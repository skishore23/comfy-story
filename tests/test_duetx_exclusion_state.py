from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from duet.duetx.cache import AuthoritativeLeaf
from duet.duetx.contracts import (
    Coverage,
    DuetXContract,
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    RawEvidencePointer,
    SparseMemoryContract,
    SpatialLocation,
    TimeFrameRange,
)
from duet.duetx.exclusion_state import (
    ExclusionDuetXProductTree,
    ExclusionDuetXState,
    exclusion_identity_state,
    exclusion_leaf_state,
    materialize_exclusion_state,
    merge_exclusion_states,
)
from duet.fusion import FusionSpec, MatrixSemigroupFusion


@pytest.fixture
def contract() -> DuetXContract:
    return replace(
        DuetXContract.default(decision1_fingerprint="1" * 64),
        sparse=SparseMemoryContract(2, 4, torch.float32, 2),
    )


def _operator(slot: int) -> torch.Tensor:
    if slot % 2:
        values = ((1.0, 0.0), (float(slot + 1), 1.0))
    else:
        values = ((1.0, float(slot + 1)), (0.0, 1.0))
    return torch.tensor([[values]], dtype=torch.float32)


def _item(
    contract: DuetXContract,
    slot: int,
    salience_q: int,
    *,
    item_id: str | None = None,
    pinned: bool = False,
) -> ExceptionItem:
    key = LeafKey(f"camera-{slot}", slot, slot)
    raw_sha256 = f"{slot + 1:064x}"
    return ExceptionItem(
        item_id or f"item-{slot}",
        EvidenceProvenance(
            leaf=key,
            time=TimeFrameRange(slot * 10, slot * 10 + 1, slot, slot + 1),
            spatial=SpatialLocation("normalized", "bbox", (0, 0, 1, 1)),
            event_type="rare-marker:v1" if slot in {2, 5} else "context:v1",
            raw=RawEvidencePointer(
                f"duet-evidence://registry/sha256/{raw_sha256}", raw_sha256, 0, 1
            ),
            source_registry_sha256="1" * 64,
            preprocessing_sha256="2" * 64,
            vae_sha256="3" * 64,
            adapter_sha256="4" * 64,
            scorer_sha256="5" * 64,
            memory_contract_sha256=contract.sparse.fingerprint(),
        ),
        torch.tensor([slot, salience_q, slot + 1, salience_q + 1], dtype=torch.float32),
        salience_q,
        pinned,
    )


def _leaf_state(contract: DuetXContract, slot: int, salience_q: int) -> ExclusionDuetXState:
    return exclusion_leaf_state(
        contract,
        slot,
        _operator(slot),
        (_item(contract, slot, salience_q),),
    )


def _cold_product(leaves: tuple[ExclusionDuetXState, ...], excluded: set[int]) -> torch.Tensor:
    result = torch.eye(2, dtype=torch.float32).reshape(1, 1, 2, 2)
    for slot, leaf in enumerate(leaves):
        if slot not in excluded:
            result = leaf.full_operator @ result
    return result


def test_root_variant_is_the_ordered_core_with_selected_exception_leaves_excluded(
    contract: DuetXContract,
) -> None:
    leaves = tuple(
        _leaf_state(contract, slot, salience)
        for slot, salience in enumerate((10, 20, 900, 30, 40, 800, 50, 60))
    )
    tree = ExclusionDuetXProductTree.from_states(leaves)

    assert tuple(item.item_id for item in tree.root.exceptions) == ("item-2", "item-5")
    assert tree.root.excluded_leaf_keys == (
        LeafKey("camera-2", 2, 2),
        LeafKey("camera-5", 5, 5),
    )
    torch.testing.assert_close(tree.root.excluded_operator, _cold_product(leaves, {2, 5}))
    assert not torch.equal(tree.root.excluded_operator, tree.root.full_operator)


def test_all_bracketings_preserve_every_exclusion_variant_and_exact_sidecars(
    contract: DuetXContract,
) -> None:
    first, second, third, fourth = tuple(
        _leaf_state(contract, slot, salience) for slot, salience in enumerate((100, 20, 80, 10))
    )
    merge = merge_exclusion_states
    bracketings = (
        merge(merge(merge(first, second), third), fourth),
        merge(merge(first, merge(second, third)), fourth),
        merge(merge(first, second), merge(third, fourth)),
        merge(first, merge(merge(second, third), fourth)),
        merge(first, merge(second, merge(third, fourth))),
    )

    expected = bracketings[0]
    assert len(expected.variants) == 4
    for state in bracketings:
        assert state.exceptions == expected.exceptions
        assert tuple(variant.excluded for variant in state.variants) == tuple(
            variant.excluded for variant in expected.variants
        )
        for actual, wanted in zip(state.variants, expected.variants, strict=True):
            torch.testing.assert_close(actual.operator, wanted.operator)

    leaves = (first, second, third, fourth)
    for variant in expected.variants:
        excluded_slots = {key.leaf.slot for key in variant.excluded}
        torch.testing.assert_close(variant.operator, _cold_product(leaves, excluded_slots))


def test_parent_winners_are_available_after_adversarial_child_pruning(
    contract: DuetXContract,
) -> None:
    leaves = tuple(
        _leaf_state(contract, slot, salience)
        for slot, salience in enumerate((100, 90, 80, 70, 110, 5, 4, 3))
    )
    left = ExclusionDuetXProductTree.from_states(leaves[:4]).root
    right = merge_exclusion_states(
        merge_exclusion_states(leaves[4], leaves[5]),
        merge_exclusion_states(leaves[6], leaves[7]),
    )
    root = merge_exclusion_states(left, right)

    assert tuple(item.item_id for item in left.exceptions) == ("item-0", "item-1")
    assert tuple(item.item_id for item in root.exceptions) == ("item-4", "item-0")
    torch.testing.assert_close(root.excluded_operator, _cold_product(leaves, {0, 4}))


def test_materialization_decodes_the_exclusion_core_and_preserves_exception_bytes(
    contract: DuetXContract,
) -> None:
    fusion = MatrixSemigroupFusion(128, FusionSpec(), operator_size=2)
    later_high_rank = _item(contract, 0, 100)
    later_high_rank = replace(
        later_high_rank,
        embedding=later_high_rank.embedding,
        provenance=replace(
            later_high_rank.provenance,
            time=TimeFrameRange(100, 110, 5, 6),
        ),
    )
    earlier_low_rank = _item(contract, 1, 90)
    left = exclusion_leaf_state(contract, 0, _operator(0), (later_high_rank,))
    right = exclusion_leaf_state(contract, 1, _operator(1), (earlier_low_rank,))
    state = merge_exclusion_states(left, right)

    result = materialize_exclusion_state(state, fusion)

    presentation_order = (earlier_low_rank, later_high_rank)
    assert state.exceptions == (later_high_rank, earlier_low_rank)
    assert result.item_ids == tuple(item.item_id for item in presentation_order)
    assert result.provenance == tuple(item.provenance for item in presentation_order)
    assert torch.equal(
        result.exception_embeddings,
        torch.stack(tuple(item.embedding for item in presentation_order)),
    )
    torch.testing.assert_close(result.dense_tokens, fusion.decode_operator(state.excluded_operator))


def test_absent_leaf_is_identity_for_every_selected_subset(contract: DuetXContract) -> None:
    absent = exclusion_identity_state(contract, Coverage(0, 1), 1, 2)
    present = _leaf_state(contract, 1, 100)

    merged = merge_exclusion_states(absent, present)

    torch.testing.assert_close(merged.full_operator, present.full_operator)
    torch.testing.assert_close(
        merged.excluded_operator,
        torch.eye(2, dtype=torch.float32).reshape(1, 1, 2, 2),
    )


def test_zero_and_one_winner_states_have_every_required_subset_variant(
    contract: DuetXContract,
) -> None:
    absent = exclusion_identity_state(contract, Coverage(0, 1), 1, 2)
    present = _leaf_state(contract, 1, 100)

    assert tuple(variant.excluded for variant in absent.variants) == ((),)
    assert tuple(variant.excluded for variant in present.variants) == (
        (),
        present.candidate_keys,
    )
    torch.testing.assert_close(present.variants[0].operator, _operator(1))
    torch.testing.assert_close(
        present.variants[1].operator,
        torch.eye(2, dtype=torch.float32).reshape(1, 1, 2, 2),
    )


def test_merge_rejects_reversed_and_nonadjacent_ranges(contract: DuetXContract) -> None:
    first = _leaf_state(contract, 0, 10)
    second = _leaf_state(contract, 1, 20)
    third = _leaf_state(contract, 2, 30)

    with pytest.raises(ValueError, match="adjacent"):
        merge_exclusion_states(second, first)
    with pytest.raises(ValueError, match="adjacent"):
        merge_exclusion_states(first, third)


def test_ties_are_deterministic_and_pin_overflow_still_fails_closed(
    contract: DuetXContract,
) -> None:
    states = tuple(_leaf_state(contract, slot, 10) for slot in range(4))
    left = merge_exclusion_states(merge_exclusion_states(states[0], states[1]), states[2])
    right = merge_exclusion_states(states[0], merge_exclusion_states(states[1], states[2]))

    assert tuple(item.item_id for item in left.exceptions) == ("item-0", "item-1")
    assert left.exceptions == right.exceptions

    pinned = tuple(
        exclusion_leaf_state(
            contract,
            slot,
            _operator(slot),
            (_item(contract, slot, 10, pinned=True),),
        )
        for slot in range(3)
    )
    with pytest.raises(ValueError, match="pinned"):
        merge_exclusion_states(merge_exclusion_states(pinned[0], pinned[1]), pinned[2])


def test_leaf_rejects_multiple_candidates_and_cross_leaf_duplicate_ids(
    contract: DuetXContract,
) -> None:
    first = _item(contract, 0, 10, item_id="duplicate")
    second_same_leaf = _item(contract, 0, 9, item_id="other")
    with pytest.raises(ValueError, match="one whole-leaf"):
        exclusion_leaf_state(contract, 0, _operator(0), (first, second_same_leaf))

    left = exclusion_leaf_state(contract, 0, _operator(0), (first,))
    right = exclusion_leaf_state(
        contract, 1, _operator(1), (_item(contract, 1, 20, item_id="duplicate"),)
    )
    with pytest.raises(ValueError, match="conflicting exception item_id"):
        merge_exclusion_states(left, right)


def _authoritative_leaf(contract: DuetXContract, slot: int, revision: int = 0) -> AuthoritativeLeaf:
    item = _item(contract, slot, 100 if slot in {2, 5} else slot)
    return AuthoritativeLeaf(
        key=item.provenance.leaf,
        source_fingerprint=f"{slot:x}" * 64,
        revision=revision,
        dense_operator=_operator(slot + revision * 10),
        candidates=(item,),
    )


def test_cached_replacement_matches_cold_rebuild_for_all_variants(
    contract: DuetXContract,
) -> None:
    tree = ExclusionDuetXProductTree(
        contract,
        dense_steps=1,
        operator_size=2,
        leaves=tuple(_authoritative_leaf(contract, slot) for slot in range(8)),
    )

    update = tree.replace(_authoritative_leaf(contract, 3, revision=1))
    cold = tree.full_rebuild_root()

    assert update.products_recomputed == 3
    assert tree.root.exceptions == cold.exceptions
    assert tuple(variant.excluded for variant in tree.root.variants) == tuple(
        variant.excluded for variant in cold.variants
    )
    for cached_variant, cold_variant in zip(tree.root.variants, cold.variants, strict=True):
        torch.testing.assert_close(cached_variant.operator, cold_variant.operator)


def test_cached_deletion_promotes_the_next_winner_and_matches_cold_rebuild(
    contract: DuetXContract,
) -> None:
    leaves = tuple(_authoritative_leaf(contract, slot) for slot in range(8))
    tree = ExclusionDuetXProductTree(contract, 1, 2, leaves=leaves)
    removed = tree.root.exceptions[0]

    tree.delete(removed.provenance.leaf)
    cold = tree.full_rebuild_root()

    assert removed.item_id not in {item.item_id for item in tree.root.exceptions}
    assert tree.root.exceptions == cold.exceptions
    for cached_variant, cold_variant in zip(tree.root.variants, cold.variants, strict=True):
        torch.testing.assert_close(cached_variant.operator, cold_variant.operator)


def test_excluded_leaf_has_no_gradient_path_but_unselected_leaves_do(
    contract: DuetXContract,
) -> None:
    operators = tuple(_operator(slot).requires_grad_() for slot in range(3))
    saliences = (100, 90, 1)
    states = tuple(
        exclusion_leaf_state(
            contract,
            slot,
            operators[slot],
            (_item(contract, slot, saliences[slot]),),
        )
        for slot in range(3)
    )
    root = merge_exclusion_states(merge_exclusion_states(states[0], states[1]), states[2])

    root.excluded_operator.sum().backward()  # type: ignore[no-untyped-call]

    assert operators[0].grad is None
    assert operators[1].grad is None
    assert operators[2].grad is not None
