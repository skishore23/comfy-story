from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from random import Random

import pytest
import torch

from duet.duetx.cache import AuthoritativeLeaf, DuetXProductTree
from duet.duetx.contracts import (
    DuetXContract,
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    RawEvidencePointer,
    SparseMemoryContract,
    SpatialLocation,
    TimeFrameRange,
)
from duet.duetx.state import DuetXState


@pytest.fixture
def contract() -> DuetXContract:
    return replace(
        DuetXContract.default(decision1_fingerprint="1" * 64),
        sparse=SparseMemoryContract(2, 4, torch.float32, 2),
    )


def _operator(value: float) -> torch.Tensor:
    return torch.tensor([[[[1.0, value], [0.0, 1.0]]]], dtype=torch.float32)


def _item(
    contract: DuetXContract,
    key: LeafKey,
    *,
    item_id: str,
    salience_q: int,
) -> ExceptionItem:
    provenance = EvidenceProvenance(
        leaf=key,
        time=TimeFrameRange(100 + key.slot, 101 + key.slot, key.slot, key.slot + 1),
        spatial=SpatialLocation("normalized", "bbox", (0, 0, 1, 1)),
        event_type="rare-detail-v1",
        raw=RawEvidencePointer(f"duet-evidence://registry/sha256/{'0' * 64}", "0" * 64, 0, 1),
        source_registry_sha256="1" * 64,
        preprocessing_sha256="2" * 64,
        vae_sha256="3" * 64,
        adapter_sha256="4" * 64,
        scorer_sha256="5" * 64,
        memory_contract_sha256=contract.sparse.fingerprint(),
    )
    return ExceptionItem(
        item_id,
        provenance,
        torch.full((4,), float(salience_q), dtype=torch.float32),
        salience_q,
    )


def _leaf(
    contract: DuetXContract,
    slot: int,
    *,
    revision: int = 0,
    value: float | None = None,
    candidates: tuple[ExceptionItem, ...] | None = None,
    source_id: str | None = None,
    source_fingerprint: str | None = None,
    dense_operator: torch.Tensor | None = None,
) -> AuthoritativeLeaf:
    key = LeafKey(source_id or f"camera-{slot}", slot, slot)
    local_candidates = candidates
    if local_candidates is None:
        local_candidates = (_item(contract, key, item_id=f"item-{slot}", salience_q=slot),)
    local_operator = dense_operator
    if local_operator is None:
        local_operator = _operator(float(slot) if value is None else value)
    return AuthoritativeLeaf(
        key=key,
        source_fingerprint=f"{slot:x}" * 64 if source_fingerprint is None else source_fingerprint,
        revision=revision,
        dense_operator=local_operator,
        candidates=local_candidates,
    )


def _assert_state_equal(actual: DuetXState, expected: DuetXState) -> None:
    assert actual.contract == expected.contract
    assert actual.coverage == expected.coverage
    torch.testing.assert_close(actual.dense_operator, expected.dense_operator)
    assert actual.exceptions == expected.exceptions


@pytest.fixture
def tree(contract: DuetXContract) -> DuetXProductTree:
    return DuetXProductTree(
        contract, dense_steps=1, operator_size=2, leaves=tuple(_leaf(contract, i) for i in range(8))
    )


def test_eight_slot_replace_recomputes_one_path(
    tree: DuetXProductTree, contract: DuetXContract
) -> None:
    replacement = _leaf(contract, 3, revision=1, value=30.0)

    update = tree.replace(replacement)

    assert update.products_recomputed == 3
    assert update.sparse_merges_recomputed == 3
    _assert_state_equal(tree.root, tree.full_rebuild_root())


def test_insert_and_replace_require_their_expected_occupancy(contract: DuetXContract) -> None:
    tree = DuetXProductTree(contract, dense_steps=1, operator_size=2)
    leaf = _leaf(contract, 0)

    tree.insert(leaf)

    with pytest.raises(ValueError, match="occupied"):
        tree.insert(leaf)
    with pytest.raises(ValueError, match="missing"):
        tree.replace(_leaf(contract, 1, revision=1))


def test_replace_requires_a_strictly_increasing_revision(contract: DuetXContract) -> None:
    initial = _leaf(contract, 0, revision=4)
    tree = DuetXProductTree(contract, dense_steps=1, operator_size=2, leaves=(initial,))

    with pytest.raises(ValueError, match="strictly increasing"):
        tree.replace(_leaf(contract, 0, revision=4, value=9.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        tree.replace(_leaf(contract, 0, revision=3, value=9.0))


def test_delete_requires_an_occupied_slot_and_can_check_its_revision(
    contract: DuetXContract,
) -> None:
    initial = _leaf(contract, 0, revision=4)
    tree = DuetXProductTree(contract, dense_steps=1, operator_size=2, leaves=(initial,))

    with pytest.raises(ValueError, match="revision"):
        tree.delete(initial.key, expected_revision=3)
    update = tree.delete(initial.key, expected_revision=4)

    assert update.products_recomputed == 3
    assert update.sparse_merges_recomputed == 3
    with pytest.raises(ValueError, match="missing"):
        tree.delete(initial.key)
    _assert_state_equal(tree.root, tree.full_rebuild_root())


def test_tree_rejects_out_of_range_keys_and_duplicate_sources(contract: DuetXContract) -> None:
    tree = DuetXProductTree(contract, dense_steps=1, operator_size=2)
    out_of_range = _leaf(contract, contract.history_items)

    with pytest.raises(ValueError, match="range"):
        tree.insert(out_of_range)
    with pytest.raises(ValueError, match="duplicate source"):
        DuetXProductTree(
            contract,
            dense_steps=1,
            operator_size=2,
            leaves=(_leaf(contract, 0, source_id="camera"), _leaf(contract, 1, source_id="camera")),
        )


def test_single_slot_tree_has_no_recomputed_internal_nodes(contract: DuetXContract) -> None:
    tree = DuetXProductTree(contract, dense_steps=1, operator_size=2, leaf_count=1)
    inserted = _leaf(contract, 0)

    update = tree.insert(inserted)

    assert update.products_recomputed == 0
    assert update.sparse_merges_recomputed == 0
    _assert_state_equal(tree.root, tree.full_rebuild_root())


def test_tree_rejects_non_power_of_two_leaf_counts(contract: DuetXContract) -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        DuetXProductTree(contract, dense_steps=1, operator_size=2, leaf_count=3)


def test_replacing_authoritative_candidates_recovers_a_previously_hidden_winner(
    contract: DuetXContract,
) -> None:
    key = LeafKey("camera-0", 0, 0)
    candidates = tuple(
        _item(contract, key, item_id=f"item-{salience}", salience_q=salience)
        for salience in (30, 20, 10)
    )
    tree = DuetXProductTree(
        contract,
        dense_steps=1,
        operator_size=2,
        leaves=(
            AuthoritativeLeaf(
                key=key,
                source_fingerprint="0" * 64,
                revision=0,
                dense_operator=_operator(1.0),
                candidates=candidates,
            ),
        ),
    )
    retained = tuple(item for item in candidates if item.item_id != "item-30")

    tree.replace(
        AuthoritativeLeaf(
            key=key,
            source_fingerprint="0" * 64,
            revision=1,
            dense_operator=_operator(1.0),
            candidates=retained,
        )
    )

    assert tuple(item.item_id for item in tree.root.exceptions) == ("item-20", "item-10")
    _assert_state_equal(tree.root, tree.full_rebuild_root())


def test_seeded_edit_trace_matches_a_cold_rebuild_after_every_edit(contract: DuetXContract) -> None:
    tree = DuetXProductTree(contract, dense_steps=1, operator_size=2)
    random = Random(41)
    revisions = [0] * contract.history_items
    occupied = [False] * contract.history_items

    for _ in range(80):
        slot = random.randrange(contract.history_items)
        if occupied[slot] and random.choice((True, False)):
            tree.delete(LeafKey(f"camera-{slot}", slot, slot), expected_revision=revisions[slot])
            occupied[slot] = False
        else:
            revisions[slot] += 1
            leaf = _leaf(contract, slot, revision=revisions[slot], value=random.random())
            if occupied[slot]:
                tree.replace(leaf)
            else:
                tree.insert(leaf)
                occupied[slot] = True
        _assert_state_equal(tree.root, tree.full_rebuild_root())


def test_authoritative_dense_operator_is_immutable_and_cache_stays_rebuildable(
    contract: DuetXContract,
) -> None:
    caller_operator = _operator(7.0)
    leaf = _leaf(contract, 0, dense_operator=caller_operator)
    tree = DuetXProductTree(contract, dense_steps=1, operator_size=2, leaves=(leaf,))

    caller_operator.add_(100.0)
    leaf.dense_operator.zero_()

    torch.testing.assert_close(leaf.dense_operator, _operator(7.0))
    assert all(not isinstance(getattr(leaf, field.name), torch.Tensor) for field in fields(leaf))
    bytes_field = next(field.name for field in fields(leaf) if field.name.endswith("_bytes"))
    with pytest.raises(FrozenInstanceError):
        setattr(leaf, bytes_field, b"mutated")
    _assert_state_equal(tree.root, tree.full_rebuild_root())


def test_authoritative_leaf_replace_and_equality_include_dense_operator(
    contract: DuetXContract,
) -> None:
    leaf = _leaf(contract, 0, value=1.0)

    same = replace(leaf, dense_operator=leaf.dense_operator.clone())
    changed = replace(leaf, dense_operator=_operator(2.0))

    assert same == leaf
    assert changed != leaf


@pytest.mark.parametrize("bad_fingerprint", ["", "A" * 64, "0" * 63])
def test_authoritative_leaf_rejects_invalid_source_fingerprint(
    contract: DuetXContract, bad_fingerprint: str
) -> None:
    with pytest.raises(ValueError, match="source_fingerprint"):
        _leaf(contract, 0, source_fingerprint=bad_fingerprint)


def test_tree_rejects_duplicate_source_fingerprint_on_construction(contract: DuetXContract) -> None:
    with pytest.raises(ValueError, match="source_fingerprint"):
        DuetXProductTree(
            contract,
            dense_steps=1,
            operator_size=2,
            leaves=(
                _leaf(contract, 0, source_fingerprint="a" * 64),
                _leaf(contract, 1, source_fingerprint="a" * 64),
            ),
        )


def test_replace_rejects_a_changed_source_fingerprint(contract: DuetXContract) -> None:
    tree = DuetXProductTree(
        contract,
        dense_steps=1,
        operator_size=2,
        leaves=(_leaf(contract, 0, source_fingerprint="a" * 64),),
    )

    with pytest.raises(ValueError, match="source_fingerprint"):
        tree.replace(_leaf(contract, 0, revision=1, source_fingerprint="b" * 64))


def test_tree_rejects_candidates_from_another_source_within_a_leaf(
    contract: DuetXContract,
) -> None:
    leaf = _leaf(contract, 0)
    foreign_key = LeafKey("other-camera", 0, 0)
    foreign_candidate = _item(contract, foreign_key, item_id="foreign", salience_q=1)
    mismatched = replace(leaf, candidates=(foreign_candidate,), dense_operator=leaf.dense_operator)

    with pytest.raises(ValueError, match="source_id"):
        DuetXProductTree(contract, dense_steps=1, operator_size=2, leaves=(mismatched,))


@pytest.mark.parametrize(
    ("dense_operator", "message"),
    [
        (torch.ones(1, 1, 2, 2, dtype=torch.float64), "float32"),
        (torch.ones(1, 1, 2, 3, dtype=torch.float32), "square"),
    ],
)
def test_tree_preserves_dense_dtype_and_shape_contracts(
    contract: DuetXContract, dense_operator: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DuetXProductTree(
            contract,
            dense_steps=1,
            operator_size=2,
            leaves=(_leaf(contract, 0, dense_operator=dense_operator),),
        )
