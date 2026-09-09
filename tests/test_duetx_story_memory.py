from __future__ import annotations

import hashlib

import pytest
import torch

from duet.duetx.cache import AuthoritativeLeaf, DuetXProductTree
from duet.duetx.contracts import (
    DuetXContract,
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    RawEvidencePointer,
    SpatialLocation,
    TimeFrameRange,
)
from duet.duetx.ltx_bridge import LTXLatentHistoryBridge
from duet.duetx.story_memory import (
    append_story_leaf,
    apply_story_memory_policy,
    compose_inclusive_story_core,
    compose_story_blocks,
    empty_story_blocks,
    story_evidence_reservoir,
)
from duet.duetx.story_product_contracts import StoryMemoryPolicy
from duet.fusion import FusionSpec, MatrixSemigroupFusion


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _operator(global_slot: int) -> torch.Tensor:
    amount = float(global_slot + 1)
    values = ((1.0, amount), (0.0, 1.0)) if global_slot % 2 == 0 else ((1.0, 0.0), (amount, 1.0))
    return torch.tensor([[values]], dtype=torch.float32)


def _leaf(contract: DuetXContract, global_slot: int, salience_q: int) -> AuthoritativeLeaf:
    local_slot = global_slot % 8
    key = LeafKey(f"story-shot-{global_slot:03d}", global_slot, local_slot)
    raw_sha256 = _digest(f"raw-{global_slot}")
    item = ExceptionItem(
        f"story-item-{global_slot:03d}",
        EvidenceProvenance(
            leaf=key,
            time=TimeFrameRange(global_slot, global_slot + 1, global_slot, global_slot + 1),
            spatial=SpatialLocation("normalized", "bbox", (0, 0, 65_536, 65_536)),
            event_type="story-shot-v1",
            raw=RawEvidencePointer(f"duet-evidence://story/sha256/{raw_sha256}", raw_sha256, 0, 1),
            source_registry_sha256="1" * 64,
            preprocessing_sha256="2" * 64,
            vae_sha256="3" * 64,
            adapter_sha256="4" * 64,
            scorer_sha256="5" * 64,
            memory_contract_sha256=contract.sparse.fingerprint(),
        ),
        torch.full((128,), float(global_slot), dtype=torch.float32),
        salience_q,
    )
    return AuthoritativeLeaf(
        key=key,
        source_fingerprint=_digest(f"source-{global_slot}"),
        revision=0,
        dense_operator=_operator(global_slot),
        candidates=(item,),
    )


def _append_range(
    contract: DuetXContract,
    fusion: MatrixSemigroupFusion,
    scores: tuple[int, ...],
) -> tuple[DuetXProductTree, ...]:
    blocks = empty_story_blocks(
        contract,
        dense_steps=1,
        operator_size=fusion.operator_size,
    )
    for slot, score in enumerate(scores):
        blocks = append_story_leaf(blocks, shot_index=slot, leaf=_leaf(contract, slot, score))
    return blocks


def test_story_memory_has_sixteen_eight_slot_blocks_and_128_capacity() -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    fusion = MatrixSemigroupFusion(128, FusionSpec(), operator_size=2)

    blocks = empty_story_blocks(contract, dense_steps=1, operator_size=fusion.operator_size)

    assert len(blocks) == 16
    assert all(block.leaf_count == 8 for block in blocks)
    assert sum(leaf is not None for block in blocks for leaf in block.leaves) == 0


def test_append_is_contiguous_and_does_not_mutate_parent_blocks() -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    blocks = empty_story_blocks(contract, dense_steps=1, operator_size=2)
    first = append_story_leaf(blocks, shot_index=0, leaf=_leaf(contract, 0, 10))

    assert all(leaf is None for block in blocks for leaf in block.leaves)
    assert first[0].leaves[0] is not None
    with pytest.raises(ValueError, match="next contiguous story slot"):
        append_story_leaf(first, shot_index=2, leaf=_leaf(contract, 2, 20))


def test_global_top_two_are_excluded_once_across_distinct_blocks() -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    fusion = MatrixSemigroupFusion(128, FusionSpec(), operator_size=2)
    scores = (700, 1, 2, 3, 4, 5, 6, 7, 1100, 500)
    blocks = _append_range(contract, fusion, scores)

    root = compose_story_blocks(blocks, fusion)

    assert tuple(item.salience_q for item in root.exceptions) == (1100, 700)
    assert tuple(key.source_rank for key in root.excluded_leaf_keys) == (8, 0)
    expected = torch.eye(2, dtype=torch.float32).reshape(1, 1, 2, 2)
    for global_slot in range(len(scores)):
        if global_slot not in {0, 8}:
            expected = _operator(global_slot) @ expected
    torch.testing.assert_close(root.core_operator, expected)
    assert root.occupied_shots == 10


def test_inclusive_core_keeps_selected_exception_leaves() -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    fusion = MatrixSemigroupFusion(128, FusionSpec(), operator_size=2)
    blocks = _append_range(contract, fusion, (700, 1, 1100))

    inclusive = compose_inclusive_story_core(blocks, fusion)
    expected = fusion.combine_ordered_operators(
        tuple(block.root.dense_operator for block in blocks)
    )

    torch.testing.assert_close(inclusive, expected, rtol=0, atol=0)
    assert not torch.equal(inclusive, compose_story_blocks(blocks, fusion).core_operator)


def test_reservoir_is_block_local_top_k_not_global_top_k() -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    fusion = MatrixSemigroupFusion(128, FusionSpec(), operator_size=2)
    blocks = _append_range(contract, fusion, tuple(range(18)))

    reservoir = story_evidence_reservoir(blocks)

    assert len(reservoir) == 6
    assert tuple(item.provenance.leaf.source_rank // 8 for item in reservoir) == (
        0,
        0,
        1,
        1,
        2,
        2,
    )


def test_forget_removes_dense_contribution_without_mutating_parent() -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    fusion = MatrixSemigroupFusion(128, FusionSpec(), operator_size=2)
    blocks = _append_range(contract, fusion, tuple(range(10)))
    before = compose_inclusive_story_core(blocks, fusion)

    updated = apply_story_memory_policy(
        blocks,
        StoryMemoryPolicy(),
        StoryMemoryPolicy(("story-item-000",), ("story-item-008",)),
    )

    assert updated[0] is not blocks[0]
    assert updated[1] is not blocks[1]
    assert all(updated[index] is blocks[index] for index in range(2, 16))
    assert story_evidence_reservoir(updated)[0].item_id == "story-item-000"
    assert story_evidence_reservoir(updated)[0].pinned is True
    assert all(item.item_id != "story-item-008" for item in story_evidence_reservoir(updated))
    assert not torch.equal(compose_inclusive_story_core(updated, fusion), before)
    assert updated[1].leaves[0] is not None
    torch.testing.assert_close(
        updated[1].leaves[0].dense_operator, torch.eye(2).reshape(1, 1, 2, 2)
    )
    torch.testing.assert_close(compose_inclusive_story_core(blocks, fusion), before)


def test_forget_survives_followup_and_append_without_resurrection() -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    fusion = MatrixSemigroupFusion(128, FusionSpec(), operator_size=2)
    blocks = _append_range(contract, fusion, (10, 20))
    policy = StoryMemoryPolicy((), ("story-item-000",))
    forgotten = apply_story_memory_policy(blocks, StoryMemoryPolicy(), policy)
    reapplied = apply_story_memory_policy(forgotten, policy, policy)
    continued = append_story_leaf(reapplied, shot_index=2, leaf=_leaf(contract, 2, 30))
    torch.testing.assert_close(
        compose_inclusive_story_core(continued, fusion), _operator(2) @ _operator(1)
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        apply_story_memory_policy(
            continued, policy, StoryMemoryPolicy((), ("invented", "story-item-000"))
        )
    with pytest.raises(ValueError, match="resurrected"):
        apply_story_memory_policy(continued, policy, StoryMemoryPolicy())


def test_legacy_tombstone_excludes_dense_source_even_without_candidate() -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    fusion = MatrixSemigroupFusion(128, FusionSpec(), operator_size=2)
    source = _leaf(contract, 0, 10)
    legacy = AuthoritativeLeaf(source.key, source.source_fingerprint, 0, source.dense_operator, ())
    blocks = append_story_leaf(
        empty_story_blocks(contract, dense_steps=1, operator_size=2), shot_index=0, leaf=legacy
    )
    policy = StoryMemoryPolicy((), ("story-item-000",))
    repaired = apply_story_memory_policy(
        blocks, policy, policy, excluded_source_ranks=frozenset({0})
    )
    torch.testing.assert_close(
        compose_inclusive_story_core(repaired, fusion), torch.eye(2).reshape(1, 1, 2, 2)
    )


def test_append_at_128_rejects_without_mutating_blocks() -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    fusion = MatrixSemigroupFusion(128, FusionSpec(), operator_size=2)
    blocks = _append_range(contract, fusion, tuple(range(128)))
    before = tuple(tuple(block.leaves) for block in blocks)

    with pytest.raises(ValueError, match="128-shot capacity"):
        append_story_leaf(blocks, shot_index=128, leaf=_leaf(contract, 128, 128))

    assert tuple(tuple(block.leaves) for block in blocks) == before


def test_materialize_operator_matches_forward_for_the_same_history() -> None:
    torch.manual_seed(7)
    bridge = LTXLatentHistoryBridge(channels=4, operator_size=2)
    history = torch.randn(1, 3, 4, 1, 2, 2)
    anchor = history[:, -1]
    expected = bridge(history, anchor=anchor)

    actual = bridge.materialize_operator(expected.core_operator, anchor=anchor)

    assert torch.equal(actual, expected.core_latent)
