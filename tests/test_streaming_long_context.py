from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from comfy_story.contracts import (
    DuetXContract,
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    RawEvidencePointer,
    SpatialLocation,
    TimeFrameRange,
    tensor_sha256,
)
from comfy_story.fusion import FusionSpec, MatrixSemigroupFusion
from comfy_story.streaming_long_context import (
    StreamingHistoryLeaf,
    StreamingLongContextContractV1,
    build_streaming_equivalence_audit,
    materialize_streaming_long_context,
)

_SOURCE_SHA256 = "a" * 64


def _contract(history_items: int, *, capacity: int = 2) -> StreamingLongContextContractV1:
    return StreamingLongContextContractV1.default(
        source_contract_sha256=_SOURCE_SHA256,
        history_items=history_items,
        exception_capacity=capacity,
        latent_channels=4,
    )


def _exception(contract: StreamingLongContextContractV1, slot: int) -> ExceptionItem:
    digest = f"{slot + 1:064x}"
    return ExceptionItem(
        f"exception-{slot}",
        EvidenceProvenance(
            LeafKey(f"camera-{slot}", slot, slot),
            TimeFrameRange(slot * 10, slot * 10 + 1, slot, slot + 1),
            SpatialLocation("whole-leaf-v1", "bbox", (0, 0, 65_536, 65_536)),
            "hard-pinned-whole-leaf-v1",
            RawEvidencePointer(f"duet-evidence://test/sha256/{digest}", digest, 0, 1),
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "5" * 64,
            contract.sparse.fingerprint(),
        ),
        torch.full((4,), float(slot), dtype=torch.float32),
        slot,
        pinned=True,
    )


def _payload(slot: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(10_000 + slot)
    return torch.randn((1, 2, 4), generator=generator, dtype=torch.float32)


def _leaves(
    contract: StreamingLongContextContractV1,
    pinned_slots: tuple[int, ...],
) -> tuple[StreamingHistoryLeaf, ...]:
    return tuple(
        StreamingHistoryLeaf(
            slot,
            _payload(slot),
            _exception(contract, slot) if slot in pinned_slots else None,
        )
        for slot in range(contract.history_items)
    )


@pytest.mark.parametrize("history_items", [8, 32, 128])
def test_streams_arbitrary_history_once_and_excludes_hard_pins_from_the_core(
    history_items: int,
) -> None:
    contract = _contract(history_items)
    pinned_slots = (2, history_items - 3)
    leaves = _leaves(contract, pinned_slots)
    fusion = MatrixSemigroupFusion(4, FusionSpec("balanced"), operator_size=3)
    encode_shapes: list[tuple[int, ...]] = []
    original_encode = fusion.encode_stream

    def observed_encode(tokens: torch.Tensor) -> torch.Tensor:
        encode_shapes.append(tuple(tokens.shape))
        return original_encode(tokens)

    fusion.encode_stream = observed_encode  # type: ignore[method-assign]
    iterations = 0

    def one_shot() -> Iterator[StreamingHistoryLeaf]:
        nonlocal iterations
        iterations += 1
        if iterations != 1:
            raise AssertionError("history iterable was consumed more than once")
        yield from leaves

    result = materialize_streaming_long_context(contract, one_shot(), fusion)

    assert iterations == 1
    assert encode_shapes == [(1, 2, 4)] * (history_items - len(pinned_slots))
    assert result.core_tokens.shape == (1, 2, 4)
    assert result.core_operator.shape == (1, 2, 3, 3)
    assert tuple(record.item.provenance.leaf.slot for record in result.exceptions) == pinned_slots
    assert result.audit.history_items == history_items
    assert result.audit.ordinary_leaf_count == history_items - len(pinned_slots)
    assert result.audit.excluded_leaf_count == len(pinned_slots)
    assert result.audit.equivalent is True
    assert result.audit.centralized_operator_sha256 == tensor_sha256(result.core_operator)

    expected_operator: torch.Tensor | None = None
    for leaf in leaves:
        if leaf.exception is not None:
            continue
        operator = original_encode(leaf.tokens)
        expected_operator = operator if expected_operator is None else operator @ expected_operator
    assert expected_operator is not None
    torch.testing.assert_close(result.core_operator, expected_operator)


def test_retains_raw_exception_bytes_and_provenance_in_temporal_slot_order() -> None:
    contract = _contract(8)
    leaves = list(_leaves(contract, (2, 5)))
    originals = {slot: leaves[slot].tokens.clone() for slot in (2, 5)}
    expected_fingerprints = tuple(leaves[slot].exception.fingerprint() for slot in (2, 5))  # type: ignore[union-attr]
    fusion = MatrixSemigroupFusion(4, FusionSpec("balanced"), operator_size=3)

    result = materialize_streaming_long_context(contract, iter(leaves), fusion)
    leaves[2].tokens.fill_(123)
    leaves[5].tokens.fill_(-456)

    assert tuple(record.item.fingerprint() for record in result.exceptions) == expected_fingerprints
    for record, slot in zip(result.exceptions, (2, 5), strict=True):
        assert torch.equal(record.raw_leaf, originals[slot])
        assert record.raw_leaf_sha256 == tensor_sha256(originals[slot])
        first = record.raw_leaf
        first.zero_()
        assert torch.equal(record.raw_leaf, originals[slot])


def test_hard_pin_payload_changes_do_not_change_the_dense_core() -> None:
    contract = _contract(8)
    original = list(_leaves(contract, (2, 5)))
    changed = list(_leaves(contract, (2, 5)))
    changed[2].tokens.fill_(100_000)
    changed[5].tokens.fill_(-100_000)
    fusion = MatrixSemigroupFusion(4, FusionSpec("balanced"), operator_size=3)

    left = materialize_streaming_long_context(contract, iter(original), fusion)
    right = materialize_streaming_long_context(contract, iter(changed), fusion)

    assert torch.equal(left.core_operator, right.core_operator)
    assert not torch.equal(left.exceptions[0].raw_leaf, right.exceptions[0].raw_leaf)
    assert not torch.equal(left.exceptions[1].raw_leaf, right.exceptions[1].raw_leaf)


def test_equivalence_audit_records_both_topologies_and_fails_closed_on_drift() -> None:
    centralized = torch.eye(2, dtype=torch.float32).reshape(1, 1, 2, 2)
    balanced = centralized.clone()

    audit = build_streaming_equivalence_audit(
        centralized,
        balanced,
        history_items=32,
        ordinary_leaf_count=30,
        excluded_leaf_count=2,
        atol=2e-6,
        rtol=2e-6,
    )

    assert audit.format == "duet-x-streaming-equivalence-audit-v1"
    assert audit.equivalent is True
    assert audit.max_abs_error == 0.0
    assert audit.centralized_operator_sha256 == audit.balanced_operator_sha256

    drifted = balanced.clone()
    drifted[..., 0, 1] = 1.0
    with pytest.raises(ValueError, match="centralized and balanced"):
        build_streaming_equivalence_audit(
            centralized,
            drifted,
            history_items=32,
            ordinary_leaf_count=30,
            excluded_leaf_count=2,
            atol=2e-6,
            rtol=2e-6,
        )


def test_audit_binds_an_independent_balanced_tree_and_materialization_is_inference_only() -> None:
    contract = _contract(10)
    leaves = _leaves(contract, (8, 9))
    for leaf in leaves:
        leaf.tokens.requires_grad_()
    fusion = MatrixSemigroupFusion(4, FusionSpec("balanced"), operator_size=3)
    ordinary_operators = tuple(
        fusion.encode_stream(leaf.tokens.detach()) for leaf in leaves if leaf.exception is None
    )
    expected_balanced = fusion.combine_ordered_operators(ordinary_operators)

    result = materialize_streaming_long_context(contract, iter(leaves), fusion)

    assert result.audit.balanced_operator_sha256 == tensor_sha256(expected_balanced)
    assert result.core_operator.requires_grad is False
    assert result.core_tokens.requires_grad is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("out-of-order", "ordered contiguous slots"),
        ("missing", "exactly 8"),
        ("too-few-pins", "exactly 2"),
        ("not-pinned", "hard-pinned"),
        ("wrong-item-slot", "same slot"),
        ("shape-drift", "same token boundary"),
    ],
)
def test_rejects_malformed_stream_and_exception_boundaries(mutation: str, message: str) -> None:
    contract = _contract(8)
    leaves = list(_leaves(contract, (2, 5)))
    if mutation == "out-of-order":
        leaves[0], leaves[1] = leaves[1], leaves[0]
    elif mutation == "missing":
        leaves.pop()
    elif mutation == "too-few-pins":
        leaves[5] = StreamingHistoryLeaf(5, leaves[5].tokens, None)
    elif mutation == "not-pinned":
        item = leaves[2].exception
        assert item is not None
        leaves[2] = StreamingHistoryLeaf(
            2,
            leaves[2].tokens,
            ExceptionItem(
                item.item_id,
                item.provenance,
                item.embedding,
                item.salience_q,
                pinned=False,
            ),
        )
    elif mutation == "wrong-item-slot":
        leaves[2] = StreamingHistoryLeaf(2, leaves[2].tokens, _exception(contract, 3))
    elif mutation == "shape-drift":
        leaves[4] = StreamingHistoryLeaf(4, torch.zeros(1, 3, 4), None)
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        materialize_streaming_long_context(
            contract,
            iter(leaves),
            MatrixSemigroupFusion(4, FusionSpec("balanced"), operator_size=3),
        )


def test_new_contract_does_not_widen_the_frozen_decision1_contract() -> None:
    frozen = DuetXContract.default(decision1_fingerprint="f" * 64)
    streaming = StreamingLongContextContractV1.default(
        source_contract_sha256=frozen.fingerprint(),
        history_items=128,
        exception_capacity=2,
        latent_channels=128,
    )

    assert frozen.format == "duet-x-ltx-decision1-v1"
    assert frozen.history_items == 8
    assert frozen.sparse.capacity == 2
    assert streaming.format == "duet-x-streaming-long-context-v1"
    assert streaming.history_items == 128
