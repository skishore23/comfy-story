from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from comfy_story.memory.contracts import (
    Coverage,
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    MemoryContract,
    RawEvidencePointer,
    SparseMemoryContract,
    SpatialLocation,
    TimeFrameRange,
)
from comfy_story.memory.fusion import FusionSpec, MatrixSemigroupFusion
from comfy_story.memory.state import (
    MemoryState,
    identity_state,
    leaf_state,
    materialize_state,
    merge_states,
)


@pytest.fixture
def contract() -> MemoryContract:
    return replace(
        MemoryContract.minimax_h3(adapter_fingerprint="1" * 64),
        sparse=SparseMemoryContract(2, 24, torch.float32, 2),
    )


@pytest.fixture
def fusion() -> MatrixSemigroupFusion:
    return MatrixSemigroupFusion(24, FusionSpec(), operator_size=2)


def _operator(value: float) -> torch.Tensor:
    return torch.tensor([[[[1.0, value], [0.0, 1.0]]]], dtype=torch.float32)


def _item(
    contract: MemoryContract,
    *,
    item_id: str,
    slot: int,
    timestamp_start_ns: int,
    source_rank: int,
    salience_q: int,
) -> ExceptionItem:
    provenance = EvidenceProvenance(
        leaf=LeafKey(f"camera-{source_rank}", source_rank, slot),
        time=TimeFrameRange(timestamp_start_ns, timestamp_start_ns + 1, slot, slot + 1),
        spatial=SpatialLocation("normalized", "bbox", (0, 0, 1, 1)),
        event_type="rare-detail-v1",
        raw=RawEvidencePointer(f"comfy-evidence://registry/sha256/{'0' * 64}", "0" * 64, 0, 1),
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
        torch.full((24,), float(slot), dtype=torch.float32),
        salience_q,
    )


def _state(contract: MemoryContract, slot: int, value: float) -> MemoryState:
    return leaf_state(contract, slot, _operator(value))


def test_merge_uses_right_times_left(contract: MemoryContract) -> None:
    left = _state(contract, 0, 2.0)
    right = _state(contract, 1, 3.0)

    merged = merge_states(left, right)

    assert torch.equal(merged.dense_operator, right.dense_operator @ left.dense_operator)
    assert merged.coverage == Coverage(0, 2)


def test_merge_rejects_reversed_adjacent_operands_and_preserves_noncommutativity(
    contract: MemoryContract,
) -> None:
    left_operator = torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]], dtype=torch.float32)
    right_operator = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]], dtype=torch.float32)
    left = leaf_state(contract, 0, left_operator)
    right = leaf_state(contract, 1, right_operator)

    merged = merge_states(left, right)

    assert torch.equal(merged.dense_operator, right_operator @ left_operator)
    assert not torch.equal(merged.dense_operator, left_operator @ right_operator)
    with pytest.raises(ValueError, match="adjacent"):
        merge_states(right, left)


def test_identity_represents_a_fixed_interval(contract: MemoryContract) -> None:
    identity = identity_state(contract, Coverage(0, 1), dense_steps=1, operator_size=2)
    leaf = _state(contract, 1, 7.0)

    merged = merge_states(identity, leaf)

    assert torch.equal(merged.dense_operator, leaf.dense_operator)
    assert merged.exceptions == ()


def test_all_five_four_leaf_bracketings_agree(contract: MemoryContract) -> None:
    items = tuple(
        _item(
            contract,
            item_id=f"item-{index}",
            slot=index,
            timestamp_start_ns=10 + index,
            source_rank=index,
            salience_q=salience_q,
        )
        for index, salience_q in enumerate((100, 20, 80, 10))
    )
    first, second, third, fourth = tuple(
        leaf_state(contract, index, _operator(float(index + 1)), (item,))
        for index, item in enumerate(items)
    )

    bracketings = (
        merge_states(merge_states(merge_states(first, second), third), fourth),
        merge_states(merge_states(first, merge_states(second, third)), fourth),
        merge_states(merge_states(first, second), merge_states(third, fourth)),
        merge_states(first, merge_states(merge_states(second, third), fourth)),
        merge_states(first, merge_states(second, merge_states(third, fourth))),
    )

    expected = bracketings[0]
    expected_ids = tuple(item.item_id for item in expected.exceptions)
    expected_fingerprints = tuple(item.fingerprint() for item in expected.exceptions)
    assert expected_ids == ("item-0", "item-2")
    for state in bracketings:
        torch.testing.assert_close(state.dense_operator, expected.dense_operator)
        assert tuple(item.item_id for item in state.exceptions) == expected_ids
        assert tuple(item.fingerprint() for item in state.exceptions) == expected_fingerprints


@pytest.mark.parametrize(
    ("left_coverage", "right_coverage", "message"),
    [
        (Coverage(0, 1), Coverage(2, 3), "adjacent"),
        (Coverage(0, 2), Coverage(1, 3), "adjacent"),
    ],
)
def test_merge_rejects_nonadjacent_or_overlapping_coverage(
    contract: MemoryContract, left_coverage: Coverage, right_coverage: Coverage, message: str
) -> None:
    left = MemoryState(contract, left_coverage, _operator(1.0), ())
    right = MemoryState(contract, right_coverage, _operator(2.0), ())

    with pytest.raises(ValueError, match=message):
        merge_states(left, right)


def test_state_rejects_contract_coverage_and_dense_boundary_mismatches(
    contract: MemoryContract,
) -> None:
    with pytest.raises(ValueError, match="coverage"):
        MemoryState(contract, Coverage(0, 9), _operator(1.0), ())
    with pytest.raises(ValueError, match="shape"):
        MemoryState(contract, Coverage(0, 1), torch.ones(1, 2, 3), ())
    with pytest.raises(ValueError, match="square"):
        MemoryState(contract, Coverage(0, 1), torch.ones(1, 1, 2, 3), ())
    with pytest.raises(ValueError, match="float32"):
        MemoryState(contract, Coverage(0, 1), _operator(1.0).double(), ())
    with pytest.raises(ValueError, match="finite"):
        MemoryState(contract, Coverage(0, 1), _operator(float("nan")), ())
    with pytest.raises(ValueError, match="B=1"):
        MemoryState(contract, Coverage(0, 1), _operator(1.0).repeat(2, 1, 1, 1), ())

    different_contract = MemoryContract.minimax_h3(adapter_fingerprint="2" * 64)
    with pytest.raises(ValueError, match="contract"):
        merge_states(_state(contract, 0, 1.0), _state(different_contract, 1, 2.0))
    with pytest.raises(ValueError, match="dense shape"):
        merge_states(
            _state(contract, 0, 1.0),
            MemoryState(contract, Coverage(1, 2), torch.eye(3).reshape(1, 1, 3, 3), ()),
        )


def test_leaf_and_state_reject_out_of_range_exception_slots(contract: MemoryContract) -> None:
    item = _item(
        contract,
        item_id="out-of-range",
        slot=contract.history_items,
        timestamp_start_ns=10,
        source_rank=0,
        salience_q=1,
    )

    with pytest.raises(ValueError, match="history_items"):
        leaf_state(contract, contract.history_items, _operator(1.0), (item,))
    with pytest.raises(ValueError, match="history_items"):
        MemoryState(contract, Coverage(0, 1), _operator(1.0), (item,))


def test_materialize_aligns_every_sidecar_and_presents_temporally(
    contract: MemoryContract, fusion: MatrixSemigroupFusion
) -> None:
    later = _item(
        contract,
        item_id="later",
        slot=0,
        timestamp_start_ns=20,
        source_rank=2,
        salience_q=100,
    )
    earlier = _item(
        contract,
        item_id="earlier",
        slot=1,
        timestamp_start_ns=10,
        source_rank=1,
        salience_q=1,
    )
    state = merge_states(
        leaf_state(contract, 0, _operator(1.0), (later,)),
        leaf_state(contract, 1, _operator(2.0), (earlier,)),
    )

    result = materialize_state(state, fusion)

    assert result.dense_tokens.shape == (1, 1, contract.latent_channels)
    assert result.dense_tokens.dtype == torch.float32
    assert result.exception_embeddings.shape == (2, 24)
    assert result.item_ids == ("earlier", "later")
    assert tuple(item.leaf.slot for item in result.provenance) == (1, 0)
    assert result.exception_embeddings.shape[0] == len(result.item_ids)
    assert len(result.provenance) == len(result.item_ids)
    for index, item in enumerate((earlier, later)):
        assert result.item_ids[index] == item.item_id
        assert result.provenance[index] == item.provenance
        torch.testing.assert_close(result.exception_embeddings[index], item.embedding)


def test_materialize_empty_exceptions_preserves_empty_embedding_shape(
    contract: MemoryContract, fusion: MatrixSemigroupFusion
) -> None:
    result = materialize_state(_state(contract, 0, 1.0), fusion)

    assert result.exception_embeddings.shape == (0, contract.sparse.embedding_dim)
    assert result.item_ids == ()
    assert result.provenance == ()


def test_leaf_requires_exception_slots_to_match_its_slot(contract: MemoryContract) -> None:
    item = _item(
        contract,
        item_id="other-slot",
        slot=1,
        timestamp_start_ns=10,
        source_rank=0,
        salience_q=1,
    )
    with pytest.raises(ValueError, match="coverage"):
        leaf_state(contract, 0, _operator(1.0), (item,))


def test_materialize_rejects_operator_size_mismatch(
    contract: MemoryContract, fusion: MatrixSemigroupFusion
) -> None:
    state = MemoryState(contract, Coverage(0, 1), torch.eye(3).reshape(1, 1, 3, 3), ())
    with pytest.raises(ValueError, match="operator_size"):
        materialize_state(state, fusion)


def test_materialize_rejects_fusion_channel_mismatch_before_decode(
    contract: MemoryContract,
) -> None:
    state = _state(contract, 0, 1.0)
    mismatched_fusion = MatrixSemigroupFusion(4, FusionSpec(), operator_size=2)

    with pytest.raises(ValueError, match="channels"):
        materialize_state(state, mismatched_fusion)


@pytest.mark.parametrize(
    ("decoded", "message"),
    [
        (torch.zeros(1, 1, 127, dtype=torch.float32), "shape"),
        (torch.zeros(1, 1, 24, dtype=torch.float64), "dtype"),
        (torch.full((1, 1, 24), float("nan"), dtype=torch.float32), "finite"),
    ],
)
def test_materialize_rejects_invalid_decoded_dense_boundary(
    contract: MemoryContract,
    fusion: MatrixSemigroupFusion,
    monkeypatch: pytest.MonkeyPatch,
    decoded: torch.Tensor,
    message: str,
) -> None:
    def invalid_decode(_: MatrixSemigroupFusion, __: torch.Tensor) -> torch.Tensor:
        return decoded

    monkeypatch.setattr(MatrixSemigroupFusion, "decode_operator", invalid_decode)

    with pytest.raises(ValueError, match=message):
        materialize_state(_state(contract, 0, 1.0), fusion)
