"""Ordered dense-times-sparse memory state for the Associative memory Decision 1 canary."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from comfy_story.memory.contracts import Coverage, EvidenceProvenance, ExceptionItem, MemoryContract
from comfy_story.memory.fusion import MatrixSemigroupFusion
from comfy_story.memory.selection import merge_top_k


def _is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _presentation_key(item: ExceptionItem) -> tuple[int, int, int, int, int, str, int, str, str]:
    provenance = item.provenance
    time = provenance.time
    leaf = provenance.leaf
    return (
        time.timestamp_start_ns,
        time.frame_start,
        time.timestamp_stop_ns,
        time.frame_stop,
        leaf.source_rank,
        leaf.source_id,
        leaf.slot,
        item.item_id,
        item.fingerprint(),
    )


@dataclass(frozen=True, slots=True)
class MemoryState:
    """A bounded exception memory and an ordered matrix product over one slot interval."""

    contract: MemoryContract
    coverage: Coverage
    dense_operator: torch.Tensor
    exceptions: tuple[ExceptionItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.contract, MemoryContract):
            raise ValueError("contract must be a MemoryContract")
        self.contract.validate()
        if not isinstance(self.coverage, Coverage):
            raise ValueError("coverage must be a Coverage")
        self.coverage.validate()
        if self.coverage.stop_slot > self.contract.history_items:
            raise ValueError("coverage exceeds contract history_items")
        _validate_dense_operator(self.dense_operator, self.contract)
        if not isinstance(self.exceptions, tuple):
            raise ValueError("exceptions must be a tuple")
        canonical = merge_top_k((), self.exceptions, self.contract.sparse)
        if canonical != self.exceptions:
            raise ValueError("exceptions must be canonical bounded Top-K retention order")
        for item in self.exceptions:
            slot = item.provenance.leaf.slot
            if slot >= self.contract.history_items:
                raise ValueError("exception LeafKey.slot exceeds contract history_items")
            if not self.coverage.start_slot <= slot < self.coverage.stop_slot:
                raise ValueError("exception LeafKey.slot must be within state coverage")


@dataclass(frozen=True, slots=True)
class MaterializedMemory:
    """Decoded dense tokens and temporally presented, aligned exception sidecars."""

    dense_tokens: torch.Tensor
    exception_embeddings: torch.Tensor
    item_ids: tuple[str, ...]
    provenance: tuple[EvidenceProvenance, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dense_tokens, torch.Tensor)
            or self.dense_tokens.layout != torch.strided
            or self.dense_tokens.ndim != 3
            or self.dense_tokens.shape[0] != 1
            or not _is_finite(self.dense_tokens)
        ):
            raise ValueError("dense_tokens must be a finite [1,dense_steps,channels] tensor")
        if (
            not isinstance(self.exception_embeddings, torch.Tensor)
            or self.exception_embeddings.layout != torch.strided
            or self.exception_embeddings.ndim != 2
            or not _is_finite(self.exception_embeddings)
        ):
            raise ValueError("exception_embeddings must be a finite [N,E] tensor")
        count = self.exception_embeddings.shape[0]
        if not isinstance(self.item_ids, tuple) or not isinstance(self.provenance, tuple):
            raise ValueError("materialized item_ids and provenance must be tuples")
        if len(self.item_ids) != count or len(self.provenance) != count:
            raise ValueError("materialized exception sidecars must have the same length")


def _validate_dense_operator(operator: torch.Tensor, contract: MemoryContract) -> None:
    if not isinstance(operator, torch.Tensor) or operator.layout != torch.strided:
        raise ValueError("dense_operator must be a strided tensor")
    if operator.ndim != 4:
        raise ValueError("dense_operator shape must be [B,dense_steps,operator_size,operator_size]")
    if operator.shape[0] != 1:
        raise ValueError("Decision 1 dense_operator requires B=1")
    if operator.shape[1] <= 0:
        raise ValueError("dense_operator dense_steps must be positive")
    if operator.shape[-2] != operator.shape[-1]:
        raise ValueError("dense_operator operator dimensions must be square")
    if operator.dtype != contract.dense_accumulation_dtype:
        raise ValueError("dense_operator must use the contract float32 dtype")
    if not _is_finite(operator):
        raise ValueError("dense_operator must be finite")


def identity_state(
    contract: MemoryContract,
    coverage: Coverage,
    dense_steps: int,
    operator_size: int,
    *,
    device: torch.device | None = None,
) -> MemoryState:
    """Build an empty fixed-interval state with an exact FP32 matrix identity."""
    if type(dense_steps) is not int or dense_steps <= 0:
        raise ValueError("dense_steps must be a positive integer")
    if type(operator_size) is not int or operator_size <= 0:
        raise ValueError("operator_size must be a positive integer")
    identity = torch.eye(operator_size, dtype=contract.dense_accumulation_dtype, device=device)
    dense_operator = (
        identity.reshape(1, 1, operator_size, operator_size)
        .expand(1, dense_steps, operator_size, operator_size)
        .clone()
    )
    return MemoryState(contract, coverage, dense_operator, ())


def leaf_state(
    contract: MemoryContract,
    slot: int,
    dense_operator: torch.Tensor,
    exceptions: tuple[ExceptionItem, ...] = (),
) -> MemoryState:
    """Build one canonical history-slot state, selecting its bounded exception memory."""
    if type(slot) is not int or not 0 <= slot < contract.history_items:
        raise ValueError("leaf slot must be within contract history_items")
    if not isinstance(exceptions, tuple):
        raise ValueError("exceptions must be a tuple")
    selected = merge_top_k((), exceptions, contract.sparse)
    return MemoryState(contract, Coverage(slot, slot + 1), dense_operator, selected)


def merge_states(left: MemoryState, right: MemoryState) -> MemoryState:
    """Concatenate adjacent ordered state intervals as ``right @ left``."""
    if not isinstance(left, MemoryState) or not isinstance(right, MemoryState):
        raise ValueError("merge requires MemoryState operands")
    if left.contract != right.contract:
        raise ValueError("cannot merge states with different contracts")
    if left.dense_operator.shape != right.dense_operator.shape:
        raise ValueError("cannot merge states with different dense shapes")
    if left.coverage.stop_slot != right.coverage.start_slot:
        raise ValueError("state coverage must be adjacent without gaps or overlaps")
    return MemoryState(
        contract=left.contract,
        coverage=Coverage(left.coverage.start_slot, right.coverage.stop_slot),
        dense_operator=right.dense_operator @ left.dense_operator,
        exceptions=merge_top_k(left.exceptions, right.exceptions, left.contract.sparse),
    )


def materialize_state(state: MemoryState, fusion: MatrixSemigroupFusion) -> MaterializedMemory:
    """Decode dense state and present exception sidecars in temporal/source order."""
    if not isinstance(state, MemoryState):
        raise ValueError("materialization requires a MemoryState")
    if not isinstance(fusion, MatrixSemigroupFusion):
        raise ValueError("materialization requires MatrixSemigroupFusion")
    if state.dense_operator.shape[-1] != fusion.operator_size:
        raise ValueError("dense operator_size does not match MatrixSemigroupFusion")
    if fusion.channels != state.contract.latent_channels:
        raise ValueError("MatrixSemigroupFusion channels do not match the state contract")
    if fusion.decoder.weight.dtype != state.contract.dense_accumulation_dtype:
        raise ValueError("MatrixSemigroupFusion decoder dtype does not match the state contract")
    dense_tokens = fusion.decode_operator(state.dense_operator)
    expected_dense_shape = (
        1,
        state.dense_operator.shape[1],
        state.contract.latent_channels,
    )
    if tuple(dense_tokens.shape) != expected_dense_shape:
        raise ValueError("decoded dense token shape does not match the state contract")
    if dense_tokens.dtype != state.contract.dense_accumulation_dtype:
        raise ValueError("decoded dense token dtype does not match the state contract")
    if not _is_finite(dense_tokens):
        raise ValueError("decoded dense tokens must be finite")
    ordered = tuple(sorted(state.exceptions, key=_presentation_key))
    if ordered:
        exception_embeddings = torch.stack(tuple(item.embedding for item in ordered))
    else:
        exception_embeddings = torch.empty(
            (0, state.contract.sparse.embedding_dim),
            dtype=state.contract.sparse.embedding_dtype,
        )
    return MaterializedMemory(
        dense_tokens=dense_tokens,
        exception_embeddings=exception_embeddings,
        item_ids=tuple(item.item_id for item in ordered),
        provenance=tuple(item.provenance for item in ordered),
    )
