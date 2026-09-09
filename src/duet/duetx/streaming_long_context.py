"""Versioned one-pass Duet-X materialization for ordered long histories.

This module is deliberately separate from the frozen Decision 1 ``N=8`` state.  It consumes one
leaf token boundary at a time, omits explicitly hard-pinned leaves from the dense product, and keeps
only a constant-size raw exception roster plus logarithmically many balanced partial products.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field

import torch

from duet.duetx.contracts import ExceptionItem, SparseMemoryContract, tensor_sha256
from duet.fusion import MatrixSemigroupFusion

_CONTRACT_FORMAT = "duet-x-streaming-long-context-v1"
_AUDIT_FORMAT = "duet-x-streaming-equivalence-audit-v1"
_SHA256_CHARS = frozenset("0123456789abcdef")


def _sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError(f"{name} must be a strided tensor")
    if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite and floating")


@dataclass(frozen=True, slots=True)
class StreamingLongContextContractV1:
    """A distinct long-context lock that cannot widen the frozen Decision 1 contract."""

    format: str
    source_contract_sha256: str
    history_items: int
    sparse: SparseMemoryContract
    latent_channels: int
    dense_accumulation_dtype: torch.dtype
    equivalence_atol: float
    equivalence_rtol: float

    @classmethod
    def default(
        cls,
        *,
        source_contract_sha256: str,
        history_items: int,
        exception_capacity: int,
        latent_channels: int = 128,
    ) -> StreamingLongContextContractV1:
        return cls(
            _CONTRACT_FORMAT,
            source_contract_sha256,
            history_items,
            SparseMemoryContract(
                exception_capacity,
                latent_channels,
                torch.float32,
                exception_capacity,
            ),
            latent_channels,
            torch.float32,
            2e-6,
            2e-6,
        ).validate()

    def validate(self) -> StreamingLongContextContractV1:
        if self.format != _CONTRACT_FORMAT:
            raise ValueError("streaming long-context contract format changed")
        _sha256(self.source_contract_sha256, "source_contract_sha256")
        if type(self.history_items) is not int or self.history_items <= 0:
            raise ValueError("history_items must be a positive integer")
        self.sparse.validate()
        if self.sparse.capacity >= self.history_items:
            raise ValueError("exception capacity must leave at least one ordinary leaf")
        if self.sparse.max_pinned != self.sparse.capacity:
            raise ValueError("streaming long-context exceptions must all be hard-pinned")
        if type(self.latent_channels) is not int or self.latent_channels <= 0:
            raise ValueError("latent_channels must be a positive integer")
        if self.sparse.embedding_dim != self.latent_channels:
            raise ValueError("exception embedding width must equal latent_channels")
        if self.sparse.embedding_dtype != torch.float32:
            raise ValueError("exception embeddings must use float32")
        if self.dense_accumulation_dtype != torch.float32:
            raise ValueError("dense accumulation must use float32")
        for name in ("equivalence_atol", "equivalence_rtol"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        return self

    def fingerprint(self) -> str:
        self.validate()
        payload = {
            "dense_accumulation_dtype": _dtype_name(self.dense_accumulation_dtype),
            "equivalence_atol": self.equivalence_atol,
            "equivalence_rtol": self.equivalence_rtol,
            "format": self.format,
            "history_items": self.history_items,
            "latent_channels": self.latent_channels,
            "source_contract_sha256": self.source_contract_sha256,
            "sparse": self.sparse.fingerprint(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class StreamingHistoryLeaf:
    """One canonical raw leaf presented without a full-history batch dimension."""

    slot: int
    tokens: torch.Tensor
    exception: ExceptionItem | None = None


class _RawLeafBytes:
    _raw_leaf_bytes: bytes
    _raw_leaf_dtype: torch.dtype
    _raw_leaf_shape: tuple[int, ...]

    @property
    def raw_leaf(self) -> torch.Tensor:
        """Return a fresh CPU tensor reconstructed from immutable canonical bytes."""
        return torch.frombuffer(
            bytearray(self._raw_leaf_bytes), dtype=self._raw_leaf_dtype
        ).reshape(self._raw_leaf_shape)


@dataclass(frozen=True, slots=True)
class StreamingRawException(_RawLeafBytes):
    """One hard-pinned exception with immutable raw bytes and existing provenance."""

    item: ExceptionItem
    raw_leaf: InitVar[torch.Tensor] = field()
    raw_leaf_sha256: str = field(init=False)
    _raw_leaf_bytes: bytes = field(init=False, repr=False)
    _raw_leaf_dtype: torch.dtype = field(init=False, repr=False)
    _raw_leaf_shape: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self, raw_leaf: torch.Tensor) -> None:
        _finite_tensor(raw_leaf, "raw exception leaf")
        canonical = raw_leaf.detach().contiguous().cpu()
        object.__setattr__(
            self,
            "_raw_leaf_bytes",
            canonical.view(torch.uint8).numpy().tobytes(),
        )
        object.__setattr__(self, "_raw_leaf_dtype", canonical.dtype)
        object.__setattr__(self, "_raw_leaf_shape", tuple(canonical.shape))
        object.__setattr__(self, "raw_leaf_sha256", tensor_sha256(canonical))

    def validate(
        self,
        contract: StreamingLongContextContractV1,
        token_shape: tuple[int, ...],
        token_dtype: torch.dtype,
    ) -> StreamingRawException:
        contract.validate()
        self.item.validate(contract.sparse)
        if self.item.pinned is not True:
            raise ValueError("streaming exception record must be hard-pinned")
        raw = self.raw_leaf
        if tuple(raw.shape) != token_shape or raw.dtype != token_dtype:
            raise ValueError("raw exception must preserve the common token boundary")
        if tensor_sha256(raw) != self.raw_leaf_sha256:
            raise ValueError("raw exception bytes changed")
        return self


@dataclass(frozen=True, slots=True)
class StreamingEquivalenceAudit:
    """Digest-only evidence that one left fold and one balanced fold agree."""

    format: str
    history_items: int
    ordinary_leaf_count: int
    excluded_leaf_count: int
    centralized_operator_sha256: str
    balanced_operator_sha256: str
    atol: float
    rtol: float
    max_abs_error: float
    equivalent: bool

    def validate(self) -> StreamingEquivalenceAudit:
        if self.format != _AUDIT_FORMAT:
            raise ValueError("streaming equivalence audit format changed")
        if (
            type(self.history_items) is not int
            or type(self.ordinary_leaf_count) is not int
            or type(self.excluded_leaf_count) is not int
            or self.history_items <= 0
            or self.ordinary_leaf_count <= 0
            or self.excluded_leaf_count < 0
            or self.ordinary_leaf_count + self.excluded_leaf_count != self.history_items
        ):
            raise ValueError("streaming equivalence audit leaf counts are inconsistent")
        _sha256(self.centralized_operator_sha256, "centralized_operator_sha256")
        _sha256(self.balanced_operator_sha256, "balanced_operator_sha256")
        for name in ("atol", "rtol", "max_abs_error"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if self.equivalent is not True:
            raise ValueError("centralized and balanced ordered products differ")
        return self


def build_streaming_equivalence_audit(
    centralized: torch.Tensor,
    balanced: torch.Tensor,
    *,
    history_items: int,
    ordinary_leaf_count: int,
    excluded_leaf_count: int,
    atol: float,
    rtol: float,
) -> StreamingEquivalenceAudit:
    """Compare independently bracketed operator products and fail closed on drift."""
    _finite_tensor(centralized, "centralized operator")
    _finite_tensor(balanced, "balanced operator")
    if centralized.shape != balanced.shape or centralized.dtype != balanced.dtype:
        raise ValueError("centralized and balanced operator boundaries differ")
    if type(atol) is not float or type(rtol) is not float:
        raise ValueError("equivalence tolerances must be floats")
    max_abs_error = float((centralized - balanced).abs().max().item())
    equivalent = bool(torch.allclose(centralized, balanced, atol=atol, rtol=rtol))
    return StreamingEquivalenceAudit(
        _AUDIT_FORMAT,
        history_items,
        ordinary_leaf_count,
        excluded_leaf_count,
        tensor_sha256(centralized),
        tensor_sha256(balanced),
        atol,
        rtol,
        max_abs_error,
        equivalent,
    ).validate()


@dataclass(frozen=True, slots=True)
class StreamingLongContextMaterialization:
    """One bounded core plus the exact fixed-size hard-pinned exception roster."""

    contract: StreamingLongContextContractV1
    core_tokens: torch.Tensor
    core_operator: torch.Tensor
    exceptions: tuple[StreamingRawException, ...]
    audit: StreamingEquivalenceAudit

    def __post_init__(self) -> None:
        self.contract.validate()
        _finite_tensor(self.core_tokens, "streaming core tokens")
        _finite_tensor(self.core_operator, "streaming core operator")
        if tuple(self.core_tokens.shape[:1]) != (1,) or self.core_tokens.ndim != 3:
            raise ValueError("streaming core tokens must have shape [1,T,C]")
        if self.core_tokens.shape[-1] != self.contract.latent_channels:
            raise ValueError("streaming core token channel width changed")
        if (
            self.core_operator.ndim != 4
            or self.core_operator.shape[:2] != self.core_tokens.shape[:2]
        ):
            raise ValueError("streaming core operator token boundary changed")
        if self.core_operator.shape[-1] != self.core_operator.shape[-2]:
            raise ValueError("streaming core operator must be square")
        if self.core_operator.dtype != self.contract.dense_accumulation_dtype:
            raise ValueError("streaming core operator accumulation dtype changed")
        if (
            not isinstance(self.exceptions, tuple)
            or len(self.exceptions) != self.contract.sparse.capacity
        ):
            raise ValueError("streaming materialization requires exactly K exceptions")
        token_shape = tuple(self.core_tokens.shape)
        exception_slots = []
        for exception in self.exceptions:
            if not isinstance(exception, StreamingRawException):
                raise ValueError("streaming exceptions must be raw exception records")
            exception.validate(self.contract, token_shape, exception.raw_leaf.dtype)
            exception_slots.append(exception.item.provenance.leaf.slot)
        if exception_slots != sorted(exception_slots) or len(set(exception_slots)) != len(
            exception_slots
        ):
            raise ValueError("streaming exceptions must retain unique temporal slot order")
        self.audit.validate()
        if self.audit.history_items != self.contract.history_items:
            raise ValueError("streaming audit history length changed")
        if self.audit.excluded_leaf_count != len(self.exceptions):
            raise ValueError("streaming audit exception count changed")
        if self.audit.centralized_operator_sha256 != tensor_sha256(self.core_operator):
            raise ValueError("streaming audit does not bind the materialized core")


@dataclass(slots=True)
class _BalancedPartial:
    start_ordinal: int
    stop_ordinal: int
    operator: torch.Tensor


class _OnlineBalancedProduct:
    """Binary-carry reduction retaining at most one operator per tree level."""

    def __init__(self) -> None:
        self._levels: dict[int, _BalancedPartial] = {}
        self._count = 0

    def append(self, operator: torch.Tensor) -> None:
        partial = _BalancedPartial(self._count, self._count + 1, operator)
        self._count += 1
        level = 0
        while level in self._levels:
            left = self._levels.pop(level)
            if left.stop_ordinal != partial.start_ordinal:
                raise RuntimeError("balanced product lost ordered adjacency")
            partial = _BalancedPartial(
                left.start_ordinal,
                partial.stop_ordinal,
                partial.operator @ left.operator,
            )
            level += 1
        self._levels[level] = partial

    def finish(self) -> torch.Tensor:
        if not self._levels:
            raise ValueError("streaming materialization requires an ordinary leaf")
        ordered = sorted(self._levels.values(), key=lambda partial: partial.start_ordinal)
        result = ordered[0]
        for partial in ordered[1:]:
            if result.stop_ordinal != partial.start_ordinal:
                raise RuntimeError("balanced product finalization lost ordered adjacency")
            result = _BalancedPartial(
                result.start_ordinal,
                partial.stop_ordinal,
                partial.operator @ result.operator,
            )
        return result.operator


def _validate_fusion(
    contract: StreamingLongContextContractV1,
    fusion: MatrixSemigroupFusion,
) -> tuple[torch.device, torch.dtype]:
    if not isinstance(fusion, MatrixSemigroupFusion):
        raise ValueError("streaming materialization requires MatrixSemigroupFusion")
    if fusion.channels != contract.latent_channels:
        raise ValueError("fusion channels do not match the streaming contract")
    device = fusion.input_norm.weight.device
    dtype = fusion.input_norm.weight.dtype
    if dtype != contract.dense_accumulation_dtype or fusion.decoder.weight.dtype != dtype:
        raise ValueError("fusion parameters must use the contract float32 dtype")
    return device, dtype


def _validate_leaf_tokens(
    tokens: torch.Tensor,
    contract: StreamingLongContextContractV1,
    *,
    expected_shape: tuple[int, ...] | None,
    expected_dtype: torch.dtype | None,
    expected_device: torch.device,
) -> tuple[tuple[int, ...], torch.dtype]:
    _finite_tensor(tokens, "streaming history leaf")
    if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[-1] != contract.latent_channels:
        raise ValueError("streaming leaf must have shape [1,T,C]")
    if tokens.device != expected_device:
        raise ValueError("streaming leaf must be on the fusion device")
    shape = tuple(tokens.shape)
    if expected_shape is not None and (shape != expected_shape or tokens.dtype != expected_dtype):
        raise ValueError("all leaves must have the same token boundary and dtype")
    return shape, tokens.dtype


def _validate_operator(
    operator: torch.Tensor,
    *,
    token_shape: tuple[int, ...],
    operator_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    _finite_tensor(operator, "encoded streaming operator")
    expected = (token_shape[0], token_shape[1], operator_size, operator_size)
    if tuple(operator.shape) != expected:
        raise ValueError("encoded streaming operator shape changed")
    if operator.dtype != dtype or operator.device != device:
        raise ValueError("encoded streaming operator dtype or device changed")


@torch.inference_mode()
def materialize_streaming_long_context(
    contract: StreamingLongContextContractV1,
    leaves: Iterable[StreamingHistoryLeaf],
    fusion: MatrixSemigroupFusion,
) -> StreamingLongContextMaterialization:
    """Consume exactly one ordered pass and return one core plus exactly ``K`` raw exceptions."""
    if not isinstance(contract, StreamingLongContextContractV1):
        raise ValueError("streaming materialization requires its versioned contract")
    contract.validate()
    fusion_device, fusion_dtype = _validate_fusion(contract, fusion)
    iterator = iter(leaves)

    token_shape: tuple[int, ...] | None = None
    token_dtype: torch.dtype | None = None
    exceptions: list[StreamingRawException] = []
    centralized: torch.Tensor | None = None
    balanced = _OnlineBalancedProduct()
    count = 0
    for expected_slot, leaf in enumerate(iterator):
        count += 1
        if not isinstance(leaf, StreamingHistoryLeaf) or leaf.slot != expected_slot:
            raise ValueError("streaming history requires ordered contiguous slots from zero")
        if expected_slot >= contract.history_items:
            raise ValueError(f"streaming history requires exactly {contract.history_items} leaves")
        observed_shape, observed_dtype = _validate_leaf_tokens(
            leaf.tokens,
            contract,
            expected_shape=token_shape,
            expected_dtype=token_dtype,
            expected_device=fusion_device,
        )
        if token_shape is None:
            token_shape = observed_shape
            token_dtype = observed_dtype
        if leaf.exception is not None:
            leaf.exception.validate(contract.sparse)
            if leaf.exception.pinned is not True:
                raise ValueError("streaming exception leaves must be explicitly hard-pinned")
            if leaf.exception.provenance.leaf.slot != leaf.slot:
                raise ValueError("streaming exception item and raw leaf must use the same slot")
            exceptions.append(StreamingRawException(leaf.exception, leaf.tokens))
            if len(exceptions) > contract.sparse.capacity:
                raise ValueError(
                    "streaming materialization requires exactly "
                    f"{contract.sparse.capacity} exceptions"
                )
            continue

        encoded = fusion.encode_stream(leaf.tokens.to(dtype=fusion_dtype))
        _validate_operator(
            encoded,
            token_shape=observed_shape,
            operator_size=fusion.operator_size,
            dtype=contract.dense_accumulation_dtype,
            device=fusion_device,
        )
        centralized = encoded if centralized is None else encoded @ centralized
        balanced.append(encoded)

    if count != contract.history_items:
        raise ValueError(f"streaming history requires exactly {contract.history_items} leaves")
    if len(exceptions) != contract.sparse.capacity:
        raise ValueError(
            f"streaming materialization requires exactly {contract.sparse.capacity} exceptions"
        )
    if centralized is None or token_shape is None or token_dtype is None:
        raise ValueError("streaming materialization requires an ordinary leaf")
    balanced_operator = balanced.finish()
    audit = build_streaming_equivalence_audit(
        centralized,
        balanced_operator,
        history_items=contract.history_items,
        ordinary_leaf_count=contract.history_items - len(exceptions),
        excluded_leaf_count=len(exceptions),
        atol=contract.equivalence_atol,
        rtol=contract.equivalence_rtol,
    )
    core_tokens = fusion.decode_operator(centralized)
    return StreamingLongContextMaterialization(
        contract,
        core_tokens,
        centralized,
        tuple(exceptions),
        audit,
    )


__all__ = (
    "StreamingEquivalenceAudit",
    "StreamingHistoryLeaf",
    "StreamingLongContextContractV1",
    "StreamingLongContextMaterialization",
    "StreamingRawException",
    "build_streaming_equivalence_audit",
    "materialize_streaming_long_context",
)
