"""Immutable provenance, exception-memory, and coverage contracts for Duet-X."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from typing import Any, Self

import torch

from duet.duetx.config import Decision1Config

_LTX_FORMAT = "duet-x-ltx-decision1-v1"
_MINIMAX_H3_FORMAT = "duet-x-minimax-h3-v1"
_SHA256_HEX = frozenset("0123456789abcdef")
_Q16_MAX = 65_536
SALIENCE_Q_MIN = -(2**63)
SALIENCE_Q_MAX = 2**63 - 1
_EVIDENCE_LOCATOR = re.compile(
    r"duet-evidence://(?P<registry>[A-Za-z0-9._-]+)/sha256/(?P<digest>[0-9a-f]{64})\Z"
)


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _SHA256_HEX for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _evidence_locator(value: str, content_sha256: str) -> None:
    match = _EVIDENCE_LOCATOR.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("locator must use the canonical duet-evidence content-addressed grammar")
    if match.group("registry") in {".", ".."}:
        raise ValueError("locator registry must not be a relative component")
    if match.group("digest") != content_sha256:
        raise ValueError("locator digest must match content_sha256")


def _salience_q(value: int) -> None:
    if type(value) is not int or not SALIENCE_Q_MIN <= value <= SALIENCE_Q_MAX:
        raise ValueError("salience_q must be a signed int64 integer")


@dataclass(frozen=True, slots=True)
class LeafKey:
    """The canonical source and fixed history slot that produced an item."""

    source_id: str
    source_rank: int
    slot: int

    def validate(self) -> Self:
        _nonempty(self.source_id, "source_id")
        if type(self.source_rank) is not int or self.source_rank < 0:
            raise ValueError("source_rank must be a nonnegative integer")
        if type(self.slot) is not int or self.slot < 0:
            raise ValueError("slot must be a nonnegative integer")
        return self


@dataclass(frozen=True, slots=True)
class TimeFrameRange:
    """A nonempty half-open timestamp and frame range from one source."""

    timestamp_start_ns: int
    timestamp_stop_ns: int
    frame_start: int
    frame_stop: int

    def validate(self) -> Self:
        if (
            type(self.timestamp_start_ns) is not int
            or type(self.timestamp_stop_ns) is not int
            or self.timestamp_start_ns < 0
            or self.timestamp_stop_ns <= self.timestamp_start_ns
        ):
            raise ValueError("timestamp range must be nonempty and half-open")
        if (
            type(self.frame_start) is not int
            or type(self.frame_stop) is not int
            or self.frame_start < 0
            or self.frame_stop <= self.frame_start
        ):
            raise ValueError("frame range must be nonempty and half-open")
        return self


@dataclass(frozen=True, slots=True)
class SpatialLocation:
    """A q16 normalized bounding box or polygon in a named coordinate system."""

    coordinate_system: str
    geometry: str
    coordinates_q16: tuple[int, ...]

    def validate(self) -> Self:
        _nonempty(self.coordinate_system, "coordinate_system")
        if self.geometry not in {"bbox", "polygon"}:
            raise ValueError("geometry must be bbox or polygon")
        if not isinstance(self.coordinates_q16, tuple) or any(
            type(coordinate) is not int or not 0 <= coordinate <= _Q16_MAX
            for coordinate in self.coordinates_q16
        ):
            raise ValueError("q16 coordinates must be integers in [0, 65536]")
        if self.geometry == "bbox":
            if len(self.coordinates_q16) != 4:
                raise ValueError("bbox must have exactly four q16 coordinates")
            left, top, right, bottom = self.coordinates_q16
            if right <= left or bottom <= top:
                raise ValueError("bbox must have positive area")
        else:
            if len(self.coordinates_q16) < 6 or len(self.coordinates_q16) % 2:
                raise ValueError("polygon must have at least three q16 points")
            vertices = tuple(
                zip(self.coordinates_q16[::2], self.coordinates_q16[1::2], strict=True)
            )
            if len(set(vertices)) < 3:
                raise ValueError("polygon must have at least three distinct vertices")
            signed_area_twice = sum(
                left[0] * right[1] - right[0] * left[1]
                for left, right in zip(vertices, vertices[1:] + vertices[:1], strict=True)
            )
            if signed_area_twice == 0:
                raise ValueError("polygon must have nonzero signed area")
        return self


@dataclass(frozen=True, slots=True)
class RawEvidencePointer:
    """Credential-free raw-media locator and its independently verified content hash."""

    locator: str
    content_sha256: str
    byte_start: int
    byte_stop: int

    def validate(self) -> Self:
        _sha256(self.content_sha256, "content_sha256")
        _evidence_locator(self.locator, self.content_sha256)
        if (
            type(self.byte_start) is not int
            or type(self.byte_stop) is not int
            or self.byte_start < 0
            or self.byte_stop <= self.byte_start
        ):
            raise ValueError("byte range must be nonempty and half-open")
        return self


@dataclass(frozen=True, slots=True)
class EvidenceProvenance:
    """Versioned sidecar provenance for one immutable exception candidate."""

    leaf: LeafKey
    time: TimeFrameRange
    spatial: SpatialLocation
    event_type: str
    raw: RawEvidencePointer
    source_registry_sha256: str
    preprocessing_sha256: str
    vae_sha256: str
    adapter_sha256: str
    scorer_sha256: str
    memory_contract_sha256: str

    @property
    def source_rank(self) -> int:
        """Expose source rank at the rank-key boundary without duplicating it."""
        return self.leaf.source_rank

    def validate(self) -> Self:
        self.leaf.validate()
        self.time.validate()
        self.spatial.validate()
        _nonempty(self.event_type, "event_type")
        self.raw.validate()
        for name in (
            "source_registry_sha256",
            "preprocessing_sha256",
            "vae_sha256",
            "adapter_sha256",
            "scorer_sha256",
            "memory_contract_sha256",
        ):
            _sha256(getattr(self, name), name)
        return self


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous CPU bytes without device identity."""
    if tensor.layout != torch.strided:
        raise ValueError("tensor_sha256 requires a strided tensor")
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    header = json.dumps(
        {"dtype": _dtype_name(tensor.dtype), "shape": list(tensor.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\n" + raw).hexdigest()


@dataclass(frozen=True, slots=True)
class SparseMemoryContract:
    """The fixed bounded exception-memory representation."""

    capacity: int
    embedding_dim: int
    embedding_dtype: torch.dtype
    max_pinned: int | None = None

    def validate(self) -> Self:
        if type(self.capacity) is not int or self.capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if type(self.embedding_dim) is not int or self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        if self.embedding_dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            raise ValueError("embedding_dtype must be a floating torch dtype")
        if self.max_pinned is not None and (
            type(self.max_pinned) is not int
            or self.max_pinned < 0
            or self.max_pinned > self.capacity
        ):
            raise ValueError("max_pinned must be within capacity")
        return self

    def fingerprint(self) -> str:
        self.validate()
        payload = {
            "capacity": self.capacity,
            "embedding_dim": self.embedding_dim,
            "embedding_dtype": _dtype_name(self.embedding_dtype),
            "max_pinned": self.max_pinned,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class DuetXContract:
    """The Decision 1 format lock consumed by sparse and dense state implementations."""

    format: str
    decision1_fingerprint: str
    sparse: SparseMemoryContract
    history_items: int
    latent_channels: int
    dense_accumulation_dtype: torch.dtype

    @classmethod
    def default(cls, *, decision1_fingerprint: str) -> Self:
        return cls(
            _LTX_FORMAT,
            decision1_fingerprint,
            SparseMemoryContract(2, 128, torch.float32, 2),
            8,
            128,
            torch.float32,
        )

    @classmethod
    def minimax_h3(cls, *, adapter_fingerprint: str) -> Self:
        """Build the exact native-latent MiniMax H3 memory contract."""
        return cls(
            _MINIMAX_H3_FORMAT,
            adapter_fingerprint,
            SparseMemoryContract(2, 24, torch.float32, 2),
            8,
            24,
            torch.float32,
        )

    @classmethod
    def from_config(cls, config: Decision1Config) -> Self:
        config.validate()
        return cls(
            config.format,
            config.fingerprint(),
            SparseMemoryContract(
                config.experiment.protected_exceptions,
                config.runtime.latent_channels,
                torch.float32,
                config.experiment.protected_exceptions,
            ),
            config.experiment.history_items,
            config.runtime.latent_channels,
            torch.float32,
        )

    def validate(self, expected_decision1_fingerprint: str | None = None) -> Self:
        if self.format not in {_LTX_FORMAT, _MINIMAX_H3_FORMAT}:
            raise ValueError(f"unsupported format: {self.format!r}")
        _sha256(self.decision1_fingerprint, "decision1_fingerprint")
        self.sparse.validate()
        if self.history_items != 8:
            raise ValueError("Decision 1 fixes history_items to 8")
        if self.sparse.capacity != 2 or self.sparse.max_pinned != 2:
            raise ValueError("Decision 1 fixes protected exceptions to 2")
        expected_channels = 128 if self.format == _LTX_FORMAT else 24
        if self.latent_channels != expected_channels:
            raise ValueError(f"{self.format} fixes latent_channels to {expected_channels}")
        if self.format == _MINIMAX_H3_FORMAT and self.sparse.embedding_dim != 24:
            raise ValueError("MiniMax H3 fixes sparse embedding_dim to 24")
        if self.dense_accumulation_dtype != torch.float32:
            raise ValueError("Decision 1 fixes dense_accumulation_dtype to torch.float32")
        if expected_decision1_fingerprint is not None:
            _sha256(expected_decision1_fingerprint, "expected_decision1_fingerprint")
            if self.decision1_fingerprint != expected_decision1_fingerprint:
                raise ValueError("Decision 1 fingerprint mismatch")
        return self

    def label(self) -> str:
        """Return the canonical JSON representation of this Decision 1 contract."""
        self.validate()
        return json.dumps(
            {
                "decision1_fingerprint": self.decision1_fingerprint,
                "dense_accumulation_dtype": _dtype_name(self.dense_accumulation_dtype),
                "format": self.format,
                "history_items": self.history_items,
                "latent_channels": self.latent_channels,
                "sparse": self.sparse.fingerprint(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def fingerprint(self) -> str:
        self.validate()
        return hashlib.sha256(self.label().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Coverage:
    """A half-open range of canonical history slots, including an identity range."""

    start_slot: int
    stop_slot: int

    def validate(self) -> Self:
        if (
            type(self.start_slot) is not int
            or type(self.stop_slot) is not int
            or self.start_slot < 0
            or self.stop_slot < self.start_slot
        ):
            raise ValueError("coverage must be a nonnegative half-open slot range")
        return self


def _canonical(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": _dtype_name(value.dtype),
            "sha256": tensor_sha256(value),
            "shape": list(value.shape),
        }
    if isinstance(value, torch.dtype):
        return _dtype_name(value)
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    return value


def canonical_item_sha256(item: ExceptionItem) -> str:
    """Return the content identity of an exception item from canonical typed data."""
    payload = json.dumps(
        _canonical(
            {
                "item_id": item.item_id,
                "provenance": item.provenance,
                "embedding": item.embedding,
                "salience_q": item.salience_q,
                "pinned": item.pinned,
            }
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _ExceptionEmbedding:
    _embedding_bytes: bytes
    _embedding_dtype: torch.dtype
    _embedding_shape: tuple[int, ...]

    @property
    def embedding(self) -> torch.Tensor:
        """Reconstruct a fresh CPU tensor from immutable canonical embedding bytes."""
        return torch.frombuffer(
            bytearray(self._embedding_bytes), dtype=self._embedding_dtype
        ).reshape(self._embedding_shape)


@dataclass(frozen=True, slots=True)
class ExceptionItem(_ExceptionEmbedding):
    item_id: str
    provenance: EvidenceProvenance
    embedding: InitVar[torch.Tensor] = field()
    salience_q: int
    pinned: bool = False
    _embedding_bytes: bytes = field(init=False, repr=False)
    _embedding_dtype: torch.dtype = field(init=False, repr=False)
    _embedding_shape: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self, embedding: torch.Tensor) -> None:
        if not isinstance(embedding, torch.Tensor) or embedding.layout != torch.strided:
            raise ValueError("exception embedding must be a strided tensor")
        canonical = embedding.detach().contiguous().cpu()
        object.__setattr__(self, "_embedding_bytes", canonical.view(torch.uint8).numpy().tobytes())
        object.__setattr__(self, "_embedding_dtype", canonical.dtype)
        object.__setattr__(self, "_embedding_shape", tuple(canonical.shape))

    def validate(self, contract: SparseMemoryContract) -> None:
        contract.validate()
        _nonempty(self.item_id, "item_id")
        self.provenance.validate()
        if self.provenance.memory_contract_sha256 != contract.fingerprint():
            raise ValueError("exception provenance memory contract fingerprint mismatch")
        embedding = self.embedding
        if tuple(embedding.shape) != (contract.embedding_dim,):
            raise ValueError("exception embedding shape mismatch")
        if embedding.dtype != contract.embedding_dtype:
            raise ValueError("exception embedding dtype mismatch")
        if not bool(torch.isfinite(embedding).all().item()):
            raise ValueError("exception embedding must be finite")
        _salience_q(self.salience_q)
        if type(self.pinned) is not bool:
            raise ValueError("pinned must be a bool")

    def fingerprint(self) -> str:
        return canonical_item_sha256(self)
