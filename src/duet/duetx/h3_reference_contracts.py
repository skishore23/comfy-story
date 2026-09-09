"""Model-neutral contracts for bounded MiniMax H3 visual reference context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Self

import torch

from duet.duetx.contracts import tensor_sha256

_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_FORMAT = "duet-x-h3-reference-compile-receipt-v1"


class H3ReferenceMethod(StrEnum):
    """Internal methods compared by the H3 compiler MVP."""

    NATIVE_FULL = "native_full"
    NATIVE_TRIMMED = "native_trimmed"
    GATED = "gated"
    RESAMPLER = "resampler"
    DUET = "duet"
    EXCEPTIONS_ONLY = "exceptions_only"
    DUET_X = "duet_x"


class H3ReferenceKind(StrEnum):
    """Visual H3 reference kinds supported by the first compiler."""

    IMAGE = "image"
    VIDEO = "video"


def _validate_source_id(value: object, field: str = "source_id") -> str:
    if not isinstance(value, str) or _SOURCE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be a portable nonempty identifier")
    return value


def _validate_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_latent(value: object, *, kind: H3ReferenceKind) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.ndim != 5
        or value.shape[0] != 1
        or value.shape[1] != 24
        or any(dimension <= 0 for dimension in value.shape[2:])
    ):
        raise ValueError("H3 visual latent must be a strided [1,24,T,H,W] tensor")
    if value.shape[3] % 2 or value.shape[4] % 2:
        raise ValueError("H3 visual latent height and width must be even")
    if kind is H3ReferenceKind.IMAGE and value.shape[2] != 1:
        raise ValueError("H3 image reference latent must contain exactly one temporal token")
    if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all().item()):
        raise ValueError("H3 visual latent must be finite and floating")
    return value


def _visual_rows(latent: torch.Tensor) -> int:
    return latent.shape[2] * (latent.shape[3] // 2) * (latent.shape[4] // 2)


@dataclass(frozen=True, slots=True)
class H3SelectedSource:
    """Ordered identity shared by Qwen presentation and H3 latent blocks."""

    source_id: str
    kind: H3ReferenceKind
    ordinal: int
    protected: bool

    def validate(self) -> Self:
        _validate_source_id(self.source_id)
        if not isinstance(self.kind, H3ReferenceKind):
            raise ValueError("H3 selected source kind is unsupported")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("H3 selected source ordinal must be nonnegative")
        if type(self.protected) is not bool:
            raise ValueError("H3 selected source protected flag must be boolean")
        return self


@dataclass(frozen=True, slots=True)
class H3VisualReference:
    """One validated H3-native visual reference latent."""

    source_id: str
    kind: H3ReferenceKind
    ordinal: int
    latent: torch.Tensor
    protected: bool

    def validate(self) -> Self:
        H3SelectedSource(self.source_id, self.kind, self.ordinal, self.protected).validate()
        _validate_latent(self.latent, kind=self.kind)
        return self

    @property
    def visual_rows(self) -> int:
        self.validate()
        return _visual_rows(self.latent)

    @property
    def tensor_sha256(self) -> str:
        self.validate()
        return tensor_sha256(self.latent)


@dataclass(frozen=True, slots=True)
class H3ReferenceBudget:
    """Hard output boundary shared by every bounded method."""

    max_visual_rows: int
    max_exceptions: int

    def validate(self) -> Self:
        if type(self.max_visual_rows) is not int or self.max_visual_rows <= 0:
            raise ValueError("max_visual_rows must be a positive integer")
        if type(self.max_exceptions) is not int or not 0 <= self.max_exceptions <= 2:
            raise ValueError("max_exceptions must be in range [0,2]")
        return self


@dataclass(frozen=True, slots=True)
class H3CompiledBlock:
    """One ordinary H3 reference block emitted by the compiler."""

    kind: H3ReferenceKind
    latent: torch.Tensor
    source_ids: tuple[str, ...]
    exact: bool

    def validate(self) -> Self:
        if not isinstance(self.kind, H3ReferenceKind):
            raise ValueError("H3 compiled block kind is unsupported")
        _validate_latent(self.latent, kind=self.kind)
        if (
            not isinstance(self.source_ids, tuple)
            or not self.source_ids
            or len(self.source_ids) != len(set(self.source_ids))
        ):
            raise ValueError("compiled block source_ids must be unique and nonempty")
        for source_id in self.source_ids:
            _validate_source_id(source_id, "compiled block source_id")
        if type(self.exact) is not bool:
            raise ValueError("compiled block exact flag must be boolean")
        if self.exact and (self.kind is not H3ReferenceKind.IMAGE or len(self.source_ids) != 1):
            raise ValueError("exact MVP blocks must contain one image source")
        return self

    @property
    def visual_rows(self) -> int:
        self.validate()
        return _visual_rows(self.latent)

    @property
    def tensor_sha256(self) -> str:
        self.validate()
        return tensor_sha256(self.latent)

    def to_minimax_ref(self) -> dict[str, object]:
        """Return the exact visual dictionary consumed by ComfyUI H3."""
        self.validate()
        block: dict[str, object] = {
            "kind": self.kind.value,
            "latent_h": self.latent.shape[3],
            "latent_w": self.latent.shape[4],
            "latent": self.latent,
        }
        if self.kind is H3ReferenceKind.VIDEO:
            block.update(
                {
                    "latent_t": self.latent.shape[2],
                    "ref_audio_t": 0,
                    "audio_latent": None,
                }
            )
        return block


CacheStatus = Literal["bypass", "miss", "hit", "updated"]


@dataclass(frozen=True, slots=True)
class H3CompileReceipt:
    """Auditable accounting for one reference compilation."""

    format: str
    method: H3ReferenceMethod
    input_semantic_items: int
    output_semantic_items: int
    input_visual_rows: int
    output_visual_rows: int
    source_sha256s: tuple[str, ...]
    output_sha256s: tuple[str, ...]
    checkpoint_sha256: str | None
    cache_status: CacheStatus
    padding_temporal_tokens: int
    trainable_parameters: int

    def validate(self) -> Self:
        if self.format != _RECEIPT_FORMAT:
            raise ValueError("H3 compile receipt format is unsupported")
        if not isinstance(self.method, H3ReferenceMethod):
            raise ValueError("H3 compile receipt method is unsupported")
        for field, value in (
            ("input_semantic_items", self.input_semantic_items),
            ("output_semantic_items", self.output_semantic_items),
            ("input_visual_rows", self.input_visual_rows),
            ("output_visual_rows", self.output_visual_rows),
            ("padding_temporal_tokens", self.padding_temporal_tokens),
            ("trainable_parameters", self.trainable_parameters),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.input_semantic_items <= 0 or self.output_semantic_items <= 0:
            raise ValueError("H3 compile receipt semantic item counts must be positive")
        if self.input_semantic_items != self.output_semantic_items:
            raise ValueError("MVP semantic roster must remain unchanged")
        if self.input_visual_rows <= 0 or self.output_visual_rows <= 0:
            raise ValueError("H3 compile receipt visual row counts must be positive")
        if len(self.source_sha256s) != self.input_semantic_items:
            raise ValueError("source hashes must align with the semantic roster")
        if not self.output_sha256s:
            raise ValueError("output hashes must be nonempty")
        for digest in self.source_sha256s:
            _validate_sha256(digest, "source_sha256")
        for digest in self.output_sha256s:
            _validate_sha256(digest, "output_sha256")
        learned = self.method in {
            H3ReferenceMethod.GATED,
            H3ReferenceMethod.RESAMPLER,
            H3ReferenceMethod.DUET,
            H3ReferenceMethod.DUET_X,
        }
        if learned:
            _validate_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        elif self.checkpoint_sha256 is not None:
            raise ValueError("non-learned methods must not bind a compiler checkpoint")
        if self.cache_status not in {"bypass", "miss", "hit", "updated"}:
            raise ValueError("H3 compile receipt cache status is unsupported")
        if type(self.trainable_parameters) is not int or self.trainable_parameters < 0:
            raise ValueError("trainable_parameters must be nonnegative")
        return self

    @property
    def compression_factor(self) -> float:
        self.validate()
        return self.input_visual_rows / self.output_visual_rows


@dataclass(frozen=True, slots=True)
class H3CompiledContext:
    """A bounded tuple of ordinary H3 reference blocks plus its receipt."""

    blocks: tuple[H3CompiledBlock, ...]
    receipt: H3CompileReceipt
    budget: H3ReferenceBudget

    def validate(self) -> Self:
        self.budget.validate()
        self.receipt.validate()
        if not isinstance(self.blocks, tuple) or not self.blocks:
            raise ValueError("compiled H3 context must contain at least one block")
        for block in self.blocks:
            if not isinstance(block, H3CompiledBlock):
                raise ValueError("compiled H3 context contains an invalid block")
            block.validate()
        source_ids = tuple(source for block in self.blocks for source in block.source_ids)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("compiled H3 context must not duplicate a source across blocks")
        rows = sum(block.visual_rows for block in self.blocks)
        if rows != self.receipt.output_visual_rows:
            raise ValueError("compiled H3 context row count differs from its receipt")
        if rows > self.budget.max_visual_rows:
            raise ValueError("compiled H3 context exceeds its visual row budget")
        exact_count = sum(block.exact for block in self.blocks)
        if exact_count > self.budget.max_exceptions:
            raise ValueError("compiled H3 context exceeds its exception budget")
        hashes = tuple(block.tensor_sha256 for block in self.blocks)
        if hashes != self.receipt.output_sha256s:
            raise ValueError("compiled H3 context output hashes differ from its receipt")
        return self


__all__ = (
    "CacheStatus",
    "H3CompileReceipt",
    "H3CompiledBlock",
    "H3CompiledContext",
    "H3ReferenceBudget",
    "H3ReferenceKind",
    "H3ReferenceMethod",
    "H3SelectedSource",
    "H3VisualReference",
)
