"""Tensor-native N=128 memory adapter for the existing LTX guide boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import torch

from duet.duetx.contracts import tensor_sha256
from duet.duetx.guide_boundary import (
    LTXGuideProvenance,
    LTXHistoryGuidePayload,
    LTXLatentGuide,
)
from duet.duetx.streaming_long_context import StreamingLongContextMaterialization

_HISTORY_ITEMS = 128
_EXCEPTION_COUNT = 2
_CHANNELS = 128
_OPERATOR_SIZE = 16
_MANDATORY_ITEM_ID = "n128-mandatory-guide"
_SHA256_HEX = frozenset("0123456789abcdef")


def _sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite_ltx_latent(value: object, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.ndim != 5
        or value.shape[0] != 1
        or value.shape[1] != _CHANNELS
        or any(dimension <= 0 for dimension in value.shape)
        or value.dtype != torch.float32
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(f"{name} must be detached finite float32 [1,128,F,H,W]")
    return value


@dataclass(frozen=True, slots=True)
class N128LTXGuideMemory:
    """Runtime-only calibrated core plus the exact streaming N=128/K=2 state."""

    materialization: StreamingLongContextMaterialization
    core_latent: torch.Tensor
    core_item_ids: tuple[str, ...]
    checkpoint_sha256: str
    core_modules_sha256: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> Self:
        if not isinstance(self.materialization, StreamingLongContextMaterialization):
            raise ValueError("N=128 guide memory requires a streaming materialization")
        materialization = self.materialization
        contract = materialization.contract.validate()
        if (
            contract.history_items != _HISTORY_ITEMS
            or contract.sparse.capacity != _EXCEPTION_COUNT
            or contract.sparse.max_pinned != _EXCEPTION_COUNT
            or contract.latent_channels != _CHANNELS
            or contract.dense_accumulation_dtype != torch.float32
        ):
            raise ValueError("guide memory requires the exact N=128/K=2/C=128 FP32 contract")
        if tuple(materialization.core_operator.shape[-2:]) != (
            _OPERATOR_SIZE,
            _OPERATOR_SIZE,
        ):
            raise ValueError("guide memory requires the frozen operator size 16")
        core = _finite_ltx_latent(self.core_latent, "calibrated core latent")
        token_count = int(core.shape[2] * core.shape[3] * core.shape[4])
        if tuple(materialization.core_tokens.shape) != (1, token_count, _CHANNELS):
            raise ValueError("calibrated core latent token boundary differs from streaming memory")
        if (
            materialization.core_tokens.dtype != core.dtype
            or materialization.core_tokens.device != core.device
        ):
            raise ValueError("calibrated core latent must match streaming memory dtype and device")
        if (
            not isinstance(self.core_item_ids, tuple)
            or len(self.core_item_ids) != _HISTORY_ITEMS - _EXCEPTION_COUNT
            or any(not isinstance(item_id, str) or not item_id for item_id in self.core_item_ids)
            or len(set(self.core_item_ids)) != len(self.core_item_ids)
        ):
            raise ValueError("core item IDs must identify exactly 126 unique ordinary leaves")
        exception_ids = tuple(exception.item.item_id for exception in materialization.exceptions)
        all_ids = (*self.core_item_ids, *exception_ids)
        if set(self.core_item_ids).intersection(exception_ids) or _MANDATORY_ITEM_ID in all_ids:
            raise ValueError("core, exception, and mandatory item IDs must be disjoint")
        _sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _sha256(self.core_modules_sha256, "core_modules_sha256")
        return self


class _CoreOnlyLTXHistoryGuidePayload(LTXHistoryGuidePayload):
    """Exact core-plus-current payload without changing historical payload modes."""

    def validate(self) -> Self:
        if (
            self.mode != "bounded"
            or self.core is None
            or self.exceptions != ()
            or not isinstance(self.mandatory, tuple)
            or len(self.mandatory) != 1
        ):
            raise ValueError("core-only mode requires one core and one mandatory guide")
        materialized = self.materialized()
        if tuple((role, guide.ordinal) for role, guide in materialized) != (
            ("core", 0),
            ("mandatory", 0),
        ):
            raise ValueError("core-only guides require the canonical core and mandatory roster")
        for _role, guide in materialized:
            if not isinstance(guide, LTXLatentGuide):
                raise ValueError("core-only payload entries must be LTXLatentGuide values")
            guide.validate()
        core = materialized[0][1].latent
        mandatory = materialized[1][1].latent
        if core.shape != mandatory.shape:
            raise ValueError("core-only LTX guides must have the same shape")
        if core.dtype != mandatory.dtype or core.device != mandatory.device:
            raise ValueError("core-only LTX guides must have the same dtype and device")
        if (
            not isinstance(self.provenance, tuple)
            or len(self.provenance) != 2
            or any(not isinstance(entry, LTXGuideProvenance) for entry in self.provenance)
        ):
            raise ValueError("core-only guide provenance must align exactly with both guides")
        for entry, (role, guide) in zip(self.provenance, materialized, strict=True):
            entry.validate()
            if (entry.role, entry.ordinal) != (role, guide.ordinal):
                raise ValueError("core-only guide provenance must align exactly with both guides")
        if set(self.provenance[0].item_ids).intersection(self.provenance[1].item_ids):
            raise ValueError("core and mandatory provenance must be disjoint")
        return self


def _latent_from_tokens(tokens: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    batch, channels, frames, height, width = reference.shape
    expected_shape = (batch, frames * height * width, channels)
    if tuple(tokens.shape) != expected_shape or tokens.dtype != reference.dtype:
        raise ValueError("raw exception token boundary differs from the calibrated core")
    source_hash = tensor_sha256(tokens)
    latent = (
        tokens.to(device=reference.device)
        .reshape(batch, frames, height, width, channels)
        .permute(0, 4, 1, 2, 3)
        .contiguous()
    )
    round_trip = latent.permute(0, 2, 3, 4, 1).flatten(1, 3).contiguous()
    if tensor_sha256(round_trip) != source_hash:
        raise ValueError("raw exception bytes changed during LTX latent conversion")
    return latent


def build_n128_ltx_history_guides(
    memory: N128LTXGuideMemory,
    mandatory: Mapping[str, object],
) -> LTXHistoryGuidePayload:
    """Map one exact N=128/K=2 memory into the existing bounded LTX payload."""
    if not isinstance(memory, N128LTXGuideMemory):
        raise ValueError("memory must be an N128LTXGuideMemory")
    memory.validate()
    if not isinstance(mandatory, Mapping):
        raise ValueError("mandatory guide must be a Comfy LATENT mapping")
    mandatory_samples = _finite_ltx_latent(mandatory.get("samples"), "mandatory guide")
    if mandatory_samples.shape != memory.core_latent.shape:
        raise ValueError(
            "calibrated core, exceptions, and mandatory guide must have the same shape"
        )
    if (
        mandatory_samples.dtype != memory.core_latent.dtype
        or mandatory_samples.device != memory.core_latent.device
    ):
        raise ValueError("mandatory guide must match memory dtype and device")

    exceptions = tuple(
        _latent_from_tokens(exception.raw_leaf, memory.core_latent)
        for exception in memory.materialization.exceptions
    )
    return LTXHistoryGuidePayload(
        mode="bounded",
        core=LTXLatentGuide(memory.core_latent, 0.75, 0),
        exceptions=(
            LTXLatentGuide(exceptions[0], 1.0, 0),
            LTXLatentGuide(exceptions[1], 0.9, 1),
        ),
        mandatory=(LTXLatentGuide(mandatory_samples, 1.0, 0),),
        provenance=(
            LTXGuideProvenance("core", 0, memory.core_item_ids),
            *tuple(
                LTXGuideProvenance(
                    "exception",
                    ordinal,
                    (exception.item.item_id,),
                    (exception.item.provenance,),
                )
                for ordinal, exception in enumerate(memory.materialization.exceptions)
            ),
            LTXGuideProvenance("mandatory", 0, (_MANDATORY_ITEM_ID,)),
        ),
    )


def build_n128_ltx_core_only_history_guides(
    memory: N128LTXGuideMemory,
    mandatory: Mapping[str, object],
) -> LTXHistoryGuidePayload:
    """Map the calibrated core and mandatory current guide without exceptions."""
    if not isinstance(memory, N128LTXGuideMemory):
        raise ValueError("memory must be an N128LTXGuideMemory")
    memory.validate()
    if not isinstance(mandatory, Mapping):
        raise ValueError("mandatory guide must be a Comfy LATENT mapping")
    mandatory_samples = _finite_ltx_latent(mandatory.get("samples"), "mandatory guide")
    if mandatory_samples.shape != memory.core_latent.shape:
        raise ValueError("calibrated core and mandatory guide must have the same shape")
    if (
        mandatory_samples.dtype != memory.core_latent.dtype
        or mandatory_samples.device != memory.core_latent.device
    ):
        raise ValueError("mandatory guide must match memory dtype and device")
    return _CoreOnlyLTXHistoryGuidePayload(
        mode="bounded",
        core=LTXLatentGuide(memory.core_latent, 0.75, 0),
        exceptions=(),
        mandatory=(LTXLatentGuide(mandatory_samples, 1.0, 0),),
        provenance=(
            LTXGuideProvenance("core", 0, memory.core_item_ids),
            LTXGuideProvenance("mandatory", 0, (_MANDATORY_ITEM_ID,)),
        ),
    )


__all__ = (
    "N128LTXGuideMemory",
    "build_n128_ltx_core_only_history_guides",
    "build_n128_ltx_history_guides",
)
