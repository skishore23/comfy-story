"""Matched-budget history methods for the Duet-X Decision 1 canary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self, cast

import torch
from torch import nn

from comfy_story.config import Method
from comfy_story.contracts import EvidenceProvenance, ExceptionItem
from comfy_story.guide_boundary import (
    GuidePayloadMode,
    LTXGuideProvenance,
    LTXHistoryGuidePayload,
    LTXLatentGuide,
)
from comfy_story.latent_bridge import LatentHistoryBridge

_HISTORY_ITEMS = 8
_LATENT_CHANNELS = 128
_GATED_HIDDEN_WIDTH = 635
_RESAMPLER_ATTENTION_WIDTH = 215


@dataclass(frozen=True, slots=True)
class MethodBatch:
    """Exact, framework-neutral input shared by all seven methods."""

    history: torch.Tensor
    item_ids: tuple[str, ...]
    evidence: tuple[EvidenceProvenance, ...]
    selected_indices: tuple[int, int]
    mandatory: LTXLatentGuide
    mandatory_provenance: LTXGuideProvenance

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> Self:
        if (
            not isinstance(self.history, torch.Tensor)
            or self.history.layout != torch.strided
            or self.history.ndim != 6
            or self.history.shape[0] != 1
            or self.history.shape[1] != _HISTORY_ITEMS
            or self.history.shape[2] != _LATENT_CHANNELS
            or any(dimension <= 0 for dimension in self.history.shape)
        ):
            raise ValueError("history must be a strided [1,8,128,F,H,W] tensor")
        if not torch.is_floating_point(self.history):
            raise ValueError("history must use a floating dtype")
        if not bool(torch.isfinite(self.history).all().item()):
            raise ValueError("history must be finite")
        if (
            not isinstance(self.item_ids, tuple)
            or len(self.item_ids) != _HISTORY_ITEMS
            or any(not isinstance(item_id, str) or not item_id for item_id in self.item_ids)
            or len(set(self.item_ids)) != _HISTORY_ITEMS
        ):
            raise ValueError("item_ids must contain eight unique nonempty strings")
        if not isinstance(self.evidence, tuple) or len(self.evidence) != _HISTORY_ITEMS:
            raise ValueError("evidence must align with all eight history items")
        for index, provenance in enumerate(self.evidence):
            if not isinstance(provenance, EvidenceProvenance):
                raise ValueError("evidence must contain EvidenceProvenance values")
            provenance.validate()
            if provenance.leaf.slot != index:
                raise ValueError("evidence must align with canonical history slots")
        if not isinstance(self.selected_indices, tuple) or len(self.selected_indices) != 2:
            raise ValueError("selected_indices must contain exactly two indices")
        left, right = self.selected_indices
        if any(
            type(index) is not int or not 0 <= index < _HISTORY_ITEMS for index in (left, right)
        ):
            raise ValueError("selected_indices must be in range [0,8)")
        if left == right:
            raise ValueError("selected_indices must be distinct")
        if left > right:
            raise ValueError("selected_indices must be in canonical ascending order")
        if not isinstance(self.mandatory, LTXLatentGuide):
            raise ValueError("mandatory must be an LTXLatentGuide")
        self.mandatory.validate()
        expected_shape = (
            self.history.shape[0],
            self.history.shape[2],
            self.history.shape[3],
            self.history.shape[4],
            self.history.shape[5],
        )
        if tuple(self.mandatory.latent.shape) != expected_shape:
            raise ValueError("mandatory latent shape must match one history item")
        if (
            self.mandatory.latent.dtype != self.history.dtype
            or self.mandatory.latent.device != self.history.device
        ):
            raise ValueError("mandatory latent dtype and device must match history")
        if self.mandatory.ordinal != 0:
            raise ValueError("mandatory guide ordinal must be zero")
        if not isinstance(self.mandatory_provenance, LTXGuideProvenance):
            raise ValueError("mandatory_provenance must be LTXGuideProvenance")
        self.mandatory_provenance.validate()
        if self.mandatory_provenance.role != "mandatory" or self.mandatory_provenance.ordinal != 0:
            raise ValueError("mandatory provenance must use mandatory ordinal zero")
        if not set(self.item_ids).isdisjoint(self.mandatory_provenance.item_ids):
            raise ValueError("mandatory provenance must be disjoint from history item_ids")
        return self


@dataclass(frozen=True, slots=True)
class MethodOutput:
    """One method result ready for the validated LTX guide boundary."""

    method: Method
    mode: GuidePayloadMode
    core: LTXLatentGuide | None
    exceptions: tuple[LTXLatentGuide, ...]
    mandatory: tuple[LTXLatentGuide, ...]
    provenance: tuple[LTXGuideProvenance, ...]

    def __post_init__(self) -> None:
        expected_mode: GuidePayloadMode
        if self.method is Method.FULL_HISTORY_TEACHER:
            expected_mode = "full_history_teacher"
        elif self.method is Method.TOPK_ONLY:
            expected_mode = "topk_only"
        elif isinstance(self.method, Method):
            expected_mode = "bounded"
        else:
            raise ValueError("method output method must be a Method")
        if self.mode != expected_mode:
            raise ValueError("method output method and payload mode do not match")
        self.to_payload()

    def to_payload(self) -> LTXHistoryGuidePayload:
        """Materialize the mode-locked payload without altering any latent tensor."""
        return LTXHistoryGuidePayload(
            mode=self.mode,
            core=self.core,
            exceptions=self.exceptions,
            mandatory=self.mandatory,
            provenance=self.provenance,
        )


class HistoryMethod(nn.Module):
    """Shared interface for one Decision 1 history method."""

    method: Method

    def __init__(self, method: Method) -> None:
        super().__init__()
        self.method = method

    def compress(self, history: torch.Tensor) -> torch.Tensor | None:
        """Return at most one core from the already-filtered history."""
        raise NotImplementedError

    def forward(self, batch: MethodBatch) -> MethodOutput:
        """Run this method through the common selection-removal boundary."""
        return run_method(self, batch)

    def trainable_parameter_count(self) -> int:
        """Return the number of trainable parameters used by this method."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class _NoCoreMethod(HistoryMethod):
    def compress(self, history: torch.Tensor) -> None:
        del history
        return None


class _RecentAnchorMethod(HistoryMethod):
    def compress(self, history: torch.Tensor) -> torch.Tensor:
        return history[:, -1]


class _MeanCoreMethod(HistoryMethod):
    def compress(self, history: torch.Tensor) -> torch.Tensor:
        return history.mean(dim=1)


class _GatedCoreMethod(HistoryMethod):
    def __init__(self, channels: int = _LATENT_CHANNELS) -> None:
        super().__init__(Method.GATED_CORE_TOPK)
        self.gate = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, _GATED_HIDDEN_WIDTH),
            nn.GELU(),
            nn.Linear(_GATED_HIDDEN_WIDTH, 1),
        )

    def compress(self, history: torch.Tensor) -> torch.Tensor:
        gate_dtype = next(self.gate.parameters()).dtype
        descriptors = history.mean(dim=(3, 4, 5)).to(dtype=gate_dtype)
        gates = self.gate(descriptors).sigmoid()
        weights = gates / gates.sum(dim=1, keepdim=True).clamp_min(torch.finfo(gates.dtype).eps)
        core = (history.to(dtype=weights.dtype) * weights[..., None, None, None]).sum(dim=1)
        return cast(torch.Tensor, core.to(dtype=history.dtype))


class _CentralizedResamplerMethod(HistoryMethod):
    def __init__(self, channels: int = _LATENT_CHANNELS) -> None:
        super().__init__(Method.CENTRALIZED_RESAMPLER_TOPK)
        width = _RESAMPLER_ATTENTION_WIDTH
        self.key = nn.Linear(channels, width, bias=False)
        self.value = nn.Linear(channels, width, bias=False)
        self.query = nn.Parameter(torch.empty(width))
        self.output = nn.Linear(width, channels, bias=False)
        nn.init.normal_(self.query, std=0.02)

    def compress(self, history: torch.Tensor) -> torch.Tensor:
        batch, streams, channels, frames, height, width = history.shape
        tokens = history.permute(0, 1, 3, 4, 5, 2).reshape(
            batch, streams, frames * height * width, channels
        )
        projection_dtype = self.key.weight.dtype
        projected = tokens.to(dtype=projection_dtype)
        logits = torch.einsum("bstw,w->bst", self.key(projected), self.query)
        weights = (logits / math.sqrt(self.query.numel())).softmax(dim=1).unsqueeze(-1)
        resampled = self.output((self.value(projected) * weights).sum(dim=1))
        core = (
            resampled.reshape(batch, frames, height, width, channels)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        return cast(torch.Tensor, core.to(dtype=history.dtype))


class _DuetCoreMethod(HistoryMethod):
    def __init__(self, channels: int = _LATENT_CHANNELS) -> None:
        super().__init__(Method.DUET_CORE_TOPK)
        self.bridge = LatentHistoryBridge(channels=channels, operator_size=16)

    def compress(self, history: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.bridge(history, anchor=history[:, -1]).core_latent)

    def compress_excluding(self, batch: MethodBatch) -> torch.Tensor:
        """Decode the exact mergeable root variant with both selected whole leaves absent."""
        selected = frozenset(batch.selected_indices)
        candidates = tuple(
            ExceptionItem(
                batch.item_ids[slot],
                batch.evidence[slot],
                batch.history[0, slot].detach().float().mean(dim=(1, 2, 3)),
                1_000_000 if slot in selected else 0,
            )
            for slot in range(_HISTORY_ITEMS)
        )
        result = self.bridge.forward_exclusion(batch.history, candidates=candidates)
        if frozenset(key.slot for key in result.excluded_leaf_keys) != selected:
            raise ValueError("mergeable Duet root excluded different leaves than the selector")
        return result.core_latent


def _guides_for_indices(
    batch: MethodBatch, indices: tuple[int, ...]
) -> tuple[tuple[LTXLatentGuide, ...], tuple[LTXGuideProvenance, ...]]:
    guides = tuple(
        LTXLatentGuide(batch.history[:, index], 1.0, ordinal)
        for ordinal, index in enumerate(indices)
    )
    provenance = tuple(
        LTXGuideProvenance(
            "exception",
            ordinal,
            (batch.item_ids[index],),
            (batch.evidence[index],),
        )
        for ordinal, index in enumerate(indices)
    )
    return guides, provenance


def run_method(method: HistoryMethod, batch: MethodBatch) -> MethodOutput:
    """Run one method after removing selected exceptions from every possible core."""
    if not isinstance(method, HistoryMethod):
        raise ValueError("method must be a HistoryMethod")
    if not isinstance(batch, MethodBatch):
        raise ValueError("batch must be a MethodBatch")
    batch.validate()
    mandatory = (batch.mandatory,)
    if method.method is Method.FULL_HISTORY_TEACHER:
        exceptions, exception_provenance = _guides_for_indices(batch, tuple(range(_HISTORY_ITEMS)))
        return MethodOutput(
            method.method,
            "full_history_teacher",
            None,
            exceptions,
            mandatory,
            (*exception_provenance, batch.mandatory_provenance),
        )

    selected = batch.selected_indices
    exceptions, exception_provenance = _guides_for_indices(batch, selected)
    if method.method is Method.TOPK_ONLY:
        return MethodOutput(
            method.method,
            "topk_only",
            None,
            exceptions,
            mandatory,
            (*exception_provenance, batch.mandatory_provenance),
        )

    selected_set = frozenset(selected)
    core_indices = tuple(index for index in range(_HISTORY_ITEMS) if index not in selected_set)
    core_history = batch.history[:, core_indices]
    core_latent = (
        method.compress_excluding(batch)
        if isinstance(method, _DuetCoreMethod)
        else method.compress(core_history)
    )
    if core_latent is None:
        raise ValueError("bounded core method did not produce a core")
    core = LTXLatentGuide(core_latent, 1.0, 0)
    core_provenance = LTXGuideProvenance(
        "core",
        0,
        tuple(batch.item_ids[index] for index in core_indices),
        tuple(batch.evidence[index] for index in core_indices),
    )
    return MethodOutput(
        method.method,
        "bounded",
        core,
        exceptions,
        mandatory,
        (core_provenance, *exception_provenance, batch.mandatory_provenance),
    )


def build_method(method: Method, *, channels: int = _LATENT_CHANNELS) -> HistoryMethod:
    """Build one of the seven locked Decision 1 method modules."""
    if not isinstance(method, Method):
        raise ValueError("method must be a Method")
    if type(channels) is not int or channels <= 0:
        raise ValueError("channels must be a positive integer")
    if method in {Method.FULL_HISTORY_TEACHER, Method.TOPK_ONLY}:
        return _NoCoreMethod(method)
    if method is Method.RECENT_ANCHOR_TOPK:
        return _RecentAnchorMethod(method)
    if method is Method.MEAN_CORE_TOPK:
        return _MeanCoreMethod(method)
    if method is Method.GATED_CORE_TOPK:
        return _GatedCoreMethod(channels)
    if method is Method.CENTRALIZED_RESAMPLER_TOPK:
        return _CentralizedResamplerMethod(channels)
    if method is Method.DUET_CORE_TOPK:
        return _DuetCoreMethod(channels)
    raise ValueError(f"unsupported method: {method!r}")


__all__ = ("HistoryMethod", "MethodBatch", "MethodOutput", "build_method", "run_method")
