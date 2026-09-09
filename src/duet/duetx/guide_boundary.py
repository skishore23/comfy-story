"""Framework-neutral contracts for ordered LTX latent-guide application."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self

import torch

from duet.duetx.contracts import EvidenceProvenance, tensor_sha256

GuideRole = Literal["core", "exception", "mandatory"]
GuidePayloadMode = Literal["bounded", "topk_only", "full_history_teacher"]


@dataclass(frozen=True, slots=True)
class LTXLatentGuide:
    """One LTX-shaped latent guide and its independent attention strength."""

    latent: torch.Tensor
    strength: float
    ordinal: int

    def validate(self) -> Self:
        if (
            not isinstance(self.latent, torch.Tensor)
            or self.latent.layout != torch.strided
            or self.latent.ndim != 5
            or self.latent.shape[1] != 128
        ):
            raise ValueError("LTX latent guide must be a strided [B,128,F,H,W] tensor")
        if any(dimension <= 0 for dimension in self.latent.shape):
            raise ValueError("LTX latent guide dimensions must be positive")
        if not torch.is_floating_point(self.latent):
            raise ValueError("LTX latent guide must use a floating dtype")
        if not bool(torch.isfinite(self.latent).all().item()):
            raise ValueError("LTX latent guide must be finite")
        if (
            isinstance(self.strength, bool)
            or not isinstance(self.strength, (int, float))
            or not math.isfinite(float(self.strength))
            or not 0.0 <= float(self.strength) <= 1.0
        ):
            raise ValueError("guide strength must be finite and within [0,1]")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("guide ordinal must be a nonnegative integer")
        return self


@dataclass(frozen=True, slots=True)
class LTXGuideProvenance:
    """One aligned, runtime-only provenance entry for a materialized guide."""

    role: GuideRole
    ordinal: int
    item_ids: tuple[str, ...]
    evidence: tuple[EvidenceProvenance, ...] = ()

    def validate(self) -> Self:
        if self.role not in {"core", "exception", "mandatory"}:
            raise ValueError("unknown guide provenance role")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("guide provenance ordinal must be a nonnegative integer")
        if (
            not isinstance(self.item_ids, tuple)
            or not self.item_ids
            or any(not isinstance(item_id, str) or not item_id for item_id in self.item_ids)
            or len(set(self.item_ids)) != len(self.item_ids)
        ):
            raise ValueError("guide provenance item_ids must be a nonempty unique string tuple")
        if not isinstance(self.evidence, tuple):
            raise ValueError("guide provenance evidence must be a tuple")
        if self.evidence and len(self.evidence) != len(self.item_ids):
            raise ValueError("guide provenance evidence must align with item_ids")
        for provenance in self.evidence:
            if not isinstance(provenance, EvidenceProvenance):
                raise ValueError("guide provenance evidence has an invalid type")
            provenance.validate()
        if self.role == "exception" and len(self.item_ids) != 1:
            raise ValueError("each exception guide must identify exactly one protected item")
        return self


@dataclass(frozen=True, slots=True)
class LTXHistoryGuidePayload:
    """A mode-locked history payload with an aligned sidecar manifest."""

    mode: GuidePayloadMode
    core: LTXLatentGuide | None
    exceptions: tuple[LTXLatentGuide, ...]
    mandatory: tuple[LTXLatentGuide, ...]
    provenance: tuple[LTXGuideProvenance, ...]

    def __post_init__(self) -> None:
        self.validate()

    def materialized(self) -> tuple[tuple[GuideRole, LTXLatentGuide], ...]:
        """Return the only valid LTX append order for this payload."""
        core = () if self.core is None else (("core", self.core),)
        exceptions = tuple(("exception", guide) for guide in self.exceptions)
        mandatory = tuple(("mandatory", guide) for guide in self.mandatory)
        return (*core, *exceptions, *mandatory)

    def validate(self) -> Self:
        if self.mode == "bounded":
            if (
                self.core is None
                or not isinstance(self.exceptions, tuple)
                or len(self.exceptions) != 2
                or not isinstance(self.mandatory, tuple)
                or len(self.mandatory) != 1
            ):
                raise ValueError(
                    "bounded mode requires one core, exactly two exceptions, "
                    "and one mandatory guide"
                )
            expected_guide_count = 4
        elif self.mode == "topk_only":
            if (
                self.core is not None
                or not isinstance(self.exceptions, tuple)
                or len(self.exceptions) != 2
                or not isinstance(self.mandatory, tuple)
                or len(self.mandatory) != 1
            ):
                raise ValueError(
                    "topk_only mode requires no core, exactly two exceptions, "
                    "and one mandatory guide"
                )
            expected_guide_count = 3
        elif self.mode == "full_history_teacher":
            if (
                self.core is not None
                or not isinstance(self.exceptions, tuple)
                or len(self.exceptions) != 8
                or not isinstance(self.mandatory, tuple)
                or len(self.mandatory) != 1
            ):
                raise ValueError(
                    "full_history_teacher mode requires no core, exactly eight raw history guides, "
                    "and one mandatory guide"
                )
            expected_guide_count = 9
        else:
            raise ValueError(
                "guide payload mode must be bounded, topk_only, or full_history_teacher"
            )
        materialized = self.materialized()
        if len(materialized) != expected_guide_count:
            raise ValueError("materialized guide count does not match the payload mode")
        shapes: set[tuple[int, ...]] = set()
        dtypes: set[torch.dtype] = set()
        devices: set[torch.device] = set()
        expected_ordinals: dict[GuideRole, int] = {
            "core": 0,
            "exception": 0,
            "mandatory": 0,
        }
        for role, guide in materialized:
            if not isinstance(guide, LTXLatentGuide):
                raise ValueError("guide payload entries must be LTXLatentGuide values")
            guide.validate()
            if guide.ordinal != expected_ordinals[role]:
                raise ValueError(f"{role} guide ordinals must be canonical and contiguous")
            expected_ordinals[role] += 1
            shapes.add(tuple(guide.latent.shape))
            dtypes.add(guide.latent.dtype)
            devices.add(guide.latent.device)
        if len(shapes) != 1:
            raise ValueError("all LTX guides must have the same shape")
        if len(dtypes) != 1 or len(devices) != 1:
            raise ValueError("all LTX guides must have the same dtype and device")
        if not isinstance(self.provenance, tuple) or len(self.provenance) != len(materialized):
            raise ValueError("guide provenance must align exactly with materialized guides")
        all_item_ids: set[str] = set()
        for sidecar, (role, guide) in zip(self.provenance, materialized, strict=True):
            if not isinstance(sidecar, LTXGuideProvenance):
                raise ValueError("guide provenance entries have an invalid type")
            sidecar.validate()
            if (sidecar.role, sidecar.ordinal) != (role, guide.ordinal):
                raise ValueError("guide provenance must align exactly with role and ordinal")
            overlap = all_item_ids.intersection(sidecar.item_ids)
            if overlap:
                raise ValueError("core, exception, and mandatory provenance must be disjoint")
            all_item_ids.update(sidecar.item_ids)
        return self


class LatentGuideBackend(Protocol):
    """Backend seam for one official LTX guide append operation."""

    def append_guide(
        self,
        positive: Any,
        negative: Any,
        latent: Mapping[str, object],
        guide: LTXLatentGuide,
        *,
        role: GuideRole,
    ) -> tuple[Any, Any, Mapping[str, object]]:
        """Append one guide or raise without reporting success."""
        ...


@dataclass(frozen=True, slots=True)
class LTXHistoryGuideReceipt:
    """Result created only after every backend append succeeds."""

    positive: Any
    negative: Any
    latent: Mapping[str, object]
    guide_count: int
    provenance: tuple[LTXGuideProvenance, ...]


def _target_samples(latent: Mapping[str, object]) -> torch.Tensor:
    if not isinstance(latent, Mapping):
        raise ValueError("target latent must be a mapping")
    samples = latent.get("samples")
    if not isinstance(samples, torch.Tensor):
        raise ValueError("target latent must contain tensor samples")
    return samples


def apply_history_guides(
    positive: Any,
    negative: Any,
    latent: Mapping[str, object],
    payload: LTXHistoryGuidePayload,
    backend: LatentGuideBackend,
) -> LTXHistoryGuideReceipt:
    """Apply ``core -> exceptions -> mandatory`` and return only after full success."""
    if not isinstance(payload, LTXHistoryGuidePayload):
        raise ValueError("payload must be an LTXHistoryGuidePayload")
    payload.validate()
    target = _target_samples(latent)
    reference_guide = payload.materialized()[0][1].latent
    guide_shape = tuple(reference_guide.shape)
    if (
        target.ndim != 5
        or target.shape[2] <= 0
        or (
            target.shape[0],
            target.shape[1],
            target.shape[3],
            target.shape[4],
        )
        != (guide_shape[0], guide_shape[1], guide_shape[3], guide_shape[4])
    ):
        raise ValueError(
            "target latent samples must match the guide batch, channels, height, and width"
        )
    if target.layout != torch.strided or not torch.is_floating_point(target):
        raise ValueError("target latent samples must be a strided floating tensor")
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("target latent samples must be finite")
    if target.dtype != reference_guide.dtype or target.device != reference_guide.device:
        raise ValueError("target latent samples must match the guide dtype and device")
    protected = tuple(
        (guide, tensor_sha256(guide.latent), guide.latent.clone())
        for role, guide in payload.materialized()
        if role == "exception"
    )
    current_positive = positive
    current_negative = negative
    current_latent = latent
    for role, guide in payload.materialized():
        applied_guide = guide
        if role == "exception":
            source = next(item for item in protected if item[0] is guide)
            applied_guide = LTXLatentGuide(source[2], guide.strength, guide.ordinal)
        current_positive, current_negative, current_latent = backend.append_guide(
            current_positive,
            current_negative,
            current_latent,
            applied_guide,
            role=role,
        )
        for source_guide, expected_sha256, defensive_clone in protected:
            if (
                tensor_sha256(source_guide.latent) != expected_sha256
                or tensor_sha256(defensive_clone) != expected_sha256
            ):
                raise ValueError("protected exception latent bytes changed during backend append")
    return LTXHistoryGuideReceipt(
        current_positive,
        current_negative,
        current_latent,
        len(payload.materialized()),
        payload.provenance,
    )


__all__ = (
    "GuidePayloadMode",
    "GuideRole",
    "LTXGuideProvenance",
    "LTXHistoryGuidePayload",
    "LTXHistoryGuideReceipt",
    "LTXLatentGuide",
    "LatentGuideBackend",
    "apply_history_guides",
)
