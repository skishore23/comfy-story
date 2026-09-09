"""Audited, prefix-only history inputs for Duet-X Decision 1."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal, Self

import torch

from duet.duetx.contracts import (
    SALIENCE_Q_MAX,
    SALIENCE_Q_MIN,
    EvidenceProvenance,
    tensor_sha256,
)
from duet.duetx.guide_boundary import LTXGuideProvenance, LTXLatentGuide
from duet.duetx.methods import MethodBatch

Split = Literal["train", "validation", "test"]
_SPLITS = frozenset({"train", "validation", "test"})
_HISTORY_ITEMS = 8
_LATENT_CHANNELS = 128
_PROCEDURAL_SCENES_PER_SPLIT = 8
_RARE_MARKER_SLOTS = (2, 5)
_SHA256_HEX = frozenset("0123456789abcdef")


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class HistoryItem:
    """One content-addressed historical latent ending before its target."""

    item_id: str
    content_sha256: str
    source_sha256: str
    media_sha256: str
    end_time_ns: int
    latent: torch.Tensor
    evidence: EvidenceProvenance

    def validate(self, *, slot: int, target_cutoff_ns: int) -> Self:
        _nonempty(self.item_id, "item_id")
        _sha256(self.content_sha256, "content_sha256")
        _sha256(self.source_sha256, "source_sha256")
        _sha256(self.media_sha256, "media_sha256")
        if type(self.end_time_ns) is not int or self.end_time_ns < 0:
            raise ValueError("end_time_ns must be a nonnegative integer")
        if self.end_time_ns >= target_cutoff_ns:
            raise ValueError("history items must end before the target cutoff")
        if (
            not isinstance(self.latent, torch.Tensor)
            or self.latent.layout != torch.strided
            or self.latent.ndim != 5
            or self.latent.shape[0] != 1
            or self.latent.shape[1] != _LATENT_CHANNELS
            or any(dimension <= 0 for dimension in self.latent.shape)
        ):
            raise ValueError("history latent must be a strided [1,128,F,H,W] tensor")
        if not torch.is_floating_point(self.latent):
            raise ValueError("history latent must use a floating dtype")
        if not bool(torch.isfinite(self.latent).all().item()):
            raise ValueError("history latent must be finite")
        if tensor_sha256(self.latent) != self.content_sha256:
            raise ValueError("content_sha256 must match the canonical history latent digest")
        if not isinstance(self.evidence, EvidenceProvenance):
            raise ValueError("history evidence must be EvidenceProvenance")
        self.evidence.validate()
        if self.evidence.leaf.slot != slot:
            raise ValueError("history evidence slots must remain canonical from 0 through 7")
        if self.evidence.time.timestamp_stop_ns > self.end_time_ns:
            raise ValueError("history evidence cannot read beyond its declared end_time_ns")
        return self


@dataclass(frozen=True, slots=True)
class HistorySample:
    """One manifest row with exactly eight prefix-only historical items."""

    sample_id: str
    split: Split
    performer_id: str
    room_id: str
    prop_ids: tuple[str, ...]
    glyph_family: str
    license_id: str
    consent_id: str
    target_cutoff_ns: int
    history: tuple[HistoryItem, ...]
    parent_manifest_sha256: str
    influence_labels: tuple[int, ...] = ()

    def validate(self) -> Self:
        for name in (
            "sample_id",
            "performer_id",
            "room_id",
            "glyph_family",
            "license_id",
            "consent_id",
        ):
            _nonempty(getattr(self, name), name)
        if self.split not in _SPLITS:
            raise ValueError("split must be train, validation, or test")
        if (
            not isinstance(self.prop_ids, tuple)
            or not self.prop_ids
            or any(not isinstance(prop_id, str) or not prop_id for prop_id in self.prop_ids)
            or len(set(self.prop_ids)) != len(self.prop_ids)
        ):
            raise ValueError("prop_ids must be a nonempty unique string tuple")
        if type(self.target_cutoff_ns) is not int or self.target_cutoff_ns <= 0:
            raise ValueError("target_cutoff_ns must be a positive integer")
        if not isinstance(self.history, tuple) or len(self.history) != _HISTORY_ITEMS:
            raise ValueError("history must contain exactly eight items")
        for slot, item in enumerate(self.history):
            if not isinstance(item, HistoryItem):
                raise ValueError("history must contain HistoryItem values")
            item.validate(slot=slot, target_cutoff_ns=self.target_cutoff_ns)
        item_ids = tuple(item.item_id for item in self.history)
        if len(set(item_ids)) != _HISTORY_ITEMS:
            raise ValueError("history item_ids must be unique")
        end_times = tuple(item.end_time_ns for item in self.history)
        if tuple(sorted(end_times)) != end_times or len(set(end_times)) != _HISTORY_ITEMS:
            raise ValueError("history items must be in strictly increasing end_time_ns order")
        shapes = {tuple(item.latent.shape) for item in self.history}
        dtypes = {item.latent.dtype for item in self.history}
        devices = {item.latent.device for item in self.history}
        if len(shapes) != 1 or len(dtypes) != 1 or len(devices) != 1:
            raise ValueError("all eight histories require equal latent shapes, dtypes, and devices")
        _sha256(self.parent_manifest_sha256, "parent_manifest_sha256")
        if not isinstance(self.influence_labels, tuple) or any(
            type(label) is not int or not SALIENCE_Q_MIN <= label <= SALIENCE_Q_MAX
            for label in self.influence_labels
        ):
            raise ValueError("influence_labels must contain signed int64 integers")
        if self.influence_labels and len(self.influence_labels) != _HISTORY_ITEMS:
            raise ValueError("influence_labels must be empty or align with all eight histories")
        if self.split == "test" and self.influence_labels:
            raise ValueError("test influence labels are sealed")
        return self


@dataclass(frozen=True, slots=True)
class HistoryBatch:
    """The exact batch-size-one tensor and metadata consumed by teacher and methods."""

    history: torch.Tensor
    sample_id: str
    split: Split
    item_ids: tuple[str, ...]
    content_sha256: tuple[str, ...]
    source_sha256: tuple[str, ...]
    media_sha256: tuple[str, ...]
    evidence: tuple[EvidenceProvenance, ...]
    parent_manifest_sha256: str

    def validate(self) -> Self:
        if (
            not isinstance(self.history, torch.Tensor)
            or self.history.layout != torch.strided
            or self.history.ndim != 6
            or tuple(self.history.shape[:3]) != (1, _HISTORY_ITEMS, _LATENT_CHANNELS)
            or any(dimension <= 0 for dimension in self.history.shape)
            or not torch.is_floating_point(self.history)
            or not bool(torch.isfinite(self.history).all().item())
        ):
            raise ValueError("batch history must be a finite floating [1,8,128,F,H,W] tensor")
        _nonempty(self.sample_id, "sample_id")
        if self.split not in _SPLITS:
            raise ValueError("split must be train, validation, or test")
        fields = (
            (self.item_ids, "item_ids"),
            (self.content_sha256, "content_sha256"),
            (self.source_sha256, "source_sha256"),
            (self.media_sha256, "media_sha256"),
            (self.evidence, "evidence"),
        )
        if any(
            not isinstance(values, tuple) or len(values) != _HISTORY_ITEMS for values, _ in fields
        ):
            raise ValueError("batch metadata must align with all eight canonical history slots")
        if any(not isinstance(item_id, str) or not item_id for item_id in self.item_ids):
            raise ValueError("batch item_ids must be nonempty strings")
        if len(set(self.item_ids)) != _HISTORY_ITEMS:
            raise ValueError("batch item_ids must be unique")
        for values, name in (
            (self.content_sha256, "content_sha256"),
            (self.source_sha256, "source_sha256"),
            (self.media_sha256, "media_sha256"),
        ):
            for value in values:
                _sha256(value, name)
        for slot, content_sha256 in enumerate(self.content_sha256):
            if tensor_sha256(self.history[:, slot]) != content_sha256:
                raise ValueError("batch content_sha256 must match each canonical history latent")
        for slot, provenance in enumerate(self.evidence):
            if not isinstance(provenance, EvidenceProvenance):
                raise ValueError("batch evidence must contain EvidenceProvenance values")
            provenance.validate()
            if provenance.leaf.slot != slot:
                raise ValueError("batch evidence slots must remain canonical from 0 through 7")
        _sha256(self.parent_manifest_sha256, "parent_manifest_sha256")
        return self

    @classmethod
    def from_sample(cls, sample: HistorySample) -> Self:
        """Build a batch without changing canonical history slot order."""
        if not isinstance(sample, HistorySample):
            raise ValueError("sample must be a HistorySample")
        sample.validate()
        history = torch.stack(tuple(item.latent for item in sample.history), dim=1)
        return cls(
            history=history,
            sample_id=sample.sample_id,
            split=sample.split,
            item_ids=tuple(item.item_id for item in sample.history),
            content_sha256=tuple(item.content_sha256 for item in sample.history),
            source_sha256=tuple(item.source_sha256 for item in sample.history),
            media_sha256=tuple(item.media_sha256 for item in sample.history),
            evidence=tuple(item.evidence for item in sample.history),
            parent_manifest_sha256=sample.parent_manifest_sha256,
        ).validate()

    def to_method_batch(
        self,
        selected_indices: tuple[int, int],
        mandatory: LTXLatentGuide,
        mandatory_provenance: LTXGuideProvenance,
    ) -> MethodBatch:
        """Adapt to the strict public method boundary while preserving slots 0 through 7."""
        self.validate()
        return MethodBatch(
            history=self.history,
            item_ids=self.item_ids,
            evidence=self.evidence,
            selected_indices=selected_indices,
            mandatory=mandatory,
            mandatory_provenance=mandatory_provenance,
        )


def _declared_rows(rows: Collection[str], name: str) -> frozenset[str]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Collection):
        raise ValueError(f"{name} must be a collection of declared row IDs")
    values = frozenset(rows)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain nonempty string IDs")
    return values


def _require_split_isolation(samples: Sequence[HistorySample], field: str) -> None:
    split_by_value: defaultdict[str, set[str]] = defaultdict(set)
    for sample in samples:
        value = getattr(sample, field)
        values = value if isinstance(value, tuple) else (value,)
        for item in values:
            split_by_value[item].add(sample.split)
    if any(len(splits) != 1 for splits in split_by_value.values()):
        raise ValueError(f"{field} must be isolated across splits")


def audit_manifest(
    samples: Sequence[HistorySample],
    *,
    license_rows: Collection[str],
    consent_rows: Collection[str],
) -> tuple[HistorySample, ...]:
    """Validate a complete manifest atomically and return its immutable sample tuple."""
    if not samples:
        raise ValueError("manifest must be a nonempty sequence of HistorySample values")
    audited = tuple(samples)
    for sample in audited:
        if not isinstance(sample, HistorySample):
            raise ValueError("manifest must contain HistorySample values")
        sample.validate()
    sample_ids = tuple(sample.sample_id for sample in audited)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id must be unique across the complete manifest")
    parents = {sample.parent_manifest_sha256 for sample in audited}
    if len(parents) != 1:
        raise ValueError("all samples must share one parent_manifest_sha256")

    licenses = _declared_rows(license_rows, "license_rows")
    consents = _declared_rows(consent_rows, "consent_rows")
    if any(sample.license_id not in licenses for sample in audited):
        raise ValueError("every sample requires a declared license row")
    if any(sample.consent_id not in consents for sample in audited):
        raise ValueError("every sample requires a declared consent row")

    for field in ("performer_id", "room_id", "prop_ids", "glyph_family"):
        _require_split_isolation(audited, field)

    source_hashes = tuple(item.source_sha256 for sample in audited for item in sample.history)
    media_hashes = tuple(item.media_sha256 for sample in audited for item in sample.history)
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("source_sha256 must be unique across the complete manifest")
    if len(set(media_hashes)) != len(media_hashes):
        raise ValueError("media_sha256 must be unique and isolated across splits")
    latent_shapes = {tuple(item.latent.shape) for sample in audited for item in sample.history}
    if len(latent_shapes) != 1:
        raise ValueError("the complete manifest requires equal latent shapes")
    return audited


def audit_decision1_procedural_canary(
    samples: Sequence[HistorySample],
    *,
    license_rows: Collection[str],
    consent_rows: Collection[str],
) -> tuple[HistorySample, ...]:
    """Audit the finite fallback canary used when no approved dataset exists.

    The fallback is deliberately small but still has eight independently clustered scenes in
    each split.  Every scene retains the fixed eight-item history, and slots 2 and 5 are the only
    explicitly declared rare-marker events.  This keeps the sealed test large enough for the
    preregistered scene-cluster bootstrap without silently turning one scene's three seeds into
    three independent samples.
    """
    audited = audit_manifest(
        samples,
        license_rows=license_rows,
        consent_rows=consent_rows,
    )
    counts = {split: sum(sample.split == split for sample in audited) for split in _SPLITS}
    if any(count != _PROCEDURAL_SCENES_PER_SPLIT for count in counts.values()):
        raise ValueError("procedural canary requires exactly eight scenes per split")
    scene_keys = {(sample.performer_id, sample.room_id, sample.glyph_family) for sample in audited}
    if len(scene_keys) != len(audited):
        raise ValueError("procedural canary scene clusters must be unique")
    for sample in audited:
        rare_slots = tuple(
            slot
            for slot, item in enumerate(sample.history)
            if item.evidence.event_type.startswith("rare-marker:")
        )
        if rare_slots != _RARE_MARKER_SLOTS:
            raise ValueError("procedural canary rare markers must be exactly slots 2 and 5")
    return audited


__all__ = (
    "HistoryBatch",
    "HistoryItem",
    "HistorySample",
    "Split",
    "audit_decision1_procedural_canary",
    "audit_manifest",
)
