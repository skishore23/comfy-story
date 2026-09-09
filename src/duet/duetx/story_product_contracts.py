"""Typed, canonical product state for selective Duet Story recall."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Self, cast

from duet.duetx.story_contracts import (
    canonical_story_json,
    decode_canonical_story_object,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VIDEO_LOCATOR = re.compile(r"^duet-evidence://story/sha256/([0-9a-f]{64})$")
_Q16_MAX = 65_536
_SALIENCE_MAX = 1_000_000


class ObservationKind(StrEnum):
    OPENING = "opening"
    CHANGE = "change"
    CLOSING = "closing"
    LEGACY = "legacy"


class CanonPresence(StrEnum):
    PRESENT = "present"
    OFF_SCREEN = "off_screen"
    UNKNOWN = "unknown"


class MemoryAction(StrEnum):
    KEEP = "keep"
    USE_IN_THIS_SHOT = "use_in_this_shot"
    UPDATE_CANON = "update_canon"
    RESTORE_ORIGINAL = "restore_original"
    LET_FADE = "let_fade"
    FORGET = "forget"


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a portable identifier")
    return value


def _text(value: object, field: str, maximum: int = 1000) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{field} must be text with at most {maximum} characters")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _mapping(value: object, field: str, names: frozenset[str]) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    result = cast(dict[str, object], value)
    if set(result) != names:
        raise ValueError(f"{field} has missing or unknown fields")
    return result


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[object], value)


def _strings(value: object, field: str) -> tuple[str, ...]:
    values = _array(value, field)
    if any(not isinstance(item, str) for item in values):
        raise ValueError(f"{field} must be an array of strings")
    return tuple(cast(list[str], values))


def _ordered_unique(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    for value in values:
        _identifier(value, field)
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must not contain duplicates")
    return values


@dataclass(frozen=True, slots=True)
class StoryObservation:
    evidence_id: str
    kind: ObservationKind
    frame_index: int
    timestamp_start_ns: int
    timestamp_stop_ns: int
    geometry_q16: tuple[int, int, int, int]
    asset_sha256: str
    preprocessing_sha256: str
    vae_sha256: str
    latent_sha256: str
    salience_q: int
    change_score_q: int
    entity_ids: tuple[str, ...]

    def validate(self) -> Self:
        _identifier(self.evidence_id, "evidence_id")
        if not isinstance(self.kind, ObservationKind) or self.kind is ObservationKind.LEGACY:
            raise ValueError("new observations require opening, change, or closing kind")
        _integer(self.frame_index, "frame_index")
        _integer(self.timestamp_start_ns, "timestamp_start_ns")
        _integer(self.timestamp_stop_ns, "timestamp_stop_ns", minimum=1)
        if self.timestamp_stop_ns <= self.timestamp_start_ns:
            raise ValueError("observation timeline range must be nonempty")
        if (
            not isinstance(self.geometry_q16, tuple)
            or len(self.geometry_q16) != 4
            or any(type(value) is not int for value in self.geometry_q16)
        ):
            raise ValueError("geometry_q16 must contain four integers")
        left, top, right, bottom = self.geometry_q16
        if not (0 <= left < right <= _Q16_MAX and 0 <= top < bottom <= _Q16_MAX):
            raise ValueError("geometry_q16 must be a nonempty normalized q16 rectangle")
        for value, field in (
            (self.asset_sha256, "asset_sha256"),
            (self.preprocessing_sha256, "preprocessing_sha256"),
            (self.vae_sha256, "vae_sha256"),
            (self.latent_sha256, "latent_sha256"),
        ):
            _digest(value, field)
        if type(self.salience_q) is not int or not 0 <= self.salience_q <= _SALIENCE_MAX:
            raise ValueError("salience_q must be an integer in [0, 1000000]")
        _integer(self.change_score_q, "change_score_q")
        _ordered_unique(self.entity_ids, "entity_ids")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> StoryObservation:
        names = frozenset(cls.__dataclass_fields__)
        data = _mapping(value, "StoryObservation", names)
        geometry = _array(data["geometry_q16"], "geometry_q16")
        if len(geometry) != 4 or any(type(item) is not int for item in geometry):
            raise ValueError("geometry_q16 must contain four integers")
        try:
            kind = ObservationKind(_text(data["kind"], "kind"))
        except ValueError as error:
            raise ValueError("observation kind is unsupported") from error
        return cls(
            _text(data["evidence_id"], "evidence_id"),
            kind,
            _integer(data["frame_index"], "frame_index"),
            _integer(data["timestamp_start_ns"], "timestamp_start_ns"),
            _integer(data["timestamp_stop_ns"], "timestamp_stop_ns"),
            cast(tuple[int, int, int, int], tuple(geometry)),
            _text(data["asset_sha256"], "asset_sha256"),
            _text(data["preprocessing_sha256"], "preprocessing_sha256"),
            _text(data["vae_sha256"], "vae_sha256"),
            _text(data["latent_sha256"], "latent_sha256"),
            _integer(data["salience_q"], "salience_q"),
            _integer(data["change_score_q"], "change_score_q"),
            _strings(data["entity_ids"], "entity_ids"),
        ).validate()


@dataclass(frozen=True, slots=True)
class NativeRGBObservation:
    """Exact RGB evidence without claiming a VAE encoding or learned latent.

    The explicit wire format preserves the historical encoded-observation schema.
    Retrieval rank is declared metadata; it is not a trained salience prediction.
    """

    evidence_id: str
    kind: ObservationKind
    frame_index: int
    timestamp_start_ns: int
    timestamp_stop_ns: int
    geometry_q16: tuple[int, int, int, int]
    asset_sha256: str
    preprocessing_sha256: str
    salience_q: int
    change_score_q: int
    entity_ids: tuple[str, ...]
    format: str = "duet-story-rgb-observation-v1"

    def validate(self) -> Self:
        if self.format != "duet-story-rgb-observation-v1":
            raise ValueError("native RGB observation format is unsupported")
        _identifier(self.evidence_id, "evidence_id")
        if not isinstance(self.kind, ObservationKind) or self.kind is ObservationKind.LEGACY:
            raise ValueError("new observations require opening, change, or closing kind")
        _integer(self.frame_index, "frame_index")
        _integer(self.timestamp_start_ns, "timestamp_start_ns")
        _integer(self.timestamp_stop_ns, "timestamp_stop_ns", minimum=1)
        if self.timestamp_stop_ns <= self.timestamp_start_ns:
            raise ValueError("observation timeline range must be nonempty")
        if (
            not isinstance(self.geometry_q16, tuple)
            or len(self.geometry_q16) != 4
            or any(type(value) is not int for value in self.geometry_q16)
        ):
            raise ValueError("geometry_q16 must contain four integers")
        left, top, right, bottom = self.geometry_q16
        if not (0 <= left < right <= _Q16_MAX and 0 <= top < bottom <= _Q16_MAX):
            raise ValueError("geometry_q16 must be a nonempty normalized q16 rectangle")
        for value, field in (
            (self.asset_sha256, "asset_sha256"),
            (self.preprocessing_sha256, "preprocessing_sha256"),
        ):
            _digest(value, field)
        if type(self.salience_q) is not int or not 0 <= self.salience_q <= _SALIENCE_MAX:
            raise ValueError("salience_q must be an integer in [0, 1000000]")
        _integer(self.change_score_q, "change_score_q")
        _ordered_unique(self.entity_ids, "entity_ids")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> NativeRGBObservation:
        names = frozenset(cls.__dataclass_fields__)
        data = _mapping(value, "NativeRGBObservation", names)
        geometry = _array(data["geometry_q16"], "geometry_q16")
        if len(geometry) != 4 or any(type(item) is not int for item in geometry):
            raise ValueError("geometry_q16 must contain four integers")
        try:
            kind = ObservationKind(_text(data["kind"], "kind"))
        except ValueError as error:
            raise ValueError("observation kind is unsupported") from error
        return cls(
            _text(data["evidence_id"], "evidence_id"),
            kind,
            _integer(data["frame_index"], "frame_index"),
            _integer(data["timestamp_start_ns"], "timestamp_start_ns"),
            _integer(data["timestamp_stop_ns"], "timestamp_stop_ns"),
            cast(tuple[int, int, int, int], tuple(geometry)),
            _text(data["asset_sha256"], "asset_sha256"),
            _text(data["preprocessing_sha256"], "preprocessing_sha256"),
            _integer(data["salience_q"], "salience_q"),
            _integer(data["change_score_q"], "change_score_q"),
            _strings(data["entity_ids"], "entity_ids"),
            _text(data["format"], "format"),
        ).validate()


def _observation_from_mapping(value: object) -> StoryObservation | NativeRGBObservation:
    if isinstance(value, dict) and "format" in value:
        return NativeRGBObservation.from_mapping(value)
    return StoryObservation.from_mapping(value)


@dataclass(frozen=True, slots=True)
class StoryObservationPacket:
    packet_id: str
    shot_index: int
    frame_count: int
    timeline_start_ns: int
    timeline_stop_ns: int
    source_video_sha256: str
    source_video_locator: str
    extraction_policy_sha256: str
    referenced_entity_ids: tuple[str, ...]
    observations: tuple[StoryObservation | NativeRGBObservation, ...]

    def validate(self) -> Self:
        _identifier(self.packet_id, "packet_id")
        _integer(self.shot_index, "shot_index")
        _integer(self.frame_count, "frame_count", minimum=1)
        _integer(self.timeline_start_ns, "timeline_start_ns")
        if (
            type(self.timeline_stop_ns) is not int
            or self.timeline_stop_ns <= self.timeline_start_ns
        ):
            raise ValueError("packet timeline range must be nonempty")
        digest = _digest(self.source_video_sha256, "source_video_sha256")
        match = (
            _VIDEO_LOCATOR.fullmatch(self.source_video_locator)
            if isinstance(self.source_video_locator, str)
            else None
        )
        if match is None or match.group(1) != digest:
            raise ValueError("source_video_locator must bind source_video_sha256")
        _digest(self.extraction_policy_sha256, "extraction_policy_sha256")
        _ordered_unique(self.referenced_entity_ids, "referenced_entity_ids")
        if not isinstance(self.observations, tuple) or not self.observations:
            raise ValueError("packet must contain observations")
        for observation in self.observations:
            if not isinstance(observation, (StoryObservation, NativeRGBObservation)):
                raise ValueError("packet observations must be StoryObservation values")
            observation.validate()
            if observation.frame_index >= self.frame_count:
                raise ValueError("observation frame index exceeds decoded frame count")
            if not (
                self.timeline_start_ns
                <= observation.timestamp_start_ns
                < observation.timestamp_stop_ns
                <= self.timeline_stop_ns
            ):
                raise ValueError("observation timeline lies outside its packet")
        ids = tuple(item.evidence_id for item in self.observations)
        if len(ids) != len(set(ids)):
            raise ValueError("packet observation IDs must be unique")
        order = tuple(
            (item.frame_index, item.kind.value, item.evidence_id) for item in self.observations
        )
        if order != tuple(sorted(order)):
            raise ValueError("packet observations must use canonical frame order")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> StoryObservationPacket:
        data = _mapping(value, "StoryObservationPacket", frozenset(cls.__dataclass_fields__))
        return cls(
            _text(data["packet_id"], "packet_id"),
            _integer(data["shot_index"], "shot_index"),
            _integer(data["frame_count"], "frame_count", minimum=1),
            _integer(data["timeline_start_ns"], "timeline_start_ns"),
            _integer(data["timeline_stop_ns"], "timeline_stop_ns", minimum=1),
            _text(data["source_video_sha256"], "source_video_sha256"),
            _text(data["source_video_locator"], "source_video_locator"),
            _text(data["extraction_policy_sha256"], "extraction_policy_sha256"),
            _strings(data["referenced_entity_ids"], "referenced_entity_ids"),
            tuple(
                _observation_from_mapping(item)
                for item in _array(data["observations"], "observations")
            ),
        ).validate()


@dataclass(frozen=True, slots=True)
class StoryEvidenceRecord:
    evidence_id: str
    packet_id: str | None
    kind: ObservationKind
    entity_ids: tuple[str, ...]
    asset_sha256: str
    source_shot_index: int | None
    frame_index: int | None
    timestamp_start_ns: int | None
    timestamp_stop_ns: int | None
    geometry_q16: tuple[int, int, int, int] | None
    salience_q: int
    fingerprint_sha256: str

    @classmethod
    def from_observation(
        cls, packet: StoryObservationPacket, observation: StoryObservation | NativeRGBObservation
    ) -> StoryEvidenceRecord:
        packet.validate()
        observation.validate()
        if observation not in packet.observations:
            raise ValueError("observation must belong to the packet")
        fingerprint = canonical_story_json(
            {
                "asset_sha256": observation.asset_sha256,
                "evidence_id": observation.evidence_id,
                "packet_id": packet.packet_id,
                "salience_q": observation.salience_q,
            }
        )
        import hashlib

        return cls(
            observation.evidence_id,
            packet.packet_id,
            observation.kind,
            observation.entity_ids,
            observation.asset_sha256,
            packet.shot_index,
            observation.frame_index,
            observation.timestamp_start_ns,
            observation.timestamp_stop_ns,
            observation.geometry_q16,
            observation.salience_q,
            hashlib.sha256(fingerprint).hexdigest(),
        ).validate()

    def validate(self) -> Self:
        _identifier(self.evidence_id, "evidence_id")
        if not isinstance(self.kind, ObservationKind):
            raise ValueError("evidence kind is unsupported")
        _ordered_unique(self.entity_ids, "entity_ids")
        _digest(self.asset_sha256, "asset_sha256")
        _digest(self.fingerprint_sha256, "fingerprint_sha256")
        if type(self.salience_q) is not int or not 0 <= self.salience_q <= _SALIENCE_MAX:
            raise ValueError("salience_q must be an integer in [0, 1000000]")
        detail = (
            self.packet_id,
            self.source_shot_index,
            self.frame_index,
            self.timestamp_start_ns,
            self.timestamp_stop_ns,
            self.geometry_q16,
        )
        if self.kind is ObservationKind.LEGACY:
            if any(value is not None for value in detail) or self.entity_ids:
                raise ValueError("legacy evidence must not claim unavailable provenance")
            return self
        if any(value is None for value in detail):
            raise ValueError("nonlegacy evidence requires exact packet and frame provenance")
        _identifier(self.packet_id, "packet_id")
        _integer(self.source_shot_index, "source_shot_index")
        _integer(self.frame_index, "frame_index")
        _integer(self.timestamp_start_ns, "timestamp_start_ns")
        if cast(int, self.timestamp_stop_ns) <= cast(int, self.timestamp_start_ns):
            raise ValueError("evidence timeline range must be nonempty")
        geometry = cast(tuple[int, int, int, int], self.geometry_q16)
        if len(geometry) != 4:
            raise ValueError("evidence geometry must contain four q16 values")
        left, top, right, bottom = geometry
        if not (0 <= left < right <= _Q16_MAX and 0 <= top < bottom <= _Q16_MAX):
            raise ValueError("evidence geometry must be a normalized q16 rectangle")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> StoryEvidenceRecord:
        data = _mapping(value, "StoryEvidenceRecord", frozenset(cls.__dataclass_fields__))
        try:
            kind = ObservationKind(_text(data["kind"], "kind"))
        except ValueError as error:
            raise ValueError("evidence kind is unsupported") from error
        geometry_value = data["geometry_q16"]
        geometry: tuple[int, int, int, int] | None
        if geometry_value is None:
            geometry = None
        else:
            items = _array(geometry_value, "geometry_q16")
            geometry = cast(tuple[int, int, int, int], tuple(items))
        optional_ints: list[int | None] = []
        for field in (
            "source_shot_index",
            "frame_index",
            "timestamp_start_ns",
            "timestamp_stop_ns",
        ):
            item = data[field]
            optional_ints.append(None if item is None else _integer(item, field))
        packet = data["packet_id"]
        if packet is not None and not isinstance(packet, str):
            raise ValueError("packet_id must be a string or null")
        return cls(
            _text(data["evidence_id"], "evidence_id"),
            packet,
            kind,
            _strings(data["entity_ids"], "entity_ids"),
            _text(data["asset_sha256"], "asset_sha256"),
            optional_ints[0],
            optional_ints[1],
            optional_ints[2],
            optional_ints[3],
            geometry,
            _integer(data["salience_q"], "salience_q"),
            _text(data["fingerprint_sha256"], "fingerprint_sha256"),
        ).validate()


@dataclass(frozen=True, slots=True)
class PendingCanonObservation:
    evidence_id: str
    state_note: str
    source_revision_sha256: str

    def validate(self) -> Self:
        _identifier(self.evidence_id, "evidence_id")
        if not _text(self.state_note, "state_note", 500).strip():
            raise ValueError("pending state_note must not be empty")
        _digest(self.source_revision_sha256, "source_revision_sha256")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> PendingCanonObservation:
        data = _mapping(value, "PendingCanonObservation", frozenset(cls.__dataclass_fields__))
        return cls(
            _text(data["evidence_id"], "evidence_id"),
            _text(data["state_note"], "state_note", 500),
            _text(data["source_revision_sha256"], "source_revision_sha256"),
        ).validate()


@dataclass(frozen=True, slots=True)
class CanonEntity:
    entity_id: str
    reference_name: str
    baseline_sha256: str
    state_note: str
    presence: CanonPresence
    supporting_evidence_ids: tuple[str, ...]
    confirmed_revision_sha256: str | None
    pending: tuple[PendingCanonObservation, ...]

    def validate(self) -> Self:
        _identifier(self.entity_id, "entity_id")
        if not _text(self.reference_name, "reference_name", 96).strip():
            raise ValueError("reference_name must not be empty")
        _digest(self.baseline_sha256, "baseline_sha256")
        _text(self.state_note, "state_note", 500)
        if not isinstance(self.presence, CanonPresence):
            raise ValueError("canon presence is unsupported")
        _ordered_unique(self.supporting_evidence_ids, "supporting_evidence_ids")
        if self.confirmed_revision_sha256 is not None:
            _digest(self.confirmed_revision_sha256, "confirmed_revision_sha256")
        if not isinstance(self.pending, tuple):
            raise ValueError("pending must be a tuple")
        for item in self.pending:
            item.validate()
        pending_ids = tuple(item.evidence_id for item in self.pending)
        if len(pending_ids) != len(set(pending_ids)):
            raise ValueError("pending evidence IDs must be unique")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> CanonEntity:
        data = _mapping(value, "CanonEntity", frozenset(cls.__dataclass_fields__))
        try:
            presence = CanonPresence(_text(data["presence"], "presence"))
        except ValueError as error:
            raise ValueError("canon presence is unsupported") from error
        revision = data["confirmed_revision_sha256"]
        if revision is not None and not isinstance(revision, str):
            raise ValueError("confirmed_revision_sha256 must be a digest or null")
        return cls(
            _text(data["entity_id"], "entity_id"),
            _text(data["reference_name"], "reference_name", 96),
            _text(data["baseline_sha256"], "baseline_sha256"),
            _text(data["state_note"], "state_note", 500),
            presence,
            _strings(data["supporting_evidence_ids"], "supporting_evidence_ids"),
            revision,
            tuple(
                PendingCanonObservation.from_mapping(item)
                for item in _array(data["pending"], "pending")
            ),
        ).validate()


@dataclass(frozen=True, slots=True)
class StoryCanon:
    entities: tuple[CanonEntity, ...]

    def validate(self) -> Self:
        if not isinstance(self.entities, tuple):
            raise ValueError("canon entities must be a tuple")
        for entity in self.entities:
            entity.validate()
        keys = tuple(entity.entity_id for entity in self.entities)
        if len(keys) != len(set(keys)):
            raise ValueError("canon entity IDs must be unique")
        if keys != tuple(sorted(keys)):
            raise ValueError("canon entities must use canonical entity order")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> StoryCanon:
        data = _mapping(value, "StoryCanon", frozenset({"entities"}))
        return cls(
            tuple(CanonEntity.from_mapping(item) for item in _array(data["entities"], "entities"))
        ).validate()


@dataclass(frozen=True, slots=True)
class StoryMemoryPolicy:
    pinned_evidence_ids: tuple[str, ...] = ()
    tombstoned_evidence_ids: tuple[str, ...] = ()

    def validate(self) -> Self:
        pinned = _ordered_unique(self.pinned_evidence_ids, "pinned_evidence_ids")
        tombstoned = _ordered_unique(self.tombstoned_evidence_ids, "tombstoned_evidence_ids")
        if pinned != tuple(sorted(pinned)) or tombstoned != tuple(sorted(tombstoned)):
            raise ValueError("memory policy IDs must use canonical order")
        if set(pinned) & set(tombstoned):
            raise ValueError("evidence cannot be both pinned and forgotten")
        if len(pinned) > 32:
            raise ValueError("memory policy supports at most 32 pinned evidence items")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> StoryMemoryPolicy:
        data = _mapping(value, "StoryMemoryPolicy", frozenset(cls.__dataclass_fields__))
        return cls(
            _strings(data["pinned_evidence_ids"], "pinned_evidence_ids"),
            _strings(data["tombstoned_evidence_ids"], "tombstoned_evidence_ids"),
        ).validate()


@dataclass(frozen=True, slots=True)
class StoryMemoryCommand:
    action: MemoryAction
    parent_revision_sha256: str
    target_id: str
    state_note: str = ""
    presence: CanonPresence | None = None
    supporting_evidence_ids: tuple[str, ...] = ()

    def validate(self) -> Self:
        if not isinstance(self.action, MemoryAction):
            raise ValueError("memory action is unsupported")
        _digest(self.parent_revision_sha256, "parent_revision_sha256")
        _identifier(self.target_id, "target_id")
        _text(self.state_note, "state_note", 500)
        if self.presence is not None and not isinstance(self.presence, CanonPresence):
            raise ValueError("command presence is unsupported")
        _ordered_unique(self.supporting_evidence_ids, "supporting_evidence_ids")
        if self.action is MemoryAction.UPDATE_CANON and self.presence is None:
            raise ValueError("Update canon requires an explicit presence")
        if self.action is not MemoryAction.UPDATE_CANON and (
            self.state_note or self.presence is not None or self.supporting_evidence_ids
        ):
            raise ValueError("only Update canon accepts canon fields")
        return self

    def to_json(self) -> bytes:
        self.validate()
        return canonical_story_json(self)

    @classmethod
    def from_mapping(cls, value: object) -> StoryMemoryCommand:
        data = _mapping(value, "StoryMemoryCommand", frozenset(cls.__dataclass_fields__))
        try:
            action = MemoryAction(_text(data["action"], "action"))
        except ValueError as error:
            raise ValueError("memory action is unsupported") from error
        presence_value = data["presence"]
        try:
            presence = (
                None if presence_value is None else CanonPresence(_text(presence_value, "presence"))
            )
        except ValueError as error:
            raise ValueError("command presence is unsupported") from error
        return cls(
            action,
            _text(data["parent_revision_sha256"], "parent_revision_sha256"),
            _text(data["target_id"], "target_id"),
            _text(data["state_note"], "state_note", 500),
            presence,
            _strings(data["supporting_evidence_ids"], "supporting_evidence_ids"),
        ).validate()

    @classmethod
    def from_json(cls, encoded: bytes) -> StoryMemoryCommand:
        return cls.from_mapping(decode_canonical_story_object(encoded, field="StoryMemoryCommand"))


@dataclass(frozen=True, slots=True)
class StoryEvidencePacketSpec:
    entity_id: str
    baseline_sha256: str
    state_evidence_id: str | None
    source_sha256s: tuple[str, ...]

    def validate(self) -> Self:
        _identifier(self.entity_id, "entity_id")
        _digest(self.baseline_sha256, "baseline_sha256")
        if self.state_evidence_id is not None:
            _identifier(self.state_evidence_id, "state_evidence_id")
        if not isinstance(self.source_sha256s, tuple) or not self.source_sha256s:
            raise ValueError("evidence packet requires source assets")
        for digest in self.source_sha256s:
            _digest(digest, "source_sha256")
        if len(self.source_sha256s) != len(set(self.source_sha256s)):
            raise ValueError("evidence packet must not duplicate source assets")
        return self


@dataclass(frozen=True, slots=True)
class GuideBinding:
    role: str
    asset_sha256: str

    def validate(self) -> Self:
        _identifier(self.role, "guide role")
        _digest(self.asset_sha256, "guide asset_sha256")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> GuideBinding:
        data = _mapping(value, "GuideBinding", frozenset(cls.__dataclass_fields__))
        return cls(
            _text(data["role"], "role"),
            _text(data["asset_sha256"], "asset_sha256"),
        ).validate()


@dataclass(frozen=True, slots=True)
class StoryRecallDecision:
    explicit_entity_ids: tuple[str, ...]
    forced_evidence_ids: tuple[str, ...]
    resolved_canon_versions: tuple[str, ...]
    rejected_candidates: tuple[str, ...]
    selected_evidence_ids: tuple[str, ...]
    packet_specs: tuple[StoryEvidencePacketSpec, ...]
    inclusive_core_sha256: str | None
    guide_bindings: tuple[GuideBinding, ...]

    def validate(self) -> Self:
        for values, field in (
            (self.explicit_entity_ids, "explicit_entity_ids"),
            (self.forced_evidence_ids, "forced_evidence_ids"),
            (self.selected_evidence_ids, "selected_evidence_ids"),
        ):
            _ordered_unique(values, field)
        if len(self.selected_evidence_ids) > 2 or len(self.packet_specs) > 2:
            raise ValueError("MiniMax recall supports at most two evidence packets")
        if self.inclusive_core_sha256 is not None:
            _digest(self.inclusive_core_sha256, "inclusive_core_sha256")
        for value in self.resolved_canon_versions:
            _digest(value, "resolved canon revision")
        for value in self.rejected_candidates:
            _text(value, "rejected candidate", 256)
        for packet in self.packet_specs:
            packet.validate()
        for binding in self.guide_bindings:
            binding.validate()
        roles = tuple(binding.role for binding in self.guide_bindings)
        if len(roles) != len(set(roles)) or len(roles) > 9:
            raise ValueError("guide bindings require unique bounded semantic roles")
        return self

    def to_json(self) -> bytes:
        self.validate()
        return canonical_story_json(self)


@dataclass(frozen=True, slots=True)
class StoryGenerationReceipt:
    prompt_sha256: str
    parent_revision_sha256: str | None
    memory_backend: str
    checkpoint_sha256: str
    sampler: str
    selected_evidence_ids: tuple[str, ...]
    guide_bindings: tuple[GuideBinding, ...]
    recall_decision_sha256: str

    def validate(self) -> Self:
        _digest(self.prompt_sha256, "prompt_sha256")
        if self.parent_revision_sha256 is not None:
            _digest(self.parent_revision_sha256, "parent_revision_sha256")
        if self.memory_backend not in {"ltx", "minimax-h3"}:
            raise ValueError("memory_backend is unsupported")
        _digest(self.checkpoint_sha256, "checkpoint_sha256")
        _identifier(self.sampler, "sampler")
        _ordered_unique(self.selected_evidence_ids, "selected_evidence_ids")
        if len(self.selected_evidence_ids) > 2:
            raise ValueError("generation receipt supports at most two recalled evidence items")
        for binding in self.guide_bindings:
            binding.validate()
        _digest(self.recall_decision_sha256, "recall_decision_sha256")
        return self

    def to_json(self) -> bytes:
        self.validate()
        return canonical_story_json(self)

    @classmethod
    def from_json(cls, encoded: bytes) -> StoryGenerationReceipt:
        data = decode_canonical_story_object(encoded, field="StoryGenerationReceipt")
        _mapping(data, "StoryGenerationReceipt", frozenset(cls.__dataclass_fields__))
        parent = data["parent_revision_sha256"]
        if parent is not None and not isinstance(parent, str):
            raise ValueError("parent_revision_sha256 must be a digest or null")
        return cls(
            _text(data["prompt_sha256"], "prompt_sha256"),
            parent,
            _text(data["memory_backend"], "memory_backend"),
            _text(data["checkpoint_sha256"], "checkpoint_sha256"),
            _text(data["sampler"], "sampler"),
            _strings(data["selected_evidence_ids"], "selected_evidence_ids"),
            tuple(
                GuideBinding.from_mapping(item)
                for item in _array(data["guide_bindings"], "guide_bindings")
            ),
            _text(data["recall_decision_sha256"], "recall_decision_sha256"),
        ).validate()


@dataclass(frozen=True, slots=True)
class StoryProductState:
    observation_packets: tuple[StoryObservationPacket, ...]
    evidence_records: tuple[StoryEvidenceRecord, ...]
    canon: StoryCanon
    policy: StoryMemoryPolicy

    def validate(self) -> Self:
        if not isinstance(self.observation_packets, tuple) or not isinstance(
            self.evidence_records, tuple
        ):
            raise ValueError("product state collections must be tuples")
        for packet in self.observation_packets:
            packet.validate()
        observed_shots = tuple(packet.shot_index for packet in self.observation_packets)
        if observed_shots != tuple(sorted(set(observed_shots))):
            raise ValueError("observation packets must use unique chronological shot order")
        for previous, current in zip(
            self.observation_packets, self.observation_packets[1:], strict=False
        ):
            if previous.timeline_stop_ns != current.timeline_start_ns:
                raise ValueError("observation packet timeline must be contiguous")
        for record in self.evidence_records:
            record.validate()
        evidence_ids = tuple(record.evidence_id for record in self.evidence_records)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence registry IDs must be unique")
        if evidence_ids != tuple(sorted(evidence_ids)):
            raise ValueError("evidence registry must use canonical evidence order")
        packet_ids = {packet.packet_id for packet in self.observation_packets}
        observations = {
            observation.evidence_id: (packet.packet_id, observation.asset_sha256)
            for packet in self.observation_packets
            for observation in packet.observations
        }
        for record in self.evidence_records:
            if record.kind is ObservationKind.LEGACY:
                continue
            expected = observations.get(record.evidence_id)
            if (
                expected != (record.packet_id, record.asset_sha256)
                or record.packet_id not in packet_ids
            ):
                raise ValueError("nonlegacy evidence must be owned by its exact observation packet")
        self.canon.validate()
        registry = set(evidence_ids)
        for entity in self.canon.entities:
            if not set(entity.supporting_evidence_ids) <= registry:
                raise ValueError("canon support must exist in the evidence registry")
            if any(item.evidence_id not in registry for item in entity.pending):
                raise ValueError("pending canon observations require registered evidence")
        self.policy.validate()
        if not set(self.policy.pinned_evidence_ids) <= registry:
            raise ValueError("pinned evidence must exist in the evidence registry")
        if not set(self.policy.tombstoned_evidence_ids) <= registry:
            raise ValueError("forgotten evidence must exist in the evidence registry")
        return self

    def to_json(self) -> bytes:
        self.validate()
        return canonical_story_json(self)

    @classmethod
    def from_json(cls, encoded: bytes) -> StoryProductState:
        data = decode_canonical_story_object(encoded, field="StoryProductState")
        _mapping(data, "StoryProductState", frozenset(cls.__dataclass_fields__))
        return cls(
            tuple(
                StoryObservationPacket.from_mapping(item)
                for item in _array(data["observation_packets"], "observation_packets")
            ),
            tuple(
                StoryEvidenceRecord.from_mapping(item)
                for item in _array(data["evidence_records"], "evidence_records")
            ),
            StoryCanon.from_mapping(data["canon"]),
            StoryMemoryPolicy.from_mapping(data["policy"]),
        ).validate()


__all__ = (
    "CanonEntity",
    "CanonPresence",
    "GuideBinding",
    "MemoryAction",
    "NativeRGBObservation",
    "ObservationKind",
    "PendingCanonObservation",
    "StoryCanon",
    "StoryEvidencePacketSpec",
    "StoryEvidenceRecord",
    "StoryGenerationReceipt",
    "StoryMemoryCommand",
    "StoryMemoryPolicy",
    "StoryObservation",
    "StoryObservationPacket",
    "StoryProductState",
    "StoryRecallDecision",
)
