"""Creator-controlled canon updates and deterministic bounded story recall."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from PIL import Image

from duet.duetx.contracts import ExceptionItem
from duet.duetx.story_contracts import StoryLibrary, canonical_story_json
from duet.duetx.story_product_contracts import (
    CanonEntity,
    CanonPresence,
    MemoryAction,
    PendingCanonObservation,
    StoryCanon,
    StoryEvidencePacketSpec,
    StoryEvidenceRecord,
    StoryMemoryCommand,
    StoryMemoryPolicy,
    StoryProductState,
    StoryRecallDecision,
)

_MENTION = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_-]{0,63})")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class AppliedStoryCommands:
    product_state: StoryProductState
    forced_evidence_ids: tuple[str, ...]
    restored_entity_ids: tuple[str, ...]


def _copy_entity(
    entity: CanonEntity,
    *,
    state_note: str | None = None,
    presence: CanonPresence | None = None,
    supporting_evidence_ids: tuple[str, ...] | None = None,
    confirmed_revision_sha256: str | None = None,
    pending: tuple[PendingCanonObservation, ...] = (),
) -> CanonEntity:
    return CanonEntity(
        entity.entity_id,
        entity.reference_name,
        entity.baseline_sha256,
        entity.state_note if state_note is None else state_note,
        entity.presence if presence is None else presence,
        (
            entity.supporting_evidence_ids
            if supporting_evidence_ids is None
            else supporting_evidence_ids
        ),
        (
            entity.confirmed_revision_sha256
            if confirmed_revision_sha256 is None
            else confirmed_revision_sha256
        ),
        pending,
    ).validate()


def apply_story_memory_commands(
    product_state: StoryProductState,
    library: StoryLibrary,
    commands: tuple[StoryMemoryCommand, ...],
    *,
    parent_revision_sha256: str,
) -> AppliedStoryCommands:
    """Apply one stale-safe command roster to an immutable product read view."""
    product_state.validate()
    library.validate()
    if (
        not isinstance(parent_revision_sha256, str)
        or _SHA256.fullmatch(parent_revision_sha256) is None
    ):
        raise ValueError("parent_revision_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(commands, tuple):
        raise ValueError("story memory commands must be a tuple")
    targets: set[str] = set()
    evidence = {item.evidence_id: item for item in product_state.evidence_records}
    entities = {item.entity_id: item for item in product_state.canon.entities}
    references = {item.name.casefold(): item for item in library.references}
    pinned = set(product_state.policy.pinned_evidence_ids)
    tombstoned = set(product_state.policy.tombstoned_evidence_ids)
    forced: list[str] = []
    restored: list[str] = []
    for command in commands:
        command.validate()
        if command.parent_revision_sha256 != parent_revision_sha256:
            raise ValueError("story memory command targets a stale parent revision")
        if command.target_id in targets:
            raise ValueError("conflicting story memory commands target the same item")
        targets.add(command.target_id)
        if (
            command.action
            in {
                MemoryAction.KEEP,
                MemoryAction.USE_IN_THIS_SHOT,
                MemoryAction.LET_FADE,
                MemoryAction.FORGET,
            }
            and command.target_id not in evidence
        ):
            raise ValueError("story memory command names unavailable evidence")
        if command.action is MemoryAction.KEEP:
            if command.target_id in tombstoned:
                raise ValueError("forgotten evidence cannot be kept")
            pinned.add(command.target_id)
        elif command.action is MemoryAction.USE_IN_THIS_SHOT:
            if command.target_id in tombstoned:
                raise ValueError("forgotten evidence cannot be recalled")
            forced.append(command.target_id)
        elif command.action is MemoryAction.LET_FADE:
            pinned.discard(command.target_id)
        elif command.action is MemoryAction.FORGET:
            pinned.discard(command.target_id)
            tombstoned.add(command.target_id)
            entities = {
                entity_id: _copy_entity(
                    entity,
                    # Keep the approved state and its provenance. Recall filters
                    # tombstones and exposes missing support instead of silently
                    # substituting the baseline appearance.
                    pending=tuple(
                        item for item in entity.pending if item.evidence_id != command.target_id
                    ),
                )
                for entity_id, entity in entities.items()
            }
        elif command.action is MemoryAction.UPDATE_CANON:
            entity = entities.get(command.target_id)
            if entity is None:
                raise ValueError("Update canon names an unavailable Story Library entity")
            if not command.state_note.strip():
                raise ValueError("Update canon requires a creator-confirmed state note")
            if any(
                item not in evidence or item in tombstoned
                for item in command.supporting_evidence_ids
            ):
                raise ValueError("Update canon requires authenticated active support")
            entities[command.target_id] = _copy_entity(
                entity,
                state_note=command.state_note.strip(),
                presence=command.presence,
                supporting_evidence_ids=tuple(sorted(command.supporting_evidence_ids)),
                confirmed_revision_sha256=parent_revision_sha256,
                pending=(),
            )
        elif command.action is MemoryAction.RESTORE_ORIGINAL:
            entity = entities.get(command.target_id)
            reference = references.get(command.target_id)
            if entity is None or reference is None:
                raise ValueError("Restore original names an unavailable Story Library entity")
            entities[command.target_id] = _copy_entity(
                entity,
                state_note=reference.note,
                presence=CanonPresence.UNKNOWN,
                supporting_evidence_ids=(),
                confirmed_revision_sha256=parent_revision_sha256,
                pending=(),
            )
            restored.append(command.target_id)
    next_state = StoryProductState(
        product_state.observation_packets,
        product_state.evidence_records,
        StoryCanon(tuple(sorted(entities.values(), key=lambda item: item.entity_id))),
        StoryMemoryPolicy(tuple(sorted(pinned)), tuple(sorted(tombstoned))),
    ).validate()
    return AppliedStoryCommands(next_state, tuple(forced), tuple(restored))


def _mentions(prompt: str, library: StoryLibrary) -> tuple[str, ...]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("What happens next? must be nonempty")
    lookup = {reference.name.casefold(): reference for reference in library.references}
    result: list[str] = []
    for match in _MENTION.finditer(prompt):
        entity_id = match.group(1).casefold()
        if entity_id not in lookup:
            raise ValueError(f"Unknown Story Library reference: @{match.group(1)}")
        if entity_id not in result:
            result.append(entity_id)
    if len(result) > 7:
        raise ValueError("MiniMax shots support at most seven explicit Story entities")
    return tuple(result)


def _latest_record(
    entity: CanonEntity,
    evidence: dict[str, StoryEvidenceRecord],
    tombstoned: set[str],
) -> StoryEvidenceRecord | None:
    candidates = tuple(
        evidence[item]
        for item in entity.supporting_evidence_ids
        if item in evidence and item not in tombstoned
    )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -(item.source_shot_index if item.source_shot_index is not None else -1),
            item.evidence_id,
            item.fingerprint_sha256,
        ),
    )


def _entity_source_rank(entity: CanonEntity, evidence: dict[str, StoryEvidenceRecord]) -> int:
    ranks = tuple(
        record.source_shot_index
        for evidence_id in entity.supporting_evidence_ids
        if (record := evidence.get(evidence_id)) is not None
        and record.source_shot_index is not None
    )
    return max(ranks, default=-1)


def _packet_for_entity(
    entity: CanonEntity,
    record: StoryEvidenceRecord | None,
    *,
    restored: bool,
) -> StoryEvidencePacketSpec:
    if record is None or restored:
        return StoryEvidencePacketSpec(
            entity.entity_id,
            entity.baseline_sha256,
            None,
            (entity.baseline_sha256,),
        ).validate()
    sources = (
        (entity.baseline_sha256,)
        if entity.baseline_sha256 == record.asset_sha256
        else (entity.baseline_sha256, record.asset_sha256)
    )
    return StoryEvidencePacketSpec(
        entity.entity_id,
        entity.baseline_sha256,
        record.evidence_id,
        sources,
    ).validate()


def parse_shot_state_evidence(encoded: object) -> tuple[tuple[str, str], ...]:
    """Parse an explicit one-shot entity/evidence roster, never a canon approval."""
    if not isinstance(encoded, str) or len(encoded.encode()) > 65536:
        raise ValueError("shot state evidence must be bounded canonical JSON")
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("shot state evidence requires JSON") from error
    if (
        not isinstance(value, dict)
        or len(value) > 2
        or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in value.items()
        )
        or canonical_story_json(value) != encoded.encode()
    ):
        raise ValueError("shot state evidence requires at most two canonical entity/evidence pairs")
    return tuple(sorted(value.items()))


def select_story_recall(
    applied: AppliedStoryCommands,
    library: StoryLibrary,
    reservoir: tuple[ExceptionItem, ...],
    *,
    prompt: str,
    reference_policy: str,
    image_references: bool = True,
    shot_state_evidence: tuple[tuple[str, str], ...] = (),
) -> StoryRecallDecision:
    """Choose up to two exact packets while retaining a larger explicit entity roster."""
    if not isinstance(applied, AppliedStoryCommands):
        raise ValueError("applied must be an AppliedStoryCommands value")
    state = applied.product_state.validate()
    library.validate()
    if reference_policy not in {"Automatic", "Prompt mentions only"}:
        raise ValueError("Reference policy is unsupported")
    explicit = _mentions(prompt, library)
    if (
        not isinstance(shot_state_evidence, tuple)
        or len(shot_state_evidence) > 2
        or any(
            not isinstance(row, tuple)
            or len(row) != 2
            or any(not isinstance(value, str) or not value for value in row)
            for row in shot_state_evidence
        )
    ):
        raise ValueError("shot state evidence requires at most two entity/evidence pairs")
    if not image_references:
        if shot_state_evidence or applied.forced_evidence_ids or applied.restored_entity_ids:
            raise ValueError("Animate frame cannot use recalled images; choose Reference shot")
        # Names still label the intended cast. No image evidence is claimed as used.
        return StoryRecallDecision(explicit, (), (), (), (), (), None, ()).validate()
    evidence = {item.evidence_id: item for item in state.evidence_records}
    entities = {item.entity_id: item for item in state.canon.entities}
    # The bounded salience reservoir is an acceleration structure, not the
    # authority for creator-approved or explicitly requested evidence.
    del reservoir
    tombstoned = set(state.policy.tombstoned_evidence_ids)
    # A one-shot read view can use selected evidence without changing creator canon.
    overrides = dict(shot_state_evidence)
    if len(overrides) != len(shot_state_evidence):
        raise ValueError("shot state evidence repeats an entity")
    if overrides and (applied.forced_evidence_ids or applied.restored_entity_ids):
        raise ValueError("shot state evidence conflicts with explicit recall or restore commands")
    for entity_id, evidence_id in overrides.items():
        entity, record = entities.get(entity_id), evidence.get(evidence_id)
        if (
            entity is None
            or entity_id not in explicit
            or record is None
            or evidence_id in tombstoned
            or entity_id not in record.entity_ids
            or record.source_shot_index is None
        ):
            raise ValueError(
                "shot state evidence must bind an active declared entity to historical evidence"
            )
        entities[entity_id] = replace(entity, supporting_evidence_ids=(evidence_id,))
    selected_ids: list[str] = []
    packets: list[StoryEvidencePacketSpec] = []
    selected_entities: set[str] = set()
    rejected: list[str] = []
    versions: list[str] = []

    def add_record(record: StoryEvidenceRecord, entity_id: str) -> None:
        if len(packets) >= 2 or record.evidence_id in selected_ids:
            return
        entity = entities.get(entity_id)
        if entity is None:
            packet = StoryEvidencePacketSpec(
                entity_id,
                record.asset_sha256,
                record.evidence_id,
                (record.asset_sha256,),
            ).validate()
        else:
            packet = _packet_for_entity(entity, record, restored=False)
            selected_entities.add(entity.entity_id)
            if entity.confirmed_revision_sha256 is not None:
                versions.append(entity.confirmed_revision_sha256)
        packets.append(packet)
        selected_ids.append(record.evidence_id)

    for evidence_id in applied.forced_evidence_ids:
        record = evidence.get(evidence_id)
        if record is None or evidence_id in tombstoned:
            raise ValueError(f"Requested evidence is unavailable: {evidence_id}")
        if len(packets) >= 2 and evidence_id not in selected_ids:
            raise ValueError("This shot requests more than two exact evidence packets")
        entity_id = record.entity_ids[0] if record.entity_ids else "memory"
        add_record(record, entity_id)

    def add_entity(entity_id: str, *, explicit_mention: bool) -> None:
        if entity_id in selected_entities:
            return
        entity = entities.get(entity_id)
        if entity is None:
            rejected.append(f"{entity_id}:canon-unavailable")
            return
        record = _latest_record(entity, evidence, tombstoned)
        if entity.supporting_evidence_ids and record is None:
            raise ValueError(
                f"Approved state for @{entity.reference_name} has no active evidence; "
                "approve replacement evidence or restore original"
            )
        if len(packets) >= 2:
            if record is not None:
                raise ValueError(
                    "This shot needs more than two approved state packets; "
                    "reduce its active cast or split the shot"
                )
            return
        packet = _packet_for_entity(
            entity,
            record,
            restored=entity_id in applied.restored_entity_ids,
        )
        packets.append(packet)
        selected_entities.add(entity_id)
        if record is not None and packet.state_evidence_id is not None:
            selected_ids.append(record.evidence_id)
        elif not explicit_mention and entity.supporting_evidence_ids:
            rejected.append(f"{entity_id}:historical-evidence-unavailable")
        if entity.confirmed_revision_sha256 is not None and entity_id not in overrides:
            versions.append(entity.confirmed_revision_sha256)

    for entity_id in sorted(
        explicit, key=lambda item: not bool(entities[item].supporting_evidence_ids)
    ):
        add_entity(entity_id, explicit_mention=True)
    if reference_policy == "Automatic":
        present = tuple(
            sorted(
                (
                    entity
                    for entity in entities.values()
                    if entity.presence is CanonPresence.PRESENT
                ),
                key=lambda entity: (
                    -_entity_source_rank(entity, evidence),
                    entity.entity_id,
                ),
            )
        )
        for entity in present:
            add_entity(entity.entity_id, explicit_mention=False)
    return StoryRecallDecision(
        explicit,
        applied.forced_evidence_ids,
        tuple(dict.fromkeys(versions)),
        tuple(rejected),
        tuple(selected_ids),
        tuple(packets),
        None,
        (),
    ).validate()


def _load_rgb(encoded: bytes, digest: str) -> Image.Image:
    if not isinstance(encoded, bytes) or hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError("evidence packet source SHA-256 changed")
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            return image.convert("RGB")
    except Exception as error:
        raise ValueError("evidence packet source is not a decodable image") from error


def render_story_evidence_packet(
    spec: StoryEvidencePacketSpec,
    *,
    load_asset: Callable[[str], bytes],
) -> bytes:
    """Render one exact or deterministic baseline-plus-state MiniMax guide."""
    spec.validate()
    if len(spec.source_sha256s) > 2:
        raise ValueError("initial Story evidence packets support at most two source images")
    encoded = tuple(load_asset(digest) for digest in spec.source_sha256s)
    images = tuple(
        _load_rgb(value, digest) for value, digest in zip(encoded, spec.source_sha256s, strict=True)
    )
    if len(images) == 1:
        return encoded[0]
    canvas = Image.new("RGB", (384, 384), (127, 127, 127))
    for index, image in enumerate(images):
        fitted = image.copy()
        fitted.thumbnail((192, 384), resample=Image.Resampling.LANCZOS)
        left = index * 192 + (192 - fitted.width) // 2
        top = (384 - fitted.height) // 2
        canvas.paste(fitted, (left, top))
    output = io.BytesIO()
    canvas.save(output, format="PNG", compress_level=9)
    return output.getvalue()


__all__ = (
    "AppliedStoryCommands",
    "apply_story_memory_commands",
    "render_story_evidence_packet",
    "select_story_recall",
)
