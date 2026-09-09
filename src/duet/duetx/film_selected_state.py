"""Carry machine-selected appearance evidence without changing creator-approved canon.

The production controller constructs the selected prefix only after authenticating each audit's
raw observations. This module checks that prefix against archive lineage and binds relevant
ending evidence to the next reference shot. It never turns a planned fact into approval.
"""

from __future__ import annotations

from dataclasses import dataclass

from duet.duetx.film_audit import ShotVisualAudit
from duet.duetx.film_plan import FilmPlan, planned_shot_conditions
from duet.duetx.film_state_audit import ending_conditions
from duet.duetx.story_native_archive import NativeReferenceArchive
from duet.duetx.story_product_contracts import ObservationKind


@dataclass(frozen=True, slots=True)
class SelectedStateSource:
    revision_sha256: str
    assessment_sha256: str
    audit: ShotVisualAudit


@dataclass(frozen=True, slots=True)
class ShotStateBinding:
    entity_id: str
    evidence_id: str
    asset_sha256: str
    source_shot_id: str
    source_revision_sha256: str
    source_video_sha256: str
    assessment_sha256: str
    facts: tuple[tuple[str, str], ...]


def select_film_state_evidence(
    plan: FilmPlan,
    index: int,
    selected: tuple[SelectedStateSource, ...],
    archive: NativeReferenceArchive,
) -> tuple[ShotStateBinding, ...]:
    """Select up to two current entity snapshots from the authenticated selected prefix."""
    plan.validate()
    if not 0 <= index < len(plan.shots) or len(selected) != index:
        raise ValueError("state recall requires the exact selected prefix")
    if not selected:
        return ()
    for shot, source in zip(plan.shots[:index], selected, strict=True):
        audit = source.audit
        if (
            audit.status != "machine_pass"
            or audit.shot_id != shot.shot_id
            or audit.shot_sha256 != shot.digest
            or len(source.assessment_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source.assessment_sha256)
        ):
            raise ValueError(
                "state recall requires authenticated passing assessments for this plan"
            )
    parent = archive.load_revision(selected[-1].revision_sha256)
    packets = parent.product_state.observation_packets
    if len(packets) != index or any(
        packet.source_video_sha256 != source.audit.video_sha256
        for packet, source in zip(packets, selected, strict=True)
    ):
        raise ValueError("state recall videos do not match the selected archive prefix")
    # Exact ancestor revisions matter even if two branches happen to contain identical video.
    cursor = parent.state
    for source in reversed(selected):
        if cursor.revision_sha256 != source.revision_sha256:
            raise ValueError("state recall revision lineage changed")
        if cursor.parent_revision_sha256 is not None:
            cursor = archive.load_revision(cursor.parent_revision_sha256).state
    entities = {name.casefold() for name in plan.shots[index].present}
    conditions = planned_shot_conditions(plan, index)
    writers = {fact.key: i for i, shot in enumerate(plan.shots[:index]) for fact in shot.effects}
    targets: dict[str, dict[str, str]] = {}
    for key, value in conditions.items():
        entity = key.split(".", 1)[0].casefold()
        if entity in entities and key in writers:
            targets.setdefault(entity, {})[key] = value
    if len(targets) > 2:
        raise ValueError("selected state requires more than two entity packets; split the shot")
    bindings = []
    tombstoned = set(parent.product_state.policy.tombstoned_evidence_ids)
    for entity, facts in sorted(targets.items()):
        newest_change = max(writers[key] for key in facts)
        for source_index in range(index - 1, newest_change - 1, -1):
            source_shot = plan.shots[source_index]
            if entity not in {name.casefold() for name in source_shot.present}:
                continue
            ending = ending_conditions(plan, source_index)
            if any(ending.get(key) != value for key, value in facts.items()):
                continue
            packet = packets[source_index]
            records = [
                record
                for record in parent.product_state.evidence_records
                if record.packet_id == packet.packet_id
                and record.kind is ObservationKind.CLOSING
                and record.frame_index == packet.frame_count - 1
                and entity in record.entity_ids
                and record.evidence_id not in tombstoned
            ]
            if len(records) != 1:
                raise ValueError("selected state lacks a unique active ending observation")
            record, source = records[0], selected[source_index]
            bindings.append(
                ShotStateBinding(
                    entity,
                    record.evidence_id,
                    record.asset_sha256,
                    source_shot.shot_id,
                    source.revision_sha256,
                    source.audit.video_sha256,
                    source.assessment_sha256,
                    tuple(sorted(facts.items())),
                )
            )
            break
        else:
            raise ValueError("no selected ending snapshot covers the required current entity state")
    return tuple(bindings)
