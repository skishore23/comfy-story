from __future__ import annotations

import json

import pytest

from comfy_story.story_product_contracts import (
    CanonEntity,
    CanonPresence,
    MemoryAction,
    ObservationKind,
    PendingCanonObservation,
    StoryCanon,
    StoryEvidenceRecord,
    StoryMemoryCommand,
    StoryMemoryPolicy,
    StoryObservation,
    StoryObservationPacket,
    StoryProductState,
)


def _digest(character: str) -> str:
    return character * 64


def _product_state() -> StoryProductState:
    observation = StoryObservation(
        evidence_id="evidence-01",
        kind=ObservationKind.OPENING,
        frame_index=0,
        timestamp_start_ns=0,
        timestamp_stop_ns=1_000_000_000,
        geometry_q16=(0, 0, 65_536, 65_536),
        asset_sha256=_digest("a"),
        preprocessing_sha256=_digest("b"),
        vae_sha256=_digest("c"),
        latent_sha256=_digest("d"),
        salience_q=900_000,
        change_score_q=0,
        entity_ids=("maya",),
    )
    packet = StoryObservationPacket(
        packet_id="shot-000",
        shot_index=0,
        frame_count=124,
        timeline_start_ns=0,
        timeline_stop_ns=5_000_000_000,
        source_video_sha256=_digest("e"),
        source_video_locator=f"duet-evidence://story/sha256/{_digest('e')}",
        extraction_policy_sha256=_digest("f"),
        referenced_entity_ids=("maya",),
        observations=(observation,),
    )
    record = StoryEvidenceRecord.from_observation(packet, observation)
    pending = PendingCanonObservation(
        evidence_id="evidence-01",
        state_note="Maya now wears a torn sleeve.",
        source_revision_sha256=_digest("1"),
    )
    canon = StoryCanon(
        (
            CanonEntity(
                entity_id="maya",
                reference_name="Maya",
                baseline_sha256=_digest("9"),
                state_note="Blue coat, torn left sleeve",
                presence=CanonPresence.PRESENT,
                supporting_evidence_ids=("evidence-01",),
                confirmed_revision_sha256=_digest("1"),
                pending=(pending,),
            ),
        )
    )
    return StoryProductState(
        observation_packets=(packet,),
        evidence_records=(record,),
        canon=canon,
        policy=StoryMemoryPolicy(("evidence-01",), ()),
    ).validate()


def test_product_state_round_trips_identical_canonical_bytes() -> None:
    state = _product_state()

    encoded = state.to_json()

    assert StoryProductState.from_json(encoded) == state
    assert StoryProductState.from_json(encoded).to_json() == encoded


def test_product_state_rejects_unknown_fields_and_noncanonical_json() -> None:
    encoded = _product_state().to_json()
    value = json.loads(encoded)
    value["surprise"] = True
    with pytest.raises(ValueError, match="missing or unknown"):
        StoryProductState.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        )

    with pytest.raises(ValueError, match="canonical"):
        StoryProductState.from_json(b'{"policy": {}, "canon": {}}')


def test_product_state_rejects_canon_support_missing_from_evidence_registry() -> None:
    state = _product_state()
    entity = state.canon.entities[0]
    broken = StoryProductState(
        state.observation_packets,
        state.evidence_records,
        StoryCanon(
            (
                CanonEntity(
                    entity.entity_id,
                    entity.reference_name,
                    entity.baseline_sha256,
                    entity.state_note,
                    entity.presence,
                    ("missing-evidence",),
                    entity.confirmed_revision_sha256,
                    entity.pending,
                ),
            )
        ),
        state.policy,
    )

    with pytest.raises(ValueError, match="canon support"):
        broken.validate()


def test_memory_command_binds_parent_target_and_canonical_payload() -> None:
    command = StoryMemoryCommand(
        action=MemoryAction.USE_IN_THIS_SHOT,
        parent_revision_sha256=_digest("a"),
        target_id="evidence-01",
    ).validate()

    assert StoryMemoryCommand.from_json(command.to_json()) == command


def test_policy_rejects_simultaneous_pin_and_tombstone() -> None:
    with pytest.raises(ValueError, match="both pinned and forgotten"):
        StoryMemoryPolicy(("evidence-01",), ("evidence-01",)).validate()
