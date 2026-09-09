from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image

from comfy_story.story_contracts import ReferenceRole, StoryLibrary, StoryReference
from comfy_story.story_product_contracts import (
    CanonEntity,
    CanonPresence,
    MemoryAction,
    NativeRGBObservation,
    ObservationKind,
    StoryCanon,
    StoryEvidencePacketSpec,
    StoryEvidenceRecord,
    StoryMemoryCommand,
    StoryMemoryPolicy,
    StoryObservationPacket,
    StoryProductState,
)
from comfy_story.story_recall import (
    apply_story_memory_commands,
    render_story_evidence_packet,
    select_story_recall,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _reference(name: str, role: ReferenceRole) -> StoryReference:
    digest = _digest(f"baseline-{name}")
    return StoryReference(
        name,
        role,
        "approved original",
        digest,
        f"comfy-story://assets/sha256/{digest}",
        (digest,),
        _digest(f"pre-{name}"),
    ).validate()


def _library() -> StoryLibrary:
    return StoryLibrary(
        "Workshop",
        (
            _reference("Mara", ReferenceRole.CHARACTER),
            _reference("Orin", ReferenceRole.CHARACTER),
            _reference("BrassKey", ReferenceRole.PROP),
        ),
    ).validate()


def _packet(shot: int, evidence_id: str, entity_id: str, asset: str) -> StoryObservationPacket:
    observation = NativeRGBObservation(
        evidence_id,
        ObservationKind.CLOSING,
        1,
        shot * 5_000_000_000 + 2_500_000_000,
        (shot + 1) * 5_000_000_000,
        (0, 0, 65_536, 65_536),
        _digest(asset),
        _digest(f"pre-{asset}"),
        800_000 + shot,
        0,
        (entity_id,),
    )
    video = _digest(f"video-{shot}")
    return StoryObservationPacket(
        f"shot-{shot:03d}",
        shot,
        2,
        shot * 5_000_000_000,
        (shot + 1) * 5_000_000_000,
        video,
        f"comfy-evidence://story/sha256/{video}",
        _digest("extractor"),
        (entity_id,),
        (observation,),
    ).validate()


def _story(*, presence: CanonPresence) -> StoryProductState:
    packets = (
        _packet(0, "mara-latest", "mara", "mara-latest"),
        _packet(1, "orin-latest", "orin", "orin-latest"),
        _packet(2, "key-bent", "brasskey", "key-bent"),
    )
    records = tuple(
        sorted(
            (
                StoryEvidenceRecord.from_observation(packet, observation)
                for packet in packets
                for observation in packet.observations
            ),
            key=lambda item: item.evidence_id,
        )
    )
    library = _library()
    baselines = {
        reference.name.casefold(): reference.media_sha256 for reference in library.references
    }
    canon = StoryCanon(
        (
            CanonEntity(
                "brasskey",
                "BrassKey",
                baselines["brasskey"],
                "bent, held by Orin",
                presence,
                ("key-bent",),
                "c" * 64,
                (),
            ),
            CanonEntity(
                "mara",
                "Mara",
                baselines["mara"],
                "",
                CanonPresence.OFF_SCREEN,
                ("mara-latest",),
                "a" * 64,
                (),
            ),
            CanonEntity(
                "orin",
                "Orin",
                baselines["orin"],
                "dusty coat",
                presence,
                ("orin-latest",),
                "b" * 64,
                (),
            ),
        )
    )
    return StoryProductState(packets, records, canon, StoryMemoryPolicy()).validate()


def test_latest_confirmed_key_state_returns_after_absence() -> None:
    applied = apply_story_memory_commands(
        _story(presence=CanonPresence.OFF_SCREEN),
        _library(),
        (),
        parent_revision_sha256="a" * 64,
    )

    unrelated = select_story_recall(
        applied,
        _library(),
        prompt="Mara crosses the workshop.",
        reference_policy="Automatic",
    )
    recalled = select_story_recall(
        applied,
        _library(),
        prompt="@Orin returns with @BrassKey.",
        reference_policy="Automatic",
    )

    assert unrelated.selected_evidence_ids == ()
    assert recalled.selected_evidence_ids == ("orin-latest", "key-bent")


def test_prompt_mentions_only_leaves_unused_capacity_empty() -> None:
    applied = apply_story_memory_commands(
        _story(presence=CanonPresence.PRESENT),
        _library(),
        (),
        parent_revision_sha256="a" * 64,
    )

    decision = select_story_recall(
        applied,
        _library(),
        prompt="An empty corridor.",
        reference_policy="Prompt mentions only",
    )

    assert decision.selected_evidence_ids == ()
    assert decision.packet_specs == ()


def test_use_command_precedes_mentions_and_binds_exact_parent() -> None:
    command = StoryMemoryCommand(
        MemoryAction.USE_IN_THIS_SHOT,
        "a" * 64,
        "key-bent",
    )
    applied = apply_story_memory_commands(
        _story(presence=CanonPresence.OFF_SCREEN),
        _library(),
        (command,),
        parent_revision_sha256="a" * 64,
    )

    decision = select_story_recall(
        applied,
        _library(),
        prompt="@Mara waits.",
        reference_policy="Automatic",
    )

    assert decision.selected_evidence_ids == ("key-bent", "mara-latest")


def test_render_packet_is_deterministic_and_preserves_two_source_panels() -> None:
    baseline = _png((255, 0, 0), (40, 20))
    state = _png((0, 0, 255), (20, 40))
    baseline_sha256 = hashlib.sha256(baseline).hexdigest()
    state_sha256 = hashlib.sha256(state).hexdigest()
    assets = {baseline_sha256: baseline, state_sha256: state}
    spec = StoryEvidencePacketSpec(
        "brasskey",
        baseline_sha256,
        "key-bent",
        (baseline_sha256, state_sha256),
    )

    first = render_story_evidence_packet(spec, load_asset=assets.__getitem__)
    second = render_story_evidence_packet(spec, load_asset=assets.__getitem__)

    assert first == second
    with Image.open(io.BytesIO(first)) as image:
        assert image.size == (384, 384)
        assert image.convert("RGB").getpixel((96, 192))[0] > 200
        assert image.convert("RGB").getpixel((288, 192))[2] > 200


def test_approved_state_survives_eviction_from_salience_reservoir() -> None:
    applied = apply_story_memory_commands(
        _story(presence=CanonPresence.PRESENT), _library(), (), parent_revision_sha256="a" * 64
    )
    decision = select_story_recall(
        applied,
        _library(),
        prompt="@BrassKey returns.",
        reference_policy="Prompt mentions only",
    )
    assert decision.selected_evidence_ids == ("key-bent",)


def test_forgetting_only_approved_support_requires_explicit_resolution() -> None:
    applied = apply_story_memory_commands(
        _story(presence=CanonPresence.PRESENT),
        _library(),
        (StoryMemoryCommand(MemoryAction.FORGET, "a" * 64, "key-bent"),),
        parent_revision_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="no active evidence"):
        select_story_recall(
            applied,
            _library(),
            prompt="@BrassKey returns.",
            reference_policy="Prompt mentions only",
        )
    restored = apply_story_memory_commands(
        applied.product_state,
        _library(),
        (StoryMemoryCommand(MemoryAction.RESTORE_ORIGINAL, "a" * 64, "brasskey"),),
        parent_revision_sha256="a" * 64,
    )
    decision = select_story_recall(
        restored,
        _library(),
        prompt="@BrassKey returns.",
        reference_policy="Prompt mentions only",
    )
    assert decision.packet_specs[0].state_evidence_id is None


def test_approved_state_capacity_conflict_is_not_silent_baseline_fallback() -> None:
    applied = apply_story_memory_commands(
        _story(presence=CanonPresence.PRESENT), _library(), (), parent_revision_sha256="a" * 64
    )
    with pytest.raises(ValueError, match="more than two approved state"):
        select_story_recall(
            applied,
            _library(),
            prompt="@Orin and @Mara hold @BrassKey.",
            reference_policy="Prompt mentions only",
        )


def _png(color: tuple[int, int, int], size: tuple[int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG", compress_level=9)
    return output.getvalue()


@pytest.mark.parametrize("action", [MemoryAction.USE_IN_THIS_SHOT, MemoryAction.RESTORE_ORIGINAL])
def test_frame_animation_does_not_silently_ignore_image_memory_commands(
    action: MemoryAction,
) -> None:
    target = "brasskey" if action is MemoryAction.RESTORE_ORIGINAL else "key-bent"
    applied = apply_story_memory_commands(
        _story(presence=CanonPresence.PRESENT),
        _library(),
        (StoryMemoryCommand(action, "a" * 64, target),),
        parent_revision_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="choose Reference shot"):
        select_story_recall(
            applied,
            _library(),
            prompt="@BrassKey returns.",
            reference_policy="Automatic",
            image_references=False,
        )


def test_one_shot_state_evidence_preserves_creator_canon() -> None:
    from comfy_story.story_recall import parse_shot_state_evidence

    state = _story(presence=CanonPresence.PRESENT)
    before = state.to_json()
    applied = apply_story_memory_commands(state, _library(), (), parent_revision_sha256="a" * 64)
    decision = select_story_recall(
        applied,
        _library(),
        prompt="@BrassKey lies on a different table.",
        reference_policy="Prompt mentions only",
        shot_state_evidence=parse_shot_state_evidence('{"brasskey":"key-bent"}'),
    )
    packet = decision.packet_specs[0]
    assert packet.entity_id == "brasskey"
    assert packet.state_evidence_id == "key-bent"
    assert packet.source_sha256s == (_digest("baseline-BrassKey"), _digest("key-bent"))
    assert applied.product_state.to_json() == before
    assert state.to_json() == before


@pytest.mark.parametrize(
    "pairs",
    [
        (("mara", "key-bent"),),
        (("brasskey", "missing"),),
        (("brasskey", "key-bent"), ("brasskey", "key-bent")),
    ],
)
def test_one_shot_state_rejects_wrong_entity_missing_or_duplicate_evidence(
    pairs: tuple[tuple[str, str], ...],
) -> None:
    applied = apply_story_memory_commands(
        _story(presence=CanonPresence.PRESENT), _library(), (), parent_revision_sha256="a" * 64
    )
    with pytest.raises(ValueError, match="shot state evidence"):
        select_story_recall(
            applied,
            _library(),
            prompt="@BrassKey lies on a table.",
            reference_policy="Automatic",
            shot_state_evidence=pairs,
        )


def test_animate_frame_cannot_claim_one_shot_state_recall() -> None:
    applied = apply_story_memory_commands(
        _story(presence=CanonPresence.PRESENT), _library(), (), parent_revision_sha256="a" * 64
    )
    with pytest.raises(ValueError, match="Animate frame cannot use recalled images"):
        select_story_recall(
            applied,
            _library(),
            prompt="@BrassKey lies on a table.",
            reference_policy="Automatic",
            image_references=False,
            shot_state_evidence=(("brasskey", "key-bent"),),
        )


@pytest.mark.parametrize("encoded", ["[]", '{"x":1}', '{"x":"y","x":"z"}', '{ "x": "y" }', "null"])
def test_shot_state_evidence_requires_unambiguous_canonical_pairs(encoded: str) -> None:
    from comfy_story.story_recall import parse_shot_state_evidence

    with pytest.raises(ValueError, match="shot state evidence"):
        parse_shot_state_evidence(encoded)


def test_one_shot_state_cannot_recall_forgotten_evidence() -> None:
    applied = apply_story_memory_commands(
        _story(presence=CanonPresence.PRESENT),
        _library(),
        (StoryMemoryCommand(MemoryAction.FORGET, "a" * 64, "key-bent"),),
        parent_revision_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="active declared entity"):
        select_story_recall(
            applied,
            _library(),
            prompt="@BrassKey rests.",
            reference_policy="Automatic",
            shot_state_evidence=(("brasskey", "key-bent"),),
        )


def test_one_shot_state_conflicts_with_explicit_restore() -> None:
    applied = apply_story_memory_commands(
        _story(presence=CanonPresence.PRESENT),
        _library(),
        (StoryMemoryCommand(MemoryAction.RESTORE_ORIGINAL, "a" * 64, "brasskey"),),
        parent_revision_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="conflicts with explicit"):
        select_story_recall(
            applied,
            _library(),
            prompt="@BrassKey rests.",
            reference_policy="Automatic",
            shot_state_evidence=(("brasskey", "key-bent"),),
        )
