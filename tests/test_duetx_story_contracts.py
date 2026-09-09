from __future__ import annotations

import json

import pytest

from duet.duetx.story_contracts import (
    DuetStoryStateRef,
    ReferenceRole,
    ShotIntent,
    StoryLibrary,
    StoryReference,
    canonical_story_json,
)


def _reference(name: str, role: ReferenceRole, digest: str) -> StoryReference:
    return StoryReference(
        name=name,
        role=role,
        note="",
        media_sha256=digest,
        evidence_locator=f"duet-story://assets/sha256/{digest}",
        keyframe_sha256s=(digest,),
        preprocessing_sha256="f" * 64,
    ).validate()


def test_library_preserves_order_and_resolves_case_insensitive_mentions() -> None:
    library = StoryLibrary(
        project_name="Coast Story",
        references=(
            _reference("Maya", ReferenceRole.CHARACTER, "1" * 64),
            _reference("WoodenChest", ReferenceRole.PROP, "2" * 64),
            _reference("Coast", ReferenceRole.LOCATION, "3" * 64),
        ),
    ).validate()

    assert library.resolve_mentions("@maya opens @WoodenChest") == (
        library.references[0],
        library.references[1],
    )
    with pytest.raises(ValueError, match="at most two exact references"):
        library.resolve_mentions("@Maya carries @WoodenChest toward @Coast")


def test_h3_resolver_allows_larger_roster_in_library_order() -> None:
    references = tuple(
        _reference(f"Ref{index}", ReferenceRole.CHARACTER, f"{index + 1:x}" * 64)
        for index in range(8)
    )
    library = StoryLibrary("H3 Story", references).validate()
    prompt = " ".join(f"@Ref{index}" for index in reversed(range(8)))

    assert tuple(reference.name for reference in library.resolve_for_h3(prompt)) == tuple(
        f"Ref{index}" for index in range(8)
    )
    with pytest.raises(ValueError, match="at most 7"):
        library.resolve_for_h3(prompt, limit=7)
    with pytest.raises(ValueError, match=r"range \[1,9\]"):
        library.resolve_for_h3(prompt, limit=10)


def test_library_rejects_unknown_duplicate_and_missing_mentions() -> None:
    maya = _reference("Maya", ReferenceRole.CHARACTER, "1" * 64)
    library = StoryLibrary("Coast Story", (maya,)).validate()

    with pytest.raises(ValueError, match="Unknown Story Library reference: @Ghost"):
        library.resolve_mentions("@Ghost arrives")
    with pytest.raises(ValueError, match="Mention one or two"):
        library.resolve_mentions("A figure arrives")
    with pytest.raises(ValueError, match="must be unique"):
        StoryLibrary(
            "Coast Story", (maya, _reference("maya", ReferenceRole.PROP, "2" * 64))
        ).validate()


def test_library_round_trip_is_canonical_and_rejects_noncanonical_json() -> None:
    library = StoryLibrary(
        "Coast Story",
        (_reference("Maya", ReferenceRole.CHARACTER, "1" * 64),),
    ).validate()
    encoded = library.to_json()

    assert StoryLibrary.from_json(encoded) == library
    assert encoded == canonical_story_json(library)
    noncanonical = json.dumps(json.loads(encoded), indent=2).encode()
    with pytest.raises(ValueError, match="canonical JSON"):
        StoryLibrary.from_json(noncanonical)


def test_state_reference_round_trips_as_canonical_content() -> None:
    state = DuetStoryStateRef(
        project_id="coast-story",
        branch_id="main",
        parent_revision_sha256=None,
        revision_sha256="1" * 64,
        library_sha256="2" * 64,
        memory_manifest_sha256="3" * 64,
        shot_count=1,
        next_history_slot=1,
        last_frame_sha256="4" * 64,
        model_configuration_sha256="5" * 64,
    ).validate()

    assert DuetStoryStateRef.from_json(state.to_json()) == state
    assert state.to_json() == canonical_story_json(state)


@pytest.mark.parametrize("shot_count", [-1, 129])
def test_state_reference_rejects_shot_counts_outside_capacity(shot_count: int) -> None:
    with pytest.raises(ValueError, match="shot_count"):
        DuetStoryStateRef(
            project_id="coast-story",
            branch_id="main",
            parent_revision_sha256=None,
            revision_sha256="1" * 64,
            library_sha256="2" * 64,
            memory_manifest_sha256="3" * 64,
            shot_count=shot_count,
            next_history_slot=shot_count,
            last_frame_sha256="4" * 64,
            model_configuration_sha256="5" * 64,
        ).validate()


def test_shot_intents_use_customer_language() -> None:
    assert tuple(intent.value for intent in ShotIntent) == (
        "Start Story",
        "Continue This Shot",
        "Next Shot",
        "New Scene",
    )
