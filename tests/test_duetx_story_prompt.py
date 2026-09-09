from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from duet.duetx.story_contracts import ReferenceRole, ShotIntent, StoryLibrary, StoryReference
from duet.duetx.story_memory_backend import StoryMemoryBackend
from duet.duetx.story_product_contracts import CanonPresence
from duet.duetx.story_prompt import _mentions, structured_reference_prompt
from duet.duetx.story_service import (
    PreparedStoryGeneration,
    StoryGenerationRequest,
    _baseline_product_state,
)


def _prepared() -> PreparedStoryGeneration:
    references = tuple(
        StoryReference(
            name,
            role,
            note,
            "a" * 64,
            "duet-story://assets/sha256/" + "a" * 64,
            ("a" * 64,),
            "b" * 64,
        )
        for name, role, note in (
            ("Crane", ReferenceRole.CHARACTER, "A blue mechanical crane"),
            ("Parcel", ReferenceRole.PROP, "A small orange parcel"),
        )
    )
    library = StoryLibrary("Warehouse", references).validate()
    frame = torch.zeros(1, 8, 8, 3)
    request = StoryGenerationRequest(
        intent=ShotIntent.START_STORY,
        project_id="warehouse",
        branch_id="main",
        previous_story=None,
        previous_frame=None,
        world_frame=frame,
        library=library,
        reference_images={ref.name: frame for ref in references},
        prompt="@Crane lowers @Parcel onto the floor.",
        shot_length_seconds=5,
        variation=41,
        checkpoint_sha256="a" * 64,
        model_configuration_sha256="b" * 64,
        memory_backend=StoryMemoryBackend.MINIMAX_H3,
    )
    return PreparedStoryGeneration(
        request,
        None,
        library,
        ("Crane", "Parcel"),
        ("current", "reference-parcel", "reference-crane"),
        (frame, frame, frame),
        "legacy",
        product_state=_baseline_product_state(library),
    )


def test_structured_prompt_preserves_roster_order_and_creator_action() -> None:
    prepared = _prepared()
    text = structured_reference_prompt(prepared)
    assert "<Subject 1> is @Parcel, the prop referenced by <Picture 2>" in text
    assert "<Subject 2> is @Crane, the character referenced by <Picture 3>" in text
    assert "[Shot 1] <Subject 2> lowers <Subject 1> onto the floor." in text
    headings = [line for line in text.splitlines() if line.endswith(":")]
    assert headings == [
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]
    assert prepared.resolved_prompt == "legacy"
    assert prepared.request.variation == 41


def test_dialogue_and_quoted_text_are_preserved_exactly() -> None:
    text = '@Crane says <d>[Hindi] @Crane घर चलो।</d> beside "@Parcel".'
    assert _mentions(text, {"crane": "<Subject 2>", "parcel": "<Subject 1>"}) == (
        '<Subject 2> says <d>[Hindi] @Crane घर चलो।</d> beside "@Parcel".'
    )


def test_offscreen_state_and_new_composition_are_retained() -> None:
    prepared = _prepared()
    product = prepared.product_state
    assert product is not None
    entities = tuple(
        replace(e, presence=CanonPresence.OFF_SCREEN) if e.entity_id == "parcel" else e
        for e in product.canon.entities
    )
    prepared = replace(
        prepared,
        request=replace(prepared.request, composition="New composition"),
        visual_roles=("current", "reference-crane"),
        visual_guides=prepared.visual_guides[:2],
        product_state=replace(product, canon=replace(product.canon, entities=entities)),
    )
    text = structured_reference_prompt(prepared)
    assert "@Parcel remains off screen" in text
    assert "not its old framing or action" in text
    assert "opening composition anchor" not in text


def test_unsupported_roster_fails_instead_of_dropping_a_reference() -> None:
    with pytest.raises(ValueError, match="semantic guide roster"):
        structured_reference_prompt(replace(_prepared(), visual_roles=("unrecognized-guide",)))


@pytest.mark.parametrize("ending", [False, True])
def test_h3_frame_format_preserves_language_and_matches_images(ending: bool) -> None:
    from duet.duetx.h3_prompt import compile_h3_prompt

    prepared = _prepared()
    action = '@Crane shows "@Parcel नमस्ते". (S1) <d>[Hindi]यह @Parcel है।</d>'
    prepared = replace(
        prepared,
        request=replace(prepared.request, render_profile="Animate frame", prompt=action),
        visual_roles=("current",),
        visual_guides=prepared.visual_guides[:1],
    )
    result = compile_h3_prompt(prepared, duration_ms=5000, ending_frame=ending)
    assert "integrated_multimodal_description:" in result
    assert "subject_definitions:" not in result
    assert action in result
    assert ("Picture 2" in result) == ending
    if ending:
        assert "5.00-second" in result
    assert compile_h3_prompt(prepared, duration_ms=124 * 1000 / 24)


def test_h3_reference_roster_and_mutable_identity() -> None:
    from duet.duetx.h3_prompt import compile_h3_prompt

    result = compile_h3_prompt(_prepared(), duration_ms=5000, ending_frame=True)
    assert "<Subject 1> is @Parcel" in result
    assert "<Subject 2> is @Crane" in result
    assert "<Picture 4> is the final frame" in result
    assert "reference generation + keyframe completion" in result
    assert "partially_preserved" in result
    assert "including explicit transformations" in result
    assert "<Subject 2> lowers <Subject 1>" in result
    assert "confirmed current evidence" not in result
    with pytest.raises(ValueError, match="roster"):
        compile_h3_prompt(replace(_prepared(), visual_roles=()), duration_ms=5000)


def test_h3_film_direction_uses_visible_facts_without_review_jargon() -> None:
    from duet.duetx.film_plan import FilmFact, FilmPlan, FilmShot, FilmStateDefinition
    from duet.duetx.h3_prompt import film_h3_direction

    plan = FilmPlan(
        "test",
        "Test",
        5000,
        (FilmFact("parcel.state", "closed"),),
        (
            FilmShot(
                "one",
                5000,
                "Reveal",
                '@Crane opens @Parcel marked "@Parcel नमस्ते".',
                ("Crane", "Parcel"),
                requires=(FilmFact("parcel.state", "closed"),),
                effects=(FilmFact("parcel.state", "open"),),
                reference_names=("Crane",),
                camera_policy="Locked frame",
            ),
        ),
        state_definitions=(
            FilmStateDefinition("parcel.state", "closed", "The lid is shut"),
            FilmStateDefinition("parcel.state", "open", "The lid is lifted"),
        ),
    )
    result = film_h3_direction(plan, 0)
    assert "At the opening, The lid is shut" in result
    assert "By the end, The lid is lifted" in result
    assert '"@Parcel नमस्ते"' in result
    assert "Crane opens Parcel" in result
    assert "parcel.state" not in result
    assert "Shot purpose" not in result
    assert "static shot" in result
    assert result.endswith("The generated clip is completely silent.")


@pytest.mark.parametrize("profile", ["Reference shot", "Animate frame"])
def test_h3_offscreen_canon_survives_prompt_compilation(profile: str) -> None:
    from duet.duetx.h3_prompt import compile_h3_prompt

    prepared = _prepared()
    assert prepared.product_state is not None
    canon = prepared.product_state.canon
    entities = tuple(
        replace(entity, presence=CanonPresence.OFF_SCREEN)
        if entity.entity_id == "parcel"
        else entity
        for entity in canon.entities
    )
    prepared = replace(
        prepared,
        request=replace(prepared.request, render_profile=profile, scene_entity_names=("Crane",)),
        product_state=replace(prepared.product_state, canon=replace(canon, entities=entities)),
    )
    if profile == "Animate frame":
        prepared = replace(
            prepared, visual_roles=("current",), visual_guides=prepared.visual_guides[:1]
        )
    result = compile_h3_prompt(prepared, duration_ms=5000)
    assert "Parcel remains off screen" in result


def test_generated_dialogue_keeps_language_and_stable_speaker_ids() -> None:
    from duet.duetx.film_plan import FilmCue, FilmPlan, FilmShot
    from duet.duetx.h3_prompt import film_h3_direction

    plan = FilmPlan(
        "voices",
        "Voices",
        2000,
        (),
        (
            FilmShot(
                "one",
                1000,
                "Ask",
                "A question",
                (),
                dialogue=(FilmCue("a", 0, 900, "Your move!", "en", "Zoe"),),
            ),
            FilmShot(
                "two",
                1000,
                "Answer",
                "An answer",
                (),
                dialogue=(FilmCue("b", 0, 900, "Ready.", "en", "Ada"),),
            ),
        ),
    ).validate()
    first = film_h3_direction(plan, 0, generated_audio=True)
    second = film_h3_direction(plan, 1, generated_audio=True)
    assert "(S2) Zoe says <d>[English]Your move!</d>" in first
    assert "(S1) Ada says <d>[English]Ready.</d>" in second
    assert "completely silent" not in first
    assert "<d>" not in film_h3_direction(plan, 0)


def test_generated_film_audio_never_enables_a_second_background_score() -> None:
    from duet.duetx.h3_prompt import compile_h3_prompt

    prepared = _prepared()
    prompt = (
        "@Crane dances to the intended score. "
        "Use the described dialogue and physical sound effects; background music is absent."
    )
    result = compile_h3_prompt(
        replace(prepared, request=replace(prepared.request, prompt=prompt)), duration_ms=5000
    )
    assert result.endswith("non_diegetic_music: N/A")
    assert "overall_soundscape: N/A" not in result


@pytest.mark.parametrize(("language", "expected"), [("en", "English"), ("ja", "Japanese")])
def test_film_language_policy_covers_non_dialogue_shots(language: str, expected: str) -> None:
    from duet.duetx.film_plan import FilmPlan, FilmShot
    from duet.duetx.h3_prompt import film_h3_direction

    plan = FilmPlan(
        "action",
        "Action",
        5000,
        (),
        (FilmShot("one", 5000, "Land", "A character lands in Tokyo and gasps.", ()),),
        languages=(language,),
    ).validate()
    text = film_h3_direction(plan, 0, generated_audio=True)
    assert f"Film speech languages: {expected} only" in text
    assert "This shot contains no dialogue" in text
    assert "explicitly described nonverbal vocalizations" in text
    assert "<d>" not in text
    assert film_h3_direction(plan, 0).endswith("The generated clip is completely silent.")


def test_film_dialogue_is_not_repeated_or_extended_between_windows() -> None:
    from duet.duetx.film_plan import FilmCue, FilmPlan, FilmShot
    from duet.duetx.h3_prompt import film_h3_direction

    plan = FilmPlan(
        "speech",
        "Speech",
        5000,
        (),
        (
            FilmShot(
                "one",
                5000,
                "Signal",
                "A wave",
                (),
                dialogue=(FilmCue("line", 1000, 2000, "Go!", "en", "Ada"),),
            ),
        ),
    ).validate()
    text = film_h3_direction(plan, 0, generated_audio=True)
    assert "exactly once" in text
    assert "Between dialogue windows there are no spoken words" in text
    assert "1.00-2.00 seconds" in text
    assert text.count("<d>[English]Go!</d>") == 1


def test_location_reference_preserves_lighting_separately_from_character_identity() -> None:
    from duet.duetx.h3_prompt import compile_h3_prompt

    prepared = _prepared()
    location = replace(
        prepared.library.references[1],
        role=ReferenceRole.LOCATION,
        note="Wet warehouse floor. Blue light from the west; fixed night exposure.",
    )
    library = replace(prepared.library, references=(prepared.library.references[0], location))
    prepared = replace(
        prepared,
        library=library,
        request=replace(prepared.request, library=library),
        product_state=_baseline_product_state(library),
    )
    result = compile_h3_prompt(prepared, duration_ms=5000)
    assert "preserve the scene architecture" in result
    assert "white balance" in result
    assert "Blue light from the west" in result
    assert "Do not copy people or old actions" in result
    # The character's action remains editable; location retention does not freeze it.
    assert "<Subject 2> lowers <Subject 1>" in result
    assert "pose, action, location and mutable" in result
