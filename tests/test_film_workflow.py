from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any

import pytest

from comfy_story.film_io import film_plan_from_json
from comfy_story.film_plan import (
    FilmFact,
    FilmPlan,
    FilmShot,
    RenderedTake,
    accepted_state,
    validate_rendered_takes,
)
from comfy_story.film_workflow import FilmShotInput, compile_film_workflow


def _plan() -> FilmPlan:
    return FilmPlan(
        "film",
        "Two shots",
        18_000,
        (FilmFact("door", "closed"),),
        (
            FilmShot(
                "one",
                9000,
                "Enter",
                "Ada opens the door",
                ("Ada",),
                effects=(FilmFact("door", "open"),),
            ),
            FilmShot(
                "two",
                9000,
                "Stay",
                "Ada sits",
                ("Ada",),
                requires=(FilmFact("door", "open"),),
                depends_on=("one",),
            ),
        ),
    ).validate()


def _library() -> dict[str, object]:
    return {
        "project_name": "A film",
        "references": [
            {"name": "Ada", "file": "ada.png", "role": "Character", "note": "Blue apron"},
        ],
    }


def test_customer_workflow_wires_generated_story_without_fabricated_approval() -> None:
    plan = _plan()
    graph: dict[str, Any] = compile_film_workflow(
        plan,
        _library(),
        (FilmShotInput("room.png", 1), FilmShotInput("room.png", 2, "voice.wav")),
    )
    first, second = graph["10"]["inputs"], graph["20"]["inputs"]
    assert first["Create"] == "Start Story"
    assert "Previous Story" not in first
    assert second["Create"] == "New Scene"
    assert second["Previous Story"] == ["10", 2]
    assert second["Composition"] == "New composition"
    assert "Previous Frame" not in second
    assert second["Authored audio"] == ["21", 0]
    assert graph["21"]["class_type"] == "LoadAudio"
    assert first["Shot length"] == second["Shot length"] == "10 seconds"
    assert first["Output duration (ms)"] == second["Output duration (ms)"] == 9000
    assert first["Memory actions"] == second["Memory actions"] == "[]"
    assert accepted_state(plan, ()) == {"door": "closed"}


def test_precise_cut_uses_supported_render_block_and_preserves_continuation() -> None:
    original = _plan()
    plan = replace(
        original,
        target_duration_ms=5375,
        shots=(
            replace(original.shots[0], duration_ms=2375),
            replace(original.shots[1], duration_ms=3000, composition="Continue frame"),
        ),
    )
    graph: dict[str, Any] = compile_film_workflow(
        plan,
        _library(),
        (FilmShotInput("room.png", 1), FilmShotInput(None, 2, intent="Next Shot")),
    )
    first, second = graph["10"]["inputs"], graph["20"]["inputs"]
    assert first["Shot length"] == second["Shot length"] == "5 seconds"
    assert first["Output duration (ms)"] == 2375
    assert second["Output duration (ms)"] == 3000
    assert second["Previous Frame"] == ["10", 1]
    assert second["Previous Story"] == ["10", 2]
    assert (first["Variation"], second["Variation"]) == (1, 2)


@pytest.mark.parametrize("filename", ["../secret.png", "/etc/passwd", "a\\b.png", "None", ""])
def test_customer_workflow_rejects_unsafe_input_names(filename: str) -> None:
    with pytest.raises(ValueError, match="input filenames"):
        compile_film_workflow(
            _plan(), _library(), (FilmShotInput(filename, 1), FilmShotInput("a.png", 2))
        )


def test_declared_cast_is_independent_of_conditioning_reference_scope() -> None:
    plan = _plan()
    plan = replace(plan, shots=tuple(replace(shot, reference_names=()) for shot in plan.shots))
    graph: dict[str, Any] = compile_film_workflow(
        plan, _library(), (FilmShotInput("room.png", 1), FilmShotInput("room.png", 2))
    )
    for key in ("10", "20"):
        values = graph[key]["inputs"]
        assert json.loads(values["Scene entities"]) == ["Ada"]
        assert "@Ada" not in values["What happens next?"]


def test_ordinary_next_shot_carries_last_frame_without_a_new_world() -> None:
    graph: dict[str, Any] = compile_film_workflow(
        _plan(),
        _library(),
        (FilmShotInput("room.png", 1), FilmShotInput(None, 2, intent="Next Shot")),
    )
    assert graph["20"]["inputs"]["Create"] == "Next Shot"
    assert graph["20"]["inputs"]["Previous Frame"] == ["10", 1]
    assert graph["20"]["inputs"]["Previous Story"] == ["10", 2]
    plan = _plan()
    plan = replace(
        plan, shots=tuple(replace(shot, composition="Continue frame") for shot in plan.shots)
    )
    with pytest.raises(ValueError, match="uploaded world"):
        compile_film_workflow(
            plan, _library(), (FilmShotInput("room.png", 1), FilmShotInput(None, 2))
        )


def test_workflow_preflights_all_shots_before_returning_graph() -> None:
    with pytest.raises(ValueError, match="every planned shot"):
        compile_film_workflow(_plan(), _library(), ())
    with pytest.raises(ValueError, match="seed range"):
        compile_film_workflow(
            _plan(), _library(), (FilmShotInput("a.png", 1), FilmShotInput("a.png", -1))
        )
    with pytest.raises(ValueError, match="missing from"):
        compile_film_workflow(
            _plan(),
            {"project_name": "Empty", "references": []},
            (FilmShotInput("a.png", 1), FilmShotInput("a.png", 2)),
        )


def test_portable_plan_roundtrip_and_malformed_input() -> None:
    plan = _plan()
    assert film_plan_from_json(json.dumps(asdict(plan))) == plan
    for bad in ("[]", "{}", "not json", '{"shots": null}'):
        with pytest.raises(ValueError, match="malformed film plan"):
            film_plan_from_json(bad)
    data = asdict(plan)
    data["surprise"] = 1
    with pytest.raises(ValueError, match="malformed film plan"):
        film_plan_from_json(json.dumps(data))


def test_draft_ancestry_and_plan_changes_cannot_be_hidden_as_approval() -> None:
    plan = _plan()
    one = RenderedTake("one", plan.shots[0].digest, "a" * 64, "b" * 64, None, 10_125)
    two = RenderedTake("two", plan.shots[1].digest, "c" * 64, "d" * 64, "a" * 64, 10_125)
    validate_rendered_takes(plan, (one, two))
    with pytest.raises(ValueError, match="ancestry"):
        validate_rendered_takes(plan, (one, replace(two, parent_revision_sha256="e" * 64)))
    with pytest.raises(ValueError, match="stale"):
        validate_rendered_takes(plan, (one, replace(two, shot_sha256="e" * 64)))
    with pytest.raises(ValueError, match="too short"):
        validate_rendered_takes(plan, (one, replace(two, source_in_ms=2000)))
    with pytest.raises(ValueError, match="reviewed acceptance"):
        accepted_state(plan, (one,))  # type: ignore[arg-type]  # Exercise the runtime boundary.


@pytest.mark.parametrize(
    "sampler",
    [
        "Native res_multistep",
        "SPEED Euler 2-stage",
        "Turbo 4-step",
        "NVFP4 Exact",
        "NVFP4 Balanced",
        "NVFP4 Ultra Fast",
        "NVFP4 Turbo 4-step",
    ],
)
def test_film_runner_exposes_public_sampler(sampler: str) -> None:
    graph: dict[str, Any] = compile_film_workflow(
        _plan(),
        _library(),
        (FilmShotInput("room.png", 1, sampler=sampler), FilmShotInput("room.png", 2)),
    )
    assert graph["10"]["inputs"]["Sampler"] == sampler
    assert graph["20"]["inputs"]["Sampler"] == "Native res_multistep"


def test_film_runner_rejects_unknown_sampler() -> None:
    with pytest.raises(ValueError, match="film sampler"):
        compile_film_workflow(
            _plan(),
            _library(),
            (
                FilmShotInput("room.png", 1, sampler="fast-ish"),
                FilmShotInput("room.png", 2),
            ),
        )


def test_non_frame_aligned_plan_is_rejected_before_any_shot_is_queued() -> None:
    plan = _plan()
    plan = replace(
        plan,
        target_duration_ms=18001,
        shots=(replace(plan.shots[0], duration_ms=9001), plan.shots[1]),
    )
    with pytest.raises(ValueError, match="exact 24 fps"):
        compile_film_workflow(
            plan, _library(), (FilmShotInput("room.png", 1), FilmShotInput("room.png", 2))
        )


def test_film_ending_guide_uses_public_image_node_and_rejects_unsafe_paths() -> None:
    inputs = (
        FilmShotInput("world.png", 7, ending_frame="guides/end.png"),
        FilmShotInput(None, 8, intent="Next Shot"),
    )
    workflow = compile_film_workflow(_plan(), _library(), inputs)
    assert workflow["12"] == {"class_type": "LoadImage", "inputs": {"image": "guides/end.png"}}
    first, second = workflow["10"], workflow["20"]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert first["inputs"]["Ending frame"] == ["12", 0]
    assert "Ending frame" not in second["inputs"]
    with pytest.raises(ValueError, match="relative Comfy"):
        compile_film_workflow(
            _plan(), _library(), (replace(inputs[0], ending_frame="../escape.png"), inputs[1])
        )


def test_frame_profile_is_explicit_and_preserves_declared_cast() -> None:
    plan = _plan()
    plan = replace(
        plan, shots=tuple(replace(shot, composition="Continue frame") for shot in plan.shots)
    )
    inputs = (FilmShotInput("room.png", 1), FilmShotInput("room.png", 2))
    old: dict[str, Any] = compile_film_workflow(plan, _library(), inputs)
    assert "Render profile" not in old["10"]["inputs"]
    animated: dict[str, Any] = compile_film_workflow(
        plan, _library(), tuple(replace(row, render_profile="Animate frame") for row in inputs)
    )
    assert animated["10"]["inputs"]["Render profile"] == "Animate frame"
    assert json.loads(animated["10"]["inputs"]["Scene entities"]) == ["Ada"]
    with pytest.raises(ValueError, match="Continue frame"):
        compile_film_workflow(
            plan,
            _library(),
            tuple(
                replace(row, render_profile="Animate frame", sampler="SPEED Euler 2-stage")
                for row in inputs
            ),
        )


@pytest.mark.parametrize("profile", ["Animate frame", "Reference shot"])
def test_turbo_eight_preserves_selected_profile_and_seed(profile: str) -> None:
    plan = _plan()
    plan = replace(
        plan, shots=tuple(replace(shot, composition="Continue frame") for shot in plan.shots)
    )
    inputs = (
        FilmShotInput("room.png", 7, sampler="Turbo 8-step", render_profile=profile),
        FilmShotInput("room.png", 8),
    )
    graph: dict[str, Any] = compile_film_workflow(plan, _library(), inputs)
    assert graph["10"]["inputs"]["Sampler"] == "Turbo 8-step"
    assert graph["10"]["inputs"]["Variation"] == 7
    assert graph["10"]["inputs"].get("Render profile", "Reference shot") == profile


def test_reference_composed_first_and_new_scene_can_omit_world() -> None:
    graph: dict[str, Any] = compile_film_workflow(
        _plan(), _library(), (FilmShotInput(None, 100), FilmShotInput(None, 200))
    )
    assert graph["10"]["inputs"]["Create"] == "Start Story"
    assert graph["20"]["inputs"]["Create"] == "New Scene"
    for key, seed in [("10", 100), ("20", 200)]:
        assert graph[key]["inputs"]["World / starting frame"] == "None"
        assert graph[key]["inputs"]["Variation"] == seed
        assert "Previous Frame" not in graph[key]["inputs"]
    assert graph["20"]["inputs"]["Previous Story"] == ["10", 2]


def test_reference_only_film_requires_selected_references() -> None:
    plan = _plan()
    plan = replace(plan, shots=tuple(replace(shot, reference_names=()) for shot in plan.shots))
    with pytest.raises(ValueError, match="uploaded world"):
        compile_film_workflow(plan, _library(), (FilmShotInput(None, 1), FilmShotInput(None, 2)))


def test_prompt_format_is_opt_in_and_does_not_change_legacy_graph_inputs() -> None:
    original = (FilmShotInput("room.png", 1), FilmShotInput("room.png", 2))
    old: dict[str, Any] = compile_film_workflow(_plan(), _library(), original)
    assert all("Prompt format" not in row["inputs"] for row in old.values())
    changed = (
        replace(original[0], prompt_format="Structured reference (experimental)"),
        original[1],
    )
    new: dict[str, Any] = compile_film_workflow(_plan(), _library(), changed)
    assert new["10"]["inputs"].pop("Prompt format") == "Structured reference (experimental)"
    assert new == old
    with pytest.raises(ValueError, match="prompt format"):
        compile_film_workflow(
            _plan(), _library(), (replace(original[0], prompt_format="unknown"), original[1])
        )


@pytest.mark.parametrize(
    "sampler", ["NVFP4 Exact", "NVFP4 Balanced", "NVFP4 Ultra Fast", "NVFP4 Turbo 4-step"]
)
def test_nvfp4_cannot_compile_against_the_animate_checkpoint(sampler: str) -> None:
    plan = _plan()
    plan = replace(
        plan, shots=tuple(replace(shot, composition="Continue frame") for shot in plan.shots)
    )
    with pytest.raises(ValueError, match="Animate frame"):
        compile_film_workflow(
            plan,
            _library(),
            (
                FilmShotInput("room.png", 1, sampler=sampler, render_profile="Animate frame"),
                FilmShotInput("room.png", 2),
            ),
        )


def test_automatic_h3_film_direction_is_versioned_without_migrating_old_recipes() -> None:
    inputs = (FilmShotInput("room.png", 1), FilmShotInput("room.png", 2))
    legacy: dict[str, Any] = compile_film_workflow(_plan(), _library(), inputs)
    automatic: dict[str, Any] = compile_film_workflow(
        _plan(),
        _library(),
        tuple(replace(item, prompt_format="H3 automatic v1") for item in inputs),
    )
    first: Any = automatic["10"]["inputs"]
    old: Any = legacy["10"]["inputs"]
    assert first["Prompt format"] == "H3 automatic v1"
    assert "Shot purpose:" not in first["What happens next?"]
    assert "Ada opens the door" in first["What happens next?"]
    assert "Shot purpose:" in old["What happens next?"]
    assert "Prompt format" not in old
    assert first["Variation"] == old["Variation"] == 1


def test_full_hd_and_generated_audio_use_public_story_inputs() -> None:
    from comfy_story.film_plan import FilmCue

    plan = _plan()
    plan = replace(
        plan,
        shots=(
            replace(
                plan.shots[0],
                composition="Continue frame",
                dialogue=(FilmCue("hello", 0, 1000, "Hello!", "en", "Ada"),),
            ),
            plan.shots[1],
        ),
    )
    settings = (
        FilmShotInput(
            "room.png",
            12,
            sampler="Full HD 2-pass",
            render_profile="Animate frame",
            prompt_format="H3 automatic v1",
        ),
        FilmShotInput("room.png", 13, prompt_format="H3 automatic v1"),
    )
    graph: dict[str, Any] = compile_film_workflow(plan, _library(), settings, generated_audio=True)
    assert graph["10"]["inputs"]["Sampler"] == "Full HD 2-pass"
    assert "<d>[English]Hello!</d>" in graph["10"]["inputs"]["What happens next?"]
    assert graph["20"]["inputs"]["Previous Story"] == ["10", 2]
    with pytest.raises(ValueError, match="Generated film audio requires"):
        compile_film_workflow(
            plan,
            _library(),
            (replace(settings[0], prompt_format="Current"), settings[1]),
            generated_audio=True,
        )
    with pytest.raises(ValueError, match="Generated film audio requires"):
        compile_film_workflow(
            plan,
            _library(),
            (replace(settings[0], audio_file="line.wav"), settings[1]),
            generated_audio=True,
        )
