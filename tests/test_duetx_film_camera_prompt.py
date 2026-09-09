from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from duet.duetx.film_io import film_plan_from_json
from duet.duetx.film_plan import FilmPlan, FilmShot, compile_candidate_prompt, compile_shot_prompt
from duet.duetx.film_workflow import FilmShotInput, compile_film_workflow
from duet.duetx.story_contracts import canonical_story_json


def _plan() -> FilmPlan:
    return FilmPlan(
        "camera-version",
        "A movement",
        5000,
        (),
        (
            FilmShot(
                "one",
                5000,
                "A movement",
                "A leaf moves.",
                (),
                camera_policy="Single continuous shot",
            ),
        ),
    ).validate()


def test_legacy_camera_prompt_and_identity_remain_recoverable() -> None:
    plan = _plan()
    assert (
        plan.shots[0].digest == "9421de4dfa10937dd6c4ad47e40531eb1c1e3ca2fc44fe3435258d2ac1381e23"
    )
    encoded = canonical_story_json(plan).decode()
    assert "direction_version" not in encoded
    assert film_plan_from_json(encoded) == plan
    assert (
        "No internal cut, dissolve, crossfade, wipe, or superimposed alternate view."
        in compile_candidate_prompt(plan, 0)
    )


def test_positive_camera_version_changes_only_its_clause_and_binds_public_request() -> None:
    old = _plan()
    new = replace(old, shots=(replace(old.shots[0], direction_version=2),)).validate()
    assert new.shots[0].digest != old.shots[0].digest
    assert film_plan_from_json(canonical_story_json(new).decode()) == new
    before = "No internal cut, dissolve, crossfade, wipe, or superimposed alternate view."
    after = "The same view and solid, opaque subjects persist throughout the shot."
    assert compile_candidate_prompt(new, 0) == compile_candidate_prompt(old, 0).replace(
        before, after
    )
    assert compile_shot_prompt(new, 0, ()) == compile_shot_prompt(old, 0, ()).replace(before, after)
    library: dict[str, object] = {"project_name": "Camera version", "references": []}
    settings = (FilmShotInput("leaf.png", 73960),)
    legacy_graph: dict[str, Any] = compile_film_workflow(old, library, settings)
    new_graph: dict[str, Any] = compile_film_workflow(new, library, settings)
    assert legacy_graph != new_graph
    assert (
        new_graph["10"]["inputs"]["Variation"] == legacy_graph["10"]["inputs"]["Variation"] == 73960
    )
    assert after in new_graph["10"]["inputs"]["What happens next?"]
    assert before in legacy_graph["10"]["inputs"]["What happens next?"]


@pytest.mark.parametrize("version", [0, 3, True, 2.0, "2", None])
def test_direction_version_is_explicit_and_strict(version: object) -> None:
    payload = json.loads(canonical_story_json(_plan()))
    payload["shots"][0]["direction_version"] = version
    with pytest.raises(ValueError, match="shot direction version"):
        film_plan_from_json(json.dumps(payload))


def test_positive_direction_keeps_exclusions_in_review_not_generation() -> None:
    from duet.duetx.film_audit import visual_audit_prompt

    original = replace(_plan().shots[0], absent=("Vehicle", "Writing"))
    positive = replace(original, direction_version=2)
    old = replace(_plan(), shots=(original,)).validate()
    new = replace(old, shots=(positive,)).validate()
    assert "Do not depict: Vehicle, Writing" in compile_candidate_prompt(old, 0)
    assert "Do not depict: Vehicle, Writing" not in compile_candidate_prompt(new, 0)
    assert positive.absent == original.absent
    assert visual_audit_prompt(positive, (0, 119), {}) == visual_audit_prompt(
        original, (0, 119), {}
    )
    assert '"Vehicle"' in visual_audit_prompt(positive, (0, 119), {})[0]


def test_direction_edit_invalidates_its_descendants_without_changing_seeds() -> None:
    from duet.duetx.film_project import edit_impact

    shot = _plan().shots[0]
    before = replace(
        _plan(),
        target_duration_ms=15000,
        shots=(shot, replace(shot, shot_id="two"), replace(shot, shot_id="three")),
    ).validate()
    after = replace(
        before, shots=(shot, replace(before.shots[1], direction_version=2), before.shots[2])
    ).validate()
    inputs: dict[str, Any] = {
        "library": {"project_name": "Camera", "references": []},
        "shots": [{"world": "leaf.png", "variation": i} for i in (73960, 73961, 73962)],
    }
    impact = edit_impact(before, inputs, after, inputs)
    assert impact["reusable_prefix"] == ["one"]
    assert impact["requires_generation_review"] == ["two", "three"]
    assert [row["variation"] for row in inputs["shots"]] == [73960, 73961, 73962]
