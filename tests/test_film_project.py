from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from comfy_story.film_narrative import FilmNarrativeContract
from comfy_story.film_plan import FilmFact, FilmPlan, FilmShot
from comfy_story.film_project import FilmProjectConflict, FilmProjectStore, edit_impact
from comfy_story.story_contracts import canonical_story_json


def _recipe() -> tuple[FilmPlan, dict[str, Any]]:
    plan = FilmPlan(
        "project",
        "Story",
        15000,
        (),
        tuple(FilmShot(name, 5000, "Show", "Hold", ()) for name in ("one", "two", "three")),
    )
    inputs = {
        "library": {"project_name": "Story", "references": []},
        "shots": [{"world": "world.png", "variation": i} for i in (11, 22, 33)],
        "burn_subtitles": False,
    }
    return plan, inputs


def test_edit_middle_shot_preserves_original_recipe_and_invalidates_descendants(
    tmp_path: Path,
) -> None:
    plan, inputs = _recipe()
    store = FilmProjectStore(tmp_path)
    original = store.save("project", plan, inputs, expected_revision=None)
    edited = replace(
        plan, shots=(plan.shots[0], replace(plan.shots[1], action="Move"), plan.shots[2])
    )
    revised = store.save(
        "project", edited, original["inputs"], expected_revision=original["revision"]
    )
    assert revised["revision"] != original["revision"]
    assert store.load("project", original["revision"]) == original
    assert store.load("project") == revised
    impact = store.impact("project", original["revision"], revised["revision"])
    assert impact["reusable_prefix"] == ["one"]
    assert impact["requires_generation_review"] == ["two", "three"]
    assert revised["inputs"]["shots_by_id"]["two"]["variation"] == 22


def test_concurrent_editor_cannot_overwrite_newer_draft(tmp_path: Path) -> None:
    plan, inputs = _recipe()
    store = FilmProjectStore(tmp_path)
    first = store.save("project", plan, inputs, expected_revision=None)
    second = store.save(
        "project", replace(plan, title="Edited"), inputs, expected_revision=first["revision"]
    )
    with pytest.raises(FilmProjectConflict, match="draft changed"):
        store.save(
            "project", replace(plan, title="Stale"), inputs, expected_revision=first["revision"]
        )
    assert store.load("project") == second
    assert (
        store.save(
            "project", replace(plan, title="Edited"), inputs, expected_revision=second["revision"]
        )
        == second
    )


def test_audio_only_edit_keeps_all_video_recipes() -> None:
    plan, inputs = _recipe()
    result = edit_impact(plan, inputs, plan, {**inputs, "audio": [{"path": "new-music.wav"}]})
    assert result["reusable_prefix"] == ["one", "two", "three"]
    assert result["requires_generation_review"] == []
    assert result["export_changed"] is True


def test_changed_initial_state_invalidates_the_entire_generation_prefix() -> None:
    plan, inputs = _recipe()
    changed = replace(plan, initial_facts=(FilmFact("world.weather", "rain"),))
    result = edit_impact(plan, inputs, changed, inputs)
    assert result["reusable_prefix"] == []
    assert result["requires_generation_review"] == ["one", "two", "three"]


def test_removed_tail_keeps_existing_prefix_but_changes_export() -> None:
    plan, inputs = _recipe()
    result = edit_impact(
        plan,
        inputs,
        replace(plan, target_duration_ms=10000, shots=plan.shots[:2]),
        {**inputs, "shots": inputs["shots"][:2]},
    )
    assert result["reusable_prefix"] == ["one", "two"]
    assert result["requires_generation_review"] == []
    assert result["removed_shots"] == ["three"]
    assert result["export_changed"] is True


def test_recipe_corruption_is_not_silently_reused(tmp_path: Path) -> None:
    plan, inputs = _recipe()
    store = FilmProjectStore(tmp_path)
    saved = store.save("project", plan, inputs, expected_revision=None)
    path = tmp_path / "project" / "recipes" / f"{saved['revision']}.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        store.load("project")


@pytest.mark.parametrize("project_id", ["../outside", "", "x/y", "x\\y", ".", ".."])
def test_project_ids_cannot_escape_storage(tmp_path: Path, project_id: str) -> None:
    with pytest.raises(ValueError, match="project ID"):
        FilmProjectStore(tmp_path).project_path(project_id)


def test_recipe_directory_symlink_cannot_write_outside_project(tmp_path: Path) -> None:
    plan, inputs = _recipe()
    store = FilmProjectStore(tmp_path / "store")
    project = store.project_path("project")
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "recipes").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        store.save("project", plan, inputs, expected_revision=None)
    assert list(outside.iterdir()) == []


def test_staging_recipe_saves_explicit_opening_and_rejects_blank(tmp_path: Path) -> None:
    plan, inputs = _recipe()
    plan = replace(
        plan, shots=(replace(plan.shots[0], composition="Continue frame"), *plan.shots[1:])
    )
    inputs["shots"][0].update(
        render_profile="Animate frame", opening_prompt="A wide empty landscape", opening_attempts=2
    )
    store = FilmProjectStore(tmp_path)
    saved = store.save("project", plan, inputs, expected_revision=None)
    assert saved["inputs"]["shots_by_id"]["one"]["opening_attempts"] == 2
    inputs["shots"][0]["opening_prompt"] = " "
    with pytest.raises(ValueError, match="opening composition"):
        store.save("project", plan, inputs, expected_revision=saved["revision"])
    assert store.load("project") == saved


def test_optional_narrative_preserves_legacy_recipe_identity_and_requires_whole_film_review(
    tmp_path: Path,
) -> None:
    plan, inputs = _recipe()
    store = FilmProjectStore(tmp_path)
    original = store.save("project", plan, inputs, expected_revision=None)
    legacy_plan = asdict(plan)
    for row in legacy_plan["shots"]:
        row.pop("direction_version")
    legacy_plan.pop("state_definitions")
    legacy_plan.pop("narrative")
    legacy = {"format": "duet-film-project-v1", "plan": legacy_plan, "inputs": original["inputs"]}
    assert original["revision"] == hashlib.sha256(canonical_story_json(legacy)).hexdigest()
    assert "narrative" not in original["plan"]
    intent = FilmNarrativeContract("Deliver", "Locked gate", "Side path", "Delivered")
    changed = store.save(
        "project", replace(plan, narrative=intent), inputs, expected_revision=original["revision"]
    )
    impact = store.impact("project", original["revision"], changed["revision"])
    assert impact["narrative_review_changed"] is True
    assert impact["reusable_prefix"] == ["one", "two", "three"]
    assert impact["requires_generation_review"] == []
    restored = store.save("project", plan, inputs, expected_revision=changed["revision"])
    assert restored == original
    assert (
        store.impact("project", changed["revision"], restored["revision"])[
            "narrative_review_changed"
        ]
        is True
    )
    assert (
        store.impact("project", original["revision"], restored["revision"])[
            "narrative_review_changed"
        ]
        is False
    )


def test_changing_opening_mode_invalidates_only_its_generation_suffix(tmp_path: Path) -> None:
    plan, inputs = _recipe()
    plan = replace(
        plan,
        shots=(plan.shots[0], replace(plan.shots[1], composition="Continue frame"), plan.shots[2]),
    )
    inputs["shots"][1].update(
        render_profile="Animate frame",
        opening_prompt="Blend an existing landscape",
        opening_mode="Compose",
    )
    store = FilmProjectStore(tmp_path)
    original = store.save("project", plan, inputs, expected_revision=None)
    inputs["shots"][1]["opening_mode"] = "Refine"
    revised = store.save("project", plan, inputs, expected_revision=original["revision"])
    assert store.load("project", original["revision"]) == original
    assert store.load("project")["inputs"]["shots_by_id"]["two"]["opening_mode"] == "Refine"
    impact = store.impact("project", original["revision"], revised["revision"])
    assert impact["reusable_prefix"] == ["one"]
    assert impact["requires_generation_review"] == ["two", "three"]
    assert revised["inputs"]["shots_by_id"]["two"]["variation"] == 22
