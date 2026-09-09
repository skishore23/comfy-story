from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from comfy_story.film_io import film_plan_from_json
from comfy_story.film_plan import (
    AcceptedTake,
    FilmCue,
    FilmFact,
    FilmPlan,
    FilmShot,
    accepted_state,
    compile_candidate_prompt,
    compile_shot_prompt,
    replace_shot,
    subtitle_srt,
)
from comfy_story.story_contracts import canonical_story_json


def _plan() -> FilmPlan:
    return FilmPlan(
        "film-test",
        "The key",
        10_000,
        (FilmFact("key.owner", "Sol"),),
        (
            FilmShot(
                "handoff",
                5_000,
                "Give Mira the means to enter",
                "Sol gives Mira the key",
                ("Mira", "Sol"),
                requires=(FilmFact("key.owner", "Sol"),),
                effects=(FilmFact("key.owner", "Mira"),),
                dialogue=(FilmCue("line1", 100, 2000, "Take the key.", "en", "Sol"),),
            ),
            FilmShot(
                "unlock",
                5_000,
                "Resolve the locked door",
                "Mira unlocks the door",
                ("Mira",),
                absent=("Sol",),
                requires=(FilmFact("key.owner", "Mira"),),
                effects=(FilmFact("door.state", "open"),),
                depends_on=("handoff",),
                dialogue=(FilmCue("line2", 500, 3000, "It worked.", "en", "Mira"),),
            ),
        ),
    ).validate()


def _take(plan: FilmPlan) -> AcceptedTake:
    return AcceptedTake(
        "handoff",
        plan.shots[0].digest,
        "a" * 64,
        "b" * 64,
        hashlib.sha256(canonical_story_json(plan.initial_facts)).hexdigest(),
        (FilmFact("key.owner", "Mira"),),
        "reviewer",
        "Key visible in Mira's hand at 3.2s",
        5_200,
    )


def test_generation_intent_does_not_advance_accepted_state() -> None:
    plan = _plan()
    assert accepted_state(plan, ()) == {"key.owner": "Sol"}
    with pytest.raises(ValueError, match="accepted prefix"):
        compile_shot_prompt(plan, 1, ())
    state = accepted_state(plan, (_take(plan),))
    assert state["key.owner"] == "Mira"
    prompt = compile_shot_prompt(plan, 1, (_take(plan),))
    assert "@Mira" in prompt
    assert "@Sol" not in prompt


def test_out_of_order_candidate_does_not_bypass_take_acceptance() -> None:
    plan = _plan()
    assert "unlocks the door" in compile_candidate_prompt(plan, 1)
    assert accepted_state(plan, ()) == {"key.owner": "Sol"}
    with pytest.raises(ValueError, match="accepted prefix"):
        compile_shot_prompt(plan, 1, ())


def test_candidate_carries_location_until_the_planned_transition() -> None:
    plan = FilmPlan(
        "arrival",
        "The invitation",
        15_000,
        (FilmFact("Leon.place", "outside"), FilmFact("Ada.place", "kitchen")),
        (
            FilmShot("wait", 5000, "Hesitation", "Leon waits", ("Leon",)),
            FilmShot(
                "enter",
                5000,
                "Accept invitation",
                "Leon enters",
                ("Leon",),
                requires=(FilmFact("Leon.place", "outside"),),
                effects=(FilmFact("Leon.place", "inside"),),
            ),
            FilmShot("sit", 5000, "Reunion", "Leon sits", ("Leon",)),
        ),
    ).validate()
    waiting = compile_candidate_prompt(plan, 0)
    assert "Planned, unverified continuity to preserve: Leon.place = outside." in waiting
    assert "Ada" not in waiting
    entering = compile_candidate_prompt(plan, 1)
    assert "starting condition: Leon.place = outside." in entering
    assert "continuity to preserve: Leon.place" not in entering
    assert (
        "Required outcome by the end of this shot (rendering goal): Leon.place = inside."
        in entering
    )
    seated = compile_candidate_prompt(plan, 2)
    assert "continuity to preserve: Leon.place = inside." in seated
    assert "outside" not in seated
    assert accepted_state(plan, ())["Leon.place"] == "outside"


def test_reviewed_compiler_carries_observed_state_and_scoped_prop_facts() -> None:
    base = _plan()
    plan = replace(
        base,
        initial_facts=(*base.initial_facts, FilmFact("key.color", "brass")),
        shots=(
            replace(
                base.shots[0], effects=(*base.shots[0].effects, FilmFact("Mira.place", "porch"))
            ),
            base.shots[1],
        ),
    ).validate()
    take = replace(_take(plan), observed_facts=plan.shots[0].effects)
    prompt = compile_shot_prompt(plan, 1, (take,))
    assert "Reviewed continuity to preserve: Mira.place = porch." in prompt
    assert "Reviewed continuity to preserve: key.color = brass." in prompt
    assert "Required outcome by the end of this shot (rendering goal): door.state = open." in prompt
    candidate = compile_candidate_prompt(plan, 1)
    assert "Planned, unverified continuity to preserve: Mira.place = porch." in candidate
    assert "Reviewed" not in candidate


def test_effect_without_explicit_start_does_not_lock_obsolete_state() -> None:
    plan = _plan()
    changed = replace_shot(plan, 0, replace(plan.shots[0], requires=()))
    prompt = compile_candidate_prompt(changed, 0)
    assert "key.owner = Sol" not in prompt
    assert "key.owner = Mira" in prompt


def test_failed_handoff_cannot_be_accepted() -> None:
    plan = _plan()
    take = replace(_take(plan), observed_facts=(FilmFact("key.owner", "Sol"),))
    with pytest.raises(ValueError, match="intended effects"):
        accepted_state(plan, (take,))


def test_edit_and_branch_change_require_new_review() -> None:
    plan = _plan()
    take = _take(plan)
    revised = replace_shot(plan, 0, replace(plan.shots[0], action="Sol tosses Mira the key"))
    with pytest.raises(ValueError, match="contract is stale"):
        accepted_state(revised, (take,))
    with pytest.raises(ValueError, match="stale parent"):
        accepted_state(plan, (replace(take, parent_acceptance_sha256="c" * 64),))


def test_replacing_upstream_take_invalidates_downstream_acceptance() -> None:
    plan = _plan()
    first = _take(plan)
    second = replace(
        first,
        shot_id="unlock",
        shot_sha256=plan.shots[1].digest,
        parent_acceptance_sha256=first.digest,
        observed_facts=plan.shots[1].effects,
    )
    assert accepted_state(plan, (first, second))["door.state"] == "open"
    with pytest.raises(ValueError, match="stale parent"):
        accepted_state(plan, (replace(first, video_sha256="d" * 64), second))


def test_plan_rejects_causal_conflict_and_wrong_runtime() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match=r"requires key\.owner"):
        replace(plan, initial_facts=(FilmFact("key.owner", "Nobody"),)).validate()
    with pytest.raises(ValueError, match="exact film duration"):
        replace(plan, target_duration_ms=180_000).validate()
    with pytest.raises(ValueError, match="too short"):
        accepted_state(plan, (replace(_take(plan), source_duration_ms=4_900),))


def test_exact_language_and_timing_are_preserved() -> None:
    plan = _plan()
    srt = subtitle_srt(plan)
    assert "00:00:00,100 --> 00:00:02,000\nTake the key." in srt
    assert "00:00:05,500 --> 00:00:08,000\nIt worked." in srt
    shot = replace(plan.shots[0], dialogue=(FilmCue("line1", 0, 2000, "Bonjour.", "fr", "Sol"),))
    with pytest.raises(ValueError, match="unapproved language"):
        replace_shot(plan, 0, shot)
    with pytest.raises(ValueError, match="fit inside"):
        FilmCue("bad", 0, 6000, "Hello", "en").validate(5000)


def test_unknown_and_conflicting_rosters_fail_preflight() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="absence"):
        replace_shot(plan, 0, replace(plan.shots[0], absent=("Mira",)))
    with pytest.raises(ValueError, match="precede"):
        replace_shot(plan, 0, replace(plan.shots[0], depends_on=("unlock",)))


def test_visible_roster_is_separate_from_active_image_references() -> None:
    plan = _plan()
    shot = replace(plan.shots[0], action="@Sol gives @Mira the key.", reference_names=("Mira",))
    scoped = replace_shot(plan, 0, shot)
    prompt = compile_shot_prompt(scoped, 0, ())
    assert "Sol gives Mira the key" in prompt
    assert "Visible entities: Mira, Sol" in prompt
    assert "Active visual references: @Mira" in prompt
    assert "@Sol" not in prompt
    assert shot.digest != plan.shots[0].digest
    for references in [("Unknown",), ("Mira", "Mira")]:
        with pytest.raises(ValueError, match="unique subset"):
            replace_shot(plan, 0, replace(shot, reference_names=references))
    without = replace_shot(plan, 0, replace(shot, reference_names=()))
    assert "@" not in compile_shot_prompt(without, 0, ())


def test_unscoped_shot_retains_its_historical_acceptance_digest() -> None:
    shot = FilmShot("one", 1000, "Setup", "Wait", ("Mira",))
    assert shot.digest == "ce081b5b9a27b5e90052922d48e00cdff699ac903c8ba701477081639096d040"


@pytest.mark.parametrize(
    "roster",
    [("Robot_17", "Drone_92", "battery", "hatch"), ("Otter_8", "Raven_6", "shell", "nest")],
)
def test_story_compilation_is_invariant_under_unfamiliar_entity_and_fact_names(
    roster: tuple[str, str, str, str],
) -> None:
    plan = _plan()
    mapping = dict(zip(("Mira", "Sol", "key", "door"), roster, strict=True))

    def rename(text: str) -> str:
        for before, after in mapping.items():
            text = text.replace(before, after)
        return text

    # Transform string values only; keep the public document's schema keys unchanged.
    def rename_values(value: object) -> object:
        if isinstance(value, str):
            return rename(value)
        if isinstance(value, dict):
            return {key: rename_values(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [rename_values(item) for item in value]
        return value

    renamed = film_plan_from_json(json.dumps(rename_values(asdict(plan))))
    for index in range(len(plan.shots)):
        assert compile_candidate_prompt(renamed, index) == rename(
            compile_candidate_prompt(plan, index)
        )
    assert subtitle_srt(renamed) == rename(subtitle_srt(plan))


def test_continuous_visibility_is_explicit_portable_and_changes_take_identity() -> None:
    plan = _plan()
    original = plan.shots[0]
    legacy = asdict(original)
    legacy.pop("reference_names")
    legacy.pop("visible_throughout")
    legacy.pop("camera_policy")
    legacy.pop("direction_version")
    legacy.pop("border_policy")
    legacy.pop("fully_visible_throughout")
    legacy.pop("ending_counts")
    assert original.digest == hashlib.sha256(canonical_story_json(legacy)).hexdigest()
    edited = replace_shot(plan, 0, replace(original, visible_throughout=("Mira", "Sol")))
    assert edited.shots[0].digest != original.digest
    assert film_plan_from_json(json.dumps(asdict(edited))) == edited
    prompt = compile_candidate_prompt(edited, 0)
    assert "Keep these entities visible throughout, including the final frame: Mira, Sol" in prompt
    assert "visible throughout" not in compile_candidate_prompt(plan, 0)


@pytest.mark.parametrize("names", [("Mira", "Mira"), ("Unknown",), ("Mira", 3), "Mira"])
def test_continuous_visibility_rejects_conflicting_or_invalid_rosters(names: object) -> None:
    payload = asdict(_plan())
    payload["shots"][0]["visible_throughout"] = names
    with pytest.raises(ValueError, match=r"visibility|string lists"):
        film_plan_from_json(json.dumps(payload))


def test_camera_policy_is_explicit_and_preserves_legacy_shot_identity() -> None:
    plan = _plan()
    shot = plan.shots[0]
    legacy = asdict(shot)
    legacy.pop("camera_policy")
    legacy.pop("direction_version")
    legacy.pop("border_policy")
    legacy.pop("fully_visible_throughout")
    legacy.pop("ending_counts")
    if not shot.visible_throughout:
        legacy.pop("visible_throughout")
    if shot.reference_names is None:
        legacy.pop("reference_names")
    assert shot.digest == hashlib.sha256(canonical_story_json(legacy)).hexdigest()
    legacy_plan = asdict(plan)
    legacy_plan.pop("state_definitions")
    legacy_plan.pop("narrative")  # Historical plans predate the optional intent contract.
    for row in legacy_plan["shots"]:
        row.pop("camera_policy")
        row.pop("direction_version")
        row.pop("border_policy")
        row.pop("fully_visible_throughout")
        row.pop("ending_counts")
    assert canonical_story_json(plan) == canonical_story_json(legacy_plan)
    locked = replace(shot, camera_policy="Locked frame")
    changed = replace(plan, shots=(locked, *plan.shots[1:])).validate()
    assert locked.digest != shot.digest
    assert "Camera: fixed position" in compile_candidate_prompt(changed, 0)
    assert "Camera: fixed position" not in compile_candidate_prompt(plan, 0)
    with pytest.raises(ValueError, match="camera policy"):
        replace(plan, shots=(replace(shot, camera_policy="invented"), *plan.shots[1:])).validate()


def test_full_visibility_is_optional_roundtrips_and_invalidates_affected_take() -> None:
    plan = _plan()
    shot = plan.shots[0]
    changed = replace_shot(plan, 0, replace(shot, fully_visible_throughout=("Mira",)))
    assert changed.shots[0].digest != shot.digest
    assert changed.shots[1].digest == plan.shots[1].digest
    assert film_plan_from_json(json.dumps(asdict(changed))) == changed
    assert "fully in frame and unobscured" in compile_candidate_prompt(changed, 0)
    assert "fully_visible_throughout" not in json.loads(canonical_story_json(plan))["shots"][0]
    assert "fully in frame and unobscured" not in compile_candidate_prompt(plan, 0)


@pytest.mark.parametrize("names", [("Mira", "Mira"), ("Unknown",), ("Mira", 3), "Mira"])
def test_full_visibility_requires_a_valid_present_roster(names: object) -> None:
    payload = asdict(_plan())
    payload["shots"][0]["fully_visible_throughout"] = names
    with pytest.raises(ValueError, match=r"visibility|string lists"):
        film_plan_from_json(json.dumps(payload))
