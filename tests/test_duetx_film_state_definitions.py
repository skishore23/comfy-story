from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from duet.duetx.film_io import film_plan_from_json
from duet.duetx.film_plan import (
    AcceptedTake,
    FilmFact,
    FilmPlan,
    FilmShot,
    FilmStateDefinition,
    accepted_state,
    compile_candidate_prompt,
    initial_acceptance_digest,
)
from duet.duetx.film_project import FilmProjectStore, edit_impact
from duet.duetx.film_state_audit import (
    ending_state_prompt,
    state_definitions,
    state_vocabularies,
)
from duet.duetx.story_contracts import canonical_story_json


def _plan() -> FilmPlan:
    return FilmPlan(
        "meanings",
        "A door",
        10000,
        (FilmFact("Door.state", "closed"),),
        (
            FilmShot("setup", 5000, "Establish", "The door remains closed.", ("Door",)),
            FilmShot(
                "open",
                5000,
                "Entry",
                "The door swings open.",
                ("Door",),
                requires=(FilmFact("Door.state", "closed"),),
                effects=(FilmFact("Door.state", "open"),),
                depends_on=("setup",),
            ),
        ),
    ).validate()


def _defined() -> FilmPlan:
    return replace(
        _plan(),
        state_definitions=(
            FilmStateDefinition("Door.state", "closed", "The door panel fills the doorway."),
            FilmStateDefinition(
                "Door.state", "open", "The panel is turned away and the doorway is clear."
            ),
        ),
    ).validate()


def test_empty_definitions_preserve_legacy_wire_and_acceptance_root() -> None:
    plan = _plan()
    encoded = canonical_story_json(plan)
    assert "state_definitions" not in json.loads(encoded)
    assert canonical_story_json(film_plan_from_json(encoded.decode())) == encoded
    assert (
        initial_acceptance_digest(plan)
        == hashlib.sha256(canonical_story_json(plan.initial_facts)).hexdigest()
    )
    with_empty = json.loads(encoded)
    with_empty["state_definitions"] = []
    assert canonical_story_json(film_plan_from_json(json.dumps(with_empty))) == encoded


def test_state_meanings_round_trip_and_include_unplanned_alternatives() -> None:
    plan = replace(
        _defined(),
        state_definitions=(
            *_defined().state_definitions,
            FilmStateDefinition("Door.state", "ajar", "A narrow gap is visible beside the panel."),
        ),
    )
    assert film_plan_from_json(canonical_story_json(plan).decode()) == plan
    assert state_vocabularies(plan)["Door.state"] == ["ajar", "closed", "open"]


@pytest.mark.parametrize(
    "definitions",
    [
        (FilmStateDefinition("Door.state", "closed", "A closed panel."),),
        (*_defined().state_definitions, _defined().state_definitions[0]),
        (FilmStateDefinition("Other.state", "visible", "@Unexpected is visible."),),
        (FilmStateDefinition("Other.state", "visible", "x" + " " * 2000),),
        (FilmStateDefinition("Other.state", "visible", ""),),
    ],
)
def test_invalid_meanings_fail_before_rendering(
    definitions: tuple[FilmStateDefinition, ...],
) -> None:
    with pytest.raises(ValueError, match=r"state definition|state key"):
        replace(_plan(), state_definitions=definitions).validate()


def test_rendering_scopes_meanings_to_relevant_starting_and_ending_states() -> None:
    plan = replace(
        _defined(),
        state_definitions=(
            *_defined().state_definitions,
            FilmStateDefinition("Other.state", "visible", "A distant unrelated object is visible."),
        ),
    )
    opening = compile_candidate_prompt(plan, 0)
    assert "The door panel fills the doorway." in opening
    assert "the doorway is clear." not in opening
    assert "distant unrelated" not in opening
    transition = compile_candidate_prompt(plan, 1)
    assert "The door panel fills the doorway." in transition
    assert "the doorway is clear." in transition


def test_independent_observer_receives_every_meaning_without_the_requested_action() -> None:
    plan = _defined()
    prompt = ending_state_prompt(state_vocabularies(plan), state_definitions(plan))
    assert "The door panel fills the doorway." in prompt
    assert "the doorway is clear." in prompt
    assert plan.shots[1].action not in prompt
    assert "no alternative is preferred" in prompt
    prefix = replace(plan, shots=plan.shots[:1], target_duration_ms=5000).validate()
    assert state_vocabularies(prefix) == state_vocabularies(plan)
    assert state_definitions(prefix) == state_definitions(plan)


def test_meaning_edit_invalidates_only_changed_prompt_and_descendants() -> None:
    plan = _defined()
    changed = replace(
        plan,
        state_definitions=(
            plan.state_definitions[0],
            replace(plan.state_definitions[1], definition="The doorway is entirely unobstructed."),
        ),
    )
    inputs = {
        "library": {"project_name": "Test", "references": []},
        "shots": [{"world": "door.png", "variation": 1}, {"world": "door.png", "variation": 2}],
    }
    impact = edit_impact(plan, inputs, changed, inputs)
    assert impact["reusable_prefix"] == ["setup"]
    assert impact["requires_generation_review"] == ["open"]


def test_old_acceptance_cannot_establish_a_newly_defined_state() -> None:
    old = _plan()
    take = AcceptedTake(
        "setup",
        old.shots[0].digest,
        "a" * 64,
        "b" * 64,
        initial_acceptance_digest(old),
        (),
        "reviewer",
        "Panel is visible.",
        5000,
    )
    assert accepted_state(old, (take,)) == {"Door.state": "closed"}
    with pytest.raises(ValueError, match="stale parent"):
        accepted_state(_defined(), (take,))
    rebound = replace(take, parent_acceptance_sha256=initial_acceptance_digest(_defined()))
    assert accepted_state(_defined(), (rebound,)) == {"Door.state": "closed"}


def test_empty_meanings_do_not_change_an_existing_recipe_on_save(tmp_path: Path) -> None:
    plan = _plan()
    # No roster is needed to exercise immutable recipe serialization.
    plan = replace(plan, shots=tuple(replace(s, present=()) for s in plan.shots))
    inputs = {
        "library": {"project_name": "Test", "references": []},
        "shots": [{"world": "door.png", "variation": 1}, {"world": "door.png", "variation": 2}],
    }
    store = FilmProjectStore(tmp_path)
    first = store.save(plan.project_id, plan, inputs, expected_revision=None)
    assert "state_definitions" not in first["plan"]
    restored = film_plan_from_json(json.dumps({**first["plan"], "state_definitions": []}))
    second = store.save(
        plan.project_id, restored, first["inputs"], expected_revision=first["revision"]
    )
    assert first == second
