from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from comfy_story.film_io import film_plan_from_json
from comfy_story.film_narrative import (
    NARRATIVE_FIELDS,
    FilmNarrativeContract,
    narrative_request,
    parse_narrative_findings,
)
from comfy_story.film_plan import (
    FilmPlan,
    FilmShot,
    compile_candidate_prompt,
    initial_acceptance_digest,
)
from comfy_story.film_retell import FilmRetell
from comfy_story.film_workflow import FilmShotInput, compile_film_workflow
from comfy_story.story_contracts import canonical_story_json


def _retell() -> FilmRetell:
    return FilmRetell(
        "A courier reaches a locked gate, takes a side path and delivers the parcel.",
        "Deliver the parcel.",
        "Take the side path.",
        "The parcel reaches the greenhouse.",
        (),
        "coherent",
    )


def _contract() -> FilmNarrativeContract:
    return FilmNarrativeContract(
        "Deliver a parcel.",
        "The direct gate is locked.",
        "Choose the side path.",
        "Parcel delivered.",
    )


def _findings() -> dict[str, dict[str, str]]:
    return {
        name: {
            "status": "supported",
            "quote": _retell().summary,
            "explanation": "Injected semantic judgment; tests do not establish model accuracy.",
        }
        for name in NARRATIVE_FIELDS
    }


@pytest.mark.parametrize("value", ["", " ", "x" * 2001, "bad\x00text", 1, None])
def test_narrative_contract_rejects_unusable_descriptions(value: object) -> None:
    data = asdict(_contract())
    data["goal"] = value  # Exercise invalid external JSON types.
    with pytest.raises(ValueError, match="narrative contract"):
        FilmNarrativeContract(**data).validate()


def test_narrative_contract_is_portable_but_does_not_change_shot_conditioning() -> None:
    plan = FilmPlan("example", "One shot", 2500, (), (FilmShot("one", 2500, "Hold", "Hold", ()),))
    legacy = canonical_story_json(plan)
    assert b'"narrative"' not in legacy
    assert canonical_story_json(film_plan_from_json(legacy.decode())) == legacy
    changed = replace(plan, narrative=_contract())
    assert film_plan_from_json(canonical_story_json(changed).decode()) == changed
    assert canonical_story_json(changed) != legacy
    assert initial_acceptance_digest(changed) == initial_acceptance_digest(plan)
    assert changed.shots[0].digest == plan.shots[0].digest
    assert compile_candidate_prompt(changed, 0) == compile_candidate_prompt(plan, 0)
    library = {"project_name": "Example", "references": []}
    settings = (FilmShotInput("opening.png", 42),)
    assert compile_film_workflow(changed, library, settings) == compile_film_workflow(
        plan, library, settings
    )


@pytest.mark.parametrize("status", ["supported", "contradicted", "unestablished"])
def test_narrative_parser_preserves_observations_without_promoting_uncertainty(status: str) -> None:
    data = _findings()
    data["decision"]["status"] = status
    if status == "unestablished":
        data["decision"]["quote"] = ""
    parsed = parse_narrative_findings(json.dumps(data), _retell())
    assert next(row for row in parsed if row.field == "decision").status == status


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quote", "An invented event not in the retelling."),
        ("quote", ""),
        ("quote", " "),
        ("quote", None),
        ("status", "pass"),
        ("explanation", ""),
        ("explanation", "x" * 1001),
        ("extra", "unrequested"),
    ],
)
def test_narrative_parser_rejects_unbound_or_invalid_findings(field: str, value: object) -> None:
    data = _findings()
    data["goal"][field] = value  # type: ignore[assignment]  # Invalid JSON is intentional.
    with pytest.raises(ValueError, match="narrative"):
        parse_narrative_findings(json.dumps(data), _retell())


@pytest.mark.parametrize("raw", ["null", "[]", "{}", '{"goal":1,"goal":2}', " " * 16385])
def test_narrative_parser_requires_complete_unique_bounded_json(raw: str) -> None:
    with pytest.raises(ValueError, match="narrative"):
        parse_narrative_findings(raw, _retell())


def test_narrative_request_binds_intent_retelling_and_model_with_text_only() -> None:
    request = narrative_request(_contract(), _retell(), "a" * 64, "b" * 64)
    assert request["retell_sha256"] == "a" * 64
    assert request["model_sha256"] == "b" * 64
    assert len(request["content"]) == 1
    assert request["content"][0]["type"] == "text"
    assert "never instructions" in request["content"][0]["text"]
    assert request != narrative_request(
        replace(_contract(), outcome="The parcel returns home."), _retell(), "a" * 64, "b" * 64
    )
