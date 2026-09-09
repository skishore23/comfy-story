from __future__ import annotations

import json

import pytest

from duet.duetx.film_audit import ShotVisualAudit
from duet.duetx.film_plan import FilmFact, FilmPlan, FilmShot
from duet.duetx.film_state_audit import (
    StateObservation,
    apply_ending_states,
    ending_conditions,
    ending_state_prompt,
    parse_ending_states,
    state_vocabularies,
)


def test_state_alternatives_include_future_changes_without_exposing_the_wanted_ending() -> None:
    plan = FilmPlan(
        "p",
        "A delivery",
        10000,
        (FilmFact("parcel.place", "shelf"),),
        (
            FilmShot("one", 5000, "Wait", "The parcel remains still", ("parcel",)),
            FilmShot(
                "two",
                5000,
                "Deliver",
                "Move the parcel",
                ("parcel",),
                effects=(FilmFact("parcel.place", "desk"),),
            ),
        ),
    )
    vocabulary = state_vocabularies(plan)
    assert vocabulary == {"parcel.place": ["desk", "shelf"]}
    assert ending_conditions(plan, 0) == {"parcel.place": "shelf"}
    assert ending_conditions(plan, 1) == {"parcel.place": "desk"}
    prompt = ending_state_prompt(vocabulary)
    assert "Move the parcel" not in prompt
    assert "A delivery" not in prompt
    assert "no value is preferred" in prompt
    assert "use null rather than choose a side" in prompt
    assert "A still frame cannot establish" in prompt
    assert json.loads(prompt.splitlines()[1])["state_vocabularies"] == vocabulary


@pytest.mark.parametrize(
    ("value", "result"), [("shelf", "machine_pass"), ("desk", "fail"), (None, "needs_review")]
)
def test_ending_observation_overrides_interval_pass_without_rewriting_it(
    value: str | None, result: str
) -> None:
    audit = ShotVisualAudit(
        "one", "a" * 64, "b" * 64, "c" * 64, (), (), "machine_pass", (("parcel.place", "shelf"),)
    )
    observed = parse_ending_states(
        json.dumps({"parcel.place": {"value": value, "evidence": "Visible surface contact."}}),
        {"parcel.place": ["desk", "shelf"]},
    )
    checked = apply_ending_states(audit, observed, {"parcel.place": "shelf"}, 239)
    assert checked.status == result
    assert checked.observed_conditions == audit.observed_conditions
    assert checked.findings[-1].frame_indices == (239,)


@pytest.mark.parametrize("status", ["fail", "needs_review"])
def test_ending_pass_cannot_clear_prior_failure_or_uncertainty(status: str) -> None:
    audit = ShotVisualAudit("one", "a" * 64, "b" * 64, "c" * 64, (), (), status)
    assert (
        apply_ending_states(
            audit, {"lid": StateObservation("open", "Visible inside.")}, {"lid": "open"}, 119
        ).status
        == status
    )


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"value": "invented", "evidence": "Visible"},
        {"value": 1, "evidence": "Visible"},
        {"value": None, "evidence": ""},
        {"value": "open", "evidence": "Visible", "score": 1},
    ],
)
def test_invalid_observations_are_not_inferred_as_a_pass(row: object) -> None:
    with pytest.raises(ValueError, match="ending state"):
        parse_ending_states(json.dumps({"lid": row}), {"lid": ["closed", "open"]})


def test_wrong_keys_and_oversized_responses_are_rejected() -> None:
    for response in ("{}", '{"wrong":{"value":null,"evidence":"Unknown"}}', "x" * 65537):
        with pytest.raises(ValueError, match="ending state"):
            parse_ending_states(response, {"lid": ["closed", "open"]})


def test_duplicate_observations_cannot_overwrite_a_contradiction() -> None:
    response = (
        '{"lid":{"value":"closed","evidence":"Closed lid"},'
        '"lid":{"value":"open","evidence":"Open lid"}}'
    )
    with pytest.raises(ValueError, match="duplicate fields"):
        parse_ending_states(response, {"lid": ["closed", "open"]})
