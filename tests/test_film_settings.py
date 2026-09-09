"""Stable customer recipes across timeline edits; no renderer or model is involved."""

from __future__ import annotations

from dataclasses import replace

import pytest

from comfy_story.film_io import bind_film_settings, normalize_film_settings
from comfy_story.film_plan import FilmPlan, FilmShot


def _plan() -> FilmPlan:
    return FilmPlan(
        "edits",
        "Two scenes",
        10000,
        (),
        (
            FilmShot("arrival", 5000, "Arrive", "Enter", ()),
            FilmShot("departure", 5000, "Leave", "Exit", ()),
        ),
    ).validate()


def test_reordering_retains_each_shots_seed_world_and_sampler() -> None:
    plan = _plan()
    settings = {
        "shots": [
            {"world": "first.png", "variation": 17, "sampler": "Turbo 4-step"},
            {"world": "second.png", "variation": 29, "sampler": "Native res_multistep"},
        ]
    }
    bound = bind_film_settings(plan, settings)
    reordered = replace(plan, shots=tuple(reversed(plan.shots))).validate()
    resolved = normalize_film_settings(reordered, bound)
    assert resolved["shots"] == list(reversed(settings["shots"]))
    assert settings["shots"][0]["variation"] == 17
    assert bound["shots_by_id"]["arrival"]["variation"] == 17
    resolved["shots"][1]["variation"] = 18
    assert bound["shots_by_id"]["arrival"]["variation"] == 17


def test_inserting_a_shot_requires_its_own_settings_without_reseeding_existing_shots() -> None:
    plan = _plan()
    bound = bind_film_settings(plan, {"shots": [{"variation": 17}, {"variation": 29}]})
    inserted = replace(
        plan,
        target_duration_ms=15000,
        shots=(plan.shots[0], FilmShot("pause", 5000, "Pause", "Wait", ()), plan.shots[1]),
    )
    with pytest.raises(ValueError, match="every planned shot ID"):
        normalize_film_settings(inserted, bound)
    bound["shots_by_id"]["pause"] = {"variation": 41}
    assert [row["variation"] for row in normalize_film_settings(inserted, bound)["shots"]] == [
        17,
        41,
        29,
    ]


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"shots": [], "shots_by_id": {}},
        {"shots_by_id": []},
        {"shots_by_id": {"arrival": {}, "wrong": {}}},
        {"shots_by_id": {"arrival": {}, "departure": None}},
        {"shots": [{}, {}, {}]},
        {"shots": [None, {}]},
    ],
)
def test_ambiguous_or_stale_bindings_fail_before_render(settings: object) -> None:
    with pytest.raises(ValueError, match=r"film inputs|shots_by_id"):
        normalize_film_settings(_plan(), settings)


def test_binding_preserves_full_python_seed_precision_and_is_idempotent() -> None:
    settings = {
        "shots": [{"variation": 2**64 - 1}, {"variation": 2**53 + 1}],
        "burn_subtitles": False,
    }
    bound = bind_film_settings(_plan(), settings)
    assert bind_film_settings(_plan(), bound) == bound
    assert normalize_film_settings(_plan(), bound) == settings
