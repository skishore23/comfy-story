"""Load the portable film sidecar without executing authoring code."""

from __future__ import annotations

import json
from typing import Any

from duet.duetx.film_narrative import FilmNarrativeContract
from duet.duetx.film_plan import (
    FilmCount,
    FilmCue,
    FilmFact,
    FilmPlan,
    FilmShot,
    FilmStateDefinition,
)


def film_plan_from_json(encoded: str) -> FilmPlan:
    """Read dataclass-shaped JSON, reject unknown fields, then validate the timeline."""
    if not encoded or len(encoded.encode("utf-8")) > 1024 * 1024:
        raise ValueError("film plan must be bounded nonempty JSON")
    try:
        data: dict[str, Any] = json.loads(encoded)
        shots = []
        for raw in data.pop("shots"):
            row = dict(raw)
            row["ending_counts"] = tuple(
                FilmCount(**value) for value in row.get("ending_counts", [])
            )
            for key in ("requires", "effects"):
                row[key] = tuple(FilmFact(**value) for value in row.get(key, []))
            for key in ("dialogue", "text"):
                row[key] = tuple(FilmCue(**value) for value in row.get(key, []))
            for key in (
                "present",
                "absent",
                "depends_on",
                "visible_throughout",
                "fully_visible_throughout",
            ):
                values = row.get(key, [])
                if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
                    raise ValueError("film rosters and dependencies must be string lists")
                row[key] = tuple(values)
            if row.get("reference_names") is not None:
                values = row["reference_names"]
                if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
                    raise ValueError("film reference names must be a string list")
                row["reference_names"] = tuple(values)
            shots.append(FilmShot(**row))
        data["initial_facts"] = tuple(FilmFact(**value) for value in data["initial_facts"])
        languages = data.get("languages", ["en"])
        if not isinstance(languages, list) or any(not isinstance(x, str) for x in languages):
            raise ValueError("film languages must be a string list")
        data["languages"] = tuple(languages)
        data["state_definitions"] = tuple(
            FilmStateDefinition(**row) for row in data.get("state_definitions", [])
        )
        if data.get("narrative") is not None:
            data["narrative"] = FilmNarrativeContract(**data["narrative"])
        return FilmPlan(shots=tuple(shots), **data).validate()
    except (TypeError, KeyError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError("malformed film plan JSON") from error


def normalize_film_settings(plan: FilmPlan, settings: object) -> dict[str, Any]:
    """Resolve stable shot IDs to timeline order without changing seeds or caller-owned data.

    Legacy positional inputs remain readable. New editors should persist shots_by_id so an
    insertion/reorder cannot silently move generation settings to a different shot.
    """
    plan.validate()
    if not isinstance(settings, dict) or ("shots" in settings) == ("shots_by_id" in settings):
        raise ValueError("film inputs require exactly one of shots or shots_by_id")
    if "shots_by_id" in settings:
        keyed = settings["shots_by_id"]
        if not isinstance(keyed, dict) or set(keyed) != {shot.shot_id for shot in plan.shots}:
            raise ValueError("shots_by_id must match every planned shot ID exactly")
        rows = [keyed[shot.shot_id] for shot in plan.shots]
    else:
        rows = settings["shots"]
    if (
        not isinstance(rows, list)
        or len(rows) != len(plan.shots)
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("film inputs require one settings object for every shot")
    result = {key: value for key, value in settings.items() if key != "shots_by_id"}
    result["shots"] = [dict(row) for row in rows]
    return result


def bind_film_settings(plan: FilmPlan, settings: object) -> dict[str, Any]:
    """Bind an existing recipe before editing its plan; never derive new random seeds."""
    result = normalize_film_settings(plan, settings)
    rows = result.pop("shots")
    result["shots_by_id"] = {shot.shot_id: row for shot, row in zip(plan.shots, rows, strict=True)}
    return result
