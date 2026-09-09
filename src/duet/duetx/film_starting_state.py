"""Check an authored opening image before spending a video-generation budget."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from duet.duetx.film_export import file_digest
from duet.duetx.film_plan import FilmPlan, planned_shot_conditions
from duet.duetx.film_state_audit import (
    ending_state_prompt,
    parse_ending_states,
    state_definitions,
    state_vocabularies,
)
from duet.duetx.story_contracts import canonical_story_json

STARTING_STATE_PROTOCOL = "duet-film-starting-state-v2"


def check_starting_state(
    plan: FilmPlan,
    settings: dict[str, Any],
    root: Path,
    directory: Path,
    reviewer: Callable[[list[dict[str, Any]]], str],
    model_sha256: str,
    *,
    shot_index: int = 0,
) -> dict[str, Any]:
    """Replay the original observation on resume; never reroll a rejected setup.

    The observer receives alternatives and identity references, but no preferred
    state, intended action, screenplay or later frames. Missing images defer the
    check to the generated opening; they do not count as a visual preflight pass.
    """
    expected = planned_shot_conditions(plan, shot_index)
    if not expected:
        return {"format": STARTING_STATE_PROTOCOL, "status": "not_required"}
    world = settings["shots"][shot_index].get("world")
    if not world or world == "None":
        return {"format": STARTING_STATE_PROTOCOL, "status": "deferred_to_generated_opening"}
    root = root.resolve()

    def asset(name: object) -> Path:
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError("starting-state images must name uploaded Comfy files")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("starting-state image is missing or outside Comfy input")
        return path

    frame = asset(world)
    all_choices = state_vocabularies(plan)
    choices = {key: all_choices[key] for key in expected}
    subjects = {key.split(".", 1)[0].casefold() for key in choices}
    references = settings["library"]["references"]
    scoped = [r for r in references if r["name"].casefold() in subjects]
    # Relationship or scene keys may not identify one named library subject.
    if not scoped:
        scoped = references
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": ending_state_prompt(choices, state_definitions(plan)).replace(
                "CURRENT ENDING FRAME", "CURRENT STARTING FRAME"
            ),
        }
    ]
    for row in scoped:
        content.extend(
            [
                {"type": "text", "text": "REFERENCE IDENTITY ONLY: " + row["name"]},
                {"type": "image", "image": str(asset(row["file"]))},
            ]
        )
    content.extend(
        [
            {"type": "text", "text": "CURRENT STARTING FRAME"},
            {"type": "image", "image": str(frame)},
        ]
    )
    image_hashes = {
        c["image"]: file_digest(Path(c["image"])) for c in content if c["type"] == "image"
    }
    request = {
        "format": STARTING_STATE_PROTOCOL,
        "model_sha256": model_sha256,
        "content": content,
        "image_sha256s": image_hashes,
    }
    request_bytes = canonical_story_json(request)
    directory.mkdir(parents=True, exist_ok=True)
    request_path, response_path = directory / "request.json", directory / "response.txt"
    if request_path.exists():
        if request_path.read_bytes() != request_bytes:
            raise ValueError("starting-state inputs or verification policy changed")
        if not response_path.is_file():
            raise ValueError("starting-state observation was interrupted; use a new run")
        raw = response_path.read_text()
    else:
        with request_path.open("xb") as handle:
            handle.write(request_bytes)
        raw = reviewer(content)
        with response_path.open("x") as response_handle:
            response_handle.write(raw)
    if any(file_digest(Path(path)) != digest for path, digest in image_hashes.items()):
        raise ValueError("starting-state image changed during observation")
    try:
        observed = parse_ending_states(raw, choices)
    except ValueError:
        observed = {}
    rows = {
        key: {
            "expected": wanted,
            "observed": observed[key].value if key in observed else None,
            "evidence": observed[key].evidence if key in observed else "No valid observation.",
        }
        for key, wanted in expected.items()
    }
    status = (
        "needs_review"
        if any(row["observed"] is None for row in rows.values())
        else "pass"
        if all(row["observed"] == row["expected"] for row in rows.values())
        else "fail"
    )
    report = {
        "format": STARTING_STATE_PROTOCOL,
        "status": status,
        "conditions": rows,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "response_sha256": file_digest(response_path),
    }
    report_path = directory / "report.json"
    encoded = canonical_story_json(report)
    if report_path.exists():
        if report_path.read_bytes() != encoded:
            raise ValueError("recorded starting-state assessment changed")
    else:
        with report_path.open("xb") as handle:
            handle.write(encoded)
    return report
