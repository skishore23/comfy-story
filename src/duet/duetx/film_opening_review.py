"""Read preserved opening failures for the customer workspace; never change selections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from duet.duetx.film_export import file_digest
from duet.duetx.story_contracts import canonical_story_json


def starting_state_review(
    directory: Path, input_root: Path, production: dict[str, Any]
) -> dict[str, Any] | None:
    """Read an uploaded opening's failed observation without rerunning its checker.

    The production digest binds the report, which binds the original request, response
    and every observed image. Preview the frame actually assessed, including any saved
    H3 crop, rather than assuming the recipe's original upload was checked.
    """
    report = production.get("starting_state")
    if not isinstance(report, dict) or report.get("status") not in {"fail", "needs_review"}:
        return None
    directory, input_root = directory.resolve(), input_root.resolve()
    row: dict[str, Any] = {"kind": "uploaded", "approval_created": False}

    def recorded(name: str) -> bytes:
        path = directory / "production" / "starting-state" / name
        if not path.resolve().is_relative_to(directory) or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("starting-state record is outside its run or exceeds its boundary")
        return path.read_bytes()

    try:
        encoded = recorded("report.json")
        if hashlib.sha256(encoded).hexdigest() != production[
            "starting_state_sha256"
        ] or encoded != canonical_story_json(report):
            raise ValueError("starting-state assessment changed")
        request_bytes, response = recorded("request.json"), recorded("response.txt")
        if (
            hashlib.sha256(request_bytes).hexdigest() != report["request_sha256"]
            or hashlib.sha256(response).hexdigest() != report["response_sha256"]
        ):
            raise ValueError("starting-state request or response changed")
        request = json.loads(request_bytes)
        content, hashes = request["content"], request["image_sha256s"]
        if request["format"] != report["format"] or content[-2] != {
            "type": "text",
            "text": "CURRENT STARTING FRAME",
        }:
            raise ValueError("starting-state frame is not identified")
        if content[-1]["type"] != "image":
            raise ValueError("starting-state frame is missing")
        images = {item["image"] for item in content if item["type"] == "image"}
        if images != set(hashes):
            raise ValueError("starting-state image bindings changed")
        for name in images:
            path = Path(name).resolve(strict=True)
            if not path.is_relative_to(input_root) or file_digest(path) != hashes[name]:
                raise ValueError("starting-state image is missing or changed")
        conditions = report["conditions"]
        if not isinstance(conditions, dict) or any(
            not isinstance(fact, dict)
            or not isinstance(fact.get("expected"), str)
            or not isinstance(fact.get("evidence"), str)
            or (fact.get("observed") is not None and not isinstance(fact["observed"], str))
            for fact in conditions.values()
        ):
            raise ValueError("starting-state conditions are malformed")
        row.update(
            image=Path(content[-1]["image"]).resolve().relative_to(input_root).as_posix(),
            state=report,
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        row["unavailable"] = "Opening evidence is incomplete, missing or changed."
    return row


def opening_candidates(
    directory: Path, input_root: Path, production: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expose the current shot's bounded original evidence after production stops.

    Missing or changed evidence remains unavailable rather than becoming a new observation.
    Legacy staged images retain their original-view label; they were not checked after fitting.
    """
    directory = directory.resolve()
    input_root = input_root.resolve()

    def recorded(path: Path) -> bytes:
        if not path.resolve().is_relative_to(directory) or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("opening record is outside the run or exceeds its boundary")
        return path.read_bytes()

    def image_file(record: dict[str, Any], field: str, digest_field: str) -> str:
        name, digest = record[field], record[digest_field]
        if not isinstance(name, str) or not isinstance(digest, str) or Path(name).is_absolute():
            raise ValueError("opening image receipt is incomplete")
        path = (input_root / name).resolve()
        if not path.is_relative_to(input_root) or file_digest(path) != digest:
            raise ValueError("opening image is missing or changed")
        return path.relative_to(input_root).as_posix()

    try:
        plan = json.loads(recorded(directory / "plan.json"))
        index = next(
            i
            for i, shot in enumerate(plan["shots"])
            if shot["shot_id"] == production.get("shot_id")
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return []
    rows = []
    for attempt in range(1, 5):
        stage = (
            directory / "production" / f"shot-{index + 1:03}" / "opening" / f"attempt-{attempt:02}"
        )
        if not (stage / "result.json").exists():
            continue
        row: dict[str, Any] = {"attempt": attempt, "approval_created": False}
        try:
            result = json.loads(recorded(stage / "result.json"))
            request_bytes = recorded(stage / "request.json")
            if (
                hashlib.sha256(request_bytes).hexdigest() != result["request_sha256"]
                or hashlib.sha256(recorded(stage / "history.json")).hexdigest()
                != result["history_sha256"]
            ):
                raise ValueError("opening request or history changed")
            image = image_file(result, "file", "sha256")
            raw = image_file(result, "raw_file", "raw_sha256") if "raw_file" in result else None
            assessment = json.loads(recorded(stage / "assessment.json"))
            row.update(
                image=image,
                raw_image=raw,
                fitted=json.loads(request_bytes).get("format")
                in ("duet-film-opening-stage-v2", "duet-film-opening-stage-v3"),
                state=assessment["state"],
                visibility=assessment["visibility"],
            )
        except (OSError, ValueError, KeyError, TypeError):
            row["unavailable"] = "Opening evidence is incomplete, missing or changed."
        rows.append(row)
    return rows
