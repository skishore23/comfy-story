"""Bounded observations of visible edit artifacts, not proof of unsampled continuity."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from itertools import pairwise

from duet.duetx.film_audit import ShotVisualAudit, VisualFinding

EDIT_ARTIFACT_PROMPT = (
    "Inspect FRAME A and FRAME B for visible editing artifacts only. They are sampled video "
    "frames, not a storyboard or reference images. Look for a dissolve or crossfade that "
    "superimposes different framings at once, ghosted duplicate scene layers, or a wipe boundary. "
    "Natural occlusion, shadows, a moving subject, camera tracking, perspective change and "
    "ordinary motion blur are not by themselves edit artifacts. Do not infer a cut merely from "
    "different camera framing across these two samples; intervening motion is not shown. "
    "Report visible_edit_artifact only with direct evidence in a supplied frame. Report "
    "no_visible_edit_artifact if neither frame visibly contains such an artifact; that label "
    "does not establish uninterrupted motion between samples. Use uncertain for ambiguous "
    "artifacts. Return only JSON with exactly status and evidence. Status must be "
    "visible_edit_artifact, no_visible_edit_artifact, or uncertain. Evidence must be a specific "
    "observation under40words. Do not judge story quality, identity or intended action."
)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("edit artifact response contains duplicate fields")
        result[key] = value
    return result


def parse_edit_artifact_observation(response: str) -> dict[str, str]:
    if len(response.encode("utf-8")) > 64 * 1024:
        raise ValueError("edit artifact response exceeds its boundary")
    try:
        data = json.loads(response, object_pairs_hook=_unique_fields)
    except json.JSONDecodeError as error:
        raise ValueError("edit artifact observation requires JSON") from error
    if (
        not isinstance(data, dict)
        or set(data) != {"status", "evidence"}
        or not isinstance(data["status"], str)
        or data["status"] not in {"visible_edit_artifact", "no_visible_edit_artifact", "uncertain"}
        or not isinstance(data["evidence"], str)
        or not 1 <= len(data["evidence"].strip()) <= 2000
    ):
        raise ValueError("edit artifact observation requires a known status and evidence")
    return {"status": data["status"], "evidence": data["evidence"]}


def _adjacent_samples(frames: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if (
        len(frames) < 2
        or any(type(frame) is not int or frame < 0 for frame in frames)
        or tuple(sorted(set(frames))) != frames
    ):
        raise ValueError("edit artifact review requires at least two ordered distinct samples")
    return tuple(pairwise(frames))


def parse_edit_artifact_pairs(
    response: str, frames: tuple[int, ...]
) -> dict[tuple[int, int], dict[str, str]]:
    adjacent = _adjacent_samples(frames)
    if len(response.encode("utf-8")) > 1024 * 1024:
        raise ValueError("edit artifact pairs exceed their boundary")
    try:
        pairs = json.loads(response, object_pairs_hook=_unique_fields)
    except json.JSONDecodeError as error:
        raise ValueError("edit artifact pairs require JSON") from error
    if not isinstance(pairs, list) or len(pairs) != len(frames) - 1:
        raise ValueError("edit artifact assessment requires every adjacent sampled pair")
    observations = {}
    for row, (first, last) in zip(pairs, adjacent, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {"frames", "response"}
            or row["frames"] != [first, last]
            or any(type(index) is not int for index in row["frames"])
            or not isinstance(row["response"], str)
        ):
            raise ValueError("edit artifact pair frame binding changed")
        try:
            observations[(first, last)] = parse_edit_artifact_observation(row["response"])
        except ValueError:
            # One malformed observation cannot erase another pair's visible artifact.
            continue
    return observations


def apply_edit_artifacts(
    audit: ShotVisualAudit,
    observations: Mapping[tuple[int, int], Mapping[str, str]],
    frames: tuple[int, ...],
) -> ShotVisualAudit:
    findings = []
    for first, last in _adjacent_samples(frames):
        row = observations.get((first, last), {})
        value = row.get("status", "uncertain")
        findings.append(
            VisualFinding(
                "temporal_stability",
                {
                    "visible_edit_artifact": "fail",
                    "no_visible_edit_artifact": "pass",
                    "uncertain": "uncertain",
                }[value],
                f"Single continuous shot; sampled edit artifacts: {value}. "
                + row.get("evidence", "No valid observation supplied.")
                + " Unsampled intervals and hard cuts without a visible blend remain unverified.",
                (first, last),
            )
        )
    failed = audit.status == "fail" or any(row.status == "fail" for row in findings)
    uncertain = audit.status == "needs_review" or any(row.status == "uncertain" for row in findings)
    return replace(
        audit,
        findings=(*audit.findings, *findings),
        status="fail" if failed else "needs_review" if uncertain else "machine_pass",
    )
