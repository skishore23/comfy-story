"""Independent single-frame extent observations, separate from screenplay requirements."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace

from duet.duetx.film_audit import ShotVisualAudit, VisualFinding


@dataclass(frozen=True, slots=True)
class VisibilityObservation:
    extent: str
    evidence: str


def visibility_observation_prompt(names: tuple[str, ...], frame: int) -> str:
    return (
        "Describe actual visible extent in each CURRENT FRAME for the named reference subjects. "
        "References establish identity only, never visibility in the current frame. No intended "
        "action or desired visibility has been supplied. For each subject and frame first describe "
        "visible parts and any covering object or picture edge, then classify extent. "
        "entire: subject is inside the picture and not covered by another subject or object; "
        "ordinary self-occlusion from viewpoint is allowed. partial: some of the subject is "
        "covered by another subject/object or cropped by a picture edge, even if recognizable "
        "or countable. absent: cannot identify the subject in this frame. uncertain: resolution "
        "or identity is insufficient to assess extent. Do not infer covered parts from other "
        'frames or a reference. Return only JSON {"observations": {subject_name: '
        '{frame_index_as_string: {"evidence": "specific visible parts and occlusion, at most '
        '15 words", "extent": "entire|partial|absent|uncertain"}}}}. Report every subject at '
        "every supplied frame. Do not return pass, approval or an overall score.\n"
        + json.dumps({"subjects": list(names), "frame_indices": [frame]})
    )


def parse_visibility_observation(
    text: str, names: tuple[str, ...], frame: int
) -> dict[str, VisibilityObservation]:
    if len(text.encode()) > 65_536:
        raise ValueError("visibility response exceeds its boundary")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("visibility response must be JSON") from error
    if not isinstance(data, dict) or set(data) != {"observations"}:
        raise ValueError("visibility response requires only observations")
    rows = data["observations"]
    if not isinstance(rows, dict) or not set(rows) <= set(names):
        raise ValueError("visibility observations must use requested entity names")
    result = {}
    for name, values in rows.items():
        if not isinstance(values, dict) or not set(values) <= {str(frame)}:
            raise ValueError("visibility observations must use the supplied frame")
        if str(frame) not in values:
            continue
        row = values[str(frame)]
        if not isinstance(row, dict) or set(row) != {"extent", "evidence"}:
            raise ValueError("visibility observation fields are invalid")
        extent, evidence = row["extent"], row["evidence"]
        if (
            extent not in ("entire", "partial", "absent", "uncertain")
            or not isinstance(evidence, str)
            or not evidence.strip()
            or len(evidence) > 1000
        ):
            raise ValueError("visibility observation requires finite extent and bounded evidence")
        result[name] = VisibilityObservation(extent, evidence)
    return result


def visibility_frame_order(frames: tuple[int, ...]) -> tuple[int, ...]:
    """Inspect the ending first; a contradiction can stop this extra assessment early."""
    return (frames[-1], *frames[:-1])


def parse_visibility_report(
    text: str, names: tuple[str, ...], frames: tuple[int, ...]
) -> dict[int, dict[str, VisibilityObservation]]:
    if len(text.encode()) > 1_000_000:
        raise ValueError("visibility report exceeds its boundary")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("visibility report must be JSON") from error
    if not isinstance(rows, list) or len(rows) > len(frames):
        raise ValueError("visibility report must contain bounded frame observations")
    result = {}
    for expected, row in zip(visibility_frame_order(frames), rows, strict=False):
        if (
            not isinstance(row, dict)
            or set(row) != {"frame_index", "response"}
            or type(row["frame_index"]) is not int
            or row["frame_index"] != expected
            or not isinstance(row["response"], str)
        ):
            raise ValueError("visibility report frame binding changed")
        result[expected] = parse_visibility_observation(row["response"], names, expected)
    return result


def apply_full_visibility(
    audit: ShotVisualAudit,
    observed: Mapping[int, Mapping[str, VisibilityObservation]],
    names: tuple[str, ...],
    frames: tuple[int, ...],
) -> ShotVisualAudit:
    findings = list(audit.findings)
    for name in names:
        failed = tuple(
            frame
            for frame in frames
            if (row := observed.get(frame, {}).get(name)) is not None
            and row.extent in ("partial", "absent")
        )
        unknown = tuple(
            frame
            for frame in frames
            if (row := observed.get(frame, {}).get(name)) is None or row.extent == "uncertain"
        )
        status = "fail" if failed else "uncertain" if unknown else "pass"
        details = "; ".join(f"{frame}: {observed[frame][name].evidence}" for frame in failed)
        findings.append(
            VisualFinding(
                "continuity",
                status,
                f"{name} must remain fully visible and unobscured throughout. "
                f"Partial or absent at {failed}; uncertain or uninspected at {unknown}. "
                + (details if failed else "Unsampled intervals remain unverified."),
                failed or unknown or frames,
            )
        )
    status = (
        "fail"
        if audit.status == "fail" or any(x.status == "fail" for x in findings)
        else "needs_review"
        if audit.status != "machine_pass" or any(x.status == "uncertain" for x in findings)
        else "machine_pass"
    )
    return replace(audit, findings=tuple(findings), status=status)
