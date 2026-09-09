from __future__ import annotations

import json

import pytest

from comfy_story.film_audit import ShotVisualAudit
from comfy_story.film_visibility_audit import (
    VisibilityObservation,
    apply_full_visibility,
    parse_visibility_observation,
    parse_visibility_report,
    visibility_observation_prompt,
)


def _audit(status: str = "machine_pass") -> ShotVisualAudit:
    return ShotVisualAudit("one", "a" * 64, "b" * 64, "c" * 64, (), (), status)


@pytest.mark.parametrize(
    ("extent", "expected"),
    [
        ("entire", "machine_pass"),
        ("partial", "fail"),
        ("absent", "fail"),
        ("uncertain", "needs_review"),
    ],
)
def test_countable_is_not_full_visibility(extent: str, expected: str) -> None:
    raw = json.dumps(
        {
            "observations": {
                "Vessel": {
                    "119": {
                        "extent": extent,
                        "evidence": "Visible parts and occluding object",
                    }
                }
            }
        }
    )
    observed = parse_visibility_observation(raw, ("Vessel",), 119)
    result = apply_full_visibility(_audit(), {119: observed}, ("Vessel",), (119,))
    assert result.status == expected
    assert result.findings[-1].frame_indices == (119,)


def test_blind_prompt_has_identity_and_frame_but_no_action_or_desired_extent() -> None:
    prompt = visibility_observation_prompt(("Lantern", "Vessel"), 42)
    assert json.loads(prompt.split("\n")[1]) == {
        "subjects": ["Lantern", "Vessel"],
        "frame_indices": [42],
    }
    assert "even if recognizable or countable" in prompt
    assert "intended_action" not in prompt
    assert "fully_visible_throughout" not in prompt


@pytest.mark.parametrize(
    "observed",
    [
        {},
        {119: {}},
        {119: {"Vessel": VisibilityObservation("entire", "Unobscured")}},
    ],
)
def test_missing_frames_or_entities_cannot_clear_review(
    observed: dict[int, dict[str, VisibilityObservation]],
) -> None:
    assert apply_full_visibility(_audit(), observed, ("Vessel",), (0, 119)).status == "needs_review"
    assert apply_full_visibility(_audit("fail"), observed, ("Vessel",), (0, 119)).status == "fail"


def test_early_contradiction_replays_without_inventing_inspection_of_remaining_frames() -> None:
    raw = json.dumps(
        {
            "observations": {
                "Vessel": {
                    "119": {
                        "extent": "partial",
                        "evidence": "Only the lid is visible behind a box",
                    }
                }
            }
        }
    )
    report = json.dumps([{"frame_index": 119, "response": raw}])
    observed = parse_visibility_report(report, ("Vessel",), (0, 59, 119))
    result = apply_full_visibility(_audit(), observed, ("Vessel",), (0, 59, 119))
    assert result.status == "fail"
    assert "uninspected at (0, 59)" in result.findings[-1].evidence
    assert result.findings[-1].frame_indices == (119,)
    with pytest.raises(ValueError, match="frame binding"):
        parse_visibility_report(
            report.replace('"frame_index": 119', '"frame_index": 59'), ("Vessel",), (0, 59, 119)
        )


@pytest.mark.parametrize(
    "rows",
    [
        {"Unknown": {}},
        {"Vessel": {"999": {}}},
        {"Vessel": []},
        {"Vessel": {"119": {"extent": "visible", "evidence": "Countable"}}},
        {"Vessel": {"119": {"extent": "entire", "evidence": ""}}},
        {"Vessel": {"119": {"extent": True, "evidence": "Countable"}}},
    ],
)
def test_observations_reject_unbound_fields_and_ambiguous_extent(rows: object) -> None:
    with pytest.raises(ValueError, match="visibility"):
        parse_visibility_observation(json.dumps({"observations": rows}), ("Vessel",), 119)
