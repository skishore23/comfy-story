from __future__ import annotations

import json
from dataclasses import replace

import pytest

from comfy_story.film_audit import ShotVisualAudit
from comfy_story.film_edit_artifacts import (
    apply_edit_artifacts,
    parse_edit_artifact_observation,
    parse_edit_artifact_pairs,
)
from comfy_story.film_plan import FilmPlan, FilmShot, compile_candidate_prompt


def _pair(first: int, last: int, status: str) -> dict[str, object]:
    return {
        "frames": [first, last],
        "response": json.dumps({"status": status, "evidence": "Visible frame evidence."}),
    }


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "[]",
        "{}",
        '{"status":"unknown","evidence":"Visible."}',
        '{"status":null,"evidence":"Visible."}',
        '{"status":"uncertain","evidence":" "}',
        '{"status":"uncertain","evidence":42}',
        '{"status":"uncertain","evidence":"Visible.","approved":true}',
        '{"status":"visible_edit_artifact","status":"no_visible_edit_artifact",'
        '"evidence":"Visible."}',
        "x" * (64 * 1024 + 1),
    ],
)
def test_edit_observation_rejects_ambiguous_or_malformed_contracts(response: str) -> None:
    with pytest.raises(ValueError, match="edit artifact"):
        parse_edit_artifact_observation(response)


def test_a_malformed_pair_cannot_erase_an_observed_blend() -> None:
    pairs = [_pair(0, 29, "visible_edit_artifact"), {"frames": [29, 59], "response": "bad JSON"}]
    observed = parse_edit_artifact_pairs(json.dumps(pairs), (0, 29, 59))
    audit = ShotVisualAudit("one", "a" * 64, "b" * 64, "c" * 64, (), (), "machine_pass")
    result = apply_edit_artifacts(audit, observed, (0, 29, 59))
    assert result.status == "fail"
    assert [(row.status, row.frame_indices) for row in result.findings] == [
        ("fail", (0, 29)),
        ("uncertain", (29, 59)),
    ]
    assert all("remain unverified" in row.evidence for row in result.findings)


@pytest.mark.parametrize(
    "pairs",
    [
        [],
        [_pair(0, 29, "uncertain")],
        [_pair(0, 29, "uncertain"), _pair(0, 59, "uncertain")],
        [_pair(0, 29, "uncertain"), _pair(59, 29, "uncertain")],
        [{"frames": [False, 29], "response": "{}"}, _pair(29, 59, "uncertain")],
        [_pair(0, 29, "uncertain"), {"frames": [29, 59], "response": {}}],
    ],
)
def test_edit_pairs_bind_every_adjacent_sample(pairs: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="edit artifact"):
        parse_edit_artifact_pairs(json.dumps(pairs), (0, 29, 59))


def test_clear_samples_never_cancel_an_existing_rejection() -> None:
    audit = ShotVisualAudit("one", "a" * 64, "b" * 64, "c" * 64, (), (), "fail")
    observed = parse_edit_artifact_pairs(
        json.dumps([_pair(0, 29, "no_visible_edit_artifact")]), (0, 29)
    )
    assert apply_edit_artifacts(audit, observed, (0, 29)).status == "fail"
    assert apply_edit_artifacts(
        replace(audit, status="needs_review"), observed, (0, 29)
    ).status == ("needs_review")
    assert apply_edit_artifacts(replace(audit, status="machine_pass"), {}, (0, 29)).status == (
        "needs_review"
    )


@pytest.mark.parametrize("frames", [(), (0,), (0, 0), (29, 0), (-1, 29), (False, 29)])
def test_invalid_sample_rosters_cannot_create_vacuous_passes(frames: tuple[int, ...]) -> None:
    audit = ShotVisualAudit("one", "a" * 64, "b" * 64, "c" * 64, (), (), "machine_pass")
    with pytest.raises(ValueError, match="ordered distinct samples"):
        parse_edit_artifact_pairs("[]", frames)
    with pytest.raises(ValueError, match="ordered distinct samples"):
        apply_edit_artifacts(audit, {}, frames)


def test_continuous_take_policy_is_explicit_and_changes_shot_identity() -> None:
    shot = FilmShot("one", 5000, "A movement", "A leaf moves.", ())
    plan = FilmPlan("edit-check", "A movement", 5000, (), (shot,))
    plan.validate()
    continuous = replace(shot, camera_policy="Single continuous shot")
    changed = replace(plan, shots=(continuous,)).validate()
    assert continuous.digest != shot.digest
    assert "one continuous take" in compile_candidate_prompt(changed, 0)
    assert "one continuous take" not in compile_candidate_prompt(plan, 0)
