from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from duet.duetx.film_audit import parse_visual_audit, sample_frame_indices, visual_audit_prompt
from duet.duetx.film_plan import FilmFact, FilmShot


def test_visual_sampling_includes_last_selected_frame_without_crossing_cut() -> None:
    samples = sample_frame_indices(9000)
    assert len(samples) == 9
    assert samples[0] == 0
    assert samples[-1] == 215
    assert tuple(sorted(set(samples))) == samples
    with pytest.raises(ValueError, match="exact frame"):
        sample_frame_indices(9001)


@pytest.mark.parametrize(
    ("observation", "status"),
    [
        ({"door": "open"}, "machine_pass"),
        ({"door": "closed"}, "fail"),
        ({}, "needs_review"),
    ],
)
def test_machine_observations_do_not_promote_intended_effects(
    observation: dict[str, str], status: str
) -> None:
    shot = FilmShot("open", 9000, "Enter", "Open door", (), effects=(FilmFact("door", "open"),))
    response = json.dumps(
        {
            "checks": {
                "action": {
                    "status": "pass",
                    "evidence": "Visible door gap",
                    "frame_indices": [215],
                }
            },
            "observed_effects": observation,
        }
    )
    result = parse_visual_audit(
        response,
        shot,
        video_sha256="a" * 64,
        model_sha256="b" * 64,
        sampled_frames=(0, 215),
        required_checks=("action",),
    )
    assert result.status == status
    assert not hasattr(result, "reviewer")
    assert result.video_sha256 == "a" * 64


def test_missing_checks_or_invented_frame_evidence_cannot_pass() -> None:
    shot = FilmShot("one", 9000, "Hold", "Wait", ())
    with pytest.raises(ValueError, match="every required check"):
        parse_visual_audit(
            '{"checks":{},"observed_effects":{}}',
            shot,
            video_sha256="a" * 64,
            model_sha256="b" * 64,
            sampled_frames=(0,),
        )

    with pytest.raises(ValueError, match="supplied source frame"):
        parse_visual_audit(
            json.dumps(
                {
                    "checks": {
                        "action": {
                            "status": "pass",
                            "evidence": "Seen",
                            "frame_indices": [99],
                        }
                    },
                    "observed_effects": {},
                }
            ),
            shot,
            video_sha256="a" * 64,
            model_sha256="b" * 64,
            sampled_frames=(0,),
            required_checks=("action",),
        )


@pytest.mark.parametrize(
    ("conditions", "expected_status"),
    [
        ({"Leon.place": "inside"}, "fail"),
        ({}, "needs_review"),
        ({"Leon.place": "outside"}, "machine_pass"),
    ],
)
def test_continuity_values_override_an_incorrect_model_pass_label(
    conditions: dict[str, str], expected_status: str
) -> None:
    shot = FilmShot("wait", 9000, "Wait", "Leon waits outside", ("Leon",))
    response = json.dumps(
        {
            "checks": {
                "continuity": {
                    "status": "pass",
                    "evidence": "Observed by the window",
                    "frame_indices": [0],
                }
            },
            "observed_effects": {},
            "observed_conditions": conditions,
        }
    )
    audit = parse_visual_audit(
        response,
        shot,
        video_sha256="a" * 64,
        model_sha256="b" * 64,
        sampled_frames=(0,),
        required_checks=("continuity",),
        expected_conditions={"Leon.place": "outside"},
    )
    assert audit.status == expected_status
    assert dict(audit.observed_conditions) == conditions


def test_prompt_separates_reference_assets_and_requests_only_applicable_checks() -> None:
    shot = FilmShot("one", 9000, "Hold", "A mug stays on a shelf", ("Cup",))
    prompt, checks = visual_audit_prompt(shot, (0, 215), {"Cup": "Prop"})
    assert "props" in checks
    assert "identity" not in checks
    assert "never evidence that an event occurred" in prompt
    assert "If intended_effects is empty, return observed_effects as {}" in prompt
    _, with_person = visual_audit_prompt(
        replace(shot, present=("Ada",)), (0, 215), {"Ada": "Character"}
    )
    assert "identity" in with_person
    assert "wardrobe" in with_person


def test_audit_requests_state_comparison_in_addition_to_matching_scenery() -> None:
    shot = FilmShot(
        "wait",
        9000,
        "Hesitation",
        "Leon waits",
        ("Leon",),
        requires=(FilmFact("Leon.activity", "waiting"),),
    )
    context = "Planned, unverified continuity to preserve: Leon.place = outside."
    prompt, checks = visual_audit_prompt(
        shot, (0, 215), {"Leon": "Character"}, continuity_context=context
    )
    assert "continuity" in checks
    assert context in prompt
    assert '"starting_conditions": {"Leon.activity": "waiting"}' in prompt
    assert "matching setting alone is insufficient" in prompt


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Actual audit frame extraction uses ffmpeg")
def test_local_audit_binds_real_frames_and_keeps_malformed_model_output_unreviewed(
    tmp_path: Path,
) -> None:
    from duet.duetx.film_audit_cli import audit_export
    from duet.duetx.film_export import file_digest
    from duet.duetx.film_plan import FilmPlan, RenderedTake
    from duet.duetx.story_contracts import canonical_story_json

    run = tmp_path / "run"
    export = run / "export"
    export.mkdir(parents=True)
    (run / "run.json").write_text('{"export_attempt":1}')
    video = export / "film.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=24:d=0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    digest = file_digest(video)
    (run / "sources").mkdir()
    shutil.copyfile(video, run / "sources" / (digest + ".mp4"))
    plan = FilmPlan("audit", "A fixture", 500, (), (FilmShot("one", 500, "Hold", "Stay blue", ()),))
    take = RenderedTake("one", plan.shots[0].digest, "a" * 64, digest, None, 500)
    (export / "film-plan.json").write_bytes(canonical_story_json(plan))
    (export / "rendered-takes.json").write_text(json.dumps([asdict(take)]))
    (export / "export-receipt.json").write_text(
        json.dumps(
            {
                "film_sha256": digest,
                "plan_sha256": hashlib.sha256(canonical_story_json(plan)).hexdigest(),
                "rendered_take_sha256s": [take.digest],
            }
        )
    )
    inputs = tmp_path / "inputs.json"
    inputs.write_text('{"library":{"references":[]}}')
    calls = []

    def reviewer(content: list[dict[str, Any]]) -> str:
        calls.append(content)
        if len(calls) <= 2:
            return "I cannot determine that."
        assert "intended_action" not in json.dumps(content)
        assert "authored captions" in content[0]["text"]
        assert content[-1]["image"].endswith("frame-09.png")
        assert [Path(row["image"]).name for row in content if row["type"] == "image"] == [
            "frame-01.png",
            "frame-05.png",
            "frame-09.png",
        ]
        assert "not contradictions by themselves" in content[0]["text"]
        return '{"summary":"A blue picture","coherence":"uncertain"}'

    report_path = audit_export(
        run,
        inputs,
        tmp_path,
        tmp_path,
        tmp_path / "audit",
        reviewer=reviewer,
        model_sha256="b" * 64,
    )
    report = json.loads(report_path.read_text())
    assert report["counts"]["needs_review"] == 1
    assert report["counts"]["machine_pass"] == 0
    assert report["approval_created"] is False
    assert len(report["shots"][0]["sampled_frames"]) == 9
    assert report["shots"][0]["sampled_frames"][-1]["frame_index"] == 11
    assert not (export / "accepted-takes.json").exists()
    assert len(calls) == 3
    assert calls[1][:-1] == calls[0]
    assert len(list((tmp_path / "audit" / "one").glob("model-response-attempt-*.txt"))) == 2


@pytest.mark.parametrize(
    ("observations", "status"),
    [
        ({"Adult": {"0": "visible", "239": "not_visible"}}, "fail"),
        ({"Adult": {"0": "not_visible", "239": "visible"}}, "fail"),
        ({"Adult": {"0": "visible", "239": "uncertain"}}, "needs_review"),
        ({"Adult": {"0": "visible"}}, "needs_review"),
        ({}, "needs_review"),
        ({"Adult": {"0": "visible", "239": "visible"}}, "machine_pass"),
    ],
)
def test_continuous_visibility_overrides_machine_pass(
    observations: dict[str, dict[str, str]], status: str
) -> None:
    shot = FilmShot(
        "walk",
        10_000,
        "Travel together",
        "Walk together",
        ("Adult", "Young"),
        visible_throughout=("Adult",),
    )
    raw = json.dumps(
        {
            "checks": {
                "identity": {
                    "status": "pass",
                    "evidence": "Recognizable subjects",
                    "frame_indices": [0, 239],
                }
            },
            "observed_effects": {},
            "observed_visibility": observations,
        }
    )
    audit = parse_visual_audit(
        raw,
        shot,
        video_sha256="a" * 64,
        model_sha256="b" * 64,
        sampled_frames=(0, 239),
        required_checks=("identity",),
    )
    assert audit.status == status
    assert audit.findings[-1].check == "continuity"
    if status == "fail":
        assert "not visible" in audit.findings[-1].evidence
        assert audit.findings[0].status == "pass"


@pytest.mark.parametrize(
    "observations",
    [
        {"Intruder": {"0": "visible"}},
        {"Adult": {"999": "visible"}},
        {"Adult": {"0": True}},
        {"Adult": []},
    ],
)
def test_visibility_rejects_invented_entities_frames_and_statuses(observations: object) -> None:
    shot = FilmShot("walk", 10_000, "Travel", "Walk", ("Adult",), visible_throughout=("Adult",))
    raw = json.dumps(
        {
            "checks": {"identity": {"status": "pass", "evidence": "Seen", "frame_indices": [0]}},
            "observed_effects": {},
            "observed_visibility": observations,
        }
    )
    with pytest.raises(ValueError, match="visibility observations"):
        parse_visual_audit(
            raw,
            shot,
            video_sha256="a" * 64,
            model_sha256="b" * 64,
            sampled_frames=(0, 239),
            required_checks=("identity",),
        )


def test_visibility_prompt_uses_explicit_scope_and_separate_source_frames() -> None:
    shot = FilmShot("hold", 10_000, "Show", "Hold", ("Box",), visible_throughout=("Box",))
    prompt, _ = visual_audit_prompt(shot, (0, 239), {"Box": "Prop"})
    assert '"visible_throughout": ["Box"]' in prompt
    assert "observed_visibility" in prompt
    assert "Judge each SHOT FRAME independently" in prompt
    unscoped, _ = visual_audit_prompt(replace(shot, visible_throughout=()), (0, 239), {})
    assert "observed_visibility" not in unscoped


@pytest.mark.parametrize("status", ["fail", "needs_review"])
def test_matching_opening_does_not_clear_other_visual_rejections(status: str) -> None:
    from duet.duetx.film_audit import ShotVisualAudit, apply_opening_conditions

    audit = ShotVisualAudit("one", "a" * 64, "b" * 64, "c" * 64, (), (), status)
    result = apply_opening_conditions(audit, {"door": "closed"}, {"door": "closed"})
    assert result.status == status
    assert result.findings[-1].frame_indices == (0,)


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"observed_conditions":{"unknown":"closed"}}',
        '{"observed_conditions":{"door":null}}',
        '{"observed_conditions":{},"approval":true}',
    ],
)
def test_opening_observations_reject_malformed_or_unrequested_facts(response: str) -> None:
    from duet.duetx.film_audit import parse_opening_conditions

    with pytest.raises(ValueError, match="opening"):
        parse_opening_conditions(response, {"door": "closed"})


@pytest.mark.parametrize(
    "response",
    [
        '{"observed_framing":{"999":{"framing":"changed","evidence":"Crop"}}}',
        '{"observed_framing":{"29":{"framing":[],"evidence":"Crop"}}}',
        '{"observed_framing":{"29":{"framing":"unchanged","evidence":""}}}',
        '{"observed_framing":{"29":"unchanged"}}',
    ],
)
def test_camera_observations_reject_invalid_frames_labels_and_missing_evidence(
    response: str,
) -> None:
    from duet.duetx.film_audit import parse_camera_observations

    with pytest.raises(ValueError, match="camera"):
        parse_camera_observations(response, (0, 29))


def test_preserved_conditions_are_also_checked_in_the_opening() -> None:
    from duet.duetx.film_audit import condition_scopes

    shot = FilmShot("return", 5000, "Return", "Hold", ("Vessel",))
    expected = {"Vessel.filled": "yes"}
    assert condition_scopes(shot, expected) == (expected, expected)
    changing = replace(shot, effects=(FilmFact("Vessel.filled", "no"),))
    assert condition_scopes(changing, expected) == (expected, {})


def test_opening_observer_cannot_see_the_preferred_value_or_future_action(tmp_path: Path) -> None:
    from duet.duetx.film_audit_cli import _opening_content

    choices = {"Vessel.filled": ["yes", "no"]}
    first = _opening_content(
        {"Vessel.filled": "yes"}, tmp_path / "frame.png", {}, vocabularies=choices
    )
    second = _opening_content(
        {"Vessel.filled": "no"}, tmp_path / "frame.png", {}, vocabularies=choices
    )
    assert first == second
    assert len([row for row in first if row["type"] == "image"]) == 1
    assert '"Vessel.filled": ["no", "yes"]' in first[0]["text"]
    assert "Required opening conditions" not in first[0]["text"]
