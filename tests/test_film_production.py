"""Exercise the public renderer and real frame/audit/export path with a simulated Comfy server.

The injected observations test scheduling, not the vision model's accuracy.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from comfy_story import film_production, film_runner
from comfy_story.film_export import file_digest
from comfy_story.film_io import film_plan_from_json
from comfy_story.film_plan import FilmFact, FilmPlan, FilmShot, planned_shot_conditions
from comfy_story.film_state_audit import ending_conditions
from comfy_story.story_contracts import canonical_story_json


@pytest.fixture
def production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> dict[str, Any]:
    if not shutil.which("ffmpeg"):
        pytest.skip("Real frame extraction and conform require FFmpeg")
    source_filter = getattr(request, "param", "null")
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=24:d=0.5",
            "-vf",
            source_filter,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    digest = file_digest(source)
    Image.new("RGB", (1344, 768), "blue").save(tmp_path / "box.png")
    plan = FilmPlan(
        "production",
        "A box",
        1000,
        (),
        (
            FilmShot("one", 500, "Setup", "Hold", ("Box",)),
            FilmShot("two", 500, "Resolve", "Hold", ("Box",), depends_on=("one",)),
        ),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(asdict(plan)))
    settings = {
        "library": {
            "project_name": "Test",
            "references": [
                {"name": "Box", "role": "Prop", "file": "box.png", "note": "Blue box"},
            ],
        },
        "shots": [{"world": "box.png", "variation": 10}, {"world": "box.png", "variation": 20}],
        "burn_subtitles": False,
    }
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(settings))
    submissions: list[dict[str, Any]] = []
    rendered: set[str] = set()
    calls: list[list[dict[str, Any]]] = []
    statuses: list[str] = []
    whole = {
        "summary": "A box remains visible.",
        "protagonist_goal": "Keep the box visible.",
        "decision": "Hold the frame.",
        "outcome": "The box remains in place.",
        "coherence": "coherent",
        "contradictions": [],
    }

    def comfy_request(server: str, path: str, payload: Any = None) -> dict[str, Any]:
        if path == "/object_info":
            return {
                "ComfyStory": {
                    "input": {"optional": {"Output duration (ms)": [], "Scene entities": []}}
                }
            }
        if path == "/prompt":
            submissions.append(payload["prompt"])
            return {"prompt_id": str(len(submissions))}
        workflow = submissions[int(path.rsplit("/", 1)[-1]) - 1]
        outputs = {}
        parent = None
        for node_id, node in workflow.items():
            revision = hashlib.sha256(canonical_story_json([node, parent])).hexdigest()
            rendered.add(revision)
            outputs[node_id] = {
                "comfy_story": [
                    {
                        "owner_node_id": node_id,
                        "revision_sha256": revision,
                        "video_sha256": digest,
                        "parent_revision_sha256": parent,
                    }
                ]
            }
            parent = revision
        return {path.rsplit("/", 1)[-1]: {"status": {"status_str": "success"}, "outputs": outputs}}

    def render(plan_path: Path, inputs: Path, server: str, run: Path) -> Path:
        (run / "sources").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, run / "sources" / (digest + ".mp4"))
        return film_runner.run_film(plan_path, inputs, server, run)

    def reviewer(content: list[dict[str, Any]]) -> str:
        calls.append(content)
        prompt = content[0]["text"]
        if "Observe CURRENT STARTING FRAME only" in prompt:
            current_plan = film_plan_from_json(plan_path.read_text())
            return json.dumps(
                {
                    fact.key: {"value": fact.value, "evidence": "Injected opening asset state."}
                    for fact in current_plan.initial_facts
                }
            )
        if "Opening state alternatives:" in prompt:
            current_plan = film_plan_from_json(plan_path.read_text())
            shot_id = Path(content[-1]["image"]).parent.name
            index = next(i for i, shot in enumerate(current_plan.shots) if shot.shot_id == shot_id)
            return json.dumps({"observed_conditions": planned_shot_conditions(current_plan, index)})
        if "Observe CURRENT ENDING FRAME only" in prompt:
            current_plan = film_plan_from_json(plan_path.read_text())
            shot_id = Path(content[-1]["image"]).parent.name
            index = next(i for i, shot in enumerate(current_plan.shots) if shot.shot_id == shot_id)
            return json.dumps(
                {
                    key: {"value": value, "evidence": "Injected ending state."}
                    for key, value in ending_conditions(current_plan, index).items()
                }
            )
        if "source_frame_indices" not in prompt:
            return json.dumps(whole)
        data = json.loads(prompt.split("\n")[1])
        status = statuses.pop(0) if statuses else "pass"
        return json.dumps(
            {
                "checks": {
                    key: {
                        "status": status if key == "action" else "pass",
                        "evidence": "The blue box holds still."
                        if status == "pass"
                        else "The box moves @Box.",
                        "frame_indices": [data["source_frame_indices"][0]],
                    }
                    for key in data["required_checks"]
                },
                "observed_effects": {},
            }
        )

    monkeypatch.setattr(film_runner, "_json_request", comfy_request)
    monkeypatch.setattr(
        film_runner, "_input_hashes", lambda *_: {"box.png": file_digest(tmp_path / "box.png")}
    )
    monkeypatch.setattr(film_production, "run_film", render)
    kwargs: dict[str, Any] = {
        "model": tmp_path,
        "comfy_input": tmp_path,
        "reviewer": reviewer,
        "model_sha256": "b" * 64,
    }

    def run() -> Path:
        return film_production.run_production(
            plan_path, inputs, "http://localhost:8188", tmp_path / "production", **kwargs
        )

    return {
        "run": run,
        "render": render,
        "root": tmp_path / "production",
        "submissions": submissions,
        "rendered": rendered,
        "calls": calls,
        "statuses": statuses,
        "whole": whole,
        "inputs": inputs,
        "settings": settings,
        "kwargs": kwargs,
        "plan": plan_path,
    }


@pytest.mark.parametrize("observed", ["open", None, "invalid-json"])
def test_invalid_starting_state_stops_before_generation_and_is_not_rerolled(
    production: dict[str, Any], observed: str | None
) -> None:
    plan = json.loads(production["plan"].read_text())
    plan["initial_facts"] = [{"key": "Box.lid", "value": "closed"}]
    plan["shots"][1]["effects"] = [{"key": "Box.lid", "value": "open"}]
    production["plan"].write_text(json.dumps(plan))
    calls = []

    def review(content: list[dict[str, Any]]) -> str:
        calls.append(content)
        prompt = content[0]["text"]
        assert "CURRENT STARTING FRAME" in prompt
        assert "required" not in prompt.lower()
        assert "Setup" not in prompt
        assert "Resolve" not in prompt
        assert '"closed"' in prompt
        assert '"open"' in prompt
        if observed == "invalid-json":
            return observed
        return json.dumps({"Box.lid": {"value": observed, "evidence": "Visible asset state."}})

    production["kwargs"]["reviewer"] = review
    for _ in range(2):
        with pytest.raises(film_production.ProductionStopped, match="No video shots queued"):
            production["run"]()
    assert len(calls) == 1
    assert production["submissions"] == []
    assert not (production["root"] / "shot-001").exists()
    report = production["root"] / "starting-state" / "report.json"
    data = json.loads(report.read_text())
    assert data["status"] == ("fail" if observed == "open" else "needs_review")
    (report.parent / "response.txt").write_text(
        json.dumps({"Box.lid": {"value": "closed", "evidence": "Tampered approval."}})
    )
    with pytest.raises(ValueError, match="assessment changed"):
        production["run"]()
    assert production["submissions"] == []


def test_failed_candidate_is_retried_before_advancing_and_resume_uses_exact_selected_parent(
    production: dict[str, Any],
) -> None:
    production["statuses"].extend(["fail", "pass", "pass"])
    output = production["run"]()
    submissions = production["submissions"]
    assert [len(row) for row in submissions] == [1, 1, 2]
    assert len(production["rendered"]) == 3
    assert submissions[1]["10"] == submissions[2]["10"]
    assert submissions[0]["10"] != submissions[1]["10"]
    assert "Correction for this attempt" in submissions[1]["10"]["inputs"]["What happens next?"]
    retry_prompt = submissions[1]["10"]["inputs"]["What happens next?"]
    assert "Action to complete visibly: Hold" in retry_prompt
    assert "moves Box" not in retry_prompt
    # Negative findings remain inspectable without becoming renderer instructions.
    assessment = production["root"] / "shot-001/attempt-01/audit-001/one/model-response.txt"
    assert "moves @Box" in assessment.read_text()
    state = json.loads((production["root"] / "production.json").read_text())
    assert state["status"] == "ready_for_review"
    assert state["approval_created"] is False
    assert [row["variation"] for row in state["selected"]] == [11, 20]
    assert not list(production["root"].rglob("accepted-takes.json"))
    assert len(production["calls"]) == 6  # Three candidate checks, two final checks, one retell.
    assert production["run"]() == output
    assert len(submissions) == 3
    assert len(production["calls"]) == 6


@pytest.mark.parametrize(
    ("statuses", "attempts", "reason"),
    [
        (["fail", "fail"], 2, "failed after 2 attempts"),
        (["uncertain"], 1, "uncertain"),
    ],
)
def test_budget_or_uncertainty_stops_without_downstream_generation(
    production: dict[str, Any], statuses: list[str], attempts: int, reason: str
) -> None:
    production["statuses"].extend(statuses)
    with pytest.raises(film_production.ProductionStopped, match=reason):
        production["run"]()
    assert [len(row) for row in production["submissions"]] == [1] * attempts
    state = json.loads((production["root"] / "production.json").read_text())
    assert state["status"] == "needs_attention"
    assert state["selected"] == []
    preview = state["preview"]
    assert preview["attempt"] == attempts
    assert preview["film_sha256"] == file_digest(Path(preview["film"]))
    assert "film" not in state
    assert not (production["root"] / "shot-002").exists()


def test_whole_film_contradiction_is_not_ready_even_when_each_shot_passes(
    production: dict[str, Any],
) -> None:
    production["whole"]["contradictions"] = ["The cause is missing."]
    with pytest.raises(film_production.ProductionStopped, match="retelling is inconsistent"):
        production["run"]()
    assert len(production["submissions"]) == 2


def test_persisted_stages_distinguish_generation_from_review(
    production: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rendering = []
    reviewing = []
    original_render = production["render"]
    original_review = production["kwargs"]["reviewer"]

    def stage() -> str:
        return str(json.loads((production["root"] / "production.json").read_text())["phase"])

    def render(*args: Any, **kwargs: Any) -> Path:
        rendering.append(stage())
        result = original_render(*args, **kwargs)
        assert isinstance(result, Path)
        return result

    def review(content: list[dict[str, Any]]) -> str:
        reviewing.append(stage())
        return str(original_review(content))

    monkeypatch.setattr(film_production, "run_film", render)
    production["kwargs"]["reviewer"] = review
    production["run"]()
    assert rendering == ["rendering_shot", "rendering_shot"]
    assert reviewing == ["checking_shot", "checking_shot"] + ["checking_film"] * 3
    assert stage() == "complete"


@pytest.mark.parametrize("field", ["summary", "protagonist_goal", "decision", "outcome"])
def test_missing_narrative_evidence_cannot_pass_on_coherence_label_alone(
    production: dict[str, Any], field: str
) -> None:
    del production["whole"][field]
    with pytest.raises(film_production.ProductionStopped, match="retelling could not be assessed"):
        production["run"]()
    state = json.loads((production["root"] / "production.json").read_text())
    assert state["status"] == "needs_attention"
    assert len(state["selected"]) == 2
    assert len(production["submissions"]) == 2
    assert state["approval_created"] is False


def test_changed_verification_protocol_cannot_reuse_old_assessments(
    production: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    production["run"]()
    submissions = len(production["submissions"])
    monkeypatch.setattr(film_production, "VISUAL_AUDIT_PROTOCOL", "future-audit-protocol")
    with pytest.raises(ValueError, match="verification policy changed"):
        production["run"]()
    assert len(production["submissions"]) == submissions


@pytest.mark.parametrize("filename", ["model-response.txt", "frame-01.png", "report.json"])
def test_changed_evidence_cannot_reuse_a_recorded_pass(
    production: dict[str, Any], filename: str
) -> None:
    production["run"]()
    root = production["root"] / "shot-001" / "attempt-01" / "audit-001"
    path = root / filename if filename == "report.json" else root / "one" / filename
    if filename == "report.json":
        row = json.loads(path.read_text())
        row["scope"] = "changed"
        path.write_text(json.dumps(row))
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        production["run"]()
    assert len(production["submissions"]) == 2


def test_all_future_audio_is_validated_before_first_render(production: dict[str, Any]) -> None:
    settings = production["settings"]
    settings["audio"] = [
        {"path": "missing.wav", "sha256": "a" * 64, "start_ms": 500, "duration_ms": 500}
    ]
    production["inputs"].write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="audio is missing"):
        production["run"]()
    assert not production["submissions"]


def test_interrupted_review_resumes_same_render_and_preserves_partial_evidence(
    production: dict[str, Any],
) -> None:
    original = production["kwargs"]["reviewer"]

    def broken(content: list[dict[str, Any]]) -> str:
        raise RuntimeError("checker interrupted")

    production["kwargs"]["reviewer"] = broken
    with pytest.raises(RuntimeError, match="interrupted"):
        production["run"]()
    assert len(production["submissions"]) == 1
    production["kwargs"]["reviewer"] = original
    production["run"]()
    assert len(production["submissions"]) == 2
    root = production["root"] / "shot-001" / "attempt-01"
    assert (root / "audit-001" / "one" / "frame-01.png").is_file()
    assert (root / "audit-002" / "report.json").is_file()


def test_changed_production_policy_requires_new_directory(production: dict[str, Any]) -> None:
    production["statuses"].append("uncertain")
    with pytest.raises(film_production.ProductionStopped):
        production["run"]()
    production["kwargs"]["max_attempts"] = 3
    with pytest.raises(ValueError, match="policy changed"):
        production["run"]()
    assert len(production["submissions"]) == 1


def test_checker_initialization_failure_spends_no_render_budget(
    production: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    production["kwargs"]["reviewer"] = None
    monkeypatch.setattr(film_production, "_model_identity", lambda _: "b" * 64)

    def unavailable(*_: object) -> Any:
        raise RuntimeError("checker cannot load")

    monkeypatch.setattr(film_production, "_local_reviewer", unavailable)
    with pytest.raises(RuntimeError, match="cannot load"):
        production["run"]()
    assert not production["submissions"]
    state = json.loads((production["root"] / "production.json").read_text())
    assert state["phase"] == "checking_verifier"
    assert state["status"] == "needs_attention"


def test_changed_selected_parent_stops_before_reviewing_its_descendant(
    production: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = production["render"]

    def changed_parent(*args: Any) -> Path:
        output: Path = original(*args)
        path = output.parent / "rendered-takes.json"
        takes = json.loads(path.read_text())
        if len(takes) == 2:
            takes[0]["revision_sha256"] = "f" * 64
            path.write_text(json.dumps(takes))
        return output

    monkeypatch.setattr(film_production, "run_film", changed_parent)
    with pytest.raises(ValueError, match="parent changed"):
        production["run"]()
    assert len(production["calls"]) == 1


def test_prefix_sound_is_clipped_without_moving_future_tracks(tmp_path: Path) -> None:
    plan = FilmPlan(
        "sound",
        "A fixture",
        1000,
        (),
        (
            FilmShot("one", 500, "Hold", "Hold", ()),
            FilmShot("two", 500, "Hold", "Hold", ()),
        ),
    )
    settings: dict[str, Any] = {
        "shots": [{}, {}],
        "audio": [
            {"path": "music.wav", "start_ms": 0, "duration_ms": 1000},
            {"path": "effect.wav", "start_ms": 700, "duration_ms": 200},
        ],
    }
    prefix = film_production._prefix_inputs(settings, plan, 1, tmp_path)
    assert prefix["audio"] == [
        {"path": str(tmp_path / "music.wav"), "start_ms": 0, "duration_ms": 500}
    ]
    full = film_production._prefix_inputs(settings, plan, 2, tmp_path)
    assert full["audio"][1]["start_ms"] == 700
    assert settings["audio"][0]["duration_ms"] == 1000


@pytest.mark.parametrize(
    "repairs",
    [{"missing": "Wrong shot"}, {"one": "use @Other"}, {"one": "x" * 2001}, {"one": None}],
)
def test_repair_notes_cannot_change_reference_roster_or_target_unknown_shots(
    production: dict[str, Any], repairs: object
) -> None:
    production["settings"]["repair_notes"] = repairs
    production["inputs"].write_text(json.dumps(production["settings"]))
    with pytest.raises(ValueError, match="repair"):
        production["run"]()
    assert not production["submissions"]


def test_required_subject_loss_stops_customer_controller_despite_pass_labels(
    production: dict[str, Any],
) -> None:
    plan_path = production["plan"]
    plan = json.loads(plan_path.read_text())
    plan["shots"][0]["visible_throughout"] = ["Box"]
    plan_path.write_text(json.dumps(plan))
    original = production["kwargs"]["reviewer"]

    def contradictory_reviewer(content: list[dict[str, Any]]) -> str:
        response = json.loads(original(content))
        data = json.loads(content[0]["text"].split("\n")[1])
        frames = data["source_frame_indices"]
        response["observed_visibility"] = {"Box": {str(frame): "visible" for frame in frames}}
        response["observed_visibility"]["Box"][str(frames[-1])] = "not_visible"
        assert all(row["status"] == "pass" for row in response["checks"].values())
        return json.dumps(response)

    production["kwargs"]["reviewer"] = contradictory_reviewer
    with pytest.raises(film_production.ProductionStopped, match="failed after 2 attempts"):
        production["run"]()
    assert [len(row) for row in production["submissions"]] == [1, 1]
    retry = production["submissions"][1]["10"]["inputs"]["What happens next?"]
    assert "Keep these entities visible throughout, including the final frame: Box" in retry
    state = json.loads((production["root"] / "production.json").read_text())
    assert "Box must remain visible throughout" in Path(state["last_assessment"]).read_text()
    assert state["selected"] == []
    assert state["status"] == "needs_attention"
    assert state["approval_created"] is False
    assert not (production["root"] / "shot-002").exists()


def test_public_production_uses_keyed_seeds_and_resumes_without_new_submissions(
    production: dict[str, Any],
) -> None:
    settings = production["settings"]
    one, two = settings.pop("shots")
    settings["shots_by_id"] = {"two": two, "one": one}
    production["inputs"].write_text(json.dumps(settings))
    output = production["run"]()
    final = production["submissions"][-1]
    assert final["10"]["inputs"]["Variation"] == 10
    assert final["20"]["inputs"]["Variation"] == 20
    count = len(production["submissions"])
    assert production["run"]() == output
    assert len(production["submissions"]) == count


def test_customer_pause_finishes_current_shot_and_resumes_exact_parent(
    production: dict[str, Any],
) -> None:
    original = production["kwargs"]["reviewer"]
    marker = production["root"] / "pause.requested"

    def pause_after_assessment(content: list[dict[str, Any]]) -> str:
        result = original(content)
        marker.write_text("pause")
        assert isinstance(result, str)
        return result

    production["kwargs"]["reviewer"] = pause_after_assessment
    with pytest.raises(film_production.ProductionPaused, match="shot boundary"):
        production["run"]()
    assert len(production["submissions"]) == 1
    state = json.loads((production["root"] / "production.json").read_text())
    assert state["status"] == "paused"
    assert len(state["selected"]) == 1
    parent = production["submissions"][0]["10"]
    marker.unlink()
    production["kwargs"]["reviewer"] = original
    production["run"]()
    assert len(production["submissions"]) == 2
    assert production["submissions"][1]["10"] == parent


def test_progress_identifies_current_shot_before_submission(
    production: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = []
    render = production["render"]

    def inspect_progress(plan: Path, inputs: Path, server: str, run: Path) -> Path:
        state = json.loads((production["root"] / "production.json").read_text())
        observed.append((state["shot_id"], state["attempt"]))
        result = render(plan, inputs, server, run)
        assert isinstance(result, Path)
        return result

    monkeypatch.setattr(film_production, "run_film", inspect_progress)
    production["run"]()
    assert observed == [("one", 1), ("two", 1)]


@pytest.mark.parametrize("opening", ["closed", "open", None])
def test_changing_state_checks_only_first_frame_for_opening_conditions(
    production: dict[str, Any], opening: str | None
) -> None:
    plan = json.loads(production["plan"].read_text())
    plan["initial_facts"] = [{"key": "Box.lid", "value": "closed"}]
    plan["shots"][0]["requires"] = [{"key": "Box.lid", "value": "closed"}]
    plan["shots"][0]["effects"] = [{"key": "Box.lid", "value": "open"}]
    production["plan"].write_text(json.dumps(plan))
    original = production["kwargs"]["reviewer"]
    opening_calls = []

    def temporal_reviewer(content: list[dict[str, Any]]) -> str:
        prompt = content[0]["text"]
        if "Opening state alternatives:" in prompt:
            opening_calls.append(content)
            images = [x["image"] for x in content if x["type"] == "image"]
            assert len(images) == 2  # One identity reference and only the first shot frame.
            assert images[-1].endswith("frame-01.png")
            assert "intended_effects" not in prompt
            assert '"Box.lid": ["closed", "open"]' in prompt
            if Path(images[-1]).parent.name != "one":
                raw = original(content)
                assert isinstance(raw, str)
                return raw
            return json.dumps(
                {"observed_conditions": {} if opening is None else {"Box.lid": opening}}
            )
        response = json.loads(original(content))
        if "source_frame_indices" in prompt:
            data = json.loads(prompt.split("\n")[1])
            response["observed_effects"] = data["intended_effects"]
            if data["required_conditions"]:
                response["observed_conditions"] = data["required_conditions"]
        return json.dumps(response)

    production["kwargs"]["reviewer"] = temporal_reviewer
    if opening == "closed":
        production["run"]()
        assert len(production["submissions"]) == 2
    else:
        with pytest.raises(film_production.ProductionStopped):
            production["run"]()
        assert not (production["root"] / "shot-002").exists()
    assert opening_calls
    audit = json.loads(
        next(production["root"].glob("shot-001/attempt-01/audit-*/one/audit.json")).read_text()
    )
    assert audit["opening_source_frame_index"] == 0
    assert len(audit["opening_response_sha256"]) == 64
    assert (
        audit["status"] == {"closed": "machine_pass", "open": "fail", None: "needs_review"}[opening]
    )
    if opening == "closed":
        opening_response = next(
            production["root"].glob(
                "shot-001/attempt-01/audit-*/one/opening-state/model-response.txt"
            )
        )
        opening_response.write_text('{"observed_conditions":{"Box.lid":"open"}}')
        with pytest.raises(ValueError, match="opening assessment"):
            production["run"]()


@pytest.mark.parametrize("framing", ["unchanged", "changed", "uncertain", None])
def test_locked_camera_observations_override_general_pass_labels(
    production: dict[str, Any], framing: str | None
) -> None:
    plan = json.loads(production["plan"].read_text())
    plan["shots"][0]["camera_policy"] = "Locked frame"
    production["plan"].write_text(json.dumps(plan))
    original = production["kwargs"]["reviewer"]
    camera_calls = []

    def reviewer(content: list[dict[str, Any]]) -> str:
        if "Assess framing only" not in content[0]["text"]:
            response = original(content)
            assert isinstance(response, str)
            return response
        camera_calls.append(content)
        assert len([x for x in content if x["type"] == "image"]) == 2
        assert "Hold" not in content[0]["text"]
        assert "REFERENCE" not in json.dumps(content)
        indices = [x["text"].split()[-1] for x in content[1:] if x["type"] == "text"][1:]
        observations = {
            index: {"framing": "unchanged", "evidence": "Fixed background."} for index in indices
        }
        if framing is None:
            del observations[indices[-1]]
        else:
            observations[indices[-1]]["framing"] = framing
        return json.dumps({"observed_framing": observations})

    production["kwargs"]["reviewer"] = reviewer
    if framing == "unchanged":
        production["run"]()
    else:
        with pytest.raises(film_production.ProductionStopped):
            production["run"]()
        assert not (production["root"] / "shot-002").exists()
    assert len(camera_calls) >= 8
    for opening in {call[2]["image"] for call in camera_calls}:
        comparisons = {call[4]["image"] for call in camera_calls if call[2]["image"] == opening}
        assert len(comparisons) == 8
        assert Path(opening).name == "frame-01.png"
        assert all(Path(frame).parent == Path(opening).parent for frame in comparisons)
    root = next(production["root"].glob("shot-001/attempt-01/audit-*/one"))
    audit = json.loads((root / "audit.json").read_text())
    assert len(audit["camera_response_sha256"]) == 64
    assert (
        audit["status"]
        == {
            "unchanged": "machine_pass",
            "changed": "fail",
            "uncertain": "needs_review",
            None: "needs_review",
        }[framing]
    )
    if framing == "unchanged":
        (root / "camera/model-response.txt").write_text('{"observed_framing":{}}')
        with pytest.raises(ValueError, match="camera assessment"):
            production["run"]()


@pytest.mark.parametrize(
    "observation", ["no_visible_edit_artifact", "visible_edit_artifact", "uncertain", None]
)
def test_continuous_take_replays_artifact_evidence_over_general_pass_labels(
    production: dict[str, Any], observation: str | None
) -> None:
    plan = json.loads(production["plan"].read_text())
    plan["shots"][0]["camera_policy"] = "Single continuous shot"
    production["plan"].write_text(json.dumps(plan))
    original = production["kwargs"]["reviewer"]
    calls = []

    def reviewer(content: list[dict[str, Any]]) -> str:
        if "Inspect FRAME A and FRAME B" not in content[0]["text"]:
            response = original(content)
            assert isinstance(response, str)
            return response
        images = [Path(row["image"]) for row in content if row["type"] == "image"]
        assert len(images) == 2
        assert images[0].parent == images[1].parent
        assert int(images[1].stem[-2:]) == int(images[0].stem[-2:]) + 1
        assert "Hold" not in content[0]["text"]
        calls.append(content)
        if observation is None:
            return "{}"
        return json.dumps({"status": observation, "evidence": "Observed sampled image content."})

    production["kwargs"]["reviewer"] = reviewer
    if observation == "no_visible_edit_artifact":
        production["run"]()
    else:
        with pytest.raises(film_production.ProductionStopped):
            production["run"]()
        assert not (production["root"] / "shot-002").exists()
    assert len(calls) >= 8
    root = next(production["root"].glob("shot-001/attempt-01/audit-*/one"))
    audit = json.loads((root / "audit.json").read_text())
    assert len(audit["edit_artifacts_response_sha256"]) == 64
    assert (
        audit["status"]
        == {
            "no_visible_edit_artifact": "machine_pass",
            "visible_edit_artifact": "fail",
            "uncertain": "needs_review",
            None: "needs_review",
        }[observation]
    )
    if observation == "no_visible_edit_artifact":
        (root / "edit-artifacts/model-response.txt").write_text("[]")
        with pytest.raises(ValueError, match="edit artifact assessment"):
            production["run"]()


def test_missing_ending_input_fails_before_any_gpu_submission(
    production: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = production["settings"]
    settings["shots"][0]["ending_frame"] = "box.png"
    production["inputs"].write_text(json.dumps(settings))
    original = film_runner._json_request

    def request(server: str, path: str, payload: Any = None) -> dict[str, Any]:
        result = original(server, path, payload)
        if path == "/object_info":
            result["LoadImage"] = {}
        return result

    monkeypatch.setattr(film_runner, "_json_request", request)
    with pytest.raises(ValueError, match="support ending frames"):
        production["run"]()
    assert not production["submissions"]


def test_missing_scene_entity_input_fails_before_any_gpu_submission(
    production: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = film_runner._json_request

    def request(server: str, path: str, payload: Any = None) -> dict[str, Any]:
        result = original(server, path, payload)
        if path == "/object_info":
            del result["ComfyStory"]["input"]["optional"]["Scene entities"]
        return result

    monkeypatch.setattr(film_runner, "_json_request", request)
    with pytest.raises(ValueError, match="support declared scene entities"):
        production["run"]()
    assert not production["submissions"]


def test_missing_frame_profile_input_fails_before_gpu_submission(
    production: dict[str, Any],
) -> None:
    plan_path = production["plan"]
    plan = json.loads(plan_path.read_text())
    for shot in plan["shots"]:
        shot["composition"] = "Continue frame"
    plan_path.write_text(json.dumps(plan))
    path = production["inputs"]
    settings = json.loads(path.read_text())
    for row in settings["shots"]:
        row["render_profile"] = "Animate frame"
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="support render profiles"):
        production["run"]()
    assert not production["submissions"]


@pytest.mark.parametrize(
    ("production", "reason", "attempts"),
    [
        (
            "drawbox=x=0:y=0:w=iw:h=12:color=black:t=fill:enable='gte(t,0.2)'",
            "failed after 2 attempts",
            2,
        ),
        ("drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='gte(t,0.2)'", "uncertain", 1),
    ],
    indirect=["production"],
)
def test_pixel_border_results_reach_controller_without_report_mismatch(
    production: dict[str, Any], reason: str, attempts: int
) -> None:
    plan = json.loads(production["plan"].read_text())
    plan["shots"][0]["border_policy"] = "Preserve opening borders"
    production["plan"].write_text(json.dumps(plan))
    with pytest.raises(film_production.ProductionStopped, match=reason):
        production["run"]()
    assert [len(row) for row in production["submissions"]] == [1] * attempts
    state = json.loads((production["root"] / "production.json").read_text())
    assert not state["selected"]
    assert not (production["root"] / "shot-002").exists()
    if attempts == 2:
        prompt = production["submissions"][1]["10"]["inputs"]["What happens next?"]
        assert "match the opening frame's picture area and margins exactly" in prompt
        assert "black top/bottom/left/right fractions" not in prompt
        assert "black top/bottom/left/right fractions" in Path(state["last_assessment"]).read_text()


@pytest.mark.parametrize(
    ("count", "reason", "attempts"), [(0, "failed after 2 attempts", 2), (None, "uncertain", 1)]
)
def test_independent_ending_counts_reach_controller_without_report_mismatch(
    production: dict[str, Any], count: int | None, reason: str, attempts: int
) -> None:
    plan = json.loads(production["plan"].read_text())
    plan["shots"][0]["ending_counts"] = [{"category": "boxes", "count": 1}]
    production["plan"].write_text(json.dumps(plan))
    original = production["kwargs"]["reviewer"]

    def review(content: list[dict[str, Any]]) -> str:
        if "Inventory separate physical instances" in content[0]["text"]:
            return json.dumps(
                {
                    "observed_counts": {
                        "boxes": {
                            "count": count,
                            "instances": [],
                            "uncertainty": "Unknown" if count is None else "",
                        }
                    }
                }
            )
        return str(original(content))

    production["kwargs"]["reviewer"] = review
    with pytest.raises(film_production.ProductionStopped, match=reason):
        production["run"]()
    assert [len(row) for row in production["submissions"]] == [1] * attempts
    state = json.loads((production["root"] / "production.json").read_text())
    assert not state["selected"]
    assert not (production["root"] / "shot-002").exists()


@pytest.mark.parametrize("tamper", ["", "response", "frame_index", "frame_sha256"])
def test_count_evidence_is_replayed_and_bound_before_selection(
    production: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    plan = json.loads(production["plan"].read_text())
    plan["shots"][0]["ending_counts"] = [{"category": "boxes", "count": 1}]
    plan["shots"][0]["border_policy"] = "Preserve opening borders"
    production["plan"].write_text(json.dumps(plan))
    original = production["kwargs"]["reviewer"]

    def review(content: list[dict[str, Any]]) -> str:
        if "Inventory separate physical instances" in content[0]["text"]:
            return json.dumps(
                {
                    "observed_counts": {
                        "boxes": {"count": 1, "instances": ["Blue box"], "uncertainty": ""}
                    }
                }
            )
        return str(original(content))

    production["kwargs"]["reviewer"] = review
    original_report = film_production._report

    def report(*args: Any, **kwargs: Any) -> Path:
        path = original_report(*args, **kwargs)
        data = json.loads(path.read_text())
        if tamper == "response":
            response = path.parent / "one" / "ending-counts" / "model-response.txt"
            response.write_text(response.read_text() + " ")
        elif tamper == "frame_index":
            data["shots"][0]["ending_counts_source_frame_index"] = -1
        elif tamper == "frame_sha256":
            data["shots"][0]["ending_counts_source_frame_sha256"] = "0" * 64
        path.write_text(json.dumps(data))
        return path

    if tamper:
        monkeypatch.setattr(film_production, "_report", report)
        with pytest.raises(ValueError, match="ending count assessment or frame binding changed"):
            production["run"]()
        assert len(production["submissions"]) == 1
        return
    output = production["run"]()
    assert production["run"]() == output
    assert len(production["submissions"]) == 2


def _with_location_transition(production: dict[str, Any]) -> None:
    plan = json.loads(production["plan"].read_text())
    fact = {"key": "Box.place", "value": "shelf"}
    plan["initial_facts"] = [fact]
    for shot in plan["shots"]:
        shot["requires"] = [fact]
    plan["shots"][1]["effects"] = [{"key": "Box.place", "value": "floor"}]
    production["plan"].write_text(json.dumps(plan))
    original = production["kwargs"]["reviewer"]

    def review(content: list[dict[str, Any]]) -> str:
        if "Opening state alternatives:" in content[0]["text"]:
            return json.dumps({"observed_conditions": {"Box.place": "shelf"}})
        response = json.loads(original(content))
        if "source_frame_indices" in content[0]["text"]:
            data = json.loads(content[0]["text"].splitlines()[1])
            if data["required_conditions"]:
                response["observed_conditions"] = data["required_conditions"]
            response["observed_effects"] = data["intended_effects"]
        return json.dumps(response)

    production["kwargs"]["reviewer"] = review


@pytest.mark.parametrize(
    ("values", "reason", "submissions"),
    [
        (["floor", "shelf"], None, [1, 1, 2]),
        (["floor", "floor"], "failed after 2 attempts", [1, 1]),
        ([None], "uncertain", [1]),
    ],
)
def test_ending_state_catches_a_premature_transition_before_it_becomes_a_parent(
    production: dict[str, Any], values: list[str | None], reason: str | None, submissions: list[int]
) -> None:
    _with_location_transition(production)
    original = production["kwargs"]["reviewer"]
    observations = iter(values)
    end_calls = []

    def review(content: list[dict[str, Any]]) -> str:
        prompt = content[0]["text"]
        if "Observe CURRENT ENDING FRAME only" in prompt:
            choices = json.loads(prompt.splitlines()[1])["state_vocabularies"]
            # The first prefix contains no floor effect; alternatives come from the complete plan.
            assert choices == {"Box.place": ["floor", "shelf"]}
            assert "intended_effects" not in prompt
            assert "required_conditions" not in prompt
            images = [row["image"] for row in content if row["type"] == "image"]
            assert len(images) == 2  # Identity reference plus this candidate's final frame only.
            assert images[-1].endswith("frame-09.png")
            end_calls.append(images[-1])
            if Path(images[-1]).parent.name == "one":
                value = next(observations, "shelf")
                return json.dumps(
                    {"Box.place": {"value": value, "evidence": "Direct surface observation."}}
                )
        return str(original(content))

    production["kwargs"]["reviewer"] = review
    if reason:
        with pytest.raises(film_production.ProductionStopped, match=reason):
            production["run"]()
        assert not (production["root"] / "shot-002").exists()
    else:
        output = production["run"]()
        assert production["submissions"][1]["10"] == production["submissions"][2]["10"]
        assert production["submissions"][0]["10"] != production["submissions"][2]["10"]
        calls_before_resume = len(end_calls)
        assert production["run"]() == output
        assert len(end_calls) == calls_before_resume
    assert [len(row) for row in production["submissions"]] == submissions
    first = json.loads(
        (production["root"] / "shot-001/attempt-01/audit-001/one/audit.json").read_text()
    )
    assert first["ending_state_source_frame_index"] == 11
    assert any(
        "Independent ending state" in row["evidence"] and row["status"] != "pass"
        for row in first["findings"]
    )


@pytest.mark.parametrize("tamper", ["response", "frame_index", "frame_sha256", "vocabulary"])
def test_ending_state_observations_are_rehashed_before_selection(
    production: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    _with_location_transition(production)
    original = film_production._report

    def report(*args: Any, **kwargs: Any) -> Path:
        path = original(*args, **kwargs)
        data = json.loads(path.read_text())
        if tamper == "response":
            response = path.parent / "one/ending-state/model-response.txt"
            response.write_text(response.read_text() + " ")
        elif tamper == "frame_index":
            data["shots"][0]["ending_state_source_frame_index"] = 0
        elif tamper == "frame_sha256":
            data["shots"][0]["ending_state_source_frame_sha256"] = "0" * 64
        else:
            data["state_vocabularies_sha256"] = "0" * 64
        path.write_text(json.dumps(data))
        return path

    monkeypatch.setattr(film_production, "_report", report)
    with pytest.raises(
        ValueError, match=r"ending state assessment or frame binding changed|stale or not bound"
    ):
        production["run"]()
    assert len(production["submissions"]) == 1


def test_retry_targets_preserve_authored_endings_without_clipping_long_actions() -> None:
    action = "Begin " + "very slowly " * 200 + "then stop before the doorway."
    plan = FilmPlan(
        "retry",
        "A parcel",
        500,
        (),
        (
            FilmShot(
                "one",
                500,
                "Arrival",
                action,
                ("Parcel",),
                effects=(FilmFact("Parcel.location", "doorway"),),
            ),
        ),
    )
    note = film_production._repair(plan, 0)
    assert "Required ending: Parcel.location = doorway." in note
    assert "Begin" not in note
    assert len(note) <= 2000


def test_retry_targets_cannot_add_reference_mentions() -> None:
    plan = FilmPlan(
        "retry",
        "A parcel",
        500,
        (),
        (FilmShot("one", 500, "Arrival", "Move @Parcel\x00 to the doorway.", ("Parcel",)),),
    )
    note = film_production._repair(plan, 0)
    assert "Action to complete visibly: Move Parcel to the doorway." in note
    assert "@" not in note
    assert "\x00" not in note


def test_authored_state_meanings_reach_each_audit_stage_and_prefix(
    production: dict[str, Any],
) -> None:
    data = json.loads(production["plan"].read_text())
    data["initial_facts"] = [{"key": "Box.state", "value": "closed"}]
    data["shots"][0]["requires"] = [{"key": "Box.state", "value": "closed"}]
    data["shots"][0]["effects"] = [{"key": "Box.state", "value": "open"}]
    data["state_definitions"] = [
        {"key": "Box.state", "value": "closed", "definition": "The lid covers the entire opening."},
        {
            "key": "Box.state",
            "value": "open",
            "definition": "The lid is raised and the opening is visible.",
        },
        {
            "key": "Other.state",
            "value": "elsewhere",
            "definition": "Unrelated offscreen definition.",
        },
    ]
    production["plan"].write_text(json.dumps(data))
    original = production["kwargs"]["reviewer"]
    seen: set[str] = set()

    def review(content: list[dict[str, Any]]) -> str:
        prompt = content[0]["text"]
        if prompt.startswith("Inspect only SHOT FRAME 0"):
            seen.add("opening")
            assert "The lid covers the entire opening." in prompt
            assert "The lid is raised and the opening is visible." in prompt
            assert "Unrelated offscreen definition." not in prompt
            raw = original(content)
            assert isinstance(raw, str)
            return raw
        if prompt.startswith("Observe CURRENT ENDING FRAME only"):
            seen.add("ending")
            assert "The lid covers the entire opening." in prompt
            assert "The lid is raised and the opening is visible." in prompt
            assert "Unrelated offscreen definition." not in prompt
        raw = json.loads(original(content))
        if "source_frame_indices" in prompt:
            seen.add("shot")
            requested = json.loads(prompt.split("\n")[1])
            assert requested["state_definitions"] == {
                "Box.state": {x["value"]: x["definition"] for x in data["state_definitions"][:2]}
            }
            raw["observed_effects"] = requested["intended_effects"]
            if requested["required_conditions"]:
                raw["observed_conditions"] = requested["required_conditions"]
        return json.dumps(raw)

    production["kwargs"]["reviewer"] = review
    production["run"]()
    assert seen == {"opening", "ending", "shot"}
    first = json.loads((production["root"] / "shot-001/attempt-01/plan.json").read_text())
    assert first["state_definitions"] == data["state_definitions"]


def test_changed_state_meanings_in_a_report_cannot_advance_production(
    production: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = film_production._report

    def changed(*args: Any, **kwargs: Any) -> Path:
        report = original(*args, **kwargs)
        data = json.loads(report.read_text())
        data["state_definitions"] = {"Box.state": {"closed": "A different meaning."}}
        report.write_text(json.dumps(data))
        return report

    monkeypatch.setattr(film_production, "_report", changed)
    with pytest.raises(ValueError, match="stale or not bound"):
        production["run"]()
    assert len(production["submissions"]) == 1
    state = json.loads((production["root"] / "production.json").read_text())
    assert state["selected"] == []


@pytest.mark.parametrize("extent", ["partial", "uncertain"])
def test_occlusion_stops_full_visibility_controller_despite_pass_labels(
    production: dict[str, Any],
    extent: str,
) -> None:
    plan_path = production["plan"]
    plan = json.loads(plan_path.read_text())
    plan["shots"][0]["fully_visible_throughout"] = ["Box"]
    plan_path.write_text(json.dumps(plan))
    original = production["kwargs"]["reviewer"]
    observed_frames = []

    def contradictory_reviewer(content: list[dict[str, Any]]) -> str:
        prompt = content[0]["text"]
        if prompt.startswith("Describe actual visible extent"):
            assert "intended_action" not in prompt
            assert "Box opens" not in prompt
            data = json.loads(prompt.split("\n")[1])
            assert data["subjects"] == ["Box"]
            frame = data["frame_indices"][0]
            observed_frames.append(frame)
            return json.dumps(
                {
                    "observations": {
                        "Box": {
                            str(frame): {
                                "extent": extent,
                                "evidence": "Box is recognizable but hidden behind another object",
                            }
                        }
                    }
                }
            )
        response = original(content)
        assert isinstance(response, str)
        return response

    production["kwargs"]["reviewer"] = contradictory_reviewer
    reason = "failed after 2 attempts" if extent == "partial" else "uncertain"
    with pytest.raises(film_production.ProductionStopped, match=reason):
        production["run"]()
    count = 2 if extent == "partial" else 1
    assert [len(row) for row in production["submissions"]] == [1] * count
    assert observed_frames == [11] * count
    state = json.loads((production["root"] / "production.json").read_text())
    assert state["selected"] == []
    assert state["status"] == "needs_attention"
    assert state["approval_created"] is False
    assert not (production["root"] / "shot-002").exists()
    if extent == "partial":
        # Same-version restart authenticates and reuses saved blind observations.
        with pytest.raises(film_production.ProductionStopped, match=reason):
            production["run"]()
        assert len(production["submissions"]) == count
        assert len(observed_frames) == count


def test_full_visibility_checks_every_sample_and_authenticates_saved_observations(
    production: dict[str, Any],
) -> None:
    payload = json.loads(production["plan"].read_text())
    payload["shots"][0]["fully_visible_throughout"] = ["Box"]
    production["plan"].write_text(json.dumps(payload))
    original = production["kwargs"]["reviewer"]
    frames_seen: list[int] = []

    def reviewer(content: list[dict[str, Any]]) -> str:
        text = content[0]["text"]
        if text.startswith("Describe actual visible extent"):
            frame = json.loads(text.split("\n")[1])["frame_indices"][0]
            frames_seen.append(frame)
            return json.dumps(
                {
                    "observations": {
                        "Box": {
                            str(frame): {
                                "extent": "entire",
                                "evidence": "Box entirely in frame and unobscured",
                            }
                        }
                    }
                }
            )
        response = original(content)
        assert isinstance(response, str)
        return response

    production["kwargs"]["reviewer"] = reviewer
    production["run"]()
    assert frames_seen[0] == 11
    assert len(frames_seen) == 18  # Candidate audit and separate assembled-film audit.
    assert frames_seen[:9] == frames_seen[9:]
    assert len(set(frames_seen)) == 9
    production["run"]()
    assert len(frames_seen) == 18
    response = (
        production["root"] / "shot-001/attempt-01/audit-001/one/full-visibility/model-response.txt"
    )
    response.write_text("[]")
    with pytest.raises(ValueError, match="full visibility assessment changed"):
        production["run"]()
    assert len(frames_seen) == 18


def test_selected_state_binding_is_carried_and_recovered_without_new_sampling(
    production: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from comfy_story.film_selected_state import SelectedStateSource, ShotStateBinding
    from comfy_story.story_native_archive import NativeReferenceArchive

    production["settings"]["recall_selected_state"] = True
    production["inputs"].write_text(json.dumps(production["settings"]))
    production["kwargs"]["story_root"] = production["root"].parent / "archive"
    original = film_runner._json_request

    def request(server: str, path: str, payload: Any = None) -> dict[str, Any]:
        result = original(server, path, payload)
        if path == "/object_info":
            result["ComfyStory"]["input"]["optional"]["Shot state evidence"] = []
        return result

    seen = []

    def select(
        plan: FilmPlan,
        index: int,
        selected: tuple[SelectedStateSource, ...],
        archive: NativeReferenceArchive,
    ) -> tuple[ShotStateBinding, ...]:
        assert len(selected) == index
        assert all(s.audit.status == "machine_pass" for s in selected)
        seen.append(index)
        if not index:
            return ()
        return (
            ShotStateBinding(
                "box",
                "checked-box",
                "a" * 64,
                "one",
                selected[0].revision_sha256,
                selected[0].audit.video_sha256,
                selected[0].assessment_sha256,
                (("Box.lid", "closed"),),
            ),
        )

    monkeypatch.setattr(film_runner, "_json_request", request)
    monkeypatch.setattr(film_production, "select_film_state_evidence", select)
    production["run"]()
    assert seen == [0, 1]
    assert (
        production["submissions"][1]["20"]["inputs"]["Shot state evidence"]
        == '{"box":"checked-box"}'
    )
    assert "Shot state evidence" not in production["submissions"][0]["10"]["inputs"]
    proof = production["root"] / "shot-002/attempt-01/selected-state.json"
    saved = proof.read_bytes()
    assert json.loads(saved)["creator_approval"] is False
    production["run"]()
    assert len(production["submissions"]) == 2
    assert proof.read_bytes() == saved
    proof.write_text("{}")
    with pytest.raises(ValueError, match="recorded selected state evidence changed"):
        production["run"]()
    assert len(production["submissions"]) == 2


def test_selected_state_requires_host_archive_before_generation(production: dict[str, Any]) -> None:
    production["settings"]["recall_selected_state"] = True
    production["inputs"].write_text(json.dumps(production["settings"]))
    with pytest.raises(ValueError, match="native Story archive"):
        production["run"]()
    assert production["submissions"] == []


@pytest.mark.parametrize("observed", ["closed", None])
def test_preserved_state_cannot_be_reestablished_after_a_wrong_opening(
    production: dict[str, Any], observed: str | None
) -> None:
    plan = json.loads(production["plan"].read_text())
    plan["initial_facts"] = [{"key": "Box.lid", "value": "open"}]
    plan["shots"][0]["requires"] = [{"key": "Box.lid", "value": "open"}]
    plan["state_definitions"] = [
        {"key": "Box.lid", "value": "open", "definition": "The lid is raised."},
        {"key": "Box.lid", "value": "closed", "definition": "The lid is shut."},
    ]
    production["plan"].write_text(json.dumps(plan))
    original = production["kwargs"]["reviewer"]

    def reviewer(content: list[dict[str, Any]]) -> str:
        prompt = content[0]["text"]
        if "Opening state alternatives:" in prompt:
            assert '"Box.lid": ["closed", "open"]' in prompt
            return json.dumps(
                {"observed_conditions": {} if observed is None else {"Box.lid": observed}}
            )
        response = json.loads(original(content))
        if "source_frame_indices" in prompt:
            response["observed_conditions"] = {"Box.lid": "open"}
        return json.dumps(response)

    production["kwargs"]["reviewer"] = reviewer
    with pytest.raises(film_production.ProductionStopped):
        production["run"]()
    submissions = len(production["submissions"])
    assert submissions == (2 if observed == "closed" else 1)
    assert not (production["root"] / "shot-002").exists()
    with pytest.raises(film_production.ProductionStopped):
        production["run"]()
    assert len(production["submissions"]) == submissions
    result = json.loads((production["root"] / "production.json").read_text())
    assert result["selected"] == []


def test_staging_failure_stops_the_customer_runner_before_any_video(
    production: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = json.loads(production["plan"].read_text())
    plan["shots"][0]["composition"] = "Continue frame"
    production["plan"].write_text(json.dumps(plan))
    settings = production["settings"]
    settings["shots"][0].update(
        opening_prompt="Wide view of the box before it moves",
        opening_attempts=1,
        render_profile="Animate frame",
        intent="New Scene",
    )
    production["inputs"].write_text(json.dumps(settings))
    production["kwargs"].update(
        story_root=production["root"].parent / "archive", staging_identity={}
    )
    seen = []

    def failed_stage(*args: Any) -> str:
        seen.append(args[1])
        raise ValueError("opening stage failed its declared image budget; no video queued")

    monkeypatch.setattr(film_production, "validate_staging_configuration", lambda *_: None)
    monkeypatch.setattr(film_production, "stage_opening", failed_stage)
    with pytest.raises(ValueError, match="no video queued"):
        production["run"]()
    assert seen == [0]
    assert production["submissions"] == []


@pytest.mark.parametrize("mode", ["Compose", "Refine"])
def test_resolved_staging_input_survives_prefix_replay_and_video_retry(
    production: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    plan = json.loads(production["plan"].read_text())
    plan["shots"][0]["composition"] = "Continue frame"
    production["plan"].write_text(json.dumps(plan))
    settings = production["settings"]
    settings["shots"][0].update(
        opening_prompt="A wide opening view",
        opening_attempts=1,
        render_profile="Animate frame",
        intent="New Scene",
        world="box.png" if mode == "Refine" else None,
        opening_mode=mode,
    )
    production["inputs"].write_text(json.dumps(settings))
    production["kwargs"].update(
        story_root=production["root"].parent / "archive", staging_identity={}
    )
    stage_calls = []

    def stage(*args: Any) -> str:
        stage_calls.append(args[1])
        return "box.png"

    original_request = film_runner._json_request

    def request(server: str, path: str, payload: Any = None) -> dict[str, Any]:
        result = original_request(server, path, payload)
        if path == "/object_info":
            result["ComfyStory"]["input"].setdefault("optional", {})["Render profile"] = []
        return result

    monkeypatch.setattr(film_runner, "_json_request", request)
    monkeypatch.setattr(film_production, "validate_staging_configuration", lambda *_: None)
    monkeypatch.setattr(film_production, "stage_opening", stage)
    production["statuses"].extend(["fail", "pass"])
    production["run"]()
    assert stage_calls == [0]
    for path in production["root"].glob("shot-*/attempt-*/inputs.json"):
        first = json.loads(path.read_text())["shots"][0]
        assert first["world"] == "box.png"
        assert first["opening_prompt"] == ""
        assert "opening_mode" not in first
    before = len(production["submissions"])
    production["run"]()
    assert len(production["submissions"]) == before
    prior_stage_calls = list(stage_calls)
    monkeypatch.setattr(film_production, "STAGING_PROTOCOL", "changed-staging-protocol")
    with pytest.raises(ValueError, match="policy changed"):
        production["run"]()
    assert len(production["submissions"]) == before
    assert stage_calls == prior_stage_calls


def _with_narrative(production: dict[str, Any], status: str = "supported") -> list[str]:
    plan = json.loads(production["plan"].read_text())
    plan["narrative"] = {
        "goal": "Keep the box visible.",
        "obstacle": "The frame could hide the box.",
        "decision": "Hold the frame.",
        "outcome": "The box remains in place.",
    }
    production["plan"].write_text(json.dumps(plan))
    original = production["kwargs"]["reviewer"]
    events: list[str] = []

    def review(content: list[dict[str, Any]]) -> str:
        prompt = content[0]["text"]
        if prompt.startswith("Compare an intended story contract"):
            events.append("comparison")
            assert list(production["root"].rglob("blind-story-retell.txt"))
            assert all(part["type"] == "text" for part in content)
            return json.dumps(
                {
                    key: {
                        "status": status if key == "decision" else "supported",
                        "quote": "Hold the frame.",
                        "explanation": "Injected observation for controller testing only.",
                    }
                    for key in plan["narrative"]
                }
            )
        return str(original(content))

    production["kwargs"]["reviewer"] = review
    return events


def test_narrative_comparison_runs_after_blind_retell_and_resumes_without_rerender(
    production: dict[str, Any],
) -> None:
    events = _with_narrative(production)
    output = production["run"]()
    state = json.loads((production["root"] / "production.json").read_text())
    assert state["status"] == "ready_for_review"
    assert state["approval_created"] is False
    assert len(state["selected"]) == 2
    assert events == ["comparison"]
    count = len(production["calls"])
    assert production["run"]() == output
    assert len(production["calls"]) == count
    assert len(production["submissions"]) == 2
    assert events == ["comparison"]


@pytest.mark.parametrize("status", ["contradicted", "unestablished"])
def test_coherent_retelling_cannot_pass_missing_intent_or_reroll_its_judgment(
    production: dict[str, Any], status: str
) -> None:
    events = _with_narrative(production, status)
    for _ in range(2):
        with pytest.raises(film_production.ProductionStopped, match="intended narrative: decision"):
            production["run"]()
    assert events == ["comparison"]
    assert len(production["submissions"]) == 2
    state = json.loads((production["root"] / "production.json").read_text())
    assert state["status"] == "needs_attention"
    assert len(state["selected"]) == 2
    assert "film" not in state
    response = next(production["root"].rglob("narrative/model-response.txt"))
    assert json.loads(response.read_text())["decision"]["status"] == status


@pytest.mark.parametrize("filename", ["request.json", "model-response.txt", "report.json"])
@pytest.mark.parametrize("remove", [False, True])
def test_narrative_evidence_tampering_stops_resume_without_new_work(
    production: dict[str, Any], filename: str, remove: bool
) -> None:
    events = _with_narrative(production)
    production["run"]()
    evidence = next(production["root"].rglob("narrative/" + filename))
    if remove:
        evidence.unlink()
    else:
        evidence.write_text(evidence.read_text() + " ")
    with pytest.raises(ValueError, match="narrative"):
        production["run"]()
    assert events == ["comparison"]
    assert len(production["submissions"]) == 2


def test_negative_blind_retelling_is_not_reinterpreted_using_intent(
    production: dict[str, Any],
) -> None:
    events = _with_narrative(production)
    production["whole"]["contradictions"] = ["Missing cause."]
    with pytest.raises(film_production.ProductionStopped, match="retelling is inconsistent"):
        production["run"]()
    assert events == []


def test_authored_preflight_and_render_share_the_fitted_frame(
    production: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = json.loads(production["plan"].read_text())
    plan["initial_facts"] = [{"key": "Box.lid", "value": "closed"}]
    production["plan"].write_text(json.dumps(plan))
    root = production["inputs"].parent
    Image.new("RGB", (300, 800), "blue").save(root / "box.png")
    Image.new("RGB", (1344, 768), "blue").save(root / "fitted.png")
    fitted_calls = []

    def fit(server: str, directory: Path, input_root: Path, world: str) -> str:
        assert world == "box.png"
        assert input_root == root
        fitted_calls.append(directory)
        return "fitted.png"

    monkeypatch.setattr(film_production, "fit_authored_opening", fit)
    original = production["kwargs"]["reviewer"]

    def review(content: list[dict[str, Any]]) -> str:
        raw = json.loads(original(content))
        if "source_frame_indices" in content[0]["text"]:
            requested = json.loads(content[0]["text"].split("\n")[1])
            if requested["required_conditions"]:
                raw["observed_conditions"] = requested["required_conditions"]
        return json.dumps(raw)

    production["kwargs"]["reviewer"] = review
    production["run"]()
    starting = next(
        call
        for call in production["calls"]
        if "Observe CURRENT STARTING FRAME only" in call[0]["text"]
    )
    assert starting[-1]["image"] == str(root / "fitted.png")
    assert any(row.get("image") == str(root / "box.png") for row in starting)
    for graph in production["submissions"]:
        assert graph["10"]["inputs"]["World / starting frame"] == "fitted.png"
    assert len(fitted_calls) == 1
    assert json.loads(production["inputs"].read_text())["shots"][0]["world"] == "box.png"
