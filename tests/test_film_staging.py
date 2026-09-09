"""Exercise budgets, blind opening checks, and exact stage recovery without GPU claims."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from comfy_story import film_staging
from comfy_story.film_plan import FilmFact, FilmPlan, FilmShot
from comfy_story.film_staging_workflow import (
    STAGING_MODELS,
    STAGING_PROTOCOL,
    compile_opening_stage,
    opening_stage_prompt,
)
from comfy_story.film_workflow import FilmShotInput, compile_film_workflow
from comfy_story.story_video_canvas import H3_VIDEO_HEIGHT, H3_VIDEO_WIDTH


@pytest.fixture
def scene(tmp_path: Path) -> tuple[FilmPlan, dict[str, Any]]:
    Image.new("RGB", (64, 32), "blue").save(tmp_path / "mug.png")
    plan = FilmPlan(
        "stage-test",
        "A change",
        5000,
        (FilmFact("mug.fill", "empty"),),
        (
            FilmShot(
                "one",
                5000,
                "Fill",
                "Pour tea into the mug",
                ("Mug",),
                requires=(FilmFact("mug.fill", "empty"),),
                effects=(FilmFact("mug.fill", "full"),),
                visible_throughout=("Mug",),
                fully_visible_throughout=("Mug",),
                composition="Continue frame",
            ),
        ),
    )
    settings: dict[str, Any] = {
        "library": {
            "project_name": "Stage",
            "references": [{"name": "Mug", "file": "mug.png", "role": "prop"}],
        },
        "shots": [
            {
                "world": None,
                "variation": 72,
                "render_profile": "Animate frame",
                "intent": "New Scene",
                "sampler": "Turbo 8-step",
                "opening_prompt": "A wide view of the mug on a table",
                "opening_attempts": 2,
            }
        ],
    }
    return plan, settings


def test_opening_prompt_excludes_future_action_and_effect(
    scene: tuple[FilmPlan, dict[str, Any]],
) -> None:
    plan, _ = scene
    prompt = opening_stage_prompt(plan, 0, "A wide view", ("Mug identity",))
    assert "mug.fill = empty" in prompt
    assert "Pour tea" not in prompt
    assert "mug.fill = full" not in prompt
    assert "inside the picture" in prompt


def test_unresolved_staging_cannot_be_queued(scene: tuple[FilmPlan, dict[str, Any]]) -> None:
    plan, settings = scene
    inputs = tuple(FilmShotInput(**s) for s in settings["shots"])
    with pytest.raises(ValueError, match="verified film production"):
        compile_film_workflow(plan, settings["library"], inputs)
    graph = compile_film_workflow(plan, settings["library"], inputs, allow_pending_staging=True)
    assert "10" in graph
    resolved = replace(inputs[0], opening_prompt="", world="mug.png")
    compile_film_workflow(plan, settings["library"], (resolved,))


@pytest.mark.parametrize("budget", [0, 5, True, 1.5])
def test_invalid_budget_rejected_before_submission(
    scene: tuple[FilmPlan, dict[str, Any]], budget: Any
) -> None:
    plan, settings = scene
    settings["shots"][0]["opening_attempts"] = budget
    with pytest.raises(ValueError, match="opening attempts"):
        compile_film_workflow(
            plan,
            settings["library"],
            (FilmShotInput(**settings["shots"][0]),),
            allow_pending_staging=True,
        )


def test_stage_workflow_keeps_three_distinct_inputs() -> None:
    workflow = compile_opening_stage("Open scene", ("scene.png", "a.png", "b.png"), 12)
    positive: Any = workflow["7"]
    assert positive["inputs"]["image2"] == ["102", 0]
    assert positive["inputs"]["image3"] == ["103", 0]
    sampler: Any = workflow["21"]
    assert sampler["inputs"]["steps"] == 40
    assert sampler["inputs"]["seed"] == 12
    fitted: Any = workflow["26"]
    assert fitted == {
        "class_type": "ImageScale",
        "inputs": {
            "image": ["22", 0],
            "upscale_method": "bilinear",
            "width": H3_VIDEO_WIDTH,
            "height": H3_VIDEO_HEIGHT,
            "crop": "center",
        },
    }
    saved: Any = workflow["24"]
    raw: Any = workflow["25"]
    assert saved["inputs"]["images"] == ["26", 0]
    assert raw["inputs"]["images"] == ["22", 0]
    with pytest.raises(ValueError, match="one to three"):
        compile_opening_stage("Open", ("1.png", "2.png", "3.png", "4.png"), 12)


def _invoke(
    plan: FilmPlan,
    settings: dict[str, Any],
    root: Path,
    reviewer: Any,
) -> str:
    return film_staging.stage_opening(
        plan,
        0,
        settings,
        root,
        root / "stage",
        "http://localhost:8188",
        reviewer,
        "a" * 64,
        {"model_files": dict.fromkeys(STAGING_MODELS, "b" * 64)},
        (),
        None,
        lambda: None,
        lambda _phase, _attempt: None,
    )


def _response(
    content: list[dict[str, Any]], *, state: str = "empty", extent: str = "entire"
) -> str:
    if "visible extent" in content[0]["text"]:
        return json.dumps(
            {"observations": {"Mug": {"0": {"extent": extent, "evidence": "Edges visible"}}}}
        )
    return json.dumps({"mug.fill": {"value": state, "evidence": "The interior is visible"}})


@pytest.mark.parametrize(("extent", "expected_attempts"), [("partial", 2), ("uncertain", 1)])
def test_stage_failures_respect_budget(
    scene: tuple[FilmPlan, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extent: str,
    expected_attempts: int,
) -> None:
    plan, settings = scene
    calls: list[object] = []

    def render(*args: Any) -> str:
        calls.append(args)
        return "mug.png"

    monkeypatch.setattr(film_staging, "_render", render)
    with pytest.raises(ValueError, match="no video queued"):
        _invoke(plan, settings, tmp_path, lambda c: _response(c, extent=extent))
    assert len(calls) == expected_attempts
    assert (tmp_path / "stage/attempt-01/assessment.json").exists()


def test_passing_stage_replays_observations_without_reroll(
    scene: tuple[FilmPlan, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, settings = scene
    monkeypatch.setattr(film_staging, "_render", lambda *args: "mug.png")
    assert _invoke(plan, settings, tmp_path, _response) == "mug.png"

    def unexpected(_content: Any) -> str:
        pytest.fail("resume must use the original raw observation")

    assert _invoke(plan, settings, tmp_path, unexpected) == "mug.png"
    path = tmp_path / "stage/attempt-01/visibility/response.txt"
    path.write_text(_response([{"text": "visible extent"}], extent="partial"))
    with pytest.raises(ValueError, match="assessment changed"):
        _invoke(plan, settings, tmp_path, unexpected)


def test_partial_visibility_is_allowed_when_full_visibility_not_required(
    scene: tuple[FilmPlan, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, settings = scene
    plan = replace(plan, shots=(replace(plan.shots[0], fully_visible_throughout=()),))
    monkeypatch.setattr(film_staging, "_render", lambda *args: "mug.png")
    assert _invoke(plan, settings, tmp_path, lambda c: _response(c, extent="partial")) == "mug.png"


@pytest.mark.parametrize(
    ("extent", "expected_attempts"),
    [("entire", 2), ("partial", 2), ("uncertain", 1), (None, 1)],
)
def test_excluded_opening_subject_cannot_pass_visibility(
    scene: tuple[FilmPlan, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extent: str | None,
    expected_attempts: int,
) -> None:
    plan, settings = scene
    plan = replace(plan, shots=(replace(plan.shots[0], absent=("Crate",)),))
    Image.new("RGB", (64, 32), "red").save(tmp_path / "crate.png")
    settings["library"]["references"].append({"name": "Crate", "file": "crate.png", "role": "prop"})
    renders: list[object] = []

    def render(*args: Any) -> str:
        renders.append(args)
        return "mug.png"

    def review(content: list[dict[str, Any]]) -> str:
        if "visible extent" not in content[0]["text"]:
            return _response(content)
        assert str(tmp_path / "crate.png") in [c.get("image") for c in content]
        assert "Exclude these entities" not in content[0]["text"]
        assert "expected" not in content[0]["text"]
        observations = json.loads(_response(content))["observations"]
        if extent is not None:
            observations["Crate"] = {"0": {"extent": extent, "evidence": "Edge of a crate"}}
        return json.dumps({"observations": observations})

    monkeypatch.setattr(film_staging, "_render", render)
    with pytest.raises(ValueError, match="no video queued"):
        _invoke(plan, settings, tmp_path, review)
    assert len(renders) == expected_attempts
    assessment = json.loads((tmp_path / "stage/attempt-01/assessment.json").read_text())
    assert assessment["visibility"]["requirements"] == {"Crate": "absent", "Mug": "entire"}
    assert assessment["approval_created"] is False


def test_absence_only_category_is_checked_and_recovers_without_new_observation(
    scene: tuple[FilmPlan, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, settings = scene
    plan = replace(
        plan,
        shots=(
            replace(
                plan.shots[0],
                visible_throughout=(),
                fully_visible_throughout=(),
                absent=("vehicles",),
            ),
        ),
    )
    monkeypatch.setattr(film_staging, "_render", lambda *args: "mug.png")
    calls = 0

    def review(content: list[dict[str, Any]]) -> str:
        nonlocal calls
        calls += 1
        if "visible extent" not in content[0]["text"]:
            return _response(content)
        assert "literal named entity or category" in content[0]["text"]
        return json.dumps(
            {"observations": {"vehicles": {"0": {"extent": "absent", "evidence": "No vehicles"}}}}
        )

    assert _invoke(plan, settings, tmp_path, review) == "mug.png"
    assert calls == 2

    def unexpected(_content: Any) -> str:
        pytest.fail("a recovered absence assessment must not reroll its observation")

    assert _invoke(plan, settings, tmp_path, unexpected) == "mug.png"
    response = tmp_path / "stage/attempt-01/visibility/response.txt"
    response.write_text(
        json.dumps(
            {"observations": {"vehicles": {"0": {"extent": "partial", "evidence": "A wheel"}}}}
        )
    )
    with pytest.raises(ValueError, match="assessment changed"):
        _invoke(plan, settings, tmp_path, unexpected)


def test_unknown_queue_outcome_never_submits_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "stage"
    directory.mkdir()
    (directory / "submission-intent.json").write_text("{}")

    def unexpected(*args: Any) -> dict[str, Any]:
        pytest.fail("ambiguous submission must not call Comfy")

    monkeypatch.setattr(film_staging, "_json_request", unexpected)
    with pytest.raises(ValueError, match="outcome unknown"):
        film_staging._render("http://localhost:8188", directory, {}, {}, tmp_path)


def test_completed_image_recovers_with_no_queue_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "stage"
    directory.mkdir()
    request = {"format": STAGING_PROTOCOL, "workflow": {}}
    from comfy_story.story_contracts import canonical_story_json

    encoded = canonical_story_json(request)
    (directory / "request.json").write_bytes(encoded)
    (directory / "history.json").write_bytes(b"history")
    (tmp_path / "image.png").write_bytes(b"original")
    (tmp_path / "raw.png").write_bytes(b"raw")
    (directory / "result.json").write_text(
        json.dumps(
            {
                "file": "image.png",
                "history_sha256": hashlib.sha256(b"history").hexdigest(),
                "sha256": hashlib.sha256(b"original").hexdigest(),
                "raw_file": "raw.png",
                "raw_sha256": hashlib.sha256(b"raw").hexdigest(),
                "request_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    )

    def unexpected(*args: Any) -> dict[str, Any]:
        pytest.fail("completed stage must not call Comfy")

    monkeypatch.setattr(film_staging, "_json_request", unexpected)
    assert film_staging._render("http://localhost:8188", directory, {}, {}, tmp_path) == "image.png"
    (tmp_path / "image.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        film_staging._render("http://localhost:8188", directory, {}, {}, tmp_path)
    (tmp_path / "image.png").write_bytes(b"original")
    (tmp_path / "raw.png").write_bytes(b"changed raw")
    with pytest.raises(ValueError, match="raw opening image changed"):
        film_staging._render("http://localhost:8188", directory, {}, {}, tmp_path)


def _image_bytes(size: tuple[int, int], color: str) -> bytes:
    data = io.BytesIO()
    Image.new("RGB", size, color).save(data, format="PNG")
    return data.getvalue()


def _mock_image_job(
    monkeypatch: pytest.MonkeyPatch, raw: bytes, fitted: bytes
) -> list[dict[str, Any]]:
    submitted: list[dict[str, Any]] = []

    def request(_server: str, path: str, payload: Any = None) -> dict[str, Any]:
        if path == "/object_info":
            return {
                node["class_type"]: {}
                for node in compile_opening_stage("test", ("ref.png",), 0).values()
                if isinstance(node, dict)
            }
        if path == "/prompt":
            submitted.append(payload["prompt"])
            return {"prompt_id": str(len(submitted))}
        job_id = path.removeprefix("/history/")
        return {
            job_id: {
                "status": {"status_str": "success"},
                "prompt": [0, job_id, submitted[int(job_id) - 1]],
                "outputs": {
                    node: {"images": [{"filename": name, "subfolder": "", "type": "output"}]}
                    for node, name in (("25", "raw.png"), ("24", "fitted.png"))
                },
            }
        }

    def fetch(url: str, **_kwargs: Any) -> io.BytesIO:
        return io.BytesIO(raw if "filename=raw.png" in url else fitted)

    monkeypatch.setattr(film_staging, "_json_request", request)
    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    return submitted


def test_opening_checks_use_fitted_pixels_and_preserve_raw_source(
    scene: tuple[FilmPlan, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, settings = scene
    settings["shots"][0]["opening_attempts"] = 1
    raw = _image_bytes((64, 128), "green")
    fitted = _image_bytes((H3_VIDEO_WIDTH, H3_VIDEO_HEIGHT), "blue")
    submitted = _mock_image_job(monkeypatch, raw, fitted)
    observed: list[bytes] = []

    def review(content: list[dict[str, Any]]) -> str:
        frame = Path(content[-1]["image"]).read_bytes()
        observed.append(frame)
        # A supplied fitted view no longer contains the green subject in the raw view.
        # Its absence must stop staging; the raw image cannot satisfy visibility instead.
        return _response(content, extent="absent" if frame == fitted else "entire")

    with pytest.raises(ValueError, match="declared image budget; no video queued"):
        _invoke(plan, settings, tmp_path, review)
    assert len(submitted) == 1
    assert observed == [fitted, fitted]
    receipt = json.loads((tmp_path / "stage/attempt-01/result.json").read_text())
    assert (tmp_path / receipt["raw_file"]).read_bytes() == raw
    assert (tmp_path / receipt["file"]).read_bytes() == fitted
    assert receipt["raw_sha256"] == hashlib.sha256(raw).hexdigest()


def test_wrong_canvas_cannot_become_an_assessed_opening(
    scene: tuple[FilmPlan, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, settings = scene
    image = _image_bytes((64, 128), "blue")
    submitted = _mock_image_job(monkeypatch, image, image)

    def unexpected(_content: Any) -> str:
        pytest.fail("an unfitted output must not reach the reviewer")

    with pytest.raises(ValueError, match="fitted H3 video canvas"):
        _invoke(plan, settings, tmp_path, unexpected)
    assert len(submitted) == 1


def test_staging_node_preflight_rejects_missing_capability_before_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def request(server: str, path: str) -> dict[str, Any]:
        calls.append(path)
        assert path == "/object_info"
        return {"DuetStory": {}}

    monkeypatch.setattr(film_staging, "_json_request", request)
    with pytest.raises(ValueError, match="missing opening staging nodes"):
        film_staging.validate_staging_configuration(
            "http://localhost:8188", {"model_files": dict.fromkeys(STAGING_MODELS, "a" * 64)}
        )
    assert calls == ["/object_info"]
    with pytest.raises(ValueError, match="exact model file"):
        film_staging.validate_staging_configuration("http://localhost:8188", {"model_files": {}})
    assert len(calls) == 1


def test_shared_scene_and_subject_pixels_are_bound_once(
    scene: tuple[FilmPlan, dict[str, Any]], tmp_path: Path
) -> None:
    plan, settings = scene
    settings["shots"][0]["world"] = "mug.png"
    names, roles = film_staging.stage_references(plan, 0, settings, tmp_path, (), None)
    assert names == ("mug.png",)
    assert "scene" in roles[0]
    assert "Mug" in roles[0]


def test_previous_scene_cannot_be_used_for_first_shot_or_with_uploaded_world(
    scene: tuple[FilmPlan, dict[str, Any]],
) -> None:
    plan, settings = scene
    settings["shots"][0]["opening_use_previous_scene"] = True
    with pytest.raises(ValueError, match="requires an earlier shot"):
        compile_film_workflow(
            plan,
            settings["library"],
            (FilmShotInput(**settings["shots"][0]),),
            allow_pending_staging=True,
        )


def test_authored_canvas_uses_native_crop_without_diffusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Image.new("RGB", (300, 800), "blue").save(tmp_path / "portrait.png")
    source_sha = hashlib.sha256((tmp_path / "portrait.png").read_bytes()).hexdigest()
    calls = []

    def render(
        server: str, directory: Path, graph: dict[str, Any], identity: dict[str, Any], root: Path
    ) -> str:
        calls.append(graph)
        assert {n["class_type"] for n in graph.values()} == {"LoadImage", "SaveImage", "ImageScale"}
        assert graph["26"]["inputs"] == {
            "image": ["101", 0],
            "width": 1344,
            "height": 768,
            "upscale_method": "bilinear",
            "crop": "center",
        }
        assert identity["references"] == {"portrait.png": source_sha}
        assert graph["25"]["inputs"]["images"] == ["101", 0]
        assert graph["24"]["inputs"]["images"] == ["26", 0]
        return "fitted.png"

    monkeypatch.setattr(film_staging, "_render", render)
    assert (
        film_staging.fit_authored_opening(
            "http://localhost:8188", tmp_path / "run", tmp_path, "portrait.png"
        )
        == "fitted.png"
    )
    assert len(calls) == 1
    assert hashlib.sha256((tmp_path / "portrait.png").read_bytes()).hexdigest() == source_sha
    Image.new("RGB", (1344, 768), "blue").save(tmp_path / "ready.png")
    assert (
        film_staging.fit_authored_opening(
            "http://localhost:8188", tmp_path / "other", tmp_path, "ready.png"
        )
        == "ready.png"
    )
    assert len(calls) == 1


def test_authored_canvas_refuses_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "portrait.png"
    Image.new("RGB", (300, 800), "blue").save(source)

    def render(*args: Any) -> str:
        Image.new("RGB", (300, 800), "red").save(source)
        return "fitted.png"

    monkeypatch.setattr(film_staging, "_render", render)
    with pytest.raises(ValueError, match="changed during canvas preparation"):
        film_staging.fit_authored_opening(
            "http://localhost:8188", tmp_path / "run", tmp_path, "portrait.png"
        )


@pytest.mark.parametrize("mode", ["", "Unknown", None, True])
def test_unknown_opening_modes_fail_before_rendering(
    scene: tuple[FilmPlan, dict[str, Any]], mode: Any
) -> None:
    plan, settings = scene
    row = FilmShotInput(**{**settings["shots"][0], "opening_mode": mode})
    with pytest.raises(ValueError, match="opening mode"):
        compile_film_workflow(plan, settings["library"], (row,), allow_pending_staging=True)
    with pytest.raises(ValueError, match="opening mode"):
        compile_opening_stage("Opening", ("mug.png",), 72, mode)


def test_refine_requires_an_existing_composition(scene: tuple[FilmPlan, dict[str, Any]]) -> None:
    plan, settings = scene
    row = FilmShotInput(**{**settings["shots"][0], "opening_mode": "Refine"})
    with pytest.raises(ValueError, match="existing scene image"):
        compile_film_workflow(plan, settings["library"], (row,), allow_pending_staging=True)
    compile_film_workflow(
        plan, settings["library"], (replace(row, world="mug.png"),), allow_pending_staging=True
    )
    with pytest.raises(ValueError, match="requires staging"):
        compile_film_workflow(
            plan, settings["library"], (replace(row, world="mug.png", opening_prompt=""),)
        )


def test_refine_profile_preserves_budget_state_checks_and_reference_roles(
    scene: tuple[FilmPlan, dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, settings = scene
    settings["shots"][0].update(world="mug.png", opening_mode="Refine")
    calls = []

    def render(*args: Any) -> str:
        graph = args[2]
        calls.append(graph)
        assert graph["21"]["inputs"]["denoise"] == 0.35
        assert graph["21"]["inputs"]["steps"] == 40
        assert graph["21"]["inputs"]["seed"] == 72 + len(calls) - 1
        prompt = graph["7"]["inputs"]["prompt"]
        assert "Image 1 fixes the layout" in prompt
        assert "mug.fill = empty" in prompt
        assert "mug.fill = full" not in prompt
        assert "original identity reference" in prompt
        return "mug.png"

    monkeypatch.setattr(film_staging, "_render", render)
    with pytest.raises(ValueError, match="declared image budget"):
        _invoke(plan, settings, tmp_path, lambda content: _response(content, state="full"))
    assert len(calls) == 2
    assert (tmp_path / "stage/attempt-02/assessment.json").exists()
