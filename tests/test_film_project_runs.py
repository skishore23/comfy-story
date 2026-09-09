from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import pytest

from comfy_story import film_project_runs
from comfy_story.film_plan import FilmPlan, FilmShot
from comfy_story.film_production import ProductionPaused
from comfy_story.film_project import FilmProjectStore
from comfy_story.film_project_runs import FilmProjectRunner
from comfy_story.film_runner import _write


@pytest.mark.parametrize(
    ("selected_ids", "expected_count", "expected_ms"),
    [
        ([], 0, 0),
        (["one"], 1, 10000),
        (["two"], 0, 0),
        (["one", "one"], 1, 10000),
        (["one", "two"], 2, 15000),
    ],
)
def test_coverage_uses_frozen_recipe_and_only_contiguous_selected_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_ids: list[str],
    expected_count: int,
    expected_ms: int,
) -> None:
    runner, first_revision = _runner(tmp_path, monkeypatch)
    plan = FilmPlan(
        "project",
        "Two events",
        15000,
        (),
        (FilmShot("one", 10000, "First", "Move", ()), FilmShot("two", 5000, "Next", "Hold", ())),
    )
    inputs = {
        "library": {"project_name": "Film", "references": []},
        "shots": [
            {"world": "world.png", "variation": 42},
            {"world": None, "variation": 43, "intent": "Next Shot"},
        ],
    }
    saved = runner.store.save("project", plan, inputs, expected_revision=first_revision)

    def stopped(p: Path, i: Path, server: str, directory: Path, **kwargs: Any) -> Path:
        _write(
            directory / "production.json",
            {
                "status": "needs_attention",
                "selected": [{"shot_id": shot_id} for shot_id in selected_ids],
                "preview": {"shot_id": "two", "attempt": 2, "duration_ms": 15000},
            },
        )
        raise ValueError("current candidate failed")

    monkeypatch.setattr(film_project_runs, "run_production", stopped)
    try:
        state = runner.start("project", saved["revision"])
        assert runner._active is not None
        runner._active.result(timeout=5)
        edited = replace(
            plan,
            target_duration_ms=20000,
            shots=(plan.shots[0], replace(plan.shots[1], duration_ms=10000)),
        )
        runner.store.save("project", edited, inputs, expected_revision=saved["revision"])
        directory = runner._run_path("project", state["run_id"])
        before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
        status = runner.status("project", state["run_id"])
        assert status["coverage"] == {
            "target_duration_ms": 15000,
            "planned_duration_ms": 15000,
            "planned_shots": 2,
            "selected_duration_ms": expected_ms,
            "selected_shots": expected_count,
        }
        assert status["preview"]["duration_ms"] == 15000
        assert status["approval_created"] is False
        assert before == {
            path: path.read_bytes() for path in directory.rglob("*") if path.is_file()
        }
    finally:
        runner._executor.shutdown(wait=True)


def _runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FilmProjectRunner, str]:
    store = FilmProjectStore(tmp_path / "store")
    plan = FilmPlan("project", "Film", 5000, (), (FilmShot("one", 5000, "Show", "Hold", ()),))
    saved = store.save(
        "project",
        plan,
        {
            "library": {"project_name": "Film", "references": []},
            "shots": [{"world": "world.png", "variation": 42}],
        },
        expected_revision=None,
    )
    monkeypatch.setattr(film_project_runs, "_model_identity", lambda _: "a" * 64)
    return FilmProjectRunner(
        store, "http://localhost:8188", tmp_path / "input", tmp_path / "model"
    ), saved["revision"]


def test_background_run_is_idempotent_and_pause_is_project_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, revision = _runner(tmp_path, monkeypatch)
    entered, finish = threading.Event(), threading.Event()
    calls = []

    def production(plan: Path, inputs: Path, server: str, directory: Path, **kwargs: Any) -> Path:
        calls.append((plan, inputs, server, kwargs))
        entered.set()
        assert finish.wait(5)
        assert (directory / "pause.requested").exists()
        raise ProductionPaused("paused")

    monkeypatch.setattr(film_project_runs, "run_production", production)
    try:
        first = runner.start("project", revision)
        assert entered.wait(5)
        assert runner.status("project", first["run_id"])["phase"] == "checking_inputs"
        assert runner.start("project", revision)["run_id"] == first["run_id"]
        assert len(calls) == 1
        with pytest.raises(ValueError, match="another film"):
            runner.start("project", revision, 3)
        paused = runner.pause("project", first["run_id"])
        assert paused["pause_requested"] is True
        assert paused["status"] == "running"
        finish.set()
        assert runner._active is not None
        runner._active.result(timeout=5)
        assert runner.status("project", first["run_id"])["status"] == "paused"
    finally:
        finish.set()
        runner._executor.shutdown(wait=True)


def test_worker_failure_survives_a_new_browser_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, revision = _runner(tmp_path, monkeypatch)

    def fail(*args: Any, **kwargs: Any) -> Path:
        raise ValueError("missing reference image")

    monkeypatch.setattr(film_project_runs, "run_production", fail)
    state = runner.start("project", revision)
    assert runner._active is not None
    runner._active.result(timeout=5)
    runner._executor.shutdown(wait=True)
    reopened = FilmProjectRunner(runner.store, runner.server, runner.comfy_input, runner.model)
    try:
        state = reopened.status("project", state["run_id"])
        assert state["status"] == "needs_attention"
        assert state["reason"] == "missing reference image"
        assert state["phase"] == "checking_inputs"
        assert state["approval_created"] is False
    finally:
        reopened._executor.shutdown(wait=True)


def test_export_must_be_inside_run_and_match_recorded_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, revision = _runner(tmp_path, monkeypatch)

    def render(plan: Path, inputs: Path, server: str, directory: Path, **kwargs: Any) -> Path:
        video = directory / "film.mp4"
        video.write_bytes(b"fixture video")
        _write(
            directory / "production.json",
            {
                "status": "ready_for_review",
                "film": str(video),
                "film_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            },
        )
        return video

    monkeypatch.setattr(film_project_runs, "run_production", render)
    try:
        state = runner.start("project", revision)
        assert runner._active is not None
        runner._active.result(timeout=5)
        video = runner.video("project", state["run_id"])
        video.write_bytes(b"modified")
        with pytest.raises(ValueError, match="integrity"):
            runner.video("project", state["run_id"])
    finally:
        runner._executor.shutdown(wait=True)


def test_unconfigured_verifier_never_submits_a_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, revision = _runner(tmp_path, monkeypatch)
    runner.model = None
    try:
        with pytest.raises(ValueError, match="VERIFY_MODEL"):
            runner.start("project", revision)
        assert runner.runs("project") == []
    finally:
        runner._executor.shutdown(wait=True)


@pytest.mark.parametrize("tamper", [None, "bytes", "outside", "symlink", "record"])
def test_stopped_preview_is_scoped_verified_and_never_approves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str | None
) -> None:
    runner, revision = _runner(tmp_path, monkeypatch)
    directories = []

    def stopped(plan: Path, inputs: Path, server: str, directory: Path, **kwargs: Any) -> Path:
        directories.append(directory)
        video = directory / "candidate.mp4"
        video.write_bytes(b"candidate with chosen soundtrack")
        _write(
            directory / "production.json",
            {
                "status": "needs_attention",
                "selected": [],
                "approval_created": False,
                "preview": {
                    "film": str(video),
                    "film_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                    "shot_id": "one",
                    "attempt": 1,
                    "duration_ms": 5000,
                },
            },
        )
        raise ValueError("candidate failed checking")

    monkeypatch.setattr(film_project_runs, "run_production", stopped)
    try:
        state = runner.start("project", revision)
        assert runner._active is not None
        runner._active.result(timeout=5)
        key = state["run_id"]
        directory = directories[0]
        record_path = directory / "production.json"
        record = json.loads(record_path.read_text())
        video = directory / "candidate.mp4"
        if tamper == "bytes":
            video.write_bytes(b"changed")
        elif tamper in ("outside", "symlink"):
            outside = tmp_path / "unrelated.mp4"
            outside.write_bytes(video.read_bytes())
            if tamper == "outside":
                record["preview"]["film"] = str(outside)
            else:
                video.unlink()
                video.symlink_to(outside)
        elif tamper == "record":
            del record["preview"]["film"]
        _write(record_path, record)
        if tamper is None:
            assert runner.preview_video("project", key) == video
        else:
            with pytest.raises(ValueError, match=r"integrity|outside|incomplete"):
                runner.preview_video("project", key)
        result = runner.status("project", key)
        assert result["status"] == "needs_attention"
        assert result["selected"] == []
        assert result["approval_created"] is False
        assert result["preview"] == {"shot_id": "one", "attempt": 1, "duration_ms": 5000}
        with pytest.raises(ValueError, match="not ready"):
            runner.video("project", key)
    finally:
        runner._executor.shutdown(wait=True)


def test_runtime_identity_covers_story_and_memory_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(film_project_runs, "file_digest", lambda _: "a" * 64)
    before = film_project_runs._runtime_identity()
    hashes = before["implementation_sha256s"]
    assert isinstance(hashes, dict)
    assert "story_service.py" in hashes
    assert "story_memory_backend.py" in hashes
    monkeypatch.setattr(
        film_project_runs,
        "file_digest",
        lambda path: ("b" if path.name == "story_service.py" else "a") * 64,
    )
    assert film_project_runs._runtime_identity() != before


def test_resume_rejects_a_changed_runtime_without_starting_another_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, revision = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(film_project_runs, "run_production", lambda *a, **kw: tmp_path / "film.mp4")
    try:
        first = runner.start("project", revision)
        assert runner._active is not None
        runner._active.result(timeout=5)
        monkeypatch.setattr(film_project_runs, "_runtime_identity", lambda: {"changed": True})
        with pytest.raises(ValueError, match="policy changed"):
            runner.start("project", revision, expected_run_id=first["run_id"])
        assert len(runner.runs("project")) == 1
        # Creating a new run is a separate, explicit customer action.
        new = runner.start("project", revision)
        assert new["run_id"] != first["run_id"]
    finally:
        runner._executor.shutdown(wait=True)


@pytest.mark.parametrize("dependency", ["model_files", "input_files", "implementation"])
def test_resume_checks_generation_dependencies_before_reusing_completed_film(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dependency: str
) -> None:
    runner, revision = _runner(tmp_path, monkeypatch)
    identity = {"model_files": "model-a", "input_files": "image-a", "implementation": "node-a"}
    runner.generation_identity = lambda _: dict(identity)
    calls = []

    def production(*args: Any, **kwargs: Any) -> Path:
        calls.append(args)
        return tmp_path / "film.mp4"

    monkeypatch.setattr(film_project_runs, "run_production", production)
    try:
        first = runner.start("project", revision)
        assert runner._active is not None
        runner._active.result(timeout=5)
        identity[dependency] = "changed"
        with pytest.raises(ValueError, match="policy changed"):
            runner.start("project", revision, expected_run_id=first["run_id"])
        assert len(calls) == 1
        assert len(runner.runs("project")) == 1
        new = runner.start("project", revision)
        runner._active.result(timeout=5)
        assert new["run_id"] != first["run_id"]
        assert len(calls) == 2
    finally:
        runner._executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "package",
    [
        "comfy-story",
        "comfy-kitchen",
        "comfy-aimdo",
        "triton",
        "kernels",
        "kernels-data",
        "huggingface-hub",
        "httpx",
        "tokenizers",
        "safetensors",
        "accelerate",
        "Pillow",
        "av",
    ],
)
def test_resume_rejects_package_only_runtime_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: str
) -> None:
    versions = {package: "1.0"}
    monkeypatch.setattr(film_project_runs, "version", lambda name: versions.get(name, "stable"))
    monkeypatch.setattr(film_project_runs, "file_digest", lambda _: "a" * 64)
    runner, revision = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(film_project_runs, "run_production", lambda *a, **kw: tmp_path / "film.mp4")
    try:
        first = runner.start("project", revision)
        assert runner._active is not None
        runner._active.result(timeout=5)
        versions[package] = "2.0"
        with pytest.raises(ValueError, match="policy changed"):
            runner.start("project", revision, expected_run_id=first["run_id"])
        assert len(runner.runs("project")) == 1
    finally:
        runner._executor.shutdown(wait=True)


def test_runtime_identity_records_absent_optional_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    def installed(name: str) -> str:
        if name == "kernels":
            raise PackageNotFoundError(name)
        return "stable"

    monkeypatch.setattr(film_project_runs, "version", installed)
    monkeypatch.setattr(film_project_runs, "file_digest", lambda _: "a" * 64)
    before = film_project_runs._runtime_identity()
    packages = before["packages"]
    assert isinstance(packages, dict)
    assert packages["kernels"] == "unavailable"
    monkeypatch.setattr(film_project_runs, "version", lambda _: "stable")
    assert film_project_runs._runtime_identity() != before


@pytest.mark.parametrize("uploaded", [False, True])
def test_opening_review_is_available_after_stop_without_reading_live_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uploaded: bool
) -> None:
    runner, revision = _runner(tmp_path, monkeypatch)
    entered, finish = threading.Event(), threading.Event()
    reviews = []
    evidence = [{"attempt": 1, "approval_created": False}]
    field = "starting_state_review" if uploaded else "opening_candidates"
    empty: list[dict[str, Any]] | None = None if uploaded else []

    def stopped(plan: Path, inputs: Path, server: str, directory: Path, **kwargs: Any) -> Path:
        _write(
            directory / "production.json",
            {"phase": "checking_inputs" if uploaded else "checking_opening", "shot_id": "one"},
        )
        entered.set()
        assert finish.wait(5)
        raise ValueError("opening stage failed")

    def inspect(directory: Path, inputs: Path, production: dict[str, Any]) -> list[dict[str, Any]]:
        reviews.append((directory, inputs, production))
        return evidence

    monkeypatch.setattr(film_project_runs, "run_production", stopped)
    monkeypatch.setattr(film_project_runs, field, inspect)
    try:
        state = runner.start("project", revision)
        assert entered.wait(5)
        assert runner.status("project", state["run_id"])[field] == empty
        assert reviews == []
        finish.set()
        assert runner._active is not None
        runner._active.result(timeout=5)
        result = runner.status("project", state["run_id"])
        assert result["status"] == "needs_attention"
        assert result[field] == evidence
        assert len(reviews) == 1
        assert reviews[0][0] == runner.store.project_path("project") / "runs" / state["run_id"]
        assert result["approval_created"] is False
    finally:
        finish.set()
        runner._executor.shutdown(wait=True)


def test_first_cut_delivers_without_reviewer_and_preserves_edit_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, revision = _runner(tmp_path, monkeypatch)
    runner.model = None
    before = runner.store.load("project", revision)
    calls = []

    def render(plan: Path, inputs: Path, server: str, directory: Path) -> Path:
        calls.append(json.loads(inputs.read_text()))
        output = directory / "film.mp4"
        output.write_bytes(b"rendered film fixture")
        (directory / "rendered-takes.json").write_text(
            json.dumps([{"shot_id": "one", "video_sha256": "c" * 64}])
        )
        return output

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("first cuts must not load or run the reviewer")

    monkeypatch.setattr(film_project_runs, "run_film", render)
    monkeypatch.setattr(film_project_runs, "run_production", forbidden)
    monkeypatch.setattr(film_project_runs, "_model_identity", forbidden)
    try:
        result = runner.start("project", revision, mode="first_cut")
        assert runner._active is not None
        runner._active.result(timeout=5)
        status = runner.status("project", result["run_id"])
        assert status["status"] == "draft_ready"
        assert status["mode"] == "first_cut"
        assert status["selected"] == []
        assert status["coverage"]["selected_shots"] == 0
        assert status["rendered"][0]["shot_id"] == "one"
        assert status["approval_created"] is False
        assert calls[0]["shots_by_id"]["one"]["variation"] == 42
        assert runner.store.load("project", revision) == before
        video = runner.video("project", result["run_id"])
        assert video.read_bytes() == b"rendered film fixture"
        with pytest.raises(ValueError, match="pause is unavailable"):
            runner.pause("project", result["run_id"])
        assert status["pause_requested"] is False
        video.write_bytes(b"changed")
        with pytest.raises(ValueError, match="integrity"):
            runner.video("project", result["run_id"])
        with pytest.raises(ValueError, match="Configure DUET_STORY_VERIFY_MODEL"):
            runner.start("project", revision)
    finally:
        runner._executor.shutdown(wait=True)


def test_first_cut_execution_failure_is_not_a_ready_film(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, revision = _runner(tmp_path, monkeypatch)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("Comfy execution failed")

    monkeypatch.setattr(film_project_runs, "run_film", fail)
    try:
        result = runner.start("project", revision, mode="first_cut")
        assert runner._active is not None
        runner._active.result(timeout=5)
        status = runner.status("project", result["run_id"])
        assert status["status"] == "needs_attention"
        assert status["phase"] == "rendering_film"
        assert status["reason"] == "Comfy execution failed"
        assert status["rendered"] == []
        with pytest.raises(ValueError, match="changed since this run"):
            runner.start("project", revision, expected_run_id=result["run_id"])
        resumed = runner.start(
            "project", revision, mode="first_cut", expected_run_id=result["run_id"]
        )
        assert resumed["run_id"] == result["run_id"]
    finally:
        runner._executor.shutdown(wait=True)


@pytest.mark.parametrize("mode", ["unknown", None, []])
def test_generation_mode_is_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: Any
) -> None:
    runner, revision = _runner(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match="generation mode"):
            runner.start("project", revision, mode=mode)
        assert runner.runs("project") == []
    finally:
        runner._executor.shutdown(wait=True)


def test_first_cut_does_not_silently_drop_checked_state_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, revision = _runner(tmp_path, monkeypatch)
    recipe = runner.store.load("project", revision)
    recipe["inputs"]["recall_selected_state"] = True
    monkeypatch.setattr(runner.store, "load", lambda *args: recipe)
    try:
        with pytest.raises(ValueError, match="require Check each shot"):
            runner.start("project", revision, mode="first_cut")
        assert runner.runs("project") == []
    finally:
        runner._executor.shutdown(wait=True)


def test_runtime_identity_hashes_only_the_installed_product_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "site-packages" / "comfy_story"
    package.mkdir(parents=True)
    module = package / "film_project_runs.py"
    module.write_text("# installed product")
    unrelated = package.parent / "unrelated.py"
    unrelated.write_text("# unrelated package")
    monkeypatch.setattr(film_project_runs, "__file__", str(module))
    identity = film_project_runs._runtime_identity()
    assert identity["implementation_sha256s"] == {
        "film_project_runs.py": hashlib.sha256(module.read_bytes()).hexdigest()
    }
    unrelated.write_text("# independently upgraded dependency")
    assert film_project_runs._runtime_identity() == identity
