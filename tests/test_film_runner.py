from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from comfy_story import film_runner
from comfy_story.film_export import export_film_draft as real_export
from comfy_story.film_export import file_digest
from comfy_story.film_plan import FilmCue, FilmPlan, FilmShot


def test_second_customer_launch_cannot_enter_an_active_run(tmp_path: Path) -> None:
    with (tmp_path / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already running"):
            film_runner.run_film(tmp_path / "plan.json", tmp_path / "inputs.json", "", tmp_path)
    # OS releases the reservation even when its owner exits; a stale lock file is harmless.
    with pytest.raises(FileNotFoundError):
        film_runner.run_film(
            tmp_path / "plan.json", tmp_path / "inputs.json", "http://localhost:8188", tmp_path
        )


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Real ffmpeg conform is required")
@pytest.mark.parametrize("caption_setting", [None, False, True])
def test_one_customer_command_submits_once_exports_unreviewed_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caption_setting: bool | None,
) -> None:
    real_run = subprocess.run
    if caption_setting is True:
        filters = real_run(
            ["ffmpeg", "-hide_banner", "-filters"], check=True, capture_output=True, text=True
        ).stdout
        if not any("subtitles" in line.split() for line in filters.splitlines()):
            pytest.skip("Burned-caption integration requires FFmpeg with libass")
    filter_checks = 0

    def checked_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        nonlocal filter_checks
        if "-filters" in args[0]:
            filter_checks += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", checked_run)
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=24:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    digest = file_digest(source)
    plan = FilmPlan(
        "unattended",
        "A film",
        1000,
        (),
        (
            FilmShot(
                "one",
                500,
                "Setup",
                "Hold",
                ("Ada",),
                dialogue=()
                if caption_setting is None
                else (FilmCue("line", 0, 500, "Hello", "en", "Ada"),),
            ),
            FilmShot("two", 500, "Resolve", "Hold", ("Ada",), depends_on=("one",)),
        ),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(asdict(plan)))
    inputs_path = tmp_path / "inputs.json"
    settings: dict[str, Any] = {
        "library": {
            "project_name": "Test",
            "references": [
                {"name": "Ada", "role": "Character", "file": "ada.png", "note": "Blue apron"},
            ],
        },
        "shots": [{"world": "room.png", "variation": 1}, {"world": "room.png", "variation": 2}],
    }
    if caption_setting is not None:
        sound = tmp_path / "sound.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=0.25",
                str(sound),
            ],
            check=True,
        )
        settings["burn_subtitles"] = caption_setting
        settings["audio"] = [
            {
                "path": sound.name,
                "sha256": file_digest(sound),
                "start_ms": 0,
                "duration_ms": 250,
                "cue_id": "line",
            }
        ]
    inputs_path.write_text(json.dumps(settings))
    run = tmp_path / "run"
    (run / "sources").mkdir(parents=True)
    shutil.copyfile(source, run / "sources" / (digest + ".mp4"))
    submissions: list[object] = []
    supports_duration = False

    def request(server: str, path: str, payload: object = None) -> dict[str, object]:
        assert server == "http://localhost:8188"
        if path == "/object_info":
            return {
                "DuetStory": {
                    "input": {
                        "optional": {"Output duration (ms)": [], "Scene entities": []}
                        if supports_duration
                        else {}
                    }
                }
            }
        if path == "/prompt":
            submissions.append(payload)
            return {"prompt_id": "owned-prompt"}
        assert path == "/history/owned-prompt"
        return {
            "owned-prompt": {
                "status": {"status_str": "success"},
                "outputs": {
                    str(i * 10): {
                        "duet_story": [
                            {
                                "owner_node_id": str(i * 10),
                                "revision_sha256": character * 64,
                                "video_sha256": digest,
                                "parent_revision_sha256": parent,
                            }
                        ]
                    }
                    for i, character, parent in [(1, "a", None), (2, "b", "a" * 64)]
                },
            }
        }

    monkeypatch.setattr(film_runner, "_json_request", request)
    uploaded_hashes = {"ada.png": "a" * 64, "room.png": "b" * 64}
    monkeypatch.setattr(film_runner, "_input_hashes", lambda server, workflow: uploaded_hashes)
    with pytest.raises(ValueError, match="Upgrade the Comfy"):
        film_runner.run_film(plan_path, inputs_path, "http://localhost:8188", run)
    assert not submissions
    assert not (run / "submission-intent.json").exists()
    supports_duration = True
    output = film_runner.run_film(plan_path, inputs_path, "http://localhost:8188", run)
    assert output.is_file()
    assert len(submissions) == 1
    receipt = json.loads((output.parent / "export-receipt.json").read_text())
    assert receipt["duration_ms"] == 1000
    assert receipt["review_status"] == "unreviewed"
    assert receipt["subtitles_burned"] is (caption_setting is True)
    if caption_setting is False:
        assert filter_checks == 0, "Captionless films must not require libass"
    assert receipt["accepted_take_sha256s"] == []
    assert len(receipt["rendered_take_sha256s"]) == 2
    assert not (output.parent / "accepted-takes.json").exists()
    assert "observed_facts" not in (output.parent / "rendered-takes.json").read_text()
    assert film_runner.run_film(plan_path, inputs_path, "http://localhost:8188", run) == output
    assert len(submissions) == 1
    settings["burn_subtitles"] = caption_setting is False
    inputs_path.write_text(json.dumps(settings))
    # Switching a captionless run to captions may fail capability preflight first
    # on FFmpeg builds without libass; neither path may reuse the old output.
    with pytest.raises(
        ValueError, match=r"inputs changed|film captions require ffmpeg with libass"
    ):
        film_runner.run_film(plan_path, inputs_path, "http://localhost:8188", run)
    if caption_setting is None:
        settings.pop("burn_subtitles")
    else:
        settings["burn_subtitles"] = caption_setting
    inputs_path.write_text(json.dumps(settings))
    uploaded_hashes["ada.png"] = "c" * 64
    with pytest.raises(ValueError, match="inputs changed"):
        film_runner.run_film(plan_path, inputs_path, "http://localhost:8188", run)
    uploaded_hashes["ada.png"] = "a" * 64
    settings["shots"][1]["variation"] = 3
    inputs_path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="inputs changed"):
        film_runner.run_film(plan_path, inputs_path, "http://localhost:8188", run)
    assert len(submissions) == 1

    settings["shots"][1]["variation"] = 2
    inputs_path.write_text(json.dumps(settings))
    interrupted = tmp_path / "interrupted"
    (interrupted / "sources").mkdir(parents=True)
    shutil.copyfile(source, interrupted / "sources" / (digest + ".mp4"))
    failed = False

    def fail_once(*args: Any, **kwargs: Any) -> Path:
        nonlocal failed
        if not failed:
            failed = True
            args[3].mkdir()
            raise RuntimeError("simulated finishing interruption")
        return real_export(*args, **kwargs)

    monkeypatch.setattr(film_runner, "export_film_draft", fail_once)
    with pytest.raises(RuntimeError, match="finishing interruption"):
        film_runner.run_film(plan_path, inputs_path, "http://localhost:8188", interrupted)
    assert len(submissions) == 2
    retried = film_runner.run_film(plan_path, inputs_path, "http://localhost:8188", interrupted)
    assert retried.parent.name == "export-002"
    assert (interrupted / "export").is_dir()
    assert len(submissions) == 2


@pytest.mark.parametrize("invalid", ["false", 0, 1, None, [], {}])
def test_caption_policy_rejects_non_booleans_before_generation(
    tmp_path: Path, invalid: object
) -> None:
    plan = FilmPlan("plain", "Plain film", 1000, (), (FilmShot("one", 1000, "Hold", "Hold", ()),))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(asdict(plan)))
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(json.dumps({"burn_subtitles": invalid}))
    with pytest.raises(ValueError, match="burn_subtitles must be a boolean"):
        film_runner.run_film(plan_path, inputs_path, "http://localhost:8188", tmp_path / "run")
