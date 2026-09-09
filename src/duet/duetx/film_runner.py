"""Customer command for one queued Story sequence and an automatic, unreviewed film export."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from duet.duetx.film_export import FilmAudio, export_film_draft, file_digest
from duet.duetx.film_io import film_plan_from_json, normalize_film_settings
from duet.duetx.film_plan import FilmPlan, RenderedTake
from duet.duetx.film_workflow import FilmShotInput, compile_film_workflow
from duet.duetx.story_contracts import canonical_story_json


def _json_request(server: str, path: str, payload: object = None) -> dict[str, Any]:
    data = None if payload is None else canonical_story_json(payload)
    request = urllib.request.Request(
        server + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("Comfy response exceeds the film runner boundary")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Comfy response must be an object")
    return value


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_story_json(value))
    temporary.replace(path)


def _duration(path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return round(float(json.loads(result.stdout)["format"]["duration"]) * 1000)


def _input_hashes(server: str, workflow: dict[str, object]) -> dict[str, str]:
    """Bind filenames to their current uploaded bytes before queueing or resuming."""
    filenames: set[str] = set()
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        values = node["inputs"]
        if node["class_type"] == "LoadAudio":
            filenames.add(values["audio"])
        elif node["class_type"] == "LoadImage":
            filenames.add(values["image"])
        else:
            if values["World / starting frame"] != "None":
                filenames.add(values["World / starting frame"])
            library = json.loads(values["Story Library"])
            filenames.update(row["file"] for row in library["references"])
    hashes = {}
    for filename in sorted(filenames):
        path = Path(filename)
        query = urllib.parse.urlencode(
            {
                "filename": path.name,
                "subfolder": str(path.parent) if path.parent != Path() else "",
                "type": "input",
            }
        )
        digest = hashlib.sha256()
        count = 0
        with urllib.request.urlopen(server + "/view?" + query, timeout=60) as response:
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > 512 * 1024 * 1024:
                    raise ValueError("film input exceeds the 512 MiB preflight boundary")
                digest.update(block)
        if not count:
            raise ValueError("film input is empty")
        hashes[filename] = digest.hexdigest()
    return hashes


def run_film(plan_path: Path, inputs_path: Path, server: str, directory: Path) -> Path:
    """Serialize callers sharing a run directory so double launches cannot submit twice."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("film command is already running in this output directory") from error
        return _run_film_locked(plan_path, inputs_path, server, directory)


def _preflight(
    plan_path: Path, inputs_path: Path, server: str, *, allow_pending_staging: bool = False
) -> tuple[FilmPlan, dict[str, Any], dict[str, object], tuple[FilmAudio, ...], dict[str, str], str]:
    """Validate the complete planned job before any GPU submission."""
    parsed = urllib.parse.urlsplit(server)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("server must be an HTTP(S) Comfy base URL")
    server = server.rstrip("/")
    plan = film_plan_from_json(plan_path.read_text())
    if any(shot.text for shot in plan.shots):
        raise ValueError("visible text overlays require an explicit typography compositor")
    settings = json.loads(inputs_path.read_text())
    if not isinstance(settings, dict):
        raise ValueError("film inputs must be an object")
    burn_subtitles = settings.get("burn_subtitles", True)
    if type(burn_subtitles) is not bool:
        raise ValueError("burn_subtitles must be a boolean")
    settings = normalize_film_settings(plan, settings)
    shot_inputs = tuple(FilmShotInput(**item) for item in settings["shots"])
    workflow = compile_film_workflow(
        plan,
        settings["library"],
        shot_inputs,
        allow_pending_staging=allow_pending_staging,
        generated_audio=settings.get("generated_audio", False),
    )
    repairs = settings.get("repair_notes", {})
    if not isinstance(repairs, dict) or not set(repairs) <= {shot.shot_id for shot in plan.shots}:
        raise ValueError("repair_notes must name existing film shots")
    for index, shot in enumerate(plan.shots):
        note = repairs.get(shot.shot_id, "")
        if not isinstance(note, str) or len(note) > 2000 or "@" in note or "\x00" in note:
            raise ValueError("repair notes must be bounded text without reference mentions")
        if note.strip():
            node = workflow[str((index + 1) * 10)]
            if not isinstance(node, dict):
                raise ValueError("film shot must compile to a node")
            node["inputs"]["What happens next?"] += (
                "\nCorrection for this attempt, based on a rejected earlier take: "
                + note.strip()
                + "\nPreserve the original shot requirements and ending goals. "
                "The earlier take is rejected, not the current scene state."
            )
    tracks = tuple(
        FilmAudio(**{**item, "path": (inputs_path.parent / item["path"]).resolve()})
        for item in settings.get("audio", [])
    )
    # Reject missing or changed authored sound before spending any GPU time.
    cues: dict[str, tuple[int, int]] = {}
    offset = 0
    for shot in plan.shots:
        for cue in shot.dialogue:
            cues[cue.cue_id] = (offset + cue.start_ms, offset + cue.end_ms)
        offset += shot.duration_ms
    spoken: set[str] = set()
    for track in tracks:
        if not track.path.is_file() or file_digest(track.path) != track.sha256:
            raise ValueError("film audio is missing or its SHA-256 changed")
        if track.duration_ms <= 0 or _duration(track.path) < track.duration_ms:
            raise ValueError("film audio is shorter than its declared use")
        if not 0 <= track.start_ms < track.start_ms + track.duration_ms <= plan.target_duration_ms:
            raise ValueError("film audio must fit the planned timeline")
        if not 0 < track.gain <= 10:
            raise ValueError("film audio gain is out of bounds")
        if track.cue_id:
            if track.cue_id not in cues or track.cue_id in spoken:
                raise ValueError("speech requires one unique planned cue")
            start, end = cues[track.cue_id]
            if track.start_ms != start or track.start_ms + track.duration_ms > end:
                raise ValueError("speech must fit its planned cue window")
            spoken.add(track.cue_id)
    if settings.get("generated_audio", False) and spoken:
        raise ValueError("Generated dialogue cannot also have authored speech tracks")
    if not settings.get("generated_audio", False) and spoken != set(cues):
        raise ValueError("every dialogue cue needs an authored audio asset before generation")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise ValueError("film export requires ffmpeg and ffprobe before generation")
    if cues and burn_subtitles:
        filters = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if not any("subtitles" in line.split() for line in filters.splitlines()):
            raise ValueError("film captions require ffmpeg with libass before generation")
    input_hashes = _input_hashes(server, workflow)
    return plan, settings, workflow, tracks, input_hashes, server


def _run_film_locked(plan_path: Path, inputs_path: Path, server: str, directory: Path) -> Path:
    """Run once, resume the same known prompt, and preserve every first-pass output.

    Interrupted or ambiguous submissions never cause an automatic second GPU submission.
    Run this on the Comfy host or any machine that can reach its configured API and media route.
    Audio assets in the inputs document are local to this command; reference/shot filenames
    are already uploaded to Comfy, as in an ordinary saved workflow.
    """
    plan, settings, workflow, tracks, input_hashes, server = _preflight(
        plan_path, inputs_path, server
    )
    burn_subtitles = settings.get("burn_subtitles", True)
    identity = hashlib.sha256(
        canonical_story_json(
            {
                "plan": plan,
                "workflow": workflow,
                "audio": settings.get("audio", []),
                "server": server,
                "input_sha256s": input_hashes,
                # Preserve recovery identities for the historical caption default.
                **({"burn_subtitles": False} if not burn_subtitles else {}),
            }
        )
    ).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "run.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            raise ValueError("run inputs changed; use a new output directory")
    else:
        state = {"identity": identity, "review_status": "unreviewed", "prompt_id": None}
        _write(state_path, state)
        _write(directory / "workflow.json", workflow)
        _write(directory / "film-plan.json", plan)
        _write(directory / "input-sha256s.json", input_hashes)
    export_attempt = state.get("export_attempt", 0)
    if type(export_attempt) is not int or export_attempt < 0:
        raise ValueError("run export attempt is invalid")
    export_name = "export" if export_attempt <= 1 else f"export-{export_attempt:03}"
    result_path = directory / export_name / "export-receipt.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        output = result_path.parent / "film.mp4"
        if result["film_sha256"] != file_digest(output):
            raise ValueError("completed film bytes changed")
        return output
    if state["prompt_id"] is None:
        intent = directory / "submission-intent.json"
        if intent.exists():
            raise ValueError("submission outcome is unknown; recover the prompt ID before retrying")
        info = _json_request(server, "/object_info")
        if any(
            node["class_type"] not in info for node in workflow.values() if isinstance(node, dict)
        ):
            raise ValueError("Comfy is missing a required public film workflow node")
        story_inputs = info["DuetStory"].get("input", {})
        if not any(
            "Output duration (ms)" in story_inputs.get(section, {})
            for section in ("required", "optional")
        ):
            raise ValueError("Upgrade the Comfy Story node to support exact film durations")
        if not any(
            "Scene entities" in story_inputs.get(section, {})
            for section in ("required", "optional")
        ):
            raise ValueError("Upgrade the Comfy Story node to support declared scene entities")
        if any(
            isinstance(node, dict) and "Render profile" in node["inputs"]
            for node in workflow.values()
        ) and not any(
            "Render profile" in story_inputs.get(section, {})
            for section in ("required", "optional")
        ):
            raise ValueError("Upgrade the Comfy Story node to support render profiles")
        if any(
            isinstance(node, dict) and "Ending frame" in node["inputs"]
            for node in workflow.values()
        ) and not any(
            "Ending frame" in story_inputs.get(section, {}) for section in ("required", "optional")
        ):
            raise ValueError("Upgrade the Comfy Story node to support ending frames")
        if any(
            isinstance(node, dict) and "Shot state evidence" in node["inputs"]
            for node in workflow.values()
        ) and not any(
            "Shot state evidence" in story_inputs.get(section, {})
            for section in ("required", "optional")
        ):
            raise ValueError("Upgrade the Comfy Story node to support shot state evidence")
        payload = {"prompt": workflow, "client_id": "duet-film-" + identity[:24]}
        _write(intent, payload)
        response = _json_request(server, "/prompt", payload)
        if not isinstance(response.get("prompt_id"), str):
            raise ValueError("Comfy did not return a film prompt ID")
        state["prompt_id"] = response["prompt_id"]
        _write(state_path, state)
        (directory / "job_ids.txt").write_text(response["prompt_id"] + "\n")
        print("Queued complete film:", response["prompt_id"], flush=True)
    prompt_id = state["prompt_id"]
    history_path = "/history/" + urllib.parse.quote(prompt_id, safe="")
    while True:
        try:
            history = _json_request(server, history_path)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10)
            continue
        if prompt_id in history:
            record = history[prompt_id]
            break
        time.sleep(10)
    _write(directory / "history.json", record)
    if record["status"]["status_str"] != "success":
        raise RuntimeError("Film workflow failed; original history saved, no automatic retake")
    summaries: dict[str, dict[str, Any]] = {}
    for output in record["outputs"].values():
        for summary in output.get("duet_story", []):
            if isinstance(summary, dict):
                owner = summary.get("owner_node_id")
                if isinstance(owner, str) and owner in workflow:
                    if owner in summaries and summaries[owner] != summary:
                        raise ValueError("conflicting Story output summaries")
                    summaries[owner] = summary
    assets: dict[str, Path] = {}
    takes = []
    media = directory / "sources"
    media.mkdir(exist_ok=True)
    for index, shot in enumerate(plan.shots):
        summary = summaries.get(str((index + 1) * 10))
        if summary is None or "parent_revision_sha256" not in summary:
            raise ValueError("Story output lacks generation ancestry; update the Duet node")
        digest = summary["video_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Story output has an invalid video digest")
        path = media / (digest + ".mp4")
        if not path.exists():
            temporary = path.with_suffix(".partial")
            with (
                urllib.request.urlopen(
                    server + "/duet/story/video/" + digest, timeout=60
                ) as response,
                temporary.open("wb") as handle,
            ):
                count = 0
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > 4 * 1024 * 1024 * 1024:
                        raise ValueError("rendered video exceeds the Story asset boundary")
                    handle.write(block)
            if file_digest(temporary) != digest:
                raise ValueError("downloaded film source SHA-256 changed")
            temporary.replace(path)
        assets[digest] = path
        takes.append(
            RenderedTake(
                shot.shot_id,
                shot.digest,
                summary["revision_sha256"],
                digest,
                summary["parent_revision_sha256"],
                _duration(path),
            )
        )
    _write(directory / "rendered-takes.json", takes)
    # A failed conform can be retried from the saved render without overwriting its evidence
    # or submitting another GPU job. Each interrupted attempt remains available for diagnosis.
    export_attempt = max(1, export_attempt)
    while True:
        export_name = "export" if export_attempt == 1 else f"export-{export_attempt:03}"
        export_directory = directory / export_name
        if not export_directory.exists():
            break
        export_attempt += 1
    state["export_attempt"] = export_attempt
    _write(state_path, state)
    export_options: dict[str, Any] = {}
    if settings.get("generated_audio", False):
        export_options["generated_audio"] = True
    if any(row.get("sampler") == "Full HD 2-pass" for row in settings["shots"]):
        export_options["output_size"] = (1920, 1088)
    return export_film_draft(
        plan,
        tuple(takes),
        assets,
        export_directory,
        audio=tracks,
        burn_subtitles=burn_subtitles,
        **export_options,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-model", type=Path, help="Local Qwen3-VL model for gated production"
    )
    parser.add_argument(
        "--comfy-input", type=Path, help="Comfy input directory for visual references"
    )
    parser.add_argument("--verify-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--max-attempts", type=int, default=2, help="Verified mode: 1-4 attempts per shot"
    )
    args = parser.parse_args()
    if args.verify_model is not None:
        from duet.duetx.film_production import ProductionStopped, run_production

        if args.comfy_input is None:
            parser.error("--verify-model requires --comfy-input")
        try:
            result = run_production(
                args.plan,
                args.inputs,
                args.server,
                args.output,
                model=args.verify_model,
                comfy_input=args.comfy_input,
                device=args.verify_device,
                max_attempts=args.max_attempts,
            )
        except ProductionStopped as error:
            parser.exit(2, f"{error}\n")
        print(result, flush=True)
        return
    print(run_film(args.plan, args.inputs, args.server, args.output), flush=True)


if __name__ == "__main__":
    main()
