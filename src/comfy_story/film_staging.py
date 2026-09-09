"""Bounded opening generation with immutable requests and fail-closed job recovery."""

from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from comfy_story.film_export import file_digest
from comfy_story.film_plan import FilmPlan
from comfy_story.film_runner import _json_request, _write
from comfy_story.film_selected_state import ShotStateBinding
from comfy_story.film_staging_workflow import (
    STAGING_MODELS,
    STAGING_PROTOCOL,
    compile_opening_stage,
    opening_stage_prompt,
)
from comfy_story.film_starting_state import check_starting_state
from comfy_story.film_visibility_audit import (
    parse_visibility_observation,
    visibility_observation_prompt,
)
from comfy_story.story_contracts import canonical_story_json
from comfy_story.story_native_archive import NativeReferenceArchive
from comfy_story.story_video_canvas import (
    H3_FIRST_FRAME_CROP,
    H3_FIRST_FRAME_METHOD,
    H3_VIDEO_HEIGHT,
    H3_VIDEO_WIDTH,
)

Reviewer = Callable[[list[dict[str, Any]]], str]


def fit_authored_opening(server: str, directory: Path, root: Path, world: str) -> str:
    """Bind preflight and animation to the same native Comfy first-frame crop.

    This performs only image scaling, not diffusion or another creative attempt.
    Keep the source untouched and retain the normal immutable job/output receipts.
    """
    source = _asset(root, world)
    digest = file_digest(source)
    with Image.open(source) as image:
        already_fitted = image.size == (H3_VIDEO_WIDTH, H3_VIDEO_HEIGHT)
    if already_fitted:
        return world
    # Reuse the established canvas and raw/fitted output contract, with no model nodes.
    graph: dict[str, object] = {
        "101": {"class_type": "LoadImage", "inputs": {"image": world}},
        "25": {
            "class_type": "SaveImage",
            "inputs": {"images": ["101", 0], "filename_prefix": "comfy_story_canvas/raw"},
        },
        "26": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["101", 0],
                "width": H3_VIDEO_WIDTH,
                "height": H3_VIDEO_HEIGHT,
                "upscale_method": H3_FIRST_FRAME_METHOD,
                "crop": H3_FIRST_FRAME_CROP,
            },
        },
        "24": {
            "class_type": "SaveImage",
            "inputs": {"images": ["26", 0], "filename_prefix": "comfy_story_canvas/fitted"},
        },
    }
    fitted = _render(
        server,
        directory,
        graph,
        {"operation": "fit_authored_opening", "references": {world: digest}, "server": server},
        root,
    )
    if file_digest(source) != digest:
        raise ValueError("authored opening changed during canvas preparation")
    return fitted


def _asset(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("staging image is missing or outside Comfy input")
    return path


def _publish(root: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    name = f"comfy_story_staging/{digest}.png"
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("staging upload directory escapes Comfy input")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        if file_digest(path) != digest:
            raise ValueError("content-addressed staging input changed") from None
    return name


def stage_references(
    plan: FilmPlan,
    index: int,
    settings: dict[str, Any],
    root: Path,
    bindings: tuple[ShotStateBinding, ...],
    archive: NativeReferenceArchive | None,
    scene_source: dict[str, str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Render from selected state; keep the original identity in the library and verifier."""
    current = {b.entity_id: b for b in bindings}
    rows = {r["name"]: r for r in settings["library"]["references"]}
    names: list[str] = []
    roles: list[str] = []
    world = settings["shots"][index].get("world")
    if scene_source is not None:
        if archive is None or world:
            raise ValueError("selected scene requires its archive and no uploaded world")
        revision = archive.load_revision(scene_source["source_revision_sha256"])
        if revision.state.last_frame_sha256 != scene_source["asset_sha256"]:
            raise ValueError("selected scene image does not match its recorded revision")
        names.append(_publish(root, revision.last_frame_png))
        roles.append(
            "preceding selected scene: preserve its location and layout "
            "while composing the described view"
        )
    if world:
        names.append(world)
        roles.append("authored scene and framing reference")
    for name in plan.shots[index].present:
        binding = current.get(name.casefold())
        if binding:
            if archive is None:
                raise ValueError("staged state requires its authenticated archive")
            filename = _publish(root, archive.assets.load_asset(binding.asset_sha256))
            role = name + " — selected current appearance; preserve this state"
        else:
            filename = rows[name]["file"]
            role = name + " — original identity reference"
        names.append(filename)
        roles.append(role)
    # A closing image may cover both subjects and the scene. Bind it once with all roles.
    unique_names: list[str] = []
    unique_roles: list[str] = []
    positions: dict[str, int] = {}
    for name, role in zip(names, roles, strict=True):
        digest = file_digest(_asset(root, name))
        if digest in positions:
            unique_roles[positions[digest]] += "; " + role
        else:
            positions[digest] = len(unique_names)
            unique_names.append(name)
            unique_roles.append(role)
    names, roles = unique_names, unique_roles
    if not 1 <= len(names) <= 3:
        raise ValueError("opening staging requires one to three images; no identities were dropped")
    for name in names:
        _asset(root, name)
    return tuple(names), tuple(roles)


def _render(
    server: str,
    directory: Path,
    workflow: dict[str, object],
    identity: dict[str, Any],
    root: Path,
) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    request = {"format": STAGING_PROTOCOL, "workflow": workflow, **identity}
    encoded = canonical_story_json(request)
    request_path = directory / "request.json"
    if request_path.exists():
        if request_path.read_bytes() != encoded:
            raise ValueError("opening stage inputs or model identity changed")
    else:
        with request_path.open("xb") as handle:
            handle.write(encoded)
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("request_sha256") != hashlib.sha256(encoded).hexdigest():
            raise ValueError("opening stage result belongs to another request")
        if not (directory / "history.json").is_file() or file_digest(
            directory / "history.json"
        ) != result.get("history_sha256"):
            raise ValueError("recorded opening history changed")
        if file_digest(_asset(root, result["file"])) != result["sha256"]:
            raise ValueError("recorded opening image changed")
        if not isinstance(result.get("raw_file"), str) or not isinstance(
            result.get("raw_sha256"), str
        ):
            raise ValueError("recorded raw opening receipt is missing")
        if file_digest(_asset(root, result["raw_file"])) != result["raw_sha256"]:
            raise ValueError("recorded raw opening image changed")
        return str(result["file"])
    job_path = directory / "job.json"
    if job_path.exists():
        prompt_id = json.loads(job_path.read_text())["prompt_id"]
    else:
        intent = directory / "submission-intent.json"
        if intent.exists():
            raise ValueError(
                "opening submission outcome unknown; recover the recorded job before retrying"
            )
        info = _json_request(server, "/object_info")
        if any(
            node["class_type"] not in info for node in workflow.values() if isinstance(node, dict)
        ):
            raise ValueError("Comfy is missing required opening staging nodes")
        payload = {
            "prompt": workflow,
            "client_id": "comfy-stage-" + hashlib.sha256(encoded).hexdigest()[:24],
        }
        with intent.open("xb") as handle:
            handle.write(canonical_story_json(payload))
        response = _json_request(server, "/prompt", payload)
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("Comfy did not return an opening stage prompt ID")
        _write(job_path, {"prompt_id": prompt_id})
    history_path = directory / "history.json"
    if history_path.exists():
        record = json.loads(history_path.read_text())
    else:
        while True:
            history = _json_request(server, "/history/" + urllib.parse.quote(prompt_id, safe=""))
            if prompt_id in history:
                record = history[prompt_id]
                break
            time.sleep(5)
        _write(history_path, record)
    if record.get("status", {}).get("status_str") != "success":
        raise ValueError(
            "opening stage job failed; original history preserved, no automatic requeue"
        )
    history_prompt = record.get("prompt", [])
    if len(history_prompt) < 3 or history_prompt[1] != prompt_id or history_prompt[2] != workflow:
        raise ValueError("opening output history does not match the submitted graph")

    def saved_image(node_id: str, *, fitted: bool) -> tuple[str, str]:
        outputs = record.get("outputs", {}).get(node_id, {}).get("images", [])
        if len(outputs) != 1 or outputs[0].get("type") != "output":
            raise ValueError("opening stage must save exactly one raw and one fitted image")
        query = urllib.parse.urlencode(
            {key: outputs[0][key] for key in ("filename", "subfolder", "type")}
        )
        with urllib.request.urlopen(server + "/view?" + query, timeout=60) as response:
            data = response.read(64 * 1024 * 1024 + 1)
        if not data or len(data) > 64 * 1024 * 1024:
            raise ValueError("opening stage image exceeds the media boundary")
        with Image.open(io.BytesIO(data)) as rendered:
            if (
                rendered.format != "PNG"
                or getattr(rendered, "n_frames", 1) != 1
                or rendered.width * rendered.height > 32_000_000
            ):
                raise ValueError("opening output must be one bounded PNG image")
            if fitted and rendered.size != (H3_VIDEO_WIDTH, H3_VIDEO_HEIGHT):
                raise ValueError("opening assessment requires the fitted H3 video canvas")
            rendered.verify()
        return _publish(root, data), hashlib.sha256(data).hexdigest()

    raw_name, raw_digest = saved_image("25", fitted=False)
    name, digest = saved_image("24", fitted=True)
    result = {
        "file": name,
        "sha256": digest,
        "raw_file": raw_name,
        "raw_sha256": raw_digest,
        "request_sha256": hashlib.sha256(encoded).hexdigest(),
        "prompt_id": prompt_id,
        "history_sha256": file_digest(history_path),
    }
    _write(result_path, result)
    return name


def _visibility(
    plan: FilmPlan,
    index: int,
    settings: dict[str, Any],
    root: Path,
    directory: Path,
    reviewer: Reviewer,
    model_id: str,
) -> dict[str, Any]:
    shot = plan.shots[index]
    names = tuple(
        sorted(set(shot.visible_throughout) | set(shot.fully_visible_throughout) | set(shot.absent))
    )
    if not names:
        return {"status": "not_required"}
    content: list[dict[str, Any]] = [
        {"type": "text", "text": visibility_observation_prompt(names, 0)}
    ]
    if shot.absent:
        # Keep the desired presence/absence out of the observer's input. Some excluded
        # subjects are categories (for example vehicles), not reference-library members.
        content[0]["text"] += (
            "\nFor a subject without an identity reference, assess the literal named entity "
            "or category in the current frame. Any recognizable instance, including cropped "
            "body parts, is visible. If the name cannot be grounded, report uncertain."
        )
    for row in settings["library"]["references"]:
        if row["name"] in names:
            content.extend(
                [
                    {"type": "text", "text": "REFERENCE IDENTITY ONLY: " + row["name"]},
                    {"type": "image", "image": str(_asset(root, row["file"]))},
                ]
            )
    content.extend(
        [
            {"type": "text", "text": "CURRENT FRAME 0"},
            {"type": "image", "image": str(_asset(root, settings["shots"][index]["world"]))},
        ]
    )
    hashes = {c["image"]: file_digest(Path(c["image"])) for c in content if c["type"] == "image"}
    request = canonical_story_json(
        {"content": content, "image_sha256s": hashes, "model_sha256": model_id}
    )
    directory.mkdir(parents=True, exist_ok=True)
    request_path, response_path = directory / "request.json", directory / "response.txt"
    if request_path.exists():
        if request_path.read_bytes() != request or not response_path.exists():
            raise ValueError("opening visibility inputs changed or observation was interrupted")
        raw = response_path.read_text()
    else:
        with request_path.open("xb") as handle:
            handle.write(request)
        raw = reviewer(content)
        with response_path.open("x") as response_handle:
            response_handle.write(raw)
    if any(file_digest(Path(p)) != digest for p, digest in hashes.items()):
        raise ValueError("opening visibility images changed during assessment")
    try:
        observed = parse_visibility_observation(raw, names, 0)
    except ValueError:
        observed = {}
    unknown = any(n not in observed or observed[n].extent == "uncertain" for n in names)
    failed = any(
        n in observed
        and (
            (n in shot.absent and observed[n].extent in ("entire", "partial"))
            or (n not in shot.absent and observed[n].extent == "absent")
            or (n in shot.fully_visible_throughout and observed[n].extent == "partial")
        )
        for n in names
    )
    return {
        "status": "needs_review" if unknown else "fail" if failed else "pass",
        "observations": observed,
        "requirements": {
            n: "absent"
            if n in shot.absent
            else "entire"
            if n in shot.fully_visible_throughout
            else "visible"
            for n in names
        },
        "request_sha256": hashlib.sha256(request).hexdigest(),
        "response_sha256": file_digest(response_path),
    }


def stage_opening(
    plan: FilmPlan,
    index: int,
    settings: dict[str, Any],
    root: Path,
    directory: Path,
    server: str,
    reviewer: Reviewer,
    model_id: str,
    staging_identity: dict[str, Any],
    bindings: tuple[ShotStateBinding, ...],
    archive: NativeReferenceArchive | None,
    pause: Callable[[], None],
    progress: Callable[[str, int], None],
    *,
    scene_source: dict[str, str] | None = None,
) -> str:
    """Reuse one passing stage across video attempts; uncertainty stops the image budget too."""
    references, roles = stage_references(
        plan, index, settings, root, bindings, archive, scene_source
    )
    hashes = {name: file_digest(_asset(root, name)) for name in references}
    row = settings["shots"][index]
    mode = row.get("opening_mode", "Compose")
    prompt = opening_stage_prompt(plan, index, row["opening_prompt"], roles, mode)
    for attempt in range(row.get("opening_attempts", 1)):
        pause()
        progress("staging_opening", attempt + 1)
        candidate = directory / f"attempt-{attempt + 1:02}"
        workflow = compile_opening_stage(
            prompt, references, (row["variation"] + attempt) % (1 << 64), mode
        )
        world = _render(
            server,
            candidate,
            workflow,
            {
                "model_identity": staging_identity,
                "references": hashes,
                "bindings": bindings,
                **({"scene_source": scene_source} if scene_source is not None else {}),
                "server": server,
            },
            root,
        )
        if any(file_digest(_asset(root, name)) != sha for name, sha in hashes.items()):
            raise ValueError("staging source image changed during rendering")
        resolved = {**settings, "shots": [dict(s) for s in settings["shots"]]}
        resolved["shots"][index]["world"] = world
        progress("checking_opening", attempt + 1)
        state = check_starting_state(
            plan, resolved, root, candidate / "starting-state", reviewer, model_id, shot_index=index
        )
        visibility = _visibility(
            plan, index, resolved, root, candidate / "visibility", reviewer, model_id
        )
        assessment = {"state": state, "visibility": visibility, "approval_created": False}
        assessment_path = candidate / "assessment.json"
        encoded = canonical_story_json(assessment)
        if assessment_path.exists() and assessment_path.read_bytes() != encoded:
            raise ValueError("recorded staging assessment changed")
        _write(assessment_path, assessment)
        statuses = {state["status"], visibility["status"]}
        if "needs_review" in statuses:
            raise ValueError("opening stage evidence is uncertain; no video queued")
        if "fail" not in statuses:
            return world
    raise ValueError("opening stage failed its declared image budget; no video queued")


def validate_staging_configuration(server: str, staging_identity: dict[str, Any]) -> None:
    """Reject unavailable models or node capabilities before any shot spends video compute."""
    model_files = staging_identity.get("model_files", {})
    if set(model_files) != set(STAGING_MODELS) or any(
        not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
        for v in model_files.values()
    ):
        raise ValueError("opening staging requires exact model file SHA-256 identities")
    graph = compile_opening_stage("Capability preflight", ("preflight.png",), 0)
    info = _json_request(server, "/object_info")
    required = {node["class_type"] for node in graph.values() if isinstance(node, dict)}
    missing = sorted(required - set(info))
    if missing:
        raise ValueError("Comfy is missing opening staging nodes: " + ", ".join(missing))
