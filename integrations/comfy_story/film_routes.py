"""Same-origin project editing and verified film jobs for the Comfy Story panel."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from aiohttp import BodyPartReader, web

from comfy_story.film_audio_asset import inspect_soundtrack
from comfy_story.film_bundle import _asset_path, export_film_inputs, import_film_inputs
from comfy_story.film_export import file_digest
from comfy_story.film_io import film_plan_from_json
from comfy_story.film_project import FilmProjectConflict, FilmProjectStore
from comfy_story.film_project_runs import FilmProjectRunner

from .comfy_adapter import _story_root

_MAX_SOUNDTRACK_UPLOAD = 256 * 1024 * 1024


def _film_generation_identity(recipe: dict[str, Any]) -> dict[str, object]:
    """Bind whole-film reuse to the same inputs, models and Comfy implementation."""
    import folder_paths

    from comfy_story.film_bundle import _asset_path, _slots
    from comfy_story.memory.settings import InspectionCodec, configured_memory

    from .comfy_adapter import (
        _generation_asset_digest,
        _generation_assets,
        _story_sampler,
        render_configuration,
    )

    memory = configured_memory()
    runtime = memory.load(InspectionCodec())
    runtime.close()
    inputs = recipe["inputs"]
    staging = {}
    if any(row.get("opening_prompt", "").strip() for row in inputs["shots_by_id"].values()):
        from comfy_story.film_staging_workflow import STAGING_MODELS, STAGING_PROTOCOL

        staging = {
            "staging": {
                "protocol": STAGING_PROTOCOL,
                "model_files": {
                    kind: _generation_asset_digest(kind, name)
                    for kind, name in STAGING_MODELS.items()
                },
            }
        }
    models = {}
    for row in inputs["shots_by_id"].values():
        profile = row.get("render_profile", "Reference shot")
        configuration = render_configuration(
            profile, _story_sampler(row.get("sampler", "Native res_multistep"))
        )
        prefix = "" if profile == "Reference shot" else profile + ":"
        generation_assets = _generation_assets(configuration)
        if (
            generation_assets["model"] != memory.foundation_sha256
            or generation_assets["video_vae"] != memory.vae_sha256
        ):
            raise ValueError("film models do not match the associative checkpoint foundation/VAE")
        models.update({prefix + key: value for key, value in generation_assets.items()})
        if row.get("sampler") in {"Turbo 4-step", "Turbo 8-step"}:
            # Preserve historical keys while binding both adapters in a mixed-profile film.
            lora_key = "lora:turbo-8step" if row["sampler"] == "Turbo 8-step" else "lora"
            models[prefix + lora_key] = _generation_asset_digest(
                "loras", str(configuration["lora"])
            )
        if row.get("sampler") == "Full HD 2-pass":
            import nodes as comfy_nodes

            from .h3_quality_nodes import require_upscaler

            require_upscaler(getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}))
            models[prefix + "lora:full-hd"] = _generation_asset_digest(
                "loras", str(configuration["lora"])
            )
    root = Path(folder_paths.get_input_directory()).resolve()
    assets = {}
    for row, field, kind in _slots(inputs):
        name = row[field]
        if name in assets:
            continue
        path = _asset_path(root, name, kind)
        before = path.stat()
        assets[name] = file_digest(path)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("film input changed while checking generation identity")
    integration = Path(__file__).parent
    comfy = Path(folder_paths.__file__).resolve().parent
    implementation = {}
    for label, base, paths in (
        ("integration", integration, integration.rglob("*.py")),
        ("comfy", comfy, comfy.glob("*.py")),
        *(
            ("comfy", comfy, (comfy / name).rglob("*.py"))
            for name in ("comfy", "comfy_api", "comfy_api_nodes", "comfy_execution", "comfy_extras")
        ),
    ):
        for path in sorted(paths):
            implementation[label + "/" + path.relative_to(base).as_posix()] = file_digest(path)
    from comfy_story.story_native_archive import NATIVE_REFERENCE_RUNTIME_SHA256

    return {
        "associative_memory": memory.binding(),
        "model_files": models,
        **staging,
        "memory_runtime": NATIVE_REFERENCE_RUNTIME_SHA256,
        "input_files": assets,
        "implementation": implementation,
    }


class FilmRoutes:
    def __init__(self) -> None:
        self.runner: FilmProjectRunner | None = None

    def service(self, request: web.Request) -> FilmProjectRunner:
        if self.runner is None:
            import folder_paths

            address = request.transport.get_extra_info("sockname") if request.transport else None
            if not address:
                raise ValueError("Comfy listener address is unavailable")
            model = os.environ.get("COMFY_STORY_VERIFY_MODEL", "").strip()
            self.runner = FilmProjectRunner(
                FilmProjectStore(_story_root() / "film_projects"),
                f"{request.scheme}://127.0.0.1:{address[1]}",
                Path(folder_paths.get_input_directory()),
                Path(model) if model else None,
                device=os.environ.get("COMFY_STORY_VERIFY_DEVICE", "cpu"),
                generation_identity=_film_generation_identity,
                story_root=_story_root(),
            )
        return self.runner

    async def upload_soundtrack(
        self, request: web.Request, service: FilmProjectRunner
    ) -> web.Response:
        """Validate a bounded upload before publishing it under a new server-owned name."""
        if not request.content_type.startswith("multipart/"):
            raise ValueError("upload one audio file as multipart form data")
        reader = await request.multipart()
        part = await reader.next()
        if not isinstance(part, BodyPartReader) or part.name != "audio" or not part.filename:
            raise ValueError("choose one soundtrack file")
        root = service.comfy_input.resolve()
        suffix = Path(part.filename).suffix.lower()
        # Only the suffix is used. Client names never choose an output path or overwrite a file.
        final = _asset_path(root, f"story-audio-{uuid.uuid4().hex}{suffix}", "audio")
        with tempfile.TemporaryDirectory(prefix="story-audio-upload-", dir=root) as temporary:
            staged = Path(temporary) / ("audio" + suffix)
            total = 0
            with staged.open("wb") as output:
                while chunk := await part.read_chunk(size=1024 * 1024):
                    total += len(chunk)
                    if total > _MAX_SOUNDTRACK_UPLOAD:
                        raise ValueError("soundtrack upload exceeds 256 MiB")
                    output.write(chunk)
            if await reader.next() is not None:
                raise ValueError("upload one soundtrack at a time")
            metadata = await asyncio.to_thread(
                inspect_soundtrack, root, staged.relative_to(root).as_posix()
            )
            os.link(staged, final)
        return web.json_response({**metadata, "path": final.name})

    async def import_inputs(self, request: web.Request, service: FilmProjectRunner) -> web.Response:
        if not request.content_type.startswith("multipart/"):
            raise ValueError("upload a multipart film inputs bundle")
        reader = await request.multipart()
        part = await reader.next()
        if not isinstance(part, BodyPartReader) or part.name != "bundle":
            raise ValueError("upload a film inputs bundle")
        with tempfile.TemporaryDirectory(
            prefix="comfy-bundle-upload-", dir=service.store.root
        ) as temporary:
            path = Path(temporary) / "upload.zip"
            total = 0
            with path.open("wb") as output:
                while chunk := await part.read_chunk(size=1024 * 1024):
                    total += len(chunk)
                    if total > 1024 * 1024 * 1024 + 2 * 1024 * 1024:
                        raise ValueError("film inputs upload exceeds 1 GiB")
                    output.write(chunk)
            if await reader.next() is not None:
                raise ValueError("upload one film inputs bundle at a time")
            result = await asyncio.to_thread(
                import_film_inputs, service.store, path, service.comfy_input
            )
        return web.json_response(result)

    def export_inputs(self, service: FilmProjectRunner, project_id: str, revision: str) -> Path:
        project = service.store.project_path(project_id)
        directory = service.store._inside(project, "input_bundles")
        directory.mkdir(exist_ok=True)
        temporary = service.store._inside(project, "input_bundles", f".{uuid.uuid4().hex}.zip")
        try:
            export_film_inputs(service.store, project_id, revision, service.comfy_input, temporary)
            digest = file_digest(temporary)
            final = service.store._inside(project, "input_bundles", digest + ".zip")
            try:
                os.link(temporary, final)
            except FileExistsError:
                if file_digest(final) != digest:
                    raise ValueError("stored film inputs bundle changed") from None
            return final
        finally:
            temporary.unlink(missing_ok=True)

    async def handle(self, request: web.Request) -> web.StreamResponse:
        try:
            origin = request.headers.get("Origin")
            if request.method != "GET" and origin and urlsplit(origin).netloc != request.host:
                raise web.HTTPForbidden(text="Use the Comfy application's own origin")
            if request.method == "POST" and request.path.endswith("/reference-review"):
                import folder_paths

                from comfy_story.reference_review import review_image_references

                from .comfy_adapter import _story_sampler, render_configuration

                payload = await request.json()
                if not isinstance(payload, dict):
                    raise ValueError("Reference review must be an object")
                config = render_configuration(
                    "Reference shot", _story_sampler(payload.get("sampler", "Native res_multistep"))
                )
                result = await asyncio.to_thread(
                    review_image_references,
                    Path(folder_paths.get_input_directory()).resolve(),
                    payload.get("references"),
                    (config["width"], config["height"]),
                    "match" if "lora" in config else "max",
                )
                return web.json_response(result)
            service = self.service(request)
            project_id = request.match_info.get("project_id")
            run_id = request.match_info.get("run_id")
            if request.method == "GET":
                if project_id is None:
                    projects = []
                    for path in sorted(service.store.root.glob("*/current"))[:100]:
                        recipe = service.store.load(path.parent.name)
                        projects.append(
                            {
                                "project_id": path.parent.name,
                                "title": recipe["plan"]["title"],
                                "revision": recipe["revision"],
                            }
                        )
                    return web.json_response({"projects": projects})
                if request.path.endswith("/inputs-bundle"):
                    revision = request.query.get("revision")
                    if revision is None:
                        raise ValueError("choose a saved recipe revision to export")
                    path = await asyncio.to_thread(
                        self.export_inputs, service, project_id, revision
                    )
                    return web.FileResponse(
                        path,
                        headers={
                            "Content-Type": "application/zip",
                            "Content-Disposition": 'attachment; filename="comfy-story-inputs.zip"',
                            "Cache-Control": "no-store",
                        },
                    )
                if run_id:
                    if request.path.endswith(("/video", "/preview")):
                        lookup = (
                            service.preview_video
                            if request.path.endswith("/preview")
                            else service.video
                        )
                        return web.FileResponse(
                            await asyncio.to_thread(lookup, project_id, run_id),
                            headers={"Content-Type": "video/mp4", "Cache-Control": "no-store"},
                        )
                    return web.json_response(service.status(project_id, run_id))
                recipe = service.store.load(project_id)
                return web.json_response(
                    {
                        "recipe": recipe,
                        "runs": service.runs(project_id),
                        "verification_configured": service.model is not None,
                    }
                )
            if project_id is None and request.path.endswith("/import-inputs"):
                return await self.import_inputs(request, service)
            if project_id is None and request.path.endswith("/upload-soundtrack"):
                return await self.upload_soundtrack(request, service)
            data = await request.json()
            if not isinstance(data, dict):
                raise ValueError("film request must be an object")
            if project_id is None and request.path.endswith("/soundtrack"):
                name = data.get("path")
                if not isinstance(name, str):
                    raise ValueError("choose an uploaded Comfy audio filename")
                return web.json_response(
                    await asyncio.to_thread(inspect_soundtrack, service.comfy_input, name)
                )
            if project_id and run_id:
                return web.json_response(service.pause(project_id, run_id))
            if project_id and request.path.endswith("/runs"):
                revision = data.get("revision")
                if not isinstance(revision, str):
                    raise ValueError("choose a saved film revision before generation")
                expected_run_id = data.get("run_id")
                if expected_run_id is not None and not isinstance(expected_run_id, str):
                    raise ValueError("run_id must identify the run being resumed")
                result = await asyncio.to_thread(
                    service.start,
                    project_id,
                    revision,
                    data.get("max_attempts", 2),
                    expected_run_id=expected_run_id,
                    mode=data.get("mode", "checked"),
                )
                return web.json_response(result)
            plan = film_plan_from_json(json.dumps(data["plan"]))
            project_id = project_id or plan.project_id
            expected = data.get("expected_revision")
            if expected is not None and not isinstance(expected, str):
                raise ValueError("expected_revision must name the version being edited")
            saved = await asyncio.to_thread(
                service.store.save, project_id, plan, data["inputs"], expected_revision=expected
            )
            impact = (
                service.store.impact(project_id, expected, saved["revision"]) if expected else None
            )
            return web.json_response({"recipe": saved, "impact": impact})
        except FilmProjectConflict as error:
            return web.json_response({"error": str(error)}, status=409)
        except FileNotFoundError:
            return web.json_response(
                {"error": "Film project, run or required asset was not found"}, status=404
            )
        except (ValueError, TypeError, KeyError) as error:
            return web.json_response({"error": str(error)}, status=400)


def register_film_routes(server: Any) -> None:
    handler = FilmRoutes()
    root = "/comfy/story/films"
    server.routes.get(root)(handler.handle)
    server.routes.post(root)(handler.handle)
    server.routes.post(root + "/import-inputs")(handler.handle)
    server.routes.post(root + "/soundtrack")(handler.handle)
    server.routes.post(root + "/upload-soundtrack")(handler.handle)
    server.routes.post(root + "/reference-review")(handler.handle)
    server.routes.get(root + "/{project_id}/inputs-bundle")(handler.handle)
    server.routes.get(root + "/{project_id}")(handler.handle)
    server.routes.put(root + "/{project_id}")(handler.handle)
    server.routes.post(root + "/{project_id}/runs")(handler.handle)
    server.routes.get(root + "/{project_id}/runs/{run_id}")(handler.handle)
    server.routes.post(root + "/{project_id}/runs/{run_id}/pause")(handler.handle)
    server.routes.get(root + "/{project_id}/runs/{run_id}/video")(handler.handle)
    server.routes.get(root + "/{project_id}/runs/{run_id}/preview")(handler.handle)
