from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from comfy_story.film_plan import FilmPlan, FilmShot
from comfy_story.film_project import FilmProjectStore
from comfy_story.film_project_runs import FilmProjectRunner


def test_project_http_save_conflict_resume_configuration_and_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = ModuleType("film_route_test")
    package.__path__ = []
    adapter = ModuleType("film_route_test.comfy_adapter")
    monkeypatch.setattr(adapter, "_story_root", lambda: tmp_path, raising=False)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    filename = Path(__file__).resolve().parents[1] / "integrations/comfy_story/film_routes.py"
    spec = importlib.util.spec_from_file_location("film_route_test.film_routes", filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    handler = module.FilmRoutes()
    runner = FilmProjectRunner(
        FilmProjectStore(tmp_path / "projects"), "http://localhost:8188", tmp_path, None
    )
    handler.runner = runner
    monkeypatch.setattr(module, "FilmRoutes", lambda: handler)
    routes = web.RouteTableDef()
    module.register_film_routes(SimpleNamespace(routes=routes))
    app = web.Application()
    app.add_routes(routes)
    plan = FilmPlan("project", "Story", 5000, (), (FilmShot("one", 5000, "Show", "Hold", ()),))
    Image.new("RGB", (16, 16), "blue").save(tmp_path / "frame.png")
    payload: dict[str, Any] = {
        "plan": asdict(plan),
        "inputs": {
            "library": {"project_name": "Story", "references": []},
            "shots": [{"world": "frame.png", "variation": 7}],
        },
    }
    preview = tmp_path / "candidate.mp4"
    preview.write_bytes(b"candidate preview fixture")
    preview_reads = []

    def preview_video(project_id: str, run_id: str) -> Path:
        preview_reads.append((project_id, run_id))
        return preview

    monkeypatch.setattr(runner, "preview_video", preview_video)
    soundtrack_reads = []

    def soundtrack(root: Path, name: str) -> dict[str, object]:
        soundtrack_reads.append((root, name))
        return {"path": name, "sha256": "a" * 64, "duration_ms": 5000}

    monkeypatch.setattr(module, "inspect_soundtrack", soundtrack)

    async def scenario() -> None:
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/comfy/story/films", json=payload)
            assert response.status == 200
            saved = (await response.json())["recipe"]
            assert saved["inputs"]["shots_by_id"]["one"]["variation"] == 7
            response = await client.get("/comfy/story/films/project")
            data = await response.json()
            assert data["recipe"] == saved
            assert data["verification_configured"] is False
            run_id = "b" * 64
            response = await client.get(f"/comfy/story/films/project/runs/{run_id}/preview")
            assert response.status == 200
            assert response.headers["Content-Type"] == "video/mp4"
            assert await response.read() == preview.read_bytes()
            assert preview_reads == [("project", run_id)]
            assert runner.runs("project") == []
            response = await client.put("/comfy/story/films/project", json=payload)
            assert response.status == 409
            payload["expected_revision"] = saved["revision"]
            response = await client.put("/comfy/story/films/project", json=payload)
            assert response.status == 200
            response = await client.post(
                "/comfy/story/films/project/runs", json={"revision": saved["revision"]}
            )
            assert response.status == 400
            assert "VERIFY_MODEL" in (await response.json())["error"]
            response = await client.put(
                "/comfy/story/films/project",
                json=payload,
                headers={"Origin": "https://unrelated.example"},
            )
            assert response.status == 403
            response = await client.post(
                "/comfy/story/films/soundtrack", json={"path": "score.wav"}
            )
            assert response.status == 200
            assert (await response.json())["duration_ms"] == 5000
            assert soundtrack_reads == [(tmp_path, "score.wav")]
            response = await client.post("/comfy/story/films/soundtrack", json={"path": 4})
            assert response.status == 400
            response = await client.post(
                "/comfy/story/films/soundtrack",
                json={"path": "score.wav"},
                headers={"Origin": "https://unrelated.example"},
            )
            assert response.status == 403
            assert len(soundtrack_reads) == 1
            response = await client.get(
                "/comfy/story/films/project/inputs-bundle?revision=" + saved["revision"]
            )
            assert response.status == 200
            archive = await response.read()
            assert response.headers["Content-Type"] == "application/zip"
            form = FormData()
            form.add_field("bundle", archive, filename="film.zip", content_type="application/zip")
            response = await client.post("/comfy/story/films/import-inputs", data=form)
            assert response.status == 200
            imported = await response.json()
            assert imported["assets_restored"] == 1
            assert imported["approvals_imported"] is False
            assert imported["recipe"]["plan"]["project_id"] != "project"
            assert imported["recipe"]["inputs"]["shots_by_id"]["one"]["variation"] == 7
            response = await client.post("/comfy/story/films/import-inputs", json={})
            assert response.status == 400
            response = await client.get("/comfy/story/films/missing")
            assert response.status == 404
            response = await client.get("/comfy/story/films")
            assert "project" in {row["project_id"] for row in (await response.json())["projects"]}

    try:
        asyncio.run(scenario())
    finally:
        runner._executor.shutdown(wait=True)


def test_film_generation_identity_tracks_inputs_models_and_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from comfy_story.memory import settings

    package = ModuleType("film_identity_test")
    package.__path__ = []
    adapter = ModuleType("film_identity_test.comfy_adapter")
    monkeypatch.setattr(adapter, "_story_root", lambda: tmp_path, raising=False)
    monkeypatch.setattr(
        adapter,
        "_generation_assets",
        lambda config: {"model": config["model"], "video_vae": "c" * 64},
        raising=False,
    )
    monkeypatch.setattr(adapter, "_story_sampler", lambda value: value, raising=False)
    monkeypatch.setattr(
        adapter,
        "render_configuration",
        lambda profile, sampler: {
            "model": "a" * 64,
            "lora": ("reference-eight" if profile == "Reference shot" else "animate-eight")
            if sampler == "Turbo 8-step"
            else ("turbo" if profile == "Reference shot" else "animate-turbo"),
        },
        raising=False,
    )

    requested = []

    def lora_digest(folder: str, name: str) -> str:
        requested.append((folder, name))
        return {"animate-eight": "8", "reference-eight": "9"}.get(name, "b") * 64

    monkeypatch.setattr(adapter, "_generation_asset_digest", lora_digest, raising=False)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    folders = ModuleType("folder_paths")
    folders.__file__ = str(tmp_path / "folder_paths.py")
    monkeypatch.setattr(folders, "get_input_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setitem(sys.modules, "folder_paths", folders)
    (tmp_path / "folder_paths.py").write_text("# Comfy source")
    (tmp_path / "frame.png").write_bytes(b"first image")
    filename = Path(__file__).resolve().parents[1] / "integrations/comfy_story/film_routes.py"
    spec = importlib.util.spec_from_file_location("film_identity_test.film_routes", filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recipe: dict[str, Any] = {
        "inputs": {
            "library": {"references": []},
            "shots_by_id": {"one": {"world": "frame.png", "sampler": "Native res_multistep"}},
        }
    }
    # Whole-film runs enforce the same required configuration as individual nodes.
    with monkeypatch.context() as environment:
        environment.delenv("COMFY_STORY_MEMORY", raising=False)
        environment.delenv("COMFY_STORY_MEMORY_CHECKPOINT", raising=False)
        with pytest.raises(ValueError, match="CHECKPOINT is required"):
            module._film_generation_identity(recipe)
        environment.setenv("COMFY_STORY_MEMORY", "native")
        with pytest.raises(ValueError, match="requires associative memory"):
            module._film_generation_identity(recipe)

    # Keep the required memory binding while isolating checkpoint I/O from asset identity tests.
    memory = settings.MemoryConfiguration(
        str(tmp_path / "memory.pt"), "b" * 64, "a" * 64, "d" * 64, "c" * 64
    )
    monkeypatch.setattr(settings, "configured_memory", lambda: memory)
    monkeypatch.setattr(
        settings.MemoryConfiguration, "load", lambda *_: SimpleNamespace(close=lambda: None)
    )
    native = module._film_generation_identity(recipe)
    assert native["associative_memory"] == memory.binding()
    assert requested == []  # Native-only installs do not require the Turbo adapter.
    assert "integration/nodes.py" in native["implementation"]
    recipe["inputs"]["shots_by_id"]["one"]["sampler"] = "Turbo 4-step"
    turbo = module._film_generation_identity(recipe)
    assert turbo["model_files"]["lora"] == "b" * 64
    assert requested == [("loras", "turbo")]
    recipe["inputs"]["shots_by_id"]["one"]["render_profile"] = "Animate frame"
    animated = module._film_generation_identity(recipe)
    assert animated["model_files"]["Animate frame:model"] == "a" * 64
    assert "model" not in animated["model_files"]
    assert requested[-1] == ("loras", "animate-turbo")
    recipe["inputs"]["shots_by_id"]["two"] = {
        "render_profile": "Animate frame",
        "sampler": "Turbo 8-step",
    }
    mixed = module._film_generation_identity(recipe)
    assert mixed["model_files"]["Animate frame:lora"] == "b" * 64
    assert mixed["model_files"]["Animate frame:lora:turbo-8step"] == "8" * 64
    recipe["inputs"]["shots_by_id"]["three"] = {
        "render_profile": "Reference shot",
        "sampler": "Turbo 8-step",
    }
    recipe["inputs"]["shots_by_id"]["four"] = {
        "render_profile": "Reference shot",
        "sampler": "Turbo 4-step",
    }
    all_profiles = module._film_generation_identity(recipe)
    assert all_profiles["model_files"]["lora"] == "b" * 64
    assert all_profiles["model_files"]["lora:turbo-8step"] == "9" * 64
    assert all_profiles["model_files"]["Animate frame:lora:turbo-8step"] == "8" * 64
    assert requested[-2:] == [("loras", "reference-eight"), ("loras", "turbo")]
    del recipe["inputs"]["shots_by_id"]["three"]
    del recipe["inputs"]["shots_by_id"]["four"]
    del recipe["inputs"]["shots_by_id"]["two"]
    del recipe["inputs"]["shots_by_id"]["one"]["render_profile"]
    (tmp_path / "frame.png").write_bytes(b"replacement image")
    image_changed = module._film_generation_identity(recipe)
    assert image_changed["input_files"] != turbo["input_files"]
    (tmp_path / "folder_paths.py").write_text("# Comfy source changed")
    source_changed = module._film_generation_identity(recipe)
    assert source_changed["implementation"] != image_changed["implementation"]
    from comfy_story import story_native_archive

    monkeypatch.setattr(story_native_archive, "NATIVE_REFERENCE_RUNTIME_SHA256", "d" * 64)
    assert module._film_generation_identity(recipe)["memory_runtime"] != native["memory_runtime"]
    recipe["inputs"]["recall_selected_state"] = True
    archive = module._film_generation_identity(recipe)
    assert "memory_checkpoint" not in archive
    assert len(archive["memory_runtime"]) == 64
    assert "staging" not in archive
    quality = ModuleType("film_identity_test.h3_quality_nodes")
    verified_nodes: list[object] = []
    monkeypatch.setattr(quality, "require_upscaler", verified_nodes.append, raising=False)
    monkeypatch.setitem(sys.modules, quality.__name__, quality)
    installed = {"MinimaxH3LatentUpscaler3D": object()}
    comfy_nodes = ModuleType("nodes")
    monkeypatch.setattr(comfy_nodes, "NODE_CLASS_MAPPINGS", installed, raising=False)
    monkeypatch.setitem(sys.modules, "nodes", comfy_nodes)
    recipe["inputs"]["shots_by_id"]["one"]["sampler"] = "Full HD 2-pass"
    full_hd = module._film_generation_identity(recipe)
    assert full_hd["model_files"]["lora:full-hd"] == "b" * 64
    assert verified_nodes == [installed]
    recipe["inputs"]["shots_by_id"]["one"]["sampler"] = "Turbo 4-step"
    recipe["inputs"]["shots_by_id"]["one"]["opening_prompt"] = "A new opening"
    staged = module._film_generation_identity(recipe)
    assert set(staged["staging"]["model_files"]) == {"diffusion_models", "text_encoders", "vae"}
    assert any(folder == "diffusion_models" for folder, _ in requested)
    monkeypatch.setattr(adapter, "_generation_asset_digest", lambda *_: "f" * 64)
    assert module._film_generation_identity(recipe)["staging"] != staged["staging"]
