"""Background customer production over immutable project recipes, shared by the Comfy panel."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from comfy_story.film_audit import VISUAL_AUDIT_PROTOCOL
from comfy_story.film_audit_cli import _model_identity
from comfy_story.film_export import file_digest
from comfy_story.film_opening_review import opening_candidates, starting_state_review
from comfy_story.film_production import ProductionPaused, run_production
from comfy_story.film_project import FilmProjectStore
from comfy_story.film_runner import _write, run_film
from comfy_story.film_starting_state import STARTING_STATE_PROTOCOL
from comfy_story.story_contracts import canonical_story_json


def _runtime_identity() -> dict[str, object]:
    package_root = Path(__file__).parent
    modules = sorted(path for path in package_root.rglob("*.py") if "__pycache__" not in path.parts)
    packages = {}
    # Attention, decoding, and verifier kernels can change independently of Comfy source.
    # Record missing optional packages too: installing one changes the available runtime.
    for name in (
        "comfy-story",
        "torch",
        "transformers",
        "numpy",
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
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    return {
        "python": list(sys.version_info[:3]),
        "platform": sys.platform,
        "machine": platform.machine(),
        "packages": packages,
        "implementation_sha256s": {
            path.relative_to(package_root).as_posix(): file_digest(path) for path in modules
        },
    }


class FilmProjectRunner:
    """One active film per service; a browser closing does not own or terminate the worker."""

    def __init__(
        self,
        store: FilmProjectStore,
        server: str,
        comfy_input: Path,
        model: Path | None,
        *,
        device: str = "cpu",
        generation_identity: Callable[[dict[str, Any]], dict[str, object]] | None = None,
        story_root: Path | None = None,
    ) -> None:
        self.store = store
        self.server = server
        self.comfy_input = comfy_input.resolve()
        self.model = model
        self.device = device
        self.generation_identity = generation_identity
        self.story_root = story_root
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="comfy-film")
        self._mutex = threading.Lock()
        self._active: Future[None] | None = None
        self._active_key: tuple[str, str] | None = None

    def start(
        self,
        project_id: str,
        revision: str,
        max_attempts: int = 2,
        *,
        expected_run_id: str | None = None,
        mode: str = "checked",
    ) -> dict[str, Any]:
        if type(max_attempts) is not int or not 1 <= max_attempts <= 4:
            raise ValueError("attempt budget must be an integer from 1 to 4")
        recipe = self.store.load(project_id, revision)
        if not isinstance(mode, str) or mode not in {"first_cut", "checked"}:
            raise ValueError("generation mode must be first_cut or checked")
        if mode == "first_cut":
            inputs = recipe["inputs"]
            if inputs.get("recall_selected_state") or any(
                "opening_prompt" in shot for shot in inputs["shots_by_id"].values()
            ):
                raise ValueError(
                    "Staged openings and selected-state recall require Check each shot in "
                    "Project tools. For a first cut, use uploaded scenes or references."
                )
        if mode == "checked" and self.model is None:
            raise ValueError(
                "Configure COMFY_STORY_VERIFY_MODEL on the Comfy host before verified generation"
            )
        policy = {
            "revision": revision,
            "max_attempts": max_attempts,
            "server": self.server,
            "device": self.device,
            "runtime": _runtime_identity(),
        }
        if mode == "checked":
            assert self.model is not None
            policy.update(
                audit_protocol=VISUAL_AUDIT_PROTOCOL,
                starting_state_protocol=STARTING_STATE_PROTOCOL,
                model_sha256=_model_identity(self.model),
            )
        else:
            policy["mode"] = "first_cut"
        if self.generation_identity is not None:
            policy["generation"] = self.generation_identity(recipe)
        key = hashlib.sha256(canonical_story_json(policy)).hexdigest()
        if expected_run_id is not None and expected_run_id != key:
            raise ValueError(
                "Generation inputs, models, runtime or verification policy changed since this run. "
                "Choose Generate saved plan to start a new run explicitly."
            )
        directory = self._run_path(project_id, key)
        with self._mutex:
            if self._active is not None and not self._active.done():
                if self._active_key == (project_id, key):
                    return self.status(project_id, key)
                raise ValueError("another film is running; wait for it to finish")
            directory.mkdir(parents=True, exist_ok=True)
            policy_path = directory / "policy.json"
            if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
                raise ValueError("frozen film run policy changed")
            _write(policy_path, policy)
            inputs = dict(recipe["inputs"])
            audio = []
            for row in inputs.get("audio", []):
                relative = Path(row["path"])
                path = (self.comfy_input / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(self.comfy_input):
                    raise ValueError("project audio must name an uploaded Comfy input file")
                audio.append({**row, "path": str(path)})
            if "audio" in inputs:
                inputs["audio"] = audio
            for name, payload in (("plan", recipe["plan"]), ("inputs", inputs)):
                target = directory / f"{name}.json"
                if target.exists() and target.read_bytes() != canonical_story_json(payload):
                    raise ValueError("frozen film run inputs changed")
                _write(target, payload)
            production = directory / "production"
            production.mkdir(exist_ok=True)
            (production / "pause.requested").unlink(missing_ok=True)
            _write(
                directory / "worker.json",
                {"status": "running", "revision": revision, "phase": "checking_inputs"},
            )
            self._active_key = (project_id, key)
            self._active = self._executor.submit(self._run, directory, max_attempts)
        return self.status(project_id, key)

    def _run_path(self, project_id: str, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid film run ID")
        project = self.store.project_path(project_id)
        path = project / "runs" / key
        if not path.resolve().is_relative_to(project):
            raise ValueError("film run path escapes its project")
        return path

    def _run(self, directory: Path, max_attempts: int) -> None:
        try:
            policy = json.loads((directory / "policy.json").read_text())
            if policy.get("mode") == "first_cut":
                self._first_cut(directory)
                return
            if self.model is None:
                raise ValueError("visual verifier is not configured")
            extra: dict[str, Any] = {}
            staging = policy.get("generation", {}).get("staging")
            if staging is not None:
                extra["staging_identity"] = staging
            result = run_production(
                directory / "plan.json",
                directory / "inputs.json",
                self.server,
                directory / "production",
                model=self.model,
                comfy_input=self.comfy_input,
                device=self.device,
                max_attempts=max_attempts,
                story_root=self.story_root,
                **extra,
            )
            state: dict[str, Any] = {
                "status": "ready_for_review",
                "phase": "complete",
                "film": str(result),
            }
        except ProductionPaused as error:
            state = {"status": "paused", "reason": str(error)}
        except Exception as error:
            # Background failures must be visible after a browser reconnect, never lost in a Future.
            state = {"status": "needs_attention", "reason": str(error)}
        worker = json.loads((directory / "worker.json").read_text())
        state.setdefault("phase", worker.get("phase", "checking_inputs"))
        _write(directory / "worker.json", state)

    def _first_cut(self, directory: Path) -> None:
        """Deliver the complete unreviewed draft through the public Story nodes."""
        production = directory / "production"
        _write(directory / "worker.json", {"status": "running", "phase": "rendering_film"})
        result = run_film(
            directory / "plan.json", directory / "inputs.json", self.server, production
        )
        # The renderer validates duration and ancestry before export. These are rendered
        # takes, never machine-selected evidence or approved associative memory.
        rendered = json.loads((production / "rendered-takes.json").read_text())
        _write(
            production / "production.json",
            {
                "status": "draft_ready",
                "phase": "complete",
                "film": str(result),
                "film_sha256": file_digest(result),
                "rendered": rendered,
                "selected": [],
            },
        )
        _write(directory / "worker.json", {"status": "draft_ready", "phase": "complete"})

    def pause(self, project_id: str, key: str) -> dict[str, Any]:
        directory = self._run_path(project_id, key)
        if not (directory / "worker.json").exists():
            raise FileNotFoundError("film run does not exist")
        policy = json.loads((directory / "policy.json").read_text())
        if policy.get("mode") == "first_cut":
            raise ValueError("first cuts run as one Comfy job; shot-boundary pause is unavailable")
        (directory / "production" / "pause.requested").write_text("pause after current shot")
        return self.status(project_id, key)

    def status(self, project_id: str, key: str) -> dict[str, Any]:
        directory = self._run_path(project_id, key)
        active = (
            self._active_key == (project_id, key)
            and self._active is not None
            and not self._active.done()
        )
        worker = json.loads((directory / "worker.json").read_text())
        policy = json.loads((directory / "policy.json").read_text())
        production_path = directory / "production" / "production.json"
        production = json.loads(production_path.read_text()) if production_path.exists() else {}
        state = worker.get("status", "needs_attention")
        if state == "running" and not active:
            state = "interrupted"
        preview = production.get("preview")
        # Use the immutable run recipe, not the editor's current (possibly longer) plan.
        plan = self.store.load(project_id, policy["revision"])["plan"]
        selected = production.get("selected", [])
        prefix = []
        for shot, take in zip(plan["shots"], selected, strict=False):
            if not isinstance(take, dict) or take.get("shot_id") != shot["shot_id"]:
                break
            prefix.append(shot)
        return {
            "run_id": key,
            "revision": policy["revision"],
            "mode": policy.get("mode", "checked"),
            "status": state,
            "phase": production.get("phase") or worker.get("phase"),
            "pause_requested": (directory / "production" / "pause.requested").exists(),
            "max_attempts": policy["max_attempts"],
            "shot_id": production.get("shot_id"),
            "attempt": production.get("attempt"),
            **(
                {"opening_attempt": production["opening_attempt"]}
                if "opening_attempt" in production
                else {}
            ),
            "rendered": production.get("rendered", []),
            "selected": production.get("selected", []),
            "coverage": {
                "target_duration_ms": plan["target_duration_ms"],
                "planned_duration_ms": sum(shot["duration_ms"] for shot in plan["shots"]),
                "planned_shots": len(plan["shots"]),
                "selected_duration_ms": sum(shot["duration_ms"] for shot in prefix),
                "selected_shots": len(prefix),
            },
            "preview": {key: preview.get(key) for key in ("shot_id", "attempt", "duration_ms")}
            if isinstance(preview, dict)
            else None,
            "reason": worker.get("reason") or production.get("reason"),
            "starting_state_review": starting_state_review(directory, self.comfy_input, production)
            if state != "running" and production.get("phase") == "checking_inputs"
            else None,
            "opening_candidates": opening_candidates(directory, self.comfy_input, production)
            if state != "running"
            and production.get("phase") in {"staging_opening", "checking_opening"}
            else [],
            "approval_created": False,
        }

    def runs(self, project_id: str) -> list[dict[str, Any]]:
        root = self.store.project_path(project_id) / "runs"
        return [
            self.status(project_id, path.parent.name)
            for path in sorted(
                root.glob("*/worker.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True
            )
        ]

    def video(self, project_id: str, key: str) -> Path:
        directory = self._run_path(project_id, key)
        production = json.loads((directory / "production" / "production.json").read_text())
        if production.get("status") not in {"ready_for_review", "draft_ready"}:
            raise ValueError("film is not ready for review")
        return self._checked_video(directory, production)

    def preview_video(self, project_id: str, key: str) -> Path:
        directory = self._run_path(project_id, key)
        path = directory / "production" / "production.json"
        production = json.loads(path.read_text())
        preview = production.get("preview")
        if not isinstance(preview, dict):
            raise ValueError("a candidate film preview is not available")
        return self._checked_video(directory, preview)

    @staticmethod
    def _checked_video(directory: Path, record: dict[str, Any]) -> Path:
        filename, digest = record.get("film"), record.get("film_sha256")
        if not isinstance(filename, str) or not isinstance(digest, str):
            raise ValueError("film output record is incomplete")
        result = Path(filename).resolve()
        if not result.is_relative_to(directory.resolve()) or not result.is_file():
            raise ValueError("film output is outside its run")
        if file_digest(result) != digest:
            raise ValueError("film output integrity check failed")
        return result
