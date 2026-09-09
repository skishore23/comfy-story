"""Immutable customer film recipes and conservative edit impact for the Comfy project editor."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from comfy_story.film_io import bind_film_settings, film_plan_from_json, normalize_film_settings
from comfy_story.film_plan import FilmPlan, compile_candidate_prompt
from comfy_story.film_workflow import FilmShotInput, compile_film_workflow
from comfy_story.story_contracts import canonical_story_json


class FilmProjectConflict(ValueError):
    """The draft changed since the caller read it; never overwrite another editor silently."""


def edit_impact(
    before: FilmPlan, before_inputs: object, after: FilmPlan, after_inputs: object
) -> dict[str, object]:
    """All Story shots carry the previous state, so invalidate a changed shot and its suffix.

    This deliberately does not claim independent reuse across a changed Story State parent.
    Soundtrack-only changes retain the exact generation recipes and need only another export.
    """
    old = normalize_film_settings(before, before_inputs)
    new = normalize_film_settings(after, after_inputs)
    export_only = {"audio", "burn_subtitles"}
    old_global = {k: v for k, v in old.items() if k not in export_only | {"shots"}}
    new_global = {k: v for k, v in new.items() if k not in export_only | {"shots"}}
    common = 0
    if (
        before.project_id == after.project_id
        and before.initial_facts == after.initial_facts
        and old_global == new_global
    ):
        for index, shot in enumerate(after.shots):
            if (
                index >= len(before.shots)
                or shot.digest != before.shots[index].digest
                or new["shots"][index] != old["shots"][index]
                or compile_candidate_prompt(before, index) != compile_candidate_prompt(after, index)
            ):
                break
            common += 1
    retained = [shot.shot_id for shot in after.shots[:common]]
    affected = [shot.shot_id for shot in after.shots[common:]]
    return {
        "reusable_prefix": retained,
        "requires_generation_review": affected,
        "removed_shots": [
            s.shot_id for s in before.shots if s.shot_id not in {x.shot_id for x in after.shots}
        ],
        "export_changed": before != after or old != new,
        "narrative_review_changed": before.narrative != after.narrative,
        "reason": "Changed shots invalidate all later Story State descendants. "
        "Unchanged recipes can recover saved clips; this is not a fresh-render pixel guarantee.",
    }


class FilmProjectStore:
    """Project drafts point to content-addressed recipes; generation consumes a frozen recipe."""

    def __init__(self, root: Path) -> None:
        if not root.is_absolute():
            raise ValueError("film project root must be absolute")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def project_path(self, project_id: str) -> Path:
        if (
            not isinstance(project_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", project_id) is None
        ):
            raise ValueError("invalid film project ID")
        path = self.root / project_id
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValueError("film project path escapes its root")
        return path

    def _inside(self, project: Path, *parts: str) -> Path:
        path = project.joinpath(*parts)
        if path.is_symlink() or not path.resolve().is_relative_to(project):
            raise ValueError("film project path escapes its project")
        return path

    @contextmanager
    def _lock(self, project: Path) -> Iterator[None]:
        project.mkdir(exist_ok=True)
        with self._inside(project, "edit.lock").open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def load(self, project_id: str, revision: str | None = None) -> dict[str, Any]:
        project = self.project_path(project_id)
        if revision is None:
            revision = self._inside(project, "current").read_text().strip()
        if re.fullmatch(r"[0-9a-f]{64}", revision) is None:
            raise ValueError("invalid film recipe revision")
        path = self._inside(project, "recipes", f"{revision}.json")
        if (
            path.resolve().parent != (project / "recipes").resolve()
            or path.stat().st_size > 2 * 1024 * 1024
        ):
            raise ValueError("invalid film recipe path or size")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != revision:
            raise ValueError("film recipe integrity check failed")
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("format") != "duet-film-project-v1":
            raise ValueError("unsupported film project recipe")
        return {**data, "revision": revision}

    def save(
        self, project_id: str, plan: FilmPlan, inputs: object, *, expected_revision: str | None
    ) -> dict[str, Any]:
        plan.validate()
        if plan.project_id != project_id:
            raise ValueError("film plan and project ID must agree")
        bound = bind_film_settings(plan, inputs)
        normalized = normalize_film_settings(plan, bound)
        for row in normalized["shots"]:
            if "opening_prompt" in row and (
                not isinstance(row["opening_prompt"], str) or not row["opening_prompt"].strip()
            ):
                raise ValueError("Stage then animate requires an opening composition")
        compile_film_workflow(
            plan,
            normalized.get("library", {}),
            tuple(FilmShotInput(**row) for row in normalized["shots"]),
            allow_pending_staging=True,
            generated_audio=normalized.get("generated_audio", False),
        )
        plan_data = asdict(plan)
        for row in plan_data["shots"]:
            if row["direction_version"] == 1:
                row.pop("direction_version")
        if not plan.state_definitions:
            plan_data.pop("state_definitions")
        if plan.narrative is None:
            plan_data.pop("narrative")
        payload = {"format": "duet-film-project-v1", "plan": plan_data, "inputs": bound}
        raw = canonical_story_json(payload)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("film project recipe exceeds 2 MiB")
        revision = hashlib.sha256(raw).hexdigest()
        project = self.project_path(project_id)
        with self._lock(project):
            current = self._inside(project, "current")
            actual = current.read_text().strip() if current.exists() else None
            if expected_revision != actual:
                raise FilmProjectConflict("film draft changed; reload before saving")
            recipes = self._inside(project, "recipes")
            recipes.mkdir(exist_ok=True)
            destination = self._inside(project, "recipes", f"{revision}.json")
            if destination.exists():
                if destination.read_bytes() != raw:
                    raise ValueError("film recipe integrity check failed")
            else:
                staging = self._inside(project, "recipes", f".{revision}.tmp")
                staging.write_bytes(raw)
                staging.replace(destination)
            temporary = self._inside(project, "current.tmp")
            temporary.write_text(revision)
            temporary.replace(current)
        return self.load(project_id, revision)

    def impact(
        self, project_id: str, before_revision: str, after_revision: str
    ) -> dict[str, object]:
        before = self.load(project_id, before_revision)
        after = self.load(project_id, after_revision)
        return edit_impact(
            film_plan_from_json(json.dumps(before["plan"])),
            before["inputs"],
            film_plan_from_json(json.dumps(after["plan"])),
            after["inputs"],
        )
