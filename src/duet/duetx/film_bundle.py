"""Portable film recipes plus uploaded inputs, without model files or approval credentials."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from duet.duetx.film_export import file_digest
from duet.duetx.film_io import film_plan_from_json, normalize_film_settings
from duet.duetx.film_project import FilmProjectStore
from duet.duetx.film_workflow import FilmShotInput, _input_file, compile_film_workflow
from duet.duetx.story_contracts import canonical_story_json

_FORMAT = "duet-film-inputs-v1"
_MAX_ASSET = 256 * 1024 * 1024
_MAX_TOTAL = 1024 * 1024 * 1024
_MAX_FILES = 512
_SUFFIXES = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
    "audio": {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".mp4", ".webm"},
}


def _slots(inputs: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str, str]]:
    for row in inputs["library"]["references"]:
        yield row, "file", "image"
    for row in inputs["shots_by_id"].values():
        if row.get("world") is not None:
            yield row, "world", "image"
        if row.get("audio_file") is not None:
            yield row, "audio_file", "audio"
        if row.get("ending_frame") is not None:
            yield row, "ending_frame", "image"
    for row in inputs.get("audio", []):
        yield row, "path", "audio"


def _asset_path(root: Path, name: str, kind: str) -> Path:
    _input_file(name)
    path = root / name
    if path.suffix.lower() not in _SUFFIXES[kind]:
        raise ValueError(f"unsupported {kind} input in film bundle")
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("film input path escapes the Comfy input directory")
    return path


def _copy(source: IO[bytes], destination: IO[bytes], limit: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    while block := source.read(min(1024 * 1024, limit - total + 1)):
        total += len(block)
        if total > limit:
            raise ValueError("film bundle asset exceeds the size limit")
        digest.update(block)
        destination.write(block)
    return digest.hexdigest(), total


def export_film_inputs(
    store: FilmProjectStore, project_id: str, revision: str, comfy_input: Path, destination: Path
) -> Path:
    """Export an immutable recipe and its input media; never include completed-take approvals."""
    root = comfy_input.resolve()
    recipe = store.load(project_id, revision)
    recipe.pop("revision")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("film inputs bundle already exists")
    with tempfile.TemporaryDirectory(prefix="duet-inputs-export-", dir=destination.parent) as temp:
        staging = Path(temp)
        assets: dict[str, dict[str, Any]] = {}
        total = 0
        for row, field, kind in _slots(recipe["inputs"]):
            path = _asset_path(root, row[field], kind)
            with path.open("rb") as source, (staging / "copy").open("wb") as target:
                digest, size = _copy(source, target, _MAX_ASSET)
            name = f"duet_story_inputs/{digest}{path.suffix.lower()}"
            if name not in assets:
                total += size
                if total > _MAX_TOTAL:
                    raise ValueError("film bundle assets exceed the total size limit")
                if len(assets) >= _MAX_FILES:
                    raise ValueError("film bundle has too many assets")
                member = f"assets/{Path(name).name}"
                (staging / "copy").rename(staging / Path(name).name)
                assets[name] = {"member": member, "sha256": digest, "size": size, "kind": kind}
            row[field] = name
        manifest = {
            "format": _FORMAT,
            "source_recipe_revision": revision,
            "recipe": recipe,
            "assets": assets,
        }
        archive_path = staging / "bundle.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            info = zipfile.ZipInfo("manifest.json", (1980, 1, 1, 0, 0, 0))
            encoded = canonical_story_json(manifest)
            if len(encoded) > 2 * 1024 * 1024:
                raise ValueError("film bundle manifest is too large")
            archive.writestr(info, encoded)
            for name in sorted(assets):
                asset = assets[name]
                info = zipfile.ZipInfo(asset["member"], (1980, 1, 1, 0, 0, 0))
                with (
                    (staging / Path(asset["member"]).name).open("rb") as source,
                    archive.open(info, "w") as output,
                ):
                    _copy(source, output, _MAX_ASSET)
        # Publish without replacing an existing file, including a concurrent export.
        os.link(archive_path, destination)
    return destination


def _import_film_inputs(
    store: FilmProjectStore, bundle: Path, comfy_input: Path, *, project_id: str | None = None
) -> dict[str, Any]:
    """Validate the entire bundle before installing inputs and creating a new saved project."""
    root = comfy_input.resolve()
    root.mkdir(parents=True, exist_ok=True)
    project_id = project_id or "film-" + str(uuid.uuid4())
    if store.project_path(project_id).exists():
        raise ValueError("choose a new project ID for importing a film")
    with tempfile.TemporaryDirectory(prefix="duet-inputs-import-", dir=root) as temp:
        staging = Path(temp)
        with zipfile.ZipFile(bundle) as archive:
            entries = archive.infolist()
            if any(
                e.flag_bits & 1 or e.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                for e in entries
            ):
                raise ValueError("film bundle uses unsupported encryption or compression")
            if len(entries) > _MAX_FILES + 1 or len({e.filename for e in entries}) != len(entries):
                raise ValueError("film bundle contains duplicate or excessive entries")
            if archive.getinfo("manifest.json").file_size > 2 * 1024 * 1024:
                raise ValueError("film bundle manifest is too large")
            manifest = json.loads(archive.read("manifest.json"))
            if not isinstance(manifest, dict) or manifest.get("format") != _FORMAT:
                raise ValueError("unsupported film inputs bundle")
            source_revision = manifest.get("source_recipe_revision")
            if (
                not isinstance(source_revision, str)
                or len(source_revision) != 64
                or any(c not in "0123456789abcdef" for c in source_revision)
            ):
                raise ValueError("invalid film bundle source revision")
            assets = manifest["assets"]
            recipe = manifest["recipe"]
            if not isinstance(assets, dict) or not isinstance(recipe, dict):
                raise ValueError("malformed film bundle manifest")
            recipe["plan"]["project_id"] = project_id
            plan = film_plan_from_json(json.dumps(recipe["plan"]))
            inputs = recipe["inputs"]
            normalized = normalize_film_settings(plan, inputs)
            compile_film_workflow(
                plan,
                normalized["library"],
                tuple(FilmShotInput(**s) for s in normalized["shots"]),
                allow_pending_staging=True,
                generated_audio=normalized.get("generated_audio", False),
            )
            required = {row[field]: kind for row, field, kind in _slots(inputs)}
            if set(required) != set(assets):
                raise ValueError("film bundle assets do not match its recipe")
            total = 0
            members = {"manifest.json"}
            for name, asset in assets.items():
                if not isinstance(asset, dict) or asset.get("kind") != required[name]:
                    raise ValueError("film bundle asset kind does not match its use")
                target = _asset_path(root, name, required[name])
                digest = asset["sha256"]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError("invalid film bundle asset digest")
                if name != f"duet_story_inputs/{digest}{target.suffix}":
                    raise ValueError("film bundle asset name is not content addressed")
                member = "assets/" + target.name
                if asset.get("member") != member:
                    raise ValueError("invalid film bundle member path")
                info = archive.getinfo(member)
                if type(asset.get("size")) is not int or info.file_size != asset["size"]:
                    raise ValueError("film bundle asset size changed")
                with archive.open(member) as source, (staging / target.name).open("wb") as output:
                    actual, size = _copy(source, output, min(_MAX_ASSET, _MAX_TOTAL - total))
                if actual != digest:
                    raise ValueError("film bundle asset integrity check failed")
                total += size
                members.add(member)
                if target.exists() and file_digest(target) != digest:
                    raise ValueError("existing Comfy input has conflicting content")
            if {e.filename for e in entries} != members:
                raise ValueError("film bundle contains unreferenced files")
        for name, asset in assets.items():
            target = _asset_path(root, name, asset["kind"])
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(staging / target.name, target)
            except FileExistsError:
                if target.is_symlink() or file_digest(target) != asset["sha256"]:
                    raise ValueError("existing Comfy input has conflicting content") from None
        saved = store.save(project_id, plan, inputs, expected_revision=None)
    return {
        "recipe": saved,
        "bundle_sha256": file_digest(bundle),
        "source_recipe_revision": manifest["source_recipe_revision"],
        "assets_restored": len(assets),
        "approvals_imported": False,
    }


def import_film_inputs(
    store: FilmProjectStore, bundle: Path, comfy_input: Path, *, project_id: str | None = None
) -> dict[str, Any]:
    """Restore a validated input bundle, reporting malformed uploads as input errors."""
    try:
        return _import_film_inputs(store, bundle, comfy_input, project_id=project_id)
    except (KeyError, TypeError, AttributeError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        raise ValueError("malformed film inputs bundle") from error
