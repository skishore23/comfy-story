from __future__ import annotations

import json
import wave
import zipfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from comfy_story import film_bundle
from comfy_story.film_bundle import export_film_inputs, import_film_inputs
from comfy_story.film_export import file_digest
from comfy_story.film_plan import FilmPlan, FilmShot
from comfy_story.film_project import FilmProjectStore


def _source(tmp_path: Path) -> tuple[FilmProjectStore, str, Path]:
    inputs = tmp_path / "input"
    inputs.mkdir()
    Image.new("RGB", (16, 16), "blue").save(inputs / "world.png")
    with wave.open(str(inputs / "score.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\x00\x00" * 24000)
    store = FilmProjectStore(tmp_path / "store")
    plan = FilmPlan(
        "original", "Film", 5000, (), (FilmShot("one", 5000, "Show", "Hold", ("Box",)),)
    )
    saved = store.save(
        "original",
        plan,
        {
            "library": {
                "project_name": "Film",
                "references": [
                    {"name": "Box", "role": "Prop", "file": "world.png", "note": "Blue box"}
                ],
            },
            "shots": [{"world": "world.png", "variation": 2**64 - 1}],
            "audio": [
                {
                    "path": "score.wav",
                    "sha256": file_digest(inputs / "score.wav"),
                    "start_ms": 0,
                    "duration_ms": 1000,
                }
            ],
            "burn_subtitles": False,
        },
        expected_revision=None,
    )
    return store, saved["revision"], inputs


def _rewrite(source: Path, destination: Path, change: Any) -> None:
    with zipfile.ZipFile(source) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    change(manifest, entries)
    entries["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(destination, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def test_bundle_restores_exact_inputs_in_another_store_without_changing_seeds(
    tmp_path: Path,
) -> None:
    store, revision, inputs = _source(tmp_path)
    bundle = export_film_inputs(store, "original", revision, inputs, tmp_path / "film.zip")
    second = export_film_inputs(store, "original", revision, inputs, tmp_path / "second.zip")
    assert file_digest(bundle) == file_digest(second)
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert len(manifest["assets"]) == 2  # Shared reference/start image is deduplicated.
        assert not any(name.endswith((".pt", ".py", ".safetensors")) for name in archive.namelist())
    target = FilmProjectStore(tmp_path / "different-store")
    restored = import_film_inputs(
        target, bundle, tmp_path / "different-input", project_id="restored"
    )
    recipe = restored["recipe"]
    assert recipe["plan"]["project_id"] == "restored"
    assert recipe["inputs"]["shots_by_id"]["one"]["variation"] == 2**64 - 1
    for name, asset in manifest["assets"].items():
        assert file_digest(tmp_path / "different-input" / name) == asset["sha256"]
    assert restored["approvals_imported"] is False
    assert restored["assets_restored"] == 2
    assert store.load("original")["revision"] == revision


@pytest.mark.parametrize("damage", ["bytes", "traversal", "extra", "missing", "size", "revision"])
def test_invalid_bundle_installs_nothing(tmp_path: Path, damage: str) -> None:
    store, revision, inputs = _source(tmp_path)
    bundle = export_film_inputs(store, "original", revision, inputs, tmp_path / "film.zip")
    bad = tmp_path / "bad.zip"

    def corrupt(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        asset = next(iter(manifest["assets"].values()))
        if damage == "bytes":
            entries[asset["member"]] = b"bad"
        elif damage == "traversal":
            asset["member"] = "../escape.png"
        elif damage == "extra":
            entries["unrelated.txt"] = b"unrelated"
        elif damage == "missing":
            del entries[asset["member"]]
        elif damage == "revision":
            del manifest["source_recipe_revision"]
        else:
            asset["size"] += 1

    _rewrite(bundle, bad, corrupt)
    target = FilmProjectStore(tmp_path / "target")
    destination = tmp_path / "target-input"
    with pytest.raises(ValueError, match=r"film bundle|film inputs bundle"):
        import_film_inputs(target, bad, destination, project_id="new")
    assert not target.project_path("new").exists()
    assert list(destination.iterdir()) == []


def test_export_rejects_symlinked_external_media_and_keeps_existing_bundle(tmp_path: Path) -> None:
    store, revision, inputs = _source(tmp_path)
    original = inputs / "world.png"
    external = tmp_path / "private.png"
    original.rename(external)
    original.symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        export_film_inputs(store, "original", revision, inputs, tmp_path / "film.zip")
    assert not (tmp_path / "film.zip").exists()
    existing = tmp_path / "existing.zip"
    existing.write_bytes(b"retain")
    with pytest.raises(FileExistsError):
        export_film_inputs(store, "original", revision, inputs, existing)
    assert existing.read_bytes() == b"retain"


def test_import_does_not_replace_conflicting_input_or_existing_project(tmp_path: Path) -> None:
    store, revision, inputs = _source(tmp_path)
    bundle = export_film_inputs(store, "original", revision, inputs, tmp_path / "film.zip")
    with pytest.raises(ValueError, match="new project"):
        import_film_inputs(store, bundle, inputs, project_id="original")
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    destination = tmp_path / "target-input"
    conflict = destination / next(iter(manifest["assets"]))
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"keep")
    with pytest.raises(ValueError, match="conflicting"):
        import_film_inputs(FilmProjectStore(tmp_path / "target"), bundle, destination)
    assert conflict.read_bytes() == b"keep"


def test_asset_size_budget_is_enforced_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, revision, inputs = _source(tmp_path)
    monkeypatch.setattr(film_bundle, "_MAX_ASSET", 10)
    with pytest.raises(ValueError, match="size limit"):
        export_film_inputs(store, "original", revision, inputs, tmp_path / "film.zip")
    assert not (tmp_path / "film.zip").exists()


def test_duplicate_inputs_count_once_toward_total_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, revision, inputs = _source(tmp_path)
    monkeypatch.setattr(film_bundle, "_MAX_TOTAL", sum(p.stat().st_size for p in inputs.iterdir()))
    bundle = export_film_inputs(store, "original", revision, inputs, tmp_path / "film.zip")
    assert bundle.is_file()


def test_ending_image_survives_portable_bundle_import(tmp_path: Path) -> None:
    from comfy_story.film_io import film_plan_from_json

    store, revision, inputs = _source(tmp_path)
    Image.new("RGB", (16, 16), "red").save(inputs / "ending.png")
    recipe = store.load("original", revision)
    recipe["inputs"]["shots_by_id"]["one"]["ending_frame"] = "ending.png"
    saved = store.save(
        "original",
        film_plan_from_json(json.dumps(recipe["plan"])),
        recipe["inputs"],
        expected_revision=revision,
    )
    archive = export_film_inputs(
        store, "original", saved["revision"], inputs, tmp_path / "bundle.zip"
    )
    target = tmp_path / "target-input"
    target.mkdir()
    imported = import_film_inputs(FilmProjectStore(tmp_path / "new-store"), archive, target)
    name = imported["recipe"]["inputs"]["shots_by_id"]["one"]["ending_frame"]
    assert file_digest(target / name) == file_digest(inputs / "ending.png")


def test_staging_recipe_round_trips_without_rendering(tmp_path: Path) -> None:
    from dataclasses import replace

    from comfy_story.film_io import film_plan_from_json

    store, revision, inputs = _source(tmp_path)
    saved = store.load("original", revision)
    plan = film_plan_from_json(json.dumps(saved["plan"]))
    plan = replace(plan, shots=(replace(plan.shots[0], composition="Continue frame"),))
    row = saved["inputs"]["shots_by_id"][plan.shots[0].shot_id]
    row.update(
        opening_prompt="A landscape before movement",
        opening_attempts=2,
        render_profile="Animate frame",
        intent="New Scene",
    )
    staged = store.save("original", plan, saved["inputs"], expected_revision=revision)
    bundle = export_film_inputs(
        store, "original", staged["revision"], inputs, tmp_path / "staged.zip"
    )
    target = FilmProjectStore(tmp_path / "target")
    imported = import_film_inputs(target, bundle, tmp_path / "target-input", project_id="restored")
    restored = imported["recipe"]["inputs"]["shots_by_id"][plan.shots[0].shot_id]
    assert restored["opening_prompt"] == row["opening_prompt"]
    assert restored["opening_attempts"] == 2
    assert restored["variation"] == row["variation"]
