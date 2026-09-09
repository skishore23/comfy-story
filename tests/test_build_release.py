from __future__ import annotations

import hashlib
import io
import json
import runpy
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable
from email.parser import BytesParser
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import NoReturn, cast

import pytest

from scripts.build_release import build_release_bundle

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_release_bundle_is_deterministic_and_contains_no_secrets(tmp_path: Path) -> None:
    first = build_release_bundle(REPOSITORY_ROOT, tmp_path / "one")
    second = build_release_bundle(REPOSITORY_ROOT, tmp_path / "two")
    assert first.read_bytes() == second.read_bytes()
    assert (
        first.with_suffix(first.suffix + ".sha256").read_text().split()[0]
        == hashlib.sha256(first.read_bytes()).hexdigest()
    )
    with zipfile.ZipFile(first) as archive:
        names = frozenset(archive.namelist())
        assert "INSTALL.md" in names
        assert "docs/getting-started.md" in names
        assert "install.py" in names
        assert "manifest.json" in names
        assert "custom_nodes/comfy_story/__init__.py" in names
        assert any(
            name.startswith("wheels/comfy_story-") and name.endswith(".whl") for name in names
        )
        assert not any("token" in name.casefold() or ".env" in name for name in names)
        manifest = json.loads(archive.read("manifest.json"))
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        install = archive.read("INSTALL.md").decode()
        installer = archive.read("install.py").decode()
        assert "docs/getting-started.md" in install
        assert "--check-only" in install
        assert "--with-ltx" not in installer
        assert "OpenImageIO" not in installer


def test_release_bundle_contains_living_canon_panel_and_minimax_install_contract(
    tmp_path: Path,
) -> None:
    bundle = build_release_bundle(REPOSITORY_ROOT, tmp_path)
    with zipfile.ZipFile(bundle) as archive:
        names = frozenset(archive.namelist())
        assert "custom_nodes/comfy_story/web/story_panel.js" in names
        assert "custom_nodes/comfy_story/web/story_state.mjs" in names
        assert "custom_nodes/comfy_story/example_workflows/two-shot-with-memory.json" in names
        install = archive.read("INSTALL.md").decode()
        assert "Add **Comfy Story**" in install
        assert "--no-deps" in install


def test_example_workflow_has_two_visible_nodes_and_dual_story_links() -> None:
    path = REPOSITORY_ROOT / "integrations/comfy_story/example_workflows/two-shot.json"
    workflow = json.loads(path.read_text())
    assert [node["type"] for node in workflow["nodes"]] == ["ComfyStory", "ComfyStory"]
    assert workflow["links"] == [
        [1, 10, 1, 20, 1, "IMAGE"],
        [2, 10, 2, 20, 0, "COMFY_STORY"],
    ]
    assert all(len(node["widgets_values"]) == 10 for node in workflow["nodes"])
    assert workflow["nodes"][0]["widgets_values"][2].endswith(".png")
    assert all(
        node["widgets_values"][-2:] == ["Native res_multistep", "[]"] for node in workflow["nodes"]
    )


def test_living_canon_example_uses_the_single_current_story_node() -> None:
    path = (
        REPOSITORY_ROOT / "integrations/comfy_story/example_workflows" / "two-shot-with-memory.json"
    )
    workflow = json.loads(path.read_text())

    assert [node["type"] for node in workflow["nodes"]] == [
        "ComfyStory",
        "ComfyStory",
    ]
    assert workflow["links"] == [
        [1, 10, 1, 20, 1, "IMAGE"],
        [2, 10, 2, 20, 0, "COMFY_STORY"],
    ]
    assert all(node["widgets_values"][-1] == "[]" for node in workflow["nodes"])
    assert all(len(node["widgets_values"]) == 10 for node in workflow["nodes"])
    assert workflow["nodes"][0]["widgets_values"][2].endswith(".png")


def test_release_contains_native_runtime_but_no_artifacts(tmp_path: Path) -> None:
    archive_path = build_release_bundle(REPOSITORY_ROOT, tmp_path)

    with zipfile.ZipFile(archive_path) as bundle:
        names = set(bundle.namelist())
        assert "custom_nodes/comfy_story/h3_reference_node.py" not in names
        wheel_name = next(name for name in names if name.startswith("wheels/comfy_story-"))
        with zipfile.ZipFile(io.BytesIO(bundle.read(wheel_name))) as wheel:
            wheel_names = set(wheel.namelist())
            metadata_name = next(
                name for name in wheel_names if name.endswith(".dist-info/METADATA")
            )
            metadata = BytesParser().parsebytes(wheel.read(metadata_name))
            assert metadata["Name"] == "comfy-story"
            assert metadata["License"] == "AGPL-3.0-only"
            license_root = metadata_name.removesuffix("METADATA") + "licenses/"
            for name, source in (
                ("LICENSE", "LICENSE"),
                ("NOTICE", "NOTICE"),
                ("LICENSING.md", "LICENSING.md"),
                ("Apache-2.0.txt", "licenses/Apache-2.0.txt"),
            ):
                assert wheel.read(license_root + name) == (REPOSITORY_ROOT / source).read_bytes()
            requirements = metadata.get_all("Requires-Dist") or []
            assert "torch>=2.8,<2.13" in requirements
            assert "cryptography>=43,<50" in requirements
            assert "Pillow>=10,<13" in requirements
            assert any(name.endswith("/licenses/LICENSE") for name in wheel_names)
            assert "comfy_story/h3_reference_compressors.py" not in wheel_names
            assert "comfy_story/h3_reference_checkpoint.py" not in wheel_names
            assert "comfy_story/h3_reference_benchmark.py" not in wheel_names
            assert "comfy_story/story_native_archive.py" in wheel_names
            assert "comfy_story/story_native_service.py" in wheel_names
            assert not any(name.startswith("comfy_story/ltx_") for name in wheel_names)
            assert (
                not {"comfy_story/training.py", "comfy_story/teacher.py", "comfy_story/losses.py"}
                & wheel_names
            )
            # Every local import must resolve within the installable wheel.
            import ast

            for name in sorted(wheel_names):
                if not name.endswith(".py"):
                    continue
                for item in ast.walk(ast.parse(wheel.read(name))):
                    modules = []
                    if isinstance(item, ast.Import):
                        modules = [alias.name for alias in item.names]
                    elif isinstance(item, ast.ImportFrom) and item.module:
                        modules = [item.module]
                    for module in modules:
                        if module == "comfy_story" or module.startswith("comfy_story."):
                            path = module.replace(".", "/")
                            assert {path + ".py", path + "/__init__.py"} & wheel_names, (
                                name,
                                module,
                            )
            entry_points = wheel.read(metadata_name.replace("METADATA", "entry_points.txt"))
            assert b"comfy-story-film = comfy_story.film_runner:main" in entry_points
            assert b"comfy-story-audit = comfy_story.film_audit_cli:main" in entry_points
        assert not any(name.startswith("artifacts/") for name in names)
        assert not any(name.endswith((".pt", ".safetensors", ".env")) for name in names)
        install = bundle.read("INSTALL.md").decode()
        assert "docs/getting-started.md" in install


def test_release_bundle_ships_product_guides_and_license(tmp_path: Path) -> None:
    bundle = build_release_bundle(REPOSITORY_ROOT, tmp_path)
    with zipfile.ZipFile(bundle) as archive:
        for name in (
            "getting-started.md",
            "prompting.md",
            "films.md",
        ):
            assert archive.read(f"docs/{name}") == (REPOSITORY_ROOT / "docs" / name).read_bytes()
        assert archive.read("LICENSE") == (REPOSITORY_ROOT / "LICENSE").read_bytes()
        for name in ("NOTICE", "LICENSING.md", "licenses/Apache-2.0.txt"):
            assert archive.read(name) == (REPOSITORY_ROOT / name).read_bytes()
        assert b"GNU AFFERO GENERAL PUBLIC LICENSE" in archive.read("LICENSE")


def _installer_fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    bundle = build_release_bundle(REPOSITORY_ROOT, tmp_path / "build")
    extracted = tmp_path / "bundle"
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(extracted)
    comfy = tmp_path / "ComfyUI"
    (comfy / "custom_nodes").mkdir(parents=True)
    (comfy / "main.py").write_text("# fixture")
    namespace = runpy.run_path(str(extracted / "install.py"))
    return namespace, comfy


def test_installer_check_only_does_not_install_or_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, comfy = _installer_fixture(tmp_path)
    main = cast(Callable[[], int], namespace["main"])
    monkeypatch.setitem(main.__globals__, "version", _compatible_version)
    monkeypatch.setattr(sys, "argv", ["install.py", "--comfy-root", str(comfy), "--check-only"])
    checks = []
    monkeypatch.setitem(main.__globals__, "check_runtime", lambda *args: checks.append(args))
    monkeypatch.setattr(subprocess, "run", _unexpected_mutation)
    monkeypatch.setattr(shutil, "copytree", _unexpected_mutation)
    assert main() == 0
    assert len(checks) == 1
    assert checks[0][1] == comfy.resolve()
    assert list((comfy / "custom_nodes").iterdir()) == []


def test_installer_rejects_missing_dependency_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, comfy = _installer_fixture(tmp_path)
    main = cast(Callable[[], int], namespace["main"])

    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setitem(main.__globals__, "version", missing)
    monkeypatch.setattr(sys, "argv", ["install.py", "--comfy-root", str(comfy)])
    monkeypatch.setattr(subprocess, "run", _unexpected_mutation)
    monkeypatch.setattr(shutil, "copytree", _unexpected_mutation)
    with pytest.raises(SystemExit, match="dependency preflight failed"):
        main()


def test_installer_rejects_existing_node_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, comfy = _installer_fixture(tmp_path)
    main = cast(Callable[[], int], namespace["main"])
    target = comfy / "custom_nodes" / "comfy_story"
    target.mkdir()
    (target / "keep.txt").write_text("existing installation")
    monkeypatch.setitem(main.__globals__, "version", _compatible_version)
    monkeypatch.setattr(sys, "argv", ["install.py", "--comfy-root", str(comfy)])
    monkeypatch.setattr(subprocess, "run", _unexpected_mutation)
    monkeypatch.setattr(shutil, "copytree", _unexpected_mutation)
    with pytest.raises(SystemExit, match="already exists"):
        main()
    assert (target / "keep.txt").read_text() == "existing installation"


def _unexpected_mutation(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("preflight must not install packages or copy nodes")


def _compatible_version(name: str) -> str:
    return {
        "aiohttp": "3.11.0",
        "cryptography": "49.0.0",
        "diffusers": "0.32.0",
        "einops": "0.8.0",
        "numpy": "2.2.6",
        "omegaconf": "2.3.0",
        "soundfile": "0.12.1",
        "torch": "2.12.1",
        "tqdm": "4.67.0",
        "Pillow": "12.0.0",
        "packaging": "24.0",
    }[name]


def test_template_browser_contains_only_workflows_and_keeps_film_examples(tmp_path: Path) -> None:
    bundle = build_release_bundle(REPOSITORY_ROOT, tmp_path)
    with zipfile.ZipFile(bundle) as archive:
        templates = [
            name
            for name in archive.namelist()
            if "/example_workflows/" in name and name.endswith(".json")
        ]
        assert len(templates) == 2
        for name in templates:
            workflow = json.loads(archive.read(name))
            assert isinstance(workflow["nodes"], list)
            assert workflow["nodes"]
            assert isinstance(workflow["links"], list)
            assert "shots" not in workflow
        for name in ("film-first-pass-plan.json", "film-first-pass-inputs.json"):
            assert f"custom_nodes/comfy_story/example_workflows/{name}" not in archive.namelist()
            assert (
                archive.read(f"custom_nodes/comfy_story/example_films/{name}")
                == (REPOSITORY_ROOT / "integrations/comfy_story/example_films" / name).read_bytes()
            )


@pytest.mark.parametrize(
    "name", ["../outside.py", "/outside.py", "x/../outside.py", "x\\outside.py", "x//file.py"]
)
def test_installer_rejects_manifest_path_escape_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    namespace, comfy = _installer_fixture(tmp_path)
    main = cast(Callable[[], int], namespace["main"])
    path = tmp_path / "bundle/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][name] = "a" * 64
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["install.py", "--comfy-root", str(comfy)])
    monkeypatch.setattr(subprocess, "run", _unexpected_mutation)
    with pytest.raises(SystemExit, match="manifest path"):
        main()


@pytest.mark.parametrize("linked", [False, True])
def test_installer_rejects_unlisted_code_and_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked: bool
) -> None:
    namespace, comfy = _installer_fixture(tmp_path)
    main = cast(Callable[[], int], namespace["main"])
    extra = tmp_path / "bundle/custom_nodes/comfy_story/unlisted.py"
    if linked:
        extra.symlink_to(tmp_path / "bundle/LICENSE")
    else:
        extra.write_text("raise RuntimeError('unreviewed local code')")
    monkeypatch.setattr(sys, "argv", ["install.py", "--comfy-root", str(comfy)])
    monkeypatch.setattr(subprocess, "run", _unexpected_mutation)
    with pytest.raises(SystemExit, match=r"symbolic links|file inventory"):
        main()


def test_distribution_ignores_untracked_code_and_rejects_tracked_links(tmp_path: Path) -> None:
    from scripts.build_release import _copy_integration

    root = tmp_path / "repo"
    source = root / "integrations/comfy_story"
    source.mkdir(parents=True)
    reviewed = source / "__init__.py"
    reviewed.write_text("# reviewed")
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    (source / "local_experiment.py").write_text("# not reviewed")
    (source / "notes.txt").write_text("private local notes")
    _copy_integration(root, tmp_path / "stage")
    assert sorted(
        path.name for path in (tmp_path / "stage/custom_nodes/comfy_story").iterdir()
    ) == ["__init__.py"]
    reviewed.unlink()
    reviewed.symlink_to(source / "local_experiment.py")
    with pytest.raises(ValueError, match="symbolic links"):
        _copy_integration(root, tmp_path / "linked-stage")


def test_release_cli_rejects_dirty_checkout_before_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import build_release as builder

    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: b" M source.py")
    monkeypatch.setattr(builder, "build_release_bundle", _unexpected_mutation)
    with pytest.raises(SystemExit, match="2"):
        builder.main(["--output-root", str(tmp_path)])


def test_bundle_inventory_includes_nested_manifest_files(tmp_path: Path) -> None:
    from scripts.build_release import _manifest

    nested = tmp_path / "example/manifest.json"
    nested.parent.mkdir()
    nested.write_text("{}")
    manifest = json.loads(_manifest(tmp_path, "a" * 40))
    assert manifest["files"]["example/manifest.json"] == hashlib.sha256(b"{}").hexdigest()


def test_ready_bundle_contains_only_the_pinned_memory_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "memory.pt"
    data = b"release packaging fixture, not trained weights"
    checkpoint.write_bytes(data)
    original = runpy.run_path

    def catalog(path: str) -> dict[str, object]:
        values = cast(dict[str, object], original(path))
        values.update(CHECKPOINT_SHA256=hashlib.sha256(data).hexdigest(), CHECKPOINT_SIZE=len(data))
        return values

    monkeypatch.setattr(runpy, "run_path", catalog)
    bundle = build_release_bundle(REPOSITORY_ROOT, tmp_path / "out", memory_checkpoint=checkpoint)
    with zipfile.ZipFile(bundle) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".whl"))
        with zipfile.ZipFile(io.BytesIO(archive.read(name))) as wheel:
            assert wheel.read("comfy_story/models/h3-associative-memory.pt") == data
    checkpoint.write_bytes(data[:-1] + b"!")
    with pytest.raises(ValueError, match="supported identity"):
        build_release_bundle(REPOSITORY_ROOT, tmp_path / "bad", memory_checkpoint=checkpoint)


def test_installer_installs_missing_dependencies_without_changing_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, comfy = _installer_fixture(tmp_path)
    main = cast(Callable[[], int], namespace["main"])

    def versions(name: str) -> str:
        if name == "cryptography":
            raise PackageNotFoundError(name)
        return _compatible_version(name)

    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> None:
        calls.append(command)
        if "-c" in command:
            assert Path(command[command.index("-c") + 1]).read_text() == "torch==2.12.1\n"

    monkeypatch.setitem(main.__globals__, "version", versions)
    monkeypatch.setitem(main.__globals__, "check_runtime", lambda *_: None)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["install.py", "--comfy-root", str(comfy)])
    assert main() == 0
    assert len(calls) == 2
    assert "cryptography<50,>=43" in calls[0]
    assert "--no-deps" in calls[1]
    assert (comfy / "custom_nodes/comfy_story/__init__.py").is_file()


def test_installer_preflight_failure_does_not_install_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, comfy = _installer_fixture(tmp_path)
    main = cast(Callable[[], int], namespace["main"])
    monkeypatch.setitem(main.__globals__, "version", _compatible_version)

    def reject(*args: object) -> NoReturn:
        raise ValueError("checkpoint failed authentication")

    monkeypatch.setitem(main.__globals__, "check_runtime", reject)
    monkeypatch.setattr(subprocess, "run", _unexpected_mutation)
    monkeypatch.setattr(sys, "argv", ["install.py", "--comfy-root", str(comfy)])
    with pytest.raises(ValueError, match="authentication"):
        main()
    assert not (comfy / "custom_nodes/comfy_story").exists()


def test_installer_resolves_shared_paths_before_temporary_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, comfy = _installer_fixture(tmp_path)
    preflight = cast(Callable[..., None], namespace["check_runtime"])
    wheel = next((tmp_path / "bundle/wheels").glob("*.whl"))
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> None:
        calls.append(command)
        assert kwargs["cwd"] != str(tmp_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", run)
    preflight(wheel, comfy, [Path("shared.yaml")])
    assert calls[0][-2:] == ["--extra-model-paths-config", str(tmp_path / "shared.yaml")]
