"""Guard the extraction boundary and the compatibility surface in the shipped source."""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import tomllib
from importlib.util import resolve_name
from pathlib import Path
from typing import Any

import duet
import duet.duetx

ROOT = Path(__file__).parents[1]


def test_product_modules_have_no_missing_local_imports() -> None:
    modules = {
        path.relative_to(ROOT / "src")
        .with_suffix("")
        .as_posix()
        .replace("/", ".")
        .removesuffix(".__init__")
        for path in (ROOT / "src").rglob("*.py")
    }
    paths = [
        *(ROOT / "src").rglob("*.py"),
        *(ROOT / "integrations/comfyui_duetx_continuity").rglob("*.py"),
    ]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if node.level:
                    if path.is_relative_to(ROOT / "src"):
                        package = (
                            path.relative_to(ROOT / "src")
                            .with_suffix("")
                            .as_posix()
                            .replace("/", ".")
                            .rsplit(".", 1)[0]
                        )
                    else:
                        continue
                    module = resolve_name("." * node.level + module, package)
                names = [module]
            for name in names:
                if name == "duet" or name.startswith("duet."):
                    assert name in modules, f"{path.relative_to(ROOT)} imports omitted {name}"


def test_every_retained_lazy_export_resolves() -> None:
    for package in (duet, duet.duetx):
        for name in package.__all__:
            assert getattr(package, name) is not None, name


def test_only_product_console_commands_are_installed() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = configuration["project"]
    assert project["name"] == "comfy-story"
    assert set(project["scripts"]) == {
        "comfy-story-film",
        "comfy-story-audit",
        "duet-story-film",
        "duet-story-audit",
    }
    for value in project["scripts"].values():
        module, name = value.split(":")
        assert callable(getattr(importlib.import_module(module), name))


def test_inventory_accounts_for_runtime_source_and_omitted_research() -> None:
    inventory: dict[str, Any] = json.loads((ROOT / "docs/SOURCE_INVENTORY.json").read_text())
    assert inventory["branch"] == "main"
    assert len(inventory["commit"]) == 40
    entries = inventory["files"]
    indexed = {row["path"]: row for row in entries}
    assert len(indexed) == len(entries)
    for row in entries:
        assert len(row["sha256"]) == 64
        if row["action"] == "retain":
            assert (ROOT / row["path"]).is_file(), row["path"]
    for path in (ROOT / "src").rglob("*.py"):
        assert indexed[path.relative_to(ROOT).as_posix()]["action"] == "retain"
    for prefix in ("paper/", "runpod/", "src/duet/salinas/", "src/duet/assembly101/"):
        assert any(row["path"].startswith(prefix) for row in entries)
        assert all(row["action"] == "omit" for row in entries if row["path"].startswith(prefix))


def test_no_generated_or_private_payload_is_tracked() -> None:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in filter(None, names):
        path = Path(name)
        assert not name.startswith(("artifacts/", "paper/", "runpod/", "data/", "assets/"))
        assert path.suffix not in {".mp4", ".wav", ".flac", ".pt", ".pth", ".safetensors", ".pyc"}
        assert not path.name.startswith(".env")
