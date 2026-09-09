"""Validate the installed package, commands, and source inventory."""

from __future__ import annotations

import ast
import importlib
import subprocess
import symtable
import tomllib
from importlib.util import resolve_name
from pathlib import Path

import comfy_story

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
        *(ROOT / "integrations/comfy_story").rglob("*.py"),
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
                if name == "comfy_story" or name.startswith("comfy_story."):
                    assert name in modules, f"{path.relative_to(ROOT)} imports omitted {name}"


def test_every_retained_lazy_export_resolves() -> None:
    for package in (comfy_story,):
        for name in package.__all__:
            assert getattr(package, name) is not None, name


def test_only_product_console_commands_are_installed() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = configuration["project"]
    assert project["name"] == "comfy-story"
    assert set(project["scripts"]) == {
        "comfy-story-film",
        "comfy-story-audit",
    }
    for value in project["scripts"].values():
        module, name = value.split(":")
        assert callable(getattr(importlib.import_module(module), name))


def test_no_generated_or_private_payload_is_tracked() -> None:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in filter(None, names):
        path = Path(name)
        assert not name.startswith(("artifacts/", "paper/", "runpod/", "data/", "assets/"))
        assert path.suffix not in {".mp4", ".wav", ".flac", ".pt", ".pth", ".safetensors", ".pyc"}
        assert not path.name.startswith(".env")


def test_local_imported_symbols_exist_without_host_test_stubs() -> None:
    # Integration fixtures replace host dependencies; a stale imported function
    # must still fail even when a test stub accidentally supplies that function.
    paths = [*(ROOT / "src").rglob("*.py"), *(ROOT / "integrations/comfy_story").rglob("*.py")]
    symbols = {
        path: {
            symbol.get_name()
            for symbol in symtable.symtable(path.read_text(), str(path), "exec").get_symbols()
            if symbol.is_assigned() or symbol.is_imported()
        }
        for path in paths
    }
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.level == 1:
                target = path.parent / (node.module.replace(".", "/") + ".py")
            elif node.level == 0 and node.module.startswith("comfy_story."):
                target = ROOT / "src" / (node.module.replace(".", "/") + ".py")
            else:
                continue
            assert target in symbols, f"{path.name} imports missing {target}"
            for alias in node.names:
                assert alias.name in symbols[target], (
                    f"{path.name} imports missing {node.module}.{alias.name}"
                )
