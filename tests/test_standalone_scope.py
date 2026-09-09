"""Validate the installed package, commands, and source inventory."""

from __future__ import annotations

import ast
import hashlib
import importlib
import subprocess
import tomllib
from importlib.util import resolve_name
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_source_origin_checks_include_the_installed_namespace(tmp_path: Path) -> None:
    from comfy_story.ltx_quality_operations import (
        SourceArchiveEntry,
        SourceArchiveReceipt,
        verify_loaded_duet_module_origins,
    )

    encoded = b"value = 1\n"
    relative = "src/comfy_story/example.py"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(encoded)
    entry = SourceArchiveEntry(
        relative,
        "100644",
        0o644,
        len(encoded),
        hashlib.sha256(encoded).hexdigest(),
        hashlib.sha1(b"blob " + str(len(encoded)).encode() + b"\0" + encoded).hexdigest(),
    )
    receipt = SourceArchiveReceipt(
        "duet-x-ltx-source-archive-v1",
        "a" * 40,
        "b" * 40,
        "c" * 64,
        "d" * 64,
        1,
        (entry,),
    ).validate()
    modules = {"comfy_story.example": SimpleNamespace(__file__=str(source))}
    verify_loaded_duet_module_origins(
        tmp_path,
        receipt,
        required_modules=("comfy_story.example",),
        loaded_modules=modules,
    )
    source.write_bytes(b"value = 2\n")
    with pytest.raises(ValueError, match="source identity changed"):
        verify_loaded_duet_module_origins(
            tmp_path,
            receipt,
            required_modules=("comfy_story.example",),
            loaded_modules=modules,
        )
