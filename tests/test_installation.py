from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from comfy_story import installation


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ComfyUI"
    (root / "comfy_extras").mkdir(parents=True)
    (root / "comfy_extras/nodes_minimax_h3.py").write_text("# H3 fixture")
    folders = ModuleType("folder_paths")

    def resolve(kind: str, name: str) -> str | None:
        path = root / "models" / kind / name
        return str(path) if path.is_file() else None

    monkeypatch.setattr(folders, "get_full_path", resolve, raising=False)
    monkeypatch.setitem(sys.modules, "folder_paths", folders)
    # Keep import-path changes local to this fixture.
    monkeypatch.setattr(sys, "path", list(sys.path))
    for kind, name in installation.REQUIRED_MODELS.items():
        folder = "vae" if kind.endswith("vae") else kind
        path = root / "models" / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model fixture")
    digest = hashlib.sha256(b"model fixture").hexdigest()
    memory = SimpleNamespace(
        foundation_sha256=digest,
        vae_sha256=digest,
        load=lambda _: SimpleNamespace(bridge=object(), close=lambda: None),
        binding=lambda: {"checkpoint_sha256": "a" * 64},
    )
    monkeypatch.setattr(installation, "configured_memory", lambda: memory)
    monkeypatch.setattr(
        installation, "history_effect_report", lambda *a, **kw: {"history_delta_rms": 1}
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/" + name)
    return root


def test_preflight_checks_models_and_reports_memory_separately_from_gpu(host: Path) -> None:
    report = installation.check_installation(host, [])
    assert report["checkpoint_authenticated"]
    assert len(report["model_files"]) == 4
    assert report["rendered_quality_validated"] is False


@pytest.mark.parametrize("kind", list(installation.REQUIRED_MODELS))
def test_preflight_reports_missing_models(host: Path, kind: str) -> None:
    folder = "vae" if kind.endswith("vae") else kind
    (host / "models" / folder / installation.REQUIRED_MODELS[kind]).unlink()
    with pytest.raises(ValueError, match="Required H3 models are missing"):
        installation.check_installation(host, [])


def test_preflight_rejects_incompatible_foundation(host: Path) -> None:
    (
        host / "models/diffusion_models" / installation.REQUIRED_MODELS["diffusion_models"]
    ).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="does not match"):
        installation.check_installation(host, [])


def test_preflight_uses_shared_model_configuration(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = ModuleType("utils.extra_config")
    calls: list[Any] = []
    monkeypatch.setattr(module, "load_extra_path_config", calls.append, raising=False)
    monkeypatch.setitem(sys.modules, "utils.extra_config", module)
    config = host / "shared.yaml"
    installation.check_installation(host, [config])
    assert calls == [str(config.resolve())]
