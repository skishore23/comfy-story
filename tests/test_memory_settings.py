from __future__ import annotations

import os
from pathlib import Path

import pytest

from comfy_story.memory.settings import configured_memory


@pytest.fixture
def memory_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    for name in tuple(os.environ):
        if name.startswith("COMFY_STORY_MEMORY"):
            monkeypatch.delenv(name)
    checkpoint = tmp_path / "memory.pt"
    checkpoint.write_bytes(b"configuration-only fixture; runtime authenticates weights separately")
    values = {
        "COMFY_STORY_MEMORY_CHECKPOINT": str(checkpoint),
        "COMFY_STORY_MEMORY_CHECKPOINT_SHA256": "1" * 64,
        "COMFY_STORY_MEMORY_FOUNDATION_SHA256": "2" * 64,
        "COMFY_STORY_MEMORY_MODEL_CONFIGURATION_SHA256": "3" * 64,
        "COMFY_STORY_MEMORY_VAE_SHA256": "4" * 64,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


def test_associative_memory_is_required_without_an_opt_in(
    memory_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    implicit = configured_memory()
    assert implicit.checkpoint == memory_environment["COMFY_STORY_MEMORY_CHECKPOINT"]
    monkeypatch.setenv("COMFY_STORY_MEMORY", "associative")
    assert configured_memory() == implicit


@pytest.mark.parametrize("mode", ["native", "off", "none", "", "typo"])
def test_memory_cannot_be_disabled(
    memory_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("COMFY_STORY_MEMORY", mode)
    with pytest.raises(ValueError, match="requires associative memory"):
        configured_memory()


@pytest.mark.parametrize(
    "suffix",
    [
        "CHECKPOINT",
        "CHECKPOINT_SHA256",
        "FOUNDATION_SHA256",
        "MODEL_CONFIGURATION_SHA256",
        "VAE_SHA256",
    ],
)
def test_each_checkpoint_pin_is_required(
    memory_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    key = "COMFY_STORY_MEMORY_" + suffix
    monkeypatch.delenv(key)
    with pytest.raises(ValueError, match=key + " is required"):
        configured_memory()


def test_required_configuration_rejects_missing_checkpoint(
    memory_environment: dict[str, str],
) -> None:
    Path(memory_environment["COMFY_STORY_MEMORY_CHECKPOINT"]).unlink()
    with pytest.raises(ValueError, match="absolute regular file"):
        configured_memory()
