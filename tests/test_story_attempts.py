from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from comfy_story.story_attempts import StoryAttemptIndex
from comfy_story.story_contracts import ComfyStoryStateRef


def _state() -> ComfyStoryStateRef:
    return ComfyStoryStateRef(
        "story", "main", None, "1" * 64, "2" * 64, "3" * 64, 1, 1, "4" * 64, "5" * 64
    ).validate()


def test_completed_attempt_survives_restart_and_duplicate_publication(tmp_path: Path) -> None:
    index = StoryAttemptIndex(tmp_path)
    assert index.load("a" * 64) is None
    index.publish("a" * 64, _state())
    restarted = StoryAttemptIndex(tmp_path)
    assert restarted.load("a" * 64) == _state()
    restarted.publish("a" * 64, _state())
    assert restarted.load("b" * 64) is None
    with pytest.raises(ValueError, match="different take"):
        restarted.publish("a" * 64, replace(_state(), revision_sha256="6" * 64))


def test_completed_attempt_rejects_tampering_and_unsafe_keys(tmp_path: Path) -> None:
    index = StoryAttemptIndex(tmp_path)
    index.publish("a" * 64, _state())
    path = index.root / f"{'a' * 64}.json"
    path.write_bytes(path.read_bytes().replace(b'"main"', b'"other"'))
    with pytest.raises(ValueError, match="binding changed"):
        index.load("a" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        index.load("../escape")
