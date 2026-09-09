from __future__ import annotations

import hashlib
import shutil
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import duet.duetx.film_audio_asset as audio_asset
from duet.duetx.film_audio_asset import inspect_soundtrack


def _wave(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\0\0" * 8000)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required")
def test_soundtrack_metadata_binds_actual_uploaded_audio(tmp_path: Path) -> None:
    path = tmp_path / "score.wav"
    _wave(path)
    result = inspect_soundtrack(tmp_path, "score.wav")
    assert {key: result[key] for key in ("path", "duration_ms", "sha256")} == {
        "path": "score.wav",
        "duration_ms": 1000,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.mark.skipif(
    not shutil.which("ffprobe") or not shutil.which("ffmpeg"), reason="FFmpeg required"
)
@pytest.mark.parametrize(
    ("sample", "full_scale", "silent"),
    [(b"\0\0", False, True), (b"\0\x40", False, False), (b"\xff\x7f", True, False)],
)
def test_levels_measure_actual_pcm_without_modifying_source(
    tmp_path: Path, sample: bytes, full_scale: bool, silent: bool
) -> None:
    path = tmp_path / "score.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(sample * 8000)
    original = path.read_bytes()
    result = inspect_soundtrack(tmp_path, "score.wav")
    levels = result["levels"]
    assert isinstance(levels, dict)
    assert levels["status"] == "measured"
    assert levels["silent"] is silent
    assert levels["reaches_full_scale"] is full_scale
    if not silent and not full_scale:
        assert levels["peak_dbfs"] == pytest.approx(-6.02, abs=0.01)
    assert path.read_bytes() == original
    assert result["sha256"] == hashlib.sha256(original).hexdigest()


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required")
def test_missing_level_analyzer_reports_unknown_without_claiming_clean_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "score.wav"
    _wave(path)
    original_run = cast(Callable[..., object], subprocess.run)

    def run(command: list[str], **kwargs: object) -> object:
        if command[0] == "ffmpeg":
            raise FileNotFoundError("not installed")
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    result = inspect_soundtrack(tmp_path, "score.wav")
    assert result["levels"] == {"status": "unavailable"}
    assert result["duration_ms"] == 1000


@pytest.mark.parametrize("name", ["../score.wav", "/tmp/score.wav", "score.txt"])
def test_soundtrack_lookup_cannot_read_arbitrary_host_files(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match=r"relative Comfy|unsupported audio"):
        inspect_soundtrack(tmp_path, name)


def test_soundtrack_lookup_rejects_symlinks_and_empty_files(tmp_path: Path) -> None:
    path = tmp_path / "score.wav"
    _wave(path)
    (tmp_path / "link.wav").symlink_to(path)
    with pytest.raises(ValueError, match="escapes"):
        inspect_soundtrack(tmp_path, "link.wav")
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="regular audio"):
        inspect_soundtrack(tmp_path, "score.wav")


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required")
def test_invalid_audio_is_not_added_as_a_soundtrack(tmp_path: Path) -> None:
    (tmp_path / "bad.wav").write_text("not audio")
    with pytest.raises(ValueError, match="could not inspect"):
        inspect_soundtrack(tmp_path, "bad.wav")


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required")
def test_soundtrack_change_during_lookup_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "score.wav"
    _wave(path)

    def changed_digest(source: Path) -> str:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with source.open("ab") as handle:
            handle.write(b"changed")
        return digest

    monkeypatch.setattr(audio_asset, "file_digest", changed_digest)
    with pytest.raises(ValueError, match="changed while reading"):
        inspect_soundtrack(tmp_path, "score.wav")
