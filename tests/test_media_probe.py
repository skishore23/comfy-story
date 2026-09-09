from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from comfy_story import media_probe
from comfy_story.media_probe import (
    _run_linux_descriptor_stable_media_tool,
    probe_decoded_video,
)


def test_explicit_ffprobe_path_uses_windows_safe_rehashed_file_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = (tmp_path / "output.mp4").resolve()
    ffprobe = (tmp_path / "ffprobe.exe").resolve()
    media.write_bytes(b"video")
    ffprobe.write_bytes(b"executable")
    seen: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            b'{"format":{"duration":"1.0416667"},"programs":[],"streams":[{"avg_frame_rate":"24/1","height":384,"nb_read_frames":"25","width":384}]}',
            b"",
        )

    monkeypatch.setattr(subprocess, "run", run)
    facts = probe_decoded_video(
        media,
        ffprobe_path=ffprobe,
        ffprobe_sha256=hashlib.sha256(ffprobe.read_bytes()).hexdigest(),
    )

    assert facts.frames == 25
    assert seen[0][0][0] == str(ffprobe)
    assert seen[0][0][-1] == str(media)
    assert "pass_fds" not in seen[0][1]


def test_descriptor_stable_ffprobe_uses_authenticated_executable_and_media_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (tmp_path / "source.mp4").resolve()
    executable = (tmp_path / "ffprobe").resolve()
    source.write_bytes(b"reviewed media")
    executable.write_bytes(b"reviewed ffprobe")
    executable.chmod(0o700)
    observed: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed.append(command)
        executable_fd, media_fd = cast(tuple[int, int], kwargs["pass_fds"])
        assert command[0] == f"/proc/self/fd/{executable_fd}"
        assert command[-1] == f"/proc/self/fd/{media_fd}"
        assert os.pread(executable_fd, 1024, 0) == b"reviewed ffprobe"
        assert os.pread(media_fd, 1024, 0) == b"reviewed media"
        return subprocess.CompletedProcess(
            command,
            0,
            b'{"format":{"duration":"1.0416667"},"programs":[],"streams":[{"avg_frame_rate":"24/1","height":384,"nb_read_frames":"25","width":384}]}',
            b"",
        )

    monkeypatch.setattr(media_probe, "_linux_descriptor_execution_available", lambda: True)
    monkeypatch.setattr(subprocess, "run", run)
    facts = probe_decoded_video(
        source,
        ffprobe_path=executable,
        ffprobe_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        descriptor_stable=True,
    )

    assert facts.frames == 25
    assert len(observed) == 1
    assert str(executable) not in observed[0]
    assert str(source) not in observed[0]


@pytest.mark.parametrize("programs", [[{}], {}, None, ""])
def test_ffprobe_programs_field_rejects_every_nonempty_or_malformed_value(
    programs: object,
) -> None:
    payload = {
        "format": {"duration": "1.0416667"},
        "programs": programs,
        "streams": [
            {
                "avg_frame_rate": "24/1",
                "height": 384,
                "nb_read_frames": "25",
                "width": 384,
            }
        ],
    }
    completed = subprocess.CompletedProcess(("ffprobe",), 0, json.dumps(payload).encode(), b"")

    with pytest.raises(ValueError, match="programs"):
        media_probe._decoded_media_facts(completed)


def test_ffprobe_44_payload_rejects_unknown_top_level_field() -> None:
    payload = {
        "chapters": [],
        "format": {"duration": "1.0416667"},
        "programs": [],
        "streams": [
            {
                "avg_frame_rate": "24/1",
                "height": 384,
                "nb_read_frames": "25",
                "width": 384,
            }
        ],
    }
    completed = subprocess.CompletedProcess(("ffprobe",), 0, json.dumps(payload).encode(), b"")

    with pytest.raises(ValueError, match="schema"):
        media_probe._decoded_media_facts(completed)


def test_legacy_ffprobe_payload_without_programs_remains_accepted() -> None:
    payload = {
        "format": {"duration": "1.0416667"},
        "streams": [
            {
                "avg_frame_rate": "24/1",
                "height": 384,
                "nb_read_frames": "25",
                "width": 384,
            }
        ],
    }
    completed = subprocess.CompletedProcess(("ffprobe",), 0, json.dumps(payload).encode(), b"")

    facts = media_probe._decoded_media_facts(completed)

    assert facts == media_probe.DecodedMediaFacts(384, 384, 25, 24.0, 1.0416667)


def test_ffprobe_44_payload_rejects_duplicate_programs_key() -> None:
    completed = subprocess.CompletedProcess(
        ("ffprobe",),
        0,
        b'{"format":{"duration":"1.0416667"},"programs":[],"programs":[],"streams":[{"avg_frame_rate":"24/1","height":384,"nb_read_frames":"25","width":384}]}',
        b"",
    )

    with pytest.raises(ValueError, match="malformed strict JSON"):
        media_probe._decoded_media_facts(completed)


def test_descriptor_stable_ffprobe_never_executes_or_reads_rename_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (tmp_path / "source.mp4").resolve()
    executable = (tmp_path / "ffprobe").resolve()
    source.write_bytes(b"reviewed media")
    executable.write_bytes(b"reviewed ffprobe")
    executable.chmod(0o700)
    expected_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    executed = False

    def run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal executed
        executed = True
        executable.rename(tmp_path / "original-ffprobe")
        executable.write_bytes(b"replacement executable")
        executable.chmod(0o700)
        source.rename(tmp_path / "original-media.mp4")
        source.write_bytes(b"replacement media")
        executable_fd, media_fd = cast(tuple[int, int], kwargs["pass_fds"])
        assert os.pread(executable_fd, 1024, 0) == b"reviewed ffprobe"
        assert os.pread(media_fd, 1024, 0) == b"reviewed media"
        return subprocess.CompletedProcess(command, 0, b"{}", b"")

    monkeypatch.setattr(media_probe, "_linux_descriptor_execution_available", lambda: True)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="path changed during descriptor-stable execution"):
        probe_decoded_video(
            source,
            ffprobe_path=executable,
            ffprobe_sha256=expected_sha256,
            descriptor_stable=True,
        )
    assert executed


def test_descriptor_stable_media_tool_fails_closed_off_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (tmp_path / "source.mp4").resolve()
    executable = (tmp_path / "ffprobe").resolve()
    source.write_bytes(b"reviewed media")
    executable.write_bytes(b"reviewed ffprobe")
    executable.chmod(0o700)
    monkeypatch.setattr(media_probe, "_linux_descriptor_execution_available", lambda: False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("non-Linux stable mode reached subprocess"),
    )

    with pytest.raises(ValueError, match="requires Linux"):
        probe_decoded_video(
            source,
            ffprobe_path=executable,
            ffprobe_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
            descriptor_stable=True,
        )


def test_descriptor_stable_ffprobe_rejects_wrong_digest_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (tmp_path / "source.mp4").resolve()
    executable = (tmp_path / "ffprobe").resolve()
    source.write_bytes(b"reviewed media")
    executable.write_bytes(b"reviewed ffprobe")
    executable.chmod(0o700)
    monkeypatch.setattr(media_probe, "_linux_descriptor_execution_available", lambda: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("wrong digest reached subprocess"),
    )

    with pytest.raises(ValueError, match="differs from the frozen bytes"):
        probe_decoded_video(
            source,
            ffprobe_path=executable,
            ffprobe_sha256="0" * 64,
            descriptor_stable=True,
        )


@pytest.mark.skipif(
    sys.platform != "linux" or not Path("/proc/self/fd").is_dir(),
    reason="descriptor-stable execution requires Linux procfs",
)
def test_linux_descriptor_stable_media_tool_executes_the_authenticated_fds(
    tmp_path: Path,
) -> None:
    system_cat = shutil.which("cat")
    if system_cat is None:
        pytest.skip("cat is unavailable")
    executable = (tmp_path / "cat").resolve()
    source = (tmp_path / "source.bin").resolve()
    shutil.copyfile(system_cat, executable)
    executable.chmod(0o700)
    source.write_bytes(b"descriptor-stable payload")

    completed = _run_linux_descriptor_stable_media_tool(
        executable_path=executable,
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        media_path=source,
        arguments=lambda media_fd_path: (media_fd_path,),
    )

    assert completed.stdout == b"descriptor-stable payload"
