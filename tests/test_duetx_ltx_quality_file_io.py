from __future__ import annotations

import os
from pathlib import Path

import pytest

from duet.duetx.ltx_quality_file_io import read_single_link_regular_file


def _trusted_file(tmp_path: Path) -> Path:
    path = tmp_path / "evidence.json"
    path.write_bytes(b'{"sealed":true}')
    path.chmod(0o600)
    return path


def test_descriptor_read_returns_exact_bytes_from_a_stable_single_link_file(
    tmp_path: Path,
) -> None:
    path = _trusted_file(tmp_path)

    assert (
        read_single_link_regular_file(
            path,
            name="evidence",
            required_mode=0o600,
            max_bytes=1024,
        )
        == b'{"sealed":true}'
    )


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "mode"])
def test_descriptor_read_rejects_namespace_and_mode_attacks(tmp_path: Path, attack: str) -> None:
    path = _trusted_file(tmp_path)
    candidate = path
    if attack == "symlink":
        candidate = tmp_path / "symlink.json"
        candidate.symlink_to(path)
    elif attack == "hardlink":
        candidate = tmp_path / "hardlink.json"
        candidate.hardlink_to(path)
    else:
        path.chmod(0o644)

    with pytest.raises(ValueError, match=r"unsafe|hardlink|single-link|mode"):
        read_single_link_regular_file(candidate, name="evidence", required_mode=0o600)


def test_descriptor_read_rejects_oversize_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _trusted_file(tmp_path)
    reads = 0
    real_read = os.read

    def counted_read(descriptor: int, size: int) -> bytes:
        nonlocal reads
        reads += 1
        return real_read(descriptor, size)

    monkeypatch.setattr("duet.duetx.ltx_quality_file_io.os.read", counted_read)
    with pytest.raises(ValueError, match="size bound"):
        read_single_link_regular_file(path, name="evidence", max_bytes=1)
    assert reads == 0


def test_descriptor_read_rejects_short_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _trusted_file(tmp_path)
    monkeypatch.setattr("duet.duetx.ltx_quality_file_io.os.read", lambda _descriptor, _size: b"")

    with pytest.raises(ValueError, match="changed during"):
        read_single_link_regular_file(path, name="evidence")


def test_descriptor_read_rejects_metadata_drift_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _trusted_file(tmp_path)
    real_fstat = os.fstat
    calls = 0

    def drifting_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        metadata = real_fstat(descriptor)
        calls += 1
        if calls == 2:
            values = list(metadata)
            values[6] = metadata.st_size + 1
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr("duet.duetx.ltx_quality_file_io.os.fstat", drifting_fstat)
    with pytest.raises(ValueError, match="changed during"):
        read_single_link_regular_file(path, name="evidence")
