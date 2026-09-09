"""Descriptor-stable reads for sealed LTX decoded-quality evidence."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def _stable_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_single_link_regular_file(
    path: Path,
    *,
    name: str,
    required_mode: int | None = None,
    max_bytes: int | None = None,
) -> bytes:
    """Read bytes from one checked descriptor and reject namespace or in-read drift."""
    if required_mode is not None and required_mode not in {0o600, 0o644}:
        raise ValueError("trusted file mode contract is invalid")
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("trusted file size bound must be positive")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{name} is unavailable or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} must be a regular non-symlink file")
        if before.st_nlink != 1:
            raise ValueError(f"{name} must be a single-link file, not a hardlink")
        if required_mode is not None and stat.S_IMODE(before.st_mode) != required_mode:
            raise ValueError(f"{name} must be mode {required_mode:04o}")
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError(f"{name} exceeds its size bound")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        if _stable_identity(after) != _stable_identity(before) or len(encoded) != before.st_size:
            raise ValueError(f"{name} changed during its descriptor read")
        return encoded
    finally:
        os.close(descriptor)


__all__ = ("read_single_link_regular_file",)
