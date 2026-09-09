"""Authenticated, content-addressed assets for Comfy Story projects."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ASSET_BYTES = 4 * 1024 * 1024 * 1024


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _read_regular(path: Path, field: str, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{field} is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= maximum
            or before.st_nlink != 1
        ):
            raise ValueError(f"{field} must be a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            value = handle.read(maximum + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ) or len(value) != after.st_size:
            raise ValueError(f"{field} changed while it was read")
        return value
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class StoryProjectStore:
    """Persist story assets and revisions beneath one configured regular root."""

    def __init__(self, root: Path) -> None:
        if sys.platform != "darwin" and not sys.platform.startswith("linux"):
            raise ValueError(
                "Comfy Story currently requires Linux or macOS for atomic revision storage"
            )
        if not isinstance(root, Path) or not root.is_absolute() or root.is_symlink():
            raise ValueError("story project root must be an absolute regular directory")
        root.mkdir(parents=True, exist_ok=True)
        if root.resolve() != root or not root.is_dir():
            raise ValueError("story project root must be an absolute regular directory")
        self._root = root
        self._assets = root / "assets" / "sha256"
        self._assets.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _put_bytes(self, directory: Path, value: bytes, *, suffix: str = "") -> str:
        digest = _sha256_bytes(value)
        target = directory / f"{digest}{suffix}"
        if target.exists():
            if _read_regular(target, "content-addressed object", len(value)) != value:
                raise ValueError("refusing conflicting content-addressed object")
            return digest
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if _read_regular(target, "content-addressed object", len(value)) != value:
                    raise ValueError("refusing conflicting content-addressed object") from None
            _fsync_directory(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return digest

    def put_asset(self, value: bytes) -> str:
        """Publish one immutable asset and return its content digest."""
        if not isinstance(value, bytes) or not 0 < len(value) <= _MAX_ASSET_BYTES:
            raise ValueError("story asset must be nonempty bounded bytes")
        return self._put_bytes(self._assets, value)

    def load_asset(self, digest: str) -> bytes:
        """Load and authenticate one public content-addressed Story asset."""
        return self._asset(digest, "story asset")

    def verified_asset_path(self, digest: str) -> Path:
        """Authenticate immutable media in bounded memory before HTTP range serving."""
        expected = _digest(digest, "story asset")
        path = self._assets / expected
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise ValueError("story asset is unavailable") from error
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or not 0 < before.st_size <= _MAX_ASSET_BYTES
                or before.st_nlink != 1
            ):
                raise ValueError("story asset must be a bounded regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                observed = hashlib.file_digest(handle, "sha256").hexdigest()
            after = os.fstat(descriptor)
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, field) != getattr(after, field) for field in fields):
                raise ValueError("story asset changed while it was read")
            if observed != expected:
                raise ValueError("story asset SHA-256 changed")
            return path
        finally:
            os.close(descriptor)

    def _asset(self, digest: str, field: str) -> bytes:
        expected = _digest(digest, field)
        value = _read_regular(self._assets / expected, field, _MAX_ASSET_BYTES)
        if _sha256_bytes(value) != expected:
            raise ValueError(f"{field} SHA-256 changed")
        return value
