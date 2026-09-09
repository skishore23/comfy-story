from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import BinaryIO, cast

import pytest

from comfy_story.secure_filesystem import (
    SecureFileIdentity,
    SecureFilesystemRoot,
    WindowsFilesystemOps,
)


def test_posix_root_reads_hashes_and_publishes_immutable_direct_children(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "root").resolve()
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "input.bin").write_bytes(b"\x00input\xff")

    with SecureFilesystemRoot.open(root, platform="posix") as custody:
        assert custody.read_regular("nested/input.bin", limit=32) == b"\x00input\xff"
        assert custody.describe_regular("nested/input.bin") == SecureFileIdentity(
            sha256="a835aec665e30cdf19c1b71e2a01948d72325191c89bdbeba7ed9ca59b44949a",
            size_bytes=7,
        )
        published = custody.write_immutable_direct_child("receipt.bin", b"\x00receipt\xff")
        assert published == custody.describe_direct_child("receipt.bin")
        assert custody.read_direct_child("receipt.bin", limit=32) == b"\x00receipt\xff"
        assert custody.write_immutable_direct_child("receipt.bin", b"\x00receipt\xff") == published
        assert custody.read_direct_child("missing.bin", limit=32, missing_ok=True) is None
        assert custody.describe_direct_child("missing.bin", missing_ok=True) is None
        assert custody.write_immutable("nested/evidence.bin", b"evidence") == (
            custody.describe_regular("nested/evidence.bin")
        )
        custody.validate_output_path("nested/future.bin")
        with pytest.raises(ValueError, match="immutable"):
            custody.write_immutable_direct_child("receipt.bin", b"different")


def test_native_factory_selects_current_platform_backend(tmp_path: Path) -> None:
    root = (tmp_path / "native").resolve()
    root.mkdir()
    (root / "input.bin").write_bytes(b"native")

    with SecureFilesystemRoot.open_native(root) as custody:
        assert custody.read_regular("input.bin") == b"native"


def test_posix_root_rejects_escape_links_and_oversized_reads(tmp_path: Path) -> None:
    root = (tmp_path / "root").resolve()
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (root / "link.bin").symlink_to(outside)
    (root / "large.bin").write_bytes(b"12345")

    with SecureFilesystemRoot.open(root, platform="posix") as custody:
        for relative in ("../outside.bin", "/absolute.bin", "nested\\escape.bin"):
            with pytest.raises(ValueError, match="normalized relative"):
                custody.read_regular(relative, limit=32)
        with pytest.raises(ValueError, match=r"unsafe|regular"):
            custody.read_regular("link.bin", limit=32)
        with pytest.raises(ValueError, match="bounded read limit"):
            custody.read_regular("large.bin", limit=4)
        with pytest.raises(ValueError, match="direct child"):
            custody.write_immutable_direct_child("nested/output.bin", b"no")


@dataclass(frozen=True)
class _WindowsStat:
    st_mode: int
    st_dev: int
    st_ino: int
    st_size: int
    st_file_attributes: int = 0


class _WindowsHandle(io.BufferedReader):
    pass


class FakeWindowsOps(WindowsFilesystemOps):
    def __init__(self, local_root: Path, windows_root: PureWindowsPath) -> None:
        self.local_root = local_root
        self.windows_root = windows_root
        self.reparse: set[str] = set()
        self.identity_overrides: dict[str, int] = {}
        self.open_directory_handles: list[tuple[PureWindowsPath, _WindowsStat]] = []

    def _local(self, path: PureWindowsPath) -> Path:
        try:
            relative = path.relative_to(self.windows_root)
        except ValueError:
            if path == PureWindowsPath(path.anchor):
                return self.local_root.parent
            raise
        return self.local_root.joinpath(*relative.parts)

    def lstat(self, path: PureWindowsPath) -> _WindowsStat:
        local = self._local(path)
        value = local.lstat()
        key = str(path)
        return _WindowsStat(
            st_mode=value.st_mode,
            st_dev=value.st_dev,
            st_ino=self.identity_overrides.get(key, value.st_ino),
            st_size=value.st_size,
            st_file_attributes=0x400 if key in self.reparse else 0,
        )

    def pin_directory(self, path: PureWindowsPath) -> object:
        handle = (path, self.lstat(path))
        self.open_directory_handles.append(handle)
        return handle

    def directory_facts(self, handle: object) -> _WindowsStat:
        _path, facts = cast(tuple[PureWindowsPath, _WindowsStat], handle)
        return facts

    def close_directory(self, handle: object) -> None:
        self.open_directory_handles.remove(cast(tuple[PureWindowsPath, _WindowsStat], handle))

    def open_binary_read(self, path: PureWindowsPath) -> BinaryIO:
        return self._local(path).open("rb", buffering=0)

    def fstat(self, handle: BinaryIO) -> _WindowsStat:
        value = os.fstat(handle.fileno())
        return _WindowsStat(value.st_mode, value.st_dev, value.st_ino, value.st_size)

    def create_binary_exclusive(self, path: PureWindowsPath, data: bytes, mode: int) -> None:
        descriptor = os.open(self._local(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def link_exclusive(self, source: PureWindowsPath, destination: PureWindowsPath) -> None:
        os.link(self._local(source), self._local(destination))

    def unlink(self, path: PureWindowsPath) -> None:
        self._local(path).unlink()


def _windows_root(tmp_path: Path) -> tuple[PureWindowsPath, FakeWindowsOps]:
    local = tmp_path / "secure"
    local.mkdir()
    root = PureWindowsPath("C:/secure")
    return root, FakeWindowsOps(local, root)


def test_windows_root_preserves_binary_identity_and_immutable_resume(tmp_path: Path) -> None:
    root, ops = _windows_root(tmp_path)
    nested = ops.local_root / "nested"
    nested.mkdir()
    (nested / "input.bin").write_bytes(b"\x00windows\xff")

    with SecureFilesystemRoot.open(root, platform="windows", windows_ops=ops) as custody:
        assert custody.read_regular("nested/input.bin", limit=64) == b"\x00windows\xff"
        identity = custody.describe_regular("nested/input.bin")
        assert identity.size_bytes == 9
        published = custody.write_immutable_direct_child("receipt.bin", b"\x00receipt\xff")
        assert published == custody.describe_direct_child("receipt.bin")
        assert custody.write_immutable_direct_child("receipt.bin", b"\x00receipt\xff") == published
        assert custody.read_direct_child("missing.bin", limit=1, missing_ok=True) is None
        with pytest.raises(ValueError, match="immutable"):
            custody.write_immutable_direct_child("receipt.bin", b"changed")
        nested_identity = custody.write_immutable("nested/evidence.bin", b"nested")
        assert nested_identity == custody.describe_regular("nested/evidence.bin")
        custody.validate_output_path("nested/future.bin")
        assert len(ops.open_directory_handles) >= 2
    assert ops.open_directory_handles == []


@pytest.mark.parametrize("target", ["root", "directory", "file"])
def test_windows_root_rejects_reparse_points_at_every_level(tmp_path: Path, target: str) -> None:
    root, ops = _windows_root(tmp_path)
    nested = ops.local_root / "nested"
    nested.mkdir()
    (nested / "input.bin").write_bytes(b"input")
    marked = {
        "root": root,
        "directory": root / "nested",
        "file": root / "nested" / "input.bin",
    }[target]
    ops.reparse.add(str(marked))

    if target == "root":
        with pytest.raises(ValueError, match="reparse"):
            SecureFilesystemRoot.open(root, platform="windows", windows_ops=ops)
        return
    with (
        SecureFilesystemRoot.open(root, platform="windows", windows_ops=ops) as custody,
        pytest.raises(ValueError, match="reparse"),
    ):
        custody.read_regular("nested/input.bin", limit=32)


def test_windows_root_detects_root_or_file_identity_change(tmp_path: Path) -> None:
    root, ops = _windows_root(tmp_path)
    path = ops.local_root / "input.bin"
    path.write_bytes(b"input")
    with SecureFilesystemRoot.open(root, platform="windows", windows_ops=ops) as custody:
        ops.identity_overrides[str(root / "input.bin")] = path.stat().st_ino + 1
        with pytest.raises(ValueError, match="identity"):
            custody.read_regular("input.bin", limit=32)
        ops.identity_overrides.clear()
        ops.identity_overrides[str(root)] = ops.local_root.stat().st_ino + 1
        with pytest.raises(ValueError, match="root identity"):
            custody.describe_regular("input.bin")


def test_windows_root_rejects_unsafe_names_and_bounded_overflow(tmp_path: Path) -> None:
    root, ops = _windows_root(tmp_path)
    (ops.local_root / "large.bin").write_bytes(b"12345")
    with SecureFilesystemRoot.open(root, platform="windows", windows_ops=ops) as custody:
        with pytest.raises(ValueError, match="bounded read limit"):
            custody.read_regular("large.bin", limit=4)
        for name in ("CON", "aux.txt", "nested/output.bin", "bad:name", "bad\\name"):
            with pytest.raises(ValueError, match="direct child"):
                custody.write_immutable_direct_child(name, b"x")


def test_content_address_hashing_streams_large_assets() -> None:
    import hashlib

    from comfy_story.secure_filesystem import _describe_handle

    data = b"asset" * 500000

    class BoundedReader(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None
            assert 0 < size <= 1024 * 1024, "Model hashing must use bounded reads"
            return super().read(size)

    assert _describe_handle(BoundedReader(data)) == SecureFileIdentity(
        hashlib.sha256(data).hexdigest(), len(data)
    )
