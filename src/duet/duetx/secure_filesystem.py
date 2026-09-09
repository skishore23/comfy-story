"""Platform-specific custody for immutable, content-addressed runtime files."""

from __future__ import annotations

import ctypes
import hashlib
import os
import secrets
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Literal, Protocol, Self, cast

_WINDOWS_REPARSE_ATTRIBUTE = 0x400
_WINDOWS_RESERVED_NAMES = frozenset(
    {"aux", "con", "nul", "prn"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


class _StatLike(Protocol):
    @property
    def st_mode(self) -> int: ...

    @property
    def st_dev(self) -> int: ...

    @property
    def st_ino(self) -> int: ...

    @property
    def st_size(self) -> int: ...


class WindowsFilesystemOps(Protocol):
    """Injectable native-Windows syscall surface used by the custody backend."""

    def lstat(self, path: PureWindowsPath) -> _StatLike: ...

    def pin_directory(self, path: PureWindowsPath) -> object: ...

    def directory_facts(self, handle: object) -> _StatLike: ...

    def close_directory(self, handle: object) -> None: ...

    def open_binary_read(self, path: PureWindowsPath) -> BinaryIO: ...

    def fstat(self, handle: BinaryIO) -> _StatLike: ...

    def create_binary_exclusive(self, path: PureWindowsPath, data: bytes, mode: int) -> None: ...

    def link_exclusive(self, source: PureWindowsPath, destination: PureWindowsPath) -> None: ...

    def unlink(self, path: PureWindowsPath) -> None: ...


@dataclass(frozen=True, slots=True)
class SecureFileIdentity:
    """Stable byte identity of one custody-checked regular file."""

    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if (
            len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
            or type(self.size_bytes) is not int
            or self.size_bytes < 0
        ):
            raise ValueError("secure file identity is invalid")


def _relative_parts(value: str | PurePath, *, direct: bool) -> tuple[str, ...]:
    raw = str(value)
    path = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or ":" in raw
        or any(ord(character) < 32 for character in raw)
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != raw
    ):
        label = "direct child" if direct else "normalized relative path"
        raise ValueError(f"secure path must be a {label}")
    if any(
        part.casefold().split(".", 1)[0] in _WINDOWS_RESERVED_NAMES or part.endswith((" ", "."))
        for part in path.parts
    ):
        label = "direct child" if direct else "normalized relative path"
        raise ValueError(f"secure path must be a {label}")
    if direct and len(path.parts) != 1:
        raise ValueError("secure path must be a direct child")
    return path.parts


def _read_handle(handle: BinaryIO, *, limit: int | None) -> bytes:
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("bounded read limit must be a nonnegative integer")
    chunks: list[bytes] = []
    total = 0
    while True:
        request = 1024 * 1024 if limit is None else min(1024 * 1024, limit + 1 - total)
        if request <= 0:
            raise ValueError("file exceeds the bounded read limit")
        chunk = handle.read(request)
        if not chunk:
            return b"".join(chunks)
        if not isinstance(chunk, bytes):
            raise ValueError("custody read did not return binary bytes")
        chunks.append(chunk)
        total += len(chunk)
        if limit is not None and total > limit:
            raise ValueError("file exceeds the bounded read limit")


def _identity(encoded: bytes) -> SecureFileIdentity:
    return SecureFileIdentity(hashlib.sha256(encoded).hexdigest(), len(encoded))


def _describe_handle(handle: BinaryIO) -> SecureFileIdentity:
    digest = hashlib.sha256()
    size = 0
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return SecureFileIdentity(digest.hexdigest(), size)


def _same_file(before: _StatLike, opened: _StatLike, after: _StatLike) -> bool:
    expected = (before.st_dev, before.st_ino, before.st_size, stat.S_IFMT(before.st_mode))
    return (
        expected
        == (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            stat.S_IFMT(opened.st_mode),
        )
        == (after.st_dev, after.st_ino, after.st_size, stat.S_IFMT(after.st_mode))
    )


class _CustodyBackend(Protocol):
    def close(self) -> None: ...

    def read(
        self, parts: tuple[str, ...], *, limit: int | None, missing_ok: bool
    ) -> bytes | None: ...

    def describe(
        self, parts: tuple[str, ...], *, missing_ok: bool
    ) -> SecureFileIdentity | None: ...

    def validate_parent(self, parts: tuple[str, ...]) -> None: ...

    def write(self, parts: tuple[str, ...], data: bytes, *, mode: int) -> SecureFileIdentity: ...


class _PosixBackend:
    def __init__(self, root: Path) -> None:
        if (
            not root.is_absolute()
            or root.is_symlink()
            or not root.is_dir()
            or root.resolve(strict=True) != root
        ):
            raise ValueError("POSIX custody root must be an existing resolved absolute directory")
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        try:
            descriptor = os.open(root.anchor, flags)
            for part in root.parts[1:]:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
                try:
                    metadata = os.fstat(next_descriptor)
                    if (
                        not stat.S_ISDIR(metadata.st_mode)
                        or metadata.st_uid not in {0, os.getuid()}
                        or stat.S_IMODE(metadata.st_mode) & 0o022
                    ):
                        raise ValueError("POSIX custody root ancestry is unsafe")
                except BaseException:
                    os.close(next_descriptor)
                    raise
                os.close(descriptor)
                descriptor = next_descriptor
        except OSError as error:
            if descriptor >= 0:
                os.close(descriptor)
            raise ValueError("POSIX custody root ancestry is unsafe") from error
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise
        self._descriptor = descriptor

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def _root_fd(self) -> int:
        if self._descriptor < 0:
            raise ValueError("secure filesystem root is closed")
        return self._descriptor

    def _open_file(self, parts: tuple[str, ...], *, missing_ok: bool) -> int | None:
        directory = os.dup(self._root_fd())
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        try:
            for part in parts[:-1]:
                try:
                    child = os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory,
                    )
                except OSError as error:
                    raise ValueError(
                        "relative custody path contains an unsafe directory"
                    ) from error
                os.close(directory)
                directory = child
            try:
                before = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
                descriptor = os.open(parts[-1], flags, dir_fd=directory)
            except FileNotFoundError:
                if missing_ok:
                    return None
                raise ValueError("custody file is missing or unsafe") from None
            except OSError as error:
                raise ValueError("custody file is missing or unsafe") from error
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened, opened):
                os.close(descriptor)
                raise ValueError("custody file is not a stable regular file")
            return descriptor
        finally:
            os.close(directory)

    def read(self, parts: tuple[str, ...], *, limit: int | None, missing_ok: bool) -> bytes | None:
        descriptor = self._open_file(parts, missing_ok=missing_ok)
        if descriptor is None:
            return None
        try:
            before = os.fstat(descriptor)
            with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
                encoded = _read_handle(handle, limit=limit)
            after = os.fstat(descriptor)
            if not _same_file(before, before, after) or len(encoded) != after.st_size:
                raise ValueError("custody file identity changed during read")
            return encoded
        finally:
            os.close(descriptor)

    def describe(self, parts: tuple[str, ...], *, missing_ok: bool) -> SecureFileIdentity | None:
        descriptor = self._open_file(parts, missing_ok=missing_ok)
        if descriptor is None:
            return None
        try:
            before = os.fstat(descriptor)
            with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
                identity = _describe_handle(handle)
            after = os.fstat(descriptor)
            if not _same_file(before, before, after) or identity.size_bytes != after.st_size:
                raise ValueError("custody file identity changed during hashing")
            return identity
        finally:
            os.close(descriptor)

    def _open_parent(self, parts: tuple[str, ...]) -> int:
        directory = os.dup(self._root_fd())
        try:
            for part in parts[:-1]:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory,
                )
                os.close(directory)
                directory = child
            return directory
        except OSError as error:
            os.close(directory)
            raise ValueError("immutable output parent is missing or unsafe") from error

    def validate_parent(self, parts: tuple[str, ...]) -> None:
        os.close(self._open_parent(parts))

    @staticmethod
    def _read_at(parent_fd: int, name: str, *, limit: int | None, missing_ok: bool) -> bytes | None:
        try:
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise ValueError("immutable output is missing") from None
        except OSError as error:
            raise ValueError("immutable output is unsafe") from error
        try:
            opened = os.fstat(descriptor)
            with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
                encoded = _read_handle(handle, limit=limit)
            after_open = os.fstat(descriptor)
            after_path = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or not _same_file(before, opened, after_open)
                or not _same_file(before, opened, after_path)
                or len(encoded) != after_path.st_size
            ):
                raise ValueError("immutable output identity changed")
            return encoded
        finally:
            os.close(descriptor)

    def write(self, parts: tuple[str, ...], data: bytes, *, mode: int) -> SecureFileIdentity:
        parent_fd = self._open_parent(parts)
        name = parts[-1]
        expected = _identity(data)
        try:
            existing = self._read_at(parent_fd, name, limit=None, missing_ok=True)
            if existing is not None:
                if existing != data:
                    raise ValueError("existing immutable artifact differs")
                return expected
            temporary = f".{name}.{secrets.token_hex(16)}.tmp"
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_BINARY", 0),
                mode,
                dir_fd=parent_fd,
            )
            try:
                offset = 0
                while offset < len(data):
                    offset += os.write(descriptor, data[offset:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                try:
                    os.link(
                        temporary,
                        name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError as error:
                    raced = self._read_at(parent_fd, name, limit=None, missing_ok=False)
                    if raced != data:
                        raise ValueError("raced immutable artifact differs") from error
                os.fsync(parent_fd)
            finally:
                with suppress(FileNotFoundError):
                    os.unlink(temporary, dir_fd=parent_fd)
            published = self._read_at(parent_fd, name, limit=None, missing_ok=False)
            if published != data:
                raise ValueError("immutable artifact publish identity mismatch")
            return expected
        finally:
            os.close(parent_fd)


class _NativeWindowsOps:
    @dataclass(frozen=True, slots=True)
    class _HandleStat:
        st_mode: int
        st_dev: int
        st_ino: int
        st_size: int
        st_file_attributes: int

    def __init__(self) -> None:
        from ctypes import wintypes

        class ByHandleFileInformation(ctypes.Structure):
            _fields_ = (
                ("dwFileAttributes", wintypes.DWORD),
                ("ftCreationTime", wintypes.FILETIME),
                ("ftLastAccessTime", wintypes.FILETIME),
                ("ftLastWriteTime", wintypes.FILETIME),
                ("dwVolumeSerialNumber", wintypes.DWORD),
                ("nFileSizeHigh", wintypes.DWORD),
                ("nFileSizeLow", wintypes.DWORD),
                ("nNumberOfLinks", wintypes.DWORD),
                ("nFileIndexHigh", wintypes.DWORD),
                ("nFileIndexLow", wintypes.DWORD),
            )

        self._information_type = ByHandleFileInformation
        win_dll = cast(Any, ctypes).WinDLL
        self._kernel32 = win_dll("kernel32", use_last_error=True)
        self._kernel32.CreateFileW.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        self._kernel32.CreateFileW.restype = wintypes.HANDLE
        self._kernel32.GetFileInformationByHandle.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ByHandleFileInformation),
        )
        self._kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    @staticmethod
    def _path(path: PureWindowsPath) -> Path:
        return Path(str(path))

    def lstat(self, path: PureWindowsPath) -> os.stat_result:
        return self._path(path).lstat()

    def pin_directory(self, path: PureWindowsPath) -> object:
        # FILE_SHARE_DELETE is deliberately omitted, so a pinned ancestor cannot be renamed or
        # replaced while a custody operation is using its absolute namespace.
        handle = self._kernel32.CreateFileW(
            str(path),
            0x80,  # FILE_READ_ATTRIBUTES
            0x1 | 0x2,  # FILE_SHARE_READ | FILE_SHARE_WRITE
            None,
            3,  # OPEN_EXISTING
            0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            raise OSError(cast(Any, ctypes).get_last_error(), "CreateFileW failed")
        return handle

    def directory_facts(self, handle: object) -> _HandleStat:
        information = self._information_type()
        if not self._kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
            raise OSError(cast(Any, ctypes).get_last_error(), "GetFileInformationByHandle failed")
        attributes = int(information.dwFileAttributes)
        mode = stat.S_IFDIR if attributes & 0x10 else stat.S_IFREG
        return self._HandleStat(
            mode,
            int(information.dwVolumeSerialNumber),
            (int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow),
            (int(information.nFileSizeHigh) << 32) | int(information.nFileSizeLow),
            attributes,
        )

    def close_directory(self, handle: object) -> None:
        if not self._kernel32.CloseHandle(handle):
            raise OSError(cast(Any, ctypes).get_last_error(), "CloseHandle failed")

    def open_binary_read(self, path: PureWindowsPath) -> BinaryIO:
        return cast(BinaryIO, self._path(path).open("rb", buffering=0))

    def fstat(self, handle: BinaryIO) -> os.stat_result:
        return os.fstat(handle.fileno())

    def create_binary_exclusive(self, path: PureWindowsPath, data: bytes, mode: int) -> None:
        descriptor = os.open(
            self._path(path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            mode,
        )
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def link_exclusive(self, source: PureWindowsPath, destination: PureWindowsPath) -> None:
        os.link(self._path(source), self._path(destination), follow_symlinks=False)

    def unlink(self, path: PureWindowsPath) -> None:
        self._path(path).unlink()


def _is_reparse(metadata: _StatLike) -> bool:
    return bool(cast(int, getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_ATTRIBUTE)


def _root_identity(metadata: _StatLike) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)


class _WindowsBackend:
    def __init__(self, root: PureWindowsPath, ops: WindowsFilesystemOps) -> None:
        if not root.is_absolute() or root == PureWindowsPath(root.anchor) or ".." in root.parts:
            raise ValueError("Windows custody root must be an absolute non-root path")
        self._root = root
        self._ops = ops
        self._ancestor_handles: list[object] = []
        for candidate in reversed((root, *root.parents)):
            handle: object | None = None
            try:
                path_metadata = ops.lstat(candidate)
                handle = ops.pin_directory(candidate)
                handle_metadata = ops.directory_facts(handle)
            except OSError as error:
                if handle is not None:
                    ops.close_directory(handle)
                self._close_handles()
                raise ValueError("Windows custody root ancestry is unreadable") from error
            if (
                _is_reparse(path_metadata)
                or _is_reparse(handle_metadata)
                or _root_identity(path_metadata) != _root_identity(handle_metadata)
            ):
                ops.close_directory(handle)
                self._close_handles()
                raise ValueError("Windows custody root ancestry contains a reparse point")
            if not stat.S_ISDIR(handle_metadata.st_mode):
                ops.close_directory(handle)
                self._close_handles()
                raise ValueError("Windows custody root ancestry contains a non-directory")
            self._ancestor_handles.append(handle)
        root_metadata = ops.directory_facts(self._ancestor_handles[-1])
        if not stat.S_ISDIR(root_metadata.st_mode):
            self._close_handles()
            raise ValueError("Windows custody root is not a directory")
        self._identity = _root_identity(root_metadata)
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._close_handles()
            self._closed = True

    def _close_handles(self) -> None:
        for handle in reversed(self._ancestor_handles):
            self._ops.close_directory(handle)
        self._ancestor_handles.clear()

    def _check_root(self) -> None:
        if self._closed:
            raise ValueError("secure filesystem root is closed")
        if not self._ancestor_handles:
            raise ValueError("Windows custody root identity changed")
        handle_metadata = self._ops.directory_facts(self._ancestor_handles[-1])
        path_metadata = self._ops.lstat(self._root)
        if (
            _is_reparse(handle_metadata)
            or _is_reparse(path_metadata)
            or _root_identity(handle_metadata) != self._identity
            or _root_identity(path_metadata) != self._identity
        ):
            raise ValueError("Windows custody root identity changed")

    def _file_path(
        self, parts: tuple[str, ...], *, missing_ok: bool
    ) -> tuple[PureWindowsPath | None, list[object]]:
        current, pinned = self._pin_parent(parts)
        path = current / parts[-1]
        try:
            metadata = self._ops.lstat(path)
        except FileNotFoundError:
            self._close_relative_handles(pinned)
            if missing_ok:
                return None, []
            raise ValueError("Windows custody file is missing") from None
        if _is_reparse(metadata):
            self._close_relative_handles(pinned)
            raise ValueError("Windows custody file is a reparse point")
        if not stat.S_ISREG(metadata.st_mode):
            self._close_relative_handles(pinned)
            raise ValueError("Windows custody file is not regular")
        return path, pinned

    def _pin_parent(self, parts: tuple[str, ...]) -> tuple[PureWindowsPath, list[object]]:
        self._check_root()
        current = self._root
        pinned: list[object] = []
        for part in parts[:-1]:
            current /= part
            handle: object | None = None
            try:
                path_metadata = self._ops.lstat(current)
                handle = self._ops.pin_directory(current)
                handle_metadata = self._ops.directory_facts(handle)
            except FileNotFoundError:
                if handle is not None:
                    self._ops.close_directory(handle)
                self._close_relative_handles(pinned)
                raise ValueError("Windows custody directory is missing") from None
            except OSError as error:
                if handle is not None:
                    self._ops.close_directory(handle)
                self._close_relative_handles(pinned)
                raise ValueError("Windows custody directory is unsafe") from error
            assert handle is not None
            if (
                _is_reparse(path_metadata)
                or _is_reparse(handle_metadata)
                or _root_identity(path_metadata) != _root_identity(handle_metadata)
            ):
                self._ops.close_directory(handle)
                self._close_relative_handles(pinned)
                raise ValueError("Windows custody path contains a reparse point")
            if not stat.S_ISDIR(handle_metadata.st_mode):
                self._ops.close_directory(handle)
                self._close_relative_handles(pinned)
                raise ValueError("Windows custody path contains a non-directory")
            pinned.append(handle)
        return current, pinned

    def _close_relative_handles(self, handles: list[object]) -> None:
        for handle in reversed(handles):
            self._ops.close_directory(handle)

    def validate_parent(self, parts: tuple[str, ...]) -> None:
        _parent, pinned = self._pin_parent(parts)
        try:
            self._check_root()
        finally:
            self._close_relative_handles(pinned)

    def read(self, parts: tuple[str, ...], *, limit: int | None, missing_ok: bool) -> bytes | None:
        path, pinned = self._file_path(parts, missing_ok=missing_ok)
        if path is None:
            return None
        try:
            return self._read_absolute(path, limit=limit)
        finally:
            self._close_relative_handles(pinned)

    def _read_absolute(self, path: PureWindowsPath, *, limit: int | None) -> bytes:
        before = self._ops.lstat(path)
        if _is_reparse(before) or not stat.S_ISREG(before.st_mode):
            raise ValueError("Windows custody file is unsafe")
        try:
            handle = self._ops.open_binary_read(path)
        except OSError as error:
            raise ValueError("Windows custody file cannot be opened") from error
        try:
            opened = self._ops.fstat(handle)
            encoded = _read_handle(handle, limit=limit)
            after_open = self._ops.fstat(handle)
        finally:
            handle.close()
        after_path = self._ops.lstat(path)
        if (
            _is_reparse(after_path)
            or not _same_file(before, opened, after_open)
            or not _same_file(before, opened, after_path)
            or len(encoded) != after_path.st_size
        ):
            raise ValueError("Windows custody file identity changed during read")
        self._check_root()
        return encoded

    def describe(self, parts: tuple[str, ...], *, missing_ok: bool) -> SecureFileIdentity | None:
        path, pinned = self._file_path(parts, missing_ok=missing_ok)
        if path is None:
            return None
        try:
            before = self._ops.lstat(path)
            with self._ops.open_binary_read(path) as handle:
                opened = self._ops.fstat(handle)
                identity = _describe_handle(handle)
                after_open = self._ops.fstat(handle)
            after_path = self._ops.lstat(path)
            if (
                _is_reparse(before)
                or _is_reparse(after_path)
                or not stat.S_ISREG(opened.st_mode)
                or not _same_file(before, opened, after_open)
                or not _same_file(before, opened, after_path)
                or identity.size_bytes != after_path.st_size
            ):
                raise ValueError("Windows custody file identity changed during hashing")
            self._check_root()
            return identity
        finally:
            self._close_relative_handles(pinned)

    def write(self, parts: tuple[str, ...], data: bytes, *, mode: int) -> SecureFileIdentity:
        self._check_root()
        parent, pinned = self._pin_parent(parts)
        name = parts[-1]
        expected = _identity(data)
        temporary = parent / f".{name}.{secrets.token_hex(16)}.tmp"
        destination = parent / name
        temporary_created = False
        try:
            try:
                destination_metadata = self._ops.lstat(destination)
            except FileNotFoundError:
                destination_metadata = None
            if destination_metadata is not None:
                if _is_reparse(destination_metadata) or not stat.S_ISREG(
                    destination_metadata.st_mode
                ):
                    raise ValueError("existing immutable artifact is unsafe")
                existing = self._read_absolute(destination, limit=None)
                if existing != data:
                    raise ValueError("existing immutable artifact differs")
                return expected
            self._ops.create_binary_exclusive(temporary, data, mode)
            temporary_created = True
            temporary_bytes = self._read_absolute(temporary, limit=len(data))
            if temporary_bytes != data:
                raise ValueError("Windows immutable temporary bytes differ")
            self._check_root()
            try:
                self._ops.link_exclusive(temporary, destination)
            except FileExistsError as error:
                raced = self._read_absolute(destination, limit=None)
                if raced != data:
                    raise ValueError("raced immutable artifact differs") from error
            published = self._read_absolute(destination, limit=None)
            if published != data:
                raise ValueError("immutable artifact publish identity mismatch")
            self._check_root()
            return expected
        finally:
            if temporary_created:
                with suppress(FileNotFoundError):
                    self._ops.unlink(temporary)
            self._close_relative_handles(pinned)


class SecureFilesystemRoot:
    """Typed facade over descriptor-confined POSIX and custody-checked Windows filesystems."""

    def __init__(self, backend: _CustodyBackend) -> None:
        self._backend = backend

    @classmethod
    def open(
        cls,
        root: Path | PureWindowsPath,
        *,
        platform: Literal["posix", "windows"],
        windows_ops: WindowsFilesystemOps | None = None,
    ) -> Self:
        if platform == "posix":
            if windows_ops is not None or not isinstance(root, Path):
                raise ValueError("POSIX custody requires a native Path and no Windows adapter")
            return cls(_PosixBackend(root))
        if not isinstance(root, PureWindowsPath):
            raise ValueError("Windows custody requires a PureWindowsPath")
        if windows_ops is None:
            if os.name != "nt":
                raise ValueError("native Windows custody is available only on Windows")
            windows_ops = cast(WindowsFilesystemOps, _NativeWindowsOps())
        return cls(_WindowsBackend(root, windows_ops))

    @classmethod
    def open_native(cls, root: Path) -> Self:
        """Select the custody backend for the current interpreter's native path type."""
        if not isinstance(root, Path):
            raise TypeError("native custody root must be a Path")
        if os.name == "nt":
            return cls.open(PureWindowsPath(str(root)), platform="windows")
        return cls.open(root, platform="posix")

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def read_regular(self, relative_path: str | PurePath, *, limit: int | None = None) -> bytes:
        parts = _relative_parts(relative_path, direct=False)
        result = self._backend.read(parts, limit=limit, missing_ok=False)
        assert result is not None
        return result

    def describe_regular(self, relative_path: str | PurePath) -> SecureFileIdentity:
        parts = _relative_parts(relative_path, direct=False)
        result = self._backend.describe(parts, missing_ok=False)
        assert result is not None
        return result

    def read_direct_child(
        self,
        name: str,
        *,
        limit: int,
        missing_ok: bool = False,
    ) -> bytes | None:
        parts = _relative_parts(name, direct=True)
        return self._backend.read(parts, limit=limit, missing_ok=missing_ok)

    def describe_direct_child(
        self, name: str, *, missing_ok: bool = False
    ) -> SecureFileIdentity | None:
        parts = _relative_parts(name, direct=True)
        return self._backend.describe(parts, missing_ok=missing_ok)

    def write_immutable_direct_child(
        self, name: str, data: bytes, *, mode: int = 0o600
    ) -> SecureFileIdentity:
        parts = _relative_parts(name, direct=True)
        return self._write(parts, data, mode=mode)

    def write_immutable(
        self,
        relative_path: str | PurePath,
        data: bytes,
        *,
        mode: int = 0o600,
    ) -> SecureFileIdentity:
        """Publish one immutable file below an existing, custody-pinned parent chain."""
        parts = _relative_parts(relative_path, direct=False)
        return self._write(parts, data, mode=mode)

    def validate_output_path(self, relative_path: str | PurePath) -> None:
        """Pin the existing parent chain for one not-yet-published output."""
        self._backend.validate_parent(_relative_parts(relative_path, direct=False))

    def _write(self, parts: tuple[str, ...], data: bytes, *, mode: int) -> SecureFileIdentity:
        if not isinstance(data, bytes):
            raise TypeError("immutable artifact data must be bytes")
        if type(mode) is not int or mode < 0 or mode > 0o777 or mode & 0o077:
            raise ValueError("immutable artifact mode must not grant group or world access")
        return self._backend.write(parts, data, mode=mode)


__all__ = (
    "SecureFileIdentity",
    "SecureFilesystemRoot",
    "WindowsFilesystemOps",
)
