"""The few filesystem calls whose POSIX form Windows lacks.

Story storage leans on three POSIX details: opening a file without following a
final symlink (`O_NOFOLLOW`), making a new directory entry durable with an fsync
of the directory, and `O_NONBLOCK` so a FIFO planted at an asset path cannot
hang a read. Windows has none of the three through `os`. Each helper keeps the
POSIX behaviour exactly and says what it does instead on Windows.
"""

from __future__ import annotations

import errno
import os
import stat
import sys
from pathlib import Path

# Windows opens file descriptors in text mode unless told otherwise, which
# rewrites CRLF and stops at Ctrl-Z on read; every caller here reads bytes.
_BINARY = getattr(os, "O_BINARY", 0)
# A FIFO cannot exist in a Windows directory, so there is nothing to guard.
NONBLOCK = getattr(os, "O_NONBLOCK", 0)


def open_no_follow(path: Path, flags: int = os.O_RDONLY) -> int:
    """Open `path` without following a link in its final component.

    POSIX refuses the link in the open itself (`O_NOFOLLOW`). Windows has no such
    flag, so a symlink or other reparse point (a junction) is refused before the
    open, and the opened file must be the one that check saw. Raises OSError, like
    a refused POSIX open, so callers keep a single error path.
    """
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is not None:
        return os.open(path, flags | nofollow | _BINARY)
    seen = os.lstat(path)
    attributes = getattr(seen, "st_file_attributes", 0)
    if stat.S_ISLNK(seen.st_mode) or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise OSError(errno.ELOOP, "refusing to follow a link", str(path))
    descriptor = os.open(path, flags | _BINARY)
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (seen.st_dev, seen.st_ino):
        os.close(descriptor)
        raise OSError(errno.ELOOP, "path changed while it was opened", str(path))
    return descriptor


def fsync_directory(path: Path) -> None:
    """Make a new entry in `path` (a link or a rename) durable.

    POSIX needs an fsync of the directory itself. Windows cannot open a directory
    through `os.open` and has no directory fsync; NTFS records the entry in its
    metadata journal, which is the durability Windows offers, so this is a no-op
    there.
    """
    if sys.platform == "win32":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
