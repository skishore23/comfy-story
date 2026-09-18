"""Exclusive advisory file locks that work on POSIX and Windows.

POSIX uses `flock` on the whole file. Windows has no `fcntl`, so it locks the first byte of the
file with `msvcrt.locking`. Either way the lock belongs to the open handle: two handles on the same
path contend even inside one process, and the OS releases the lock when the handle closes.
"""

from __future__ import annotations

import os
import sys
import time
from typing import IO, Any

if sys.platform == "win32":
    import msvcrt

    # msvcrt's own blocking mode gives up after ten one-second retries, so poll instead.
    _RETRY_SECONDS = 0.05

    def _lock(fd: int, blocking: bool) -> bool:
        while True:
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                if not blocking:
                    return False
                time.sleep(_RETRY_SECONDS)
            else:
                return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fd: int, blocking: bool) -> bool:
        if blocking:
            fcntl.flock(fd, fcntl.LOCK_EX)
            return True
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def lock_exclusive(handle: IO[Any], *, blocking: bool = True) -> None:
    """Take an exclusive lock on `handle`.

    With `blocking=False`, raise BlockingIOError when another handle holds the lock.
    """
    if not _lock(handle.fileno(), blocking):
        raise BlockingIOError(f"{getattr(handle, 'name', handle.fileno())} is locked")


def unlock(handle: IO[Any]) -> None:
    """Release a lock taken with `lock_exclusive`."""
    _unlock(handle.fileno())
