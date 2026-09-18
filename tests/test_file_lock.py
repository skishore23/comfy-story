from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

from comfy_story.file_lock import lock_exclusive, unlock

ROOT = Path(__file__).resolve().parents[1]


def test_a_second_handle_cannot_take_a_held_lock(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    with path.open("a") as first, path.open("a") as second:
        lock_exclusive(first, blocking=False)
        with pytest.raises(BlockingIOError):
            lock_exclusive(second, blocking=False)


def test_unlock_lets_another_handle_take_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "edit.lock"
    with path.open("a") as first, path.open("a") as second:
        lock_exclusive(first)
        unlock(first)
        lock_exclusive(second, blocking=False)


def test_closing_the_handle_releases_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    with path.open("a") as first:
        lock_exclusive(first, blocking=False)
    with path.open("a") as second:
        lock_exclusive(second, blocking=False)


def test_a_blocking_lock_waits_for_the_holder(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    released = threading.Event()
    acquired = threading.Event()
    with path.open("a+b") as holder:
        lock_exclusive(holder)

        def wait_for_lock() -> None:
            with path.open("a+b") as waiter:
                lock_exclusive(waiter)
                assert released.is_set(), "took the lock while it was still held"
                acquired.set()

        thread = threading.Thread(target=wait_for_lock)
        thread.start()
        assert not acquired.wait(0.3)
        released.set()
        unlock(holder)
    thread.join(timeout=10)
    assert acquired.is_set()


def test_only_the_lock_module_imports_fcntl() -> None:
    """fcntl does not exist on Windows; importing it anywhere else stops the node pack loading."""
    offenders = []
    for path in sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("integrations/**/*.py")]):
        if path.name == "file_lock.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if "fcntl" in names:
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
