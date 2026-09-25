from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from comfy_story.platform_io import fsync_directory, open_no_follow
from comfy_story.story_store import StoryProjectStore


def test_open_no_follow_reads_bytes_exactly(tmp_path: Path) -> None:
    """A text-mode descriptor on Windows would rewrite CRLF and stop at Ctrl-Z."""
    payload = b"a\r\nb\x1ac\n\x00end"
    path = tmp_path / "blob"
    path.write_bytes(payload)
    descriptor = open_no_follow(path)
    try:
        assert os.read(descriptor, 64) == payload
    finally:
        os.close(descriptor)


def test_open_no_follow_refuses_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    link = tmp_path / "link"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("this account cannot create symlinks (Windows without Developer Mode)")
    # POSIX: O_NOFOLLOW's ELOOP ("Too many levels of symbolic links"); Windows: the lstat check.
    with pytest.raises(OSError, match=r"symbolic links|follow a link") as refused:
        open_no_follow(link)
    assert refused.value.errno == errno.ELOOP


def test_fsync_directory_accepts_a_directory(tmp_path: Path) -> None:
    fsync_directory(tmp_path)


def test_story_store_round_trips_an_asset(tmp_path: Path) -> None:
    """The store refused to construct off Linux and macOS; it must work everywhere."""
    store = StoryProjectStore(tmp_path.resolve())
    payload = b"frame\r\n\x1a" * 100
    digest = store.put_asset(payload)
    assert store.load_asset(digest) == payload
    assert store.verified_asset_path(digest).read_bytes() == payload
