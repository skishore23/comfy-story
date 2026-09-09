from __future__ import annotations

from pathlib import Path

import pytest

from comfy_story.story_store import (
    StoryProjectStore,
)


@pytest.mark.parametrize("corruption", ["bytes", "symlink", "hardlink"])
def test_streamed_asset_verification_rejects_corruption(tmp_path: Path, corruption: str) -> None:
    import os

    store = StoryProjectStore(tmp_path / "story")
    digest = store.put_asset(b"immutable movie bytes")
    path = store.verified_asset_path(digest)
    if corruption == "bytes":
        path.write_bytes(b"different movie bytes")
    elif corruption == "symlink":
        source = tmp_path / "outside"
        source.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(source)
    else:
        os.link(path, tmp_path / "second-link")
    with pytest.raises(ValueError, match=r"SHA-256|unavailable|regular file"):
        store.verified_asset_path(digest)


def test_streamed_asset_verification_does_not_use_buffered_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StoryProjectStore(tmp_path / "story")
    digest = store.put_asset(b"a" * (3 * 1024 * 1024))

    def reject_read_bytes(*args: object) -> bytes:
        raise AssertionError("asset must be hashed incrementally")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    assert store.verified_asset_path(digest).name == digest
