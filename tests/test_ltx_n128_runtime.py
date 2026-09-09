from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from comfy_story.ltx_n128_materializer import AuthenticatedObservationPNG
from comfy_story.ltx_n128_runtime import (
    CURRENT_SLOT,
    HISTORY_ITEMS,
    ORDINARY_SOURCE_COUNT,
    PINNED_SLOTS,
    authenticate,
    build_history,
    canonical_sha256,
    file_sha256,
)


def source(index: int) -> AuthenticatedObservationPNG:
    return AuthenticatedObservationPNG(
        Path(f"/tmp/source-{index:03d}.png"), hashlib.sha256(str(index).encode()).hexdigest()
    ).validate()


def test_build_history_has_exact_roster_and_cycle() -> None:
    ordinary = tuple(source(index) for index in range(ORDINARY_SOURCE_COUNT))
    identity = source(100)
    object_ = source(101)
    current = source(102)

    history = build_history(ordinary, identity, object_, current)

    assert len(history) == HISTORY_ITEMS
    assert history[PINNED_SLOTS[0]] is identity
    assert history[PINNED_SLOTS[1]] is object_
    assert history[CURRENT_SLOT] is current
    ordinary_rows = tuple(
        value for slot, value in enumerate(history) if slot not in (*PINNED_SLOTS, CURRENT_SLOT)
    )
    assert ordinary_rows == tuple(
        ordinary[index % ORDINARY_SOURCE_COUNT] for index in range(len(ordinary_rows))
    )


def test_build_history_rejects_wrong_ordinary_count() -> None:
    with pytest.raises(ValueError, match="exactly twelve"):
        build_history(
            tuple(source(index) for index in range(11)), source(100), source(101), source(102)
        )


def test_authenticate_binds_regular_png(tmp_path: Path) -> None:
    path = tmp_path / "source.png"
    path.write_bytes(b"png bytes")

    result = authenticate(path)

    assert result.path == path.resolve()
    assert result.sha256 == hashlib.sha256(b"png bytes").hexdigest()
    assert file_sha256(path) == result.sha256


def test_authenticate_rejects_non_png_and_symlink(tmp_path: Path) -> None:
    non_png = tmp_path / "source.jpg"
    non_png.write_bytes(b"bytes")
    with pytest.raises(ValueError, match="authenticated PNG"):
        authenticate(non_png)
    png = tmp_path / "source.png"
    png.write_bytes(b"png")
    link = tmp_path / "link.png"
    link.symlink_to(png)
    with pytest.raises(ValueError, match="authenticated PNG"):
        authenticate(link)


def test_canonical_sha256_is_order_independent_for_objects() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})
