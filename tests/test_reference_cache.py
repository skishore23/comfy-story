from pathlib import Path

import pytest
import torch
from PIL import Image

from comfy_story.reference_cache import ReferenceCache
from comfy_story.reference_review import image_reference_size, review_image_references


def test_reference_cache_restart_invalidation_and_corruption(tmp_path: Path) -> None:
    calls = []

    def encode(pixels: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return torch.full((1, 24, 1, 2, 2), float(pixels.mean()), dtype=torch.bfloat16)

    pixels = torch.ones(1, 32, 32, 3)
    cold = ReferenceCache(tmp_path, "vae-a").encode(pixels, encode)
    warm = ReferenceCache(tmp_path, "vae-a").encode(pixels, encode)
    assert torch.equal(cold, warm)
    assert cold.dtype == warm.dtype
    assert len(calls) == 1
    # Changing history pixels or the encoder identity cannot reuse an earlier encoding.
    ReferenceCache(tmp_path, "vae-a").encode(pixels * 2, encode)
    ReferenceCache(tmp_path, "vae-b").encode(pixels, encode)
    assert len(calls) == 3
    for path in tmp_path.glob("*.pt"):
        path.write_bytes(b"damaged cache")
    assert torch.equal(ReferenceCache(tmp_path, "vae-a").encode(pixels, encode), cold)
    assert len(calls) == 4


def test_cache_eviction_disabled_and_unavailable(tmp_path: Path) -> None:
    calls = []

    def encode(pixels: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return torch.ones(1, 24, 1, 2, 2) * pixels.mean()

    pixels = torch.ones(1, 32, 32, 3)
    cache = ReferenceCache(tmp_path / "cache", "vae", maximum_bytes=3000)
    for scale in range(5):
        cache.encode(pixels * scale, encode)
    assert sum(path.stat().st_size for path in cache.root.glob("*.pt")) <= 3000
    assert len(list(cache.root.glob("*.pt"))) == 1
    cache.encode(pixels * 4, encode)
    assert len(calls) == 5
    ReferenceCache(cache.root, "vae", 0).encode(pixels * 4, encode)
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    ReferenceCache(blocked, "vae").encode(pixels, encode)
    assert len(calls) == 7


def test_reference_review_matches_native_sizing_and_rejects_unsafe_inputs(tmp_path: Path) -> None:
    Image.new("RGB", (2688, 1536)).save(tmp_path / "cast.png")
    result = review_image_references(
        tmp_path, [{"name": "Aiko", "file": "cast.png"}], (1344, 768), "match"
    )
    assert result["named_image_tokens"] == 42 * 24
    assert image_reference_size(2688, 1536, (1344, 768), "max") == (2688, 1536)
    assert image_reference_size(64, 32, (1344, 768), "match") == (64, 32)
    for refs in [[{"name": "bad", "file": "../cast.png"}], [{}], [{}] * 10, "bad"]:
        with pytest.raises(ValueError, match=r"reference|input|filename"):
            review_image_references(tmp_path, refs, (1344, 768), "match")


def test_encoder_failure_is_not_retried_or_cached(tmp_path: Path) -> None:
    calls = []

    def fail(pixels: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        raise RuntimeError("encoder failed")

    with pytest.raises(RuntimeError, match="encoder failed"):
        ReferenceCache(tmp_path, "vae").encode(torch.ones(1, 32, 32, 3), fail)
    assert len(calls) == 1
    assert not list(tmp_path.glob("*.pt"))
