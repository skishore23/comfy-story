from __future__ import annotations

import pytest
import torch

from comfy_story.h3_reference_alignment import (
    fold_video_into_streams,
    pack_aligned_images,
)
from comfy_story.h3_reference_contracts import H3ReferenceKind, H3VisualReference


def _image(source_id: str, height: int = 4, width: int = 6) -> H3VisualReference:
    return H3VisualReference(
        source_id,
        H3ReferenceKind.IMAGE,
        0,
        torch.zeros(1, 24, 1, height, width),
        False,
    )


def test_video_is_split_into_eight_ordered_chunks() -> None:
    latent = torch.arange(37, dtype=torch.float32).reshape(1, 1, 37, 1, 1).repeat(1, 24, 1, 2, 2)
    reference = H3VisualReference("drive", H3ReferenceKind.VIDEO, 0, latent, False)

    pack = fold_video_into_streams(reference)

    assert pack.history.shape == (1, 8, 24, 5, 2, 2)
    assert pack.padding_temporal_tokens == 3
    assert pack.history[0, 0, 0, :, 0, 0].tolist() == [0, 1, 2, 3, 4]
    assert pack.history[0, 7, 0, :, 0, 0].tolist() == [35, 36, 36, 36, 36]
    torch.testing.assert_close(pack.last_available, pack.history[:, 7])


def test_video_without_padding_preserves_exact_chunk_order() -> None:
    latent = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1, 1)
    latent = latent.repeat(1, 24, 1, 2, 2)
    pack = fold_video_into_streams(
        H3VisualReference("drive", H3ReferenceKind.VIDEO, 0, latent, False)
    )

    assert pack.padding_temporal_tokens == 0
    assert pack.history[:, 3, :, 0].equal(latent[:, :, 6])
    assert pack.history[:, 3, :, 1].equal(latent[:, :, 7])


def test_image_pack_rejects_unaligned_spatial_shapes() -> None:
    with pytest.raises(ValueError, match="same latent shape"):
        pack_aligned_images((_image("a", 48, 84), _image("b", 40, 72)))


def test_image_pack_marks_only_real_sources_available() -> None:
    references = tuple(_image(f"view-{index}") for index in range(3))

    pack = pack_aligned_images(references)

    assert pack.history.shape == (1, 8, 24, 1, 4, 6)
    assert pack.availability.tolist() == [[True, True, True, False, False, False, False, False]]
    assert pack.source_ids == ("view-0", "view-1", "view-2")
    torch.testing.assert_close(pack.last_available, references[-1].latent)


def test_alignment_rejects_protected_or_wrong_kind_sources() -> None:
    protected = H3VisualReference(
        "face", H3ReferenceKind.IMAGE, 0, torch.zeros(1, 24, 1, 4, 6), True
    )
    with pytest.raises(ValueError, match="outside the dense"):
        pack_aligned_images((protected,))
    with pytest.raises(ValueError, match="image references only"):
        pack_aligned_images(
            (
                H3VisualReference(
                    "drive", H3ReferenceKind.VIDEO, 0, torch.zeros(1, 24, 2, 4, 6), False
                ),
            )
        )
    with pytest.raises(ValueError, match="protected video"):
        fold_video_into_streams(
            H3VisualReference("drive", H3ReferenceKind.VIDEO, 0, torch.zeros(1, 24, 2, 4, 6), True)
        )
