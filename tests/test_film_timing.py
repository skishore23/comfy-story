from __future__ import annotations

import pytest
import torch

from comfy_story.film_timing import trim_shot_media


@pytest.mark.parametrize(
    ("source_frames", "milliseconds", "expected_frames"),
    [(124, 5000, 120), (243, 10000, 240), (362, 15000, 360), (243, 7000, 168)],
)
def test_continuation_frame_is_the_last_frame_retained_by_the_edit(
    source_frames: int, milliseconds: int, expected_frames: int
) -> None:
    images = (
        torch.arange(source_frames, dtype=torch.float32).reshape(-1, 1, 1, 1).expand(-1, 2, 2, 3)
    )
    waveform = torch.arange(source_frames * 2000, dtype=torch.float32).reshape(1, 1, -1)
    selected, audio = trim_shot_media(
        images, {"waveform": waveform, "sample_rate": 48000}, milliseconds
    )
    assert selected.shape[0] == expected_frames
    assert torch.equal(selected[-1], images[expected_frames - 1])
    assert audio["waveform"].shape[-1] == milliseconds * 48
    assert images.shape[0] == source_frames
    assert waveform.shape[-1] == source_frames * 2000


@pytest.mark.parametrize("duration", [0, -1, True, 5001, 15001])
def test_invalid_intervals_fail_before_media_can_be_published(duration: int) -> None:
    with pytest.raises(ValueError, match="duration"):
        trim_shot_media(
            torch.zeros(124, 2, 2, 3),
            {"waveform": torch.zeros(1, 1, 250000), "sample_rate": 48000},
            duration,
        )


def test_short_decodes_are_rejected_instead_of_padded() -> None:
    with pytest.raises(ValueError, match="video"):
        trim_shot_media(
            torch.zeros(119, 2, 2, 3),
            {"waveform": torch.zeros(1, 1, 240000), "sample_rate": 48000},
            5000,
        )
    with pytest.raises(ValueError, match="audio"):
        trim_shot_media(
            torch.zeros(124, 2, 2, 3),
            {"waveform": torch.zeros(1, 1, 239999), "sample_rate": 48000},
            5000,
        )
