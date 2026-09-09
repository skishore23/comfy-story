from __future__ import annotations

import pytest
import torch

from comfy_story.story_language import prepare_authored_audio


def test_approved_audio_is_snapshotted_and_padded_without_generated_speech() -> None:
    original = torch.full((1, 1, 8_000), 0.25)
    prepared = prepare_authored_audio({"waveform": original, "sample_rate": 8_000}, frame_count=48)
    assert prepared is not None
    binding = prepared.binding
    original.zero_()
    output = prepared.as_comfy()
    assert output["sample_rate"] == 8_000
    assert output["waveform"].shape == (1, 1, 16_000)
    torch.testing.assert_close(output["waveform"][..., :8_000], torch.full((1, 1, 8_000), 0.25))
    assert output["waveform"][..., 8_000:].count_nonzero() == 0
    assert prepared.binding == binding
    changed = prepare_authored_audio({"waveform": original, "sample_rate": 8_000}, frame_count=48)
    assert changed is not None
    assert changed.binding != binding


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"sample_rate": True, "waveform": torch.zeros(1, 1, 10)},
        {"sample_rate": 8_000, "waveform": torch.zeros(2, 1, 10)},
        {"sample_rate": 8_000, "waveform": torch.zeros(1, 3, 10)},
        {"sample_rate": 8_000, "waveform": torch.zeros(1, 1, 0)},
        {"sample_rate": 8_000, "waveform": torch.full((1, 1, 10), float("nan"))},
        {"sample_rate": 8_000, "waveform": torch.full((1, 1, 10), 1.1)},
        {"sample_rate": 8_000, "waveform": torch.zeros(1, 1, 8_001)},
    ],
)
def test_invalid_or_overlong_authored_audio_fails_before_generation(value: object) -> None:
    with pytest.raises(ValueError, match="Authored audio"):
        prepare_authored_audio(value, frame_count=24)


def test_absent_authored_audio_preserves_the_existing_generation_path() -> None:
    assert prepare_authored_audio(None, frame_count=124) is None
