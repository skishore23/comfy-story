"""Exact film intervals shared by saved media and the next shot's visual state."""

from __future__ import annotations

from typing import Any

import torch


def trim_shot_media(
    images: torch.Tensor, audio: dict[str, Any], duration_ms: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Trim decoded padding without interpolation, retiming or mutating source tensors."""
    if type(duration_ms) is not int or not 0 < duration_ms <= 15000 or duration_ms * 24 % 1000:
        raise ValueError("shot duration must select exact 24 fps frames within 15 seconds")
    frames = duration_ms * 24 // 1000
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] < frames:
        raise ValueError("decoded video is shorter than the requested interval")
    waveform = audio.get("waveform")
    rate = audio.get("sample_rate")
    if type(rate) is not int or rate <= 0 or duration_ms * rate % 1000:
        raise ValueError("shot duration must select exact audio samples")
    samples = duration_ms * rate // 1000
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[-1] < samples:
        raise ValueError("decoded audio is shorter than the requested interval")
    return images[:frames].contiguous(), {**audio, "waveform": waveform[..., :samples].contiguous()}
