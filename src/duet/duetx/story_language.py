"""Explicit language assets for Story generation; never infer exact speech from picture prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from duet.duetx.contracts import tensor_sha256


@dataclass(frozen=True, slots=True)
class AuthoredStoryAudio:
    waveform: torch.Tensor
    sample_rate: int
    frame_count: int

    @property
    def binding(self) -> dict[str, str | int]:
        return {
            "waveform_sha256": tensor_sha256(self.waveform),
            "sample_rate": self.sample_rate,
            "frame_count": self.frame_count,
        }

    def as_comfy(self) -> dict[str, Any]:
        samples = round(self.frame_count * self.sample_rate / 24)
        return {
            "waveform": functional.pad(self.waveform, (0, samples - self.waveform.shape[-1])),
            "sample_rate": self.sample_rate,
        }


def prepare_authored_audio(value: object, *, frame_count: int) -> AuthoredStoryAudio | None:
    """Validate and snapshot one approved track, rejecting truncation before GPU generation.

    Exact pronunciation and agreement with a script still require review of the source audio.
    This boundary preserves that source, pads its ending with silence, and excludes generated
    H3 speech. It does not claim that an arbitrary connected track is linguistically correct.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Authored audio must be a Comfy AUDIO input")
    rate = value.get("sample_rate")
    wave = value.get("waveform")
    if type(rate) is not int or not 8_000 <= rate <= 192_000:
        raise ValueError("Authored audio requires a sample rate between 8000 and 192000 Hz")
    if (
        not isinstance(wave, torch.Tensor)
        or wave.ndim != 3
        or wave.shape[0] != 1
        or wave.shape[1] not in (1, 2)
        or wave.shape[2] == 0
        or not wave.is_floating_point()
    ):
        raise ValueError("Authored audio requires one nonempty mono or stereo waveform")
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("Authored audio requires a positive shot frame count")
    if wave.shape[-1] > round(frame_count * rate / 24):
        raise ValueError(
            "Authored audio exceeds the shot duration; edit the track or lengthen the shot"
        )
    if not torch.isfinite(wave).all() or wave.abs().max() > 1:
        raise ValueError("Authored audio samples must be finite and within [-1, 1]")
    return AuthoredStoryAudio(
        wave.detach().to(device="cpu", dtype=torch.float32).clone(), rate, frame_count
    )
