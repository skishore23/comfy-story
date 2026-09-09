"""Typed model-specific seam for the shared Duet Story memory engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import torch

from comfy_story.contracts import DuetXContract
from comfy_story.latent_bridge import LatentHistoryBridge

_SHA256_HEX = frozenset("0123456789abcdef")


def _digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


class StoryMemoryBackend(StrEnum):
    """The exact latent representation used by one Story branch."""

    LTX = "ltx"
    MINIMAX_H3 = "minimax-h3"


class StorySampler(StrEnum):
    """MiniMax generation strategy recorded independently from memory identity."""

    NATIVE_RES_MULTISTEP = "native-res-multistep"
    SPEED_EULER_2STAGE = "speed-euler-2stage"
    TURBO_4STEP = "turbo-4step"
    TURBO_8STEP = "turbo-8step"
    FULL_HD_2PASS = "full-hd-2pass"
    NVFP4_EXACT = "nvfp4-exact"
    NVFP4_BALANCED = "nvfp4-balanced"
    NVFP4_ULTRA_FAST = "nvfp4-ultra-fast"
    NVFP4_TURBO = "nvfp4-turbo-4step"


@dataclass(frozen=True, slots=True)
class StoryMemoryIdentity:
    """Immutable interpretation boundary for all cached operators on a branch."""

    backend: StoryMemoryBackend
    contract: DuetXContract
    checkpoint_sha256: str
    model_configuration_sha256: str

    def validate(self) -> Self:
        if not isinstance(self.backend, StoryMemoryBackend):
            raise ValueError("backend must be a StoryMemoryBackend")
        self.contract.validate()
        expected_format = {
            StoryMemoryBackend.LTX: "duet-x-ltx-decision1-v1",
            StoryMemoryBackend.MINIMAX_H3: "duet-x-minimax-h3-v1",
        }[self.backend]
        if self.contract.format != expected_format:
            raise ValueError("memory backend does not match its contract format")
        _digest(self.checkpoint_sha256, "checkpoint_sha256")
        _digest(self.model_configuration_sha256, "model_configuration_sha256")
        return self


@dataclass(frozen=True, slots=True)
class EncodedStoryFrame:
    """One backend-native latent and the fingerprints required for provenance."""

    latent: torch.Tensor
    preprocessing_sha256: str
    vae_sha256: str
    adapter_sha256: str
    scorer_sha256: str

    def validate(self, contract: DuetXContract) -> Self:
        contract.validate()
        expected_channels = contract.latent_channels
        if (
            not isinstance(self.latent, torch.Tensor)
            or self.latent.layout != torch.strided
            or self.latent.ndim != 5
            or self.latent.shape[0] != 1
            or self.latent.shape[1] != expected_channels
            or any(dimension <= 0 for dimension in self.latent.shape[2:])
            or self.latent.dtype != torch.float32
            or self.latent.device.type != "cpu"
            or not self.latent.is_contiguous()
            or not bool(torch.isfinite(self.latent).all().item())
        ):
            raise ValueError(
                "encoded story frame must be finite contiguous CPU float32 with "
                f"one batch and {expected_channels} channels"
            )
        for field in (
            "preprocessing_sha256",
            "vae_sha256",
            "adapter_sha256",
            "scorer_sha256",
        ):
            _digest(getattr(self, field), field)
        return self


@runtime_checkable
class StoryMemoryRuntime(Protocol):
    """Only model-dependent operations required by shared Story orchestration."""

    @property
    def identity(self) -> StoryMemoryIdentity: ...

    @property
    def bridge(self) -> LatentHistoryBridge: ...

    def materialize_frame(self, frame: np.ndarray[Any, Any]) -> EncodedStoryFrame: ...

    def score(self, latent: torch.Tensor, event_type: int, timestamp: float) -> float: ...

    def decode(self, latent: torch.Tensor) -> np.ndarray[Any, Any]: ...

    def close(self) -> None: ...


__all__ = (
    "EncodedStoryFrame",
    "StoryMemoryBackend",
    "StoryMemoryIdentity",
    "StoryMemoryRuntime",
    "StorySampler",
)
