from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from duet.duetx.contracts import DuetXContract
from duet.duetx.latent_bridge import LatentHistoryBridge
from duet.duetx.story_memory_backend import (
    EncodedStoryFrame,
    StoryMemoryBackend,
    StoryMemoryIdentity,
    StoryMemoryRuntime,
    StorySampler,
)


class _Runtime:
    def __init__(self) -> None:
        contract = DuetXContract.minimax_h3(adapter_fingerprint="1" * 64)
        self._bridge = LatentHistoryBridge(24, operator_size=4)
        self._identity = StoryMemoryIdentity(
            StoryMemoryBackend.MINIMAX_H3,
            contract,
            "2" * 64,
            "3" * 64,
        )

    @property
    def identity(self) -> StoryMemoryIdentity:
        return self._identity

    @property
    def bridge(self) -> LatentHistoryBridge:
        return self._bridge

    def materialize_frame(self, frame: np.ndarray[Any, Any]) -> EncodedStoryFrame:
        del frame
        return EncodedStoryFrame(torch.zeros(1, 24, 1, 2, 2), *("4" * 64 for _ in range(4)))

    def score(self, latent: torch.Tensor, event_type: int, timestamp: float) -> float:
        del latent, event_type, timestamp
        return 0.0

    def decode(self, latent: torch.Tensor) -> np.ndarray[Any, Any]:
        del latent
        return np.zeros((384, 384, 3), dtype=np.uint8)

    def close(self) -> None:
        return None


def test_runtime_protocol_exposes_only_the_model_specific_memory_seam() -> None:
    runtime = _Runtime()

    assert isinstance(runtime, StoryMemoryRuntime)
    assert runtime.identity.backend is StoryMemoryBackend.MINIMAX_H3
    assert runtime.identity.contract.latent_channels == 24
    assert runtime.bridge.fusion.channels == 24


def test_encoded_frame_uses_the_selected_contract_channel_boundary() -> None:
    minimax = DuetXContract.minimax_h3(adapter_fingerprint="1" * 64)
    encoded = EncodedStoryFrame(
        torch.zeros(1, 24, 1, 2, 2),
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
    )

    assert encoded.validate(minimax) is encoded
    with pytest.raises(ValueError, match="128 channels"):
        encoded.validate(DuetXContract.default(decision1_fingerprint="1" * 64))


def test_story_memory_identity_rejects_a_backend_contract_mismatch() -> None:
    identity = StoryMemoryIdentity(
        StoryMemoryBackend.MINIMAX_H3,
        DuetXContract.default(decision1_fingerprint="1" * 64),
        "2" * 64,
        "3" * 64,
    )

    with pytest.raises(ValueError, match="backend does not match"):
        identity.validate()


def test_sampler_names_are_stable_authenticated_metadata_values() -> None:
    assert StorySampler.NATIVE_RES_MULTISTEP.value == "native-res-multistep"
    assert StorySampler.SPEED_EULER_2STAGE.value == "speed-euler-2stage"
