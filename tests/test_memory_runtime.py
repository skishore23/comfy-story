from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from comfy_story.memory.backend import StoryMemoryBackend, StoryMemoryRuntime
from comfy_story.memory.bridge import H3LatentHistoryBridge
from comfy_story.memory.checkpoint import (
    MemoryProtocol,
    build_memory_modules,
    save_minimax_h3_runtime_checkpoint,
)
from comfy_story.memory.runtime import (
    MiniMaxH3StoryRuntime,
    inspect_minimax_h3_checkpoint,
)


class _FakeCodec:
    def __init__(self) -> None:
        self.closed = False

    def encode_frame(self, frame: np.ndarray[Any, Any]) -> torch.Tensor:
        assert frame.dtype == np.uint8
        assert frame.shape == (384, 384, 3)
        return torch.full((1, 24, 1, 4, 4), 0.25, dtype=torch.float32)

    def decode_frame(self, latent: torch.Tensor) -> np.ndarray[Any, Any]:
        assert latent.shape[1] == 24
        return np.full((384, 384, 3), 127, dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def _checkpoint(path: Path) -> tuple[str, str, str]:
    foundation = "1" * 64
    model_configuration = "2" * 64
    protocol = MemoryProtocol.default().fingerprint()
    modules = build_memory_modules()
    assert isinstance(modules.bridge, H3LatentHistoryBridge)
    with torch.no_grad():
        modules.bridge.residual.weight.copy_(torch.eye(24) * 0.1)
    save_minimax_h3_runtime_checkpoint(
        path,
        modules,
        foundation_sha256=foundation,
        model_configuration_sha256=model_configuration,
        optimizer_steps=2_000,
        training_trace=({"loss": 0.25, "optimizer_step": 2_000},),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest(), foundation, protocol


def test_minimax_runtime_loads_trained_native_24_channel_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "minimax-memory.pt"
    checkpoint_sha256, foundation_sha256, protocol_sha256 = _checkpoint(path)
    codec = _FakeCodec()

    runtime = MiniMaxH3StoryRuntime.load_pinned(
        path,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_foundation_sha256=foundation_sha256,
        expected_protocol_sha256=protocol_sha256,
        model_configuration_sha256="2" * 64,
        vae_sha256="3" * 64,
        codec=codec,
        device=torch.device("cpu"),
    )

    assert isinstance(runtime, StoryMemoryRuntime)
    assert runtime.identity.backend is StoryMemoryBackend.MINIMAX_H3
    assert runtime.identity.contract.latent_channels == 24
    assert runtime.identity.checkpoint_sha256 == checkpoint_sha256
    encoded = runtime.materialize_frame(np.zeros((384, 384, 3), dtype=np.uint8))
    assert encoded.latent.shape == (1, 24, 1, 4, 4)
    encoded.validate(runtime.identity.contract)
    assert runtime.decode(encoded.latent).shape == (384, 384, 3)
    assert isinstance(runtime.score(encoded.latent, 0, 0.0), float)
    runtime.close()
    assert codec.closed


def test_minimax_checkpoint_exposes_pinned_bridge_identity_before_runtime_load(
    tmp_path: Path,
) -> None:
    path = tmp_path / "minimax-memory.pt"
    checkpoint_sha256, foundation_sha256, protocol_sha256 = _checkpoint(path)

    identity = inspect_minimax_h3_checkpoint(
        path,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_foundation_sha256=foundation_sha256,
        expected_protocol_sha256=protocol_sha256,
        model_configuration_sha256="2" * 64,
    )

    assert identity.checkpoint_sha256 == checkpoint_sha256
    assert len(identity.adapter_sha256) == 64
    assert len(identity.scorer_sha256) == 64
    assert identity.optimizer_steps == 2_000


def test_minimax_runtime_rejects_declared_bridge_identity_that_does_not_match_tensors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong-bridge-identity.pt"
    _, foundation_sha256, protocol_sha256 = _checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["adapter_sha256"] = "9" * 64
    torch.save(payload, path)
    checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="bridge tensor digest"):
        MiniMaxH3StoryRuntime.load_pinned(
            path,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_foundation_sha256=foundation_sha256,
            expected_protocol_sha256=protocol_sha256,
            model_configuration_sha256="2" * 64,
            vae_sha256="3" * 64,
            codec=_FakeCodec(),
            device=torch.device("cpu"),
        )


def test_minimax_runtime_rejects_an_untrained_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "untrained.pt"
    checkpoint_sha256, foundation_sha256, protocol_sha256 = _checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["optimizer_steps"] = 0
    payload["training_trace"] = ()
    torch.save(payload, path)
    checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="trained MiniMax H3 checkpoint"):
        MiniMaxH3StoryRuntime.load_pinned(
            path,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_foundation_sha256=foundation_sha256,
            expected_protocol_sha256=protocol_sha256,
            model_configuration_sha256="2" * 64,
            vae_sha256="3" * 64,
            codec=_FakeCodec(),
            device=torch.device("cpu"),
        )


def test_minimax_runtime_rejects_wrong_foundation_before_codec_use(tmp_path: Path) -> None:
    path = tmp_path / "wrong-foundation.pt"
    checkpoint_sha256, _, protocol_sha256 = _checkpoint(path)

    with pytest.raises(ValueError, match="foundation"):
        MiniMaxH3StoryRuntime.load_pinned(
            path,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_foundation_sha256="9" * 64,
            expected_protocol_sha256=protocol_sha256,
            model_configuration_sha256="2" * 64,
            vae_sha256="3" * 64,
            codec=_FakeCodec(),
            device=torch.device("cpu"),
        )
