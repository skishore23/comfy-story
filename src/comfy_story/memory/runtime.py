"""Native 24-channel MiniMax H3 runtime for Comfy Story associative memory."""

from __future__ import annotations

import gc
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, cast, runtime_checkable

import numpy as np
import torch

from comfy_story.memory.backend import (
    EncodedStoryFrame,
    StoryMemoryBackend,
    StoryMemoryIdentity,
)
from comfy_story.memory.bridge import H3LatentHistoryBridge
from comfy_story.memory.checkpoint import (
    MINIMAX_H3_RUNTIME_CHECKPOINT_FORMAT,
    MemoryProtocol,
    build_memory_modules,
    module_sha256,
    module_state_sha256,
)
from comfy_story.memory.contracts import MemoryContract
from comfy_story.memory.health import history_effect_report
from comfy_story.memory.modules import LocalExceptionScorer, MemoryModules
from comfy_story.story_store import _read_regular

_FRAME_SHAPE = (384, 384, 3)
_CHECKPOINT_KEYS = frozenset(
    {
        "adapter_sha256",
        "format",
        "foundation_sha256",
        "model_configuration_sha256",
        "module_state_sha256",
        "modules",
        "optimizer_steps",
        "protocol_sha256",
        "scorer_sha256",
        "training_trace",
        "training_trace_sha256",
    }
)
_MODULE_NAMES = frozenset({"bridge", "gated", "resampler", "scorer"})
_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024


def _digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("MiniMax H3 training trace must be canonical JSON data") from error


def _mapping(value: object, field: str, keys: frozenset[str]) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed mapping")
    result = cast(dict[str, object], value)
    if set(result) != keys:
        raise ValueError(f"{field} field roster changed")
    return result


@runtime_checkable
class MiniMaxH3Codec(Protocol):
    """Comfy-owned frozen MiniMax video-VAE operations used by Story memory."""

    def encode_frame(self, frame: np.ndarray[Any, Any]) -> torch.Tensor: ...

    def decode_frame(self, latent: torch.Tensor) -> np.ndarray[Any, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MiniMaxH3CheckpointIdentity:
    """Authenticated training and foundation identity for one runtime bundle."""

    checkpoint_sha256: str
    foundation_sha256: str
    model_configuration_sha256: str
    module_state_sha256: str
    adapter_sha256: str
    scorer_sha256: str
    protocol_sha256: str
    optimizer_steps: int

    def validate(self) -> Self:
        for field in (
            "checkpoint_sha256",
            "foundation_sha256",
            "model_configuration_sha256",
            "module_state_sha256",
            "adapter_sha256",
            "scorer_sha256",
            "protocol_sha256",
        ):
            _digest(getattr(self, field), field)
        if (
            type(self.optimizer_steps) is not int
            or self.optimizer_steps != MemoryProtocol.default().optimizer_steps
        ):
            raise ValueError("trained MiniMax H3 checkpoint requires 2000 optimizer steps")
        return self


def _load_checkpoint_identity(
    path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_foundation_sha256: str,
    expected_protocol_sha256: str,
    model_configuration_sha256: str,
) -> tuple[dict[str, object], MiniMaxH3CheckpointIdentity]:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError("trained MiniMax H3 checkpoint must be an absolute regular file")
    for value, field in (
        (expected_checkpoint_sha256, "expected_checkpoint_sha256"),
        (expected_foundation_sha256, "expected_foundation_sha256"),
        (expected_protocol_sha256, "expected_protocol_sha256"),
        (model_configuration_sha256, "model_configuration_sha256"),
    ):
        _digest(value, field)
    size = path.stat().st_size
    if not 0 < size <= _MAX_CHECKPOINT_BYTES:
        raise ValueError("trained MiniMax H3 checkpoint size is invalid")
    encoded = _read_regular(path, "memory checkpoint", _MAX_CHECKPOINT_BYTES)
    observed_checkpoint_sha256 = hashlib.sha256(encoded).hexdigest()
    if observed_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("trained MiniMax H3 checkpoint content SHA-256 mismatch")
    try:
        raw = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("trained MiniMax H3 checkpoint could not be loaded safely") from error
    payload = _mapping(raw, "trained MiniMax H3 checkpoint", _CHECKPOINT_KEYS)
    if payload["format"] != MINIMAX_H3_RUNTIME_CHECKPOINT_FORMAT:
        raise ValueError("trained MiniMax H3 checkpoint format changed")
    if payload["foundation_sha256"] != expected_foundation_sha256:
        raise ValueError("trained MiniMax H3 checkpoint foundation mismatch")
    protocol_sha256 = MemoryProtocol.default().fingerprint()
    if (
        payload["protocol_sha256"] != expected_protocol_sha256
        or payload["protocol_sha256"] != protocol_sha256
    ):
        raise ValueError("trained MiniMax H3 checkpoint protocol mismatch")
    if payload["model_configuration_sha256"] != model_configuration_sha256:
        raise ValueError("trained MiniMax H3 checkpoint model configuration mismatch")
    optimizer_steps = payload["optimizer_steps"]
    trace = payload["training_trace"]
    expected_optimizer_steps = MemoryProtocol.default().optimizer_steps
    if (
        type(optimizer_steps) is not int
        or optimizer_steps != expected_optimizer_steps
        or not isinstance(trace, (tuple, list))
        or not trace
    ):
        raise ValueError("trained MiniMax H3 checkpoint has no completed training trace")
    final_trace = trace[-1]
    if (
        not isinstance(final_trace, dict)
        or final_trace.get("optimizer_step") != expected_optimizer_steps
    ):
        raise ValueError("trained MiniMax H3 checkpoint trace is incomplete")
    if hashlib.sha256(_canonical_json(trace)).hexdigest() != payload["training_trace_sha256"]:
        raise ValueError("trained MiniMax H3 checkpoint training trace changed")
    identity = MiniMaxH3CheckpointIdentity(
        observed_checkpoint_sha256,
        expected_foundation_sha256,
        model_configuration_sha256,
        _digest(payload["module_state_sha256"], "module_state_sha256"),
        _digest(payload["adapter_sha256"], "adapter_sha256"),
        _digest(payload["scorer_sha256"], "scorer_sha256"),
        expected_protocol_sha256,
        optimizer_steps,
    ).validate()
    return payload, identity


def inspect_minimax_h3_checkpoint(
    path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_foundation_sha256: str,
    expected_protocol_sha256: str,
    model_configuration_sha256: str,
) -> MiniMaxH3CheckpointIdentity:
    """Authenticate memory identity before Comfy allocates foundation models."""
    _, identity = _load_checkpoint_identity(
        path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_foundation_sha256=expected_foundation_sha256,
        expected_protocol_sha256=expected_protocol_sha256,
        model_configuration_sha256=model_configuration_sha256,
    )
    return identity


class MiniMaxH3StoryRuntime:
    """Own the trained memory adapter and a frozen MiniMax H3 video codec."""

    def __init__(
        self,
        modules: MemoryModules,
        codec: MiniMaxH3Codec,
        checkpoint: MiniMaxH3CheckpointIdentity,
        *,
        vae_sha256: str,
    ) -> None:
        self.modules = modules.validate()
        if not isinstance(self.modules.bridge, H3LatentHistoryBridge):
            raise ValueError("MiniMax H3 runtime bridge architecture changed")
        if not isinstance(self.modules.scorer, LocalExceptionScorer):
            raise ValueError("MiniMax H3 runtime scorer architecture changed")
        if not isinstance(codec, MiniMaxH3Codec):
            raise ValueError("codec must satisfy the MiniMaxH3Codec contract")
        self.codec = codec
        self.checkpoint = checkpoint.validate()
        if module_state_sha256(self.modules) != self.checkpoint.module_state_sha256:
            raise ValueError("MiniMax H3 runtime module state changed")
        self.vae_sha256 = _digest(vae_sha256, "vae_sha256")
        self.adapter_sha256 = module_sha256(
            self.modules.bridge,
            domain=b"duet-x-minimax-h3-bridge-v1",
        )
        self.scorer_sha256 = module_sha256(
            self.modules.scorer,
            domain=b"duet-x-minimax-h3-scorer-v1",
        )
        if self.adapter_sha256 != self.checkpoint.adapter_sha256:
            raise ValueError("MiniMax H3 bridge tensor digest changed")
        if self.scorer_sha256 != self.checkpoint.scorer_sha256:
            raise ValueError("MiniMax H3 scorer tensor digest changed")
        contract = MemoryContract.minimax_h3(adapter_fingerprint=self.adapter_sha256).validate()
        self._identity = StoryMemoryIdentity(
            StoryMemoryBackend.MINIMAX_H3,
            contract,
            self.checkpoint.checkpoint_sha256,
            self.checkpoint.model_configuration_sha256,
        ).validate()

    @property
    def identity(self) -> StoryMemoryIdentity:
        return self._identity

    @property
    def bridge(self) -> H3LatentHistoryBridge:
        return cast(H3LatentHistoryBridge, self.modules.bridge)

    @classmethod
    def load_pinned(
        cls,
        path: Path,
        *,
        expected_checkpoint_sha256: str,
        expected_foundation_sha256: str,
        expected_protocol_sha256: str,
        model_configuration_sha256: str,
        vae_sha256: str,
        codec: MiniMaxH3Codec,
        device: torch.device,
    ) -> MiniMaxH3StoryRuntime:
        """Load an exact completed checkpoint and reject every identity mismatch."""
        if not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device")
        _digest(vae_sha256, "vae_sha256")
        payload, identity = _load_checkpoint_identity(
            path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_foundation_sha256=expected_foundation_sha256,
            expected_protocol_sha256=expected_protocol_sha256,
            model_configuration_sha256=model_configuration_sha256,
        )
        states = _mapping(payload["modules"], "MiniMax H3 modules", _MODULE_NAMES)
        modules = build_memory_modules()
        for name, module in modules.named().items():
            state = states[name]
            if not isinstance(state, dict):
                raise ValueError(f"trained MiniMax H3 module {name} is malformed")
            try:
                module.load_state_dict(state, strict=True)
            except (RuntimeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"trained MiniMax H3 module {name} does not match its architecture"
                ) from error
            if any(
                not bool(torch.isfinite(tensor).all().item())
                for tensor in module.state_dict().values()
            ):
                raise ValueError(f"memory module {name} contains nonfinite weights")
            module.to(device).eval().requires_grad_(False)
        observed_module_state = module_state_sha256(modules)
        if observed_module_state != payload["module_state_sha256"]:
            raise ValueError("trained MiniMax H3 checkpoint tensor digest changed")
        history_effect_report(cast(H3LatentHistoryBridge, modules.bridge), require_effect=True)
        return cls(modules, codec, identity, vae_sha256=vae_sha256)

    def materialize_frame(self, frame: np.ndarray[Any, Any]) -> EncodedStoryFrame:
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.shape != _FRAME_SHAPE
            or not frame.flags.c_contiguous
        ):
            raise ValueError("MiniMax H3 frame must be contiguous RGB uint8 384x384")
        try:
            latent = self.codec.encode_frame(frame)
        except Exception as error:
            raise ValueError("MiniMax H3 VAE frame encoding failed") from error
        if not isinstance(latent, torch.Tensor):
            raise ValueError("MiniMax H3 VAE must return a tensor latent")
        canonical = latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
        preprocessing_sha256 = hashlib.sha256(
            b"duet-story-minimax-h3-rgb384-v1\0" + frame.tobytes()
        ).hexdigest()
        return EncodedStoryFrame(
            canonical,
            preprocessing_sha256,
            self.vae_sha256,
            self.adapter_sha256,
            self.scorer_sha256,
        ).validate(self.identity.contract)

    @torch.inference_mode()
    def score(self, latent: torch.Tensor, event_type: int, timestamp: float) -> float:
        device = next(self.modules.scorer.parameters()).device
        score = self.modules.scorer(
            latent.to(device=device),
            event_types=torch.tensor([event_type], device=device, dtype=torch.int64),
            timestamps=torch.tensor([timestamp], device=device, dtype=torch.float32),
        )
        if not isinstance(score, torch.Tensor) or tuple(score.shape) != (1,):
            raise ValueError("trained MiniMax H3 scorer returned an unexpected boundary")
        value = float(score.item())
        if not np.isfinite(value):
            raise ValueError("trained MiniMax H3 scorer returned a nonfinite value")
        return value

    def decode(self, latent: torch.Tensor) -> np.ndarray[Any, Any]:
        if (
            not isinstance(latent, torch.Tensor)
            or latent.ndim != 5
            or latent.shape[0] != 1
            or latent.shape[1] != 24
            or not torch.is_floating_point(latent)
            or not bool(torch.isfinite(latent).all().item())
        ):
            raise ValueError("MiniMax H3 decode requires a finite [1,24,F,H,W] latent")
        try:
            value = self.codec.decode_frame(latent)
        except Exception as error:
            raise ValueError("MiniMax H3 VAE frame decoding failed") from error
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.uint8
            or value.shape != _FRAME_SHAPE
        ):
            raise ValueError("MiniMax H3 VAE decoder returned an unexpected RGB boundary")
        return np.ascontiguousarray(value)

    def close(self) -> None:
        try:
            self.codec.close()
        finally:
            for module in self.modules.named().values():
                module.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


__all__ = (
    "MiniMaxH3CheckpointIdentity",
    "MiniMaxH3Codec",
    "MiniMaxH3StoryRuntime",
    "inspect_minimax_h3_checkpoint",
)
