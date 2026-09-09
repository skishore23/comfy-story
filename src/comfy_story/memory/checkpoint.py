"""Versioned H3 memory checkpoint architecture and atomic export.

Wire identifiers and hash domains preserve authentication of existing checkpoints.
No research runner or training configuration is imported."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import torch

from comfy_story.memory.bridge import H3LatentHistoryBridge
from comfy_story.memory.modules import (
    CheckpointGatedModule,
    CheckpointResamplerModule,
    LocalExceptionScorer,
    MemoryModules,
)
from comfy_story.tensors import tensor_sha256

MODULE_SEED = 20_260_828
TOTAL_OPTIMIZER_STEPS = 2_000

_FORMAT = "duet-x-minimax-h3-training-protocol-v1"
_TRAINABLE_MODULES = ("bridge", "gated", "resampler", "scorer")
_FROZEN_MODULES = ("minimax_h3", "qwen3_vl", "video_vae", "audio_vae")
MINIMAX_H3_RUNTIME_CHECKPOINT_FORMAT = "duet-x-minimax-h3-trainable-v1"
_SHA256_HEX = frozenset("0123456789abcdef")


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


@dataclass(frozen=True, slots=True)
class MemoryProtocol:
    """Auditable architecture and foundation freeze rules for MiniMax memory."""

    format: str
    history_items: int
    protected_exceptions: int
    latent_channels: int
    operator_size: int
    optimizer_steps: int
    module_seed: int
    trainable_modules: tuple[str, ...]
    frozen_modules: tuple[str, ...]

    @classmethod
    def default(cls) -> Self:
        return cls(
            _FORMAT,
            8,
            2,
            24,
            16,
            TOTAL_OPTIMIZER_STEPS,
            MODULE_SEED,
            _TRAINABLE_MODULES,
            _FROZEN_MODULES,
        )

    def validate(self) -> Self:
        if self.format != _FORMAT:
            raise ValueError("MiniMax H3 training protocol format changed")
        if self.history_items != 8:
            raise ValueError("MiniMax H3 training requires eight history items")
        if self.protected_exceptions != 2:
            raise ValueError("MiniMax H3 training requires two protected exceptions")
        if self.latent_channels != 24:
            raise ValueError("MiniMax H3 training requires 24 latent channels")
        if self.operator_size != 16:
            raise ValueError("MiniMax H3 training requires operator size 16")
        if self.optimizer_steps != TOTAL_OPTIMIZER_STEPS:
            raise ValueError("MiniMax H3 training requires 2000 optimizer steps")
        if self.module_seed != MODULE_SEED:
            raise ValueError("MiniMax H3 module seed changed")
        if self.trainable_modules != _TRAINABLE_MODULES:
            raise ValueError("MiniMax H3 trainable module roster changed")
        if self.frozen_modules != _FROZEN_MODULES:
            raise ValueError("MiniMax H3 frozen foundation roster changed")
        return self

    def fingerprint(self) -> str:
        self.validate()
        encoded = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def build_memory_modules() -> MemoryModules:
    """Initialize the only four trainable MiniMax memory module families."""
    protocol = MemoryProtocol.default().validate()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(protocol.module_seed)
        if devices:
            torch.cuda.manual_seed_all(protocol.module_seed)
        modules = MemoryModules(
            bridge=H3LatentHistoryBridge(operator_size=protocol.operator_size),
            gated=CheckpointGatedModule(channels=protocol.latent_channels),
            resampler=CheckpointResamplerModule(channels=protocol.latent_channels),
            scorer=LocalExceptionScorer(channels=protocol.latent_channels),
        )
    return modules.validate()


def module_state_sha256(modules: MemoryModules) -> str:
    """Hash the complete named MiniMax trainable tensor roster."""
    rows = [
        [module_name, tensor_name, tensor_sha256(tensor)]
        for module_name, module in sorted(modules.validate().named().items())
        for tensor_name, tensor in sorted(module.state_dict().items())
    ]
    return hashlib.sha256(
        b"duet-x-minimax-h3-modules-v1\0" + _canonical_json(rows) + b"\n"
    ).hexdigest()


def module_sha256(module: torch.nn.Module, *, domain: bytes) -> str:
    """Hash one named module state under an explicit domain separator."""
    if not isinstance(domain, bytes) or not domain:
        raise ValueError("module hash domain must be nonempty bytes")
    rows = [[name, tensor_sha256(tensor)] for name, tensor in sorted(module.state_dict().items())]
    return hashlib.sha256(domain + b"\0" + _canonical_json(rows) + b"\n").hexdigest()


def save_minimax_h3_runtime_checkpoint(
    path: Path,
    modules: MemoryModules,
    *,
    foundation_sha256: str,
    model_configuration_sha256: str,
    optimizer_steps: int,
    training_trace: tuple[dict[str, object], ...],
) -> None:
    """Atomically publish one trained, runtime-only MiniMax memory checkpoint."""
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise ValueError("MiniMax H3 checkpoint path must be absolute and non-symlinked")
    path.parent.mkdir(parents=True, exist_ok=True)
    _digest(foundation_sha256, "foundation_sha256")
    _digest(model_configuration_sha256, "model_configuration_sha256")
    protocol = MemoryProtocol.default().validate()
    if type(optimizer_steps) is not int or optimizer_steps != protocol.optimizer_steps:
        raise ValueError("trained MiniMax H3 checkpoint requires 2000 optimizer_steps")
    if not isinstance(training_trace, tuple) or not training_trace:
        raise ValueError("trained MiniMax H3 checkpoint requires a nonempty training trace")
    trace_bytes = _canonical_json(training_trace)
    validated_modules = modules.validate()
    payload: dict[str, object] = {
        "adapter_sha256": module_sha256(
            validated_modules.bridge,
            domain=b"duet-x-minimax-h3-bridge-v1",
        ),
        "format": MINIMAX_H3_RUNTIME_CHECKPOINT_FORMAT,
        "foundation_sha256": foundation_sha256,
        "model_configuration_sha256": model_configuration_sha256,
        "module_state_sha256": module_state_sha256(validated_modules),
        "modules": {
            name: {
                tensor_name: tensor.detach().contiguous().cpu()
                for tensor_name, tensor in module.state_dict().items()
            }
            for name, module in validated_modules.named().items()
        },
        "optimizer_steps": optimizer_steps,
        "protocol_sha256": protocol.fingerprint(),
        "scorer_sha256": module_sha256(
            validated_modules.scorer,
            domain=b"duet-x-minimax-h3-scorer-v1",
        ),
        "training_trace": training_trace,
        "training_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = (
    "MINIMAX_H3_RUNTIME_CHECKPOINT_FORMAT",
    "MemoryProtocol",
    "build_memory_modules",
    "module_sha256",
    "module_state_sha256",
    "save_minimax_h3_runtime_checkpoint",
)
