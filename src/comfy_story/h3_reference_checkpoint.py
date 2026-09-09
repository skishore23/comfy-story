"""Authenticated runtime checkpoints for the H3 reference-context compiler."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import torch

from comfy_story.contracts import tensor_sha256
from comfy_story.h3_reference_compressors import (
    AlignedReferenceCompressor,
    DuetReferenceCompressor,
    GatedReferenceCompressor,
    ResamplerReferenceCompressor,
)
from comfy_story.h3_reference_contracts import H3ReferenceMethod
from comfy_story.minimax_h3_memory import inspect_minimax_h3_checkpoint
from comfy_story.minimax_h3_training import (
    build_minimax_h3_training_modules,
)
from comfy_story.minimax_h3_training import (
    module_state_sha256 as parent_module_state_sha256,
)

COMPILER_CHECKPOINT_FORMAT = "duet-x-h3-reference-compiler-v1"
_WARM_START_SCOPE = "story-memory-visual-latents"
_MODULE_NAMES = frozenset({"bridge", "gated", "resampler"})
_PAYLOAD_KEYS = frozenset(
    {
        "format",
        "foundation_sha256",
        "latent_channels",
        "model_configuration_sha256",
        "module_state_sha256",
        "modules",
        "operator_size",
        "parent_checkpoint_sha256",
        "parent_module_state_sha256",
        "warm_start_scope",
    }
)
_PARENT_KEYS = frozenset(
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
_MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024


def _digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, field: str, keys: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be a plain mapping")
    result = cast(dict[str, object], value)
    if any(type(key) is not str for key in result) or set(result) != keys:
        raise ValueError(f"{field} field roster changed")
    return result


def _regular_file_bytes(path: Path, field: str) -> bytes:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError(f"{field} must be an absolute regular file")
    size = path.stat().st_size
    if not 0 < size <= _MAX_CHECKPOINT_BYTES:
        raise ValueError(f"{field} size is invalid")
    return path.read_bytes()


def _safe_load(encoded: bytes, field: str) -> object:
    try:
        return torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"{field} could not be loaded safely") from error


def _module_states(value: object) -> dict[str, dict[str, torch.Tensor]]:
    modules = _mapping(value, "compiler modules", _MODULE_NAMES)
    checked: dict[str, dict[str, torch.Tensor]] = {}
    for module_name, raw_state in modules.items():
        if type(raw_state) is not dict:
            raise ValueError(f"compiler module {module_name} must be a plain state mapping")
        state = cast(dict[str, object], raw_state)
        if not state or any(type(name) is not str for name in state):
            raise ValueError(f"compiler module {module_name} state is malformed")
        tensors: dict[str, torch.Tensor] = {}
        for name, value_tensor in state.items():
            if (
                type(value_tensor) is not torch.Tensor
                or value_tensor.layout != torch.strided
                or not bool(torch.isfinite(value_tensor).all().item())
            ):
                raise ValueError(f"compiler module {module_name}.{name} is not a finite tensor")
            tensors[name] = value_tensor.detach().contiguous().cpu()
        checked[module_name] = tensors
    return checked


def _module_state_sha256(states: dict[str, dict[str, torch.Tensor]]) -> str:
    rows = [
        [module_name, tensor_name, tensor_sha256(tensor)]
        for module_name, state in sorted(states.items())
        for tensor_name, tensor in sorted(state.items())
    ]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(
        b"duet-x-h3-reference-compiler-modules-v1\0" + encoded + b"\n"
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class H3ReferenceCompilerIdentity:
    """Complete immutable identity for compiler inference weights."""

    format: str
    checkpoint_sha256: str
    parent_checkpoint_sha256: str
    parent_module_state_sha256: str
    foundation_sha256: str
    model_configuration_sha256: str
    module_state_sha256: str
    latent_channels: int
    operator_size: int
    warm_start_scope: str

    def validate(self) -> Self:
        if self.format != COMPILER_CHECKPOINT_FORMAT:
            raise ValueError("H3 reference compiler checkpoint format changed")
        for field in (
            "checkpoint_sha256",
            "parent_checkpoint_sha256",
            "parent_module_state_sha256",
            "foundation_sha256",
            "model_configuration_sha256",
            "module_state_sha256",
        ):
            _digest(getattr(self, field), field)
        if self.latent_channels != 24 or self.operator_size != 16:
            raise ValueError("H3 reference compiler architecture changed")
        if self.warm_start_scope != _WARM_START_SCOPE:
            raise ValueError("H3 reference compiler warm-start scope changed")
        return self


@dataclass(frozen=True, slots=True)
class LoadedH3ReferenceCompilerCheckpoint:
    """Authenticated identity and inference modules loaded from one file."""

    identity: H3ReferenceCompilerIdentity
    gated: GatedReferenceCompressor
    resampler: ResamplerReferenceCompressor
    duet: DuetReferenceCompressor

    def compressor_for(self, method: H3ReferenceMethod) -> AlignedReferenceCompressor:
        if method is H3ReferenceMethod.GATED:
            return self.gated
        if method is H3ReferenceMethod.RESAMPLER:
            return self.resampler
        if method is H3ReferenceMethod.DUET:
            return self.duet
        if method is H3ReferenceMethod.DUET_X:
            compressor = DuetReferenceCompressor(method=H3ReferenceMethod.DUET_X)
            compressor.bridge.load_state_dict(self.duet.bridge.state_dict(), strict=True)
            device = next(self.duet.parameters()).device
            return compressor.to(device).eval().requires_grad_(False)
        raise ValueError(f"{method.value} does not use compiler inference weights")


def _load_parent_payload(path: Path) -> tuple[dict[str, object], str]:
    encoded = _regular_file_bytes(path, "parent MiniMax H3 checkpoint")
    parent_sha256 = hashlib.sha256(encoded).hexdigest()
    payload = _mapping(_safe_load(encoded, "parent MiniMax H3 checkpoint"), "parent", _PARENT_KEYS)
    foundation = _digest(payload["foundation_sha256"], "foundation_sha256")
    model_configuration = _digest(
        payload["model_configuration_sha256"], "model_configuration_sha256"
    )
    protocol = _digest(payload["protocol_sha256"], "protocol_sha256")
    inspect_minimax_h3_checkpoint(
        path,
        expected_checkpoint_sha256=parent_sha256,
        expected_foundation_sha256=foundation,
        expected_protocol_sha256=protocol,
        model_configuration_sha256=model_configuration,
    )
    raw_modules = _mapping(
        payload["modules"],
        "parent MiniMax H3 modules",
        frozenset({"bridge", "gated", "resampler", "scorer"}),
    )
    modules = build_minimax_h3_training_modules()
    try:
        for name, module in modules.named().items():
            state = raw_modules[name]
            if type(state) is not dict:
                raise ValueError(f"parent MiniMax H3 module {name} state is malformed")
            module.load_state_dict(cast(dict[str, torch.Tensor], state), strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("parent MiniMax H3 module architecture mismatch") from error
    if parent_module_state_sha256(modules) != payload["module_state_sha256"]:
        raise ValueError("parent MiniMax H3 module state SHA-256 mismatch")
    return payload, parent_sha256


def export_h3_reference_compiler_checkpoint(
    source: Path, output: Path
) -> H3ReferenceCompilerIdentity:
    """Extract only visual compiler modules from an authenticated Story checkpoint."""
    if (
        not isinstance(output, Path)
        or not output.is_absolute()
        or output.is_symlink()
        or output == source
    ):
        raise ValueError("compiler checkpoint output must be a distinct absolute non-symlink path")
    parent, parent_sha256 = _load_parent_payload(source)
    parent_modules = _mapping(
        parent["modules"],
        "parent MiniMax H3 modules",
        frozenset({"bridge", "gated", "resampler", "scorer"}),
    )
    states = _module_states({name: parent_modules[name] for name in _MODULE_NAMES})
    module_digest = _module_state_sha256(states)
    payload: dict[str, object] = {
        "format": COMPILER_CHECKPOINT_FORMAT,
        "foundation_sha256": _digest(parent["foundation_sha256"], "foundation_sha256"),
        "latent_channels": 24,
        "model_configuration_sha256": _digest(
            parent["model_configuration_sha256"], "model_configuration_sha256"
        ),
        "module_state_sha256": module_digest,
        "modules": states,
        "operator_size": 16,
        "parent_checkpoint_sha256": parent_sha256,
        "parent_module_state_sha256": _digest(
            parent["module_state_sha256"], "parent_module_state_sha256"
        ),
        "warm_start_scope": _WARM_START_SCOPE,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    checkpoint_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    return H3ReferenceCompilerIdentity(
        COMPILER_CHECKPOINT_FORMAT,
        checkpoint_sha256,
        parent_sha256,
        cast(str, payload["parent_module_state_sha256"]),
        cast(str, payload["foundation_sha256"]),
        cast(str, payload["model_configuration_sha256"]),
        module_digest,
        24,
        16,
        _WARM_START_SCOPE,
    ).validate()


def load_h3_reference_compiler_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_model_configuration_sha256: str,
    device: torch.device,
) -> LoadedH3ReferenceCompilerCheckpoint:
    """Authenticate and instantiate the exact compiler module roster."""
    _digest(expected_sha256, "expected_sha256")
    _digest(expected_model_configuration_sha256, "expected_model_configuration_sha256")
    if not isinstance(device, torch.device):
        raise ValueError("compiler checkpoint device must be a torch.device")
    encoded = _regular_file_bytes(path, "H3 reference compiler checkpoint")
    observed_sha256 = hashlib.sha256(encoded).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("H3 reference compiler checkpoint content SHA-256 mismatch")
    payload = _mapping(
        _safe_load(encoded, "H3 reference compiler checkpoint"),
        "H3 reference compiler checkpoint",
        _PAYLOAD_KEYS,
    )
    identity = H3ReferenceCompilerIdentity(
        cast(str, payload["format"]),
        observed_sha256,
        _digest(payload["parent_checkpoint_sha256"], "parent_checkpoint_sha256"),
        _digest(payload["parent_module_state_sha256"], "parent_module_state_sha256"),
        _digest(payload["foundation_sha256"], "foundation_sha256"),
        _digest(payload["model_configuration_sha256"], "model_configuration_sha256"),
        _digest(payload["module_state_sha256"], "module_state_sha256"),
        cast(int, payload["latent_channels"]),
        cast(int, payload["operator_size"]),
        cast(str, payload["warm_start_scope"]),
    ).validate()
    if identity.model_configuration_sha256 != expected_model_configuration_sha256:
        raise ValueError("H3 reference compiler model configuration mismatch")
    states = _module_states(payload["modules"])
    if _module_state_sha256(states) != identity.module_state_sha256:
        raise ValueError("H3 reference compiler module state SHA-256 mismatch")
    gated = GatedReferenceCompressor()
    resampler = ResamplerReferenceCompressor()
    duet = DuetReferenceCompressor()
    try:
        gated.load_state_dict(states["gated"], strict=True)
        resampler.load_state_dict(states["resampler"], strict=True)
        duet.bridge.load_state_dict(states["bridge"], strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("H3 reference compiler module architecture mismatch") from error
    for module in (gated, resampler, duet):
        module.to(device).eval().requires_grad_(False)
    return LoadedH3ReferenceCompilerCheckpoint(identity, gated, resampler, duet)


__all__ = (
    "COMPILER_CHECKPOINT_FORMAT",
    "H3ReferenceCompilerIdentity",
    "LoadedH3ReferenceCompilerCheckpoint",
    "export_h3_reference_compiler_checkpoint",
    "load_h3_reference_compiler_checkpoint",
)
