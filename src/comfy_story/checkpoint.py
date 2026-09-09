"""Strict atomic training bundles for the Duet-X Decision 1 bridge."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self, cast

import torch
from torch import nn

_CHECKPOINT_FORMAT = "duet-x-bridge-checkpoint-v2"
_MODULE_NAMES = ("bridge", "gated", "resampler", "scorer")
_TOP_LEVEL_KEYS = frozenset(
    {
        "format",
        "modules",
        "optimizer",
        "scheduler",
        "scaler",
        "cursor",
        "rng_states",
        "fingerprints",
        "parameter_counts",
        "state_hashes",
        "manifest_sha256",
    }
)
_HASHED_STATE_KEYS = (
    "modules",
    "optimizer",
    "scheduler",
    "scaler",
    "cursor",
    "rng_states",
    "fingerprints",
    "parameter_counts",
)
_SHA256_HEX = frozenset("0123456789abcdef")


def _mapping(value: object, name: str, expected_keys: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a plain mapping")
    result = cast(dict[str, Any], value)
    if any(type(key) is not str for key in result) or set(result) != expected_keys:
        raise ValueError(f"{name} fields do not match the checkpoint schema")
    return result


def _open_mapping(value: object, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a plain mapping")
    result = cast(dict[str, Any], value)
    if any(type(key) is not str for key in result):
        raise ValueError(f"{name} keys must be strings")
    return result


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint")
    return value


def _canonical_state_bytes(value: object) -> bytes:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError("checkpoint hashes require strided tensors")
        tensor = value.detach().contiguous().cpu()
        header = json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        raw = bytes(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return b"tensor:" + len(header).to_bytes(8, "big") + header + raw
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        entries: list[tuple[bytes, bytes]] = []
        for key, item in mapping.items():
            if type(key) not in {str, int}:
                raise ValueError("checkpoint hash mapping keys must be strings or integers")
            key_bytes = _canonical_state_bytes(key)
            entries.append((key_bytes, _canonical_state_bytes(item)))
        result = bytearray(b"dict:")
        for key_bytes, item_bytes in sorted(entries, key=lambda entry: entry[0]):
            result.extend(len(key_bytes).to_bytes(8, "big"))
            result.extend(key_bytes)
            result.extend(len(item_bytes).to_bytes(8, "big"))
            result.extend(item_bytes)
        return bytes(result)
    if type(value) in {list, tuple}:
        prefix = b"list:" if type(value) is list else b"tuple:"
        result = bytearray(prefix)
        for item in cast(Sequence[object], value):
            item_bytes = _canonical_state_bytes(item)
            result.extend(len(item_bytes).to_bytes(8, "big"))
            result.extend(item_bytes)
        return bytes(result)
    if value is None or type(value) in {str, int, float, bool}:
        return b"primitive:" + json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    raise ValueError("checkpoint hash state contains an unsupported value")


def _state_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_state_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointFingerprints:
    """Every immutable run identity required to resume Decision 1."""

    config: str
    runtime: str
    source: str
    data: str
    teacher: str
    product: str

    def validate(self) -> Self:
        for name in self.names():
            _sha256(getattr(self, name), name)
        return self

    @staticmethod
    def names() -> tuple[str, ...]:
        return ("config", "runtime", "source", "data", "teacher", "product")

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        data = _mapping(value, "fingerprints", frozenset(cls.names()))
        return cls(**{name: _sha256(data[name], name) for name in cls.names()}).validate()


@dataclass(frozen=True, slots=True)
class TrainingCursor:
    """Exact completed optimizer/micro-step and next-data positions."""

    optimizer_step: int
    micro_step: int
    data_index: int

    def validate(self) -> Self:
        for name in ("optimizer_step", "micro_step", "data_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"cursor {name} must be a nonnegative integer")
        if self.micro_step != self.optimizer_step * 8:
            raise ValueError("cursor micro_step must equal eight times optimizer_step")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        keys = frozenset({"optimizer_step", "micro_step", "data_index"})
        data = _mapping(value, "cursor", keys)
        if any(type(data[name]) is not int for name in keys):
            raise ValueError("cursor fields must be integers")
        return cls(
            optimizer_step=data["optimizer_step"],
            micro_step=data["micro_step"],
            data_index=data["data_index"],
        ).validate()


@dataclass(frozen=True, slots=True)
class TrainingModules:
    """The only four trainable module families allowed in a bridge bundle."""

    bridge: nn.Module
    gated: nn.Module
    resampler: nn.Module
    scorer: nn.Module

    def validate(self) -> Self:
        modules = self.named()
        if any(not isinstance(module, nn.Module) for module in modules.values()):
            raise ValueError("training modules must be torch modules")
        if len({id(module) for module in modules.values()}) != len(_MODULE_NAMES):
            raise ValueError("training modules must be distinct")
        return self

    def named(self) -> dict[str, nn.Module]:
        return {name: getattr(self, name) for name in _MODULE_NAMES}

    def parameters(self) -> Iterator[nn.Parameter]:
        for module in self.named().values():
            yield from module.parameters()

    def parameter_counts(self) -> dict[str, int]:
        self.validate()
        return {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in self.named().items()
        }


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().contiguous().cpu() for name, tensor in module.state_dict().items()
    }


def _rng_states() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone().cpu() for state in torch.cuda.get_rng_state_all()],
    }


def _build_payload(
    modules: TrainingModules,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    cursor: TrainingCursor,
    fingerprints: CheckpointFingerprints,
) -> dict[str, object]:
    modules.validate()
    cursor.validate()
    fingerprints.validate()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    scaler_state = scaler.state_dict()
    _validate_cursor_alignment(cursor, scheduler_state)
    _validate_optimizer_state(optimizer_state, optimizer, modules, cursor)
    _validate_finite_state(optimizer_state, "optimizer")
    _validate_finite_state(scheduler_state, "scheduler")
    _validate_finite_state(scaler_state, "scaler")
    payload: dict[str, object] = {
        "format": _CHECKPOINT_FORMAT,
        "modules": {name: _cpu_state(module) for name, module in modules.named().items()},
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "scaler": scaler_state,
        "cursor": asdict(cursor),
        "rng_states": _rng_states(),
        "fingerprints": asdict(fingerprints),
        "parameter_counts": modules.parameter_counts(),
    }
    payload["state_hashes"] = _build_state_hashes(payload)
    payload["manifest_sha256"] = _state_sha256(payload)
    return payload


def save_bridge_checkpoint(
    path: Path,
    *,
    modules: TrainingModules,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    cursor: TrainingCursor,
    fingerprints: CheckpointFingerprints,
) -> Path:
    """Atomically save exactly the permitted bridge/scorer training state."""
    if not isinstance(path, Path):
        raise ValueError("checkpoint path must be a pathlib.Path")
    payload = _build_payload(modules, optimizer, scheduler, scaler, cursor, fingerprints)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return path


def _validate_module_states(
    value: object, modules: TrainingModules
) -> dict[str, dict[str, torch.Tensor]]:
    data = _mapping(value, "modules", frozenset(_MODULE_NAMES))
    result: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in modules.named().items():
        state = _open_mapping(data[name], f"modules.{name}")
        current = module.state_dict()
        if set(state) != set(current):
            raise ValueError(f"modules.{name} state fields do not match the module")
        checked: dict[str, torch.Tensor] = {}
        for key, expected in current.items():
            value_tensor = state[key]
            if (
                type(value_tensor) is not torch.Tensor
                or value_tensor.layout != torch.strided
                or value_tensor.shape != expected.shape
                or value_tensor.dtype != expected.dtype
                or not bool(torch.isfinite(value_tensor).all().item())
            ):
                raise ValueError(f"modules.{name} state tensor {key!r} is incompatible")
            checked[key] = value_tensor
        result[name] = checked
    return result


def _validate_parameter_counts(value: object, modules: TrainingModules) -> None:
    data = _mapping(value, "parameter_counts", frozenset(_MODULE_NAMES))
    if any(type(data[name]) is not int or data[name] < 0 for name in _MODULE_NAMES):
        raise ValueError("checkpoint parameter counts must be nonnegative integers")
    if data != modules.parameter_counts():
        raise ValueError("checkpoint module parameter count mismatch")


def _validate_state_schema(
    value: object, current: Mapping[str, object], name: str, *, fixed: frozenset[str] | None = None
) -> dict[str, Any]:
    expected = frozenset(current) if fixed is None else fixed
    return _mapping(value, name, expected)


def _validate_finite_state(value: object, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError(f"{name} tensors must be strided")
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError(f"{name} tensors must be finite")
        return
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        for key, item in mapping.items():
            if type(key) not in {str, int}:
                raise ValueError(f"{name} mapping keys must be strings or integers")
            _validate_finite_state(item, f"{name}.{key}")
        return
    if type(value) in {list, tuple}:
        for index, item in enumerate(cast(Sequence[object], value)):
            _validate_finite_state(item, f"{name}[{index}]")
        return
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{name} numbers must be finite")
    if value is not None and type(value) not in {str, int, float, bool}:
        raise ValueError(f"{name} contains unsupported state")


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> tuple[nn.Parameter, ...]:
    return tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in cast(list[nn.Parameter], group["params"])
    )


def _validate_optimizer_state(
    value: object,
    optimizer: torch.optim.Optimizer,
    modules: TrainingModules,
    cursor: TrainingCursor,
) -> dict[str, Any]:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Decision 1 checkpoints require AdamW")
    module_parameters = tuple(modules.parameters())
    optimizer_parameters = _optimizer_parameters(optimizer)
    if tuple(map(id, optimizer_parameters)) != tuple(map(id, module_parameters)):
        raise ValueError("optimizer parameter coverage must exactly match training modules")

    data = _mapping(value, "optimizer", frozenset({"state", "param_groups"}))
    current = optimizer.state_dict()
    current_groups = current["param_groups"]
    recorded_groups = data["param_groups"]
    if type(recorded_groups) is not list or len(recorded_groups) != len(current_groups):
        raise ValueError("optimizer parameter-group count mismatch")
    expected_parameter_ids: list[int] = []
    parameter_by_id: dict[int, nn.Parameter] = {}
    parameter_offset = 0
    for group_index, (recorded_value, current_value) in enumerate(
        zip(recorded_groups, current_groups, strict=True)
    ):
        if type(recorded_value) is not dict or type(current_value) is not dict:
            raise ValueError("optimizer parameter groups must be plain mappings")
        recorded_group = cast(dict[str, object], recorded_value)
        current_group = cast(dict[str, object], current_value)
        if set(recorded_group) != set(current_group):
            raise ValueError(f"optimizer parameter group {group_index} fields mismatch")
        recorded_ids = recorded_group["params"]
        current_ids = current_group["params"]
        if (
            type(recorded_ids) is not list
            or type(current_ids) is not list
            or recorded_ids != current_ids
            or any(type(parameter_id) is not int for parameter_id in recorded_ids)
        ):
            raise ValueError(f"optimizer parameter group {group_index} IDs mismatch")
        typed_ids = cast(list[int], recorded_ids)
        group_parameters = optimizer_parameters[
            parameter_offset : parameter_offset + len(typed_ids)
        ]
        parameter_offset += len(typed_ids)
        expected_parameter_ids.extend(typed_ids)
        parameter_by_id.update(zip(typed_ids, group_parameters, strict=True))
    if len(set(expected_parameter_ids)) != len(expected_parameter_ids):
        raise ValueError("optimizer parameter IDs must be unique")

    raw_state = data["state"]
    if type(raw_state) is not dict:
        raise ValueError("optimizer state must be a plain mapping")
    state = cast(dict[object, object], raw_state)
    expected_state_ids = set(expected_parameter_ids) if cursor.optimizer_step else set()
    if set(state) != expected_state_ids:
        raise ValueError("optimizer state parameter IDs do not exactly cover the modules")
    for parameter_id, raw_parameter_state in state.items():
        if type(parameter_id) is not int or type(raw_parameter_state) is not dict:
            raise ValueError("optimizer per-parameter state has invalid structure")
        parameter_state = cast(dict[str, object], raw_parameter_state)
        if set(parameter_state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("optimizer per-parameter state fields mismatch")
        parameter = parameter_by_id[parameter_id]
        step = parameter_state["step"]
        if (
            type(step) is not torch.Tensor
            or step.shape != torch.Size([])
            or step.dtype != torch.float32
            or float(step.item()) != cursor.optimizer_step
        ):
            raise ValueError("optimizer step state does not match the exact cursor")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state[moment_name]
            if (
                type(moment) is not torch.Tensor
                or moment.layout != torch.strided
                or moment.shape != parameter.shape
                or moment.dtype != parameter.dtype
            ):
                raise ValueError(f"optimizer {moment_name} schema mismatch")
    return data


def _build_state_hashes(payload: Mapping[str, object]) -> dict[str, object]:
    modules = cast(Mapping[str, object], payload["modules"])
    return {
        "modules": {name: _state_sha256(modules[name]) for name in _MODULE_NAMES},
        **{name: _state_sha256(payload[name]) for name in _HASHED_STATE_KEYS if name != "modules"},
    }


def _validate_state_hashes(payload: Mapping[str, Any]) -> None:
    hashes = _mapping(payload["state_hashes"], "state_hashes", frozenset(_HASHED_STATE_KEYS))
    module_hashes = _mapping(hashes["modules"], "state_hashes.modules", frozenset(_MODULE_NAMES))
    expected = _build_state_hashes(payload)
    expected_modules = cast(dict[str, str], expected["modules"])
    for name in _MODULE_NAMES:
        recorded = _sha256(module_hashes[name], f"state_hashes.modules.{name}")
        if recorded != expected_modules[name]:
            raise ValueError(f"checkpoint module {name} content hash mismatch")
    for name in _HASHED_STATE_KEYS:
        if name == "modules":
            continue
        recorded = _sha256(hashes[name], f"state_hashes.{name}")
        if recorded != expected[name]:
            raise ValueError(f"checkpoint {name} content hash mismatch")
    recorded_manifest = _sha256(payload["manifest_sha256"], "manifest_sha256")
    manifest_payload = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded_manifest != _state_sha256(manifest_payload):
        raise ValueError("checkpoint manifest hash mismatch")


def _validate_cursor_alignment(
    cursor: TrainingCursor, scheduler_state: Mapping[str, object]
) -> None:
    last_epoch = scheduler_state.get("last_epoch")
    if type(last_epoch) is not int or last_epoch != cursor.optimizer_step:
        raise ValueError("scheduler state does not match the exact checkpoint cursor")


def _validate_rng_states(value: object) -> dict[str, object]:
    data = _mapping(value, "rng_states", frozenset({"python", "torch", "cuda"}))
    if type(data["torch"]) is not torch.Tensor or data["torch"].dtype != torch.uint8:
        raise ValueError("rng_states.torch must be a uint8 tensor")
    if type(data["cuda"]) is not list or any(
        type(state) is not torch.Tensor or state.dtype != torch.uint8 for state in data["cuda"]
    ):
        raise ValueError("rng_states.cuda must be a list of uint8 tensors")
    if type(data["python"]) is not tuple:
        raise ValueError("rng_states.python must be a Python random state tuple")
    probe = random.Random()
    try:
        probe.setstate(data["python"])
    except (TypeError, ValueError) as error:
        raise ValueError("rng_states.python is invalid") from error
    return cast(dict[str, object], data)


def load_bridge_checkpoint(
    path: Path,
    *,
    modules: TrainingModules,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    expected_fingerprints: CheckpointFingerprints,
) -> TrainingCursor:
    """Strictly verify and atomically restore a complete resumable bundle."""
    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError(path)
    modules.validate()
    expected_fingerprints.validate()
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("bridge checkpoint could not be loaded safely") from error
    payload = _mapping(raw, "checkpoint", _TOP_LEVEL_KEYS)
    if payload["format"] != _CHECKPOINT_FORMAT:
        raise ValueError("unsupported bridge checkpoint format")
    recorded_fingerprints = CheckpointFingerprints.from_mapping(payload["fingerprints"])
    if recorded_fingerprints != expected_fingerprints:
        raise ValueError("bridge checkpoint fingerprint mismatch")
    cursor = TrainingCursor.from_mapping(payload["cursor"])
    _validate_parameter_counts(payload["parameter_counts"], modules)
    module_states = _validate_module_states(payload["modules"], modules)
    optimizer_state = _validate_optimizer_state(payload["optimizer"], optimizer, modules, cursor)
    scheduler_state = _validate_state_schema(
        payload["scheduler"], scheduler.state_dict(), "scheduler"
    )
    scaler_state = _validate_state_schema(payload["scaler"], scaler.state_dict(), "scaler")
    _validate_cursor_alignment(cursor, scheduler_state)
    _validate_finite_state(optimizer_state, "optimizer")
    _validate_finite_state(scheduler_state, "scheduler")
    _validate_finite_state(scaler_state, "scaler")
    rng_states = _validate_rng_states(payload["rng_states"])
    _validate_state_hashes(payload)

    module_backups = {
        name: copy.deepcopy(module.state_dict()) for name, module in modules.named().items()
    }
    optimizer_backup = copy.deepcopy(optimizer.state_dict())
    scheduler_backup = copy.deepcopy(scheduler.state_dict())
    scaler_backup = copy.deepcopy(scaler.state_dict())
    python_backup = random.getstate()
    torch_backup = torch.get_rng_state()
    cuda_backup = torch.cuda.get_rng_state_all()
    try:
        for name, module in modules.named().items():
            module.load_state_dict(module_states[name], strict=True)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        scaler.load_state_dict(scaler_state)
        random.setstate(cast(tuple[Any, ...], rng_states["python"]))
        torch.set_rng_state(cast(torch.Tensor, rng_states["torch"]))
        cuda_states = cast(list[torch.Tensor], rng_states["cuda"])
        if cuda_states:
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError("checkpoint CUDA RNG device count mismatch")
            torch.cuda.set_rng_state_all(cuda_states)
    except Exception as error:
        for name, module in modules.named().items():
            module.load_state_dict(module_backups[name], strict=True)
        optimizer.load_state_dict(optimizer_backup)
        scheduler.load_state_dict(scheduler_backup)
        scaler.load_state_dict(scaler_backup)
        random.setstate(python_backup)
        torch.set_rng_state(torch_backup)
        if cuda_backup:
            torch.cuda.set_rng_state_all(cuda_backup)
        if isinstance(error, ValueError):
            raise
        raise ValueError("bridge checkpoint state is incompatible") from error
    return cursor


def inspect_bridge_checkpoint_fingerprints(path: Path) -> CheckpointFingerprints:
    """Read fingerprints only after validating the complete safe checkpoint hash envelope."""
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("bridge checkpoint could not be loaded safely") from error
    payload = _mapping(raw, "checkpoint", _TOP_LEVEL_KEYS)
    if payload["format"] != _CHECKPOINT_FORMAT:
        raise ValueError("unsupported bridge checkpoint format")
    _validate_state_hashes(payload)
    return CheckpointFingerprints.from_mapping(payload["fingerprints"])


__all__ = (
    "CheckpointFingerprints",
    "TrainingCursor",
    "TrainingModules",
    "inspect_bridge_checkpoint_fingerprints",
    "load_bridge_checkpoint",
    "save_bridge_checkpoint",
)
