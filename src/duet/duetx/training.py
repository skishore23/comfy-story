"""Frozen-foundation, resumable training for Duet-X Decision 1."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from duet.duetx.checkpoint import (
    CheckpointFingerprints,
    TrainingCursor,
    TrainingModules,
    load_bridge_checkpoint,
    save_bridge_checkpoint,
)
from duet.duetx.config import Decision1Config, Method
from duet.duetx.losses import Decision1Losses, LocalExceptionScorer
from duet.duetx.ltx_bridge import LTXLatentHistoryBridge
from duet.duetx.methods import build_method
from duet.duetx.teacher import InfluenceRecord, validate_completed_influence_records

TOTAL_OPTIMIZER_STEPS = 2_000
WARMUP_STEPS = 200
ACCUMULATION_STEPS = 8
GRADIENT_CLIP_NORM = 1.0
CHECKPOINT_INTERVAL = 100
MODULE_SEED = 20_260_828
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01

LossFunction = Callable[
    [int, int, tuple[InfluenceRecord, ...], TrainingModules, nn.Module | None], Decision1Losses
]
FoundationDigestProvider = Callable[[], str]

_METRIC_KEYS = frozenset(
    {
        *Decision1Losses.component_names(),
        "total",
        "optimizer_step",
        "micro_step",
        "data_index",
        "learning_rate",
        "gradient_norm",
        "module_seed",
        "product_fingerprint",
        "teacher_fingerprint",
    }
)
_METRIC_FLOAT_KEYS = frozenset(
    {
        *Decision1Losses.component_names(),
        "total",
        "learning_rate",
        "gradient_norm",
    }
)
_METRIC_INT_KEYS = frozenset({"optimizer_step", "micro_step", "data_index", "module_seed"})


class TrainStepResult:
    """Primitive-ready outcome of one gradient-accumulation micro-step."""

    __slots__ = (
        "applied_learning_rate",
        "clipped_gradient_norm",
        "losses",
        "optimizer_stepped",
    )

    def __init__(
        self,
        *,
        losses: dict[str, float],
        optimizer_stepped: bool,
        clipped_gradient_norm: float,
        applied_learning_rate: float | None,
    ) -> None:
        self.losses = losses
        self.optimizer_stepped = optimizer_stepped
        self.clipped_gradient_norm = clipped_gradient_norm
        self.applied_learning_rate = applied_learning_rate


def _foundation_items_digest(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Hash a logical named foundation-state roster with the canonical algorithm."""
    entries: list[dict[str, str]] = []
    observed_names: set[str] = set()
    for name, tensor in sorted(items):
        if type(name) is not str or not name or name in observed_names:
            raise ValueError("foundation state names must be unique nonempty strings")
        observed_names.add(name)
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError("foundation state must contain only strided tensors")
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("foundation state tensors must be finite")
        contiguous = tensor.detach().contiguous().cpu()
        raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
        header = json.dumps(
            {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        entries.append({"name": name, "sha256": hashlib.sha256(header + b"\n" + raw).hexdigest()})
    encoded = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def foundation_digest(foundation: nn.Module) -> str:
    """Hash every named foundation tensor without serializing it."""
    if not isinstance(foundation, nn.Module):
        raise ValueError("foundation must be a torch module")
    return _foundation_items_digest(foundation.state_dict().items())


def freeze_foundation(foundation: nn.Module) -> str:
    """Put the external foundation in inference mode and freeze every parameter."""
    digest = foundation_digest(foundation)
    foundation.eval()
    for parameter in foundation.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return digest


def _assert_foundation_frozen(foundation: nn.Module) -> None:
    if foundation.training:
        raise ValueError("foundation must remain in evaluation mode")
    for parameter in foundation.parameters():
        if parameter.requires_grad or parameter.grad is not None:
            raise ValueError("foundation parameters must remain fully frozen")


def _assert_foundation_excluded(foundation: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    foundation_ids = {id(parameter) for parameter in foundation.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in cast(list[nn.Parameter], group["params"])
    }
    if foundation_ids & optimizer_ids:
        raise ValueError("optimizer must exclude every foundation parameter")


def build_training_modules() -> TrainingModules:
    """Initialize all trainable families from the one locked module seed."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(MODULE_SEED)
        if devices:
            torch.cuda.manual_seed_all(MODULE_SEED)
        modules = TrainingModules(
            bridge=LTXLatentHistoryBridge(channels=128, operator_size=16),
            gated=build_method(Method.GATED_CORE_TOPK),
            resampler=build_method(Method.CENTRALIZED_RESAMPLER_TOPK),
            scorer=LocalExceptionScorer(channels=128),
        )
    return modules.validate()


def build_optimizer(
    modules: TrainingModules,
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> torch.optim.AdamW:
    """Build the locked optimizer over only the four permitted module families."""
    modules.validate()
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(float(weight_decay))
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be finite and nonnegative")
    parameters = tuple(parameter for parameter in modules.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("training modules must expose trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm linearly for exactly 200 completed optimizer steps, then remain constant."""

    def scale(step: int) -> float:
        return min(1.0, (step + 1) / WARMUP_STEPS)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _module_device(modules: TrainingModules) -> torch.device:
    parameters = tuple(modules.parameters())
    if not parameters:
        raise ValueError("training modules must have parameters")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("training modules must share one device")
    return parameters[0].device


def _autocast_context(device: torch.device) -> AbstractContextManager[Any]:
    # PyTorch 2.8's stub omits the typed CUDA BF16 capability query.
    is_bf16_supported = cast(Callable[[], bool], torch.cuda.is_bf16_supported)
    cuda_bf16 = device.type == "cuda" and torch.cuda.is_available() and is_bf16_supported()
    supported = device.type == "cpu" or cuda_bf16
    if supported:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _finite_gradients(parameters: Sequence[nn.Parameter]) -> None:
    for parameter in parameters:
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            raise ValueError("training produced a nonfinite gradient")


def train_step(
    loss_fn: Callable[[], Decision1Losses],
    *,
    modules: TrainingModules,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    accumulation_index: int,
) -> TrainStepResult:
    """Run one of eight micro-steps and update only on the eighth."""
    if type(accumulation_index) is not int or not 0 <= accumulation_index < ACCUMULATION_STEPS:
        raise ValueError("accumulation_index must be in range [0,8)")
    parameters = tuple(parameter for parameter in modules.parameters() if parameter.requires_grad)
    if accumulation_index == 0:
        optimizer.zero_grad(set_to_none=True)
    with _autocast_context(_module_device(modules)):
        losses = loss_fn()
        if not isinstance(losses, Decision1Losses):
            raise ValueError("loss_fn must return Decision1Losses")
        total = losses.total()
    torch.autograd.backward(scaler.scale(total / ACCUMULATION_STEPS))
    _finite_gradients(parameters)
    stepped = accumulation_index == ACCUMULATION_STEPS - 1
    clipped_norm = 0.0
    applied_learning_rate: float | None = None
    if stepped:
        scaler.unscale_(optimizer)
        _finite_gradients(parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        if not bool(torch.isfinite(norm).item()):
            raise ValueError("training produced a nonfinite gradient norm")
        clipped_norm = min(float(norm), GRADIENT_CLIP_NORM)
        applied_learning_rate = float(optimizer.param_groups[0]["lr"])
        if not math.isfinite(applied_learning_rate):
            raise ValueError("optimizer learning rate must be finite")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return TrainStepResult(
        losses=losses.logged(),
        optimizer_stepped=stepped,
        clipped_gradient_norm=clipped_norm,
        applied_learning_rate=applied_learning_rate,
    )


def _validate_labels(
    labels: Sequence[InfluenceRecord], fingerprints: CheckpointFingerprints
) -> tuple[InfluenceRecord, ...]:
    try:
        records = validate_completed_influence_records(labels)
    except ValueError as error:
        raise ValueError(f"training labels are invalid: {error}") from error
    for record in records:
        if (
            record.config_fingerprint != fingerprints.config
            or record.runtime_fingerprint != fingerprints.runtime
            or record.source_fingerprint != fingerprints.source
            or record.parent_manifest_sha256 != fingerprints.data
            or record.teacher_artifact_sha256 != fingerprints.teacher
        ):
            raise ValueError("training label fingerprints do not match the locked run")
    return records


def _validate_metric_schema(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("metrics JSONL rows must be plain objects")
    row = cast(dict[str, object], value)
    if (
        set(row) != _METRIC_KEYS
        or any(type(row[name]) is not float for name in _METRIC_FLOAT_KEYS)
        or any(type(row[name]) is not int for name in _METRIC_INT_KEYS)
        or type(row["product_fingerprint"]) is not str
        or type(row["teacher_fingerprint"]) is not str
    ):
        raise ValueError("metrics JSONL rows must match the exact primitive schema")
    if any(not math.isfinite(cast(float, row[name])) for name in _METRIC_FLOAT_KEYS):
        raise ValueError("metrics JSONL values must be finite")
    return row


def _strict_existing_metrics(
    path: Path,
    cursor: TrainingCursor,
    *,
    group_count: int,
    fingerprints: CheckpointFingerprints,
) -> None:
    if not path.exists():
        if cursor.optimizer_step:
            raise ValueError("metrics JSONL is missing for the resume cursor")
        return
    if not path.is_file():
        raise ValueError("metrics path must be a regular file")
    lines = path.read_text().splitlines()
    if len(lines) != cursor.optimizer_step:
        raise ValueError("metrics JSONL must end exactly at the resume cursor")
    last_data_index: int | None = None
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"metrics JSONL has a blank row at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"metrics JSONL has invalid JSON at line {line_number}") from error
        row = _validate_metric_schema(row)
        step = row.get("optimizer_step")
        if type(step) is not int or step != line_number:
            raise ValueError("metrics optimizer steps must be contiguous from one")
        if row["micro_step"] != step * ACCUMULATION_STEPS:
            raise ValueError("metrics micro-step does not match the optimizer step")
        data_index = row["data_index"]
        if type(data_index) is not int or not 0 <= data_index < group_count:
            raise ValueError("metrics data index is outside the validated label groups")
        if row["module_seed"] != MODULE_SEED:
            raise ValueError("metrics module seed does not match the locked seed")
        if (
            row["product_fingerprint"] != fingerprints.product
            or row["teacher_fingerprint"] != fingerprints.teacher
        ):
            raise ValueError("metrics fingerprints do not match the locked run")
        last_data_index = data_index
    if cursor.optimizer_step and last_data_index != cursor.data_index:
        raise ValueError("metrics JSONL data position does not match the resume cursor")


def _append_metric(path: Path, row: dict[str, object]) -> None:
    _validate_metric_schema(row)
    encoded = json.dumps(row, allow_nan=False, separators=(",", ":"), sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def train(
    *,
    config: Decision1Config,
    modules: TrainingModules,
    foundation: nn.Module | None,
    foundation_digest_provider: FoundationDigestProvider | None = None,
    labels: Sequence[InfluenceRecord],
    loss_fn: LossFunction,
    fingerprints: CheckpointFingerprints,
    checkpoint_dir: Path,
    metrics_path: Path,
    resume_from: Path | None = None,
    final_checkpoint_path: Path | None = None,
) -> TrainingCursor:
    """Train to the locked production limit or the explicit two-step smoke limit."""
    if not isinstance(config, Decision1Config):
        raise ValueError("config must be a Decision1Config")
    config.validate()
    optimizer_steps = config.training.optimizer_steps
    if fingerprints.config != config.fingerprint():
        raise ValueError("checkpoint config fingerprint does not match the validated config")
    if final_checkpoint_path is not None and not isinstance(final_checkpoint_path, Path):
        raise ValueError("final_checkpoint_path must be a pathlib.Path")
    fingerprints.validate()
    validated_labels = _validate_labels(labels, fingerprints)
    label_groups = tuple(
        validated_labels[start : start + 8] for start in range(0, len(validated_labels), 8)
    )
    if (foundation is None) == (foundation_digest_provider is None):
        raise ValueError("training requires exactly one loaded foundation or digest provider")
    frozen_digest = (
        freeze_foundation(foundation)
        if foundation is not None
        else cast(FoundationDigestProvider, foundation_digest_provider)()
    )
    if frozen_digest != fingerprints.product:
        raise ValueError("frozen product digest does not match the locked fingerprint")
    optimizer = build_optimizer(modules)
    scheduler = build_scheduler(optimizer)
    _module_device(modules)
    if foundation is not None:
        _assert_foundation_frozen(foundation)
        _assert_foundation_excluded(foundation, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    cursor = TrainingCursor(0, 0, 0)
    if resume_from is not None:
        cursor = load_bridge_checkpoint(
            resume_from,
            modules=modules,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_fingerprints=fingerprints,
        )
    if cursor.optimizer_step > optimizer_steps:
        raise ValueError("resume cursor exceeds the requested optimizer steps")
    if cursor.data_index >= len(label_groups):
        raise ValueError("resume data index is outside the validated label groups")
    _strict_existing_metrics(
        metrics_path,
        cursor,
        group_count=len(label_groups),
        fingerprints=fingerprints,
    )

    while cursor.optimizer_step < optimizer_steps:
        component_sums = dict.fromkeys((*Decision1Losses.component_names(), "total"), 0.0)
        gradient_norm = 0.0
        applied_learning_rate: float | None = None
        next_data_index = cursor.data_index
        for accumulation_index in range(ACCUMULATION_STEPS):
            micro_step = cursor.micro_step + accumulation_index
            data_index = next_data_index
            group = label_groups[data_index]
            result = train_step(
                partial(loss_fn, micro_step, data_index, group, modules, foundation),
                modules=modules,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                accumulation_index=accumulation_index,
            )
            next_data_index = (data_index + 1) % len(label_groups)
            for name, value in result.losses.items():
                component_sums[name] += value
            gradient_norm = result.clipped_gradient_norm
            if result.applied_learning_rate is not None:
                applied_learning_rate = result.applied_learning_rate
        if applied_learning_rate is None:
            raise ValueError("optimizer did not step after eight accumulated micro-steps")
        next_step = cursor.optimizer_step + 1
        cursor = TrainingCursor(
            optimizer_step=next_step,
            micro_step=cursor.micro_step + ACCUMULATION_STEPS,
            data_index=next_data_index,
        ).validate()
        row: dict[str, object] = {
            name: value / ACCUMULATION_STEPS for name, value in component_sums.items()
        }
        row.update(
            {
                "optimizer_step": cursor.optimizer_step,
                "micro_step": cursor.micro_step,
                "data_index": cursor.data_index,
                "learning_rate": applied_learning_rate,
                "gradient_norm": gradient_norm,
                "module_seed": MODULE_SEED,
                "product_fingerprint": fingerprints.product,
                "teacher_fingerprint": fingerprints.teacher,
            }
        )
        _append_metric(metrics_path, row)
        if cursor.optimizer_step % CHECKPOINT_INTERVAL == 0:
            if foundation is not None:
                _assert_foundation_frozen(foundation)
                _assert_foundation_excluded(foundation, optimizer)
            elif cast(FoundationDigestProvider, foundation_digest_provider)() != frozen_digest:
                raise ValueError("external foundation/product digest changed during training")
            save_bridge_checkpoint(
                checkpoint_dir / f"step-{cursor.optimizer_step:06d}.pt",
                modules=modules,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                cursor=cursor,
                fingerprints=fingerprints,
            )
    if foundation is not None:
        _assert_foundation_frozen(foundation)
        _assert_foundation_excluded(foundation, optimizer)
        final_digest = foundation_digest(foundation)
    else:
        final_digest = cast(FoundationDigestProvider, foundation_digest_provider)()
    if final_digest != frozen_digest:
        raise ValueError("frozen foundation/product digest changed during training")
    if final_checkpoint_path is not None:
        save_bridge_checkpoint(
            final_checkpoint_path,
            modules=modules,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cursor=cursor,
            fingerprints=fingerprints,
        )
    return cursor


__all__ = (
    "ACCUMULATION_STEPS",
    "CHECKPOINT_INTERVAL",
    "GRADIENT_CLIP_NORM",
    "MODULE_SEED",
    "TOTAL_OPTIMIZER_STEPS",
    "WARMUP_STEPS",
    "TrainStepResult",
    "build_optimizer",
    "build_scheduler",
    "build_training_modules",
    "foundation_digest",
    "freeze_foundation",
    "train",
    "train_step",
)
