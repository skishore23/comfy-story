from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from duet.duetx.checkpoint import (
    CheckpointFingerprints,
    TrainingCursor,
    TrainingModules,
    save_bridge_checkpoint,
)
from duet.duetx.config import Decision1Config, load_config
from duet.duetx.losses import Decision1Losses
from duet.duetx.ltx_bridge import LTXLatentHistoryBridge
from duet.duetx.teacher import InfluenceRecord
from duet.duetx.training import (
    ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    MODULE_SEED,
    build_optimizer,
    build_scheduler,
    build_training_modules,
    foundation_digest,
    freeze_foundation,
    train,
    train_step,
)

_PRODUCTION_CONFIG = load_config(Path("configs/duetx/decision1.yaml"))


def _config_fingerprint(config: Decision1Config = _PRODUCTION_CONFIG) -> str:
    return config.fingerprint()


def _modules() -> TrainingModules:
    torch.manual_seed(1)
    return TrainingModules(
        bridge=nn.Linear(2, 2, bias=False),
        gated=nn.Linear(2, 2, bias=False),
        resampler=nn.Linear(2, 2, bias=False),
        scorer=nn.Linear(2, 1, bias=False),
    )


def _losses(modules: TrainingModules, scale: float = 1.0) -> Decision1Losses:
    features = torch.ones(1, 2)
    prediction = (
        modules.bridge(features).sum()
        + modules.gated(features).sum()
        + modules.resampler(features).sum()
        + modules.scorer(features).sum()
    ) * scale
    zero = prediction * 0.0
    return Decision1Losses(prediction.square(), zero, zero, zero, zero, zero)


def _fingerprints(foundation: nn.Module) -> CheckpointFingerprints:
    return CheckpointFingerprints(
        config=_config_fingerprint(),
        runtime="2" * 64,
        source="3" * 64,
        data="4" * 64,
        teacher="5" * 64,
        product=foundation_digest(foundation),
    )


def _labels(
    sample_offset: int = 0,
    *,
    split: str = "train",
    coordinate_seed: int = 7,
) -> tuple[InfluenceRecord, ...]:
    return tuple(
        InfluenceRecord(
            sample_id=f"sample-{sample_offset}",
            split=split,  # type: ignore[arg-type]
            item_index=index,
            item_id=f"item-{sample_offset}-{index}",
            item_sha256=f"{100 + sample_offset * 20 + index:064x}",
            source_sha256=f"{200 + sample_offset * 20 + index:064x}",
            media_sha256=f"{300 + sample_offset * 20 + index:064x}",
            velocity=0.0,
            hidden=0.0,
            marker=0.0,
            structural=0.0,
            score_q=0,
            teacher_artifact_sha256="5" * 64,
            coordinate_seed=coordinate_seed,
            timesteps=(0.2, 0.5, 0.8),
            config_fingerprint=_config_fingerprint(),
            source_fingerprint="3" * 64,
            runtime_fingerprint="2" * 64,
            parent_manifest_sha256="4" * 64,
        )
        for index in range(8)
    )


def _label_groups(count: int = 5) -> tuple[InfluenceRecord, ...]:
    return tuple(record for sample in range(count) for record in _labels(sample))


def _metric_row(
    step: int, data_index: int, fingerprints: CheckpointFingerprints
) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys((*Decision1Losses.component_names(), "total"), 0.0)
    row.update(
        {
            "optimizer_step": step,
            "micro_step": step * ACCUMULATION_STEPS,
            "data_index": data_index,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "gradient_norm": 0.0,
            "module_seed": MODULE_SEED,
            "product_fingerprint": fingerprints.product,
            "teacher_fingerprint": fingerprints.teacher,
        }
    )
    return row


def _write_metrics_to_cursor(
    path: Path,
    *,
    cursor: TrainingCursor,
    fingerprints: CheckpointFingerprints,
    group_count: int,
) -> None:
    first_index = (
        cursor.data_index - (cursor.optimizer_step - 1) * ACCUMULATION_STEPS
    ) % group_count
    rows = [
        _metric_row(
            step,
            (first_index + (step - 1) * ACCUMULATION_STEPS) % group_count,
            fingerprints,
        )
        for step in range(1, cursor.optimizer_step + 1)
    ]
    path.write_text("".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows))


def _save_resume(
    path: Path,
    *,
    modules: TrainingModules,
    fingerprints: CheckpointFingerprints,
    cursor: TrainingCursor,
) -> None:
    optimizer = build_optimizer(modules)
    scheduler = build_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    for _ in range(cursor.optimizer_step):
        for parameter in modules.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    save_bridge_checkpoint(
        path,
        modules=modules,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        cursor=cursor,
        fingerprints=fingerprints,
    )


@pytest.mark.parametrize("mutation", ["duplicate", "mixed_split", "mixed_coordinate"])
def test_train_reuses_canonical_completed_label_validation(tmp_path: Path, mutation: str) -> None:
    records = _labels()
    if mutation == "duplicate":
        invalid = (*records, *records)
    elif mutation == "mixed_split":
        invalid = (*records[:4], replace(records[4], split="validation"), *records[5:])
    else:
        invalid = (*records[:4], replace(records[4], coordinate_seed=99), *records[5:])
    foundation = nn.Linear(2, 2)

    def should_not_run(
        micro_step: int,
        data_index: int,
        group: tuple[InfluenceRecord, ...],
        active_modules: TrainingModules,
        frozen: nn.Module | None,
    ) -> Decision1Losses:
        del micro_step, data_index, group, active_modules, frozen
        raise AssertionError("invalid labels reached the loss function")

    with pytest.raises(ValueError, match=r"duplicate|completed|mix|fingerprint|coordinate"):
        train(
            config=_PRODUCTION_CONFIG,
            modules=_modules(),
            foundation=foundation,
            labels=invalid,
            loss_fn=should_not_run,
            fingerprints=_fingerprints(foundation),
            checkpoint_dir=tmp_path / "checkpoints",
            metrics_path=tmp_path / "metrics.jsonl",
        )


def test_train_step_freezes_foundation_accumulates_eight_and_clips_gradients() -> None:
    foundation = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    digest = freeze_foundation(foundation)
    modules = _modules()
    optimizer = build_optimizer(modules)
    scheduler = build_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    assert isinstance(modules.bridge, nn.Linear)
    before = modules.bridge.weight.detach().clone()
    result = None

    for accumulation_index in range(ACCUMULATION_STEPS):
        result = train_step(
            lambda: _losses(modules, scale=1000.0),
            modules=modules,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            accumulation_index=accumulation_index,
        )
        if accumulation_index < ACCUMULATION_STEPS - 1:
            assert not result.optimizer_stepped
            assert torch.equal(modules.bridge.weight, before)

    assert result is not None
    assert result.optimizer_stepped
    assert not torch.equal(modules.bridge.weight, before)
    assert result.clipped_gradient_norm <= 1.0
    assert all(not parameter.requires_grad for parameter in foundation.parameters())
    assert all(
        id(parameter)
        not in {id(item) for group in optimizer.param_groups for item in group["params"]}
        for parameter in foundation.parameters()
    )
    assert foundation_digest(foundation) == digest
    assert result.applied_learning_rate == pytest.approx(DEFAULT_LEARNING_RATE / 200)


def test_train_accepts_actual_external_foundation_digest_guard(tmp_path: Path) -> None:
    config = replace(
        _PRODUCTION_CONFIG,
        training=replace(_PRODUCTION_CONFIG.training, optimizer_steps=2, synthetic_smoke=True),
    )
    product_sha = "7" * 64
    fingerprints = CheckpointFingerprints(
        config=config.fingerprint(),
        runtime="2" * 64,
        source="3" * 64,
        data="4" * 64,
        teacher="5" * 64,
        product=product_sha,
    )
    modules = _modules()
    calls = 0

    def guard() -> str:
        nonlocal calls
        calls += 1
        return product_sha

    train(
        config=config,
        modules=modules,
        foundation=None,
        foundation_digest_provider=guard,
        labels=tuple(
            replace(record, config_fingerprint=config.fingerprint()) for record in _label_groups(1)
        ),
        loss_fn=lambda _micro, _data, _group, active, _foundation: _losses(active),
        fingerprints=fingerprints,
        checkpoint_dir=tmp_path / "checkpoints",
        metrics_path=tmp_path / "metrics.jsonl",
        final_checkpoint_path=tmp_path / "final.pt",
    )

    assert calls >= 2


def test_scheduler_has_exact_200_optimizer_step_warmup() -> None:
    modules = _modules()
    optimizer = build_optimizer(modules, learning_rate=1.0)
    scheduler = build_scheduler(optimizer)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1 / 200)
    for _ in range(199):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_locked_module_seed_is_deterministic_without_changing_run_rng() -> None:
    torch.manual_seed(91)
    run_rng = torch.get_rng_state()

    first = build_training_modules()
    actual_next = torch.rand(3)
    torch.set_rng_state(run_rng)
    expected_next = torch.rand(3)
    second = build_training_modules()

    assert torch.equal(actual_next, expected_next)
    for name, module in first.named().items():
        for key, value in module.state_dict().items():
            assert torch.equal(value, second.named()[name].state_dict()[key])


def test_train_step_uses_bfloat16_autocast_where_cpu_supports_it() -> None:
    modules = _modules()
    optimizer = build_optimizer(modules)
    scheduler = build_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    observed: list[torch.dtype] = []

    def loss_fn() -> Decision1Losses:
        prediction = modules.bridge(torch.ones(1, 2))
        observed.append(prediction.dtype)
        value = prediction.float().square().mean()
        zero = value * 0.0
        return Decision1Losses(value, zero, zero, zero, zero, zero)

    train_step(
        loss_fn,
        modules=modules,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        accumulation_index=0,
    )

    assert observed == [torch.bfloat16]


def test_bfloat16_training_autocast_preserves_fp32_dense_accumulation() -> None:
    bridge = LTXLatentHistoryBridge(operator_size=4)
    history = torch.randn(1, 3, 128, 1, 1, 1)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = bridge(history, anchor=history[:, -1])

    assert output.core_operator.dtype == torch.float32


def test_resumable_loop_reaches_2000_and_writes_strict_metrics_and_checkpoint(
    tmp_path: Path,
) -> None:
    modules = _modules()
    foundation = nn.Linear(2, 2)
    fingerprints = _fingerprints(foundation)
    resume = tmp_path / "resume.pt"
    resume_cursor = TrainingCursor(optimizer_step=1999, micro_step=1999 * 8, data_index=3)
    _save_resume(
        resume,
        modules=modules,
        fingerprints=fingerprints,
        cursor=resume_cursor,
    )
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics_to_cursor(
        metrics,
        cursor=resume_cursor,
        fingerprints=fingerprints,
        group_count=5,
    )
    observed: list[tuple[int, int, str]] = []

    def loss_fn(
        micro_step: int,
        data_index: int,
        records: tuple[InfluenceRecord, ...],
        active_modules: TrainingModules,
        frozen: nn.Module | None,
    ) -> Decision1Losses:
        del frozen
        observed.append((micro_step, data_index, records[0].sample_id))
        return _losses(active_modules)

    cursor = train(
        config=_PRODUCTION_CONFIG,
        modules=modules,
        foundation=foundation,
        labels=_label_groups(),
        loss_fn=loss_fn,
        fingerprints=fingerprints,
        checkpoint_dir=tmp_path / "checkpoints",
        metrics_path=metrics,
        resume_from=resume,
    )

    assert cursor == TrainingCursor(optimizer_step=2000, micro_step=16000, data_index=1)
    assert observed == [
        (15992 + offset, (3 + offset) % 5, f"sample-{(3 + offset) % 5}") for offset in range(8)
    ]
    assert (tmp_path / "checkpoints" / "step-002000.pt").is_file()
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert len(rows) == 2000
    assert rows[-1]["optimizer_step"] == 2000
    assert rows[-1]["learning_rate"] == pytest.approx(DEFAULT_LEARNING_RATE)
    assert all(
        value is None or type(value) in {str, int, float, bool} for value in rows[-1].values()
    )
    assert MODULE_SEED == 20_260_828


def test_train_hashes_whole_foundation_only_at_run_boundaries(tmp_path: Path) -> None:
    modules = _modules()
    foundation = nn.Linear(2, 2)
    fingerprints = _fingerprints(foundation)
    cursor = TrainingCursor(optimizer_step=1999, micro_step=15992, data_index=3)
    resume = tmp_path / "resume.pt"
    _save_resume(resume, modules=modules, fingerprints=fingerprints, cursor=cursor)
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics_to_cursor(metrics, cursor=cursor, fingerprints=fingerprints, group_count=5)

    with patch.object(foundation, "state_dict", wraps=foundation.state_dict) as state_dict:
        train(
            config=_PRODUCTION_CONFIG,
            modules=modules,
            foundation=foundation,
            labels=_label_groups(),
            loss_fn=lambda micro, data, records, active, frozen: _losses(active),
            fingerprints=fingerprints,
            checkpoint_dir=tmp_path / "checkpoints",
            metrics_path=metrics,
            resume_from=resume,
        )

    assert state_dict.call_count == 2


@pytest.mark.parametrize(
    "mutation", ["missing_field", "wrong_type", "gap", "behind", "wrong_cursor"]
)
def test_resume_rejects_nonexact_metrics_lineage(tmp_path: Path, mutation: str) -> None:
    modules = _modules()
    foundation = nn.Linear(2, 2)
    fingerprints = _fingerprints(foundation)
    cursor = TrainingCursor(optimizer_step=1999, micro_step=15992, data_index=3)
    resume = tmp_path / "resume.pt"
    _save_resume(resume, modules=modules, fingerprints=fingerprints, cursor=cursor)
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics_to_cursor(metrics, cursor=cursor, fingerprints=fingerprints, group_count=5)
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    if mutation == "missing_field":
        del rows[-1]["velocity"]
    elif mutation == "wrong_type":
        rows[-1]["velocity"] = None
    elif mutation == "gap":
        rows[-1]["optimizer_step"] = 1997
    elif mutation == "behind":
        rows.pop()
    else:
        rows[-1]["data_index"] = 4
    metrics.write_text("".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows))

    with pytest.raises(ValueError, match=r"metrics|schema|cursor|contiguous"):
        train(
            config=_PRODUCTION_CONFIG,
            modules=modules,
            foundation=foundation,
            labels=_label_groups(),
            loss_fn=lambda micro, data, records, active, frozen: _losses(active),
            fingerprints=fingerprints,
            checkpoint_dir=tmp_path / "checkpoints",
            metrics_path=metrics,
            resume_from=resume,
        )


def test_train_fails_closed_on_missing_labels_or_changed_product_digest(tmp_path: Path) -> None:
    modules = _modules()
    foundation = nn.Linear(2, 2)
    fingerprints = _fingerprints(foundation)

    with pytest.raises(ValueError, match="labels"):
        train(
            config=_PRODUCTION_CONFIG,
            modules=modules,
            foundation=foundation,
            labels=(),
            loss_fn=lambda micro, data, records, active, frozen: _losses(active),
            fingerprints=fingerprints,
            checkpoint_dir=tmp_path / "checkpoints",
            metrics_path=tmp_path / "metrics.jsonl",
        )

    with torch.no_grad():
        next(foundation.parameters()).add_(1)
    with pytest.raises(ValueError, match="product"):
        train(
            config=_PRODUCTION_CONFIG,
            modules=modules,
            foundation=foundation,
            labels=_labels(),
            loss_fn=lambda micro, data, records, active, frozen: _losses(active),
            fingerprints=fingerprints,
            checkpoint_dir=tmp_path / "checkpoints",
            metrics_path=tmp_path / "metrics.jsonl",
        )


def test_two_step_smoke_rejects_production_config_identity_before_work_or_write(
    tmp_path: Path,
) -> None:
    smoke = load_config(Path("configs/duetx/synthetic_smoke.yaml"))
    modules = _modules()
    foundation = nn.Linear(2, 2)
    fingerprints = _fingerprints(foundation)
    checkpoint = tmp_path / "checkpoint.pt"
    metrics = tmp_path / "metrics.jsonl"
    called = False

    def forbidden_loss(
        micro_step: int,
        data_index: int,
        records: tuple[InfluenceRecord, ...],
        active_modules: TrainingModules,
        frozen: nn.Module | None,
    ) -> Decision1Losses:
        del micro_step, data_index, records, active_modules, frozen
        nonlocal called
        called = True
        raise AssertionError("identity mismatch reached a training step")

    with pytest.raises(ValueError, match="config fingerprint"):
        train(
            config=smoke,
            modules=modules,
            foundation=foundation,
            labels=_labels(),
            loss_fn=forbidden_loss,
            fingerprints=fingerprints,
            checkpoint_dir=tmp_path / "periodic",
            metrics_path=metrics,
            final_checkpoint_path=checkpoint,
        )

    assert called is False
    assert not checkpoint.exists()
    assert not metrics.exists()
