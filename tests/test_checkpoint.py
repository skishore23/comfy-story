from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from comfy_story.checkpoint import (
    CheckpointFingerprints,
    TrainingCursor,
    TrainingModules,
    load_bridge_checkpoint,
    save_bridge_checkpoint,
)


def _modules(width: int = 3) -> TrainingModules:
    return TrainingModules(
        bridge=nn.Linear(width, width),
        gated=nn.Linear(width, width),
        resampler=nn.Linear(width, width),
        scorer=nn.Linear(width, 1),
    )


def _fingerprints() -> CheckpointFingerprints:
    return CheckpointFingerprints(
        config="1" * 64,
        runtime="2" * 64,
        source="3" * 64,
        data="4" * 64,
        teacher="5" * 64,
        product="6" * 64,
    )


def _state(
    modules: TrainingModules,
    *,
    steps: int = 17,
) -> tuple[
    torch.optim.AdamW,
    torch.optim.lr_scheduler.LambdaLR,
    torch.cuda.amp.GradScaler,
]:
    optimizer = torch.optim.AdamW(modules.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    for _ in range(steps):
        loss = torch.stack(
            tuple(parameter.square().sum() for parameter in modules.parameters())
        ).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return optimizer, scheduler, scaler


def _save(path: Path, modules: TrainingModules) -> None:
    optimizer, scheduler, scaler = _state(modules)
    save_bridge_checkpoint(
        path,
        modules=modules,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        cursor=TrainingCursor(optimizer_step=17, micro_step=136, data_index=9),
        fingerprints=_fingerprints(),
    )


def test_checkpoint_round_trip_restores_exact_training_and_rng_state(tmp_path: Path) -> None:
    torch.manual_seed(41)
    random.seed(43)
    modules = _modules()
    optimizer, scheduler, scaler = _state(modules)
    path = tmp_path / "bridge.pt"
    save_bridge_checkpoint(
        path,
        modules=modules,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        cursor=TrainingCursor(optimizer_step=17, micro_step=136, data_index=9),
        fingerprints=_fingerprints(),
    )
    expected_tensor = torch.rand(3)
    expected_python = random.random()
    saved_state = {
        name: {key: value.detach().clone() for key, value in module.state_dict().items()}
        for name, module in modules.named().items()
    }

    with torch.no_grad():
        for parameter in modules.parameters():
            parameter.add_(100)
    torch.manual_seed(999)
    random.seed(999)

    cursor = load_bridge_checkpoint(
        path,
        modules=modules,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_fingerprints=_fingerprints(),
    )

    assert cursor == TrainingCursor(optimizer_step=17, micro_step=136, data_index=9)
    for name, module in modules.named().items():
        for key, value in module.state_dict().items():
            assert torch.equal(value, saved_state[name][key])
    assert torch.equal(torch.rand(3), expected_tensor)
    assert random.random() == expected_python


def test_checkpoint_contains_only_locked_modules_and_training_state(tmp_path: Path) -> None:
    path = tmp_path / "bridge.pt"
    _save(path, _modules())

    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert set(payload) == {
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
    assert payload["format"] == "duet-x-bridge-checkpoint-v2"
    assert set(payload["modules"]) == {"bridge", "gated", "resampler", "scorer"}
    assert set(payload["fingerprints"]) == {
        "config",
        "runtime",
        "source",
        "data",
        "teacher",
        "product",
    }
    assert not any("foundation" in key or "ltx" in key for key in payload)
    assert not tuple(tmp_path.glob(".bridge.pt.*"))


@pytest.mark.parametrize("target", ["module", "optimizer"])
def test_checkpoint_content_hashes_reject_finite_state_tampering(
    tmp_path: Path, target: str
) -> None:
    path = tmp_path / "bridge.pt"
    _save(path, _modules())
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if target == "module":
        first_weight = next(iter(payload["modules"]["bridge"].values()))
        first_weight.add_(1.0)
    else:
        first_state = next(iter(payload["optimizer"]["state"].values()))
        first_state["exp_avg"].add_(1.0)
    torch.save(payload, path)
    destination = _modules()
    optimizer, scheduler, scaler = _state(destination)

    with pytest.raises(ValueError, match=r"hash|manifest"):
        load_bridge_checkpoint(
            path,
            modules=destination,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_fingerprints=_fingerprints(),
        )


@pytest.mark.parametrize(
    "mutation",
    ["unknown_group_key", "missing_state", "extra_state", "missing_moment"],
)
def test_strict_load_rejects_nested_optimizer_schema_before_apply(
    tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / "bridge.pt"
    _save(path, _modules())
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["optimizer"]["state"]
    if mutation == "unknown_group_key":
        payload["optimizer"]["param_groups"][0]["unexpected"] = True
    elif mutation == "missing_state":
        del state[next(iter(state))]
    elif mutation == "extra_state":
        state[99_999] = next(iter(state.values()))
    else:
        del next(iter(state.values()))["exp_avg"]
    torch.save(payload, path)
    destination = _modules()
    optimizer, scheduler, scaler = _state(destination)

    with pytest.raises(ValueError, match=r"optimizer|parameter|state|group"):
        load_bridge_checkpoint(
            path,
            modules=destination,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_fingerprints=_fingerprints(),
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "module_extra", "state_extra"])
def test_strict_load_rejects_missing_or_extra_keys(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "bridge.pt"
    source_modules = _modules()
    _save(path, source_modules)
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if mutation == "missing":
        del payload["rng_states"]
    elif mutation == "extra":
        payload["foundation"] = {}
    elif mutation == "module_extra":
        payload["modules"]["foundation"] = {}
    else:
        payload["modules"]["bridge"]["unknown"] = torch.zeros(1)
    torch.save(payload, path)
    destination = _modules()
    optimizer, scheduler, scaler = _state(destination)

    with pytest.raises(ValueError, match=r"fields|modules|state"):
        load_bridge_checkpoint(
            path,
            modules=destination,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_fingerprints=_fingerprints(),
        )


def test_strict_load_rejects_fingerprint_and_parameter_count_mismatches(tmp_path: Path) -> None:
    path = tmp_path / "bridge.pt"
    _save(path, _modules())
    destination = _modules()
    optimizer, scheduler, scaler = _state(destination)

    with pytest.raises(ValueError, match="fingerprint"):
        load_bridge_checkpoint(
            path,
            modules=destination,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_fingerprints=replace(_fingerprints(), teacher="a" * 64),
        )

    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    payload["parameter_counts"]["scorer"] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="parameter count"):
        load_bridge_checkpoint(
            path,
            modules=destination,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_fingerprints=_fingerprints(),
        )


def test_strict_load_rejects_nonfinite_optimizer_state(tmp_path: Path) -> None:
    path = tmp_path / "bridge.pt"
    _save(path, _modules())
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    first_state = next(iter(payload["optimizer"]["state"].values()))
    first_state["exp_avg"].fill_(float("nan"))
    torch.save(payload, path)
    destination = _modules()
    optimizer, scheduler, scaler = _state(destination)

    with pytest.raises(ValueError, match="finite"):
        load_bridge_checkpoint(
            path,
            modules=destination,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_fingerprints=_fingerprints(),
        )


def test_save_rejects_scheduler_state_that_disagrees_with_exact_cursor(tmp_path: Path) -> None:
    modules = _modules()
    optimizer, scheduler, scaler = _state(modules, steps=1)

    with pytest.raises(ValueError, match="cursor"):
        save_bridge_checkpoint(
            tmp_path / "bridge.pt",
            modules=modules,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cursor=TrainingCursor(optimizer_step=17, micro_step=136, data_index=9),
            fingerprints=_fingerprints(),
        )
