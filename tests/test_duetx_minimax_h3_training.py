from __future__ import annotations

import torch

from duet.duetx.checkpoint import TrainingModules
from duet.duetx.config import Method
from duet.duetx.losses import Decision1Losses, LocalExceptionScorer
from duet.duetx.methods import _CentralizedResamplerMethod, _GatedCoreMethod, build_method
from duet.duetx.minimax_h3_bridge import MiniMaxH3LatentHistoryBridge
from duet.duetx.minimax_h3_training import (
    MiniMaxH3TrainingProtocol,
    build_minimax_h3_training_modules,
)
from duet.duetx.training import ACCUMULATION_STEPS, build_optimizer, build_scheduler, train_step


def test_minimax_training_factory_uses_only_native_24_channel_modules() -> None:
    modules = build_minimax_h3_training_modules().validate()

    assert isinstance(modules, TrainingModules)
    assert isinstance(modules.bridge, MiniMaxH3LatentHistoryBridge)
    assert modules.bridge.fusion.channels == 24
    assert isinstance(modules.gated, _GatedCoreMethod)
    assert isinstance(modules.gated.gate[0], torch.nn.LayerNorm)
    assert isinstance(modules.resampler, _CentralizedResamplerMethod)
    assert modules.gated.gate[0].normalized_shape == (24,)
    assert modules.resampler.key.in_features == 24
    assert modules.resampler.output.out_features == 24
    assert isinstance(modules.scorer, LocalExceptionScorer)
    assert modules.scorer.item_projection.in_features == 24


def test_minimax_training_factory_is_deterministic_without_changing_global_rng() -> None:
    torch.manual_seed(901)
    expected_next = torch.rand(3)
    torch.manual_seed(901)

    first = build_minimax_h3_training_modules()
    observed_next = torch.rand(3)
    second = build_minimax_h3_training_modules()

    torch.testing.assert_close(observed_next, expected_next)
    for name in first.named():
        first_state = first.named()[name].state_dict()
        second_state = second.named()[name].state_dict()
        assert tuple(first_state) == tuple(second_state)
        for tensor_name in first_state:
            assert torch.equal(first_state[tensor_name], second_state[tensor_name])


def test_minimax_trainable_cores_accept_native_latents() -> None:
    history = torch.randn(1, 6, 24, 1, 2, 2)

    gated = build_method(Method.GATED_CORE_TOPK, channels=24).compress(history)
    resampled = build_method(Method.CENTRALIZED_RESAMPLER_TOPK, channels=24).compress(history)
    duet = build_method(Method.DUET_CORE_TOPK, channels=24).compress(history)

    assert gated is not None
    assert gated.shape == (1, 24, 1, 2, 2)
    assert resampled is not None
    assert resampled.shape == (1, 24, 1, 2, 2)
    assert duet is not None
    assert duet.shape == (1, 24, 1, 2, 2)


def test_minimax_training_protocol_locks_foundations_and_memory_budget() -> None:
    protocol = MiniMaxH3TrainingProtocol.default()

    assert protocol.validate() is protocol
    assert protocol.format == "duet-x-minimax-h3-training-protocol-v1"
    assert protocol.history_items == 8
    assert protocol.protected_exceptions == 2
    assert protocol.latent_channels == 24
    assert protocol.operator_size == 16
    assert protocol.optimizer_steps == 2_000
    assert protocol.trainable_modules == ("bridge", "gated", "resampler", "scorer")
    assert protocol.frozen_modules == (
        "minimax_h3",
        "qwen3_vl",
        "video_vae",
        "audio_vae",
    )
    assert len(protocol.fingerprint()) == 64


def test_minimax_training_modules_complete_one_real_optimizer_step() -> None:
    modules = build_minimax_h3_training_modules()
    optimizer = build_optimizer(modules)
    scheduler = build_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    generator = torch.Generator().manual_seed(409)
    history = torch.randn((1, 8, 24, 1, 2, 2), generator=generator)
    before = {
        module_name: {
            name: parameter.detach().clone() for name, parameter in module.named_parameters()
        }
        for module_name, module in modules.named().items()
    }

    def loss() -> Decision1Losses:
        bridge = modules.bridge(history, anchor=history[:, -1]).core_latent
        assert isinstance(modules.gated, _GatedCoreMethod)
        assert isinstance(modules.resampler, _CentralizedResamplerMethod)
        gated = modules.gated.compress(history)
        resampled = modules.resampler.compress(history)
        assert gated is not None
        assert resampled is not None
        scores = modules.scorer(
            history[:, 0],
            event_types=torch.zeros(1, dtype=torch.int64),
            timestamps=torch.zeros(1),
        )
        prediction = bridge.square().mean() + gated.square().mean()
        prediction = prediction + resampled.square().mean() + scores.square().mean()
        zero = prediction * 0.0
        return Decision1Losses(prediction, zero, zero, zero, zero, zero)

    result = None
    for accumulation_index in range(ACCUMULATION_STEPS):
        result = train_step(
            loss,
            modules=modules,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            accumulation_index=accumulation_index,
        )

    assert result is not None
    assert result.optimizer_stepped
    assert all(
        any(
            not torch.equal(before[module_name][name], parameter)
            for name, parameter in module.named_parameters()
        )
        for module_name, module in modules.named().items()
    )
