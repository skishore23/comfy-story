"""Counterfactual checks for an active historical readout, separate from image quality."""

from __future__ import annotations

import torch

from comfy_story.memory.bridge import H3LatentHistoryBridge


@torch.inference_mode()
def history_effect_report(
    bridge: H3LatentHistoryBridge, *, require_effect: bool = False
) -> dict[str, float | int]:
    parameter = next(bridge.parameters())
    generator = torch.Generator(device="cpu").manual_seed(20260909)
    histories = torch.randn((2, 8, 24, 1, 2, 2), generator=generator).to(parameter.device)
    anchor = histories[0:1, -1].clone()
    first = bridge(histories[0:1], anchor=anchor).core_latent
    changed = bridge(histories[1:2], anchor=anchor).core_latent
    reordered = bridge(histories[0:1].flip(1), anchor=anchor).core_latent
    report: dict[str, float | int] = {
        "readout_nonzero": int(torch.count_nonzero(bridge.residual.weight).item()),
        "history_delta_rms": float((first - changed).square().mean().sqrt().item()),
        "order_delta_rms": float((first - reordered).square().mean().sqrt().item()),
        "anchor_delta_rms": float((first - anchor).square().mean().sqrt().item()),
    }
    if not all(bool(torch.isfinite(value).all().item()) for value in (first, changed, reordered)):
        raise ValueError("memory checkpoint produces nonfinite historical guides")
    if require_effect and (
        report["history_delta_rms"] <= 1e-8 or report["order_delta_rms"] <= 1e-8
    ):
        raise ValueError(
            "memory checkpoint has no measurable history/order effect; "
            "install an active, compatible associative memory checkpoint"
        )
    return report
