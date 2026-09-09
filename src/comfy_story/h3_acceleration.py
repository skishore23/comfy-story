"""Bounded H3 block reuse on native Comfy hooks, with sampling-local state.

Algorithm references: mrfakename/minimax-h3-ultra-fast at 5620192, and
Cache-DiT F1B0/TaylorSeer. This implementation keeps Comfy's conditioning,
packing, model weights, attention, and output heads in charge.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch

from comfy_story.h3_sol_attention import H3SolAttention, load_sol_kernel

_LOG = logging.getLogger(__name__)
NVFP4_MODEL = "minimax_h3_ref2va_pruned_nvfp4.safetensors"
CACHE_CONFIGURATION = {
    "format": "duet-h3-balanced-cache-v1",
    "threshold": 0.08,
    "dense_start": 3,
    "dense_end": 2,
    "max_cached": 2,
    "probe_stride": 8,
    "tail_forecast_order": 1,
    "attention": "native-comfy-dense",
}


@dataclass
class H3BlockCache:
    """One Euler trajectory; never retained across shots or sampling attempts."""

    total_steps: int
    step: int = 0
    consecutive: int = 0
    reused: int = 0
    probe: torch.Tensor | None = None
    head: torch.Tensor | None = None
    tail: torch.Tensor | None = None
    slope: torch.Tensor | None = None
    anchor_step: int = -1
    skip: bool = False
    layout: object = None

    def first(self, before: torch.Tensor, after: torch.Tensor, layout: object) -> None:
        if self.step >= self.total_steps:
            raise ValueError("H3 Balanced requires one model evaluation per Euler step")
        if self.layout is not None and layout != self.layout:
            raise ValueError("H3 conditioning layout changed within a cached trajectory")
        self.layout = layout
        current = after[::8].float() - before.float()
        eligible = (
            3 <= self.step < self.total_steps - 2
            and self.consecutive < 2
            and self.probe is not None
            and self.tail is not None
            and self.tail.shape == after.shape
        )
        self.skip = False
        if eligible and self.probe is not None:
            relative = (current - self.probe).abs().mean() / self.probe.abs().mean().clamp_min(1e-8)
            self.skip = bool(torch.isfinite(relative).item() and relative.item() <= 0.08)
        if not self.skip:
            self.probe = current.detach()
            self.head = after.detach().clone()

    def finish(self, value: torch.Tensor) -> torch.Tensor:
        if self.skip:
            if self.tail is None:
                raise RuntimeError("H3 cache has no dense anchor")
            value = value + self.tail
            if self.slope is not None:
                value.add_(self.slope, alpha=self.step - self.anchor_step)
            self.reused += 1
            self.consecutive += 1
        else:
            if self.head is None:
                raise RuntimeError("H3 cache did not execute its first block")
            tail = value.detach() - self.head
            self.slope = (
                None if self.tail is None else (tail - self.tail) / (self.step - self.anchor_step)
            )
            self.tail = tail
            self.anchor_step = self.step
            self.consecutive = 0
        self.head = None
        self.step += 1
        return value


def patch_h3_balanced(model: Any, *, sol_attention: bool = False) -> Any:
    """Clone an H3 patcher and install bounded per-request block reuse.

    The hooks operate on the entire packed text/reference/audio/video stream.
    Existing block replacements are rejected rather than silently overwritten.
    """
    diffusion = model.get_model_object("diffusion_model")
    if type(diffusion).__name__ != "MiniMaxH3Model" or len(diffusion.blocks) != 50:
        raise ValueError("H3 Balanced requires native MiniMaxH3Model with 50 blocks")
    if model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit"):
        raise ValueError("H3 Balanced cannot overwrite existing DiT block patches")
    if sol_attention and model.model_options.get("transformer_options", {}).get(
        "optimized_attention_override"
    ):
        raise ValueError("H3 Ultra Fast cannot overwrite an existing attention override")
    kernel = load_sol_kernel() if sol_attention else None
    sol_state: ContextVar[H3SolAttention | None] = ContextVar("duet_h3_sol", default=None)
    patched = model.clone()
    state: ContextVar[H3BlockCache | None] = ContextVar("duet_h3_cache", default=None)

    def outer(executor: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        sigmas = kwargs.get("sigmas", args[3] if len(args) > 3 else None)
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 7:
            raise ValueError("H3 Balanced requires a complete Euler sigma schedule")
        if not bool(torch.isfinite(sigmas).all()) or not bool((sigmas[:-1] > sigmas[1:]).all()):
            raise ValueError("H3 Balanced sigmas must be finite and strictly decreasing")
        cache = H3BlockCache(len(sigmas) - 1)
        token = state.set(cache)
        sol = H3SolAttention(kernel) if kernel is not None else None
        sol_token = sol_state.set(sol)
        try:
            result = executor(*args, **kwargs)
            if cache.step != cache.total_steps:
                raise ValueError("H3 Balanced did not observe exactly one H3 call per Euler step")
            _LOG.info(
                "Duet H3 Balanced: %d dense, %d cached steps",
                cache.step - cache.reused,
                cache.reused,
            )
            if sol is not None:
                _LOG.info(
                    "Duet H3 Sol-Attn: %d sparse, %d dense calls", sol.sparse_calls, sol.dense_calls
                )
            return result
        finally:
            sol_state.reset(sol_token)
            state.reset(token)

    def block_patch(index: int) -> Callable[..., Any]:
        def run(args: dict[str, Any], extra: dict[str, Any]) -> dict[str, torch.Tensor]:
            cache = state.get()
            if cache is None:
                raise ValueError("H3 Balanced block executed outside its sampling context")
            sol = sol_state.get()
            if sol is not None:
                sol.observe(args["layout"], cache.step, index)
            value = args["img"]
            if index == 0:
                before = value[::8].detach().clone()
                value = extra["original_block"](args)["img"]
                layout = (tuple(value.shape), tuple(args["layout"].segments))
                cache.first(before, value, layout)
            elif not cache.skip:
                value = extra["original_block"](args)["img"]
            if index == 49:
                value = cache.finish(value)
                if sol is not None:
                    sol.layer = -1
            return {"img": value}

        return run

    if sol_attention:

        def attention(original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            sol = sol_state.get()
            if sol is None:
                raise ValueError("H3 Sol-Attn called outside its sampling context")
            return sol(original, *args, **kwargs)

        patched.model_options.setdefault("transformer_options", {})[
            "optimized_attention_override"
        ] = attention

    for index in range(50):
        patched.set_model_patch_replace(block_patch(index), "dit", "double_block", index)
    patched.add_wrapper_with_key("outer_sample", "duet_h3_balanced", outer)
    return patched
