"""Model-scoped Sol-Attn policy for native H3 packed reference/audio/video streams.

Matches the Space's Ref2VA policy at 5620192: ten dense steps, two dense
layers, and a dense prefix for every non-target-video query and key.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

SOL_CONFIGURATION = {
    "format": "duet-h3-sol-v1",
    "backend": "sol-attn-triton",
    "source_revision": "46031940ba8af5d18054217e571149579424c0b1",
    "tau": 1.0,
    "thresh_type": "diag",
    "dense_steps": 10,
    "dense_layers": 2,
    "min_tokens": 24576,
    "dense_prefix_queries": True,
    "dense_prefix_keys": True,
}


def load_sol_kernel() -> Callable[..., torch.Tensor]:
    """Optional GPU dependency; never silently claim acceleration when unavailable."""
    import importlib

    try:
        module = importlib.import_module("sol_attn.triton_ref")
    except ImportError as error:
        raise RuntimeError(
            "NVFP4 Ultra Fast requires the pinned sol-attn Triton backend"
        ) from error
    kernel: Callable[..., torch.Tensor] = module.sol_attn
    return kernel


@dataclass
class H3SolAttention:
    kernel: Callable[..., torch.Tensor]
    step: int = 0
    layer: int = -1
    video_start: int = 0
    sequence: int = 0
    sparse_calls: int = 0
    dense_calls: int = 0

    def observe(self, layout: Any, step: int, layer: int) -> None:
        segments = tuple(layout.segments)
        targets = [(a, b) for a, b, kind in segments if kind == "video"]
        if len(targets) != 1 or targets[0][1] != layout.seq_len:
            raise ValueError("Sol-Attn requires one contiguous target-video suffix")
        self.video_start, self.sequence = targets[0][0], layout.seq_len
        if not 0 < self.video_start < self.sequence:
            raise ValueError("Sol-Attn requires a nonempty dense conditioning/audio prefix")
        self.step, self.layer = step, layer

    def __call__(
        self,
        original: Callable[..., torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        eligible = self.layer >= 2 and self.step >= 10 and self.sequence >= 24576
        if not eligible:
            self.dense_calls += 1
            return original(q, k, v, heads, **kwargs)
        if (
            kwargs.get("mask") is not None
            or not kwargs.get("skip_reshape")
            or q.ndim != 4
            or q.shape != k.shape
            or q.shape != v.shape
            or q.shape[0] != 1
            or q.shape[1] != heads
            or q.shape[2] != self.sequence
            or q.dtype not in (torch.float16, torch.bfloat16)
            or q.device.type != "cuda"
        ):
            raise ValueError("Sol-Attn requires unmasked single-batch native H3 CUDA attention")
        # Comfy supplies [B,H,S,D]; Sol takes contiguous [B,S,H,D].
        query, key, value = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        attended = self.kernel(
            query,
            key,
            value,
            tau=1.0,
            thresh_type="diag",
            sink_start=0,
            sink_tokens=self.video_start,
        )
        # KV sinks alone do not protect audio/text/reference QUERY rows.
        prefix = F.scaled_dot_product_attention(
            q[:, :, : self.video_start], k, v, dropout_p=0.0, is_causal=False
        )
        attended[:, : self.video_start] = prefix.transpose(1, 2)
        self.sparse_calls += 1
        if kwargs.get("skip_output_reshape", False):
            return attended.transpose(1, 2)
        return attended.reshape(q.shape[0], self.sequence, -1)
