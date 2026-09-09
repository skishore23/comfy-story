"""Protect the native packed prefix, model scope, and Sol backend contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from comfy_story.h3_sol_attention import H3SolAttention, load_sol_kernel


def test_layout_uses_target_suffix_after_all_reference_and_audio_segments() -> None:
    policy = H3SolAttention(lambda *a, **kw: torch.empty(0))
    policy.observe(
        SimpleNamespace(
            segments=[
                (0, 10, "text"),
                (10, 500, "ref_video"),
                (500, 700, "audio"),
                (700, 30000, "video"),
            ],
            seq_len=30000,
        ),
        12,
        4,
    )
    assert (policy.video_start, policy.sequence, policy.step, policy.layer) == (700, 30000, 12, 4)
    with pytest.raises(ValueError, match="suffix"):
        policy.observe(
            SimpleNamespace(segments=[(0, 200, "video"), (200, 400, "audio")], seq_len=400), 12, 4
        )


@pytest.mark.parametrize(
    ("step", "layer", "sequence"),
    [(9, 49, 30000), (20, 1, 30000), (20, -1, 30000), (20, 49, 20000)],
)
def test_warmup_refiner_and_short_sequences_use_original_attention(
    step: int, layer: int, sequence: int
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("Sparse kernel must not run")

    policy = H3SolAttention(forbidden, step=step, layer=layer, sequence=sequence)
    x = torch.ones(1, 1, 4, 2)
    result = policy(lambda *a, **kw: x * 2, x, x, x, 1, skip_reshape=True)
    assert torch.equal(result, x * 2)
    assert policy.dense_calls == 1
    assert policy.sparse_calls == 0


def test_sparse_path_rejects_incompatible_cpu_inputs() -> None:
    policy = H3SolAttention(lambda *a, **kw: torch.empty(0), step=10, layer=2, sequence=24576)
    x = torch.empty(1, 1, 24576, 2)
    with pytest.raises(ValueError, match="native H3 CUDA"):
        policy(lambda *a, **kw: x, x, x, x, 1, skip_reshape=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Sol-Attn requires NVIDIA CUDA")
def test_real_sol_kernel_preserves_dense_prefix_queries_and_returns_native_shape() -> None:
    kernel = load_sol_kernel()
    torch.manual_seed(2026)
    q, k, v = (torch.randn(1, 2, 24576, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    policy = H3SolAttention(kernel, step=10, layer=2, video_start=193, sequence=24576)

    def forbidden(*a: Any, **kw: Any) -> torch.Tensor:
        raise AssertionError("Eligible call unexpectedly used dense fallback")

    out = policy(forbidden, q, k, v, 2, skip_reshape=True)
    assert out.shape == (1, 24576, 256)
    assert torch.isfinite(out).all()
    expected = (
        F.scaled_dot_product_attention(q[:, :, :193], k, v).transpose(1, 2).reshape(1, 193, 256)
    )
    torch.testing.assert_close(out[:, :193], expected, rtol=0, atol=0)
    assert policy.sparse_calls == 1
