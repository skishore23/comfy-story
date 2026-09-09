from __future__ import annotations

import pytest
import torch

from comfy_story.contracts import (
    DuetXContract,
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    RawEvidencePointer,
    SpatialLocation,
    TimeFrameRange,
)
from comfy_story.ltx_bridge import LTXLatentHistoryBridge


def _history(
    *,
    batch: int = 1,
    streams: int = 6,
    channels: int = 128,
    frames: int = 1,
    height: int = 2,
    width: int = 3,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.randn(batch, streams, channels, frames, height, width, dtype=dtype)


def _candidate(slot: int, score: int) -> ExceptionItem:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    digest = f"{slot + 1:064x}"
    return ExceptionItem(
        f"item-{slot}",
        EvidenceProvenance(
            LeafKey(f"camera-{slot}", slot, slot),
            TimeFrameRange(slot * 10, slot * 10 + 1, slot, slot + 1),
            SpatialLocation("whole-latent-v1", "bbox", (0, 0, 65_536, 65_536)),
            "whole-history-latent-v1",
            RawEvidencePointer(f"duet-evidence://test/sha256/{digest}", digest, 0, 1),
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "5" * 64,
            contract.sparse.fingerprint(),
        ),
        torch.full((128,), float(slot)),
        score,
    )


def test_zero_init_bridge_is_exact_anchor_and_reports_operator() -> None:
    history = _history(height=4, width=5, dtype=torch.bfloat16)
    anchor = history[:, -1].clone()

    output = LTXLatentHistoryBridge()(history, anchor=anchor)

    assert torch.equal(output.core_latent, anchor)
    assert output.core_operator.shape == (1, 20, 16, 16)
    assert output.core_operator.dtype == torch.float32
    assert output.core_operator.device == anchor.device
    assert output.per_reference_rmse.shape == (1, 6)
    assert output.per_reference_rmse.dtype == torch.float32
    assert output.per_reference_rmse.device == anchor.device
    assert output.uncertainty.shape == (1,)
    assert output.uncertainty.dtype == torch.float32
    assert output.uncertainty.device == anchor.device


@pytest.mark.parametrize(
    ("history", "anchor", "availability", "message"),
    [
        (torch.randn(1, 2, 128, 1, 2), torch.randn(1, 128, 1, 2, 1), None, "history"),
        (_history(channels=127), torch.randn(1, 127, 1, 2, 3), None, "channels"),
        (_history(), torch.randn(1, 128, 1, 2, 2), None, "anchor"),
        (_history(), torch.randn(1, 128, 1, 2, 3), torch.ones(1, 6), "bool"),
        (
            _history(),
            torch.randn(1, 128, 1, 2, 3),
            torch.ones(1, 5, dtype=torch.bool),
            "availability",
        ),
    ],
)
def test_bridge_rejects_invalid_boundaries(
    history: torch.Tensor,
    anchor: torch.Tensor,
    availability: torch.Tensor | None,
    message: str,
) -> None:
    bridge = LTXLatentHistoryBridge()

    with pytest.raises(ValueError, match=message):
        bridge(history, anchor=anchor, availability=availability)


def test_bridge_requires_one_available_reference_per_batch_row() -> None:
    history = _history(batch=2, streams=3)
    availability = torch.tensor([[True, False, False], [False, False, False]])

    with pytest.raises(ValueError, match="at least one"):
        LTXLatentHistoryBridge()(history, anchor=history[:, -1], availability=availability)


def test_identity_masking_ignores_unavailable_reference_values() -> None:
    torch.manual_seed(7)
    bridge = LTXLatentHistoryBridge(operator_size=4)
    history = _history(batch=2, streams=4, channels=128, height=2, width=2)
    availability = torch.tensor(
        [[True, False, True, False], [False, True, True, False]], dtype=torch.bool
    )
    changed = history.clone()
    changed[~availability] = torch.randn_like(changed[~availability]) * 100

    original = bridge(history, anchor=history[:, 0], availability=availability)
    replaced = bridge(changed, anchor=history[:, 0], availability=availability)

    assert torch.equal(original.core_operator, replaced.core_operator)
    assert torch.equal(original.core_latent, replaced.core_latent)
    expected_uncertainty = original.per_reference_rmse.masked_fill(~availability, -torch.inf).amax(
        dim=1
    )
    torch.testing.assert_close(original.uncertainty, expected_uncertainty)
    torch.testing.assert_close(original.uncertainty, replaced.uncertainty)


def test_bridge_operator_is_order_sensitive_and_regroupable() -> None:
    torch.manual_seed(11)
    bridge = LTXLatentHistoryBridge(operator_size=4)
    history = _history(streams=5, height=1, width=2)
    output = bridge(history, anchor=history[:, -1])
    reordered = bridge(history[:, [1, 0, 2, 3, 4]], anchor=history[:, -1])
    tokens = history.permute(0, 1, 3, 4, 5, 2).flatten(2, 4)
    leaves = tuple(bridge.fusion.encode_stream(tokens).unbind(dim=1))
    prefix = bridge.fusion.combine_ordered_operators(leaves[:3])
    suffix = bridge.fusion.combine_ordered_operators(leaves[3:])
    regrouped = bridge.fusion.combine_ordered_operators((prefix, suffix))

    assert not torch.allclose(output.core_operator, reordered.core_operator)
    torch.testing.assert_close(output.core_operator, regrouped, atol=2e-6, rtol=2e-6)


def test_prefix_cache_matches_cold_bridge_operator() -> None:
    torch.manual_seed(13)
    bridge = LTXLatentHistoryBridge(operator_size=4)
    history = _history(streams=4, height=1, width=2)
    tokens = history.permute(0, 1, 3, 4, 5, 2).flatten(2, 4)
    leaves = tuple(bridge.fusion.encode_stream(tokens).unbind(dim=1))
    prefix = bridge.fusion.combine_ordered_operators(leaves[:3])
    cached = bridge.fusion.combine_ordered_operators((prefix, leaves[3]))

    output = bridge(history, anchor=history[:, -1])

    torch.testing.assert_close(output.core_operator, cached, atol=2e-6, rtol=2e-6)


def test_bridge_supports_gradients_and_preserves_anchor_dtype_and_device() -> None:
    torch.manual_seed(17)
    bridge = LTXLatentHistoryBridge(operator_size=4)
    history = _history(batch=2, streams=3, height=1, width=2, dtype=torch.bfloat16)
    history.requires_grad_()
    anchor = history[:, -1].detach().clone()

    output = bridge(history, anchor=anchor)
    output.core_latent.float().sum().backward()

    assert output.core_latent.dtype == anchor.dtype
    assert output.core_latent.device == anchor.device
    assert bridge.residual.weight.grad is not None
    assert torch.isfinite(bridge.residual.weight.grad).all()
    assert torch.count_nonzero(bridge.residual.weight.grad) > 0


def test_nonzero_residual_propagates_gradients_through_fusion() -> None:
    torch.manual_seed(19)
    bridge = LTXLatentHistoryBridge(operator_size=4)
    with torch.no_grad():
        bridge.residual.weight.fill_(0.01)
    history = _history(batch=2, streams=3, height=1, width=2)
    history.requires_grad_()

    output = bridge(history, anchor=history[:, -1].detach())
    output.core_latent.square().mean().backward()

    assert history.grad is not None
    assert torch.isfinite(history.grad).all()
    assert torch.count_nonzero(history.grad) > 0
    for parameter in bridge.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_exclusion_bridge_uses_mergeable_root_and_latest_nonselected_anchor() -> None:
    torch.manual_seed(23)
    bridge = LTXLatentHistoryBridge(operator_size=4)
    with torch.no_grad():
        bridge.residual.weight.copy_(torch.eye(128))
    history = _history(streams=8, height=1, width=2)
    candidates = tuple(
        _candidate(slot, 100 if slot == 2 else 90 if slot == 5 else slot) for slot in range(8)
    )
    changed = history.clone()
    changed[:, 2].fill_(10_000)
    changed[:, 5].fill_(-10_000)

    original = bridge.forward_exclusion(history, candidates=candidates)
    mutated = bridge.forward_exclusion(changed, candidates=candidates)

    assert original.excluded_leaf_keys == (
        LeafKey("camera-2", 2, 2),
        LeafKey("camera-5", 5, 5),
    )
    assert original.anchor_slot == 7
    assert torch.equal(original.core_operator, mutated.core_operator)
    assert torch.equal(original.core_latent, mutated.core_latent)
    assert torch.equal(original.anchor_latent, history[:, 7])
    assert original.exception_items == (candidates[2], candidates[5])


def test_diagnostics_are_finite_and_exclude_unavailable_references() -> None:
    bridge = LTXLatentHistoryBridge(operator_size=4)
    history = torch.zeros(2, 3, 128, 1, 1, 1)
    history[0, 0] = 3
    history[0, 1] = 100
    history[1, 0] = 4
    history[1, 1] = 12
    availability = torch.tensor([[True, False, True], [True, True, False]])
    anchor = torch.zeros(2, 128, 1, 1, 1)

    output = bridge(history, anchor=anchor, availability=availability)

    torch.testing.assert_close(
        output.per_reference_rmse,
        torch.tensor([[3.0, 100.0, 0.0], [4.0, 12.0, 0.0]]),
    )
    torch.testing.assert_close(output.uncertainty, torch.tensor([3.0, 12.0]))
    assert torch.isfinite(output.uncertainty).all()


def test_uncertainty_cannot_select_a_large_unavailable_rmse() -> None:
    bridge = LTXLatentHistoryBridge(operator_size=4)
    history = torch.zeros(1, 3, 128, 1, 1, 1)
    history[0, 0] = 2
    history[0, 1] = 10_000
    history[0, 2] = 5
    availability = torch.tensor([[True, False, True]])
    anchor = torch.zeros(1, 128, 1, 1, 1)

    output = bridge(history, anchor=anchor, availability=availability)

    torch.testing.assert_close(output.uncertainty, torch.tensor([5.0]))


def test_bridge_state_contains_only_fusion_and_zero_initialized_residual() -> None:
    bridge = LTXLatentHistoryBridge()
    keys = tuple(bridge.state_dict())

    assert keys
    assert all(key.startswith(("fusion.", "residual.")) for key in keys)
    assert not any(key.startswith(("ltx", "transformer", "vae", "text_encoder")) for key in keys)
    assert torch.count_nonzero(bridge.residual.weight) == 0
