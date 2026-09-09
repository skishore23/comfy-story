from __future__ import annotations

import pytest
import torch

from comfy_story.losses import Decision1Losses, LocalExceptionScorer, decision1_losses


def test_total_loss_uses_locked_weights() -> None:
    parts = Decision1Losses(
        velocity=torch.tensor(1.0),
        hidden=torch.tensor(2.0),
        latent_calibration=torch.tensor(3.0),
        influence_huber=torch.tensor(4.0),
        influence_pairwise=torch.tensor(5.0),
        rare_detail_velocity=torch.tensor(6.0),
    )

    expected = torch.tensor(1.0 + 0.25 * 2.0 + 0.10 * 3.0 + 0.25 * 4.0 + 0.10 * 5.0 + 0.50 * 6.0)

    assert torch.equal(parts.total(), expected)
    assert parts.logged() == pytest.approx(
        {
            "hidden": 2.0,
            "influence_huber": 4.0,
            "influence_pairwise": 5.0,
            "latent_calibration": 3.0,
            "rare_detail_velocity": 6.0,
            "total": 6.3,
            "velocity": 1.0,
        }
    )


def test_loss_components_have_finite_gradients_with_locked_inner_arithmetic() -> None:
    predicted_velocity = torch.tensor([1.0, 3.0], requires_grad=True)
    predicted_hidden = torch.tensor([2.0, 4.0], requires_grad=True)
    predicted_latent = torch.tensor([1.0, 5.0], requires_grad=True)
    predicted_influence = torch.tensor([0.0, 2.0, 1.0], requires_grad=True)

    parts = decision1_losses(
        predicted_velocity=predicted_velocity,
        teacher_velocity=torch.tensor([0.0, 1.0]),
        predicted_hidden=predicted_hidden,
        teacher_hidden=torch.tensor([0.0, 2.0]),
        core_latent=predicted_latent,
        calibration_target=torch.tensor([1.0, 1.0]),
        predicted_influence=predicted_influence,
        target_influence=torch.tensor([0.0, 1.0, 3.0]),
        rare_detail_mask=torch.tensor([False, True]),
    )

    torch.testing.assert_close(parts.velocity, torch.tensor(2.5))
    torch.testing.assert_close(parts.hidden, torch.tensor(4.0))
    torch.testing.assert_close(parts.latent_calibration, torch.tensor(8.0))
    torch.testing.assert_close(parts.influence_huber, torch.tensor(2.0 / 3.0))
    torch.testing.assert_close(parts.influence_pairwise, torch.tensor(1.5))
    torch.testing.assert_close(parts.rare_detail_velocity, torch.tensor(4.0))

    parts.total().backward()  # type: ignore[no-untyped-call]
    for value in (
        predicted_velocity,
        predicted_hidden,
        predicted_latent,
        predicted_influence,
    ):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_empty_rare_mask_and_pairwise_ties_are_differentiable_zeros() -> None:
    predicted_velocity = torch.tensor([1.0, 2.0], requires_grad=True)
    predicted_influence = torch.tensor([3.0, 4.0], requires_grad=True)

    parts = decision1_losses(
        predicted_velocity=predicted_velocity,
        teacher_velocity=torch.zeros(2),
        predicted_hidden=torch.zeros(2, requires_grad=True),
        teacher_hidden=torch.zeros(2),
        core_latent=torch.zeros(2, requires_grad=True),
        calibration_target=torch.zeros(2),
        predicted_influence=predicted_influence,
        target_influence=torch.ones(2),
        rare_detail_mask=torch.zeros(2, dtype=torch.bool),
    )

    assert parts.rare_detail_velocity.item() == 0.0
    assert parts.influence_pairwise.item() == 0.0
    (parts.rare_detail_velocity + parts.influence_pairwise).backward()  # type: ignore[no-untyped-call]
    velocity_grad = predicted_velocity.grad
    influence_grad = predicted_influence.grad
    assert velocity_grad is not None
    assert influence_grad is not None
    assert torch.equal(velocity_grad, torch.zeros_like(predicted_velocity))
    assert torch.equal(influence_grad, torch.zeros_like(predicted_influence))


def test_pairwise_influence_compares_items_only_within_each_sample() -> None:
    parts = decision1_losses(
        predicted_velocity=torch.zeros(2, 2),
        teacher_velocity=torch.zeros(2, 2),
        predicted_hidden=torch.zeros(2, 2),
        teacher_hidden=torch.zeros(2, 2),
        core_latent=torch.zeros(2, 2),
        calibration_target=torch.zeros(2, 2),
        predicted_influence=torch.tensor([[0.0, 1.0], [100.0, 101.0]]),
        target_influence=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        rare_detail_mask=torch.zeros(2, dtype=torch.bool),
    )

    assert parts.influence_pairwise.item() == 0.0


def test_rare_detail_mask_selects_whole_batch_examples() -> None:
    parts = decision1_losses(
        predicted_velocity=torch.tensor([[1.0, 3.0], [10.0, 20.0]]),
        teacher_velocity=torch.zeros(2, 2),
        predicted_hidden=torch.zeros(2, 2),
        teacher_hidden=torch.zeros(2, 2),
        core_latent=torch.zeros(2, 2),
        calibration_target=torch.zeros(2, 2),
        predicted_influence=torch.zeros(2),
        target_influence=torch.zeros(2),
        rare_detail_mask=torch.tensor([True, False]),
    )

    torch.testing.assert_close(parts.rare_detail_velocity, torch.tensor(5.0))


def test_local_exception_scorer_is_item_local() -> None:
    torch.manual_seed(13)
    scorer = LocalExceptionScorer(channels=4, event_types=3, hidden_width=8)
    items = torch.randn(3, 4, 1, 2, 2)
    event_types = torch.tensor([0, 1, 2])
    timestamps = torch.tensor([0.1, 0.5, 0.9])

    baseline = scorer(items, event_types=event_types, timestamps=timestamps)
    changed = items.clone()
    changed[1:].mul_(1000)
    changed_scores = scorer(changed, event_types=event_types, timestamps=timestamps)

    assert baseline.shape == (3,)
    assert torch.equal(baseline[0], changed_scores[0])


def test_loss_rejects_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        decision1_losses(
            predicted_velocity=torch.tensor([float("nan")]),
            teacher_velocity=torch.zeros(1),
            predicted_hidden=torch.zeros(1),
            teacher_hidden=torch.zeros(1),
            core_latent=torch.zeros(1),
            calibration_target=torch.zeros(1),
            predicted_influence=torch.zeros(1),
            target_influence=torch.zeros(1),
            rare_detail_mask=torch.zeros(1, dtype=torch.bool),
        )
