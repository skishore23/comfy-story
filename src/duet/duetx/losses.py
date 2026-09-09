"""Locked bridge and item-local scorer objectives for Duet-X Decision 1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self, cast

import torch
from torch import nn
from torch.nn import functional


def _finite_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or not torch.is_floating_point(value)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(f"{name} must be a finite floating strided tensor")
    return value


@dataclass(frozen=True, slots=True)
class Decision1Losses:
    """The six separately auditable components of the locked training objective."""

    velocity: torch.Tensor
    hidden: torch.Tensor
    latent_calibration: torch.Tensor
    influence_huber: torch.Tensor
    influence_pairwise: torch.Tensor
    rare_detail_velocity: torch.Tensor

    def validate(self) -> Self:
        components = tuple((name, getattr(self, name)) for name in self.component_names())
        for name, value in components:
            _finite_tensor(value, name)
            if value.numel() != 1:
                raise ValueError(f"{name} must be a scalar tensor")
        devices = {value.device for _, value in components}
        if len(devices) != 1:
            raise ValueError("loss components must share one device")
        return self

    @staticmethod
    def component_names() -> tuple[str, ...]:
        """Return the stable names logged for every optimizer step."""
        return (
            "velocity",
            "hidden",
            "latent_calibration",
            "influence_huber",
            "influence_pairwise",
            "rare_detail_velocity",
        )

    def total(self) -> torch.Tensor:
        """Apply the exact preregistered Decision 1 outer weights."""
        self.validate()
        return (
            self.velocity
            + 0.25 * self.hidden
            + 0.10 * self.latent_calibration
            + 0.25 * self.influence_huber
            + 0.10 * self.influence_pairwise
            + 0.50 * self.rare_detail_velocity
        )

    def logged(self) -> dict[str, float]:
        """Detach every component into strict JSON-number primitives."""
        self.validate()
        result = {
            name: float(getattr(self, name).detach().to(torch.float64).item())
            for name in self.component_names()
        }
        result["total"] = float(self.total().detach().to(torch.float64).item())
        if any(not math.isfinite(value) for value in result.values()):
            raise ValueError("logged loss components must be finite")
        return result


def _matching_pair(
    predicted: torch.Tensor,
    target: torch.Tensor,
    predicted_name: str,
    target_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = _finite_tensor(predicted, predicted_name)
    target = _finite_tensor(target, target_name)
    if predicted.shape != target.shape:
        raise ValueError(f"{predicted_name} and {target_name} shapes must match exactly")
    if predicted.device != target.device:
        raise ValueError(f"{predicted_name} and {target_name} devices must match")
    return predicted, target


def _pairwise_influence_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.ndim == 0:
        return predicted * 0.0
    predicted_rows = predicted.reshape(1, -1) if predicted.ndim == 1 else predicted.flatten(1)
    target_rows = target.reshape(1, -1) if target.ndim == 1 else target.flatten(1)
    items = predicted_rows.shape[1]
    if items < 2:
        return predicted.sum() * 0.0
    left, right = torch.triu_indices(items, items, offset=1, device=predicted.device)
    target_difference = target_rows[:, left] - target_rows[:, right]
    non_ties = target_difference != 0
    if not bool(non_ties.any().item()):
        return predicted.sum() * 0.0
    predicted_difference = predicted_rows[:, left] - predicted_rows[:, right]
    return functional.smooth_l1_loss(predicted_difference[non_ties], target_difference[non_ties])


def decision1_losses(
    *,
    predicted_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    predicted_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    core_latent: torch.Tensor,
    calibration_target: torch.Tensor,
    predicted_influence: torch.Tensor,
    target_influence: torch.Tensor,
    rare_detail_mask: torch.Tensor,
) -> Decision1Losses:
    """Compute the locked inner distances and return separately named components."""
    predicted_velocity, teacher_velocity = _matching_pair(
        predicted_velocity, teacher_velocity, "predicted_velocity", "teacher_velocity"
    )
    predicted_hidden, teacher_hidden = _matching_pair(
        predicted_hidden, teacher_hidden, "predicted_hidden", "teacher_hidden"
    )
    core_latent, calibration_target = _matching_pair(
        core_latent, calibration_target, "core_latent", "calibration_target"
    )
    predicted_influence, target_influence = _matching_pair(
        predicted_influence,
        target_influence,
        "predicted_influence",
        "target_influence",
    )
    if (
        not isinstance(rare_detail_mask, torch.Tensor)
        or rare_detail_mask.layout != torch.strided
        or rare_detail_mask.dtype is not torch.bool
        or rare_detail_mask.device != predicted_velocity.device
    ):
        raise ValueError("rare_detail_mask must be a bool tensor on the velocity device")
    element_mask = rare_detail_mask.shape == predicted_velocity.shape
    batch_mask = predicted_velocity.ndim > 0 and rare_detail_mask.shape == (
        predicted_velocity.shape[0],
    )
    if not element_mask and not batch_mask:
        raise ValueError("rare_detail_mask must match velocity elements or batch examples")

    velocity_error = (predicted_velocity - teacher_velocity).square()
    if bool(rare_detail_mask.any().item()):
        rare_velocity = velocity_error[rare_detail_mask].mean()
    else:
        rare_velocity = predicted_velocity.sum() * 0.0
    return Decision1Losses(
        velocity=velocity_error.mean(),
        hidden=(predicted_hidden - teacher_hidden).square().mean(),
        latent_calibration=(core_latent - calibration_target).square().mean(),
        influence_huber=functional.smooth_l1_loss(predicted_influence, target_influence),
        influence_pairwise=_pairwise_influence_loss(predicted_influence, target_influence),
        rare_detail_velocity=rare_velocity,
    ).validate()


class LocalExceptionScorer(nn.Module):
    """Score each item independently from its own latent, type, and timestamp."""

    def __init__(
        self, channels: int = 128, *, event_types: int = 16, hidden_width: int = 128
    ) -> None:
        super().__init__()
        if min(channels, event_types, hidden_width) <= 0:
            raise ValueError("scorer dimensions must be positive")
        self.channels = channels
        self.event_types = event_types
        self.type_embedding = nn.Embedding(event_types, hidden_width)
        self.item_projection = nn.Linear(channels, hidden_width)
        self.timestamp_projection = nn.Linear(1, hidden_width, bias=False)
        self.output = nn.Sequential(nn.GELU(), nn.Linear(hidden_width, 1))

    def forward(
        self,
        items: torch.Tensor,
        *,
        event_types: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Return one score per row without any cross-item operation."""
        _finite_tensor(items, "items")
        if items.ndim != 5 or items.shape[1] != self.channels or items.shape[0] <= 0:
            raise ValueError(f"items must have shape [B,{self.channels},F,H,W]")
        batch = items.shape[0]
        if (
            not isinstance(event_types, torch.Tensor)
            or event_types.dtype != torch.int64
            or event_types.shape != (batch,)
            or event_types.device != items.device
            or not bool(((event_types >= 0) & (event_types < self.event_types)).all().item())
        ):
            raise ValueError("event_types must be in-range int64 values aligned with items")
        _finite_tensor(timestamps, "timestamps")
        if timestamps.shape != (batch,) or timestamps.device != items.device:
            raise ValueError("timestamps must align with items")
        parameter = self.item_projection.weight
        descriptor = items.mean(dim=(2, 3, 4)).to(dtype=parameter.dtype)
        hidden = (
            self.item_projection(descriptor)
            + self.type_embedding(event_types)
            + self.timestamp_projection(timestamps.to(parameter.dtype).unsqueeze(-1))
        )
        return cast(torch.Tensor, self.output(hidden).squeeze(-1))


__all__ = ("Decision1Losses", "LocalExceptionScorer", "decision1_losses")
