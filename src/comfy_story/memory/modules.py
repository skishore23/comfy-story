"""Fixed module architecture for authenticated H3 memory checkpoints."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self, cast

import torch
from torch import nn

_MODULE_NAMES = ("bridge", "gated", "resampler", "scorer")


def _finite_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or not torch.is_floating_point(value)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(f"{name} must be a finite floating strided tensor")
    return value


class LocalExceptionScorer(nn.Module):
    """Score each item independently from its own latent, type, and timestamp."""

    def __init__(
        self, channels: int = 24, *, event_types: int = 16, hidden_width: int = 128
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


class CheckpointGatedModule(nn.Module):
    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 635),
            nn.GELU(),
            nn.Linear(635, 1),
        )


class CheckpointResamplerModule(nn.Module):
    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        width = 215
        self.key = nn.Linear(channels, width, bias=False)
        self.value = nn.Linear(channels, width, bias=False)
        self.query = nn.Parameter(torch.empty(width))
        self.output = nn.Linear(width, channels, bias=False)
        nn.init.normal_(self.query, std=0.02)


@dataclass(frozen=True, slots=True)
class MemoryModules:
    """The only four trainable module families allowed in a bridge bundle."""

    bridge: nn.Module
    gated: nn.Module
    resampler: nn.Module
    scorer: nn.Module

    def validate(self) -> Self:
        modules = self.named()
        if any(not isinstance(module, nn.Module) for module in modules.values()):
            raise ValueError("training modules must be torch modules")
        if len({id(module) for module in modules.values()}) != len(_MODULE_NAMES):
            raise ValueError("training modules must be distinct")
        return self

    def named(self) -> dict[str, nn.Module]:
        return {name: getattr(self, name) for name in _MODULE_NAMES}

    def parameters(self) -> Iterator[nn.Parameter]:
        for module in self.named().values():
            yield from module.parameters()

    def parameter_counts(self) -> dict[str, int]:
        self.validate()
        return {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in self.named().items()
        }
