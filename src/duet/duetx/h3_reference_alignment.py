"""Explicit alignment rules for H3 visual reference compression."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

import torch

from duet.duetx.h3_reference_contracts import H3ReferenceKind, H3VisualReference


@dataclass(frozen=True, slots=True)
class AlignedReferencePack:
    """Eight aligned streams plus the exact sources and padding that formed them."""

    history: torch.Tensor
    availability: torch.Tensor
    source_ids: tuple[str, ...]
    output_kind: H3ReferenceKind
    padding_temporal_tokens: int

    def validate(self) -> Self:
        if (
            not isinstance(self.history, torch.Tensor)
            or self.history.layout != torch.strided
            or self.history.ndim != 6
            or self.history.shape[0] != 1
            or self.history.shape[1] != 8
            or self.history.shape[2] != 24
            or any(dimension <= 0 for dimension in self.history.shape[3:])
        ):
            raise ValueError("aligned H3 history must have shape [1,8,24,F,H,W]")
        if self.history.shape[4] % 2 or self.history.shape[5] % 2:
            raise ValueError("aligned H3 history height and width must be even")
        if not torch.is_floating_point(self.history) or not bool(
            torch.isfinite(self.history).all().item()
        ):
            raise ValueError("aligned H3 history must be finite and floating")
        if (
            not isinstance(self.availability, torch.Tensor)
            or self.availability.dtype is not torch.bool
            or tuple(self.availability.shape) != (1, 8)
            or self.availability.device != self.history.device
        ):
            raise ValueError("aligned H3 availability must be bool[1,8] on the history device")
        available = self.availability[0].tolist()
        available_count = sum(available)
        if available_count <= 0 or available != [True] * available_count + [False] * (
            8 - available_count
        ):
            raise ValueError("aligned H3 available streams must form a nonempty prefix")
        if not isinstance(self.source_ids, tuple) or not self.source_ids:
            raise ValueError("aligned H3 source_ids must be a nonempty tuple")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("aligned H3 source_ids must be unique")
        if not isinstance(self.output_kind, H3ReferenceKind):
            raise ValueError("aligned H3 output kind is unsupported")
        if self.output_kind is H3ReferenceKind.IMAGE:
            if self.history.shape[3] != 1:
                raise ValueError("aligned image history must contain one temporal token")
            if available_count != len(self.source_ids):
                raise ValueError("aligned image sources must match available streams")
            if self.padding_temporal_tokens != 0:
                raise ValueError("aligned image history cannot have temporal padding")
        else:
            if available_count != 8 or len(self.source_ids) != 1:
                raise ValueError("aligned video history requires eight chunks from one source")
            if (
                type(self.padding_temporal_tokens) is not int
                or not 0 <= self.padding_temporal_tokens < 8
            ):
                raise ValueError("aligned video temporal padding must be in range [0,8)")
        return self

    @property
    def last_available(self) -> torch.Tensor:
        self.validate()
        index = int(self.availability[0].to(torch.int64).sum().item()) - 1
        return self.history[:, index]


def pack_aligned_images(
    references: tuple[H3VisualReference, ...],
    *,
    streams: int = 8,
) -> AlignedReferencePack:
    """Stack same-shape non-protected image references and pad unavailable slots."""
    if streams != 8:
        raise ValueError("H3 image alignment requires exactly eight streams")
    if not isinstance(references, tuple) or not references or len(references) > streams:
        raise ValueError("H3 image alignment requires one to eight references")
    for reference in references:
        reference.validate()
        if reference.kind is not H3ReferenceKind.IMAGE:
            raise ValueError("H3 image alignment accepts image references only")
        if reference.protected:
            raise ValueError("protected references must remain outside the dense image pack")
    first = references[0].latent
    if any(
        tuple(reference.latent.shape) != tuple(first.shape)
        or reference.latent.dtype != first.dtype
        or reference.latent.device != first.device
        for reference in references[1:]
    ):
        raise ValueError(
            "aligned image references must have the same latent shape, dtype, and device"
        )
    items = [reference.latent for reference in references]
    items.extend(references[-1].latent.clone() for _ in range(streams - len(items)))
    history = torch.stack(items, dim=1)
    availability = torch.tensor(
        [[True] * len(references) + [False] * (streams - len(references))],
        dtype=torch.bool,
        device=history.device,
    )
    return AlignedReferencePack(
        history,
        availability,
        tuple(reference.source_id for reference in references),
        H3ReferenceKind.IMAGE,
        0,
    ).validate()


def fold_video_into_streams(
    reference: H3VisualReference,
    *,
    streams: int = 8,
) -> AlignedReferencePack:
    """Treat eight chronological video chunks as Duet's ordered stream dimension."""
    reference.validate()
    if reference.kind is not H3ReferenceKind.VIDEO or streams != 8:
        raise ValueError("temporal folding requires one video and exactly eight streams")
    if reference.protected:
        raise ValueError("a protected video cannot enter the dense temporal fold")
    batch, channels, frames, height, width = reference.latent.shape
    folded_frames = math.ceil(frames / streams)
    padding = streams * folded_frames - frames
    if padding:
        tail = reference.latent[:, :, -1:].expand(batch, channels, padding, height, width)
        padded = torch.cat((reference.latent, tail), dim=2)
    else:
        padded = reference.latent
    history = padded.reshape(batch, channels, streams, folded_frames, height, width)
    history = history.permute(0, 2, 1, 3, 4, 5).contiguous()
    return AlignedReferencePack(
        history,
        torch.ones((batch, streams), dtype=torch.bool, device=history.device),
        (reference.source_id,),
        H3ReferenceKind.VIDEO,
        padding,
    ).validate()


__all__ = ("AlignedReferencePack", "fold_video_into_streams", "pack_aligned_images")
