"""Bounded visual-reference compilers for MiniMax H3 conditioning."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable
from typing import cast

import torch
from torch import nn

from duet.duetx.h3_reference_alignment import (
    AlignedReferencePack,
    fold_video_into_streams,
    pack_aligned_images,
)
from duet.duetx.h3_reference_contracts import (
    H3CompiledBlock,
    H3CompiledContext,
    H3CompileReceipt,
    H3ReferenceBudget,
    H3ReferenceKind,
    H3ReferenceMethod,
    H3VisualReference,
)
from duet.duetx.latent_bridge import LatentHistoryBridge

_GATED_HIDDEN_WIDTH = 635
_RESAMPLER_ATTENTION_WIDTH = 215
_RECEIPT_FORMAT = "duet-x-h3-reference-compile-receipt-v1"


class AlignedReferenceCompressor(nn.Module):
    """Shared interface for a trainable aligned-reference core."""

    method: H3ReferenceMethod

    def __init__(self, method: H3ReferenceMethod) -> None:
        super().__init__()
        self.method = method

    def compress(self, pack: AlignedReferencePack) -> torch.Tensor:
        """Return one H3-native core for an already aligned pack."""
        raise NotImplementedError

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class GatedReferenceCompressor(AlignedReferenceCompressor):
    """Learn one global mixture weight per available reference stream."""

    def __init__(self, channels: int = 24) -> None:
        super().__init__(H3ReferenceMethod.GATED)
        self.gate = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, _GATED_HIDDEN_WIDTH),
            nn.GELU(),
            nn.Linear(_GATED_HIDDEN_WIDTH, 1),
        )

    def compress(self, pack: AlignedReferencePack) -> torch.Tensor:
        pack.validate()
        gate_dtype = next(self.gate.parameters()).dtype
        descriptors = pack.history.mean(dim=(3, 4, 5)).to(dtype=gate_dtype)
        gates = self.gate(descriptors).sigmoid()
        gates = gates.masked_fill(~pack.availability.unsqueeze(-1), 0)
        denominator = gates.sum(dim=1, keepdim=True).clamp_min(torch.finfo(gates.dtype).eps)
        weights = gates / denominator
        core = (pack.history.to(dtype=weights.dtype) * weights[..., None, None, None]).sum(dim=1)
        return cast(torch.Tensor, core.to(dtype=pack.history.dtype))


class ResamplerReferenceCompressor(AlignedReferenceCompressor):
    """Attend across references independently at every H3 latent position."""

    def __init__(self, channels: int = 24) -> None:
        super().__init__(H3ReferenceMethod.RESAMPLER)
        width = _RESAMPLER_ATTENTION_WIDTH
        self.key = nn.Linear(channels, width, bias=False)
        self.value = nn.Linear(channels, width, bias=False)
        self.query = nn.Parameter(torch.empty(width))
        self.output = nn.Linear(width, channels, bias=False)
        nn.init.normal_(self.query, std=0.02)

    def compress(self, pack: AlignedReferencePack) -> torch.Tensor:
        pack.validate()
        batch, streams, channels, frames, height, width = pack.history.shape
        tokens = pack.history.permute(0, 1, 3, 4, 5, 2).reshape(
            batch, streams, frames * height * width, channels
        )
        projected = tokens.to(dtype=self.key.weight.dtype)
        logits = torch.einsum("bstw,w->bst", self.key(projected), self.query)
        logits = logits.masked_fill(~pack.availability.unsqueeze(-1), -torch.inf)
        weights = (logits / math.sqrt(self.query.numel())).softmax(dim=1).unsqueeze(-1)
        resampled = self.output((self.value(projected) * weights).sum(dim=1))
        core = (
            resampled.reshape(batch, frames, height, width, channels)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        return cast(torch.Tensor, core.to(dtype=pack.history.dtype))


class DuetReferenceCompressor(AlignedReferenceCompressor):
    """Apply ordered associative Duet fusion to aligned H3 visual streams."""

    def __init__(
        self, channels: int = 24, *, method: H3ReferenceMethod = H3ReferenceMethod.DUET
    ) -> None:
        if method not in {H3ReferenceMethod.DUET, H3ReferenceMethod.DUET_X}:
            raise ValueError("Duet H3 compressor method must be duet or duet_x")
        super().__init__(method)
        self.bridge = LatentHistoryBridge(channels=channels, operator_size=16)

    def compress(self, pack: AlignedReferencePack) -> torch.Tensor:
        pack.validate()
        return cast(
            torch.Tensor,
            self.bridge(
                pack.history,
                anchor=pack.last_available,
                availability=pack.availability,
            ).core_latent,
        )


def build_reference_compressor(method: H3ReferenceMethod) -> AlignedReferenceCompressor:
    """Construct the trainable core for one learned compiler method."""
    if method is H3ReferenceMethod.GATED:
        return GatedReferenceCompressor()
    if method is H3ReferenceMethod.RESAMPLER:
        return ResamplerReferenceCompressor()
    if method in {H3ReferenceMethod.DUET, H3ReferenceMethod.DUET_X}:
        return DuetReferenceCompressor(method=method)
    raise ValueError(f"{method.value} does not use a trainable reference compressor")


def _native_block(reference: H3VisualReference, *, exact: bool = False) -> H3CompiledBlock:
    return H3CompiledBlock(
        reference.kind, reference.latent, (reference.source_id,), exact
    ).validate()


def _without_protection(reference: H3VisualReference) -> H3VisualReference:
    return H3VisualReference(
        reference.source_id,
        reference.kind,
        reference.ordinal,
        reference.latent,
        False,
    ).validate()


def _group_images(
    references: Iterable[H3VisualReference],
) -> tuple[tuple[H3VisualReference, ...], ...]:
    groups: OrderedDict[
        tuple[tuple[int, ...], torch.dtype, torch.device], list[H3VisualReference]
    ] = OrderedDict()
    for reference in references:
        key = (tuple(reference.latent.shape), reference.latent.dtype, reference.latent.device)
        groups.setdefault(key, []).append(reference)
    chunks: list[tuple[H3VisualReference, ...]] = []
    for group in groups.values():
        chunks.extend(tuple(group[index : index + 8]) for index in range(0, len(group), 8))
    return tuple(chunks)


class H3ReferenceCompiler(nn.Module):
    """Compile a semantic roster into bounded ordinary ``minimax_refs`` blocks."""

    def __init__(
        self,
        method: H3ReferenceMethod,
        budget: H3ReferenceBudget,
        *,
        checkpoint_sha256: str | None = None,
        compressor: AlignedReferenceCompressor | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(method, H3ReferenceMethod):
            raise ValueError("H3 compiler method is unsupported")
        self.method = method
        self.budget = budget.validate()
        learned = method in {
            H3ReferenceMethod.GATED,
            H3ReferenceMethod.RESAMPLER,
            H3ReferenceMethod.DUET,
            H3ReferenceMethod.DUET_X,
        }
        if learned:
            resolved = build_reference_compressor(method) if compressor is None else compressor
            if resolved.method is not method:
                raise ValueError("H3 compiler compressor method does not match")
            if (
                not isinstance(checkpoint_sha256, str)
                or len(checkpoint_sha256) != 64
                or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
            ):
                raise ValueError("learned H3 compiler requires a lowercase checkpoint SHA-256")
            self.compressor: AlignedReferenceCompressor | None = resolved
        else:
            if compressor is not None or checkpoint_sha256 is not None:
                raise ValueError("non-learned H3 compiler cannot bind a checkpoint or compressor")
            self.compressor = None
        self.checkpoint_sha256 = checkpoint_sha256

    def _compile_dense(
        self, references: tuple[H3VisualReference, ...]
    ) -> tuple[tuple[H3CompiledBlock, ...], int]:
        if self.compressor is None:
            raise RuntimeError("dense compilation requires a compressor")
        blocks: list[H3CompiledBlock] = []
        padding = 0
        images = tuple(
            reference for reference in references if reference.kind is H3ReferenceKind.IMAGE
        )
        videos = tuple(
            reference for reference in references if reference.kind is H3ReferenceKind.VIDEO
        )
        for group in _group_images(images):
            if len(group) == 1:
                blocks.append(_native_block(group[0]))
                continue
            pack = pack_aligned_images(group)
            blocks.append(
                H3CompiledBlock(
                    H3ReferenceKind.IMAGE,
                    self.compressor.compress(pack),
                    pack.source_ids,
                    False,
                ).validate()
            )
        for reference in videos:
            pack = fold_video_into_streams(reference)
            blocks.append(
                H3CompiledBlock(
                    H3ReferenceKind.VIDEO,
                    self.compressor.compress(pack),
                    pack.source_ids,
                    False,
                ).validate()
            )
            padding += pack.padding_temporal_tokens
        return tuple(blocks), padding

    def compile_references(self, references: tuple[H3VisualReference, ...]) -> H3CompiledContext:
        """Compile once without changing the Qwen semantic roster or source order."""
        if not isinstance(references, tuple) or not references:
            raise ValueError("H3 compiler requires a nonempty reference tuple")
        for reference in references:
            reference.validate()
        if len({reference.source_id for reference in references}) != len(references):
            raise ValueError("H3 compiler source IDs must be unique")
        ordinals = tuple(reference.ordinal for reference in references)
        if ordinals != tuple(sorted(ordinals)) or len(set(ordinals)) != len(ordinals):
            raise ValueError("H3 compiler references must use unique ascending ordinals")

        padding = 0
        if self.method is H3ReferenceMethod.NATIVE_FULL:
            blocks = tuple(_native_block(reference) for reference in references)
        elif self.method is H3ReferenceMethod.NATIVE_TRIMMED:
            selected: list[H3CompiledBlock] = []
            rows = 0
            for reference in references:
                if rows + reference.visual_rows <= self.budget.max_visual_rows:
                    selected.append(_native_block(reference))
                    rows += reference.visual_rows
            if not selected:
                raise ValueError("H3 visual row budget cannot hold any native reference")
            blocks = tuple(selected)
        elif self.method is H3ReferenceMethod.EXCEPTIONS_ONLY:
            protected = tuple(reference for reference in references if reference.protected)
            if not protected:
                raise ValueError("exceptions-only H3 compilation requires a protected image")
            if any(reference.kind is not H3ReferenceKind.IMAGE for reference in protected):
                raise ValueError("exact H3 exceptions must be images")
            blocks = tuple(_native_block(reference, exact=True) for reference in protected)
        elif self.method is H3ReferenceMethod.DUET_X:
            protected = tuple(reference for reference in references if reference.protected)
            if len(protected) > self.budget.max_exceptions:
                raise ValueError("protected H3 references exceed the exception budget")
            if any(reference.kind is not H3ReferenceKind.IMAGE for reference in protected):
                raise ValueError("exact H3 exceptions must be images")
            dense = tuple(reference for reference in references if not reference.protected)
            dense_blocks, padding = self._compile_dense(dense) if dense else ((), 0)
            blocks = (
                *dense_blocks,
                *(_native_block(reference, exact=True) for reference in protected),
            )
        else:
            dense = tuple(_without_protection(reference) for reference in references)
            blocks, padding = self._compile_dense(dense)

        if not blocks:
            raise ValueError("H3 compiler produced no visual blocks")
        output_rows = sum(block.visual_rows for block in blocks)
        trainable_parameters = (
            0 if self.compressor is None else self.compressor.trainable_parameter_count()
        )
        receipt = H3CompileReceipt(
            _RECEIPT_FORMAT,
            self.method,
            len(references),
            len(references),
            sum(reference.visual_rows for reference in references),
            output_rows,
            tuple(reference.tensor_sha256 for reference in references),
            tuple(block.tensor_sha256 for block in blocks),
            self.checkpoint_sha256,
            "bypass",
            padding,
            trainable_parameters,
        ).validate()
        return H3CompiledContext(tuple(blocks), receipt, self.budget).validate()


__all__ = (
    "AlignedReferenceCompressor",
    "DuetReferenceCompressor",
    "GatedReferenceCompressor",
    "H3ReferenceCompiler",
    "ResamplerReferenceCompressor",
    "build_reference_compressor",
)
