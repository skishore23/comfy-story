"""Strict model-neutral adapter for ComfyUI MiniMax H3 reference conditioning."""

from __future__ import annotations

from typing import cast

import torch

from comfy_story.contracts import tensor_sha256
from comfy_story.h3_reference_contracts import (
    H3CompiledContext,
    H3ReferenceKind,
    H3SelectedSource,
    H3VisualReference,
)

_IMAGE_FIELDS = frozenset({"kind", "latent_h", "latent_w", "latent"})
_VIDEO_FIELDS = frozenset(
    {"kind", "latent_t", "latent_h", "latent_w", "ref_audio_t", "latent", "audio_latent"}
)


def _conditioning_entries(value: object) -> list[tuple[object, dict[str, object]]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("H3 conditioning must be a nonempty sequence")
    entries: list[tuple[object, dict[str, object]]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("each H3 conditioning entry must contain value and metadata")
        metadata = row[1]
        if not isinstance(metadata, dict) or any(not isinstance(key, str) for key in metadata):
            raise ValueError("H3 conditioning metadata must be a string-keyed mapping")
        entries.append((row[0], cast(dict[str, object], metadata)))
    return entries


def _native_blocks(metadata: dict[str, object]) -> tuple[dict[str, object], ...]:
    refs = metadata.get("minimax_refs")
    if not isinstance(refs, (list, tuple)) or not refs:
        raise ValueError("H3 conditioning must contain nonempty minimax_refs")
    if any(
        not isinstance(block, dict) or any(not isinstance(key, str) for key in block)
        for block in refs
    ):
        raise ValueError("each minimax_refs block must be a string-keyed mapping")
    return tuple(cast(dict[str, object], block) for block in refs)


def _block_signature(block: dict[str, object]) -> tuple[object, ...]:
    kind = block.get("kind")
    scalars: tuple[object, ...]
    if kind == "image":
        fields = _IMAGE_FIELDS
        scalars = (kind, block.get("latent_h"), block.get("latent_w"))
    elif kind in {"video", "video_audio"}:
        fields = _VIDEO_FIELDS
        scalars = (
            kind,
            block.get("latent_t"),
            block.get("latent_h"),
            block.get("latent_w"),
            block.get("ref_audio_t"),
            block.get("audio_latent") is None,
        )
    elif kind == "audio":
        raise ValueError("audio references are outside the visual compiler MVP")
    else:
        raise ValueError("minimax_refs contains an unsupported reference kind")
    if set(block) != fields:
        raise ValueError("minimax_refs block has missing or unknown fields")
    latent = block.get("latent")
    if not isinstance(latent, torch.Tensor):
        raise ValueError("minimax_refs visual block must contain a latent tensor")
    audio = block.get("audio_latent")
    audio_digest = None if audio is None else tensor_sha256(cast(torch.Tensor, audio))
    return (*scalars, tensor_sha256(latent), audio_digest)


def _consistent_native_blocks(
    entries: list[tuple[object, dict[str, object]]],
) -> tuple[dict[str, object], ...]:
    first = _native_blocks(entries[0][1])
    signature = tuple(_block_signature(block) for block in first)
    for _, metadata in entries[1:]:
        candidate = _native_blocks(metadata)
        if tuple(_block_signature(block) for block in candidate) != signature:
            raise ValueError("conditioning entries contain inconsistent minimax_refs")
    return first


def _reference_from_native(block: dict[str, object], source: H3SelectedSource) -> H3VisualReference:
    source.validate()
    kind_value = block.get("kind")
    if kind_value == "video_audio" or block.get("audio_latent") is not None:
        raise ValueError("audio-bearing references are outside the visual compiler MVP")
    try:
        kind = H3ReferenceKind(cast(str, kind_value))
    except ValueError as error:
        raise ValueError("minimax_refs contains an unsupported visual kind") from error
    if source.kind is not kind:
        raise ValueError("H3 source manifest kind does not match minimax_refs")
    latent = block.get("latent")
    reference = H3VisualReference(
        source.source_id,
        kind,
        source.ordinal,
        cast(torch.Tensor, latent),
        source.protected,
    ).validate()
    if block.get("latent_h") != reference.latent.shape[3]:
        raise ValueError("minimax_refs latent_h does not match its tensor")
    if block.get("latent_w") != reference.latent.shape[4]:
        raise ValueError("minimax_refs latent_w does not match its tensor")
    if kind is H3ReferenceKind.VIDEO:
        if block.get("latent_t") != reference.latent.shape[2]:
            raise ValueError("minimax_refs latent_t does not match its tensor")
        if block.get("ref_audio_t") != 0:
            raise ValueError("silent visual video must declare ref_audio_t zero")
    return reference


def extract_visual_references(
    conditioning: object,
    sources: tuple[H3SelectedSource, ...],
) -> tuple[H3VisualReference, ...]:
    """Extract visual H3 blocks and bind them to their shared semantic source IDs."""
    if not isinstance(sources, tuple) or not sources:
        raise ValueError("H3 source manifest must be a nonempty tuple")
    entries = _conditioning_entries(conditioning)
    blocks = _consistent_native_blocks(entries)
    if len(blocks) != len(sources):
        raise ValueError("H3 source manifest does not match minimax_refs")
    if tuple(source.ordinal for source in sources) != tuple(range(len(sources))):
        raise ValueError("H3 source manifest ordinals must be contiguous from zero")
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("H3 source manifest IDs must be unique")
    return tuple(
        _reference_from_native(block, source) for block, source in zip(blocks, sources, strict=True)
    )


def native_visual_rows(conditioning: object) -> int:
    """Count the fixed visual rows H3 will carry through every transformer block."""
    entries = _conditioning_entries(conditioning)
    blocks = _consistent_native_blocks(entries)
    rows = 0
    for block in blocks:
        kind = H3ReferenceKind(cast(str, block["kind"]))
        source = H3SelectedSource(f"row:{len(str(rows))}", kind, 0, False)
        rows += _reference_from_native(block, source).visual_rows
    return rows


def replace_visual_references(
    conditioning: object,
    context: H3CompiledContext,
) -> list[list[object]]:
    """Copy conditioning metadata and replace only its native reference roster."""
    context.validate()
    entries = _conditioning_entries(conditioning)
    _consistent_native_blocks(entries)
    native = tuple(block.to_minimax_ref() for block in context.blocks)
    copied: list[list[object]] = []
    for value, metadata in entries:
        updated = dict(metadata)
        updated["minimax_refs"] = [dict(block) for block in native]
        copied.append([value, updated])
    return copied


__all__ = ("extract_visual_references", "native_visual_rows", "replace_visual_references")
