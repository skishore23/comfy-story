from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from comfy_story.contracts import tensor_sha256
from comfy_story.h3_reference_contracts import (
    H3CompiledBlock,
    H3CompiledContext,
    H3CompileReceipt,
    H3ReferenceBudget,
    H3ReferenceKind,
    H3ReferenceMethod,
    H3SelectedSource,
    H3VisualReference,
)


def _receipt(
    block: H3CompiledBlock,
    *,
    method: H3ReferenceMethod = H3ReferenceMethod.EXCEPTIONS_ONLY,
    input_rows: int | None = None,
) -> H3CompileReceipt:
    rows = block.visual_rows
    return H3CompileReceipt(
        format="duet-x-h3-reference-compile-receipt-v1",
        method=method,
        input_semantic_items=1,
        output_semantic_items=1,
        input_visual_rows=rows if input_rows is None else input_rows,
        output_visual_rows=rows,
        source_sha256s=(tensor_sha256(block.latent),),
        output_sha256s=(tensor_sha256(block.latent),),
        checkpoint_sha256=("a" * 64 if method is H3ReferenceMethod.DUET_X else None),
        cache_status="bypass",
        padding_temporal_tokens=0,
        trainable_parameters=0,
    ).validate()


def test_video_reference_counts_h3_visual_rows() -> None:
    latent = torch.zeros(1, 24, 37, 48, 84)
    reference = H3VisualReference("drive", H3ReferenceKind.VIDEO, 0, latent, False)

    assert reference.visual_rows == 37 * 24 * 42


def test_compiled_context_rejects_budget_overflow() -> None:
    latent = torch.zeros(1, 24, 1, 48, 84)
    block = H3CompiledBlock(H3ReferenceKind.IMAGE, latent, ("face",), True)

    with pytest.raises(ValueError, match="visual row budget"):
        H3CompiledContext(
            blocks=(block,),
            receipt=_receipt(block),
            budget=H3ReferenceBudget(max_visual_rows=1007, max_exceptions=1),
        ).validate()


def test_compiled_block_emits_exact_comfy_h3_dictionary() -> None:
    latent = torch.zeros(1, 24, 7, 48, 84)
    block = H3CompiledBlock(H3ReferenceKind.VIDEO, latent, ("drive",), False).validate()

    assert block.to_minimax_ref() == {
        "kind": "video",
        "latent_t": 7,
        "latent_h": 48,
        "latent_w": 84,
        "ref_audio_t": 0,
        "latent": latent,
        "audio_latent": None,
    }


def test_visual_reference_rejects_invalid_h3_latents() -> None:
    with pytest.raises(ValueError, match=r"\[1,24,T,H,W\]"):
        H3VisualReference(
            "portrait", H3ReferenceKind.IMAGE, 0, torch.zeros(1, 23, 1, 48, 84), False
        ).validate()
    with pytest.raises(ValueError, match="exactly one temporal token"):
        H3VisualReference(
            "portrait", H3ReferenceKind.IMAGE, 0, torch.zeros(1, 24, 2, 48, 84), False
        ).validate()
    with pytest.raises(ValueError, match="finite and floating"):
        H3VisualReference(
            "portrait",
            H3ReferenceKind.IMAGE,
            0,
            torch.full((1, 24, 1, 48, 84), torch.nan),
            False,
        ).validate()


def test_selected_source_requires_portable_identity_and_boolean_protection() -> None:
    assert H3SelectedSource("story:maya", H3ReferenceKind.IMAGE, 0, True).validate()
    with pytest.raises(ValueError, match="portable"):
        H3SelectedSource("../maya", H3ReferenceKind.IMAGE, 0, False).validate()
    with pytest.raises(ValueError, match="boolean"):
        H3SelectedSource("maya", H3ReferenceKind.IMAGE, 0, 1).validate()  # type: ignore[arg-type]


def test_receipt_requires_checkpoint_only_for_learned_methods() -> None:
    latent = torch.zeros(1, 24, 1, 48, 84)
    block = H3CompiledBlock(H3ReferenceKind.IMAGE, latent, ("face",), True)
    native = _receipt(block)
    learned = replace(
        native,
        method=H3ReferenceMethod.DUET_X,
        checkpoint_sha256="b" * 64,
    )

    assert learned.validate().compression_factor == 1.0
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        replace(learned, checkpoint_sha256=None).validate()
    with pytest.raises(ValueError, match="must not bind"):
        replace(native, checkpoint_sha256="b" * 64).validate()


def test_context_rejects_duplicate_sources_and_hash_drift() -> None:
    latent = torch.zeros(1, 24, 1, 48, 84)
    first = H3CompiledBlock(H3ReferenceKind.IMAGE, latent, ("face",), True)
    second = H3CompiledBlock(H3ReferenceKind.IMAGE, latent.clone(), ("face",), False)
    receipt = replace(
        _receipt(first),
        output_visual_rows=first.visual_rows + second.visual_rows,
        output_sha256s=(first.tensor_sha256, second.tensor_sha256),
    )

    with pytest.raises(ValueError, match="duplicate"):
        H3CompiledContext((first, second), receipt, H3ReferenceBudget(3000, 1)).validate()
    with pytest.raises(ValueError, match="output hashes"):
        H3CompiledContext(
            (first,),
            replace(_receipt(first), output_sha256s=("f" * 64,)),
            H3ReferenceBudget(1008, 1),
        ).validate()
