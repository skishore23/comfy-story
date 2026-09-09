from __future__ import annotations

import pytest
import torch

from duet.duetx.h3_reference_alignment import AlignedReferencePack, pack_aligned_images
from duet.duetx.h3_reference_compressors import (
    AlignedReferenceCompressor,
    DuetReferenceCompressor,
    GatedReferenceCompressor,
    H3ReferenceCompiler,
    ResamplerReferenceCompressor,
)
from duet.duetx.h3_reference_contracts import (
    H3ReferenceBudget,
    H3ReferenceKind,
    H3ReferenceMethod,
    H3VisualReference,
)
from duet.duetx.latent_bridge import LatentHistoryBridge

_CHECKPOINT_SHA256 = "a" * 64


def _image(
    source_id: str, ordinal: int, value: float, *, protected: bool = False
) -> H3VisualReference:
    return H3VisualReference(
        source_id,
        H3ReferenceKind.IMAGE,
        ordinal,
        torch.full((1, 24, 1, 4, 6), value, dtype=torch.float32),
        protected,
    )


def _video(frames: int = 17) -> H3VisualReference:
    latent = torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1)
    return H3VisualReference(
        "motion",
        H3ReferenceKind.VIDEO,
        0,
        latent.repeat(1, 24, 1, 4, 6),
        False,
    )


@pytest.mark.parametrize(
    "compressor",
    [GatedReferenceCompressor(), ResamplerReferenceCompressor(), DuetReferenceCompressor()],
)
def test_learned_compressors_emit_one_native_shape_core(
    compressor: AlignedReferenceCompressor,
) -> None:
    references = (_image("a", 0, 1), _image("b", 1, 2), _image("c", 2, 3))
    pack = pack_aligned_images(references)

    core = compressor.compress(pack)

    assert core.shape == references[0].latent.shape
    assert core.dtype == references[0].latent.dtype
    assert torch.isfinite(core).all()


def test_duet_is_anchor_preserving_at_initialization() -> None:
    references = (_image("a", 0, 1), _image("b", 1, 7))

    core = DuetReferenceCompressor().compress(pack_aligned_images(references))

    torch.testing.assert_close(core, references[-1].latent)


@pytest.mark.parametrize(
    "compressor",
    [GatedReferenceCompressor(), ResamplerReferenceCompressor(), DuetReferenceCompressor()],
)
def test_unavailable_padding_cannot_change_a_learned_core(
    compressor: AlignedReferenceCompressor,
) -> None:
    pack = pack_aligned_images((_image("a", 0, 1), _image("b", 1, 2)))
    poisoned_history = pack.history.clone()
    poisoned_history[:, 2:] = 1_000_000
    poisoned = AlignedReferencePack(
        poisoned_history,
        pack.availability,
        pack.source_ids,
        pack.output_kind,
        pack.padding_temporal_tokens,
    ).validate()

    expected = compressor.compress(pack)
    actual = compressor.compress(poisoned)

    torch.testing.assert_close(actual, expected)


def test_bridge_exposes_masked_per_stream_operators() -> None:
    pack = pack_aligned_images((_image("a", 0, 1), _image("b", 1, 2)))
    bridge = LatentHistoryBridge(24, operator_size=16)

    operators = bridge.encode_operators(pack.history, availability=pack.availability)

    assert operators.shape == (1, 8, 24, 16, 16)
    combined = bridge.fusion.combine_ordered_operators(tuple(operators.unbind(dim=1)))
    direct = bridge(
        pack.history,
        anchor=pack.last_available,
        availability=pack.availability,
    ).core_operator
    torch.testing.assert_close(combined, direct)


def test_native_full_is_an_exact_bypass_with_receipt() -> None:
    references = (_image("a", 0, 1), _image("b", 1, 2))
    compiler = H3ReferenceCompiler(
        H3ReferenceMethod.NATIVE_FULL,
        H3ReferenceBudget(max_visual_rows=12, max_exceptions=2),
    )

    compiled = compiler.compile_references(references)

    assert [
        block.latent is source.latent
        for block, source in zip(compiled.blocks, references, strict=True)
    ] == [
        True,
        True,
    ]
    assert compiled.receipt.input_visual_rows == 12
    assert compiled.receipt.output_visual_rows == 12
    assert compiled.receipt.compression_factor == 1
    assert compiled.receipt.checkpoint_sha256 is None
    assert compiled.receipt.cache_status == "bypass"


def test_duet_x_compiles_dense_history_and_preserves_exact_exceptions() -> None:
    references = (
        _image("world", 0, 1),
        _image("hero", 1, 2, protected=True),
        _image("prop", 2, 3),
        _image("villain", 3, 4, protected=True),
    )
    compiler = H3ReferenceCompiler(
        H3ReferenceMethod.DUET_X,
        H3ReferenceBudget(max_visual_rows=18, max_exceptions=2),
        checkpoint_sha256=_CHECKPOINT_SHA256,
    )

    compiled = compiler.compile_references(references)

    assert len(compiled.blocks) == 3
    assert compiled.blocks[0].source_ids == ("world", "prop")
    assert compiled.blocks[0].exact is False
    assert compiled.blocks[1].source_ids == ("hero",)
    assert compiled.blocks[1].exact is True
    assert compiled.blocks[1].latent is references[1].latent
    assert compiled.blocks[2].source_ids == ("villain",)
    assert compiled.receipt.input_semantic_items == 4
    assert compiled.receipt.output_semantic_items == 4
    assert compiled.receipt.input_visual_rows == 24
    assert compiled.receipt.output_visual_rows == 18
    assert compiled.receipt.trainable_parameters > 0


def test_incompatible_singleton_shapes_remain_native() -> None:
    references = (
        _image("wide", 0, 1),
        H3VisualReference(
            "tall",
            H3ReferenceKind.IMAGE,
            1,
            torch.full((1, 24, 1, 6, 4), 2.0),
            False,
        ),
    )
    compiler = H3ReferenceCompiler(
        H3ReferenceMethod.DUET,
        H3ReferenceBudget(max_visual_rows=12, max_exceptions=0),
        checkpoint_sha256=_CHECKPOINT_SHA256,
    )

    compiled = compiler.compile_references(references)

    assert len(compiled.blocks) == 2
    assert compiled.blocks[0].latent is references[0].latent
    assert compiled.blocks[1].latent is references[1].latent


def test_video_uses_ordered_eight_way_temporal_fold() -> None:
    compiler = H3ReferenceCompiler(
        H3ReferenceMethod.DUET,
        H3ReferenceBudget(max_visual_rows=18, max_exceptions=0),
        checkpoint_sha256=_CHECKPOINT_SHA256,
    )

    compiled = compiler.compile_references((_video(),))

    assert compiled.blocks[0].kind is H3ReferenceKind.VIDEO
    assert compiled.blocks[0].latent.shape == (1, 24, 3, 4, 6)
    assert compiled.receipt.padding_temporal_tokens == 7
    assert compiled.receipt.input_visual_rows == 102
    assert compiled.receipt.output_visual_rows == 18


def test_native_trimmed_keeps_source_order_within_budget() -> None:
    references = tuple(_image(f"ref-{index}", index, float(index)) for index in range(4))
    compiler = H3ReferenceCompiler(
        H3ReferenceMethod.NATIVE_TRIMMED,
        H3ReferenceBudget(max_visual_rows=13, max_exceptions=0),
    )

    compiled = compiler.compile_references(references)

    assert tuple(block.source_ids for block in compiled.blocks) == (("ref-0",), ("ref-1",))
    assert compiled.receipt.output_visual_rows == 12


def test_exceptions_only_requires_and_emits_only_protected_images() -> None:
    references = (_image("world", 0, 1), _image("hero", 1, 2, protected=True))
    compiler = H3ReferenceCompiler(
        H3ReferenceMethod.EXCEPTIONS_ONLY,
        H3ReferenceBudget(max_visual_rows=6, max_exceptions=1),
    )

    compiled = compiler.compile_references(references)

    assert len(compiled.blocks) == 1
    assert compiled.blocks[0].source_ids == ("hero",)
    assert compiled.blocks[0].exact is True
    with pytest.raises(ValueError, match="requires a protected"):
        compiler.compile_references((references[0],))


def test_compiler_rejects_budget_overflow_and_checkpoint_mismatch() -> None:
    reference = _image("hero", 0, 2, protected=True)
    with pytest.raises(ValueError, match="lowercase checkpoint"):
        H3ReferenceCompiler(
            H3ReferenceMethod.DUET_X,
            H3ReferenceBudget(max_visual_rows=6, max_exceptions=1),
            checkpoint_sha256="bad",
        )
    compiler = H3ReferenceCompiler(
        H3ReferenceMethod.DUET_X,
        H3ReferenceBudget(max_visual_rows=5, max_exceptions=1),
        checkpoint_sha256=_CHECKPOINT_SHA256,
    )
    with pytest.raises(ValueError, match="visual row budget"):
        compiler.compile_references((reference,))


def test_compiler_rejects_ambiguous_source_order() -> None:
    compiler = H3ReferenceCompiler(
        H3ReferenceMethod.NATIVE_FULL,
        H3ReferenceBudget(max_visual_rows=12, max_exceptions=0),
    )

    with pytest.raises(ValueError, match="ascending ordinals"):
        compiler.compile_references((_image("later", 2, 1), _image("earlier", 1, 2)))


def test_reference_compilation_does_not_override_torch_module_compile() -> None:
    assert H3ReferenceCompiler.compile is torch.nn.Module.compile
