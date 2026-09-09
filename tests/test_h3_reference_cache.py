from __future__ import annotations

from pathlib import Path

import pytest
import torch

from comfy_story.h3_reference_alignment import pack_aligned_images
from comfy_story.h3_reference_cache import CompiledReferenceCache, H3DuetProductCache
from comfy_story.h3_reference_compressors import H3ReferenceCompiler
from comfy_story.h3_reference_contracts import (
    H3ReferenceBudget,
    H3ReferenceKind,
    H3ReferenceMethod,
    H3VisualReference,
)
from comfy_story.latent_bridge import LatentHistoryBridge


def _image(source_id: str, ordinal: int, value: float) -> H3VisualReference:
    return H3VisualReference(
        source_id,
        H3ReferenceKind.IMAGE,
        ordinal,
        torch.full((1, 24, 1, 4, 6), value, dtype=torch.float32),
        False,
    )


def test_compiled_pack_cache_is_content_addressed(tmp_path: Path) -> None:
    references = (_image("a", 0, 1), _image("b", 1, 2))
    budget = H3ReferenceBudget(12, 0)
    expected = H3ReferenceCompiler(H3ReferenceMethod.NATIVE_FULL, budget).compile_references(
        references
    )
    cache = CompiledReferenceCache(tmp_path)
    key = cache.key_for(
        references,
        H3ReferenceMethod.NATIVE_FULL,
        budget,
        checkpoint_sha256=None,
    )

    cache.store(key, expected)
    actual = cache.load(key)

    assert actual is not None
    assert actual.receipt == expected.receipt
    assert actual.budget == expected.budget
    assert actual.blocks[0].source_ids == expected.blocks[0].source_ids
    torch.testing.assert_close(actual.blocks[0].latent, expected.blocks[0].latent)


def test_cache_key_changes_with_method_budget_content_and_protection(tmp_path: Path) -> None:
    cache = CompiledReferenceCache(tmp_path)
    reference = _image("a", 0, 1)
    base = cache.key_for(
        (reference,),
        H3ReferenceMethod.NATIVE_FULL,
        H3ReferenceBudget(6, 0),
        checkpoint_sha256=None,
    )

    changed_content = cache.key_for(
        (_image("a", 0, 2),),
        H3ReferenceMethod.NATIVE_FULL,
        H3ReferenceBudget(6, 0),
        checkpoint_sha256=None,
    )
    protected = H3VisualReference(
        reference.source_id,
        reference.kind,
        reference.ordinal,
        reference.latent,
        True,
    )
    changed_protection = cache.key_for(
        (protected,),
        H3ReferenceMethod.NATIVE_FULL,
        H3ReferenceBudget(6, 0),
        checkpoint_sha256=None,
    )
    changed_budget = cache.key_for(
        (reference,),
        H3ReferenceMethod.NATIVE_FULL,
        H3ReferenceBudget(7, 0),
        checkpoint_sha256=None,
    )

    assert len({base, changed_content, changed_protection, changed_budget}) == 4
    assert cache.load("f" * 64) is None


def test_cache_rejects_tensor_drift(tmp_path: Path) -> None:
    references = (_image("a", 0, 1),)
    budget = H3ReferenceBudget(6, 0)
    context = H3ReferenceCompiler(H3ReferenceMethod.NATIVE_FULL, budget).compile_references(
        references
    )
    cache = CompiledReferenceCache(tmp_path)
    key = cache.key_for(
        references,
        H3ReferenceMethod.NATIVE_FULL,
        budget,
        checkpoint_sha256=None,
    )
    path = cache.store(key, context)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["blocks"][0]["latent"][0, 0, 0, 0, 0] = 20
    torch.save(payload, path)

    with pytest.raises(ValueError, match="output hashes"):
        cache.load(key)


def test_one_duet_leaf_replacement_recomputes_log2_path() -> None:
    references = tuple(_image(f"ref-{index}", index, float(index)) for index in range(8))
    pack = pack_aligned_images(references)
    bridge = LatentHistoryBridge(24, operator_size=16)
    cache = H3DuetProductCache.build(bridge, pack)

    replacement = torch.ones((1, 24, 1, 4, 6), dtype=torch.float32)
    update = cache.replace(slot=5, latent=replacement, revision=1)

    assert update.products_recomputed == 3
    torch.testing.assert_close(cache.root_operator, cache.full_rebuild_root())
    assert cache.materialize().shape == replacement.shape


def test_product_cache_rejects_stale_or_padding_replacement() -> None:
    pack = pack_aligned_images((_image("a", 0, 1), _image("b", 1, 2)))
    cache = H3DuetProductCache.build(LatentHistoryBridge(24), pack)
    replacement = torch.ones_like(pack.history[:, 0])

    cache.replace(slot=1, latent=replacement, revision=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        cache.replace(slot=1, latent=replacement, revision=1)
    with pytest.raises(ValueError, match="unavailable padding"):
        cache.replace(slot=5, latent=replacement, revision=1)
