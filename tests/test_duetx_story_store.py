from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from duet.duetx.cache import AuthoritativeLeaf
from duet.duetx.contracts import (
    DuetXContract,
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    RawEvidencePointer,
    SpatialLocation,
    TimeFrameRange,
)
from duet.duetx.story_contracts import ReferenceRole, StoryLibrary, StoryReference
from duet.duetx.story_memory import append_story_leaf, compose_story_blocks, empty_story_blocks
from duet.duetx.story_product_contracts import (
    CanonEntity,
    CanonPresence,
    GuideBinding,
    StoryCanon,
    StoryGenerationReceipt,
    StoryMemoryPolicy,
    StoryProductState,
)
from duet.duetx.story_store import (
    LoadedStoryRevision,
    StoryCommitPayload,
    StoryGuideBundle,
    StoryProjectStore,
    migrate_minimax_v2_product_state,
)
from duet.fusion import FusionSpec, MatrixSemigroupFusion


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _library() -> StoryLibrary:
    digest = _digest("maya")
    return StoryLibrary(
        "Coast Story",
        (
            StoryReference(
                "Maya",
                ReferenceRole.CHARACTER,
                "",
                digest,
                f"duet-story://assets/sha256/{digest}",
                (digest,),
                _digest("reference-preprocessing"),
            ),
        ),
    ).validate()


def _store(root: Path) -> StoryProjectStore:
    store = StoryProjectStore(root)
    assert store.put_asset(b"maya") == _digest("maya")
    return store


def _leaf(contract: DuetXContract, shot_index: int) -> AuthoritativeLeaf:
    local_slot = shot_index % 8
    key = LeafKey(f"shot-{shot_index}", shot_index, local_slot)
    raw_digest = _digest(f"video-{shot_index}")
    item = ExceptionItem(
        f"item-{shot_index}",
        EvidenceProvenance(
            key,
            TimeFrameRange(shot_index, shot_index + 1, shot_index, shot_index + 1),
            SpatialLocation("normalized", "bbox", (0, 0, 65_536, 65_536)),
            "story-shot-v1",
            RawEvidencePointer(f"duet-evidence://story/sha256/{raw_digest}", raw_digest, 0, 1),
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "5" * 64,
            contract.sparse.fingerprint(),
        ),
        torch.full((128,), float(shot_index), dtype=torch.float32),
        shot_index,
    )
    return AuthoritativeLeaf(
        key,
        _digest(f"source-{shot_index}"),
        0,
        torch.eye(2, dtype=torch.float32).reshape(1, 1, 2, 2),
        (item,),
    )


def _payload(
    contract: DuetXContract,
    *,
    parent: LoadedStoryRevision | None = None,
    branch_id: str = "main",
    seed: int = 101,
) -> StoryCommitPayload:
    if parent is None:
        blocks = empty_story_blocks(contract, dense_steps=1, operator_size=2)
        shot_index = 0
    else:
        blocks = parent.blocks
        shot_index = parent.state.shot_count
    blocks = append_story_leaf(
        blocks,
        shot_index=shot_index,
        leaf=_leaf(contract, shot_index),
    )
    return StoryCommitPayload(
        project_id="coast-story",
        branch_id=branch_id,
        parent=None if parent is None else parent.state,
        library=_library(),
        blocks=blocks,
        checkpoint_sha256="6" * 64,
        model_configuration_sha256="7" * 64,
        last_frame_png=f"frame-{shot_index}".encode(),
        story_context_png=f"context-{shot_index}".encode(),
        shot_metadata={"intent": "Start Story", "seed": seed},
    )


def test_publish_load_and_identical_retry_are_content_addressed(tmp_path: Path) -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    store = _store(tmp_path)
    payload = _payload(contract)

    first = store.publish(payload)
    second = store.publish(payload)
    loaded = store.load(first, contract, payload.checkpoint_sha256)

    assert first == second
    assert loaded.state == first
    assert len(loaded.blocks) == 16
    assert loaded.last_frame_png == b"frame-0"
    assert loaded.story_context_png == b"context-0"
    assert loaded.guide_bundle is None
    with pytest.raises(ValueError, match="materialized guide bundle"):
        loaded.require_guide_bundle()


def _minimax_leaf(contract: DuetXContract, shot_index: int) -> AuthoritativeLeaf:
    local_slot = shot_index % 8
    key = LeafKey(f"minimax-shot-{shot_index}", shot_index, local_slot)
    frame = f"minimax-frame-{shot_index}".encode()
    raw_digest = hashlib.sha256(frame).hexdigest()
    item = ExceptionItem(
        f"minimax-item-{shot_index}",
        EvidenceProvenance(
            key,
            TimeFrameRange(shot_index, shot_index + 1, shot_index, shot_index + 1),
            SpatialLocation("normalized", "bbox", (0, 0, 65_536, 65_536)),
            "story-shot-v1",
            RawEvidencePointer(
                f"duet-evidence://story-frame/sha256/{raw_digest}", raw_digest, 0, len(frame)
            ),
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "5" * 64,
            contract.sparse.fingerprint(),
        ),
        torch.full((24,), float(shot_index), dtype=torch.float32),
        shot_index,
    )
    return AuthoritativeLeaf(
        key,
        _digest(f"minimax-source-{shot_index}"),
        0,
        torch.eye(2, dtype=torch.float32).reshape(1, 1, 2, 2),
        (item,),
    )


def _minimax_payload(
    contract: DuetXContract, parent: LoadedStoryRevision | None = None
) -> StoryCommitPayload:
    shot_index = 0 if parent is None else parent.state.shot_count
    blocks = (
        empty_story_blocks(contract, dense_steps=1, operator_size=2)
        if parent is None
        else parent.blocks
    )
    blocks = append_story_leaf(
        blocks,
        shot_index=shot_index,
        leaf=_minimax_leaf(contract, shot_index),
    )
    memory = compose_story_blocks(
        blocks,
        MatrixSemigroupFusion(24, FusionSpec(), operator_size=2),
    )
    exception_pngs = tuple(
        f"minimax-frame-{item.provenance.leaf.source_rank}".encode() for item in memory.exceptions
    )
    return StoryCommitPayload(
        project_id="coast-story",
        branch_id="main",
        parent=None if parent is None else parent.state,
        library=_library(),
        blocks=blocks,
        checkpoint_sha256="8" * 64,
        model_configuration_sha256="9" * 64,
        last_frame_png=f"minimax-frame-{shot_index}".encode(),
        story_context_png=f"minimax-context-{shot_index}".encode(),
        shot_metadata={"memory_backend": "minimax-h3", "seed": 101},
        guide_bundle=StoryGuideBundle(
            None if shot_index < 2 else f"minimax-core-{shot_index}".encode(),
            exception_pngs,
            tuple(item.fingerprint() for item in memory.exceptions),
        ),
    )


def test_minimax_guide_bundle_round_trips_exact_exception_assets(tmp_path: Path) -> None:
    contract = DuetXContract.minimax_h3(adapter_fingerprint="a" * 64)
    store = _store(tmp_path)
    parent: LoadedStoryRevision | None = None
    for _ in range(3):
        payload = _minimax_payload(contract, parent)
        state = store.publish(payload)
        parent = store.load(state, contract, payload.checkpoint_sha256)

    assert parent is not None
    bundle = parent.require_guide_bundle()
    assert bundle.core_png == b"minimax-core-2"
    assert bundle.exception_pngs == (b"minimax-frame-2", b"minimax-frame-1")
    assert bundle.exception_fingerprints == tuple(
        item.fingerprint() for item in parent.memory.exceptions
    )
    assert store.load_asset(hashlib.sha256(b"minimax-frame-2").hexdigest()) == (b"minimax-frame-2")


def _empty_product_state() -> StoryProductState:
    return StoryProductState(
        (),
        (),
        StoryCanon(
            (
                CanonEntity(
                    "maya",
                    "Maya",
                    _digest("maya"),
                    "",
                    CanonPresence.UNKNOWN,
                    (),
                    None,
                    (),
                ),
            )
        ),
        StoryMemoryPolicy(),
    ).validate()


def _receipt() -> StoryGenerationReceipt:
    return StoryGenerationReceipt(
        prompt_sha256=_digest("prompt"),
        parent_revision_sha256=None,
        memory_backend="minimax-h3",
        checkpoint_sha256="8" * 64,
        sampler="native-res-multistep",
        selected_evidence_ids=(),
        guide_bindings=(GuideBinding("current", _digest("current")),),
        recall_decision_sha256=_digest("decision"),
    ).validate()


def test_v3_revision_round_trips_product_state_and_receipt(tmp_path: Path) -> None:
    contract = DuetXContract.minimax_h3(adapter_fingerprint="a" * 64)
    store = _store(tmp_path)
    payload = replace(
        _minimax_payload(contract),
        guide_bundle=None,
        product_state=_empty_product_state(),
        generation_receipt=_receipt(),
    )

    state = store.publish(payload)
    loaded = store.load(state, contract, payload.checkpoint_sha256)

    assert loaded.require_product_state() == payload.product_state
    assert loaded.generation_receipt == payload.generation_receipt


def test_v3_requires_product_state_and_receipt_as_one_transaction(tmp_path: Path) -> None:
    contract = DuetXContract.minimax_h3(adapter_fingerprint="a" * 64)
    payload = replace(_minimax_payload(contract), product_state=_empty_product_state())

    with pytest.raises(ValueError, match="together"):
        _store(tmp_path).publish(payload)


def test_v3_load_rejects_changed_product_sidecar(tmp_path: Path) -> None:
    contract = DuetXContract.minimax_h3(adapter_fingerprint="a" * 64)
    store = _store(tmp_path)
    payload = replace(
        _minimax_payload(contract),
        guide_bundle=None,
        product_state=_empty_product_state(),
        generation_receipt=_receipt(),
    )
    state = store.publish(payload)
    manifest = json.loads(
        (tmp_path / "revisions" / "sha256" / state.revision_sha256 / "manifest.json").read_bytes()
    )
    product = tmp_path / "products" / "sha256" / f"{manifest['product_state_sha256']}.json"
    product.write_bytes(b'{"canon": {}}')

    with pytest.raises(ValueError, match="product state"):
        store.load(state, contract, payload.checkpoint_sha256)


def test_v2_migration_preserves_unknown_evidence_as_legacy(tmp_path: Path) -> None:
    contract = DuetXContract.minimax_h3(adapter_fingerprint="a" * 64)
    store = _store(tmp_path)
    payload = _minimax_payload(contract)
    state = store.publish(payload)
    loaded = store.load(state, contract, payload.checkpoint_sha256)

    product = migrate_minimax_v2_product_state(loaded)

    assert tuple(entity.entity_id for entity in product.canon.entities) == ("maya",)
    assert all(record.kind.value == "legacy" for record in product.evidence_records)
    assert all(record.frame_index is None for record in product.evidence_records)


def test_minimax_guide_bundle_rejects_tampered_exception_asset(tmp_path: Path) -> None:
    contract = DuetXContract.minimax_h3(adapter_fingerprint="a" * 64)
    store = _store(tmp_path)
    payload = _minimax_payload(contract)
    state = store.publish(payload)
    bundle = payload.guide_bundle
    assert bundle is not None
    digest = hashlib.sha256(bundle.exception_pngs[0]).hexdigest()
    (tmp_path / "assets" / "sha256" / digest).write_bytes(b"corrupted-frame")

    with pytest.raises(ValueError, match="SHA-256"):
        store.load(state, contract, payload.checkpoint_sha256)


def test_minimax_guide_bundle_must_match_memory_topk(tmp_path: Path) -> None:
    contract = DuetXContract.minimax_h3(adapter_fingerprint="a" * 64)
    store = _store(tmp_path)
    payload = _minimax_payload(contract)
    bundle = payload.guide_bundle
    assert bundle is not None

    with pytest.raises(ValueError, match="fingerprints do not match"):
        store.publish(
            replace(
                payload,
                guide_bundle=replace(bundle, exception_fingerprints=("f" * 64,)),
            )
        )


def test_minimax_guide_core_presence_tracks_nonexception_history(tmp_path: Path) -> None:
    contract = DuetXContract.minimax_h3(adapter_fingerprint="a" * 64)
    store = _store(tmp_path)
    parent: LoadedStoryRevision | None = None
    for _ in range(2):
        payload = _minimax_payload(contract, parent)
        state = store.publish(payload)
        parent = store.load(state, contract, payload.checkpoint_sha256)
    payload = _minimax_payload(contract, parent)
    bundle = payload.guide_bundle
    assert bundle is not None
    assert bundle.core_png is not None

    with pytest.raises(ValueError, match="core presence"):
        store.publish(replace(payload, guide_bundle=replace(bundle, core_png=None)))


def test_two_children_share_parent_without_mutating_each_other(tmp_path: Path) -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    store = _store(tmp_path)
    parent_state = store.publish(_payload(contract))
    parent = store.load(parent_state, contract, "6" * 64)

    take_a = store.publish(_payload(contract, parent=parent, branch_id="take-a", seed=101))
    take_b = store.publish(_payload(contract, parent=parent, branch_id="take-b", seed=202))

    assert take_a.parent_revision_sha256 == parent_state.revision_sha256
    assert take_b.parent_revision_sha256 == parent_state.revision_sha256
    assert take_a.revision_sha256 != take_b.revision_sha256
    assert store.load(parent_state, contract, "6" * 64).state == parent_state


def test_invalid_payload_publishes_no_revision(tmp_path: Path) -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    store = _store(tmp_path)
    payload = _payload(contract)
    invalid = StoryCommitPayload(
        project_id=payload.project_id,
        branch_id=payload.branch_id,
        parent=payload.parent,
        library=payload.library,
        blocks=payload.blocks,
        checkpoint_sha256="bad",
        model_configuration_sha256=payload.model_configuration_sha256,
        last_frame_png=payload.last_frame_png,
        story_context_png=payload.story_context_png,
        shot_metadata=payload.shot_metadata,
    )

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        store.publish(invalid)

    assert list((tmp_path / "revisions" / "sha256").iterdir()) == []
    assert list((tmp_path / "staging").iterdir()) == []


def test_load_rejects_changed_revision_manifest(tmp_path: Path) -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    store = _store(tmp_path)
    state = store.publish(_payload(contract))
    manifest = tmp_path / "revisions" / "sha256" / state.revision_sha256 / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="revision manifest SHA-256"):
        store.load(state, contract, "6" * 64)


@pytest.mark.parametrize("corruption", ["bytes", "symlink", "hardlink"])
def test_streamed_asset_verification_rejects_corruption(tmp_path: Path, corruption: str) -> None:
    import os

    store = StoryProjectStore(tmp_path / "story")
    digest = store.put_asset(b"immutable movie bytes")
    path = store.verified_asset_path(digest)
    if corruption == "bytes":
        path.write_bytes(b"different movie bytes")
    elif corruption == "symlink":
        source = tmp_path / "outside"
        source.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(source)
    else:
        os.link(path, tmp_path / "second-link")
    with pytest.raises(ValueError, match=r"SHA-256|unavailable|regular file"):
        store.verified_asset_path(digest)


def test_streamed_asset_verification_does_not_use_buffered_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StoryProjectStore(tmp_path / "story")
    digest = store.put_asset(b"a" * (3 * 1024 * 1024))

    def reject_read_bytes(*args: object) -> bytes:
        raise AssertionError("asset must be hashed incrementally")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    assert store.verified_asset_path(digest).name == digest
