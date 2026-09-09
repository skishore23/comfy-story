from __future__ import annotations

from pathlib import Path

import torch

import comfy_story as duetx
from comfy_story.fusion import FusionSpec, MatrixSemigroupFusion

EXPECTED_PUBLIC_API = {
    "FusionSpec",
    "MatrixSemigroupFusion",
    "TamariFusion",
    "prepare_story_generation",
    "train",
    "load_runtime_snapshot",
    "LocalExceptionScorer",
    "save_runtime_snapshot",
    "LTXHistoryGuidePayload",
    "Decision1Config",
    "CacheUpdate",
    "AuthoritativeLeaf",
    "TrainingModules",
    "RawEvidencePointer",
    "label_influence",
    "CheckpointFingerprints",
    "load_config",
    "quantize_half_up",
    "InfluenceRecord",
    "HistorySample",
    "TeacherOutput",
    "StoryMemoryRuntime",
    "StorySampler",
    "SparseMemoryContract",
    "Decision1Losses",
    "MethodBatch",
    "StoryMemoryIdentity",
    "materialize_state",
    "MaterializedDuetXMemory",
    "Selector",
    "commit_story_generation",
    "DuetStoryStateRef",
    "LTXLatentGuide",
    "tensor_sha256",
    "DuetXContract",
    "leaf_state",
    "load_influence_shard",
    "StoryLibrary",
    "StoryLTXRuntime",
    "TrainingCursor",
    "train_step",
    "write_influence_shard",
    "audit_manifest",
    "EvidenceProvenance",
    "ShotIntent",
    "TeacherBackend",
    "Coverage",
    "TimeFrameRange",
    "load_bridge_checkpoint",
    "merge_states",
    "StoryGenerationRequest",
    "Method",
    "canonicalize_items",
    "HistoryBatch",
    "HistoryItem",
    "LatentGuideBackend",
    "select_candidates",
    "LeafKey",
    "DuetXProductTree",
    "merge_top_k",
    "save_bridge_checkpoint",
    "build_method",
    "StoryCommitResult",
    "item_rank",
    "run_method",
    "StoryCommitRequest",
    "MethodOutput",
    "HistoryMethod",
    "identity_state",
    "StoryMemoryBackend",
    "ReferenceRole",
    "SpatialLocation",
    "DuetXState",
    "ExceptionItem",
    "GuidePayloadMode",
    "StoryReference",
    "select_top_k",
    "apply_history_guides",
}


def _item(contract: duetx.DuetXContract, key: duetx.LeafKey) -> duetx.ExceptionItem:
    provenance = duetx.EvidenceProvenance(
        leaf=key,
        time=duetx.TimeFrameRange(100 + key.slot, 101 + key.slot, key.slot, key.slot + 1),
        spatial=duetx.SpatialLocation("normalized", "bbox", (0, 0, 4, 4)),
        event_type="rare-detail-v1",
        raw=duetx.RawEvidencePointer(
            f"duet-evidence://registry/sha256/{f'{key.slot:x}' * 64}",
            f"{key.slot:x}" * 64,
            0,
            1,
        ),
        source_registry_sha256="1" * 64,
        preprocessing_sha256="2" * 64,
        vae_sha256="3" * 64,
        adapter_sha256="4" * 64,
        scorer_sha256="5" * 64,
        memory_contract_sha256=contract.sparse.fingerprint(),
    )
    return duetx.ExceptionItem(
        f"item-{key.slot}",
        provenance,
        torch.full((128,), float(key.slot), dtype=torch.float32),
        key.slot,
        key.slot in {2, 5},
    )


def _leaf(
    contract: duetx.DuetXContract, slot: int, *, revision: int = 0
) -> duetx.AuthoritativeLeaf:
    key = duetx.LeafKey(f"camera-{slot}", slot, slot)
    operators = (
        torch.tensor([[1.0, slot + 1.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [slot + 1.0, 1.0]]),
    )
    return duetx.AuthoritativeLeaf(
        key=key,
        source_fingerprint=f"{slot:x}" * 64,
        revision=revision,
        dense_operator=operators[slot % 2].reshape(1, 1, 2, 2),
        candidates=(_item(contract, key),),
    )


def test_public_api_is_narrow_and_does_not_export_refmax_formats() -> None:
    assert set(duetx.__all__) == EXPECTED_PUBLIC_API
    assert all("refmax" not in name.lower() for name in duetx.__all__)


def test_public_api_runs_the_complete_structural_memory_canary(tmp_path: Path) -> None:
    contract = duetx.DuetXContract.default(decision1_fingerprint="1" * 64)
    leaves = tuple(_leaf(contract, slot) for slot in range(8))
    tree = duetx.DuetXProductTree(contract, 1, 2, leaves)
    expected_dense = leaves[7].dense_operator
    for leaf in reversed(leaves[:-1]):
        expected_dense = expected_dense @ leaf.dense_operator

    assert torch.equal(tree.root.dense_operator, expected_dense)
    assert len(tree.root.exceptions) == contract.sparse.capacity
    assert tuple(item.item_id for item in tree.root.exceptions) == ("item-5", "item-2")
    assert all(item.pinned for item in tree.root.exceptions)

    fusion = MatrixSemigroupFusion(128, FusionSpec(), operator_size=2)
    materialized = duetx.materialize_state(tree.root, fusion)
    assert materialized.item_ids == ("item-2", "item-5")
    assert tuple(provenance.leaf.slot for provenance in materialized.provenance) == (2, 5)
    assert torch.equal(materialized.exception_embeddings[0], leaves[2].candidates[0].embedding)
    assert torch.equal(materialized.exception_embeddings[1], leaves[5].candidates[0].embedding)

    path = tmp_path / "runtime.pt"
    duetx.save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)
    restored = duetx.load_runtime_snapshot(
        path,
        expected_contract=contract,
        expected_model_checkpoint_sha256="0" * 64,
    )
    assert tuple(item.item_id for item in restored.root.exceptions) == ("item-5", "item-2")
    assert tuple(item.fingerprint() for item in restored.root.exceptions) == tuple(
        item.fingerprint() for item in tree.root.exceptions
    )
    assert torch.equal(restored.root.dense_operator, tree.root.dense_operator)

    restored.replace(_leaf(contract, 3, revision=1))
    restored.delete(leaves[6].key, expected_revision=0)
    cold = duetx.DuetXProductTree(
        contract,
        1,
        2,
        tuple(leaf for leaf in restored.leaves if leaf is not None),
    )
    assert torch.equal(restored.root.dense_operator, cold.root.dense_operator)
    assert tuple(item.item_id for item in restored.root.exceptions) == tuple(
        item.item_id for item in cold.root.exceptions
    )
    assert tuple(item.fingerprint() for item in restored.root.exceptions) == tuple(
        item.fingerprint() for item in cold.root.exceptions
    )
