"""Comfy Story memory surface with stable historical module paths."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "FusionSpec": ("comfy_story.fusion", "FusionSpec"),
    "MatrixSemigroupFusion": ("comfy_story.fusion", "MatrixSemigroupFusion"),
    "TamariFusion": ("comfy_story.fusion", "TamariFusion"),
    "AuthoritativeLeaf": ("comfy_story.cache", "AuthoritativeLeaf"),
    "CacheUpdate": ("comfy_story.cache", "CacheUpdate"),
    "CheckpointFingerprints": ("comfy_story.checkpoint", "CheckpointFingerprints"),
    "Coverage": ("comfy_story.contracts", "Coverage"),
    "Decision1Config": ("comfy_story.config", "Decision1Config"),
    "Decision1Losses": ("comfy_story.losses", "Decision1Losses"),
    "DuetXContract": ("comfy_story.contracts", "DuetXContract"),
    "DuetStoryStateRef": ("comfy_story.story_contracts", "DuetStoryStateRef"),
    "DuetXProductTree": ("comfy_story.cache", "DuetXProductTree"),
    "DuetXState": ("comfy_story.state", "DuetXState"),
    "EvidenceProvenance": ("comfy_story.contracts", "EvidenceProvenance"),
    "ExceptionItem": ("comfy_story.contracts", "ExceptionItem"),
    "GuidePayloadMode": ("comfy_story.guide_boundary", "GuidePayloadMode"),
    "HistoryBatch": ("comfy_story.data", "HistoryBatch"),
    "HistoryItem": ("comfy_story.data", "HistoryItem"),
    "HistoryMethod": ("comfy_story.methods", "HistoryMethod"),
    "HistorySample": ("comfy_story.data", "HistorySample"),
    "InfluenceRecord": ("comfy_story.teacher", "InfluenceRecord"),
    "LTXHistoryGuidePayload": ("comfy_story.guide_boundary", "LTXHistoryGuidePayload"),
    "LTXLatentGuide": ("comfy_story.guide_boundary", "LTXLatentGuide"),
    "LatentGuideBackend": ("comfy_story.guide_boundary", "LatentGuideBackend"),
    "LeafKey": ("comfy_story.contracts", "LeafKey"),
    "LocalExceptionScorer": ("comfy_story.losses", "LocalExceptionScorer"),
    "MaterializedDuetXMemory": ("comfy_story.state", "MaterializedDuetXMemory"),
    "Method": ("comfy_story.config", "Method"),
    "MethodBatch": ("comfy_story.methods", "MethodBatch"),
    "MethodOutput": ("comfy_story.methods", "MethodOutput"),
    "RawEvidencePointer": ("comfy_story.contracts", "RawEvidencePointer"),
    "ReferenceRole": ("comfy_story.story_contracts", "ReferenceRole"),
    "Selector": ("comfy_story.config", "Selector"),
    "ShotIntent": ("comfy_story.story_contracts", "ShotIntent"),
    "SparseMemoryContract": ("comfy_story.contracts", "SparseMemoryContract"),
    "SpatialLocation": ("comfy_story.contracts", "SpatialLocation"),
    "StoryLibrary": ("comfy_story.story_contracts", "StoryLibrary"),
    "StoryMemoryBackend": ("comfy_story.story_memory_backend", "StoryMemoryBackend"),
    "StoryMemoryIdentity": ("comfy_story.story_memory_backend", "StoryMemoryIdentity"),
    "StoryMemoryRuntime": ("comfy_story.story_memory_backend", "StoryMemoryRuntime"),
    "StorySampler": ("comfy_story.story_memory_backend", "StorySampler"),
    "StoryCommitRequest": ("comfy_story.story_service", "StoryCommitRequest"),
    "StoryCommitResult": ("comfy_story.story_service", "StoryCommitResult"),
    "StoryGenerationRequest": ("comfy_story.story_service", "StoryGenerationRequest"),
    "StoryLTXRuntime": ("comfy_story.story_service", "StoryLTXRuntime"),
    "StoryReference": ("comfy_story.story_contracts", "StoryReference"),
    "TeacherBackend": ("comfy_story.teacher", "TeacherBackend"),
    "TeacherOutput": ("comfy_story.teacher", "TeacherOutput"),
    "TimeFrameRange": ("comfy_story.contracts", "TimeFrameRange"),
    "TrainingCursor": ("comfy_story.checkpoint", "TrainingCursor"),
    "TrainingModules": ("comfy_story.checkpoint", "TrainingModules"),
    "apply_history_guides": ("comfy_story.guide_boundary", "apply_history_guides"),
    "audit_manifest": ("comfy_story.data", "audit_manifest"),
    "build_method": ("comfy_story.methods", "build_method"),
    "canonicalize_items": ("comfy_story.selection", "canonicalize_items"),
    "identity_state": ("comfy_story.state", "identity_state"),
    "item_rank": ("comfy_story.selection", "item_rank"),
    "label_influence": ("comfy_story.teacher", "label_influence"),
    "load_influence_shard": ("comfy_story.teacher", "load_influence_shard"),
    "leaf_state": ("comfy_story.state", "leaf_state"),
    "load_bridge_checkpoint": ("comfy_story.checkpoint", "load_bridge_checkpoint"),
    "load_config": ("comfy_story.config", "load_config"),
    "load_runtime_snapshot": ("comfy_story.snapshot", "load_runtime_snapshot"),
    "materialize_state": ("comfy_story.state", "materialize_state"),
    "merge_states": ("comfy_story.state", "merge_states"),
    "merge_top_k": ("comfy_story.selection", "merge_top_k"),
    "prepare_story_generation": ("comfy_story.story_service", "prepare_story_generation"),
    "quantize_half_up": ("comfy_story.teacher", "quantize_half_up"),
    "run_method": ("comfy_story.methods", "run_method"),
    "save_bridge_checkpoint": ("comfy_story.checkpoint", "save_bridge_checkpoint"),
    "save_runtime_snapshot": ("comfy_story.snapshot", "save_runtime_snapshot"),
    "select_candidates": ("comfy_story.selection", "select_candidates"),
    "select_top_k": ("comfy_story.selection", "select_top_k"),
    "tensor_sha256": ("comfy_story.contracts", "tensor_sha256"),
    "commit_story_generation": ("comfy_story.story_service", "commit_story_generation"),
    "train": ("comfy_story.training", "train"),
    "train_step": ("comfy_story.training", "train_step"),
    "write_influence_shard": ("comfy_story.teacher", "write_influence_shard"),
}

__all__ = tuple(sorted(_EXPORTS))


def __getattr__(name: str) -> Any:
    """Resolve one public symbol only when a caller requests it."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
