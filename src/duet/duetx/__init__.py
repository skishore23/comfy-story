"""Comfy Story memory surface with stable historical module paths."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AuthoritativeLeaf": ("duet.duetx.cache", "AuthoritativeLeaf"),
    "CacheUpdate": ("duet.duetx.cache", "CacheUpdate"),
    "CheckpointFingerprints": ("duet.duetx.checkpoint", "CheckpointFingerprints"),
    "Coverage": ("duet.duetx.contracts", "Coverage"),
    "Decision1Config": ("duet.duetx.config", "Decision1Config"),
    "Decision1Losses": ("duet.duetx.losses", "Decision1Losses"),
    "DuetXContract": ("duet.duetx.contracts", "DuetXContract"),
    "DuetStoryStateRef": ("duet.duetx.story_contracts", "DuetStoryStateRef"),
    "DuetXProductTree": ("duet.duetx.cache", "DuetXProductTree"),
    "DuetXState": ("duet.duetx.state", "DuetXState"),
    "EvidenceProvenance": ("duet.duetx.contracts", "EvidenceProvenance"),
    "ExceptionItem": ("duet.duetx.contracts", "ExceptionItem"),
    "GuidePayloadMode": ("duet.duetx.guide_boundary", "GuidePayloadMode"),
    "HistoryBatch": ("duet.duetx.data", "HistoryBatch"),
    "HistoryItem": ("duet.duetx.data", "HistoryItem"),
    "HistoryMethod": ("duet.duetx.methods", "HistoryMethod"),
    "HistorySample": ("duet.duetx.data", "HistorySample"),
    "InfluenceRecord": ("duet.duetx.teacher", "InfluenceRecord"),
    "LTXHistoryGuidePayload": ("duet.duetx.guide_boundary", "LTXHistoryGuidePayload"),
    "LTXLatentGuide": ("duet.duetx.guide_boundary", "LTXLatentGuide"),
    "LatentGuideBackend": ("duet.duetx.guide_boundary", "LatentGuideBackend"),
    "LeafKey": ("duet.duetx.contracts", "LeafKey"),
    "LocalExceptionScorer": ("duet.duetx.losses", "LocalExceptionScorer"),
    "MaterializedDuetXMemory": ("duet.duetx.state", "MaterializedDuetXMemory"),
    "Method": ("duet.duetx.config", "Method"),
    "MethodBatch": ("duet.duetx.methods", "MethodBatch"),
    "MethodOutput": ("duet.duetx.methods", "MethodOutput"),
    "RawEvidencePointer": ("duet.duetx.contracts", "RawEvidencePointer"),
    "ReferenceRole": ("duet.duetx.story_contracts", "ReferenceRole"),
    "Selector": ("duet.duetx.config", "Selector"),
    "ShotIntent": ("duet.duetx.story_contracts", "ShotIntent"),
    "SparseMemoryContract": ("duet.duetx.contracts", "SparseMemoryContract"),
    "SpatialLocation": ("duet.duetx.contracts", "SpatialLocation"),
    "StoryLibrary": ("duet.duetx.story_contracts", "StoryLibrary"),
    "StoryMemoryBackend": ("duet.duetx.story_memory_backend", "StoryMemoryBackend"),
    "StoryMemoryIdentity": ("duet.duetx.story_memory_backend", "StoryMemoryIdentity"),
    "StoryMemoryRuntime": ("duet.duetx.story_memory_backend", "StoryMemoryRuntime"),
    "StorySampler": ("duet.duetx.story_memory_backend", "StorySampler"),
    "StoryCommitRequest": ("duet.duetx.story_service", "StoryCommitRequest"),
    "StoryCommitResult": ("duet.duetx.story_service", "StoryCommitResult"),
    "StoryGenerationRequest": ("duet.duetx.story_service", "StoryGenerationRequest"),
    "StoryLTXRuntime": ("duet.duetx.story_service", "StoryLTXRuntime"),
    "StoryReference": ("duet.duetx.story_contracts", "StoryReference"),
    "TeacherBackend": ("duet.duetx.teacher", "TeacherBackend"),
    "TeacherOutput": ("duet.duetx.teacher", "TeacherOutput"),
    "TimeFrameRange": ("duet.duetx.contracts", "TimeFrameRange"),
    "TrainingCursor": ("duet.duetx.checkpoint", "TrainingCursor"),
    "TrainingModules": ("duet.duetx.checkpoint", "TrainingModules"),
    "apply_history_guides": ("duet.duetx.guide_boundary", "apply_history_guides"),
    "audit_manifest": ("duet.duetx.data", "audit_manifest"),
    "build_method": ("duet.duetx.methods", "build_method"),
    "canonicalize_items": ("duet.duetx.selection", "canonicalize_items"),
    "identity_state": ("duet.duetx.state", "identity_state"),
    "item_rank": ("duet.duetx.selection", "item_rank"),
    "label_influence": ("duet.duetx.teacher", "label_influence"),
    "load_influence_shard": ("duet.duetx.teacher", "load_influence_shard"),
    "leaf_state": ("duet.duetx.state", "leaf_state"),
    "load_bridge_checkpoint": ("duet.duetx.checkpoint", "load_bridge_checkpoint"),
    "load_config": ("duet.duetx.config", "load_config"),
    "load_runtime_snapshot": ("duet.duetx.snapshot", "load_runtime_snapshot"),
    "materialize_state": ("duet.duetx.state", "materialize_state"),
    "merge_states": ("duet.duetx.state", "merge_states"),
    "merge_top_k": ("duet.duetx.selection", "merge_top_k"),
    "prepare_story_generation": ("duet.duetx.story_service", "prepare_story_generation"),
    "quantize_half_up": ("duet.duetx.teacher", "quantize_half_up"),
    "run_method": ("duet.duetx.methods", "run_method"),
    "save_bridge_checkpoint": ("duet.duetx.checkpoint", "save_bridge_checkpoint"),
    "save_runtime_snapshot": ("duet.duetx.snapshot", "save_runtime_snapshot"),
    "select_candidates": ("duet.duetx.selection", "select_candidates"),
    "select_top_k": ("duet.duetx.selection", "select_top_k"),
    "tensor_sha256": ("duet.duetx.contracts", "tensor_sha256"),
    "commit_story_generation": ("duet.duetx.story_service", "commit_story_generation"),
    "train": ("duet.duetx.training", "train"),
    "train_step": ("duet.duetx.training", "train_step"),
    "write_influence_shard": ("duet.duetx.teacher", "write_influence_shard"),
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
