from __future__ import annotations

import hashlib
import io
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from comfy_story.memory.bridge import H3LatentHistoryBridge
from comfy_story.memory.checkpoint import build_memory_modules, save_minimax_h3_runtime_checkpoint
from comfy_story.memory.health import history_effect_report
from comfy_story.memory.service import _parent_blocks
from comfy_story.memory.settings import InspectionCodec, MemoryConfiguration
from comfy_story.story_contracts import ReferenceRole, ShotIntent, StoryLibrary, StoryReference
from comfy_story.story_native_archive import NativeReferenceArchive
from comfy_story.story_native_service import NativeStoryCommitResult, commit_native_story_generation
from comfy_story.story_product_contracts import MemoryAction, StoryMemoryCommand
from comfy_story.story_service import (
    PreparedStoryGeneration,
    StoryCommitRequest,
    StoryGenerationRequest,
    prepare_story_generation,
)
from comfy_story.story_store import StoryProjectStore


class PixelCodec:
    """Deterministic test codec; this fixture makes no H3 rendering claim."""

    def encode_frame(self, frame: np.ndarray[Any, Any]) -> torch.Tensor:
        intensity = float(frame.mean()) / 255
        base = torch.linspace(-1, 1, 24)
        features = base + intensity * base.square()
        return features.reshape(1, 24, 1, 1, 1).expand(1, 24, 1, 2, 2).contiguous()

    def decode_frame(self, latent: torch.Tensor) -> np.ndarray[Any, Any]:
        rgb = latent[0, :3].mean(dim=(1, 2, 3)).sigmoid().mul(255).round().to(torch.uint8).numpy()
        return np.broadcast_to(rgb, (384, 384, 3)).copy()

    def close(self) -> None:
        pass


def _config(path: Path, *, active: bool = True) -> MemoryConfiguration:
    modules = build_memory_modules()
    if active:
        bridge = cast(H3LatentHistoryBridge, modules.bridge)
        with torch.no_grad():
            bridge.residual.weight.copy_(torch.eye(24) * 0.1)
    save_minimax_h3_runtime_checkpoint(
        path,
        modules,
        foundation_sha256="1" * 64,
        model_configuration_sha256="2" * 64,
        optimizer_steps=2000,
        training_trace=({"optimizer_step": 2000, "fixture": "synthetic-architecture-test"},),
    )
    return MemoryConfiguration(
        str(path), hashlib.sha256(path.read_bytes()).hexdigest(), "1" * 64, "2" * 64, "3" * 64
    )


@pytest.fixture
def setup(tmp_path: Path) -> tuple[StoryProjectStore, StoryGenerationRequest]:
    config = _config(tmp_path / "memory.pt")
    store = StoryProjectStore(tmp_path / "story")
    image = torch.full((1, 8, 12, 3), 0.3)
    encoded = io.BytesIO()
    Image.fromarray(np.full((8, 12, 3), 76, dtype=np.uint8)).save(encoded, format="PNG")
    digest = store.put_asset(encoded.getvalue())
    library = StoryLibrary(
        "Boat",
        (
            StoryReference(
                "Boat",
                ReferenceRole.PROP,
                "Blue paper",
                digest,
                f"comfy-story://assets/sha256/{digest}",
                (digest,),
                "4" * 64,
            ),
        ),
    )
    request = StoryGenerationRequest(
        intent=ShotIntent.START_STORY,
        project_id="boat",
        branch_id="main",
        previous_story=None,
        previous_frame=None,
        world_frame=image,
        library=library,
        reference_images={"Boat": image},
        prompt="@Boat drifts.",
        shot_length_seconds=5,
        variation=7,
        model_configuration_sha256="5" * 64,
        associative_memory=config,
    )
    return store, request


def _commit(store: StoryProjectStore, prepared: PreparedStoryGeneration) -> NativeStoryCommitResult:
    config = prepared.request.associative_memory
    assert config is not None
    runtime = config.load(PixelCodec())
    try:
        execution = store.put_asset(f"shot-{prepared.parent_shot_count}".encode())
        prepared = replace(prepared, request=replace(prepared.request, execution_sha256=execution))
        video = store.put_asset(f"test video {prepared.parent_shot_count}".encode())
        images = torch.stack([torch.full((8, 12, 3), value) for value in (0.1, 0.8, 0.3)])
        return commit_native_story_generation(
            StoryCommitRequest(prepared, images, video, f"comfy-evidence://story/sha256/{video}"),
            store=store,
            memory_runtime=runtime,
        )
    finally:
        runtime.close()


def test_associative_chain_reopens_snapshots_and_uses_history(
    setup: tuple[StoryProjectStore, StoryGenerationRequest],
) -> None:
    store, request = setup
    first = _commit(store, prepare_story_generation(request, store=store))
    reopened = NativeReferenceArchive(StoryProjectStore(store.root)).load(
        first.state, model_configuration_sha256=request.model_configuration_sha256
    )
    assert "associative_memory" in reopened.shot_metadata
    child = replace(
        request,
        intent=ShotIntent.NEXT_SHOT,
        previous_story=reopened.state,
        previous_frame=first.last_frame,
    )
    prepared = prepare_story_generation(child, store=store)
    assert prepared.visual_roles[-1] == "associative-history"
    assert prepared.recall_decision is not None
    assert prepared.recall_decision.guide_bindings[-1].role == "associative-history"
    second = _commit(store, prepared)
    assert second.state.shot_count == 2
    assert second.state.parent_revision_sha256 == first.state.revision_sha256
    assert (
        NativeReferenceArchive(store)
        .load(first.state, model_configuration_sha256=request.model_configuration_sha256)
        .shot_metadata
        == reopened.shot_metadata
    )


@pytest.mark.parametrize(
    ("composition", "profile"),
    [("New composition", "Reference shot"), ("Continue frame", "Animate frame")],
)
def test_memory_does_not_leak_scene_guides_into_cuts_or_frame_animation(
    setup: tuple[StoryProjectStore, StoryGenerationRequest], composition: str, profile: str
) -> None:
    store, request = setup
    first = _commit(store, prepare_story_generation(request, store=store))
    child = replace(
        request,
        intent=ShotIntent.NEXT_SHOT,
        previous_story=first.state,
        previous_frame=first.last_frame,
        composition=composition,
        render_profile=profile,
    )
    assert "associative-history" not in prepare_story_generation(child, store=store).visual_roles


def test_forget_excludes_dense_leaf_and_survives_later_shots(
    setup: tuple[StoryProjectStore, StoryGenerationRequest],
) -> None:
    store, request = setup
    first = _commit(store, prepare_story_generation(request, store=store))
    target = first.loaded_revision.product_state.evidence_records[0].evidence_id
    command = StoryMemoryCommand(MemoryAction.FORGET, first.state.revision_sha256, target)
    child = replace(
        request,
        intent=ShotIntent.NEXT_SHOT,
        previous_story=first.state,
        previous_frame=first.last_frame,
        memory_commands=(command,),
    )
    prepared = prepare_story_generation(child, store=store)
    assert "associative-history" not in prepared.visual_roles
    second = _commit(store, prepared)
    config = request.associative_memory
    assert config is not None
    runtime = config.load(InspectionCodec())
    try:
        blocks = _parent_blocks(second.loaded_revision, config, store, runtime.identity.contract)
        leaf = blocks[0].leaves[0]
        assert leaf is not None
        torch.testing.assert_close(
            leaf.dense_operator, torch.eye(16).reshape(1, 1, 16, 16).expand(1, 4, 16, 16)
        )
        original = _parent_blocks(first.loaded_revision, config, store, runtime.identity.contract)[
            0
        ].leaves[0]
        assert original is not None
        assert not torch.equal(original.dense_operator, leaf.dense_operator)
    finally:
        runtime.close()
    third = _commit(
        store,
        prepare_story_generation(
            replace(
                child,
                previous_story=second.state,
                previous_frame=second.last_frame,
                memory_commands=(),
            ),
            store=store,
        ),
    )
    assert target in third.loaded_revision.product_state.policy.tombstoned_evidence_ids


def test_associative_branch_cannot_silently_switch_to_native(
    setup: tuple[StoryProjectStore, StoryGenerationRequest],
) -> None:
    store, request = setup
    first = _commit(store, prepare_story_generation(request, store=store))
    child = replace(
        request,
        intent=ShotIntent.NEXT_SHOT,
        previous_story=first.state,
        previous_frame=first.last_frame,
        associative_memory=None,
    )
    with pytest.raises(ValueError, match="requires its associative"):
        prepare_story_generation(child, store=store)


def test_checkpoint_change_and_corrupt_snapshot_fail_before_generation(
    setup: tuple[StoryProjectStore, StoryGenerationRequest],
) -> None:
    store, request = setup
    first = _commit(store, prepare_story_generation(request, store=store))
    child = replace(
        request,
        intent=ShotIntent.NEXT_SHOT,
        previous_story=first.state,
        previous_frame=first.last_frame,
    )
    config = request.associative_memory
    assert config is not None
    with pytest.raises(ValueError, match="content SHA-256"):
        prepare_story_generation(
            replace(child, associative_memory=replace(config, checkpoint_sha256="f" * 64)),
            store=store,
        )
    memory = cast(dict[str, Any], first.loaded_revision.shot_metadata["associative_memory"])
    (store.root / "assets" / "sha256" / memory["blocks"][0]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_story_generation(child, store=store)


def test_inactive_checkpoint_is_rejected_and_active_checkpoint_changes_guides(
    tmp_path: Path,
) -> None:
    inactive = _config(tmp_path / "inactive.pt", active=False)
    with pytest.raises(ValueError, match="no measurable history/order effect"):
        inactive.load(InspectionCodec())
    active = _config(tmp_path / "active.pt")
    runtime = active.load(InspectionCodec())
    try:
        report = history_effect_report(runtime.bridge, require_effect=True)
        assert report["history_delta_rms"] > 0
        assert report["order_delta_rms"] > 0
    finally:
        runtime.close()


def test_checkpoint_inspection_reports_mechanism_without_claiming_render_quality(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from comfy_story.memory.inspect import main

    config = _config(tmp_path / "active.pt")
    assert (
        main(
            [
                "--checkpoint",
                config.checkpoint,
                "--sha256",
                config.checkpoint_sha256,
                "--foundation-sha256",
                config.foundation_sha256,
                "--model-configuration-sha256",
                config.model_configuration_sha256,
                "--vae-sha256",
                config.vae_sha256,
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["checkpoint_authenticated"] is True
    assert report["history_effect"]["history_delta_rms"] > 0
    assert report["rendered_quality_validated"] is False
    assert report["foundation_and_vae_loaded"] is False


def test_recovery_rejects_missing_associative_assets(
    setup: tuple[StoryProjectStore, StoryGenerationRequest],
) -> None:
    from comfy_story.memory.service import verify_memory_revision

    store, request = setup
    first = _commit(store, prepare_story_generation(request, store=store))
    verify_memory_revision(first.loaded_revision, request.associative_memory, store)
    data = cast(dict[str, Any], first.loaded_revision.shot_metadata["associative_memory"])
    (store.root / "assets" / "sha256" / data["context_sha256"]).unlink()
    with pytest.raises(ValueError, match="unavailable"):
        verify_memory_revision(first.loaded_revision, request.associative_memory, store)
