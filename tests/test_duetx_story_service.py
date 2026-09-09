from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from duet.duetx.contracts import DuetXContract, tensor_sha256
from duet.duetx.ltx_bridge import LTXLatentHistoryBridge
from duet.duetx.minimax_h3_bridge import MiniMaxH3LatentHistoryBridge
from duet.duetx.story_contracts import (
    ReferenceRole,
    ShotIntent,
    StoryLibrary,
    StoryReference,
)
from duet.duetx.story_memory_backend import (
    StoryMemoryBackend,
    StoryMemoryIdentity,
    StoryMemoryRuntime,
    StorySampler,
)
from duet.duetx.story_product_contracts import ObservationKind
from duet.duetx.story_service import (
    EncodedStoryFrame,
    StoryCommitRequest,
    StoryGenerationRequest,
    commit_story_generation,
    normalize_story_frame,
    prepare_story_generation,
)
from duet.duetx.story_store import StoryProjectStore


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _image(value: float) -> torch.Tensor:
    return torch.full((1, 24, 32, 3), value, dtype=torch.float32)


def _reference(name: str, role: ReferenceRole, value: str) -> StoryReference:
    digest = _digest(value)
    return StoryReference(
        name,
        role,
        "",
        digest,
        f"duet-story://assets/sha256/{digest}",
        (digest,),
        _digest(f"preprocess-{value}"),
    ).validate()


def _library() -> StoryLibrary:
    return StoryLibrary(
        "Coast Story",
        (
            _reference("Maya", ReferenceRole.CHARACTER, "maya"),
            _reference("WoodenChest", ReferenceRole.PROP, "chest"),
            _reference("Coast", ReferenceRole.LOCATION, "coast"),
        ),
    ).validate()


def _store(root: Path) -> StoryProjectStore:
    store = StoryProjectStore(root)
    for value in (b"maya", b"chest", b"coast"):
        store.put_asset(value)
    return store


def _request(**changes: object) -> StoryGenerationRequest:
    library = _library()
    values: dict[str, object] = {
        "intent": ShotIntent.START_STORY,
        "project_id": "coast-story",
        "branch_id": "main",
        "previous_story": None,
        "previous_frame": None,
        "world_frame": _image(0.1),
        "library": library,
        "reference_images": {
            "Maya": _image(0.2),
            "WoodenChest": _image(0.3),
            "Coast": _image(0.4),
        },
        "prompt": "@Maya opens @WoodenChest.",
        "shot_length_seconds": 5,
        "variation": 101,
        "checkpoint_sha256": "6" * 64,
        "model_configuration_sha256": "7" * 64,
    }
    values.update(changes)
    return StoryGenerationRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("intent", "has_state", "has_previous", "has_world", "message"),
    [
        (ShotIntent.START_STORY, True, False, True, "must not receive Previous Story"),
        (ShotIntent.START_STORY, False, False, False, "requires World / starting frame"),
        (ShotIntent.CONTINUE_THIS_SHOT, True, False, False, "requires Previous Frame"),
        (ShotIntent.NEXT_SHOT, False, False, True, "requires Previous Story"),
        (ShotIntent.NEW_SCENE, True, True, False, "requires World / starting frame"),
    ],
)
def test_intent_preflight_fails_before_state_load(
    tmp_path: Path,
    intent: ShotIntent,
    has_state: bool,
    has_previous: bool,
    has_world: bool,
    message: str,
) -> None:
    class SentinelStore:
        def load(self, *args: object) -> object:
            raise AssertionError("state load must not run")

    request = _request(
        intent=intent,
        previous_story=(object() if has_state else None),
        previous_frame=(_image(0.8) if has_previous else None),
        world_frame=(_image(0.1) if has_world else None),
    )
    with pytest.raises(ValueError, match=message):
        prepare_story_generation(
            request,
            store=cast(StoryProjectStore, SentinelStore()),
            contract=DuetXContract.default(decision1_fingerprint="1" * 64),
        )


def test_prepared_roles_are_current_context_and_two_exact_mentions(tmp_path: Path) -> None:
    prepared = prepare_story_generation(
        _request(),
        store=_store(tmp_path),
        contract=DuetXContract.default(decision1_fingerprint="1" * 64),
    )

    assert prepared.active_reference_names == ("Maya", "WoodenChest")
    assert prepared.visual_roles == (
        "current-or-starting-frame",
        "duet-x-story-context",
        "exact-reference:Maya",
        "exact-reference:WoodenChest",
    )
    assert torch.count_nonzero(prepared.visual_guides[1]) == 0


def test_story_frame_normalization_is_rgb_384_and_deterministic() -> None:
    images = torch.linspace(0, 1, 5 * 24 * 32 * 3).reshape(5, 24, 32, 3)
    first = normalize_story_frame(images[-1:])
    second = normalize_story_frame(images[-1:].clone())
    assert first.shape == (384, 384, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


class _FakeRuntime:
    def __init__(self, bridge: LTXLatentHistoryBridge, contract: DuetXContract) -> None:
        self._bridge = bridge
        self._identity = StoryMemoryIdentity(
            StoryMemoryBackend.LTX,
            contract,
            "6" * 64,
            "7" * 64,
        ).validate()

    @property
    def identity(self) -> StoryMemoryIdentity:
        return self._identity

    @property
    def bridge(self) -> LTXLatentHistoryBridge:
        return self._bridge

    def materialize_frame(self, frame: np.ndarray[Any, Any]) -> EncodedStoryFrame:
        assert frame.shape == (384, 384, 3)
        return EncodedStoryFrame(
            torch.full((1, 128, 1, 12, 12), 0.25, dtype=torch.float32),
            _digest("preprocessed"),
            _digest("vae"),
            _digest("adapter"),
            _digest("scorer"),
        ).validate(self.identity.contract)

    def score(self, latent: torch.Tensor, event_type: int, timestamp: float) -> float:
        assert latent.shape == (1, 128, 1, 12, 12)
        assert event_type == 0
        assert timestamp == 0.0
        return 1.0

    def decode(self, latent: torch.Tensor) -> np.ndarray[Any, Any]:
        assert latent.shape == (1, 128, 1, 12, 12)
        return np.full((384, 384, 3), 127, dtype=np.uint8)

    def close(self) -> None:
        return None


def test_story_request_defaults_preserve_the_ltx_native_path() -> None:
    request = _request()

    assert request.memory_backend is StoryMemoryBackend.LTX
    assert request.sampler is StorySampler.NATIVE_RES_MULTISTEP


def test_commit_adds_exactly_one_official_boundary_leaf_after_decode(tmp_path: Path) -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    store = _store(tmp_path)
    bridge = LTXLatentHistoryBridge(channels=128, operator_size=16)
    prepared = prepare_story_generation(_request(), store=store, contract=contract)

    result = commit_story_generation(
        StoryCommitRequest(
            prepared=prepared,
            decoded_images=torch.rand(4, 24, 32, 3),
            saved_video_sha256="8" * 64,
            saved_video_locator="duet-evidence://story/sha256/" + "8" * 64,
        ),
        store=store,
        runtime=_FakeRuntime(bridge, contract),
    )

    assert result.state.shot_count == 1
    assert result.last_frame.shape == (1, 24, 32, 3)
    leaf = result.loaded_revision.blocks[0].leaves[0]
    assert leaf is not None
    assert leaf.key.source_rank == 0
    assert leaf.dense_operator.shape == (1, 144, 16, 16)
    assert result.loaded_revision.story_context_png.startswith(b"\x89PNG")


def test_second_shot_loads_shared_library_and_authenticated_context(tmp_path: Path) -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    store = _store(tmp_path)
    bridge = LTXLatentHistoryBridge(channels=128, operator_size=16)
    first = prepare_story_generation(_request(), store=store, contract=contract)
    committed = commit_story_generation(
        StoryCommitRequest(
            first,
            torch.rand(2, 24, 32, 3),
            "8" * 64,
            "duet-evidence://story/sha256/" + "8" * 64,
        ),
        store=store,
        runtime=_FakeRuntime(bridge, contract),
    )

    second = prepare_story_generation(
        _request(
            intent=ShotIntent.NEXT_SHOT,
            previous_story=committed.state,
            previous_frame=committed.last_frame,
            world_frame=None,
            library=None,
            reference_images={"Maya": _image(0.2), "WoodenChest": _image(0.3)},
        ),
        store=store,
        contract=contract,
    )

    assert second.parent_shot_count == 1
    assert second.library == _library()
    assert torch.count_nonzero(second.visual_guides[1]) > 0


def test_commit_rejects_runtime_identity_before_encoding(tmp_path: Path) -> None:
    contract = DuetXContract.default(decision1_fingerprint="1" * 64)
    store = _store(tmp_path)
    bridge = LTXLatentHistoryBridge(operator_size=16)
    prepared = prepare_story_generation(_request(), store=store, contract=contract)
    runtime = _FakeRuntime(bridge, contract)
    runtime._identity = StoryMemoryIdentity(
        StoryMemoryBackend.LTX,
        contract,
        "9" * 64,
        "7" * 64,
    ).validate()

    assert isinstance(runtime, StoryMemoryRuntime)
    with pytest.raises(ValueError, match="checkpoint"):
        commit_story_generation(
            StoryCommitRequest(
                prepared,
                torch.rand(2, 24, 32, 3),
                "8" * 64,
                "duet-evidence://story/sha256/" + "8" * 64,
            ),
            store=store,
            runtime=runtime,
        )


class _MiniMaxCommitRuntime:
    def __init__(self) -> None:
        self._bridge = MiniMaxH3LatentHistoryBridge(operator_size=4)
        with torch.no_grad():
            self._bridge.residual.weight.copy_(torch.eye(24))
        self._identity = StoryMemoryIdentity(
            StoryMemoryBackend.MINIMAX_H3,
            DuetXContract.minimax_h3(adapter_fingerprint="a" * 64),
            "b" * 64,
            "c" * 64,
        ).validate()
        self.encoded_means: list[float] = []

    @property
    def identity(self) -> StoryMemoryIdentity:
        return self._identity

    @property
    def bridge(self) -> MiniMaxH3LatentHistoryBridge:
        return self._bridge

    def materialize_frame(self, frame: np.ndarray[Any, Any]) -> EncodedStoryFrame:
        mean = float(frame.mean() / 255.0)
        self.encoded_means.append(mean)
        return EncodedStoryFrame(
            torch.full((1, 24, 1, 4, 4), mean),
            _digest(f"preprocessed-{mean}"),
            "d" * 64,
            "e" * 64,
            "f" * 64,
        ).validate(self.identity.contract)

    def score(self, latent: torch.Tensor, event_type: int, timestamp: float) -> float:
        del latent, event_type
        return timestamp * 127.0

    def decode(self, latent: torch.Tensor) -> np.ndarray[Any, Any]:
        value = round(float(latent.mean().item()) * 255.0)
        return np.full((384, 384, 3), max(0, min(255, value)), dtype=np.uint8)

    def close(self) -> None:
        return None


def test_minimax_commit_publishes_exact_exceptions_and_nonexception_core(
    tmp_path: Path,
) -> None:
    runtime = _MiniMaxCommitRuntime()
    contract = runtime.identity.contract
    store = _store(tmp_path)
    state = None
    previous_frame = None
    results = []
    prepared_roles: list[tuple[str, ...]] = []
    for shot_index, value in enumerate((0.1, 0.4, 0.8)):
        request = _request(
            intent=ShotIntent.START_STORY if shot_index == 0 else ShotIntent.NEXT_SHOT,
            previous_story=state,
            previous_frame=previous_frame,
            world_frame=_image(value) if shot_index == 0 else None,
            library=_library() if shot_index == 0 else None,
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
        )
        prepared = prepare_story_generation(request, store=store, contract=contract)
        prepared_roles.append(prepared.visual_roles)
        result = commit_story_generation(
            StoryCommitRequest(
                prepared,
                torch.full((2, 24, 32, 3), value),
                f"{shot_index + 1:064x}",
                f"duet-evidence://story/sha256/{shot_index + 1:064x}",
            ),
            store=store,
            runtime=runtime,
        )
        results.append(result)
        state = result.state
        previous_frame = result.last_frame

    loaded = results[-1].loaded_revision
    bundle = loaded.require_guide_bundle()
    assert bundle.core_png is not None
    assert len(bundle.exception_pngs) == 2
    assert tuple(item.provenance.leaf.source_rank for item in loaded.memory.exceptions) == (2, 1)
    for item, image in zip(loaded.memory.exceptions, bundle.exception_pngs, strict=True):
        assert hashlib.sha256(image).hexdigest() == item.provenance.raw.content_sha256
        assert item.provenance.raw.locator.startswith("duet-evidence://story-frame/sha256/")
    assert len(runtime.encoded_means) == 4
    assert runtime.encoded_means[-1] == pytest.approx(0.1, abs=1 / 255)
    fourth = prepare_story_generation(
        _request(
            intent=ShotIntent.NEXT_SHOT,
            previous_story=state,
            previous_frame=previous_frame,
            world_frame=None,
            library=None,
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
        ),
        store=store,
        contract=contract,
    )
    prepared_roles.append(fourth.visual_roles)
    assert prepared_roles == [
        (
            "current-or-starting-frame",
            "exact-reference:Maya",
            "exact-reference:WoodenChest",
        ),
        (
            "current-or-starting-frame",
            "duet-x-exception:0",
            "exact-reference:Maya",
            "exact-reference:WoodenChest",
        ),
        (
            "current-or-starting-frame",
            "duet-x-exception:0",
            "duet-x-exception:1",
            "exact-reference:Maya",
            "exact-reference:WoodenChest",
        ),
        (
            "current-or-starting-frame",
            "duet-x-core",
            "duet-x-exception:0",
            "duet-x-exception:1",
            "exact-reference:Maya",
            "exact-reference:WoodenChest",
        ),
    ]


def test_living_canon_commit_captures_three_frames_but_appends_one_leaf(
    tmp_path: Path,
) -> None:
    runtime = _MiniMaxCommitRuntime()
    store = _store(tmp_path)
    prepared = prepare_story_generation(
        _request(
            prompt="@Maya opens @WoodenChest.",
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            living_canon=True,
        ),
        store=store,
        contract=runtime.identity.contract,
    )
    images = torch.zeros(5, 24, 32, 3)
    images[2:] = 1.0

    result = commit_story_generation(
        StoryCommitRequest(
            prepared,
            images,
            "4" * 64,
            f"duet-evidence://story/sha256/{'4' * 64}",
        ),
        store=store,
        runtime=runtime,
    )

    product = result.loaded_revision.require_product_state()
    assert result.state.shot_count == 1
    assert tuple(item.kind for item in product.observation_packets[0].observations) == (
        ObservationKind.OPENING,
        ObservationKind.CHANGE,
        ObservationKind.CLOSING,
    )
    assert (
        sum(leaf is not None for block in result.loaded_revision.blocks for leaf in block.leaves)
        == 1
    )
    leaf = result.loaded_revision.blocks[0].leaves[0]
    assert leaf is not None
    assert len(leaf.candidates) == 3
    assert product.observation_packets[0].timeline_stop_ns == round(5 * 1_000_000_000 / 24)
    assert result.loaded_revision.shot_metadata["decoded_frame_count"] == 5
    assert result.loaded_revision.shot_metadata["fps"] == 24
    assert store.load_asset(str(result.loaded_revision.shot_metadata["last_frame_full_sha256"]))


@pytest.mark.parametrize("profile", ["Reference shot", "Animate frame"])
def test_declared_cast_retains_pending_evidence_without_reference_images(
    tmp_path: Path, profile: str
) -> None:
    runtime = _MiniMaxCommitRuntime()
    store = _store(tmp_path)
    prepared = prepare_story_generation(
        _request(
            prompt="Maya opens WoodenChest.",
            reference_policy="Prompt mentions only",
            scene_entity_names=("Maya", "WoodenChest"),
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            living_canon=True,
            render_profile=profile,
        ),
        store=store,
        contract=runtime.identity.contract,
    )
    assert prepared.active_reference_names == ()
    assert prepared.visual_roles == ("current",)
    result = commit_story_generation(
        StoryCommitRequest(
            prepared,
            torch.zeros(5, 24, 32, 3),
            "4" * 64,
            f"duet-evidence://story/sha256/{'4' * 64}",
        ),
        store=store,
        runtime=runtime,
    )
    product = result.loaded_revision.require_product_state()
    packet = product.observation_packets[0]
    assert packet.referenced_entity_ids == ("maya", "woodenchest")
    assert all(
        observation.entity_ids == packet.referenced_entity_ids
        for observation in packet.observations
    )
    assert result.loaded_revision.shot_metadata["active_references"] == []
    assert result.loaded_revision.shot_metadata["declared_scene_entities"] == [
        "maya",
        "woodenchest",
    ]
    for entity in product.canon.entities:
        assert bool(entity.pending) == (entity.entity_id in packet.referenced_entity_ids)
        assert entity.confirmed_revision_sha256 is None
    assert result.loaded_revision.shot_metadata.get("render_profile", "Reference shot") == profile


@pytest.mark.parametrize("names", [("Unknown",), ("Maya", "maya"), ("",), ["Maya"]])
def test_declared_scene_names_are_validated(tmp_path: Path, names: object) -> None:
    with pytest.raises(ValueError, match=r"scene_entit|scene entities"):
        prepare_story_generation(
            _request(scene_entity_names=names, living_canon=True),
            store=_store(tmp_path),
            contract=DuetXContract.default(decision1_fingerprint="1" * 64),
        )


def test_living_canon_minimax_prepare_uses_four_semantic_guides_without_duplicates(
    tmp_path: Path,
) -> None:
    runtime = _MiniMaxCommitRuntime()
    store = _store(tmp_path)
    first = prepare_story_generation(
        _request(
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            living_canon=True,
        ),
        store=store,
        contract=runtime.identity.contract,
    )
    committed = commit_story_generation(
        StoryCommitRequest(
            first,
            torch.full((2, 24, 32, 3), 0.25),
            "5" * 64,
            f"duet-evidence://story/sha256/{'5' * 64}",
        ),
        store=store,
        runtime=runtime,
    )

    second = prepare_story_generation(
        _request(
            intent=ShotIntent.NEXT_SHOT,
            previous_story=committed.state,
            previous_frame=committed.last_frame,
            world_frame=None,
            library=None,
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            living_canon=True,
        ),
        store=store,
        contract=runtime.identity.contract,
    )

    assert second.visual_roles == (
        "current",
        "inclusive-core",
        "evidence-maya",
        "evidence-woodenchest",
    )
    assert len(second.visual_guides) == 4
    assert len({tensor_sha256(value) for value in second.visual_guides}) == 4
    assert second.recall_decision is not None
    assert tuple(source.source_id for source in second.compiler_sources) == (
        "story:current",
        "story:inclusive-core",
        "story:evidence-maya",
        "story:evidence-woodenchest",
    )
    assert tuple(source.protected for source in second.compiler_sources) == (
        False,
        False,
        True,
        True,
    )
    for intent in (ShotIntent.NEXT_SHOT, ShotIntent.CONTINUE_THIS_SHOT):
        cut = prepare_story_generation(
            replace(
                second.request,
                intent=intent,
                composition="New composition",
                prompt="@WoodenChest sits alone.",
                reference_policy="Prompt mentions only",
            ),
            store=store,
            contract=runtime.identity.contract,
        )
        assert cut.visual_roles == ("evidence-woodenchest",)
        assert torch.equal(cut.visual_guides[0], _image(0.3))
        assert "<Picture 1> is a baseline prop reference" in cut.resolved_prompt
        assert "<Picture 2>" not in cut.resolved_prompt
        assert cut.recall_decision is not None
        assert cut.recall_decision.inclusive_core_sha256 is None
        assert tuple(b.role for b in cut.recall_decision.guide_bindings) == cut.visual_roles
        assert tuple(s.ordinal for s in cut.compiler_sources) == (0,)
        assert cut.parent is not None
        assert second.parent is not None
        assert cut.parent.state == second.parent.state
        assert cut.recall_decision != second.recall_decision
        with pytest.raises(ValueError, match="requires a selected visual reference"):
            prepare_story_generation(
                replace(cut.request, prompt="An empty scene."),
                store=store,
                contract=runtime.identity.contract,
            )
    new_scene = prepare_story_generation(
        replace(
            second.request,
            intent=ShotIntent.NEW_SCENE,
            composition="New composition",
            world_frame=_image(0.7),
        ),
        store=store,
        contract=runtime.identity.contract,
    )
    assert new_scene.visual_roles[:2] == ("current", "inclusive-core")
    assert torch.equal(new_scene.visual_guides[0], _image(0.7))
    animated = prepare_story_generation(
        replace(second.request, render_profile="Animate frame"),
        store=store,
        contract=runtime.identity.contract,
    )
    assert animated.visual_roles == ("current",)
    assert animated.active_reference_names == ()
    assert animated.recall_decision is not None
    assert animated.recall_decision.explicit_entity_ids == ("maya", "woodenchest")
    assert animated.recall_decision.selected_evidence_ids == ()
    assert animated.recall_decision.inclusive_core_sha256 is None


def test_retrieval_only_ablation_keeps_current_and_exact_recall_without_story_core(
    tmp_path: Path,
) -> None:
    runtime = _MiniMaxCommitRuntime()
    store = _store(tmp_path)
    first = prepare_story_generation(
        _request(
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            living_canon=True,
        ),
        store=store,
        contract=runtime.identity.contract,
    )
    committed = commit_story_generation(
        StoryCommitRequest(
            first,
            torch.full((2, 24, 32, 3), 0.25),
            "5" * 64,
            f"duet-evidence://story/sha256/{'5' * 64}",
        ),
        store=store,
        runtime=runtime,
    )

    second = prepare_story_generation(
        _request(
            intent=ShotIntent.NEXT_SHOT,
            previous_story=committed.state,
            previous_frame=committed.last_frame,
            world_frame=None,
            library=None,
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            living_canon=True,
            include_associative_core=False,
        ),
        store=store,
        contract=runtime.identity.contract,
    )

    assert second.visual_roles == (
        "current",
        "evidence-maya",
        "evidence-woodenchest",
    )
    assert "inclusive chronological Story Context" not in second.resolved_prompt
    assert tuple(source.source_id for source in second.compiler_sources) == (
        "story:current",
        "story:evidence-maya",
        "story:evidence-woodenchest",
    )
    assert tuple(source.protected for source in second.compiler_sources) == (
        False,
        True,
        True,
    )
    assert second.recall_decision is not None
    assert second.recall_decision.inclusive_core_sha256 is None


def test_native_evidence_keeps_identity_and_approved_state_separate_without_a_collage(
    tmp_path: Path,
) -> None:
    import io
    from dataclasses import replace

    from PIL import Image

    from duet.duetx.story_product_contracts import (
        CanonPresence,
        MemoryAction,
        StoryMemoryCommand,
    )

    runtime = _MiniMaxCommitRuntime()
    store = _store(tmp_path)
    baseline = io.BytesIO()
    Image.new("RGB", (32, 24), (51, 51, 51)).save(baseline, format="PNG")
    digest = store.put_asset(baseline.getvalue())
    maya = replace(
        _library().references[0],
        media_sha256=digest,
        evidence_locator=f"duet-story://assets/sha256/{digest}",
        keyframe_sha256s=(digest,),
    )
    library = StoryLibrary("Separate evidence", (maya,)).validate()
    request = _request(
        library=library,
        prompt="@Maya waits.",
        memory_backend=StoryMemoryBackend.MINIMAX_H3,
        checkpoint_sha256=runtime.identity.checkpoint_sha256,
        model_configuration_sha256=runtime.identity.model_configuration_sha256,
        living_canon=True,
        include_associative_core=False,
    )
    first = prepare_story_generation(request, store=store, contract=runtime.identity.contract)
    committed = commit_story_generation(
        StoryCommitRequest(
            first,
            torch.full((2, 24, 32, 3), 0.8),
            "5" * 64,
            f"duet-evidence://story/sha256/{'5' * 64}",
        ),
        store=store,
        runtime=runtime,
    )
    product = committed.loaded_revision.require_product_state()
    record = next(item for item in product.evidence_records if item.kind is ObservationKind.CLOSING)
    command = StoryMemoryCommand(
        MemoryAction.UPDATE_CANON,
        committed.state.revision_sha256,
        "maya",
        "The current coat is wet.",
        CanonPresence.PRESENT,
        (record.evidence_id,),
    )
    next_request = replace(
        request,
        intent=ShotIntent.NEW_SCENE,
        previous_story=committed.state,
        library=None,
        memory_commands=(command,),
        separate_evidence_images=True,
    )
    separate = prepare_story_generation(
        next_request, store=store, contract=runtime.identity.contract
    )
    assert separate.visual_roles == ("current", "identity-maya", "evidence-maya")
    torch.testing.assert_close(separate.visual_guides[1], _image(0.2))
    # Neither source contains gray padding or a half-width contact-sheet panel.
    assert separate.visual_guides[2].std().item() < 1e-6
    assert separate.visual_guides[2].mean().item() == pytest.approx(0.8, abs=1 / 255)
    assert "takes precedence for confirmed state" in separate.resolved_prompt
    assert "<Picture 3> is exact current evidence for @maya" in separate.resolved_prompt
    assert "Approved state of @Maya: The current coat is wet." in separate.resolved_prompt
    assert separate.recall_decision is not None
    assert separate.recall_decision.selected_evidence_ids == (record.evidence_id,)
    assert len(separate.recall_decision.packet_specs) == 1
    legacy = prepare_story_generation(
        replace(next_request, separate_evidence_images=False),
        store=store,
        contract=runtime.identity.contract,
    )
    assert legacy.visual_roles == ("current", "evidence-maya")
    assert legacy.visual_guides[1].shape == (1, 384, 384, 3)
    assert legacy.recall_decision is not None
    assert legacy.recall_decision.packet_specs == separate.recall_decision.packet_specs
    assert legacy.recall_decision.guide_bindings != separate.recall_decision.guide_bindings
    # New conditioning still appends to the authenticated existing memory tree.
    second = commit_story_generation(
        StoryCommitRequest(
            separate,
            torch.full((2, 24, 32, 3), 0.7),
            "8" * 64,
            f"duet-evidence://story/sha256/{'8' * 64}",
        ),
        store=store,
        runtime=runtime,
    )
    assert second.state.shot_count == 2
    assert second.state.parent_revision_sha256 == committed.state.revision_sha256


def test_living_canon_first_shot_bounds_exact_compiler_exceptions(tmp_path: Path) -> None:
    runtime = _MiniMaxCommitRuntime()

    prepared = prepare_story_generation(
        _request(
            prompt="@Maya opens @WoodenChest on @Coast.",
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            living_canon=True,
        ),
        store=_store(tmp_path),
        contract=runtime.identity.contract,
    )

    assert prepared.visual_roles == (
        "current",
        "evidence-maya",
        "evidence-woodenchest",
        "reference-coast",
    )
    assert tuple(source.protected for source in prepared.compiler_sources) == (
        False,
        True,
        True,
        False,
    )
    assert "baseline character reference for @Maya" in prepared.resolved_prompt
    assert "baseline prop reference for @WoodenChest" in prepared.resolved_prompt
    assert "baseline location reference for @Coast" in prepared.resolved_prompt
    assert (
        "incidental background, pose, other objects and actions are not current story state"
        in prepared.resolved_prompt
    )
    assert "exact current evidence" not in prepared.resolved_prompt


def test_minimax_first_shot_does_not_duplicate_one_named_reference(tmp_path: Path) -> None:
    runtime = _MiniMaxCommitRuntime()
    request = _request(
        prompt="@Maya walks along the coast.",
        memory_backend=StoryMemoryBackend.MINIMAX_H3,
        checkpoint_sha256=runtime.identity.checkpoint_sha256,
        model_configuration_sha256=runtime.identity.model_configuration_sha256,
    )

    prepared = prepare_story_generation(
        request,
        store=_store(tmp_path),
        contract=runtime.identity.contract,
    )

    assert prepared.visual_roles == (
        "current-or-starting-frame",
        "exact-reference:Maya",
    )
    assert len(prepared.visual_guides) == 2
    assert "<Picture 1>" in prepared.resolved_prompt
    assert "<Picture 2> is @Maya" in prepared.resolved_prompt
    assert "<Picture 3>" not in prepared.resolved_prompt


def test_minimax_preparation_carries_ordered_compiler_sources(tmp_path: Path) -> None:
    runtime = _MiniMaxCommitRuntime()
    request = _request(
        prompt="@Maya opens @WoodenChest.",
        protected_reference_names=("WoodenChest",),
        memory_backend=StoryMemoryBackend.MINIMAX_H3,
        checkpoint_sha256=runtime.identity.checkpoint_sha256,
        model_configuration_sha256=runtime.identity.model_configuration_sha256,
    )

    prepared = prepare_story_generation(
        request,
        store=_store(tmp_path),
        contract=runtime.identity.contract,
    )

    assert tuple(source.source_id for source in prepared.compiler_sources) == (
        "story:current-or-starting-frame",
        "library:Maya",
        "library:WoodenChest",
    )
    assert tuple(source.ordinal for source in prepared.compiler_sources) == (0, 1, 2)
    assert tuple(source.protected for source in prepared.compiler_sources) == (False, False, True)


def test_minimax_protected_references_must_be_selected(tmp_path: Path) -> None:
    runtime = _MiniMaxCommitRuntime()
    request = _request(
        prompt="@Maya walks.",
        protected_reference_names=("WoodenChest",),
        memory_backend=StoryMemoryBackend.MINIMAX_H3,
        checkpoint_sha256=runtime.identity.checkpoint_sha256,
        model_configuration_sha256=runtime.identity.model_configuration_sha256,
    )

    with pytest.raises(ValueError, match="selected by @mention"):
        prepare_story_generation(
            request, store=_store(tmp_path), contract=runtime.identity.contract
        )


def test_story_branch_locks_memory_backend_but_not_sampler(tmp_path: Path) -> None:
    runtime = _MiniMaxCommitRuntime()
    store = _store(tmp_path)
    first = prepare_story_generation(
        _request(
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
        ),
        store=store,
        contract=runtime.identity.contract,
    )
    committed = commit_story_generation(
        StoryCommitRequest(
            first,
            torch.full((2, 24, 32, 3), 0.25),
            "1" * 64,
            f"duet-evidence://story/sha256/{'1' * 64}",
        ),
        store=store,
        runtime=runtime,
    )

    with pytest.raises(ValueError, match=r"snapshot|contract|checkpoint"):
        prepare_story_generation(
            _request(
                intent=ShotIntent.NEXT_SHOT,
                previous_story=committed.state,
                previous_frame=committed.last_frame,
                world_frame=None,
                library=None,
                memory_backend=StoryMemoryBackend.LTX,
            ),
            store=store,
            contract=DuetXContract.default(decision1_fingerprint="0" * 64),
        )

    speed = prepare_story_generation(
        _request(
            intent=ShotIntent.NEXT_SHOT,
            previous_story=committed.state,
            previous_frame=committed.last_frame,
            world_frame=None,
            library=None,
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
            sampler=StorySampler.SPEED_EULER_2STAGE,
        ),
        store=store,
        contract=runtime.identity.contract,
    )

    assert speed.request.sampler is StorySampler.SPEED_EULER_2STAGE
    assert speed.request.checkpoint_sha256 == first.request.checkpoint_sha256
    assert speed.request.model_configuration_sha256 == first.request.model_configuration_sha256
    speed_commit = commit_story_generation(
        StoryCommitRequest(
            speed,
            torch.full((2, 24, 32, 3), 0.5),
            "2" * 64,
            f"duet-evidence://story/sha256/{'2' * 64}",
        ),
        store=store,
        runtime=runtime,
    )
    assert speed_commit.loaded_revision.shot_metadata["sampler"] == "speed-euler-2stage"


@pytest.mark.parametrize("living_canon", [False, True])
def test_minimax_new_composition_retains_references_without_anchor_instruction(
    tmp_path: Path, living_canon: bool
) -> None:
    runtime = _MiniMaxCommitRuntime()
    request = _request(
        prompt="@Maya walks along the coast.",
        memory_backend=StoryMemoryBackend.MINIMAX_H3,
        checkpoint_sha256=runtime.identity.checkpoint_sha256,
        model_configuration_sha256=runtime.identity.model_configuration_sha256,
        living_canon=living_canon,
        composition="New composition",
    )
    prepared = prepare_story_generation(
        request, store=_store(tmp_path), contract=runtime.identity.contract
    )
    assert "composition anchor" not in prepared.resolved_prompt
    assert "do not copy its framing" in prepared.resolved_prompt
    assert len(prepared.visual_guides) == 2


def test_unconfirmed_library_notes_are_not_labeled_as_approved_story_state(tmp_path: Path) -> None:
    from dataclasses import replace

    runtime = _MiniMaxCommitRuntime()
    original = _library()
    library = replace(
        original,
        references=tuple(
            replace(ref, note="The chest starts plain and will be painted red later.")
            if ref.name == "WoodenChest"
            else ref
            for ref in original.references
        ),
    )
    direction = "@WoodenChest is unpainted. Nobody has painted it yet."
    prepared = prepare_story_generation(
        _request(
            library=library,
            prompt=direction,
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            living_canon=True,
        ),
        store=_store(tmp_path),
        contract=runtime.identity.contract,
    )
    assert (
        "Baseline reference notes for @WoodenChest (not approved current state)"
        in prepared.resolved_prompt
    )
    assert "Approved state of @WoodenChest" not in prepared.resolved_prompt
    assert "Do not enact future events mentioned in reference notes" in prepared.resolved_prompt
    assert prepared.resolved_prompt.endswith("User direction: " + direction)
    assert prepared.product_state is not None
    chest = next(e for e in prepared.product_state.canon.entities if e.entity_id == "woodenchest")
    assert chest.confirmed_revision_sha256 is None
    assert not chest.supporting_evidence_ids


def test_multiple_baseline_notes_share_one_direction_precedence_rule(tmp_path: Path) -> None:
    runtime = _MiniMaxCommitRuntime()
    library = replace(
        _library(),
        references=tuple(
            replace(ref, note=f"Recognizable {ref.name}." if ref.name == "Maya" else "Plain wood")
            for ref in _library().references
        ),
    )
    direction = "@Maya waits beside @WoodenChest."
    prepared = prepare_story_generation(
        _request(
            library=library,
            prompt=direction,
            memory_backend=StoryMemoryBackend.MINIMAX_H3,
            checkpoint_sha256=runtime.identity.checkpoint_sha256,
            model_configuration_sha256=runtime.identity.model_configuration_sha256,
            living_canon=True,
        ),
        store=_store(tmp_path),
        contract=runtime.identity.contract,
    )
    prompt = prepared.resolved_prompt
    assert "Recognizable Maya.." not in prompt
    assert "(not approved current state): Recognizable Maya." in prompt
    assert "(not approved current state): Plain wood." in prompt
    assert prompt.count("Do not enact future events mentioned in reference notes.") == 1
    assert "Baseline reference notes for @Coast" not in prompt
    assert prompt.endswith("User direction: " + direction)


@pytest.mark.parametrize(
    "change",
    [
        {"living_canon": False},
        {"composition": "New composition"},
        {"sampler": StorySampler.SPEED_EULER_2STAGE},
        {"memory_backend": StoryMemoryBackend.LTX},
    ],
)
def test_frame_profile_rejects_incompatible_contracts(
    tmp_path: Path, change: dict[str, object]
) -> None:
    request = _request(
        **{
            "render_profile": "Animate frame",
            "living_canon": True,
            "memory_backend": StoryMemoryBackend.MINIMAX_H3,
            **change,
        }
    )
    with pytest.raises(ValueError, match="Animate frame"):
        prepare_story_generation(
            request,
            store=_store(tmp_path),
            contract=DuetXContract.default(decision1_fingerprint="1" * 64),
        )


@pytest.mark.parametrize("world", [None, 0.1])
def test_native_reference_start_uses_only_explicit_scene_and_cast(
    tmp_path: Path, world: float | None
) -> None:
    request = _request(
        world_frame=None if world is None else _image(world),
        composition="New composition",
        memory_backend=StoryMemoryBackend.MINIMAX_H3,
        native_reference_archive=True,
        living_canon=True,
        include_associative_core=False,
        checkpoint_sha256=None,
        prompt="@Maya walks along the coast.",
    )
    prepared = prepare_story_generation(request, store=_store(tmp_path), contract=None)
    assert prepared.active_reference_names == ("Maya",)
    assert len(prepared.visual_guides) == (1 if world is None else 2)
    assert ("current" in prepared.visual_roles) is (world is not None)
    assert ("prior visual context" in prepared.resolved_prompt) is (world is not None)
    assert torch.equal(prepared.visual_guides[-1], _image(0.2))
    assert prepared.request.variation == 101


def test_reference_only_start_rejects_empty_selected_roster(tmp_path: Path) -> None:
    request = _request(
        world_frame=None,
        composition="New composition",
        memory_backend=StoryMemoryBackend.MINIMAX_H3,
        native_reference_archive=True,
        living_canon=True,
        include_associative_core=False,
        checkpoint_sha256=None,
        prompt="A quiet landscape.",
        reference_policy="Prompt mentions only",
    )
    with pytest.raises(ValueError, match="selected visual reference"):
        prepare_story_generation(request, store=_store(tmp_path), contract=None)
