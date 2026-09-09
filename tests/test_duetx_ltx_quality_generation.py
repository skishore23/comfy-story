from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch

import duet.duetx.ltx_quality_generation as generation
from duet.duetx.checkpoint import TrainingModules
from duet.duetx.contracts import tensor_sha256
from duet.duetx.ltx_media import (
    PINNED_LTX_OFFLOAD_MODE,
    STREAMING_FOUNDATION_GUARD_SHA256,
    OneStageMediaGenerationResult,
)
from duet.duetx.ltx_quality_generation import (
    ArtifactInstallReceipt,
    BuiltMethodGuides,
    CellStore,
    FrozenMethodModules,
    GenerationAudit,
    GenerationPrerequisites,
    HostRuntimeIdentity,
    aggregate_generation,
    build_cell_keys,
    build_method_guides,
    build_partitions,
    finalize_generation,
    generate_cell_once,
    load_frozen_trainable_checkpoint,
    load_generation_prerequisites,
)
from duet.duetx.ltx_quality_protocol import (
    FROZEN_TRAINABLE_CHECKPOINT_SHA256,
    MATCHED_SEEDS,
    CellState,
    CommittedGenerationCell,
    CounterfactualCluster,
    GenerationCellKey,
    GenerationCellLifecycle,
    GuideSource,
    GuideSourceKind,
    Method,
    MethodGuideReceipt,
    QualityCategory,
    QualityProtocol,
    ReviewPhase,
    ReviewPhaseHistory,
    RuntimeCoordinate,
    SceneVariant,
    canonical_sha256,
    scene_manifest_sha256,
)

_LIFECYCLE_AUTHORIZATION_SHA256 = "e" * 64


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.cache
def _guide_digest(value: int) -> str:
    return tensor_sha256(torch.full((1, 128, 1, 12, 12), float(value), dtype=torch.float32))


def _clusters() -> tuple[CounterfactualCluster, ...]:
    clusters = []
    categories = tuple(QualityCategory)
    for index in range(18):
        cluster_id = f"cluster-{index:02d}"
        category = categories[index // 6]
        visible = (0, 1, 3, 4)[index % 4]
        variants = []
        for variant_index, variant_id in enumerate(("A", "B")):
            rgb = [_digest(f"rgb-{index}-{slot}") for slot in range(9)]
            pre = [_digest(f"pre-{index}-{slot}") for slot in range(9)]
            latent = [_guide_digest(slot + 1) for slot in range(9)]
            rgb[visible] = _digest(f"rgb-{index}-{variant_id}-{visible}")
            pre[visible] = _digest(f"pre-{index}-{variant_id}-{visible}")
            latent[visible] = _guide_digest(visible + 1 + variant_index * 100)
            variants.append(
                SceneVariant(
                    scene_id=f"{cluster_id}-{variant_id.lower()}",
                    cluster_id=cluster_id,
                    variant_id=variant_id,
                    category=category,
                    fact_attribute="hidden cue",
                    fact_value=("alpha", "beta")[variant_index],
                    counterfactual_value=("beta", "alpha")[variant_index],
                    accepted_values=("alpha", "beta"),
                    fact_visible_slot=visible,
                    fact_absent_slots=(2, 5, 6, 7, 8),
                    prompt="Continue and reveal the same hidden cue.",
                    negative_prompt="text, watermark",
                    forbidden_fact_terms=("alpha", "beta"),
                    base_plate_sha256=_digest(f"base-{index}"),
                    rgb_frame_sha256=tuple(rgb),
                    preprocessed_tensor_sha256=tuple(pre),
                    vae_latent_sha256=tuple(latent),
                    scoring_region=(10, 10, 100, 100),
                    reveal_interval=(10, 20),
                    objective_fact_rubric="The continuation reproduces the hidden cue.",
                    source_workflow_sha256=_digest("workflow"),
                    rights_receipt_sha256=_digest(f"rights-{index}"),
                    predecessor_exclusion_ledger_sha256=_digest("ledger"),
                ).validate()
            )
        clusters.append(
            CounterfactualCluster(
                cluster_id, category, cast(tuple[SceneVariant, SceneVariant], tuple(variants))
            ).validate()
        )
    return tuple(clusters)


def _scene_with_guides() -> tuple[SceneVariant, tuple[torch.Tensor, ...]]:
    scene = _clusters()[0].variants[0]
    guides = tuple(
        torch.full((1, 128, 1, 12, 12), float(slot + 1), dtype=torch.float32) for slot in range(9)
    )
    return scene, guides


class _Gated(torch.nn.Module):
    def compress(self, history: torch.Tensor) -> torch.Tensor:
        return history.mean(dim=1)


class _Bridge(torch.nn.Module):
    def forward_exclusion(self, history: torch.Tensor, *, candidates: object) -> object:
        del candidates
        selected = (2, 5)
        keep = tuple(index for index in range(8) if index not in selected)

        class Result:
            core_latent = history[:, keep].sum(dim=1)
            excluded_leaf_keys = tuple(type("Key", (), {"slot": item})() for item in selected)

        return Result()


class _MutatingGated(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def compress(self, history: torch.Tensor) -> torch.Tensor:
        self.scale.add_(1)
        return history.mean(dim=1) * self.scale


class _ReportedDeviceTensor(torch.Tensor):
    _reported_device: torch.device

    @staticmethod
    def __new__(
        cls,
        value: torch.Tensor,
        reported_device: torch.device,
    ) -> _ReportedDeviceTensor:
        result = torch.Tensor._make_subclass(cls, value, require_grad=False)
        result._reported_device = reported_device
        return result

    def __getattribute__(self, name: str) -> Any:
        if name == "device":
            return super().__getattribute__("_reported_device")
        return super().__getattribute__(name)


class _DeviceStateModule(torch.nn.Module):
    def __init__(self, reported_device: torch.device) -> None:
        super().__init__()
        self.state = _ReportedDeviceTensor(torch.ones(1), reported_device)

    def _save_to_state_dict(
        self,
        destination: dict[str, Any],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        del keep_vars
        destination[f"{prefix}state"] = self.state


def _modules() -> FrozenMethodModules:
    gated = _Gated().eval().requires_grad_(False)
    bridge = _Bridge().eval().requires_grad_(False)
    return FrozenMethodModules.capture(
        FROZEN_TRAINABLE_CHECKPOINT_SHA256,
        gated,
        bridge,
        torch.device("cpu"),
        expected_gated_type=_Gated,
        expected_bridge_type=_Bridge,
    )


def _prerequisites(clusters: tuple[CounterfactualCluster, ...]) -> GenerationPrerequisites:
    return GenerationPrerequisites(
        format="duet-x-ltx-quality-preregistered-v1",
        phases=ReviewPhaseHistory(
            (
                ReviewPhase.SOURCE_AUTHORED,
                ReviewPhase.DEVELOPMENT_REHEARSED,
                ReviewPhase.PREREGISTERED,
            )
        ),
        protocol_sha256=QualityProtocol.default().fingerprint(),
        scene_manifest_sha256=scene_manifest_sha256(clusters),
        clusters=clusters,
        materialized_population_sha256=_digest("materialized"),
        trainable_checkpoint_sha256=FROZEN_TRAINABLE_CHECKPOINT_SHA256,
        runtime=RuntimeCoordinate.default(),
        pilot_evidence_sha256=_digest("pilot"),
        rehearsal_evidence_sha256=_digest("rehearsal"),
        source_commit="b" * 40,
        source_archive_sha256=_digest("confirmatory-source-archive"),
        operational_configuration_sha256=_digest("operational-configuration"),
        runtime_equivalence_seal_sha256=_digest("runtime-equivalence-seal"),
        hosts=(
            HostRuntimeIdentity(
                "forge1",
                _digest("forge1-runtime"),
                _digest("foundation"),
                _digest("forge1-runtime-equivalence-receipt"),
            ),
            HostRuntimeIdentity(
                "forge2",
                _digest("forge2-runtime"),
                _digest("foundation"),
                _digest("forge2-runtime-equivalence-receipt"),
            ),
        ),
    ).validate()


def test_method_guides_use_the_one_frozen_boundary_and_exact_order() -> None:
    scene, guides = _scene_with_guides()
    manifest = _digest("scene-manifest")
    modules = _modules()

    full = build_method_guides(
        GenerationCellKey(scene.scene_id, 101, Method.FULL_HISTORY),
        scene,
        guides,
        manifest,
        modules,
    )
    recent = build_method_guides(
        GenerationCellKey(scene.scene_id, 101, Method.RECENT_ANCHOR),
        scene,
        guides,
        manifest,
        modules,
    )
    gated = build_method_guides(
        GenerationCellKey(scene.scene_id, 101, Method.GATED_CORE), scene, guides, manifest, modules
    )
    duet = build_method_guides(
        GenerationCellKey(scene.scene_id, 101, Method.DUET_CORE), scene, guides, manifest, modules
    )

    assert full.guides == guides
    assert recent.guides == (guides[7], guides[2], guides[5], guides[8])
    assert torch.equal(
        gated.guides[0],
        torch.stack(tuple(guides[index] for index in (0, 1, 3, 4, 6, 7)), dim=1).mean(dim=1),
    )
    assert gated.guides[1:] == (guides[2], guides[5], guides[8])
    assert torch.equal(
        duet.guides[0],
        sum((guides[index] for index in (0, 1, 3, 4, 6, 7)), torch.zeros_like(guides[0])),
    )
    assert duet.guides[1:] == (guides[2], guides[5], guides[8])
    assert tuple(source.scene_slot for source in recent.receipt.sources) == (7, 2, 5, 8)
    assert duet.receipt.core_input_sha256 is not None


def test_frozen_modules_reject_concrete_architecture_and_finite_state_drift() -> None:
    modules = _modules()
    with pytest.raises(ValueError, match="architecture changed"):
        dataclasses.replace(modules, expected_gated_type=torch.nn.Identity).validate()

    gated = _MutatingGated().eval().requires_grad_(False)
    captured = FrozenMethodModules.capture(
        FROZEN_TRAINABLE_CHECKPOINT_SHA256,
        gated,
        _Bridge().eval().requires_grad_(False),
        torch.device("cpu"),
        expected_gated_type=_MutatingGated,
        expected_bridge_type=_Bridge,
    )
    gated.scale.add_(1)
    with pytest.raises(ValueError, match="tensor state changed"):
        captured.validate()


def test_frozen_modules_match_declared_cuda_device_semantically() -> None:
    cases = (
        (torch.device("cuda"), torch.device("cuda"), True),
        (torch.device("cuda:0"), torch.device("cuda"), True),
        (torch.device("cuda:1"), torch.device("cuda"), False),
        (torch.device("cuda:2"), torch.device("cuda"), False),
        (torch.device("cpu"), torch.device("cuda"), False),
        (torch.device("mps"), torch.device("cuda"), False),
        (torch.device("meta"), torch.device("cuda"), False),
        (torch.device("cuda:1"), torch.device("cuda:1"), True),
        (torch.device("cuda"), torch.device("cuda:1"), False),
        (torch.device("cuda:0"), torch.device("cuda:1"), False),
        (torch.device("cuda:2"), torch.device("cuda:1"), False),
        (torch.device("cpu"), torch.device("cpu"), True),
        (torch.device("mps"), torch.device("mps"), True),
        (torch.device("meta"), torch.device("meta"), True),
    )

    for actual, expected, accepted in cases:
        gated = _DeviceStateModule(actual).eval().requires_grad_(False)
        bridge = _DeviceStateModule(actual).eval().requires_grad_(False)
        capture = functools.partial(
            FrozenMethodModules.capture,
            FROZEN_TRAINABLE_CHECKPOINT_SHA256,
            gated,
            bridge,
            expected,
            expected_gated_type=_DeviceStateModule,
            expected_bridge_type=_DeviceStateModule,
        )
        if accepted:
            assert capture().device == expected
        else:
            with pytest.raises(ValueError, match="declared device"):
                capture()


def test_frozen_checkpoint_loader_rejects_wrong_bytes_and_freezes_loaded_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "trainable.pt"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="content SHA-256"):
        load_frozen_trainable_checkpoint(path, device=torch.device("cpu"))

    modules = TrainingModules(
        bridge=torch.nn.Identity(),
        gated=torch.nn.Identity(),
        resampler=torch.nn.Identity(),
        scorer=torch.nn.Identity(),
    ).validate()
    state_digest = hashlib.sha256(b"duet-x-ltx-probe-modules-v1\0[]\n").hexdigest()
    payload = {
        "format": "duet-x-ltx-probe-trainable-v1",
        "foundation_sha256": _digest("foundation"),
        "module_state_sha256": state_digest,
        "modules": {name: {} for name in modules.named()},
        "optimizer_steps": 1,
        "protocol_sha256": _digest("probe-protocol"),
        "training_trace": {},
        "training_trace_sha256": _digest("trace"),
    }
    torch.save(payload, path)
    monkeypatch.setattr(generation, "FROZEN_TRAINABLE_CHECKPOINT_SHA256", _file_digest(path))
    loaded = load_frozen_trainable_checkpoint(
        path,
        device=torch.device("cpu"),
        module_factory=lambda: modules,
        expected_module_types=(torch.nn.Identity, torch.nn.Identity),
    )
    assert not loaded.gated.training
    assert not loaded.bridge.training
    assert not any(parameter.requires_grad for parameter in loaded.gated.parameters())
    with pytest.raises(ValueError, match="explicit expected types"):
        load_frozen_trainable_checkpoint(
            path,
            device=torch.device("cpu"),
            module_factory=lambda: modules,
        )


def test_frozen_checkpoint_loader_rejects_envelope_and_module_roster_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "trainable.pt"
    torch.save({"format": "wrong"}, path)
    monkeypatch.setattr(generation, "FROZEN_TRAINABLE_CHECKPOINT_SHA256", _file_digest(path))
    with pytest.raises(ValueError, match="envelope"):
        load_frozen_trainable_checkpoint(path, device=torch.device("cpu"))


def test_checkpoint_and_prerequisite_loads_reject_symlinks_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    clusters = _clusters()
    prerequisite = _prerequisites(clusters)
    canonical = tmp_path / "preregistration.json"
    canonical.write_bytes(prerequisite.to_json())
    canonical.chmod(0o600)
    assert load_generation_prerequisites(canonical) == prerequisite

    linked = tmp_path / "linked.json"
    linked.symlink_to(canonical)
    with pytest.raises(ValueError, match="unsafe"):
        load_generation_prerequisites(linked)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(json.loads(canonical.read_bytes()), indent=2))
    noncanonical.chmod(0o600)
    with pytest.raises(ValueError, match="not canonical"):
        load_generation_prerequisites(noncanonical)

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_link = tmp_path / "checkpoint-link.pt"
    checkpoint_link.symlink_to(checkpoint)
    with pytest.raises(ValueError, match="unsafe"):
        load_frozen_trainable_checkpoint(checkpoint_link, device=torch.device("cpu"))


@pytest.mark.parametrize("kind", ["prerequisite", "checkpoint"])
def test_generation_loaders_reject_hardlinked_inputs(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    if kind == "prerequisite":
        source.write_bytes(_prerequisites(_clusters()).to_json())
        source.chmod(0o600)
    else:
        source.write_bytes(b"checkpoint")
    linked = tmp_path / "linked"
    linked.hardlink_to(source)

    def load() -> object:
        if kind == "prerequisite":
            return load_generation_prerequisites(linked)
        return load_frozen_trainable_checkpoint(linked, device=torch.device("cpu"))

    with pytest.raises(ValueError, match=r"hardlink|single-link|unsafe"):
        load()


def test_generation_loader_rejects_metadata_drift_during_descriptor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "preregistration.json"
    path.write_bytes(_prerequisites(_clusters()).to_json())
    path.chmod(0o600)
    real_fstat = os.fstat
    calls = 0

    def drifting_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        metadata = real_fstat(descriptor)
        calls += 1
        if calls == 2:
            values = list(metadata)
            values[6] = metadata.st_size + 1
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr("duet.duetx.ltx_quality_file_io.os.fstat", drifting_fstat)
    with pytest.raises(ValueError, match=r"changed|stable"):
        load_generation_prerequisites(path)


def test_checkpoint_loader_hashes_and_decodes_one_captured_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"format": "wrong"}, checkpoint)
    monkeypatch.setattr(generation, "FROZEN_TRAINABLE_CHECKPOINT_SHA256", _file_digest(checkpoint))
    original = generation._read_regular_bytes
    reads = 0

    def counted(path: Path, name: str = "required file") -> bytes:
        nonlocal reads
        reads += 1
        return original(path, name)

    monkeypatch.setattr(generation, "_read_regular_bytes", counted)
    with pytest.raises(ValueError, match="envelope"):
        load_frozen_trainable_checkpoint(checkpoint, device=torch.device("cpu"))
    assert reads == 1


@pytest.mark.parametrize("mutation", ["shape", "dtype", "hash", "device"])
def test_method_guides_reject_every_latent_coordinate_drift(mutation: str) -> None:
    scene, guides = _scene_with_guides()
    values = list(guides)
    if mutation == "shape":
        values[0] = torch.zeros(1, 128, 1, 6, 6)
    elif mutation == "dtype":
        values[0] = values[0].to(torch.float64)
    elif mutation == "hash":
        values[0] = values[0] + 1
    else:
        values[0] = torch.empty((1, 128, 1, 12, 12), device="meta")
    with pytest.raises(ValueError, match=r"shape|CPU float32|hash"):
        build_method_guides(
            GenerationCellKey(scene.scene_id, 101, Method.FULL_HISTORY),
            scene,
            tuple(values),
            _digest("manifest"),
            _modules(),
        )


def test_exact_matrix_rotates_method_order_without_changing_seed() -> None:
    clusters = _clusters()
    keys = build_cell_keys(clusters)
    assert len(keys) == 432
    assert len(set(keys)) == 432
    assert {key.seed for key in keys} == set(MATCHED_SEEDS)
    first_scene = sorted(key.scene_id for key in keys)[0]
    orders = [
        tuple(key.method for key in keys if key.scene_id == first_scene and key.seed == seed)
        for seed in MATCHED_SEEDS
    ]
    assert len(set(orders)) == 3
    assert all(set(order) == set(Method) for order in orders)


def test_two_host_partitions_are_deterministic_disjoint_complete_and_bound() -> None:
    clusters = _clusters()
    prerequisites = _prerequisites(clusters)
    keys = build_cell_keys(clusters)
    first, second = build_partitions(keys, prerequisites)
    again = build_partitions(keys, prerequisites)

    assert (first, second) == again
    assert first.host_id == "forge1"
    assert second.host_id == "forge2"
    assert len(first.keys) == len(second.keys) == 216
    assert set(first.keys).isdisjoint(second.keys)
    assert {*first.keys, *second.keys} == set(keys)
    with pytest.raises(ValueError, match="host"):
        dataclasses.replace(first, host_id="forge3").validate()


def test_cross_host_key_is_rejected_before_guide_start_or_model_call(tmp_path: Path) -> None:
    clusters = _clusters()
    prerequisites = _prerequisites(clusters)
    forge1 = CellStore(
        tmp_path,
        prerequisites,
        prerequisites.hosts[0],
        _LIFECYCLE_AUTHORIZATION_SHA256,
    ).validate()
    foreign_key = build_partitions(build_cell_keys(clusters), prerequisites)[1].keys[0]
    scene = next(
        scene
        for cluster in clusters
        for scene in cluster.variants
        if scene.scene_id == foreign_key.scene_id
    )
    engine = _Engine()
    with pytest.raises(ValueError, match="not assigned"):
        generate_cell_once(
            store=forge1,
            key=foreign_key,
            scene=scene,
            scene_guides=(),
            modules=_modules(),
            engine=engine,
            decoded_frames_sha256=lambda _: _digest("frames"),
            peak_memory_bytes=lambda: 1,
        )
    with pytest.raises(ValueError, match="not assigned"):
        forge1.begin(foreign_key, _modules().core_modules_sha256)
    with pytest.raises(ValueError, match="not assigned"):
        forge1.install(
            foreign_key,
            cast(Any, None),
            engine,
            cast(Any, None),
            scene,
            (),
            _modules(),
            lambda _: _digest("frames"),
            lambda: 1,
        )
    assert engine.calls == 0
    assert not forge1.cell_root(foreign_key).exists()


def test_self_mutating_core_is_terminal_after_one_core_call_and_before_media(
    tmp_path: Path,
) -> None:
    clusters = _clusters()
    scene, guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 202, Method.GATED_CORE)
    store = _store_for_key(tmp_path, clusters, key)
    gated = _MutatingGated().eval().requires_grad_(False)
    bridge = _Bridge().eval().requires_grad_(False)
    modules = FrozenMethodModules.capture(
        FROZEN_TRAINABLE_CHECKPOINT_SHA256,
        gated,
        bridge,
        torch.device("cpu"),
        expected_gated_type=_MutatingGated,
        expected_bridge_type=_Bridge,
    )
    engine = _Engine()
    with pytest.raises(ValueError, match=r"tensor state changed|identity or version changed"):
        generate_cell_once(
            store=store,
            key=key,
            scene=scene,
            scene_guides=guides,
            modules=modules,
            engine=engine,
            decoded_frames_sha256=lambda _: _digest("frames"),
            peak_memory_bytes=lambda: 1,
        )
    assert engine.calls == 0
    assert store.status(key) is CellState.STARTED
    with pytest.raises(RuntimeError, match="terminal evidence failure"):
        generate_cell_once(
            store=store,
            key=key,
            scene=scene,
            scene_guides=guides,
            modules=_modules(),
            engine=engine,
            decoded_frames_sha256=lambda _: _digest("frames"),
            peak_memory_bytes=lambda: 1,
        )


class _Engine:
    def __init__(
        self,
        *,
        offload_mode: str = PINNED_LTX_OFFLOAD_MODE,
        foundation_guard_sha256: str = STREAMING_FOUNDATION_GUARD_SHA256,
    ) -> None:
        self.calls = 0
        self.offload_mode = offload_mode
        self.foundation_guard_sha256 = foundation_guard_sha256

    def generate(
        self,
        *,
        guides: tuple[torch.Tensor, ...],
        prompt: str,
        negative_prompt: str,
        seed: int,
        output_path: Path,
    ) -> OneStageMediaGenerationResult:
        del prompt, negative_prompt
        self.calls += 1
        output_path.write_bytes(b"opaque-video" + seed.to_bytes(2, "big") + bytes([len(guides)]))
        latent = torch.full((1, 128, 4, 12, 12), float(seed), dtype=torch.float32)
        return OneStageMediaGenerationResult(
            path=output_path,
            final_latent=latent,
            final_latent_sha256=tensor_sha256(latent),
            checkpoint_file_sha256=RuntimeCoordinate.default().model_checkpoint_sha256,
            foundation_before_sha256=_digest("foundation"),
            foundation_after_sha256=_digest("foundation"),
            offload_mode=self.offload_mode,
            foundation_guard_sha256=self.foundation_guard_sha256,
            media_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
            width=384,
            height=384,
            frames=25,
            fps=24,
            duration_seconds=25.0 / 24.0,
        ).validate()


def _store(tmp_path: Path, clusters: tuple[CounterfactualCluster, ...]) -> CellStore:
    root = tmp_path / "cells"
    root.mkdir(mode=0o700)
    prerequisites = _prerequisites(clusters)
    return CellStore(
        root,
        prerequisites,
        prerequisites.hosts[0],
        _LIFECYCLE_AUTHORIZATION_SHA256,
    ).validate()


def _store_for_key(
    tmp_path: Path, clusters: tuple[CounterfactualCluster, ...], key: GenerationCellKey
) -> CellStore:
    root = tmp_path / "cells"
    root.mkdir(mode=0o700)
    prerequisites = _prerequisites(clusters)
    partition = next(
        item
        for item in build_partitions(build_cell_keys(clusters), prerequisites)
        if key in item.keys
    )
    host = next(item for item in prerequisites.hosts if item.host_id == partition.host_id)
    return CellStore(root, prerequisites, host, _LIFECYCLE_AUTHORIZATION_SHA256).validate()


def _example_install(
    key: GenerationCellKey,
    receipt: MethodGuideReceipt,
    host_id: str,
    seal: str,
) -> ArtifactInstallReceipt:
    digest = _digest("example")
    return ArtifactInstallReceipt(
        "duet-x-ltx-quality-generation-v1",
        GenerationCellLifecycle(key, tuple(CellState)[:3]),
        seal,
        digest,
        _LIFECYCLE_AUTHORIZATION_SHA256,
        receipt,
        digest,
        digest,
        digest,
        digest,
        digest,
        1.0,
        0,
        1.0,
        digest,
        digest,
        RuntimeCoordinate.default().fingerprint(),
        digest,
        host_id,
        QualityProtocol.default().fingerprint(),
    ).validate()


def test_cell_lifecycle_is_append_only_atomic_and_committed_never_reruns(tmp_path: Path) -> None:
    clusters = _clusters()
    scene, guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 101, Method.DUET_CORE)
    store = _store_for_key(tmp_path, clusters, key)
    engine = _Engine()

    first = generate_cell_once(
        store=store,
        key=key,
        scene=scene,
        scene_guides=guides,
        modules=_modules(),
        engine=engine,
        decoded_frames_sha256=lambda path: _digest(path.read_bytes().hex()),
        peak_memory_bytes=lambda: 123,
    )
    second = generate_cell_once(
        store=store,
        key=key,
        scene=scene,
        scene_guides=guides,
        modules=_modules(),
        engine=engine,
        decoded_frames_sha256=lambda path: _digest(path.read_bytes().hex()),
        peak_memory_bytes=lambda: 999,
    )

    assert engine.calls == 1
    assert first == second
    assert first.lifecycle.states == tuple(CellState)
    cell_root = store.cell_root(key)
    assert sorted(path.name for path in cell_root.glob("*.json")) == [
        "00-absent.json",
        "01-guide-bound.json",
        "01-started.json",
        "03-committed.json",
    ]
    assert (cell_root / "artifact" / "02-artifact-installed.json").is_file()
    assert (cell_root / "artifact" / "media.mp4").read_bytes().startswith(b"opaque-video")
    assert not tuple(cell_root.glob(".*.tmp-*"))


def test_installed_artifact_is_promoted_disk_only_after_commit_crash(tmp_path: Path) -> None:
    clusters = _clusters()
    scene, guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 202, Method.GATED_CORE)
    store = _store_for_key(tmp_path, clusters, key)
    engine = _Engine()
    built = build_method_guides(
        key, scene, guides, store.prerequisites.scene_manifest_sha256, _modules()
    )
    started = store.begin(key, built.core_modules_sha256)
    store.bind_guides(started, built)
    store.install(
        key,
        started,
        engine,
        built,
        scene,
        guides,
        _modules(),
        lambda _: _digest("frames"),
        lambda: 7,
    )
    committed = generate_cell_once(
        store=store,
        key=key,
        scene=scene,
        scene_guides=guides,
        modules=_modules(),
        engine=engine,
        decoded_frames_sha256=lambda _: _digest("must-not-run"),
        peak_memory_bytes=lambda: 999,
    )
    assert engine.calls == 1
    assert committed.lifecycle.current is CellState.COMMITTED


@pytest.mark.parametrize(
    ("offload_mode", "foundation_guard_sha256"),
    [
        ("none", STREAMING_FOUNDATION_GUARD_SHA256),
        (PINNED_LTX_OFFLOAD_MODE, "0" * 64),
    ],
)
def test_install_rejects_generation_engine_identity_drift_before_atomic_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offload_mode: str,
    foundation_guard_sha256: str,
) -> None:
    clusters = _clusters()
    scene, guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 202, Method.GATED_CORE)
    store = _store_for_key(tmp_path, clusters, key)
    built = build_method_guides(
        key, scene, guides, store.prerequisites.scene_manifest_sha256, _modules()
    )
    started = store.begin(key, built.core_modules_sha256)
    store.bind_guides(started, built)
    engine = _Engine(
        offload_mode=offload_mode,
        foundation_guard_sha256=foundation_guard_sha256,
    )
    monkeypatch.setattr(OneStageMediaGenerationResult, "validate", lambda self: self)

    with pytest.raises(ValueError, match="generation engine identity"):
        store.install(
            key,
            started,
            engine,
            built,
            scene,
            guides,
            _modules(),
            lambda _: _digest("frames"),
            lambda: 7,
        )

    assert engine.calls == 1
    assert not (store.cell_root(key) / "artifact").exists()


def test_absent_receipt_recovery_is_still_allowed_before_started(tmp_path: Path) -> None:
    clusters = _clusters()
    scene, guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 202, Method.RECENT_ANCHOR)
    store = _store_for_key(tmp_path, clusters, key)
    built = build_method_guides(
        key, scene, guides, store.prerequisites.scene_manifest_sha256, _modules()
    )
    store.begin(key, built.core_modules_sha256)
    (store.cell_root(key) / "01-started.json").unlink()
    engine = _Engine()
    committed = generate_cell_once(
        store=store,
        key=key,
        scene=scene,
        scene_guides=guides,
        modules=_modules(),
        engine=engine,
        decoded_frames_sha256=lambda _: _digest("frames"),
        peak_memory_bytes=lambda: 1,
    )
    assert engine.calls == 1
    assert committed.lifecycle.current is CellState.COMMITTED


def test_started_without_valid_installed_artifact_is_terminal_and_never_reruns(
    tmp_path: Path,
) -> None:
    clusters = _clusters()
    scene, guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 303, Method.RECENT_ANCHOR)
    store = _store_for_key(tmp_path, clusters, key)
    engine = _Engine()
    store.begin(
        key,
        build_method_guides(
            key, scene, guides, store.prerequisites.scene_manifest_sha256, _modules()
        ).core_modules_sha256,
    )
    with pytest.raises(RuntimeError, match="terminal evidence failure"):
        generate_cell_once(
            store=store,
            key=key,
            scene=scene,
            scene_guides=guides,
            modules=_modules(),
            engine=engine,
            decoded_frames_sha256=lambda _: _digest("frames"),
            peak_memory_bytes=lambda: 1,
        )
    assert engine.calls == 0


def test_foreign_scene_parent_is_rejected_before_any_model_call(tmp_path: Path) -> None:
    clusters = _clusters()
    scene, guides = _scene_with_guides()
    foreign = dataclasses.replace(scene, objective_fact_rubric="Substituted rubric.").validate()
    key = GenerationCellKey(scene.scene_id, 101, Method.FULL_HISTORY)
    store = _store_for_key(tmp_path, clusters, key)
    engine = _Engine()
    with pytest.raises(ValueError, match="preregistered scene manifest"):
        generate_cell_once(
            store=store,
            key=key,
            scene=foreign,
            scene_guides=guides,
            modules=_modules(),
            engine=engine,
            decoded_frames_sha256=lambda _: _digest("frames"),
            peak_memory_bytes=lambda: 1,
        )
    assert engine.calls == 0


def test_damaged_installed_media_is_terminal_not_regenerated(tmp_path: Path) -> None:
    clusters = _clusters()
    scene, guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 101, Method.FULL_HISTORY)
    store = _store_for_key(tmp_path, clusters, key)
    engine = _Engine()
    built = build_method_guides(
        key, scene, guides, store.prerequisites.scene_manifest_sha256, _modules()
    )
    started = store.begin(key, built.core_modules_sha256)
    store.bind_guides(started, built)
    store.install(
        key,
        started,
        engine,
        built,
        scene,
        guides,
        _modules(),
        lambda _: _digest("frames"),
        lambda: 1,
    )
    (store.cell_root(key) / "artifact" / "media.mp4").write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="terminal evidence failure"):
        generate_cell_once(
            store=store,
            key=key,
            scene=scene,
            scene_guides=guides,
            modules=_modules(),
            engine=engine,
            decoded_frames_sha256=lambda _: _digest("frames"),
            peak_memory_bytes=lambda: 1,
        )
    assert engine.calls == 1


def test_install_rejects_post_receipt_guide_mutation_before_media_forward(
    tmp_path: Path,
) -> None:
    clusters = _clusters()
    scene, scene_guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 202, Method.GATED_CORE)
    store = _store_for_key(tmp_path, clusters, key)
    built = build_method_guides(
        key,
        scene,
        scene_guides,
        store.prerequisites.scene_manifest_sha256,
        _modules(),
    )
    started = store.begin(key, built.core_modules_sha256)
    store.bind_guides(started, built)
    built.guides[0].add_(1)
    engine = _Engine()
    with pytest.raises(ValueError, match="guide tensor"):
        store.install(
            key,
            started,
            engine,
            built,
            scene,
            scene_guides,
            _modules(),
            lambda _: _digest("frames"),
            lambda: 1,
        )
    assert engine.calls == 0
    assert not tuple(store.cell_root(key).glob(".artifact.tmp-*"))


@pytest.mark.parametrize("method", [Method.GATED_CORE, Method.DUET_CORE])
def test_install_rejects_forged_method_core_before_media_forward(
    tmp_path: Path, method: Method
) -> None:
    clusters = _clusters()
    scene, scene_guides = _scene_with_guides()
    key = next(
        candidate
        for candidate in build_cell_keys(clusters)
        if candidate.scene_id == scene.scene_id
        and candidate.method is method
        and candidate
        in build_partitions(build_cell_keys(clusters), _prerequisites(clusters))[0].keys
    )
    store = _store_for_key(tmp_path, clusters, key)
    modules = _modules()
    legitimate = build_method_guides(
        key,
        scene,
        scene_guides,
        store.prerequisites.scene_manifest_sha256,
        modules,
    )
    forged_core = torch.full_like(legitimate.guides[0], 19)
    forged_source = dataclasses.replace(
        legitimate.receipt.sources[0], sha256=tensor_sha256(forged_core)
    ).validate()
    forged_receipt = dataclasses.replace(
        legitimate.receipt,
        sources=(forged_source, *legitimate.receipt.sources[1:]),
    ).validate()
    forged = BuiltMethodGuides(
        (forged_core, *legitimate.guides[1:]),
        forged_receipt,
        legitimate.core_modules_sha256,
    ).validate()
    started = store.begin(key, forged.core_modules_sha256)
    store.bind_guides(started, forged)
    engine = _Engine()
    with pytest.raises(ValueError, match="authoritative method guides"):
        store.install(
            key,
            started,
            engine,
            forged,
            scene,
            scene_guides,
            modules,
            lambda _: _digest("frames"),
            lambda: 1,
        )
    assert engine.calls == 0


def test_install_rejects_foreign_prompt_scene_before_media_forward(tmp_path: Path) -> None:
    clusters = _clusters()
    scene, scene_guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 101, Method.FULL_HISTORY)
    store = _store_for_key(tmp_path, clusters, key)
    built = build_method_guides(
        key,
        scene,
        scene_guides,
        store.prerequisites.scene_manifest_sha256,
        _modules(),
    )
    started = store.begin(key, built.core_modules_sha256)
    store.bind_guides(started, built)
    foreign = dataclasses.replace(scene, prompt="Foreign generation prompt.").validate()
    engine = _Engine()
    with pytest.raises(ValueError, match="authoritative preregistered scene"):
        store.install(
            key,
            started,
            engine,
            built,
            foreign,
            scene_guides,
            _modules(),
            lambda _: _digest("frames"),
            lambda: 1,
        )
    assert engine.calls == 0
    assert not tuple(store.cell_root(key).glob(".artifact.tmp-*"))


def test_install_rejects_started_receipt_for_another_same_host_key_before_forward(
    tmp_path: Path,
) -> None:
    clusters = _clusters()
    prerequisites = _prerequisites(clusters)
    partition = build_partitions(build_cell_keys(clusters), prerequisites)[0]
    key_a, key_b = partition.keys[:2]
    root = tmp_path / "cells"
    root.mkdir(mode=0o700)
    store = CellStore(
        root,
        prerequisites,
        prerequisites.hosts[0],
        _LIFECYCLE_AUTHORIZATION_SHA256,
    ).validate()
    scene_b = next(
        scene
        for cluster in clusters
        for scene in cluster.variants
        if scene.scene_id == key_b.scene_id
    )
    _, scene_guides = _scene_with_guides()
    modules = _modules()
    started_a = store.begin(key_a, modules.core_modules_sha256)
    built_b = build_method_guides(
        key_b,
        scene_b,
        scene_guides,
        prerequisites.scene_manifest_sha256,
        modules,
    )
    engine = _Engine()
    with pytest.raises(ValueError, match=r"exactly started|key"):
        store.install(
            key_b,
            started_a,
            engine,
            built_b,
            scene_b,
            scene_guides,
            modules,
            lambda _: _digest("frames"),
            lambda: 1,
        )
    assert engine.calls == 0
    assert not store.cell_root(key_b).exists()


def test_install_rejects_reinstall_after_artifact_and_commit_before_forward(
    tmp_path: Path,
) -> None:
    clusters = _clusters()
    scene, scene_guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 101, Method.FULL_HISTORY)
    store = _store_for_key(tmp_path, clusters, key)
    built = build_method_guides(
        key,
        scene,
        scene_guides,
        store.prerequisites.scene_manifest_sha256,
        _modules(),
    )
    started = store.begin(key, built.core_modules_sha256)
    store.bind_guides(started, built)
    engine = _Engine()
    store.install(
        key,
        started,
        engine,
        built,
        scene,
        scene_guides,
        _modules(),
        lambda _: _digest("frames"),
        lambda: 1,
    )
    with pytest.raises(ValueError, match="exactly started"):
        store.install(
            key,
            started,
            engine,
            built,
            scene,
            scene_guides,
            _modules(),
            lambda _: _digest("frames"),
            lambda: 1,
        )
    store.promote(key)
    with pytest.raises(ValueError, match="exactly started"):
        store.install(
            key,
            started,
            engine,
            built,
            scene,
            scene_guides,
            _modules(),
            lambda _: _digest("frames"),
            lambda: 1,
        )
    assert engine.calls == 1
    assert not tuple(store.cell_root(key).glob(".artifact.tmp-*"))


def test_extra_or_damaged_lifecycle_evidence_is_terminal_not_regenerated(tmp_path: Path) -> None:
    clusters = _clusters()
    store = _store(tmp_path, clusters)
    scene, guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 101, Method.FULL_HISTORY)
    engine = _Engine()
    generate_cell_once(
        store=store,
        key=key,
        scene=scene,
        scene_guides=guides,
        modules=_modules(),
        engine=engine,
        decoded_frames_sha256=lambda _: _digest("frames"),
        peak_memory_bytes=lambda: 1,
    )
    (store.cell_root(key) / "extra.json").write_text("{}")
    with pytest.raises(RuntimeError, match="terminal evidence failure"):
        generate_cell_once(
            store=store,
            key=key,
            scene=scene,
            scene_guides=guides,
            modules=_modules(),
            engine=engine,
            decoded_frames_sha256=lambda _: _digest("frames"),
            peak_memory_bytes=lambda: 1,
        )
    assert engine.calls == 1


def test_unrecognized_cell_evidence_is_rejected_before_model_call(tmp_path: Path) -> None:
    clusters = _clusters()
    store = _store(tmp_path, clusters)
    scene, guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 101, Method.FULL_HISTORY)
    store.cell_root(key).mkdir()
    (store.cell_root(key) / "foreign.json").write_text("{}")
    engine = _Engine()
    with pytest.raises(RuntimeError, match="terminal evidence failure"):
        generate_cell_once(
            store=store,
            key=key,
            scene=scene,
            scene_guides=guides,
            modules=_modules(),
            engine=engine,
            decoded_frames_sha256=lambda _: _digest("frames"),
            peak_memory_bytes=lambda: 1,
        )
    assert engine.calls == 0


def _generate_committed_cell(
    tmp_path: Path,
) -> tuple[CellStore, GenerationCellKey, SceneVariant, tuple[torch.Tensor, ...], _Engine]:
    clusters = _clusters()
    scene, guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 101, Method.FULL_HISTORY)
    store = _store_for_key(tmp_path, clusters, key)
    engine = _Engine()
    generate_cell_once(
        store=store,
        key=key,
        scene=scene,
        scene_guides=guides,
        modules=_modules(),
        engine=engine,
        decoded_frames_sha256=lambda _: _digest("frames"),
        peak_memory_bytes=lambda: 1,
    )
    return store, key, scene, guides, engine


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime_coordinate_sha256", _digest("foreign-runtime-coordinate")),
        ("model_foundations_before_sha256", _digest("foreign-foundation")),
        ("model_foundations_after_sha256", _digest("foreign-foundation")),
        ("protocol_sha256", _digest("foreign-protocol")),
        ("initial_noise_sha256", _digest("foreign-noise")),
    ],
)
def test_recovery_recomputes_every_preregistered_execution_parent(
    tmp_path: Path, field: str, value: str
) -> None:
    store, key, _, _, engine = _generate_committed_cell(tmp_path)
    installed = store.cell_root(key) / "artifact" / "02-artifact-installed.json"
    payload = json.loads(installed.read_bytes())
    payload[field] = value
    if field.startswith("model_foundations_"):
        payload["model_foundations_before_sha256"] = value
        payload["model_foundations_after_sha256"] = value
    installed.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    with pytest.raises(ValueError, match=r"parent|host binding"):
        store.status(key)
    assert engine.calls == 1


@pytest.mark.parametrize(
    "relative",
    [
        "00-absent.json",
        "01-started.json",
        "01-guide-bound.json",
        "artifact/02-artifact-installed.json",
        "03-committed.json",
    ],
)
@pytest.mark.parametrize("attack", ["symlink", "hardlink"])
def test_every_receipt_link_is_rejected(relative: str, attack: str, tmp_path: Path) -> None:
    store, key, _, _, _ = _generate_committed_cell(tmp_path)
    receipt = store.cell_root(key) / relative
    external = tmp_path / f"external-{receipt.name}"
    if attack == "symlink":
        external.write_bytes(receipt.read_bytes())
        receipt.unlink()
        receipt.symlink_to(external)
    else:
        external.hardlink_to(receipt)
    with pytest.raises(ValueError, match=r"hardlink|single-link|unsafe|non-symlink"):
        store.status(key)


def test_aggregate_rejects_any_extra_partial_directory(tmp_path: Path) -> None:
    clusters = _clusters()
    prerequisites = _prerequisites(clusters)
    partition = build_partitions(build_cell_keys(clusters), prerequisites)[0]
    root = tmp_path / "store"
    root.mkdir(mode=0o700)
    store = CellStore(
        root,
        prerequisites,
        prerequisites.hosts[0],
        _LIFECYCLE_AUTHORIZATION_SHA256,
    ).validate()
    (root / "foreign-partial-evidence").mkdir()
    with pytest.raises(ValueError, match="missing or extra"):
        store.audit_assigned_committed(partition)


def test_aggregate_rejects_missing_extra_duplicate_conflicting_cross_host_and_parent_drift(
    tmp_path: Path,
) -> None:
    clusters = _clusters()
    prerequisites = _prerequisites(clusters)
    keys = build_cell_keys(clusters)
    partitions = build_partitions(keys, prerequisites)
    stores = []
    lifecycle_hashes = (_digest("forge1-lifecycle"), _digest("forge2-lifecycle"))
    for index, host in enumerate(prerequisites.hosts):
        root = tmp_path / f"host-{index}"
        root.mkdir(mode=0o700)
        stores.append(
            CellStore(
                root,
                prerequisites,
                host,
                _LIFECYCLE_AUTHORIZATION_SHA256,
            ).validate()
        )

    with pytest.raises(ValueError, match="missing"):
        aggregate_generation(tuple(stores), partitions, clusters, lifecycle_hashes)
    wrong = dataclasses.replace(partitions[0], host_id="forge2")
    with pytest.raises(ValueError, match="host"):
        aggregate_generation(tuple(stores), (wrong, partitions[1]), clusters, lifecycle_hashes)
    with pytest.raises(ValueError, match="preregistration"):
        aggregate_generation(
            (
                dataclasses.replace(
                    stores[0],
                    prerequisites=dataclasses.replace(
                        prerequisites, pilot_evidence_sha256=_digest("drift")
                    ),
                ),
                stores[1],
            ),
            partitions,
            clusters,
            lifecycle_hashes,
        )


def _committed_cells_for_audit(
    clusters: tuple[CounterfactualCluster, ...],
) -> tuple[CommittedGenerationCell, ...]:
    manifest = scene_manifest_sha256(clusters)
    scenes = {variant.scene_id: variant for cluster in clusters for variant in cluster.variants}
    cells = []
    for key in build_cell_keys(clusters):
        scene = scenes[key.scene_id]
        if key.method is Method.FULL_HISTORY:
            slots = tuple(range(9))
            sources = tuple(
                GuideSource(GuideSourceKind.SCENE_VAE_LATENT, slot, scene.vae_latent_sha256[slot])
                for slot in slots
            )
            core_input = None
        elif key.method is Method.RECENT_ANCHOR:
            slots = (7, 2, 5, 8)
            sources = tuple(
                GuideSource(GuideSourceKind.SCENE_VAE_LATENT, slot, scene.vae_latent_sha256[slot])
                for slot in slots
            )
            core_input = None
        else:
            slots = (2, 5, 8)
            sources = (
                GuideSource(
                    GuideSourceKind.METHOD_CORE, None, _digest(f"core-{key.fingerprint()}")
                ),
                *(
                    GuideSource(
                        GuideSourceKind.SCENE_VAE_LATENT,
                        slot,
                        scene.vae_latent_sha256[slot],
                    )
                    for slot in slots
                ),
            )
            core_input = canonical_sha256(
                {
                    "format": "duet-x-method-core-input-v1",
                    "method": key.method.value,
                    "trainable_checkpoint_sha256": FROZEN_TRAINABLE_CHECKPOINT_SHA256,
                    "scene_manifest_sha256": manifest,
                    "scene_variant_sha256": scene.fingerprint(),
                    "history_vae_latent_sha256": tuple(
                        scene.vae_latent_sha256[index] for index in (0, 1, 3, 4, 6, 7)
                    ),
                }
            )
        guide = MethodGuideReceipt(
            key, manifest, scene.fingerprint(), sources, core_input
        ).validate()
        cells.append(_example_install(key, guide, "forge1", _digest("seal")).committed())
    method_order = {method: index for index, method in enumerate(Method)}
    return tuple(
        sorted(
            cells,
            key=lambda cell: (
                cell.key.scene_id,
                MATCHED_SEEDS.index(cell.key.seed),
                method_order[cell.key.method],
            ),
        )
    )


def test_generation_audit_rejects_missing_extra_duplicate_conflicting_and_nonfinite_cells() -> None:
    cells = _committed_cells_for_audit(_clusters())
    base = GenerationAudit(
        "duet-x-ltx-quality-generation-audit-v1",
        _digest("seal"),
        (_digest("partition-1"), _digest("partition-2")),
        (_digest("forge1-lifecycle"), _digest("forge2-lifecycle")),
        cells,
    ).validate()
    assert len(base.cells) == 432
    with pytest.raises(ValueError, match="exactly 432"):
        dataclasses.replace(base, cells=cells[:-1]).validate()
    with pytest.raises(ValueError, match="exactly 432"):
        dataclasses.replace(base, cells=(*cells, cells[0])).validate()
    conflict = dataclasses.replace(cells[0], media_sha256=_digest("conflict")).validate()
    with pytest.raises(ValueError, match="duplicate or conflicting"):
        dataclasses.replace(base, cells=(conflict, *cells[1:-1], cells[0])).validate()
    nonfinite = dataclasses.replace(cells[0], wall_time_seconds=float("nan"))
    with pytest.raises(ValueError, match="finite positive"):
        dataclasses.replace(base, cells=(nonfinite, *cells[1:])).validate()


def test_finalize_is_disk_only_and_byte_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clusters = _clusters()
    prerequisites = _prerequisites(clusters)
    partitions = build_partitions(build_cell_keys(clusters), prerequisites)
    marker = {"calls": 0}
    audit = GenerationAudit(
        "duet-x-ltx-quality-generation-audit-v1",
        prerequisites.fingerprint(),
        (partitions[0].fingerprint(), partitions[1].fingerprint()),
        (_digest("forge1-lifecycle"), _digest("forge2-lifecycle")),
        _committed_cells_for_audit(clusters),
    ).validate()

    def fake_aggregate(*args: object, **kwargs: object) -> GenerationAudit:
        marker["calls"] += 1
        return audit

    monkeypatch.setattr("duet.duetx.ltx_quality_generation.aggregate_generation", fake_aggregate)
    output = tmp_path / "final.json"
    lifecycle_hashes = (_digest("forge1-lifecycle"), _digest("forge2-lifecycle"))
    first = finalize_generation((), partitions, clusters, prerequisites, lifecycle_hashes, output)
    second = finalize_generation((), partitions, clusters, prerequisites, lifecycle_hashes, output)
    assert first == second == output.read_bytes()
    assert marker["calls"] == 2
    assert b"first_finalization_sha256" in first
    assert b"confirmatory_forge_receipt_sha256" in first


def test_installed_receipt_rejects_nonfinite_or_conflicting_evidence() -> None:
    scene, guides = _scene_with_guides()
    key = GenerationCellKey(scene.scene_id, 101, Method.FULL_HISTORY)
    receipt = build_method_guides(key, scene, guides, _digest("manifest"), _modules()).receipt
    installed = _example_install(key, receipt, "forge1", _digest("seal"))
    with pytest.raises(ValueError, match="finite positive"):
        dataclasses.replace(installed, wall_time_seconds=float("nan")).validate()
    with pytest.raises(ValueError, match="foundation mutation"):
        dataclasses.replace(installed, model_foundations_after_sha256=_digest("other")).validate()
