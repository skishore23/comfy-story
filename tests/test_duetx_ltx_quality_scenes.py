"""Tests for deterministic authored decoded-quality scenes."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import duet.duetx.ltx_quality_scenes as scenes
from duet.duetx.ltx_quality_protocol import (
    FACT_ABSENT_SLOTS,
    QualityCategory,
    from_canonical_json,
)
from duet.duetx.ltx_quality_scenes import (
    BasePlateManifest,
    BasePlateReceipt,
    ComposedCluster,
    ComposedFrames,
    ComposedPopulationManifest,
    MaterializationManifest,
    MaterializedSceneGuides,
    PredecessorExclusionLedger,
    RejectedPredecessor,
    RightsProvenanceReceipt,
    compose_population,
    frozen_cluster_specs,
    load_content_addressed_base_plate,
    materialize_rgb_frames,
    rgb_frame_sha256,
    validate_materialized_population,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plate(index: int) -> NDArray[np.uint8]:
    rows = np.arange(1024, dtype=np.uint16)[:, None]
    columns = np.arange(1024, dtype=np.uint16)[None, :]
    return np.ascontiguousarray(
        np.stack(
            (
                np.broadcast_to((rows + index) % 256, (1024, 1024)),
                np.broadcast_to((columns + index * 3) % 256, (1024, 1024)),
                (rows + columns + index * 11) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
    )


def _sources() -> tuple[
    BasePlateManifest,
    dict[str, NDArray[np.uint8]],
    tuple[RightsProvenanceReceipt, ...],
    PredecessorExclusionLedger,
]:
    specs = frozen_cluster_specs()
    plates = {spec.cluster_id: _plate(index) for index, spec in enumerate(specs)}
    receipts = tuple(
        BasePlateReceipt(
            cluster_id=spec.cluster_id,
            archive_name=f"plates/{spec.cluster_id}.png",
            source_file_sha256=_digest(f"file-{spec.cluster_id}"),
            decoded_rgb_sha256=rgb_frame_sha256(plates[spec.cluster_id]),
            decoded_shape=(1024, 1024, 3),
            accepted_job_id=f"accepted-{spec.cluster_id}",
            executed_prompt_sha256=_digest(f"prompt-{spec.cluster_id}"),
            workflow_sha256=_digest("workflow"),
            checkpoint_sha256=_digest("checkpoint"),
            authored_source_receipt_sha256=_digest(f"source-{spec.cluster_id}"),
        ).validate()
        for spec in specs
    )
    manifest = BasePlateManifest("duet-x-ltx-quality-base-plates-v1", receipts).validate()
    rights = tuple(
        RightsProvenanceReceipt(
            cluster_id=spec.cluster_id,
            accepted_job_id=f"accepted-{spec.cluster_id}",
            base_plate_sha256=rgb_frame_sha256(plates[spec.cluster_id]),
            workflow_sha256=_digest("workflow"),
            checkpoint_sha256=_digest("checkpoint"),
            self_authored_sdxl_source=True,
            no_third_party_footage=True,
            no_third_party_audio=True,
            no_third_party_logos=True,
            no_private_person_media=True,
        ).validate()
        for spec in specs
    )
    ledger = PredecessorExclusionLedger(
        "duet-x-ltx-quality-predecessors-v1",
        manifest.fingerprint(),
        tuple(spec.cluster_id for spec in specs),
        (
            RejectedPredecessor(
                "rejected-object-05", _digest("old-object"), "fact leak", "object-05"
            ),
            RejectedPredecessor(
                "rejected-temporal-03a", _digest("old-temporal-a"), "confound", "temporal-03"
            ),
            RejectedPredecessor(
                "rejected-temporal-03b", _digest("old-temporal-b"), "empty plane", "temporal-03"
            ),
        ),
    ).validate()
    return manifest, plates, rights, ledger


def _with_plate(
    manifest: BasePlateManifest,
    plates: dict[str, NDArray[np.uint8]],
    rights: tuple[RightsProvenanceReceipt, ...],
    ledger: PredecessorExclusionLedger,
    cluster_id: str,
    plate: NDArray[np.uint8],
) -> tuple[
    BasePlateManifest,
    dict[str, NDArray[np.uint8]],
    tuple[RightsProvenanceReceipt, ...],
    PredecessorExclusionLedger,
]:
    index = tuple(receipt.cluster_id for receipt in manifest.plates).index(cluster_id)
    source_hash = rgb_frame_sha256(plate)
    changed_manifest = BasePlateManifest(
        manifest.format,
        (
            *manifest.plates[:index],
            replace(
                manifest.plates[index],
                source_file_sha256=_digest(f"changed-file-{cluster_id}"),
                decoded_rgb_sha256=source_hash,
            ),
            *manifest.plates[index + 1 :],
        ),
    ).validate()
    changed_rights = (
        *rights[:index],
        replace(rights[index], base_plate_sha256=source_hash),
        *rights[index + 1 :],
    )
    changed_ledger = replace(
        ledger,
        source_manifest_sha256=changed_manifest.fingerprint(),
    ).validate()
    return changed_manifest, {**plates, cluster_id: plate}, changed_rights, changed_ledger


def _leaking_plate(spec_index: int, variant_index: int) -> tuple[str, NDArray[np.uint8]]:
    spec = frozen_cluster_specs()[spec_index]
    layer = spec.layers[variant_index]
    left, top, right, bottom = spec.scoring_region
    width, height = right - left, bottom - top
    if layer.kind in {"stripes", "arrow_left"}:
        x, y = 0, 0
    elif layer.kind == "arrow_right":
        x, y = width - 1, 0
    else:
        x, y = width // 2, height // 2
    output_x, output_y = left + x, top + y
    source_x = output_x * 1024 // 384
    source_y = output_y * 1024 // 384
    plate = np.zeros((1024, 1024, 3), dtype=np.uint8)
    plate[source_y, source_x] = np.asarray(layer.rgb, dtype=np.uint8)
    return spec.cluster_id, plate


def test_frozen_specs_are_exact_balanced_and_prompts_do_not_leak_aliases() -> None:
    specs = frozen_cluster_specs()

    assert len(specs) == 18
    assert tuple(spec.cluster_id for spec in specs) == tuple(
        [f"identity-{index:02d}" for index in range(1, 7)]
        + [f"object-{index:02d}" for index in range(1, 7)]
        + [f"temporal-{index:02d}" for index in range(1, 7)]
    )
    assert all(
        sum(spec.category is category for spec in specs) == 6 for category in QualityCategory
    )
    for spec in specs:
        text = f"{spec.prompt}\n{spec.negative_prompt}".casefold()
        assert all(term.casefold() not in text for term in spec.forbidden_fact_terms)


def test_population_composition_is_canonical_and_changes_exactly_one_old_slot() -> None:
    manifest, plates, rights, ledger = _sources()

    first = compose_population(manifest, plates, rights, ledger)
    second = compose_population(manifest, plates, rights, ledger)

    assert first.manifest().to_json() == second.manifest().to_json()
    assert len(first.clusters) == 18
    assert sum(len(cluster.variants) for cluster in first.clusters) == 36
    for cluster in first.clusters:
        left, right = cluster.variants
        changed = [
            slot
            for slot, (left_frame, right_frame) in enumerate(
                zip(left.frames.frames, right.frames.frames, strict=True)
            )
            if not np.array_equal(left_frame, right_frame)
        ]
        assert changed == [cluster.spec.fact_visible_slot]
        for slot in FACT_ABSENT_SLOTS:
            assert np.array_equal(left.frames.frames[slot], right.frames.frames[slot])
            assert left.frames.rgb_frame_sha256[slot] == right.frames.rgb_frame_sha256[slot]


@pytest.mark.parametrize("spec_index", [0, 6, 12])
@pytest.mark.parametrize("variant_index", [0, 1])
def test_composition_rejects_each_fact_layer_cue_already_visible_in_shared_frames(
    spec_index: int, variant_index: int
) -> None:
    manifest, plates, rights, ledger = _sources()
    cluster_id, leaked_plate = _leaking_plate(spec_index, variant_index)
    changed = _with_plate(manifest, plates, rights, ledger, cluster_id, leaked_plate)

    with pytest.raises(ValueError, match=r"fact layer.*absent"):
        compose_population(*changed)


def test_base_plate_manifest_rejects_duplicate_source_or_decoded_bytes() -> None:
    manifest, _, _, _ = _sources()
    first, second = manifest.plates[:2]

    with pytest.raises(ValueError, match="source-file"):
        BasePlateManifest(
            manifest.format,
            (
                first,
                replace(second, source_file_sha256=first.source_file_sha256),
                *manifest.plates[2:],
            ),
        ).validate()
    with pytest.raises(ValueError, match="decoded-pixel"):
        BasePlateManifest(
            manifest.format,
            (
                first,
                replace(second, decoded_rgb_sha256=first.decoded_rgb_sha256),
                *manifest.plates[2:],
            ),
        ).validate()


def test_hash_only_wire_manifests_round_trip_strictly_and_preserve_parents() -> None:
    manifest, plates, rights, ledger = _sources()
    population = compose_population(manifest, plates, rights, ledger)
    rgb = population.rgb_manifest()
    population_manifest = population.manifest()

    loaded_rgb = from_canonical_json(rgb.to_json(), type(rgb))
    loaded_population = from_canonical_json(
        population_manifest.to_json(), ComposedPopulationManifest
    )

    assert loaded_rgb.to_json() == rgb.to_json()
    assert loaded_rgb.fingerprint() == rgb.fingerprint()
    assert loaded_population.to_json() == population_manifest.to_json()
    assert loaded_population.rgb_frame_manifest_sha256 == rgb.fingerprint()
    with pytest.raises(ValueError, match="unknown field"):
        type(rgb).from_dict({**rgb.to_dict(), "unknown": "field"})
    with pytest.raises(ValueError, match="duplicate key"):
        from_canonical_json(
            b'{"format":"x","format":"x"}',
            ComposedPopulationManifest,
        )


def test_private_materialization_boundary_rejects_unrelated_ledger_or_source_chain() -> None:
    manifest, plates, rights, ledger = _sources()
    population = compose_population(manifest, plates, rights, ledger)
    composed = population.clusters[0]
    materialized = tuple(
        materialize_rgb_frames(
            variant.frames.frames,
            video_preprocess=lambda frame: torch.full(
                (3, 384, 384), float(frame.sum()), dtype=torch.float32
            ),
            image_conditioner=lambda value: value,
            video_encoder=lambda value: torch.full(
                (1, 128, 1, 12, 12), value[0, 0, 0].item(), dtype=torch.float32
            ),
        )
        for variant in composed.variants
    )

    from duet.duetx.ltx_quality_scenes import _protocol_cluster_from_materialized

    assert "protocol_cluster_from_materialized" not in scenes.__all__
    with pytest.raises(ValueError, match="ledger source manifest drift"):
        _protocol_cluster_from_materialized(
            replace(
                population,
                predecessor_ledger=replace(
                    ledger,
                    source_manifest_sha256=_digest("unrelated-ledger-parent"),
                ),
            ),
            composed,
            (materialized[0], materialized[1]),
        )
    substituted_source = replace(
        manifest.plates[0],
        source_file_sha256=_digest("substituted-file"),
        decoded_rgb_sha256=_digest("substituted-base"),
        accepted_job_id="substituted-job",
        workflow_sha256=_digest("substituted-workflow"),
        checkpoint_sha256=_digest("substituted-checkpoint"),
    )
    substituted_manifest = BasePlateManifest(
        manifest.format,
        (substituted_source, *manifest.plates[1:]),
    ).validate()
    substituted_rights = replace(
        rights[0],
        accepted_job_id=substituted_source.accepted_job_id,
        base_plate_sha256=substituted_source.decoded_rgb_sha256,
        workflow_sha256=substituted_source.workflow_sha256,
        checkpoint_sha256=substituted_source.checkpoint_sha256,
    )
    substituted_population = replace(
        population,
        source_manifest=substituted_manifest,
        rights_receipts=(substituted_rights, *rights[1:]),
        predecessor_ledger=replace(
            ledger,
            source_manifest_sha256=substituted_manifest.fingerprint(),
        ),
    )
    with pytest.raises(ValueError, match="composed base plate"):
        _protocol_cluster_from_materialized(
            substituted_population,
            composed,
            (materialized[0], materialized[1]),
        )


def test_private_materialization_boundary_rejects_non_member_cluster_with_one_pixel_rgb_drift() -> (
    None
):
    manifest, plates, rights, ledger = _sources()
    population = compose_population(manifest, plates, rights, ledger)
    original = population.clusters[0]
    altered_frames = list(original.variants[0].frames.frames)
    altered = np.array(altered_frames[original.spec.fact_visible_slot], copy=True, order="C")
    altered[0, 0, 0] ^= 1
    altered_frames[original.spec.fact_visible_slot] = altered
    crafted_variant = replace(
        original.variants[0],
        frames=ComposedFrames(
            tuple(altered_frames),
            tuple(rgb_frame_sha256(frame) for frame in altered_frames),
        ).validate(),
    )
    crafted = ComposedCluster(
        original.spec,
        original.base_plate_sha256,
        (crafted_variant, original.variants[1]),
    ).validate()
    guides = tuple(
        materialize_rgb_frames(
            variant.frames.frames,
            video_preprocess=lambda frame: torch.full(
                (3, 384, 384), float(frame.sum()), dtype=torch.float32
            ),
            image_conditioner=lambda value: value,
            video_encoder=lambda value: torch.full(
                (1, 128, 1, 12, 12), value[0, 0, 0].item(), dtype=torch.float32
            ),
        )
        for variant in crafted.variants
    )

    from duet.duetx.ltx_quality_scenes import _protocol_cluster_from_materialized

    with pytest.raises(ValueError, match="registered population member"):
        _protocol_cluster_from_materialized(population, crafted, (guides[0], guides[1]))


def test_materializer_defensively_copies_each_frame_and_hashes_every_boundary() -> None:
    frames = tuple(np.full((384, 384, 3), index, dtype=np.uint8) for index in range(9))
    calls: list[str] = []

    def preprocess(frame: NDArray[np.uint8]) -> torch.Tensor:
        calls.append("preprocess")
        frame.fill(255)
        return torch.zeros((3, 384, 384), dtype=torch.float32)

    def condition(value: torch.Tensor) -> torch.Tensor:
        calls.append("condition")
        return value.unsqueeze(0)

    def encode(value: torch.Tensor) -> torch.Tensor:
        calls.append("encode")
        return torch.ones((1, 128, 1, 12, 12), dtype=torch.float64)

    result = materialize_rgb_frames(
        frames,
        video_preprocess=preprocess,
        image_conditioner=condition,
        video_encoder=encode,
    )

    assert isinstance(result, MaterializedSceneGuides)
    assert calls == ["preprocess", "condition", "encode"] * 9
    assert all(frame[0, 0, 0] == index for index, frame in enumerate(frames))
    assert len(result.manifest.raw_encoder_tensor_sha256) == 9
    assert result.manifest.raw_encoder_tensor_sha256 != result.manifest.normalized_latent_sha256
    assert all(
        latent.dtype is torch.float32 and latent.device.type == "cpu" for latent in result.latents
    )


def test_composed_population_rejects_prompt_rights_and_predecessor_drift() -> None:
    manifest, plates, rights, ledger = _sources()
    leaked = replace(frozen_cluster_specs()[0], prompt="Reveal the amber stripe shown earlier.")

    with pytest.raises(ValueError, match="leaks"):
        leaked.validate()
    with pytest.raises(ValueError, match="leaks"):
        replace(frozen_cluster_specs()[-1], negative_prompt="avoid leftward motion").validate()
    invalid_rights = (
        replace(rights[0], self_authored_sdxl_source=False),
        replace(rights[0], no_third_party_footage=False),
        replace(rights[0], no_third_party_audio=False),
        replace(rights[0], no_third_party_logos=False),
        replace(rights[0], no_private_person_media=False),
    )
    for receipt in invalid_rights:
        with pytest.raises(ValueError, match="source-rights"):
            receipt.validate()
    with pytest.raises(ValueError, match="rights provenance"):
        compose_population(
            manifest,
            plates,
            (*rights[:-1], replace(rights[-1], checkpoint_sha256=_digest("wrong"))),
            ledger,
        )
    with pytest.raises(ValueError, match="ledger source manifest drift"):
        compose_population(
            manifest,
            plates,
            rights,
            replace(ledger, source_manifest_sha256=_digest("wrong")),
        )
    with pytest.raises(ValueError, match="pixel hash"):
        compose_population(
            manifest,
            {**plates, "identity-01": np.zeros((1024, 1024, 3), dtype=np.uint8)},
            rights,
            ledger,
        )
    reused_source = BasePlateManifest(
        manifest.format,
        (
            replace(manifest.plates[0], source_file_sha256=ledger.rejected[0].source_file_sha256),
            *manifest.plates[1:],
        ),
    ).validate()
    reused_ledger = replace(ledger, source_manifest_sha256=reused_source.fingerprint()).validate()
    with pytest.raises(ValueError, match="source bytes"):
        compose_population(reused_source, plates, rights, reused_ledger)


def test_base_plate_loader_detects_source_replacement_during_decode(tmp_path: Path) -> None:
    path = tmp_path / "plate.bin"
    path.write_bytes(b"authoring-source")
    plate = _plate(0)
    receipt = BasePlateReceipt(
        cluster_id="identity-01",
        archive_name="plates/identity-01.png",
        source_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        decoded_rgb_sha256=rgb_frame_sha256(plate),
        decoded_shape=(1024, 1024, 3),
        accepted_job_id="accepted-identity-01",
        executed_prompt_sha256=_digest("prompt"),
        workflow_sha256=_digest("workflow"),
        checkpoint_sha256=_digest("checkpoint"),
        authored_source_receipt_sha256=_digest("source"),
    )

    assert np.array_equal(
        load_content_addressed_base_plate(path, receipt, decoder=lambda _: plate), plate
    )

    def mutate(_: Path) -> NDArray[np.uint8]:
        path.write_bytes(b"replaced")
        return plate

    with pytest.raises(ValueError, match="changed during decode"):
        load_content_addressed_base_plate(path, receipt, decoder=mutate)


def test_materialized_population_preserves_task_one_pair_hash_contract() -> None:
    manifest, plates, rights, ledger = _sources()
    population = compose_population(manifest, plates, rights, ledger)

    def preprocess(frame: NDArray[np.uint8]) -> torch.Tensor:
        return torch.full((3, 384, 384), float(frame.sum()), dtype=torch.float32)

    def encode(value: torch.Tensor) -> torch.Tensor:
        return torch.full((1, 128, 1, 12, 12), value[0, 0, 0].item(), dtype=torch.float32)

    guides: dict[str, tuple[MaterializedSceneGuides, MaterializedSceneGuides]] = {}
    for cluster in population.clusters:
        left, right = cluster.variants
        guides[cluster.spec.cluster_id] = (
            materialize_rgb_frames(
                left.frames.frames,
                video_preprocess=preprocess,
                image_conditioner=lambda value: value,
                video_encoder=encode,
            ),
            materialize_rgb_frames(
                right.frames.frames,
                video_preprocess=preprocess,
                image_conditioner=lambda value: value,
                video_encoder=encode,
            ),
        )
    materialized_population = validate_materialized_population(population, guides)

    assert len(materialized_population.clusters) == 18
    for protocol_cluster in materialized_population.clusters:
        protocol_left, protocol_right = protocol_cluster.variants
        assert all(
            protocol_left.preprocessed_tensor_sha256[slot]
            == protocol_right.preprocessed_tensor_sha256[slot]
            and protocol_left.vae_latent_sha256[slot] == protocol_right.vae_latent_sha256[slot]
            for slot in FACT_ABSENT_SLOTS
        )

    def assert_intermediate_drift(changed_manifest: MaterializationManifest) -> None:
        changed_guide = replace(materialized_population.guides[0], manifest=changed_manifest)
        with pytest.raises(ValueError, match="intermediate receipt drift"):
            replace(
                materialized_population,
                guides=(changed_guide, *materialized_population.guides[1:]),
            ).validate()

    guide_manifest = materialized_population.guides[0].manifest
    assert_intermediate_drift(
        replace(
            guide_manifest,
            conditioner_tensor_sha256=(
                _digest("conditioner"),
                *guide_manifest.conditioner_tensor_sha256[1:],
            ),
        )
    )
    with pytest.raises(ValueError, match="source parent drift"):
        replace(
            materialized_population,
            manifest=replace(
                materialized_population.manifest,
                source_manifest_sha256=_digest("different-source-parent"),
            ),
        ).validate()
    with pytest.raises(ValueError, match="RGB parent drift"):
        replace(
            materialized_population,
            manifest=replace(
                materialized_population.manifest,
                rgb_frame_manifest_sha256=_digest("different-rgb-parent"),
            ),
        ).validate()
    assert_intermediate_drift(
        replace(
            guide_manifest,
            raw_encoder_tensor_sha256=(
                _digest("raw"),
                *guide_manifest.raw_encoder_tensor_sha256[1:],
            ),
        )
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: tuple(np.zeros((384, 384, 3), dtype=np.uint8) for _ in range(8)), "nine"),
        (lambda: tuple(np.zeros((384, 384, 3), dtype=np.float32) for _ in range(9)), "RGB"),
    ],
)
def test_materializer_rejects_invalid_rgb_inputs(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        materialize_rgb_frames(
            factory(),  # type: ignore[operator]
            video_preprocess=lambda frame: torch.zeros((3, 384, 384)),
            image_conditioner=lambda value: value,
            video_encoder=lambda value: torch.zeros((1, 128, 1, 12, 12)),
        )


@pytest.mark.parametrize(
    ("preprocess", "condition", "encode"),
    [
        (
            lambda frame: torch.ones((3, 384, 383)),
            lambda value: value,
            lambda value: torch.ones((1, 128, 1, 12, 12)),
        ),
        (
            lambda frame: torch.ones((3, 384, 384), requires_grad=True),
            lambda value: value,
            lambda value: torch.ones((1, 128, 1, 12, 12)),
        ),
        (
            lambda frame: torch.ones((3, 384, 384)),
            lambda value: value,
            lambda value: torch.full((1, 128, 1, 12, 12), float("nan")),
        ),
    ],
)
def test_materializer_rejects_bad_intermediate_tensors(
    preprocess: object, condition: object, encode: object
) -> None:
    frames = tuple(np.zeros((384, 384, 3), dtype=np.uint8) for _ in range(9))

    with pytest.raises(ValueError, match="finite detached"):
        materialize_rgb_frames(
            frames,
            video_preprocess=preprocess,  # type: ignore[arg-type]
            image_conditioner=condition,  # type: ignore[arg-type]
            video_encoder=encode,  # type: ignore[arg-type]
        )
