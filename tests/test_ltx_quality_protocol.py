from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import pytest

from comfy_story.ltx_quality_protocol import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CATEGORY_CLUSTER_COUNT,
    CONFIRMATORY_CLUSTER_COUNT,
    CONFIRMATORY_SCENE_COUNT,
    CSR_DELTA_THRESHOLD,
    FACT_ABSENT_SLOTS,
    FACT_ELIGIBLE_SLOTS,
    GENERATION_CELL_COUNT,
    MATCHED_SEEDS,
    METHOD_IDS,
    PAIRWISE_THRESHOLD,
    PROTECTED_SLOTS,
    REVIEW_PAIR_COUNT,
    Ballot,
    BlindPackageSealReceipt,
    BlindPresentationSide,
    BlindReviewPackage,
    BootstrapSettings,
    CellState,
    ClusterAggregate,
    CommittedGenerationCell,
    CommittedGenerationMatrix,
    CounterfactualCluster,
    Defect,
    GenerationApparatusFailure,
    GenerationCellKey,
    GenerationCellLifecycle,
    GuideSource,
    GuideSourceKind,
    Method,
    MethodGuideReceipt,
    QualityCategory,
    QualityProtocol,
    ReviewComparison,
    ReviewEvidenceBinding,
    ReviewPhase,
    ReviewPhaseHistory,
    ReviewPresentationReceipt,
    RuntimeCoordinate,
    SceneVariant,
    ScoredReviewPair,
    SidePreference,
    Verdict,
    VerdictEvidence,
    answer_key_sha256,
    canonical_json,
    canonical_sha256,
    classify_verdict,
    committed_generation_manifest_sha256,
    csr_delta_meets_threshold,
    derive_cluster_aggregates,
    from_canonical_json,
    scene_manifest_sha256,
    scored_review_manifest_sha256,
    side_assignment_commitment_sha256,
    side_assignment_methods,
    validate_confirmatory_clusters,
    validate_generation_cell_keys,
)

_HASHES = tuple(f"{index:064x}" for index in range(1, 80))
_SIDE_ASSIGNMENT_SECRET = "confirmatory-side-assignment-secret-20260830"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _variant(
    cluster_index: int,
    variant_id: str,
    category: QualityCategory,
    *,
    visible_slot: int,
) -> SceneVariant:
    offset = 0 if variant_id == "A" else 20
    rgb = list(_HASHES[0:9])
    tensors = list(_HASHES[10:19])
    latents = list(_HASHES[30:39])
    if variant_id == "B":
        rgb[visible_slot] = _HASHES[offset + 40]
        tensors[visible_slot] = _HASHES[offset + 41]
        latents[visible_slot] = _HASHES[offset + 42]
    fact_value, counterfactual = (
        ("striped badge", "dotted badge")
        if variant_id == "A"
        else ("dotted badge", "striped badge")
    )
    return SceneVariant(
        scene_id=f"cluster-{cluster_index:02d}-{variant_id.lower()}",
        cluster_id=f"cluster-{cluster_index:02d}",
        variant_id=variant_id,
        category=category,
        fact_attribute="badge pattern",
        fact_value=fact_value,
        counterfactual_value=counterfactual,
        accepted_values=("striped badge", "dotted badge"),
        fact_visible_slot=visible_slot,
        fact_absent_slots=FACT_ABSENT_SLOTS,
        prompt="Reveal the same marking that was shown before.",
        negative_prompt="text, logo, watermark",
        forbidden_fact_terms=("striped badge", "dotted badge"),
        base_plate_sha256=_HASHES[70],
        rgb_frame_sha256=tuple(rgb),
        preprocessed_tensor_sha256=tuple(tensors),
        vae_latent_sha256=tuple(latents),
        scoring_region=(32, 48, 224, 256),
        reveal_interval=(12, 21),
        objective_fact_rubric="The revealed badge has the historical pattern.",
        source_workflow_sha256=_HASHES[71],
        rights_receipt_sha256=_HASHES[72],
        predecessor_exclusion_ledger_sha256=_HASHES[73],
    ).validate()


def _cluster(index: int, category: QualityCategory) -> CounterfactualCluster:
    visible_slot = FACT_ELIGIBLE_SLOTS[index % len(FACT_ELIGIBLE_SLOTS)]
    return CounterfactualCluster(
        cluster_id=f"cluster-{index:02d}",
        category=category,
        variants=(
            _variant(index, "A", category, visible_slot=visible_slot),
            _variant(index, "B", category, visible_slot=visible_slot),
        ),
    ).validate()


def _confirmatory_clusters() -> tuple[CounterfactualCluster, ...]:
    categories = (
        (QualityCategory.IDENTITY_DETAIL,) * 6
        + (QualityCategory.OBJECT_WORLD_STATE,) * 6
        + (QualityCategory.TEMPORAL_CONTINUITY,) * 6
    )
    return tuple(_cluster(index, category) for index, category in enumerate(categories))


def _committed_cell(
    key: GenerationCellKey,
    scene: SceneVariant,
    runtime: RuntimeCoordinate | None = None,
    manifest_sha256: str | None = None,
) -> CommittedGenerationCell:
    coordinate = runtime or RuntimeCoordinate.default()
    scene_manifest = manifest_sha256 or canonical_sha256((scene.to_dict(),))
    if key.method is Method.FULL_HISTORY:
        sources = tuple(
            GuideSource(GuideSourceKind.SCENE_VAE_LATENT, index, digest)
            for index, digest in enumerate(scene.vae_latent_sha256)
        )
        core_input_sha256 = None
    elif key.method is Method.RECENT_ANCHOR:
        sources = tuple(
            GuideSource(GuideSourceKind.SCENE_VAE_LATENT, index, scene.vae_latent_sha256[index])
            for index in (7, 2, 5, 8)
        )
        core_input_sha256 = None
    else:
        sources = (
            GuideSource(
                GuideSourceKind.METHOD_CORE,
                None,
                _digest(f"{key.scene_id}-{key.seed}-{key.method.value}-core"),
            ),
            *(
                GuideSource(
                    GuideSourceKind.SCENE_VAE_LATENT,
                    index,
                    scene.vae_latent_sha256[index],
                )
                for index in (2, 5, 8)
            ),
        )
        core_input_sha256 = canonical_sha256(
            {
                "format": "duet-x-method-core-input-v1",
                "method": key.method.value,
                "trainable_checkpoint_sha256": (
                    QualityProtocol.default().trainable_checkpoint_sha256
                ),
                "scene_manifest_sha256": scene_manifest,
                "scene_variant_sha256": scene.fingerprint(),
                "history_vae_latent_sha256": tuple(
                    scene.vae_latent_sha256[index] for index in (0, 1, 3, 4, 6, 7)
                ),
            }
        )
    guide_receipt = MethodGuideReceipt(
        key=key,
        scene_manifest_sha256=scene_manifest,
        scene_variant_sha256=scene.fingerprint(),
        sources=sources,
        core_input_sha256=core_input_sha256,
    )
    lifecycle = GenerationCellLifecycle(
        key,
        tuple(CellState),
    )
    return CommittedGenerationCell(
        key=key,
        lifecycle=lifecycle,
        guide_receipt=guide_receipt,
        initial_noise_sha256=_digest(f"{key.scene_id}-{key.seed}-noise"),
        final_latent_sha256=_digest(f"{key}-final-latent"),
        decoded_frames_sha256=_digest(f"{key}-decoded-frames"),
        media_sha256=_digest(f"{key}-media"),
        duration_seconds=1.0,
        peak_memory_bytes=1024,
        wall_time_seconds=2.0,
        model_foundations_before_sha256=_digest("model-foundations"),
        model_foundations_after_sha256=_digest("model-foundations"),
        runtime_coordinate_sha256=coordinate.fingerprint(),
        runtime_identity_sha256=_digest("host-runtime-receipt"),
        host_id="forge-1" if key.seed != 303 else "forge-2",
        lifecycle_authorization_sha256=_digest("confirmatory-lifecycle-authorization"),
        protocol_sha256=QualityProtocol.default().fingerprint(),
    ).validate()


def _committed_matrix() -> CommittedGenerationMatrix:
    clusters = _confirmatory_clusters()
    scenes = tuple(variant for cluster in clusters for variant in cluster.variants)
    scene_ids = tuple(sorted(scene.scene_id for scene in scenes))
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    runtime = RuntimeCoordinate.default()
    scene_manifest = scene_manifest_sha256(clusters)
    cells = tuple(
        _committed_cell(
            GenerationCellKey(scene_id, seed, method),
            scene_by_id[scene_id],
            runtime,
            scene_manifest,
        )
        for scene_id in scene_ids
        for seed in MATCHED_SEEDS
        for method in Method
    )
    generation_manifest = committed_generation_manifest_sha256(cells, clusters, runtime)
    return CommittedGenerationMatrix(
        scene_ids=scene_ids,
        clusters=clusters,
        scene_manifest_sha256=scene_manifest,
        runtime=runtime,
        confirmatory_forge_receipt_sha256=("a" * 64, "b" * 64),
        cells=cells,
        first_finalization_sha256=generation_manifest,
        second_finalization_sha256=generation_manifest,
    ).validate()


def _preference(
    score: float,
    target: Method,
    left_method: Method,
) -> SidePreference:
    if score == 0.5:
        return SidePreference.TIE
    selected = (
        target
        if score == 1.0
        else {
            Method.DUET_CORE: Method.GATED_CORE,
            Method.FULL_HISTORY: Method.RECENT_ANCHOR,
        }[target]
    )
    return SidePreference.LEFT if selected is left_method else SidePreference.RIGHT


def _presentation(
    generation: CommittedGenerationMatrix,
    key: GenerationCellKey,
    pair_id: str,
    side: str,
) -> ReviewPresentationReceipt:
    cell = next(cell for cell in generation.cells if cell.key == key)
    return ReviewPresentationReceipt(
        cell_key=key,
        source_media_sha256=cell.media_sha256,
        normalized_media_sha256=_digest(f"{pair_id}-{side}-normalized"),
        codec="test-h264",
        width=384,
        height=384,
        frames=25,
        fps=24,
        audio_policy="absent",
        metadata_policy="identifying_metadata_stripped",
    ).validate()


def _scored_pairs(
    generation: CommittedGenerationMatrix,
    changes_by_cluster: Mapping[int, Mapping[str, object]] | None = None,
    secret: str = _SIDE_ASSIGNMENT_SECRET,
) -> tuple[ScoredReviewPair, ...]:
    changes = changes_by_cluster or {}
    pairs: list[ScoredReviewPair] = []
    for cluster_index in range(18):
        settings: dict[str, object] = {
            "fact_duet": True,
            "fact_gated": False,
            "primary_preference": 1.0,
            "prompt_duet": True,
            "prompt_gated": True,
            "fact_full": True,
            "fact_recent": False,
            "calibration_preference": 1.0,
            "duet_defects": (),
        }
        settings.update(changes.get(cluster_index, {}))
        category = tuple(QualityCategory)[cluster_index // 6]
        for variant_offset in range(2):
            scene_id = f"cluster-{cluster_index:02d}-{('a', 'b')[variant_offset]}"
            for seed in MATCHED_SEEDS:
                primary_id = f"primary-{scene_id}-{seed}"
                primary_left, primary_right = side_assignment_methods(
                    secret,
                    generation.first_finalization_sha256,
                    scene_id,
                    seed,
                    ReviewComparison.PRIMARY,
                )
                primary_fact = {
                    Method.DUET_CORE: cast(bool, settings["fact_duet"]),
                    Method.GATED_CORE: cast(bool, settings["fact_gated"]),
                }
                primary_prompt = {
                    Method.DUET_CORE: cast(bool, settings["prompt_duet"]),
                    Method.GATED_CORE: cast(bool, settings["prompt_gated"]),
                }
                primary_defects = {
                    Method.DUET_CORE: cast(tuple[Defect, ...], settings["duet_defects"]),
                    Method.GATED_CORE: (),
                }
                primary_ballot = Ballot(
                    pair_id=primary_id,
                    rater_id="authorized-rater-1",
                    left_fact_present=primary_fact[primary_left],
                    right_fact_present=primary_fact[primary_right],
                    left_prompt_adherent=primary_prompt[primary_left],
                    right_prompt_adherent=primary_prompt[primary_right],
                    fact_preference=_preference(
                        cast(float, settings["primary_preference"]),
                        Method.DUET_CORE,
                        primary_left,
                    ),
                    continuity_preference=SidePreference.TIE,
                    left_major_defects=primary_defects[primary_left],
                    right_major_defects=primary_defects[primary_right],
                )
                pairs.append(
                    ScoredReviewPair(
                        pair_id=primary_id,
                        scene_id=scene_id,
                        cluster_id=f"cluster-{cluster_index:02d}",
                        category=category,
                        seed=seed,
                        comparison=ReviewComparison.PRIMARY,
                        left_method=primary_left,
                        right_method=primary_right,
                        left_presentation=_presentation(
                            generation,
                            GenerationCellKey(scene_id, seed, primary_left),
                            primary_id,
                            "left",
                        ),
                        right_presentation=_presentation(
                            generation,
                            GenerationCellKey(scene_id, seed, primary_right),
                            primary_id,
                            "right",
                        ),
                        ballot=primary_ballot,
                    ).validate()
                )
                calibration_id = f"calibration-{scene_id}-{seed}"
                calibration_left, calibration_right = side_assignment_methods(
                    secret,
                    generation.first_finalization_sha256,
                    scene_id,
                    seed,
                    ReviewComparison.CALIBRATION,
                )
                calibration_fact = {
                    Method.FULL_HISTORY: cast(bool, settings["fact_full"]),
                    Method.RECENT_ANCHOR: cast(bool, settings["fact_recent"]),
                }
                calibration_ballot = Ballot(
                    pair_id=calibration_id,
                    rater_id="authorized-rater-1",
                    left_fact_present=calibration_fact[calibration_left],
                    right_fact_present=calibration_fact[calibration_right],
                    left_prompt_adherent=True,
                    right_prompt_adherent=True,
                    fact_preference=_preference(
                        cast(float, settings["calibration_preference"]),
                        Method.FULL_HISTORY,
                        calibration_left,
                    ),
                    continuity_preference=SidePreference.TIE,
                    left_major_defects=(),
                    right_major_defects=(),
                )
                pairs.append(
                    ScoredReviewPair(
                        pair_id=calibration_id,
                        scene_id=scene_id,
                        cluster_id=f"cluster-{cluster_index:02d}",
                        category=category,
                        seed=seed,
                        comparison=ReviewComparison.CALIBRATION,
                        left_method=calibration_left,
                        right_method=calibration_right,
                        left_presentation=_presentation(
                            generation,
                            GenerationCellKey(scene_id, seed, calibration_left),
                            calibration_id,
                            "left",
                        ),
                        right_presentation=_presentation(
                            generation,
                            GenerationCellKey(scene_id, seed, calibration_right),
                            calibration_id,
                            "right",
                        ),
                        ballot=calibration_ballot,
                    ).validate()
                )
    return tuple(sorted(pairs, key=lambda pair: pair.canonical_key))


def _blind_package(
    generation: CommittedGenerationMatrix,
    pairs: tuple[ScoredReviewPair, ...],
    secret: str,
) -> BlindReviewPackage:
    commitment = side_assignment_commitment_sha256(
        secret,
        generation.first_finalization_sha256,
    )
    sides = tuple(
        sorted(
            (
                side
                for pair in pairs
                for side in (
                    BlindPresentationSide.from_presentation(
                        pair.pair_id, "left", pair.left_presentation
                    ),
                    BlindPresentationSide.from_presentation(
                        pair.pair_id, "right", pair.right_presentation
                    ),
                )
            ),
            key=lambda side: (side.pair_id, side.side),
        )
    )
    return BlindReviewPackage(
        generation_manifest_sha256=generation.first_finalization_sha256,
        side_assignment_commitment_sha256=commitment,
        sides=sides,
    ).validate()


def _verdict_evidence(
    *,
    changes_by_cluster: Mapping[int, Mapping[str, object]] | None = None,
    matrix: CommittedGenerationMatrix | None = None,
    secret: str = _SIDE_ASSIGNMENT_SECRET,
) -> tuple[VerdictEvidence, str]:
    generation = matrix or _committed_matrix()
    scored_pairs = _scored_pairs(generation, changes_by_cluster, secret)
    phases = tuple(ReviewPhase)[: tuple(ReviewPhase).index(ReviewPhase.ANSWER_KEY_REVEALED) + 1]
    review = ReviewEvidenceBinding(
        phase_history=ReviewPhaseHistory(phases),
        scored_pairs=scored_pairs,
        scored_pairs_sha256=scored_review_manifest_sha256(scored_pairs),
        answer_key_sha256=answer_key_sha256(scored_pairs),
        side_assignment_secret=secret,
        side_assignment_commitment_sha256=side_assignment_commitment_sha256(
            secret,
            generation.first_finalization_sha256,
        ),
        generation_manifest_sha256=generation.first_finalization_sha256,
        blind_package=_blind_package(generation, scored_pairs, secret),
    ).validate()
    seal = BlindPackageSealReceipt.for_package(review.blind_package)
    evidence = VerdictEvidence(
        generation=generation,
        review=review,
        bootstrap=BootstrapSettings.default(),
        blind_package_seal=seal,
    )
    return evidence, seal.fingerprint()


def test_default_protocol_freezes_the_complete_confirmatory_coordinate() -> None:
    protocol = QualityProtocol.default().validate()

    assert tuple(category.value for category in protocol.categories) == (
        "identity_detail",
        "object_world_state",
        "temporal_continuity",
    )
    assert tuple(method.value for method in protocol.methods) == (
        "full_history",
        "recent_anchor",
        "gated_core",
        "duet_core",
    )
    assert METHOD_IDS == ("full_history", "recent_anchor", "gated_core", "duet_core")
    assert MATCHED_SEEDS == (101, 202, 303)
    assert PROTECTED_SLOTS == (2, 5)
    assert FACT_ELIGIBLE_SLOTS == (0, 1, 3, 4)
    assert FACT_ABSENT_SLOTS == (2, 5, 6, 7, 8)
    assert (
        CONFIRMATORY_CLUSTER_COUNT,
        CONFIRMATORY_SCENE_COUNT,
        CATEGORY_CLUSTER_COUNT,
        GENERATION_CELL_COUNT,
        REVIEW_PAIR_COUNT,
    ) == (18, 36, 6, 432, 216)
    assert (CSR_DELTA_THRESHOLD, PAIRWISE_THRESHOLD) == (0.10, 0.60)
    assert (protocol.bootstrap.samples, protocol.bootstrap.seed) == (
        BOOTSTRAP_SAMPLES,
        BOOTSTRAP_SEED,
    )
    assert (
        protocol.runtime.pipeline,
        protocol.runtime.guide_lattice,
        protocol.runtime.width,
        protocol.runtime.height,
        protocol.runtime.frames,
        protocol.runtime.fps,
        protocol.runtime.inference_steps,
        protocol.runtime.dtype,
        protocol.runtime.batch_size,
    ) == ("TI2VidOneStagePipeline", (12, 12), 384, 384, 25, 24, 30, "bfloat16", 1)
    assert len(protocol.fingerprint()) == 64


def test_protocol_rejects_confirmatory_configuration_drift_and_unknown_fields() -> None:
    protocol = QualityProtocol.default()
    with pytest.raises(ValueError, match="frozen"):
        dataclasses.replace(protocol, csr_delta_threshold=0.05).validate()
    with pytest.raises(ValueError, match="frozen"):
        dataclasses.replace(protocol, pairwise_threshold=0.55).validate()

    raw = protocol.to_dict()
    raw["post_hoc_threshold"] = 0.05
    with pytest.raises(ValueError, match="unknown field"):
        QualityProtocol.from_dict(raw)


def test_canonical_serialization_is_finite_strict_and_rejects_duplicate_json_keys() -> None:
    protocol = QualityProtocol.default()
    encoded = canonical_json(protocol)

    assert encoded == canonical_json(protocol.to_dict())
    assert canonical_sha256(protocol) == protocol.fingerprint()
    assert json.loads(encoded)["methods"] == [
        "full_history",
        "recent_anchor",
        "gated_core",
        "duet_core",
    ]
    with pytest.raises(ValueError, match="finite"):
        canonical_json({"value": float("nan")})
    with pytest.raises(ValueError, match="duplicate key"):
        from_canonical_json(b'{"samples":10000,"samples":9999}', BootstrapSettings)


def test_invalid_direct_records_and_mutable_sequences_cannot_be_canonical_evidence() -> None:
    invalid_key = GenerationCellKey("scene", 404, Method.DUET_CORE)
    with pytest.raises(ValueError, match="frozen seed"):
        invalid_key.to_dict()
    with pytest.raises(ValueError, match="frozen seed"):
        canonical_json(invalid_key)

    ballot = Ballot(
        pair_id="opaque-pair-001",
        rater_id="authorized-rater-1",
        left_fact_present=True,
        right_fact_present=False,
        left_prompt_adherent=True,
        right_prompt_adherent=True,
        fact_preference=SidePreference.LEFT,
        continuity_preference=SidePreference.TIE,
        left_major_defects=cast(Any, [Defect.FACT_SUBSTITUTION]),
        right_major_defects=cast(Any, []),
    )
    with pytest.raises(ValueError, match="tuple"):
        ballot.validate()
    with pytest.raises(ValueError, match="tuple"):
        ballot.to_json()


def test_scene_and_counterfactual_pair_validate_fact_isolation_and_exact_equality() -> None:
    cluster = _cluster(0, QualityCategory.IDENTITY_DETAIL)
    assert cluster.variants[0].fact_visible_slot == 0
    assert cluster.variants[0].prompt_bytes == b"Reveal the same marking that was shown before."

    leaked = dataclasses.replace(cluster.variants[0], prompt="Reveal the striped badge.")
    with pytest.raises(ValueError, match="fact leakage"):
        leaked.validate()
    unequal = dataclasses.replace(
        cluster.variants[1],
        rgb_frame_sha256=(
            *cluster.variants[1].rgb_frame_sha256[:2],
            _HASHES[50],
            *cluster.variants[1].rgb_frame_sha256[3:],
        ),
    )
    with pytest.raises(ValueError, match="shared slot"):
        dataclasses.replace(cluster, variants=(cluster.variants[0], unequal)).validate()
    equal_fact = dataclasses.replace(
        cluster.variants[1],
        rgb_frame_sha256=cluster.variants[0].rgb_frame_sha256,
    )
    with pytest.raises(ValueError, match="fact-visible"):
        dataclasses.replace(cluster, variants=(cluster.variants[0], equal_fact)).validate()


def test_confirmatory_cluster_grid_rejects_duplicates_counts_and_category_imbalance() -> None:
    clusters = _confirmatory_clusters()
    validated = validate_confirmatory_clusters(clusters)
    assert len(validated) == 18
    assert sum(len(cluster.variants) for cluster in validated) == 36

    with pytest.raises(ValueError, match="exactly 18"):
        validate_confirmatory_clusters(clusters[:-1])
    with pytest.raises(ValueError, match="duplicate cluster"):
        validate_confirmatory_clusters((*clusters[:-1], clusters[0]))
    imbalanced = dataclasses.replace(clusters[-1], category=QualityCategory.IDENTITY_DETAIL)
    with pytest.raises(ValueError, match="six clusters"):
        validate_confirmatory_clusters((*clusters[:-1], imbalanced))


def test_generation_cell_key_and_append_only_lifecycle_are_strict() -> None:
    key = GenerationCellKey("cluster-00-a", 101, Method.DUET_CORE).validate()
    lifecycle = GenerationCellLifecycle(key, (CellState.ABSENT,)).validate()

    assert lifecycle.may_execute_model_forward
    lifecycle = lifecycle.transition(CellState.STARTED)
    assert not lifecycle.may_execute_model_forward
    lifecycle = lifecycle.transition(CellState.ARTIFACT_INSTALLED)
    assert lifecycle.may_finalize_disk_only
    lifecycle = lifecycle.transition(CellState.COMMITTED)
    assert lifecycle.at_committed_state
    assert not hasattr(lifecycle, "is_committed")
    with pytest.raises(ValueError, match="transition"):
        lifecycle.transition(CellState.STARTED)
    with pytest.raises(ValueError, match="frozen seed"):
        dataclasses.replace(key, seed=404).validate()
    with pytest.raises(ValueError, match="prefix"):
        GenerationCellLifecycle(key, (CellState.ABSENT, CellState.ARTIFACT_INSTALLED)).validate()


def test_generation_cell_grid_requires_the_exact_unique_cartesian_product() -> None:
    scene_ids = tuple(f"scene-{index:02d}" for index in range(36))
    keys = tuple(
        GenerationCellKey(scene_id, seed, method)
        for scene_id in scene_ids
        for seed in MATCHED_SEEDS
        for method in Method
    )

    assert len(validate_generation_cell_keys(keys, scene_ids)) == 432
    with pytest.raises(ValueError, match="432 unique"):
        validate_generation_cell_keys((*keys[:-1], keys[0]), scene_ids)


def test_committed_cell_binds_artifact_execution_and_runtime_evidence() -> None:
    scene = _variant(0, "A", QualityCategory.IDENTITY_DETAIL, visible_slot=0)
    key = GenerationCellKey(scene.scene_id, 101, Method.DUET_CORE)
    committed = _committed_cell(key, scene)

    assert committed.lifecycle.at_committed_state
    assert len(committed.guide_receipt.sources) == 4
    with pytest.raises(ValueError, match="committed lifecycle"):
        dataclasses.replace(
            committed,
            lifecycle=GenerationCellLifecycle(key, (CellState.ABSENT, CellState.STARTED)),
        ).validate()
    with pytest.raises(ValueError, match="final latent"):
        dataclasses.replace(committed, final_latent_sha256="").validate()
    with pytest.raises(ValueError, match="foundation mutation"):
        dataclasses.replace(
            committed,
            model_foundations_after_sha256=_digest("mutated-foundations"),
        ).validate()


def test_complete_committed_matrix_requires_432_bound_cells_and_two_identical_finalizations() -> (
    None
):
    matrix = _committed_matrix()

    assert len(matrix.cells) == 432
    assert matrix.first_finalization_sha256 == committed_generation_manifest_sha256(
        matrix.cells, matrix.clusters, matrix.runtime
    )
    with pytest.raises(ValueError, match="432 unique"):
        dataclasses.replace(matrix, cells=(*matrix.cells[:-1], matrix.cells[0])).validate()
    with pytest.raises(ValueError, match="disk-only finalizations"):
        dataclasses.replace(matrix, second_finalization_sha256=_digest("different")).validate()
    damaged = dataclasses.replace(matrix.cells[0], media_sha256="not-a-hash")
    with pytest.raises(ValueError, match="media_sha256"):
        dataclasses.replace(matrix, cells=(damaged, *matrix.cells[1:])).validate()

    drifted = dataclasses.replace(matrix.cells[0], initial_noise_sha256=_digest("drifted-noise"))
    drifted_cells = (drifted, *matrix.cells[1:])
    drifted_manifest = committed_generation_manifest_sha256(
        drifted_cells, matrix.clusters, matrix.runtime
    )
    with pytest.raises(ValueError, match="matched initial-noise"):
        dataclasses.replace(
            matrix,
            cells=drifted_cells,
            first_finalization_sha256=drifted_manifest,
            second_finalization_sha256=drifted_manifest,
        ).validate()


def test_committed_matrix_rejects_unrelated_scenes_guide_drift_and_nonfrozen_runtime() -> None:
    matrix = _committed_matrix()
    first = matrix.clusters[0]
    unrelated_variants = tuple(
        dataclasses.replace(
            variant,
            scene_id=f"unrelated-{variant.variant_id.lower()}",
            cluster_id="unrelated-cluster",
        )
        for variant in first.variants
    )
    unrelated_cluster = CounterfactualCluster(
        cluster_id="unrelated-cluster",
        category=first.category,
        variants=cast(tuple[SceneVariant, SceneVariant], unrelated_variants),
    ).validate()
    unrelated_clusters = (unrelated_cluster, *matrix.clusters[1:])
    with pytest.raises(ValueError, match=r"scene manifest|scene IDs"):
        dataclasses.replace(
            matrix,
            clusters=unrelated_clusters,
            scene_manifest_sha256=scene_manifest_sha256(unrelated_clusters),
        ).validate()

    cell = matrix.cells[1]
    wrong_sources = (
        *cell.guide_receipt.sources[:-1],
        dataclasses.replace(cell.guide_receipt.sources[-1], sha256=_digest("wrong-current-guide")),
    )
    changed_cell = dataclasses.replace(
        cell,
        guide_receipt=dataclasses.replace(cell.guide_receipt, sources=wrong_sources),
    )
    changed_cells = (matrix.cells[0], changed_cell, *matrix.cells[2:])
    changed_manifest = committed_generation_manifest_sha256(
        changed_cells, matrix.clusters, matrix.runtime
    )
    with pytest.raises(ValueError, match="guide receipt raw source"):
        dataclasses.replace(
            matrix,
            cells=changed_cells,
            first_finalization_sha256=changed_manifest,
            second_finalization_sha256=changed_manifest,
        ).validate()

    core_cell = matrix.cells[2]
    changed_core = dataclasses.replace(
        core_cell,
        guide_receipt=dataclasses.replace(
            core_cell.guide_receipt,
            core_input_sha256=_digest("wrong-core-input-binding"),
        ),
    )
    changed_core_cells = (*matrix.cells[:2], changed_core, *matrix.cells[3:])
    changed_core_manifest = committed_generation_manifest_sha256(
        changed_core_cells, matrix.clusters, matrix.runtime
    )
    with pytest.raises(ValueError, match="core input receipt"):
        dataclasses.replace(
            matrix,
            cells=changed_core_cells,
            first_finalization_sha256=changed_core_manifest,
            second_finalization_sha256=changed_core_manifest,
        ).validate()

    with pytest.raises(ValueError, match="runtime coordinate is frozen"):
        dataclasses.replace(
            matrix,
            runtime=dataclasses.replace(matrix.runtime, width=385),
        ).validate()


def test_method_guide_receipts_bind_exact_raw_layouts_and_frozen_core_inputs() -> None:
    matrix = _committed_matrix()
    scene = next(
        variant
        for cluster in matrix.clusters
        for variant in cluster.variants
        if variant.scene_id == matrix.scene_ids[0]
    )
    cells = {
        cell.key.method: cell
        for cell in matrix.cells
        if cell.key.scene_id == scene.scene_id and cell.key.seed == MATCHED_SEEDS[0]
    }

    assert tuple(
        source.scene_slot for source in cells[Method.FULL_HISTORY].guide_receipt.sources
    ) == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    )
    assert tuple(
        source.scene_slot for source in cells[Method.RECENT_ANCHOR].guide_receipt.sources
    ) == (
        7,
        2,
        5,
        8,
    )
    for method in (Method.GATED_CORE, Method.DUET_CORE):
        receipt = cells[method].guide_receipt
        assert receipt.sources[0].kind is GuideSourceKind.METHOD_CORE
        assert tuple(source.scene_slot for source in receipt.sources[1:]) == (2, 5, 8)
        assert receipt.core_input_sha256 is not None
        assert receipt.core_input_sha256 == canonical_sha256(
            {
                "format": "duet-x-method-core-input-v1",
                "method": method.value,
                "trainable_checkpoint_sha256": (
                    QualityProtocol.default().trainable_checkpoint_sha256
                ),
                "scene_manifest_sha256": matrix.scene_manifest_sha256,
                "scene_variant_sha256": scene.fingerprint(),
                "history_vae_latent_sha256": tuple(
                    scene.vae_latent_sha256[index] for index in (0, 1, 3, 4, 6, 7)
                ),
            }
        )

    recent = cells[Method.RECENT_ANCHOR]
    permuted_receipt = dataclasses.replace(
        recent.guide_receipt,
        sources=(
            recent.guide_receipt.sources[1],
            recent.guide_receipt.sources[0],
            *recent.guide_receipt.sources[2:],
        ),
    )
    permuted_cells = tuple(
        dataclasses.replace(cell, guide_receipt=permuted_receipt)
        if cell.key == recent.key
        else cell
        for cell in matrix.cells
    )
    permuted_manifest = committed_generation_manifest_sha256(
        permuted_cells, matrix.clusters, matrix.runtime
    )
    with pytest.raises(ValueError, match="guide receipt layout"):
        dataclasses.replace(
            matrix,
            cells=permuted_cells,
            first_finalization_sha256=permuted_manifest,
            second_finalization_sha256=permuted_manifest,
        ).validate()

    core = cells[Method.DUET_CORE]
    forged_core = dataclasses.replace(
        core.guide_receipt,
        core_input_sha256=_digest("arbitrary-core-input"),
    )
    forged_cells = tuple(
        dataclasses.replace(cell, guide_receipt=forged_core) if cell.key == core.key else cell
        for cell in matrix.cells
    )
    forged_manifest = committed_generation_manifest_sha256(
        forged_cells, matrix.clusters, matrix.runtime
    )
    with pytest.raises(ValueError, match="core input"):
        dataclasses.replace(
            matrix,
            cells=forged_cells,
            first_finalization_sha256=forged_manifest,
            second_finalization_sha256=forged_manifest,
        ).validate()

    protected_history_core = dataclasses.replace(
        core.guide_receipt,
        core_input_sha256=canonical_sha256(
            {
                "format": "duet-x-method-core-input-v1",
                "method": Method.DUET_CORE.value,
                "trainable_checkpoint_sha256": (
                    QualityProtocol.default().trainable_checkpoint_sha256
                ),
                "scene_manifest_sha256": matrix.scene_manifest_sha256,
                "scene_variant_sha256": scene.fingerprint(),
                "history_vae_latent_sha256": scene.vae_latent_sha256[:8],
            }
        ),
    )
    protected_history_cells = tuple(
        dataclasses.replace(cell, guide_receipt=protected_history_core)
        if cell.key == core.key
        else cell
        for cell in matrix.cells
    )
    protected_history_manifest = committed_generation_manifest_sha256(
        protected_history_cells, matrix.clusters, matrix.runtime
    )
    with pytest.raises(ValueError, match="core input"):
        dataclasses.replace(
            matrix,
            cells=protected_history_cells,
            first_finalization_sha256=protected_history_manifest,
            second_finalization_sha256=protected_history_manifest,
        ).validate()


def test_committed_matrix_allows_distinct_host_receipts_at_one_frozen_coordinate() -> None:
    matrix = _committed_matrix()
    changed_cell = dataclasses.replace(
        matrix.cells[0], runtime_identity_sha256=_digest("forge-1-host-receipt")
    )
    changed_cells = (changed_cell, *matrix.cells[1:])
    manifest = committed_generation_manifest_sha256(changed_cells, matrix.clusters, matrix.runtime)

    assert dataclasses.replace(
        matrix,
        cells=changed_cells,
        first_finalization_sha256=manifest,
        second_finalization_sha256=manifest,
    ).validate()


def test_started_cell_without_artifact_has_a_typed_terminal_apparatus_failure() -> None:
    key = GenerationCellKey("scene-00", 101, Method.DUET_CORE)
    failure = GenerationApparatusFailure(
        lifecycle=GenerationCellLifecycle(key, (CellState.ABSENT, CellState.STARTED)),
        reason="started_without_valid_installed_artifact",
    ).validate()

    assert failure.lifecycle.current is CellState.STARTED
    with pytest.raises(ValueError, match="started lifecycle"):
        dataclasses.replace(
            failure,
            lifecycle=GenerationCellLifecycle(key, tuple(CellState)),
        ).validate()


def test_review_phase_history_accepts_only_the_authenticated_chain() -> None:
    history = ReviewPhaseHistory((ReviewPhase.SOURCE_AUTHORED,)).validate()
    for phase in tuple(ReviewPhase)[1:]:
        history = history.transition(phase)

    assert history.current is ReviewPhase.FINALIZED
    with pytest.raises(ValueError, match="transition"):
        history.transition(ReviewPhase.ANSWER_KEY_REVEALED)
    with pytest.raises(ValueError, match="prefix"):
        ReviewPhaseHistory((ReviewPhase.SOURCE_AUTHORED, ReviewPhase.PREREGISTERED)).validate()


def test_ballot_fields_freeze_choices_taxonomy_and_single_rater_shape() -> None:
    ballot = Ballot(
        pair_id="opaque-pair-001",
        rater_id="authorized-rater-1",
        left_fact_present=True,
        right_fact_present=False,
        left_prompt_adherent=True,
        right_prompt_adherent=True,
        fact_preference=SidePreference.LEFT,
        continuity_preference=SidePreference.TIE,
        left_major_defects=(Defect.FACT_SUBSTITUTION,),
        right_major_defects=(),
    ).validate()

    assert Ballot.from_dict(ballot.to_dict()) == ballot
    with pytest.raises(ValueError, match="duplicate defect"):
        dataclasses.replace(
            ballot,
            left_major_defects=(Defect.FACT_SUBSTITUTION, Defect.FACT_SUBSTITUTION),
        ).validate()
    raw = ballot.to_dict()
    raw["technical_failure"] = True
    with pytest.raises(ValueError, match="unknown field"):
        Ballot.from_dict(raw)


def test_verdict_derives_invalid_evidence_from_the_committed_matrix() -> None:
    matrix = _committed_matrix()
    damaged = dataclasses.replace(matrix, second_finalization_sha256=_digest("different"))
    evidence, anchor = _verdict_evidence(matrix=damaged)

    assert classify_verdict(evidence, anchor).verdict is Verdict.INVALID_EVIDENCE


def test_verdict_cannot_false_pass_from_incomplete_ballots_or_caller_aggregate() -> None:
    evidence, anchor = _verdict_evidence()
    incomplete_pairs = evidence.review.scored_pairs[:-1]
    incomplete_review = dataclasses.replace(
        evidence.review,
        scored_pairs=incomplete_pairs,
        scored_pairs_sha256=scored_review_manifest_sha256(incomplete_pairs),
    )

    assert (
        classify_verdict(dataclasses.replace(evidence, review=incomplete_review), anchor).verdict
        is Verdict.INVALID_EVIDENCE
    )
    invented_aggregate = derive_cluster_aggregates(evidence.review, evidence.generation)[0]
    assert classify_verdict(invented_aggregate, anchor).verdict is Verdict.INVALID_EVIDENCE


def test_verdict_rejects_swapped_review_sides_with_the_sealed_commitment_retained() -> None:
    evidence, anchor = _verdict_evidence()
    original = evidence.review.scored_pairs[1]
    swapped = dataclasses.replace(
        original,
        left_method=original.right_method,
        right_method=original.left_method,
        left_presentation=original.right_presentation,
        right_presentation=original.left_presentation,
    ).validate()
    swapped_pairs = (
        evidence.review.scored_pairs[0],
        swapped,
        *evidence.review.scored_pairs[2:],
    )
    review = dataclasses.replace(
        evidence.review,
        scored_pairs=swapped_pairs,
        scored_pairs_sha256=scored_review_manifest_sha256(swapped_pairs),
    )

    assert answer_key_sha256(swapped_pairs) != evidence.review.answer_key_sha256
    assert (
        classify_verdict(dataclasses.replace(evidence, review=review), anchor).verdict
        is Verdict.INVALID_EVIDENCE
    )


def test_verdict_rejects_media_or_normalized_presentation_not_bound_to_its_cell() -> None:
    evidence, anchor = _verdict_evidence()
    original = evidence.review.scored_pairs[1]
    substituted = dataclasses.replace(
        original,
        left_presentation=dataclasses.replace(
            original.left_presentation,
            source_media_sha256=_digest("unrelated-source-media"),
            normalized_media_sha256=_digest("unrelated-normalized-media"),
        ),
    ).validate()
    substituted_pairs = (
        evidence.review.scored_pairs[0],
        substituted,
        *evidence.review.scored_pairs[2:],
    )
    review = dataclasses.replace(
        evidence.review,
        scored_pairs=substituted_pairs,
        scored_pairs_sha256=scored_review_manifest_sha256(substituted_pairs),
        answer_key_sha256=answer_key_sha256(substituted_pairs),
        side_assignment_commitment_sha256=side_assignment_commitment_sha256(
            evidence.review.side_assignment_secret,
            evidence.generation.first_finalization_sha256,
        ),
    )

    assert (
        classify_verdict(dataclasses.replace(evidence, review=review), anchor).verdict
        is Verdict.INVALID_EVIDENCE
    )


def test_verdict_rejects_normalized_side_swap_or_secret_change_against_sealed_package() -> None:
    evidence, anchor = _verdict_evidence()
    original = evidence.review.scored_pairs[1]
    swapped_normalized = dataclasses.replace(
        original,
        left_presentation=dataclasses.replace(
            original.left_presentation,
            normalized_media_sha256=original.right_presentation.normalized_media_sha256,
        ),
    ).validate()
    swapped_pairs = (
        evidence.review.scored_pairs[0],
        swapped_normalized,
        *evidence.review.scored_pairs[2:],
    )
    review = dataclasses.replace(
        evidence.review,
        scored_pairs=swapped_pairs,
        scored_pairs_sha256=scored_review_manifest_sha256(swapped_pairs),
        answer_key_sha256=answer_key_sha256(swapped_pairs),
        blind_package=evidence.review.blind_package,
    )

    assert (
        classify_verdict(dataclasses.replace(evidence, review=review), anchor).verdict
        is Verdict.INVALID_EVIDENCE
    )
    changed_secret = dataclasses.replace(
        evidence.review,
        side_assignment_secret="a-different-revealed-secret",
        blind_package=evidence.review.blind_package,
    )
    assert (
        classify_verdict(dataclasses.replace(evidence, review=changed_secret), anchor).verdict
        is Verdict.INVALID_EVIDENCE
    )


def test_verdict_requires_an_externally_preserved_pre_ballot_seal_anchor() -> None:
    earlier, authority_anchor = _verdict_evidence()
    replacement, regenerated_candidate_anchor = _verdict_evidence(
        secret="replacement-side-assignment-secret"
    )

    assert isinstance(earlier.blind_package_seal, BlindPackageSealReceipt)
    assert regenerated_candidate_anchor != authority_anchor
    assert not hasattr(replacement, "external_blind_package_seal_anchor_sha256")
    with pytest.raises(TypeError):
        cast(Any, classify_verdict)(replacement)
    assert classify_verdict(replacement, authority_anchor).verdict is Verdict.INVALID_EVIDENCE


def test_verdict_rejects_scored_pair_cluster_or_category_relabeling() -> None:
    evidence, anchor = _verdict_evidence()
    relabeled_pairs = tuple(
        dataclasses.replace(
            pair,
            cluster_id=f"renamed-{pair.cluster_id}",
            category=tuple(QualityCategory)[
                (tuple(QualityCategory).index(pair.category) + 1) % len(QualityCategory)
            ],
        )
        for pair in evidence.review.scored_pairs
    )
    relabeled_review = dataclasses.replace(
        evidence.review,
        scored_pairs=relabeled_pairs,
        scored_pairs_sha256=scored_review_manifest_sha256(relabeled_pairs),
    )

    assert (
        classify_verdict(dataclasses.replace(evidence, review=relabeled_review), anchor).verdict
        is Verdict.INVALID_EVIDENCE
    )


def test_verdict_derives_calibration_systematic_defect_and_prompt_branches() -> None:
    uninformative = {
        index: {
            "calibration_preference": 0.5,
            "fact_full": True,
            "fact_recent": True,
        }
        for index in range(18)
    }
    uninformative_evidence, anchor = _verdict_evidence(changes_by_cluster=uninformative)
    assert (
        classify_verdict(uninformative_evidence, anchor).verdict
        is Verdict.HUMAN_EVALUATION_UNINFORMATIVE
    )

    defects = {index: {"duet_defects": (Defect.FACT_SUBSTITUTION,)} for index in range(3)}
    defect_evidence, anchor = _verdict_evidence(changes_by_cluster=defects)
    assert classify_verdict(defect_evidence, anchor).verdict is Verdict.SYSTEMATIC_DEFECT

    prompt_failure = {index: {"prompt_duet": False} for index in range(18)}
    prompt_evidence, anchor = _verdict_evidence(changes_by_cluster=prompt_failure)
    assert classify_verdict(prompt_evidence, anchor).verdict is Verdict.PROMPT_ADHERENCE_FAILED


def test_verdict_derives_advance_tie_and_negative_branches_from_cluster_aggregates() -> None:
    evidence, anchor = _verdict_evidence()
    assert len(derive_cluster_aggregates(evidence.review, evidence.generation)) == 18
    assert classify_verdict(evidence, anchor).verdict is Verdict.ADVANCE_QUALITY_FIRST

    tie = {
        index: {"fact_duet": True, "fact_gated": True, "primary_preference": 0.5}
        for index in range(18)
    }
    tie_evidence, anchor = _verdict_evidence(changes_by_cluster=tie)
    assert classify_verdict(tie_evidence, anchor).verdict is Verdict.CLOSE_LTX_TIE

    category_guardrail = {
        index: {"fact_duet": True, "fact_gated": True, "primary_preference": 0.5}
        for index in range(6, 18)
    }
    guardrail_evidence, anchor = _verdict_evidence(changes_by_cluster=category_guardrail)
    assert classify_verdict(guardrail_evidence, anchor).verdict is Verdict.CLOSE_LTX_TIE

    negative = {
        index: {"fact_duet": False, "fact_gated": True, "primary_preference": 0.0}
        for index in range(18)
    }
    negative_evidence, anchor = _verdict_evidence(changes_by_cluster=negative)
    assert classify_verdict(negative_evidence, anchor).verdict is Verdict.CLOSE_LTX_NEGATIVE


def test_verdict_uses_derived_cluster_bootstrap_not_a_caller_supplied_interval() -> None:
    heterogeneous = {
        index: {
            "fact_duet": True,
            "fact_gated": True,
            "primary_preference": 0.0 if index < 7 else 1.0,
        }
        for index in range(18)
    }
    evidence, anchor = _verdict_evidence(changes_by_cluster=heterogeneous)
    result = classify_verdict(evidence, anchor)

    assert result.pairwise_duet == pytest.approx(11.0 / 18.0)
    assert result.pairwise_lower_95 < 0.50
    assert result.verdict is Verdict.CLOSE_LTX_TIE


def test_csr_exact_decimal_boundary_advances_without_pairwise_superiority() -> None:
    assert csr_delta_meets_threshold(0.30, 0.20)
    assert not csr_delta_meets_threshold(0.299_999, 0.20)


def test_cluster_aggregate_rejects_nonfinite_and_mutable_defects() -> None:
    base = ClusterAggregate(
        cluster_id="cluster-00",
        category=QualityCategory.IDENTITY_DETAIL,
        csr_full=1.0,
        csr_recent=0.0,
        csr_gated=0.0,
        csr_duet=1.0,
        primary_pairwise_duet=1.0,
        calibration_pairwise_full=1.0,
        prompt_adherence_gated=1.0,
        prompt_adherence_duet=1.0,
        duet_defects=(),
    ).validate()
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(base, csr_duet=float("inf")).validate()
    with pytest.raises(ValueError, match="tuple"):
        dataclasses.replace(
            base,
            duet_defects=cast(Any, [Defect.FACT_SUBSTITUTION]),
        ).validate()
