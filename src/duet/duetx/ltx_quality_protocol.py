"""Immutable contracts for the frozen Duet-X decoded-quality gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any, Self, TypeVar, cast, get_args, get_origin, get_type_hints

import numpy as np
import numpy.typing as npt

QUALITY_PROTOCOL_FORMAT = "duet-x-ltx-decoded-quality-v1"
FROZEN_TRAINABLE_CHECKPOINT_SHA256 = (
    "c40a90d38319ed4019b79ebc0b175c9a8f094dcc007d375ad8397ca1cd18a30f"
)
LTX_DEVELOPMENT_CHECKPOINT_SHA256 = (
    "7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f"
)
LTX_SOURCE_COMMIT = "598ab41247a77dbfe29b5186e915bcf4f9040ec7"

MATCHED_SEEDS = (101, 202, 303)
PROTECTED_SLOTS = (2, 5)
FACT_ELIGIBLE_SLOTS = (0, 1, 3, 4)
FACT_ABSENT_SLOTS = (2, 5, 6, 7, 8)
CONFIRMATORY_CLUSTER_COUNT = 18
CONFIRMATORY_SCENE_COUNT = 36
CATEGORY_CLUSTER_COUNT = 6
GENERATION_CELL_COUNT = 432
REVIEW_PAIR_COUNT = 216
COMPLETE_RATER_COUNT = 1
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_830
CSR_DELTA_THRESHOLD = 0.10
PAIRWISE_THRESHOLD = 0.60
PAIRWISE_CI_LOWER_THRESHOLD = 0.50
PROMPT_ADHERENCE_DELTA_FLOOR = -0.02
SYSTEMATIC_DEFECT_CLUSTER_COUNT = 3
CATEGORY_DIRECTION_MINIMUM = 2

_SHA256_HEX = frozenset("0123456789abcdef")


class QualityCategory(StrEnum):
    """The three equally weighted confirmatory scene categories."""

    IDENTITY_DETAIL = "identity_detail"
    OBJECT_WORLD_STATE = "object_world_state"
    TEMPORAL_CONTINUITY = "temporal_continuity"


class QualityMethod(StrEnum):
    """The four frozen methods in the decoded primary matrix."""

    FULL_HISTORY = "full_history"
    RECENT_ANCHOR = "recent_anchor"
    GATED_CORE = "gated_core"
    DUET_CORE = "duet_core"


Method = QualityMethod
METHOD_IDS = tuple(method.value for method in QualityMethod)


class CellState(StrEnum):
    """One append-only exactly-once generation-cell state."""

    ABSENT = "absent"
    STARTED = "started"
    ARTIFACT_INSTALLED = "artifact_installed"
    COMMITTED = "committed"


class ReviewPhase(StrEnum):
    """One authenticated evidence phase."""

    SOURCE_AUTHORED = "source_authored"
    DEVELOPMENT_REHEARSED = "development_rehearsed"
    PREREGISTERED = "preregistered"
    GENERATION_COMPLETE = "generation_complete"
    BLIND_PACKAGE_SEALED = "blind_package_sealed"
    BALLOTS_SEALED = "ballots_sealed"
    ANSWER_KEY_REVEALED = "answer_key_revealed"
    FINALIZED = "finalized"


class SidePreference(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    TIE = "tie"


class ReviewComparison(StrEnum):
    PRIMARY = "primary"
    CALIBRATION = "calibration"


class Defect(StrEnum):
    """The exhaustive major quality-defect taxonomy."""

    IDENTITY_CORRUPTION = "identity_corruption"
    FACT_SUBSTITUTION = "fact_substitution"
    TEMPORAL_DISCONTINUITY = "temporal_discontinuity"
    PROMPT_CONTRADICTION = "prompt_contradiction"
    RENDERING_COLLAPSE = "rendering_collapse"


class Verdict(StrEnum):
    """Final decoded-quality verdicts in precedence order."""

    INVALID_EVIDENCE = "invalid_evidence"
    HUMAN_EVALUATION_UNINFORMATIVE = "human_evaluation_uninformative"
    SYSTEMATIC_DEFECT = "systematic_defect"
    PROMPT_ADHERENCE_FAILED = "prompt_adherence_failed"
    ADVANCE_QUALITY_FIRST = "advance_quality_first"
    CLOSE_LTX_TIE = "close_ltx_tie"
    CLOSE_LTX_NEGATIVE = "close_ltx_negative"


_RecordT = TypeVar("_RecordT", bound="FrozenRecord")


def _canonical_value(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, FrozenRecord):
        value.validate()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical JSON object keys must be strings")
            result[key] = _canonical_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical JSON values must be finite")
        return value
    raise ValueError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: object) -> bytes:
    """Encode a value in the protocol's sole finite canonical JSON representation."""
    try:
        return json.dumps(
            _canonical_value(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except ValueError as error:
        if "Out of range float" in str(error):
            raise ValueError("canonical JSON values must be finite") from error
        raise


def canonical_sha256(value: object) -> str:
    """Hash the canonical JSON bytes for a protocol record or JSON value."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in canonical JSON: {key}")
        result[key] = value
    return result


def from_canonical_json(data: bytes | str, cls: type[_RecordT]) -> _RecordT:
    """Decode strict JSON into a validated record, rejecting duplicate object keys."""
    try:
        raw = json.loads(data, object_pairs_hook=_object_without_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError("record is not valid JSON") from error
    return cls.from_dict(raw)


def _decode_value(annotation: object, raw: object, name: str) -> object:
    origin = get_origin(annotation)
    if origin is tuple:
        if not isinstance(raw, (tuple, list)):
            raise ValueError(f"{name} must be an array")
        arguments = get_args(annotation)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_decode_value(arguments[0], item, f"{name} item") for item in raw)
        if len(raw) != len(arguments):
            raise ValueError(f"{name} must contain exactly {len(arguments)} items")
        return tuple(
            _decode_value(expected, item, f"{name}[{index}]")
            for index, (expected, item) in enumerate(zip(arguments, raw, strict=True))
        )
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        if not isinstance(raw, str):
            raise ValueError(f"{name} must be a string enum value")
        try:
            return annotation(raw)
        except ValueError as error:
            raise ValueError(f"{name} has an unknown value: {raw!r}") from error
    if isinstance(annotation, type) and issubclass(annotation, FrozenRecord):
        return _decode_record(annotation, raw, name)
    if annotation in {str, int, float, bool}:
        if type(raw) is not annotation:
            expected = cast(type[object], annotation).__name__
            raise ValueError(f"{name} must have exact type {expected}")
        return raw
    raise TypeError(f"unsupported record annotation for {name}: {annotation!r}")


def _decode_record(cls: type[FrozenRecord], raw: object, name: str) -> FrozenRecord:
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{name} must be an object")
    record_fields = fields(cast(Any, cls))
    field_names = {field.name for field in record_fields}
    unknown = set(raw) - field_names
    if unknown:
        raise ValueError(f"unknown field: {name}.{sorted(unknown)[0]}")
    missing = field_names - set(raw)
    if missing:
        raise ValueError(f"missing field: {name}.{sorted(missing)[0]}")
    annotations = get_type_hints(cls)
    values = {
        field.name: _decode_value(annotations[field.name], raw[field.name], f"{name}.{field.name}")
        for field in record_fields
    }
    instance = cls(**values)
    return instance.validate()


class FrozenRecord:
    """Strict parsing and canonical serialization shared by frozen records."""

    @classmethod
    def from_dict(cls: type[_RecordT], raw: object) -> _RecordT:
        return cast(_RecordT, _decode_record(cls, raw, cls.__name__))

    def validate(self) -> Self:
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:
        self.validate()
        value = _canonical_value(self)
        if not isinstance(value, dict):
            raise RuntimeError("record serialization did not produce an object")
        return value

    def to_json(self) -> bytes:
        self.validate()
        return canonical_json(self)

    def fingerprint(self) -> str:
        self.validate()
        return canonical_sha256(self)


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty trimmed string")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_unit(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return value


def _require_tuple(value: object, name: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be an immutable tuple")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeCoordinate(FrozenRecord):
    """The exact successful one-stage decoded generation coordinate."""

    pipeline: str
    model_checkpoint_sha256: str
    source_commit: str
    guide_lattice: tuple[int, int]
    width: int
    height: int
    frames: int
    fps: int
    inference_steps: int
    dtype: str
    batch_size: int
    scheduler: str
    attention_backend: str
    decoder: str
    reference_boundary: str
    reference_downscale_factor: int
    reference_temporal_scale_factor: int
    reference_strength: float
    reference_attention_mask: float
    identical_settings_within_scene_seed: bool

    @classmethod
    def default(cls) -> Self:
        return cls(
            pipeline="TI2VidOneStagePipeline",
            model_checkpoint_sha256=LTX_DEVELOPMENT_CHECKPOINT_SHA256,
            source_commit=LTX_SOURCE_COMMIT,
            guide_lattice=(12, 12),
            width=384,
            height=384,
            frames=25,
            fps=24,
            inference_steps=30,
            dtype="bfloat16",
            batch_size=1,
            scheduler="checkpoint_pinned_flow_matching",
            attention_backend="pinned_ltx_source_default",
            decoder="official_ltx_video_decoder",
            reference_boundary="official_reference_conditioning",
            reference_downscale_factor=1,
            reference_temporal_scale_factor=1,
            reference_strength=1.0,
            reference_attention_mask=1.0,
            identical_settings_within_scene_seed=True,
        )

    def validate(self) -> Self:
        if self != type(self).default():
            raise ValueError("decoded runtime coordinate is frozen")
        return self


@dataclass(frozen=True, slots=True)
class BootstrapSettings(FrozenRecord):
    """The sole permissible cluster-bootstrap settings."""

    samples: int
    seed: int
    confidence: float
    quantile_method: str
    resampling_unit: str

    @classmethod
    def default(cls) -> Self:
        return cls(BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, 0.95, "linear", "cluster")

    def validate(self) -> Self:
        if self != type(self).default():
            raise ValueError("bootstrap settings are frozen")
        return self


@dataclass(frozen=True, slots=True)
class SceneVariant(FrozenRecord):
    """One content-addressed member of a counterfactual A/B scene pair."""

    scene_id: str
    cluster_id: str
    variant_id: str
    category: QualityCategory
    fact_attribute: str
    fact_value: str
    counterfactual_value: str
    accepted_values: tuple[str, str]
    fact_visible_slot: int
    fact_absent_slots: tuple[int, ...]
    prompt: str
    negative_prompt: str
    forbidden_fact_terms: tuple[str, ...]
    base_plate_sha256: str
    rgb_frame_sha256: tuple[str, ...]
    preprocessed_tensor_sha256: tuple[str, ...]
    vae_latent_sha256: tuple[str, ...]
    scoring_region: tuple[int, int, int, int]
    reveal_interval: tuple[int, int]
    objective_fact_rubric: str
    source_workflow_sha256: str
    rights_receipt_sha256: str
    predecessor_exclusion_ledger_sha256: str

    @property
    def prompt_bytes(self) -> bytes:
        return self.prompt.encode("utf-8")

    @property
    def negative_prompt_bytes(self) -> bytes:
        return self.negative_prompt.encode("utf-8")

    def validate(self) -> Self:
        for name in (
            "scene_id",
            "cluster_id",
            "fact_attribute",
            "fact_value",
            "counterfactual_value",
            "prompt",
            "negative_prompt",
            "objective_fact_rubric",
        ):
            _nonempty(getattr(self, name), name)
        if self.variant_id not in {"A", "B"}:
            raise ValueError("variant_id must be A or B")
        if not isinstance(self.category, QualityCategory):
            raise ValueError("category must be a frozen quality category")
        _require_tuple(self.accepted_values, "accepted_values")
        if len(set(self.accepted_values)) != 2 or set(self.accepted_values) != {
            self.fact_value,
            self.counterfactual_value,
        }:
            raise ValueError("accepted_values must contain the two distinct counterfactual values")
        if (
            type(self.fact_visible_slot) is not int
            or self.fact_visible_slot not in FACT_ELIGIBLE_SLOTS
        ):
            raise ValueError("fact-visible slot must be one frozen old unprotected slot")
        _require_tuple(self.fact_absent_slots, "fact_absent_slots")
        if self.fact_absent_slots != FACT_ABSENT_SLOTS:
            raise ValueError("fact-absent slots are frozen")
        _require_tuple(self.forbidden_fact_terms, "forbidden_fact_terms")
        if not self.forbidden_fact_terms or len(
            {term.casefold() for term in self.forbidden_fact_terms}
        ) != len(self.forbidden_fact_terms):
            raise ValueError("forbidden fact terms must be nonempty and unique")
        for term in self.forbidden_fact_terms:
            _nonempty(term, "forbidden fact term")
        if not {self.fact_value.casefold(), self.counterfactual_value.casefold()} <= {
            term.casefold() for term in self.forbidden_fact_terms
        }:
            raise ValueError("forbidden fact terms must cover both counterfactual values")
        prompt_text = f"{self.prompt}\n{self.negative_prompt}".casefold()
        if any(term.casefold() in prompt_text for term in self.forbidden_fact_terms):
            raise ValueError("prompt fact leakage is forbidden")
        _sha256(self.base_plate_sha256, "base_plate_sha256")
        for name in (
            "rgb_frame_sha256",
            "preprocessed_tensor_sha256",
            "vae_latent_sha256",
        ):
            values = getattr(self, name)
            _require_tuple(values, name)
            if len(values) != 9:
                raise ValueError(f"{name} must bind exactly nine guide slots")
            for index, digest in enumerate(values):
                _sha256(digest, f"{name}[{index}]")
        _require_tuple(self.scoring_region, "scoring_region")
        left, top, right, bottom = self.scoring_region
        if not (0 <= left < right <= 384 and 0 <= top < bottom <= 384):
            raise ValueError("scoring_region must have positive area inside 384x384")
        _require_tuple(self.reveal_interval, "reveal_interval")
        reveal_start, reveal_stop = self.reveal_interval
        if not (0 <= reveal_start < reveal_stop <= 25):
            raise ValueError("reveal_interval must be nonempty inside 25 frames")
        for name in (
            "source_workflow_sha256",
            "rights_receipt_sha256",
            "predecessor_exclusion_ledger_sha256",
        ):
            _sha256(getattr(self, name), name)
        return self


@dataclass(frozen=True, slots=True)
class CounterfactualCluster(FrozenRecord):
    """The independent inferential unit containing variants A and B."""

    cluster_id: str
    category: QualityCategory
    variants: tuple[SceneVariant, SceneVariant]

    def validate(self) -> Self:
        _nonempty(self.cluster_id, "cluster_id")
        if not isinstance(self.category, QualityCategory):
            raise ValueError("category must be a frozen quality category")
        _require_tuple(self.variants, "variants")
        left, right = self.variants
        left.validate()
        right.validate()
        if (left.variant_id, right.variant_id) != ("A", "B"):
            raise ValueError("cluster variants must be ordered A then B")
        if left.scene_id == right.scene_id:
            raise ValueError("cluster variants must have distinct scene IDs")
        if any(
            variant.cluster_id != self.cluster_id or variant.category is not self.category
            for variant in self.variants
        ):
            raise ValueError("cluster identity/category must match both variants")
        shared_fields = (
            "fact_attribute",
            "accepted_values",
            "fact_visible_slot",
            "fact_absent_slots",
            "prompt",
            "negative_prompt",
            "forbidden_fact_terms",
            "base_plate_sha256",
            "scoring_region",
            "reveal_interval",
            "objective_fact_rubric",
            "source_workflow_sha256",
            "rights_receipt_sha256",
            "predecessor_exclusion_ledger_sha256",
        )
        if any(getattr(left, name) != getattr(right, name) for name in shared_fields):
            raise ValueError("counterfactual variants disagree on a shared field")
        if (
            left.fact_value != right.counterfactual_value
            or right.fact_value != left.counterfactual_value
        ):
            raise ValueError("counterfactual fact values must swap between A and B")
        visible_slot = left.fact_visible_slot
        for name in (
            "rgb_frame_sha256",
            "preprocessed_tensor_sha256",
            "vae_latent_sha256",
        ):
            left_hashes = getattr(left, name)
            right_hashes = getattr(right, name)
            for slot in range(9):
                if slot == visible_slot:
                    if left_hashes[slot] == right_hashes[slot]:
                        raise ValueError("fact-visible bytes must differ across A/B")
                elif left_hashes[slot] != right_hashes[slot]:
                    if slot in FACT_ABSENT_SLOTS:
                        raise ValueError("A/B pair equality failed at a shared slot")
                    raise ValueError("variants may differ only at the fact-visible slot")
        return self


def validate_confirmatory_clusters(
    clusters: Sequence[CounterfactualCluster],
) -> tuple[CounterfactualCluster, ...]:
    """Validate the complete balanced 18-cluster, 36-variant population."""
    values = tuple(clusters)
    if len(values) != CONFIRMATORY_CLUSTER_COUNT:
        raise ValueError("confirmatory population requires exactly 18 clusters")
    cluster_ids = tuple(cluster.cluster_id for cluster in values)
    if len(set(cluster_ids)) != len(cluster_ids):
        raise ValueError("duplicate cluster key")
    categories = tuple(cluster.category for cluster in values)
    if any(categories.count(category) != CATEGORY_CLUSTER_COUNT for category in QualityCategory):
        raise ValueError("confirmatory population requires exactly six clusters per category")
    scene_ids = tuple(variant.scene_id for cluster in values for variant in cluster.variants)
    if len(scene_ids) != CONFIRMATORY_SCENE_COUNT or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("scene keys must contain exactly 36 unique variants")
    for cluster in values:
        cluster.validate()
    return values


def scene_manifest_sha256(clusters: Sequence[CounterfactualCluster]) -> str:
    """Hash the complete validated counterfactual scene population."""
    values = validate_confirmatory_clusters(clusters)
    return canonical_sha256(tuple(cluster.to_dict() for cluster in values))


@dataclass(frozen=True, slots=True)
class GenerationCellKey(FrozenRecord):
    """The immutable identity of one exactly-once generation cell."""

    scene_id: str
    seed: int
    method: Method

    def validate(self) -> Self:
        _nonempty(self.scene_id, "scene_id")
        if type(self.seed) is not int or self.seed not in MATCHED_SEEDS:
            raise ValueError("generation cell seed must be a frozen seed")
        if not isinstance(self.method, Method):
            raise ValueError("generation cell method must be frozen")
        return self


@dataclass(frozen=True, slots=True)
class GenerationCellLifecycle(FrozenRecord):
    """The complete append-only state chain recorded for one cell."""

    key: GenerationCellKey
    states: tuple[CellState, ...]

    def validate(self) -> Self:
        self.key.validate()
        _require_tuple(self.states, "states")
        chain = tuple(CellState)
        if not self.states or self.states != chain[: len(self.states)]:
            raise ValueError("generation lifecycle must be a prefix of the frozen state chain")
        return self

    @property
    def current(self) -> CellState:
        self.validate()
        return self.states[-1]

    @property
    def may_execute_model_forward(self) -> bool:
        return self.current is CellState.ABSENT

    @property
    def may_finalize_disk_only(self) -> bool:
        return self.current is CellState.ARTIFACT_INSTALLED

    @property
    def at_committed_state(self) -> bool:
        """Report state only; artifact-backed commitment requires CommittedGenerationCell."""
        return self.current is CellState.COMMITTED

    def transition(self, next_state: CellState) -> Self:
        self.validate()
        chain = tuple(CellState)
        expected_index = len(self.states)
        if expected_index >= len(chain) or next_state is not chain[expected_index]:
            raise ValueError(f"invalid generation lifecycle transition to {next_state.value}")
        return type(self)(self.key, (*self.states, next_state)).validate()


@dataclass(frozen=True, slots=True)
class GenerationApparatusFailure(FrozenRecord):
    """Terminal exactly-once failure after a forward started without a valid artifact."""

    lifecycle: GenerationCellLifecycle
    reason: str

    def validate(self) -> Self:
        if not isinstance(self.lifecycle, GenerationCellLifecycle):
            raise ValueError("apparatus failure requires a generation lifecycle")
        self.lifecycle.validate()
        if self.lifecycle.current is not CellState.STARTED:
            raise ValueError("terminal apparatus failure requires a started lifecycle")
        if self.reason != "started_without_valid_installed_artifact":
            raise ValueError("terminal apparatus failure reason is frozen")
        return self


class GuideSourceKind(StrEnum):
    """The two provenance classes permitted in a method-guide receipt."""

    SCENE_VAE_LATENT = "scene_vae_latent"
    METHOD_CORE = "method_core"


@dataclass(frozen=True, slots=True)
class GuideSource(FrozenRecord):
    """One ordered guide input, either raw scene data or a method-produced core."""

    kind: GuideSourceKind
    scene_slot: int | None
    sha256: str

    def validate(self) -> Self:
        if not isinstance(self.kind, GuideSourceKind):
            raise ValueError("guide source kind must be frozen")
        _sha256(self.sha256, "guide source SHA-256")
        if self.kind is GuideSourceKind.SCENE_VAE_LATENT:
            if type(self.scene_slot) is not int or self.scene_slot not in range(9):
                raise ValueError("raw guide source must name one scene VAE slot")
        elif self.scene_slot is not None:
            raise ValueError("method core guide source cannot name a scene slot")
        return self


@dataclass(frozen=True, slots=True)
class MethodGuideReceipt(FrozenRecord):
    """Ordered guide provenance bound to one scene, method, and core input when applicable."""

    key: GenerationCellKey
    scene_manifest_sha256: str
    scene_variant_sha256: str
    sources: tuple[GuideSource, ...]
    core_input_sha256: str | None

    def validate(self) -> Self:
        if not isinstance(self.key, GenerationCellKey):
            raise ValueError("method guide receipt key is malformed")
        self.key.validate()
        _sha256(self.scene_manifest_sha256, "scene_manifest_sha256")
        _sha256(self.scene_variant_sha256, "scene_variant_sha256")
        _require_tuple(self.sources, "guide sources")
        expected_count = 9 if self.key.method is Method.FULL_HISTORY else 4
        if len(self.sources) != expected_count:
            raise ValueError(f"{self.key.method.value} guide receipt has the wrong source count")
        for source in self.sources:
            if not isinstance(source, GuideSource):
                raise ValueError("guide receipt sources must be typed provenance records")
            source.validate()
        if self.core_input_sha256 is None:
            if self.key.method in {Method.GATED_CORE, Method.DUET_CORE}:
                raise ValueError("core methods require a frozen core input receipt")
        else:
            _sha256(self.core_input_sha256, "core_input_sha256")
            if self.key.method not in {Method.GATED_CORE, Method.DUET_CORE}:
                raise ValueError("raw-guide methods cannot claim a core input receipt")
        return self


@dataclass(frozen=True, slots=True)
class CommittedGenerationCell(FrozenRecord):
    """Artifact, execution, foundation, and runtime bindings for one committed cell."""

    key: GenerationCellKey
    lifecycle: GenerationCellLifecycle
    guide_receipt: MethodGuideReceipt
    initial_noise_sha256: str
    final_latent_sha256: str
    decoded_frames_sha256: str
    media_sha256: str
    duration_seconds: float
    peak_memory_bytes: int
    wall_time_seconds: float
    model_foundations_before_sha256: str
    model_foundations_after_sha256: str
    runtime_coordinate_sha256: str
    runtime_identity_sha256: str
    host_id: str
    lifecycle_authorization_sha256: str
    protocol_sha256: str

    def validate(self) -> Self:
        if not isinstance(self.key, GenerationCellKey):
            raise ValueError("committed cell key is malformed")
        self.key.validate()
        if not isinstance(self.lifecycle, GenerationCellLifecycle):
            raise ValueError("committed cell lifecycle is malformed")
        self.lifecycle.validate()
        if self.lifecycle.key != self.key or self.lifecycle.current is not CellState.COMMITTED:
            raise ValueError("committed cell requires its complete committed lifecycle")
        if not isinstance(self.guide_receipt, MethodGuideReceipt):
            raise ValueError("committed cell requires a typed method guide receipt")
        self.guide_receipt.validate()
        if self.guide_receipt.key != self.key:
            raise ValueError("committed cell guide receipt key does not match its cell")
        for name in (
            "initial_noise_sha256",
            "final_latent_sha256",
            "decoded_frames_sha256",
            "media_sha256",
            "model_foundations_before_sha256",
            "model_foundations_after_sha256",
            "runtime_coordinate_sha256",
            "runtime_identity_sha256",
            "lifecycle_authorization_sha256",
            "protocol_sha256",
        ):
            try:
                _sha256(getattr(self, name), name)
            except ValueError as error:
                if name == "final_latent_sha256":
                    raise ValueError("final latent SHA-256 binding is invalid") from error
                raise
        for name in ("duration_seconds", "wall_time_seconds"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive float")
        if type(self.peak_memory_bytes) is not int or self.peak_memory_bytes < 0:
            raise ValueError("peak_memory_bytes must be a nonnegative integer")
        if self.model_foundations_before_sha256 != self.model_foundations_after_sha256:
            raise ValueError("model foundation mutation invalidates a committed cell")
        _nonempty(self.host_id, "host_id")
        if self.protocol_sha256 != QualityProtocol.default().fingerprint():
            raise ValueError("committed cell protocol binding does not match the frozen protocol")
        return self


def _core_input_sha256(scene: SceneVariant, method: Method, manifest_sha256: str) -> str:
    """Bind a core to its frozen checkpoint, scene manifest, and ordered history inputs."""
    if method not in {Method.GATED_CORE, Method.DUET_CORE}:
        raise ValueError("only core methods have a core input receipt")
    return canonical_sha256(
        {
            "format": "duet-x-method-core-input-v1",
            "method": method.value,
            "trainable_checkpoint_sha256": FROZEN_TRAINABLE_CHECKPOINT_SHA256,
            "scene_manifest_sha256": manifest_sha256,
            "scene_variant_sha256": scene.fingerprint(),
            "history_vae_latent_sha256": tuple(
                scene.vae_latent_sha256[index] for index in (0, 1, 3, 4, 6, 7)
            ),
        }
    )


def committed_generation_manifest_sha256(
    cells: Sequence[CommittedGenerationCell],
    clusters: Sequence[CounterfactualCluster],
    runtime: RuntimeCoordinate,
) -> str:
    """Hash committed cells with their validated scene and runtime coordinates."""
    values = tuple(cells)
    for cell in values:
        cell.validate()
    cluster_values = validate_confirmatory_clusters(clusters)
    if not isinstance(runtime, RuntimeCoordinate):
        raise ValueError("generation manifest requires a frozen runtime coordinate")
    runtime.validate()
    return canonical_sha256(
        {
            "cells": tuple(cell.to_dict() for cell in values),
            "clusters": tuple(cluster.to_dict() for cluster in cluster_values),
            "runtime": runtime.to_dict(),
        }
    )


@dataclass(frozen=True, slots=True)
class CommittedGenerationMatrix(FrozenRecord):
    """Complete 432-cell evidence with two byte-identical disk-only finalizations."""

    scene_ids: tuple[str, ...]
    clusters: tuple[CounterfactualCluster, ...]
    scene_manifest_sha256: str
    runtime: RuntimeCoordinate
    confirmatory_forge_receipt_sha256: tuple[str, str]
    cells: tuple[CommittedGenerationCell, ...]
    first_finalization_sha256: str
    second_finalization_sha256: str

    def validate(self) -> Self:
        _require_tuple(self.scene_ids, "scene_ids")
        if self.scene_ids != tuple(sorted(self.scene_ids)):
            raise ValueError("generation matrix scene IDs must be canonically sorted")
        _require_tuple(self.clusters, "clusters")
        clusters = validate_confirmatory_clusters(self.clusters)
        expected_scene_ids = tuple(
            sorted(variant.scene_id for cluster in clusters for variant in cluster.variants)
        )
        if self.scene_ids != expected_scene_ids:
            raise ValueError("generation matrix scene IDs do not match the scene manifest")
        _sha256(self.scene_manifest_sha256, "scene_manifest_sha256")
        if self.scene_manifest_sha256 != scene_manifest_sha256(clusters):
            raise ValueError("generation matrix scene manifest hash does not match its clusters")
        if not isinstance(self.runtime, RuntimeCoordinate):
            raise ValueError("generation matrix requires a frozen runtime coordinate")
        self.runtime.validate()
        _require_tuple(
            self.confirmatory_forge_receipt_sha256,
            "confirmatory Forge receipt SHA-256",
        )
        if (
            len(self.confirmatory_forge_receipt_sha256) != 2
            or len(set(self.confirmatory_forge_receipt_sha256)) != 2
        ):
            raise ValueError("generation matrix requires two distinct Forge lifecycle receipts")
        for digest in self.confirmatory_forge_receipt_sha256:
            _sha256(digest, "confirmatory Forge receipt SHA-256")
        _require_tuple(self.cells, "cells")
        for cell in self.cells:
            if not isinstance(cell, CommittedGenerationCell):
                raise ValueError("generation matrix cells must be committed evidence records")
            cell.validate()
        validate_generation_cell_keys(tuple(cell.key for cell in self.cells), self.scene_ids)
        method_order = {method: index for index, method in enumerate(Method)}
        expected_order = tuple(
            sorted(
                self.cells,
                key=lambda cell: (
                    cell.key.scene_id,
                    MATCHED_SEEDS.index(cell.key.seed),
                    method_order[cell.key.method],
                ),
            )
        )
        if self.cells != expected_order:
            raise ValueError("committed generation cells must be in canonical key order")
        scene_by_id = {
            variant.scene_id: variant for cluster in clusters for variant in cluster.variants
        }
        runtime_sha256 = self.runtime.fingerprint()
        for cell in self.cells:
            scene = scene_by_id[cell.key.scene_id]
            receipt = cell.guide_receipt
            if receipt.scene_manifest_sha256 != self.scene_manifest_sha256:
                raise ValueError("committed cell guide receipt does not bind the scene manifest")
            if receipt.scene_variant_sha256 != scene.fingerprint():
                raise ValueError("committed cell does not match its scene manifest binding")
            if cell.runtime_coordinate_sha256 != runtime_sha256:
                raise ValueError("committed cell does not match the frozen runtime coordinate")
            _validate_method_guide_receipt(receipt, scene)
        if len({cell.model_foundations_before_sha256 for cell in self.cells}) != 1:
            raise ValueError("all committed cells must share one model-foundation identity")
        for scene_id in self.scene_ids:
            for seed in MATCHED_SEEDS:
                noise_identities = {
                    cell.initial_noise_sha256
                    for cell in self.cells
                    if cell.key.scene_id == scene_id and cell.key.seed == seed
                }
                if len(noise_identities) != 1:
                    raise ValueError("matched initial-noise identity drifted across methods")
        _sha256(self.first_finalization_sha256, "first_finalization_sha256")
        _sha256(self.second_finalization_sha256, "second_finalization_sha256")
        expected_sha256 = committed_generation_manifest_sha256(
            self.cells, self.clusters, self.runtime
        )
        if (
            self.first_finalization_sha256 != expected_sha256
            or self.second_finalization_sha256 != expected_sha256
        ):
            raise ValueError("two disk-only finalizations must be byte-identical to the manifest")
        return self


def _validate_method_guide_receipt(receipt: MethodGuideReceipt, scene: SceneVariant) -> None:
    """Validate the exact preregistered raw/core layouts against one source scene."""
    raw_slots: tuple[int, ...]
    if receipt.key.method is Method.FULL_HISTORY:
        raw_slots = tuple(range(9))
    elif receipt.key.method is Method.RECENT_ANCHOR:
        raw_slots = (7, 2, 5, 8)
    else:
        raw_slots = (2, 5, 8)
        core = receipt.sources[0]
        if core.kind is not GuideSourceKind.METHOD_CORE or core.scene_slot is not None:
            raise ValueError("core guide receipt layout must begin with one method core")
        expected_core = _core_input_sha256(scene, receipt.key.method, receipt.scene_manifest_sha256)
        if receipt.core_input_sha256 != expected_core:
            raise ValueError("core input receipt does not match frozen checkpoint and history")
    expected_sources = (
        receipt.sources
        if receipt.key.method
        in {
            Method.FULL_HISTORY,
            Method.RECENT_ANCHOR,
        }
        else receipt.sources[1:]
    )
    if tuple(source.scene_slot for source in expected_sources) != raw_slots:
        raise ValueError("guide receipt layout does not match the frozen raw source order")
    for source, slot in zip(expected_sources, raw_slots, strict=True):
        if (
            source.kind is not GuideSourceKind.SCENE_VAE_LATENT
            or source.sha256 != scene.vae_latent_sha256[slot]
        ):
            raise ValueError("guide receipt raw source does not match its scene VAE latent")


def validate_generation_cell_keys(
    keys: Sequence[GenerationCellKey], scene_ids: Sequence[str]
) -> tuple[GenerationCellKey, ...]:
    """Require the exact Cartesian matrix and reject duplicate or foreign keys."""
    scenes = tuple(scene_ids)
    if len(scenes) != CONFIRMATORY_SCENE_COUNT or len(set(scenes)) != len(scenes):
        raise ValueError("generation matrix requires exactly 36 unique scene IDs")
    values = tuple(keys)
    if len(values) != GENERATION_CELL_COUNT or len(set(values)) != len(values):
        raise ValueError("generation matrix requires exactly 432 unique cell keys")
    for key in values:
        key.validate()
    expected = {
        GenerationCellKey(scene_id, seed, method)
        for scene_id in scenes
        for seed in MATCHED_SEEDS
        for method in Method
    }
    if set(values) != expected:
        raise ValueError("generation cell keys do not match the frozen Cartesian matrix")
    return values


@dataclass(frozen=True, slots=True)
class ReviewPhaseHistory(FrozenRecord):
    """Authenticated append-only phase history for the evidence package."""

    phases: tuple[ReviewPhase, ...]

    def validate(self) -> Self:
        _require_tuple(self.phases, "phases")
        chain = tuple(ReviewPhase)
        if not self.phases or self.phases != chain[: len(self.phases)]:
            raise ValueError("review phase history must be a prefix of the frozen phase chain")
        return self

    @property
    def current(self) -> ReviewPhase:
        self.validate()
        return self.phases[-1]

    def transition(self, next_phase: ReviewPhase) -> Self:
        self.validate()
        chain = tuple(ReviewPhase)
        expected_index = len(self.phases)
        if expected_index >= len(chain) or next_phase is not chain[expected_index]:
            raise ValueError(f"invalid review phase transition to {next_phase.value}")
        return type(self)((*self.phases, next_phase)).validate()


@dataclass(frozen=True, slots=True)
class Ballot(FrozenRecord):
    """One complete blind A/B ballot from the single authorized PoC rater."""

    pair_id: str
    rater_id: str
    left_fact_present: bool
    right_fact_present: bool
    left_prompt_adherent: bool
    right_prompt_adherent: bool
    fact_preference: SidePreference
    continuity_preference: SidePreference
    left_major_defects: tuple[Defect, ...]
    right_major_defects: tuple[Defect, ...]

    def validate(self) -> Self:
        _nonempty(self.pair_id, "pair_id")
        _nonempty(self.rater_id, "rater_id")
        for name in (
            "left_fact_present",
            "right_fact_present",
            "left_prompt_adherent",
            "right_prompt_adherent",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if not isinstance(self.fact_preference, SidePreference) or not isinstance(
            self.continuity_preference, SidePreference
        ):
            raise ValueError("ballot preferences must be left, right, or tie")
        for name in ("left_major_defects", "right_major_defects"):
            values = getattr(self, name)
            _require_tuple(values, name)
            if any(not isinstance(value, Defect) for value in values):
                raise ValueError("major defects must use the frozen taxonomy")
            if len(values) != len(set(values)):
                raise ValueError("duplicate defect on one ballot side")
        return self


@dataclass(frozen=True, slots=True)
class ReviewPresentationReceipt(FrozenRecord):
    """Exact normalized blind-review asset derived from one committed media cell."""

    cell_key: GenerationCellKey
    source_media_sha256: str
    normalized_media_sha256: str
    codec: str
    width: int
    height: int
    frames: int
    fps: int
    audio_policy: str
    metadata_policy: str

    def validate(self) -> Self:
        if not isinstance(self.cell_key, GenerationCellKey):
            raise ValueError("review presentation cell key is malformed")
        self.cell_key.validate()
        _sha256(self.source_media_sha256, "source_media_sha256")
        _sha256(self.normalized_media_sha256, "normalized_media_sha256")
        _nonempty(self.codec, "codec")
        runtime = RuntimeCoordinate.default()
        if (
            self.width,
            self.height,
            self.frames,
            self.fps,
        ) != (runtime.width, runtime.height, runtime.frames, runtime.fps):
            raise ValueError("review presentation dimensions and timing are frozen")
        if self.audio_policy != "absent":
            raise ValueError("review presentation audio policy is frozen")
        if self.metadata_policy != "identifying_metadata_stripped":
            raise ValueError("review presentation metadata policy is frozen")
        return self


@dataclass(frozen=True, slots=True)
class BlindPresentationSide(FrozenRecord):
    """One public, method-free side of the package sealed before ballots are accepted."""

    pair_id: str
    side: str
    normalized_media_sha256: str
    codec: str
    width: int
    height: int
    frames: int
    fps: int
    audio_policy: str
    metadata_policy: str

    @classmethod
    def from_presentation(cls, pair_id: str, side: str, receipt: ReviewPresentationReceipt) -> Self:
        receipt.validate()
        return cls(
            pair_id=pair_id,
            side=side,
            normalized_media_sha256=receipt.normalized_media_sha256,
            codec=receipt.codec,
            width=receipt.width,
            height=receipt.height,
            frames=receipt.frames,
            fps=receipt.fps,
            audio_policy=receipt.audio_policy,
            metadata_policy=receipt.metadata_policy,
        )

    def validate(self) -> Self:
        _nonempty(self.pair_id, "blind presentation pair_id")
        if self.side not in {"left", "right"}:
            raise ValueError("blind presentation side must be left or right")
        _sha256(self.normalized_media_sha256, "normalized_media_sha256")
        _nonempty(self.codec, "codec")
        runtime = RuntimeCoordinate.default()
        if (self.width, self.height, self.frames, self.fps) != (
            runtime.width,
            runtime.height,
            runtime.frames,
            runtime.fps,
        ):
            raise ValueError("blind presentation dimensions and timing are frozen")
        if self.audio_policy != "absent":
            raise ValueError("blind presentation audio policy is frozen")
        if self.metadata_policy != "identifying_metadata_stripped":
            raise ValueError("blind presentation metadata policy is frozen")
        return self


@dataclass(frozen=True, slots=True)
class BlindReviewPackage(FrozenRecord):
    """The pre-ballot public presentation receipt; it intentionally has no method assignments."""

    generation_manifest_sha256: str
    side_assignment_commitment_sha256: str
    sides: tuple[BlindPresentationSide, ...]

    def validate(self) -> Self:
        _sha256(self.generation_manifest_sha256, "generation_manifest_sha256")
        _sha256(self.side_assignment_commitment_sha256, "side_assignment_commitment_sha256")
        _require_tuple(self.sides, "blind presentation sides")
        if len(self.sides) != REVIEW_PAIR_COUNT * 2:
            raise ValueError("blind package requires two public sides for every review pair")
        for side in self.sides:
            if not isinstance(side, BlindPresentationSide):
                raise ValueError("blind package sides must be typed presentation records")
            side.validate()
        keys = tuple((side.pair_id, side.side) for side in self.sides)
        if len(set(keys)) != len(keys):
            raise ValueError("blind package contains a duplicate public side")
        if keys != tuple(sorted(keys)):
            raise ValueError("blind package sides must be in canonical order")
        return self


@dataclass(frozen=True, slots=True)
class BlindPackageSealReceipt(FrozenRecord):
    """The pre-ballot package receipt whose fingerprint must be preserved outside this evidence."""

    protocol_sha256: str
    generation_manifest_sha256: str
    side_assignment_commitment_sha256: str
    blind_package_sha256: str
    blind_package_sealed_phase_sha256: str

    @classmethod
    def for_package(cls, package: BlindReviewPackage) -> Self:
        package.validate()
        sealed_history = ReviewPhaseHistory(
            tuple(ReviewPhase)[: tuple(ReviewPhase).index(ReviewPhase.BLIND_PACKAGE_SEALED) + 1]
        ).validate()
        return cls(
            protocol_sha256=QualityProtocol.default().fingerprint(),
            generation_manifest_sha256=package.generation_manifest_sha256,
            side_assignment_commitment_sha256=package.side_assignment_commitment_sha256,
            blind_package_sha256=package.fingerprint(),
            blind_package_sealed_phase_sha256=sealed_history.fingerprint(),
        ).validate()

    def validate(self) -> Self:
        for name in (
            "protocol_sha256",
            "generation_manifest_sha256",
            "side_assignment_commitment_sha256",
            "blind_package_sha256",
            "blind_package_sealed_phase_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.protocol_sha256 != QualityProtocol.default().fingerprint():
            raise ValueError("blind package seal receipt protocol is not frozen")
        expected_phase_sha256 = ReviewPhaseHistory(
            tuple(ReviewPhase)[: tuple(ReviewPhase).index(ReviewPhase.BLIND_PACKAGE_SEALED) + 1]
        ).fingerprint()
        if self.blind_package_sealed_phase_sha256 != expected_phase_sha256:
            raise ValueError("blind package seal receipt does not bind the pre-ballot phase")
        return self


@dataclass(frozen=True, slots=True)
class ScoredReviewPair(FrozenRecord):
    """One answer-key-revealed blind pair bound to its scene, category, and methods."""

    pair_id: str
    scene_id: str
    cluster_id: str
    category: QualityCategory
    seed: int
    comparison: ReviewComparison
    left_method: Method
    right_method: Method
    left_presentation: ReviewPresentationReceipt
    right_presentation: ReviewPresentationReceipt
    ballot: Ballot

    @property
    def canonical_key(self) -> tuple[str, int, str]:
        return (self.scene_id, self.seed, self.comparison.value)

    def validate(self) -> Self:
        for name in ("pair_id", "scene_id", "cluster_id"):
            _nonempty(getattr(self, name), name)
        if not isinstance(self.category, QualityCategory):
            raise ValueError("scored pair category is not frozen")
        if type(self.seed) is not int or self.seed not in MATCHED_SEEDS:
            raise ValueError("scored pair seed must be frozen")
        if not isinstance(self.comparison, ReviewComparison):
            raise ValueError("scored pair comparison is malformed")
        if not isinstance(self.left_method, Method) or not isinstance(self.right_method, Method):
            raise ValueError("scored pair methods must be frozen")
        expected_methods = (
            {Method.DUET_CORE, Method.GATED_CORE}
            if self.comparison is ReviewComparison.PRIMARY
            else {Method.FULL_HISTORY, Method.RECENT_ANCHOR}
        )
        if {self.left_method, self.right_method} != expected_methods:
            raise ValueError("scored pair methods do not match its frozen comparison")
        for side, method in (
            (self.left_presentation, self.left_method),
            (self.right_presentation, self.right_method),
        ):
            if not isinstance(side, ReviewPresentationReceipt):
                raise ValueError("scored pair requires two presentation receipts")
            side.validate()
            if side.cell_key != GenerationCellKey(self.scene_id, self.seed, method):
                raise ValueError("scored pair side does not match its answer-key cell")
        if not isinstance(self.ballot, Ballot):
            raise ValueError("scored pair requires a complete ballot")
        self.ballot.validate()
        if self.ballot.pair_id != self.pair_id:
            raise ValueError("scored pair and ballot IDs do not match")
        return self


def scored_review_manifest_sha256(pairs: Sequence[ScoredReviewPair]) -> str:
    """Hash answer-key-revealed scored pairs in their canonical order."""
    values = tuple(pairs)
    for pair in values:
        pair.validate()
    return canonical_sha256(tuple(pair.to_dict() for pair in values))


def _answer_key_entries(pairs: Sequence[ScoredReviewPair]) -> tuple[dict[str, object], ...]:
    values = tuple(pairs)
    for pair in values:
        pair.validate()
    return tuple(
        {
            "pair_id": pair.pair_id,
            "left": pair.left_presentation.to_dict(),
            "right": pair.right_presentation.to_dict(),
        }
        for pair in values
    )


def answer_key_sha256(pairs: Sequence[ScoredReviewPair]) -> str:
    """Hash the exact revealed pair-side-to-presentation answer mapping."""
    return canonical_sha256(
        {"format": "duet-x-answer-key-v1", "entries": _answer_key_entries(pairs)}
    )


def side_assignment_commitment_sha256(secret: str, generation_manifest_sha256: str) -> str:
    """Commit the hidden assignment preimage to the protocol and generation seal."""
    _nonempty(secret, "side_assignment_secret")
    _sha256(generation_manifest_sha256, "generation_manifest_sha256")
    return canonical_sha256(
        {
            "format": "duet-x-side-assignment-commitment-v1",
            "protocol_sha256": QualityProtocol.default().fingerprint(),
            "generation_manifest_sha256": generation_manifest_sha256,
            "secret": secret,
        }
    )


def side_assignment_methods(
    secret: str,
    generation_manifest_sha256: str,
    scene_id: str,
    seed: int,
    comparison: ReviewComparison,
) -> tuple[Method, Method]:
    """Recompute one pair's left/right method order from the revealed preimage."""
    _nonempty(secret, "side_assignment_secret")
    _sha256(generation_manifest_sha256, "generation_manifest_sha256")
    _nonempty(scene_id, "scene_id")
    if type(seed) is not int or seed not in MATCHED_SEEDS:
        raise ValueError("side assignment seed must be frozen")
    if not isinstance(comparison, ReviewComparison):
        raise ValueError("side assignment comparison is malformed")
    methods = (
        (Method.DUET_CORE, Method.GATED_CORE)
        if comparison is ReviewComparison.PRIMARY
        else (Method.FULL_HISTORY, Method.RECENT_ANCHOR)
    )
    digest = canonical_sha256(
        {
            "format": "duet-x-pair-side-assignment-v1",
            "generation_manifest_sha256": generation_manifest_sha256,
            "protocol_sha256": QualityProtocol.default().fingerprint(),
            "secret": secret,
            "scene_id": scene_id,
            "seed": seed,
            "comparison": comparison.value,
        }
    )
    return methods if int(digest, 16) % 2 == 0 else (methods[1], methods[0])


@dataclass(frozen=True, slots=True)
class ClusterAggregate(FrozenRecord):
    """Task-5-produced averages for one independent counterfactual cluster."""

    cluster_id: str
    category: QualityCategory
    csr_full: float
    csr_recent: float
    csr_gated: float
    csr_duet: float
    primary_pairwise_duet: float
    calibration_pairwise_full: float
    prompt_adherence_gated: float
    prompt_adherence_duet: float
    duet_defects: tuple[Defect, ...]

    def validate(self) -> Self:
        _nonempty(self.cluster_id, "cluster_id")
        if not isinstance(self.category, QualityCategory):
            raise ValueError("cluster aggregate category is not frozen")
        for name in (
            "csr_full",
            "csr_recent",
            "csr_gated",
            "csr_duet",
            "primary_pairwise_duet",
            "calibration_pairwise_full",
            "prompt_adherence_gated",
            "prompt_adherence_duet",
        ):
            _finite_unit(getattr(self, name), name)
        _require_tuple(self.duet_defects, "duet_defects")
        if any(not isinstance(defect, Defect) for defect in self.duet_defects):
            raise ValueError("cluster defects must use the frozen taxonomy")
        if len(self.duet_defects) != len(set(self.duet_defects)):
            raise ValueError("cluster defects must be unique")
        return self


@dataclass(frozen=True, slots=True)
class ReviewEvidenceBinding(FrozenRecord):
    """Sealed review provenance required before decision aggregates can be consumed."""

    phase_history: ReviewPhaseHistory
    scored_pairs: tuple[ScoredReviewPair, ...]
    scored_pairs_sha256: str
    answer_key_sha256: str
    side_assignment_secret: str
    side_assignment_commitment_sha256: str
    generation_manifest_sha256: str
    blind_package: BlindReviewPackage

    def validate(self) -> Self:
        if not isinstance(self.phase_history, ReviewPhaseHistory):
            raise ValueError("review evidence requires an authenticated phase history")
        self.phase_history.validate()
        phases = tuple(ReviewPhase)
        if phases.index(self.phase_history.current) < phases.index(ReviewPhase.ANSWER_KEY_REVEALED):
            raise ValueError("verdict evidence requires answer-key reveal")
        _require_tuple(self.scored_pairs, "scored_pairs")
        if len(self.scored_pairs) != REVIEW_PAIR_COUNT:
            raise ValueError("review evidence requires exactly 216 complete scored pairs")
        for pair in self.scored_pairs:
            if not isinstance(pair, ScoredReviewPair):
                raise ValueError("review evidence contains a malformed scored pair")
            pair.validate()
        pair_ids = tuple(pair.pair_id for pair in self.scored_pairs)
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("review evidence contains a duplicate pair key")
        keys = tuple(pair.canonical_key for pair in self.scored_pairs)
        if len(set(keys)) != len(keys):
            raise ValueError("review evidence contains a duplicate scene/seed/comparison key")
        if keys != tuple(sorted(keys)):
            raise ValueError("review scored pairs must be in canonical key order")
        rater_ids = {pair.ballot.rater_id for pair in self.scored_pairs}
        if len(rater_ids) != COMPLETE_RATER_COUNT:
            raise ValueError("review evidence requires one complete authorized rater")
        scene_ids = {pair.scene_id for pair in self.scored_pairs}
        cluster_ids = {pair.cluster_id for pair in self.scored_pairs}
        if (
            len(scene_ids) != CONFIRMATORY_SCENE_COUNT
            or len(cluster_ids) != CONFIRMATORY_CLUSTER_COUNT
        ):
            raise ValueError("review evidence requires 36 scenes and 18 clusters")
        for cluster_id in cluster_ids:
            cluster_pairs = tuple(
                pair for pair in self.scored_pairs if pair.cluster_id == cluster_id
            )
            cluster_scenes = {pair.scene_id for pair in cluster_pairs}
            cluster_categories = {pair.category for pair in cluster_pairs}
            if len(cluster_scenes) != 2 or len(cluster_categories) != 1:
                raise ValueError("each review cluster must contain two scenes in one category")
        category_by_cluster = {
            cluster_id: next(
                pair.category for pair in self.scored_pairs if pair.cluster_id == cluster_id
            )
            for cluster_id in cluster_ids
        }
        if any(
            tuple(category_by_cluster.values()).count(category) != CATEGORY_CLUSTER_COUNT
            for category in QualityCategory
        ):
            raise ValueError("review evidence requires six clusters per category")
        expected_keys = {
            (scene_id, seed, comparison.value)
            for scene_id in scene_ids
            for seed in MATCHED_SEEDS
            for comparison in ReviewComparison
        }
        if set(keys) != expected_keys:
            raise ValueError("review evidence does not contain the complete scored-pair grid")
        for name in (
            "scored_pairs_sha256",
            "answer_key_sha256",
            "side_assignment_commitment_sha256",
            "generation_manifest_sha256",
        ):
            _sha256(getattr(self, name), name)
        _nonempty(self.side_assignment_secret, "side_assignment_secret")
        if self.scored_pairs_sha256 != scored_review_manifest_sha256(self.scored_pairs):
            raise ValueError("scored review manifest hash does not match its canonical evidence")
        if self.answer_key_sha256 != answer_key_sha256(self.scored_pairs):
            raise ValueError("answer-key hash does not match the revealed side mapping")
        if self.side_assignment_commitment_sha256 != side_assignment_commitment_sha256(
            self.side_assignment_secret,
            self.generation_manifest_sha256,
        ):
            raise ValueError("revealed assignment secret does not match the sealed commitment")
        if not isinstance(self.blind_package, BlindReviewPackage):
            raise ValueError("review evidence requires its pre-ballot blind package")
        self.blind_package.validate()
        if (
            self.blind_package.generation_manifest_sha256 != self.generation_manifest_sha256
            or self.blind_package.side_assignment_commitment_sha256
            != self.side_assignment_commitment_sha256
        ):
            raise ValueError(
                "blind package does not match the sealed generation and assignment context"
            )
        expected_sides = tuple(
            sorted(
                (
                    side
                    for pair in self.scored_pairs
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
        if self.blind_package.sides != expected_sides:
            raise ValueError("answer key presentations do not match the sealed blind package")
        for pair in self.scored_pairs:
            expected = side_assignment_methods(
                self.side_assignment_secret,
                self.generation_manifest_sha256,
                pair.scene_id,
                pair.seed,
                pair.comparison,
            )
            if (pair.left_method, pair.right_method) != expected:
                raise ValueError("answer-key side mapping does not match the revealed assignment")
        return self


@dataclass(frozen=True, slots=True)
class VerdictEvidence(FrozenRecord):
    """Complete provenance plus raw cluster aggregates consumed by the verdict boundary."""

    generation: CommittedGenerationMatrix
    review: ReviewEvidenceBinding
    bootstrap: BootstrapSettings
    blind_package_seal: BlindPackageSealReceipt

    def validate(self) -> Self:
        if not isinstance(self.generation, CommittedGenerationMatrix):
            raise ValueError("verdict evidence requires a committed generation matrix")
        self.generation.validate()
        if not isinstance(self.review, ReviewEvidenceBinding):
            raise ValueError("verdict evidence requires sealed review provenance")
        self.review.validate()
        if self.review.generation_manifest_sha256 != self.generation.first_finalization_sha256:
            raise ValueError("review package is not bound to the committed generation manifest")
        if not isinstance(self.blind_package_seal, BlindPackageSealReceipt):
            raise ValueError("verdict evidence requires a pre-ballot blind package seal receipt")
        self.blind_package_seal.validate()
        if (
            self.blind_package_seal.generation_manifest_sha256
            != self.generation.first_finalization_sha256
            or self.blind_package_seal.side_assignment_commitment_sha256
            != self.review.side_assignment_commitment_sha256
            or self.blind_package_seal.blind_package_sha256
            != self.review.blind_package.fingerprint()
        ):
            raise ValueError("blind package seal receipt does not match revealed review evidence")
        cells_by_key = {cell.key: cell for cell in self.generation.cells}
        profiles: set[tuple[str, int, int, int, int, str, str]] = set()
        for pair in self.review.scored_pairs:
            for presentation in (pair.left_presentation, pair.right_presentation):
                cell = cells_by_key.get(presentation.cell_key)
                if cell is None or presentation.source_media_sha256 != cell.media_sha256:
                    raise ValueError("review presentation is not bound to its committed media")
                profiles.add(
                    (
                        presentation.codec,
                        presentation.width,
                        presentation.height,
                        presentation.frames,
                        presentation.fps,
                        presentation.audio_policy,
                        presentation.metadata_policy,
                    )
                )
        if len(profiles) != 1:
            raise ValueError("review presentations do not share one normalized profile")
        if not isinstance(self.bootstrap, BootstrapSettings):
            raise ValueError("verdict evidence requires frozen bootstrap settings")
        self.bootstrap.validate()
        derive_cluster_aggregates(self.review, self.generation)
        return self

    def validate_against_external_anchor(self, preserved_anchor_sha256: str) -> Self:
        """Require the authority-retained pre-ballot seal anchor at the decision boundary."""
        self.validate()
        _sha256(preserved_anchor_sha256, "preserved_blind_package_seal_anchor_sha256")
        if preserved_anchor_sha256 != self.blind_package_seal.fingerprint():
            raise ValueError(
                "external blind package seal anchor does not match the candidate receipt"
            )
        return self


def _method_boolean(pair: ScoredReviewPair, method: Method, field: str) -> float:
    if method is pair.left_method:
        value = getattr(pair.ballot, f"left_{field}")
    elif method is pair.right_method:
        value = getattr(pair.ballot, f"right_{field}")
    else:
        raise ValueError("method is absent from scored pair")
    if type(value) is not bool:
        raise ValueError("scored ballot indicator must be boolean")
    return float(value)


def _method_preference(pair: ScoredReviewPair, method: Method) -> float:
    preference = pair.ballot.fact_preference
    if preference is SidePreference.TIE:
        return 0.5
    selected = pair.left_method if preference is SidePreference.LEFT else pair.right_method
    return float(selected is method)


def _method_defects(pair: ScoredReviewPair, method: Method) -> tuple[Defect, ...]:
    if method is pair.left_method:
        return pair.ballot.left_major_defects
    if method is pair.right_method:
        return pair.ballot.right_major_defects
    raise ValueError("method is absent from scored pair")


def derive_cluster_aggregates(
    review: ReviewEvidenceBinding,
    generation: CommittedGenerationMatrix,
) -> tuple[ClusterAggregate, ...]:
    """Derive the sole 18 cluster-level decision rows from complete scored ballots."""
    review.validate()
    if not isinstance(generation, CommittedGenerationMatrix):
        raise ValueError("aggregate derivation requires the committed generation matrix")
    generation.validate()
    scene_ids = generation.scene_ids
    scene_by_id = {
        variant.scene_id: variant for cluster in generation.clusters for variant in cluster.variants
    }
    for pair in review.scored_pairs:
        scene = scene_by_id.get(pair.scene_id)
        if scene is None:
            raise ValueError("review pair scene is absent from the generation manifest")
        if pair.cluster_id != scene.cluster_id or pair.category is not scene.category:
            raise ValueError("review pair cluster/category does not match its generation scene")
    if {pair.scene_id for pair in review.scored_pairs} != set(scene_ids):
        raise ValueError("review scenes do not match the committed generation matrix")
    cluster_ids = tuple(sorted({pair.cluster_id for pair in review.scored_pairs}))
    aggregates: list[ClusterAggregate] = []
    for cluster_id in cluster_ids:
        cluster_pairs = tuple(pair for pair in review.scored_pairs if pair.cluster_id == cluster_id)
        primary = tuple(
            pair for pair in cluster_pairs if pair.comparison is ReviewComparison.PRIMARY
        )
        calibration = tuple(
            pair for pair in cluster_pairs if pair.comparison is ReviewComparison.CALIBRATION
        )
        categories = {pair.category for pair in cluster_pairs}
        if len(primary) != 6 or len(calibration) != 6 or len(categories) != 1:
            raise ValueError("cluster aggregation requires six primary and six calibration pairs")
        duet_defects = tuple(
            defect
            for defect in Defect
            if any(defect in _method_defects(pair, Method.DUET_CORE) for pair in primary)
        )
        aggregates.append(
            ClusterAggregate(
                cluster_id=cluster_id,
                category=next(iter(categories)),
                csr_full=_mean(
                    tuple(
                        _method_boolean(pair, Method.FULL_HISTORY, "fact_present")
                        for pair in calibration
                    )
                ),
                csr_recent=_mean(
                    tuple(
                        _method_boolean(pair, Method.RECENT_ANCHOR, "fact_present")
                        for pair in calibration
                    )
                ),
                csr_gated=_mean(
                    tuple(
                        _method_boolean(pair, Method.GATED_CORE, "fact_present") for pair in primary
                    )
                ),
                csr_duet=_mean(
                    tuple(
                        _method_boolean(pair, Method.DUET_CORE, "fact_present") for pair in primary
                    )
                ),
                primary_pairwise_duet=_mean(
                    tuple(_method_preference(pair, Method.DUET_CORE) for pair in primary)
                ),
                calibration_pairwise_full=_mean(
                    tuple(_method_preference(pair, Method.FULL_HISTORY) for pair in calibration)
                ),
                prompt_adherence_gated=_mean(
                    tuple(
                        _method_boolean(pair, Method.GATED_CORE, "prompt_adherent")
                        for pair in primary
                    )
                ),
                prompt_adherence_duet=_mean(
                    tuple(
                        _method_boolean(pair, Method.DUET_CORE, "prompt_adherent")
                        for pair in primary
                    )
                ),
                duet_defects=duet_defects,
            ).validate()
        )
    return tuple(aggregates)


@dataclass(frozen=True, slots=True)
class _DerivedVerdictMetrics:
    csr_duet: float
    csr_gated: float
    pairwise_duet: float
    pairwise_lower_95: float
    pairwise_upper_95: float
    prompt_adherence_duet: float
    prompt_adherence_gated: float
    systematic_defects: tuple[Defect, ...]
    category_superiority: tuple[bool, bool, bool]
    calibration_full_pairwise: float
    csr_full: float
    csr_recent: float


def _mean(values: Sequence[float]) -> float:
    if not values or any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError("decision mean requires a nonempty finite float sequence")
    return math.fsum(values) / len(values)


def _bootstrap_pairwise_interval(
    values: tuple[float, ...], settings: BootstrapSettings
) -> tuple[float, float]:
    settings.validate()
    array: npt.NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(settings.seed))
    indices: npt.NDArray[np.int64] = generator.integers(
        0,
        CONFIRMATORY_CLUSTER_COUNT,
        size=(settings.samples, CONFIRMATORY_CLUSTER_COUNT),
        dtype=np.int64,
    )
    bootstrap_means = np.mean(array[indices], axis=1, dtype=np.float64)
    quantiles = np.quantile(
        bootstrap_means,
        (0.025, 0.975),
        method="linear",
    )
    return float(quantiles[0]), float(quantiles[1])


def _derive_verdict_metrics(
    evidence: VerdictEvidence, preserved_blind_package_seal_anchor_sha256: str
) -> _DerivedVerdictMetrics:
    evidence.validate_against_external_anchor(preserved_blind_package_seal_anchor_sha256)
    aggregates = derive_cluster_aggregates(evidence.review, evidence.generation)
    pairwise_values = tuple(item.primary_pairwise_duet for item in aggregates)
    lower, upper = _bootstrap_pairwise_interval(pairwise_values, evidence.bootstrap)
    defect_counts = {
        defect: sum(defect in aggregate.duet_defects for aggregate in aggregates)
        for defect in Defect
    }
    systematic = tuple(
        defect for defect in Defect if defect_counts[defect] >= SYSTEMATIC_DEFECT_CLUSTER_COUNT
    )
    category_superiority = tuple(
        _mean(tuple(item.csr_duet for item in aggregates if item.category is category))
        > _mean(tuple(item.csr_gated for item in aggregates if item.category is category))
        or _mean(
            tuple(item.primary_pairwise_duet for item in aggregates if item.category is category)
        )
        > PAIRWISE_CI_LOWER_THRESHOLD
        for category in QualityCategory
    )
    return _DerivedVerdictMetrics(
        csr_duet=_mean(tuple(item.csr_duet for item in aggregates)),
        csr_gated=_mean(tuple(item.csr_gated for item in aggregates)),
        pairwise_duet=_mean(pairwise_values),
        pairwise_lower_95=lower,
        pairwise_upper_95=upper,
        prompt_adherence_duet=_mean(tuple(item.prompt_adherence_duet for item in aggregates)),
        prompt_adherence_gated=_mean(tuple(item.prompt_adherence_gated for item in aggregates)),
        systematic_defects=systematic,
        category_superiority=cast(tuple[bool, bool, bool], category_superiority),
        calibration_full_pairwise=_mean(
            tuple(item.calibration_pairwise_full for item in aggregates)
        ),
        csr_full=_mean(tuple(item.csr_full for item in aggregates)),
        csr_recent=_mean(tuple(item.csr_recent for item in aggregates)),
    )


@dataclass(frozen=True, slots=True)
class VerdictResult(FrozenRecord):
    """The verdict plus its decision-bearing derived quantities."""

    verdict: Verdict
    csr_delta: float
    prompt_adherence_delta: float
    pairwise_duet: float
    pairwise_lower_95: float
    pairwise_upper_95: float
    quality_superiority: bool
    category_guardrail_pass: bool
    systematic_defects: tuple[Defect, ...]
    category_superiority: tuple[bool, bool, bool]

    def validate(self) -> Self:
        if not isinstance(self.verdict, Verdict):
            raise ValueError("verdict must use the frozen classification")
        for name in (
            "csr_delta",
            "prompt_adherence_delta",
            "pairwise_duet",
            "pairwise_lower_95",
            "pairwise_upper_95",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (
            type(self.quality_superiority) is not bool
            or type(self.category_guardrail_pass) is not bool
        ):
            raise ValueError("verdict guardrail fields must be booleans")
        _require_tuple(self.systematic_defects, "systematic_defects")
        if any(not isinstance(defect, Defect) for defect in self.systematic_defects):
            raise ValueError("systematic defects must use the frozen taxonomy")
        _require_tuple(self.category_superiority, "category_superiority")
        if len(self.category_superiority) != 3 or any(
            type(value) is not bool for value in self.category_superiority
        ):
            raise ValueError("category superiority must contain three derived booleans")
        return self


def _invalid_verdict_result() -> VerdictResult:
    return VerdictResult(
        verdict=Verdict.INVALID_EVIDENCE,
        csr_delta=0.0,
        prompt_adherence_delta=0.0,
        pairwise_duet=0.0,
        pairwise_lower_95=0.0,
        pairwise_upper_95=0.0,
        quality_superiority=False,
        category_guardrail_pass=False,
        systematic_defects=(),
        category_superiority=(False, False, False),
    ).validate()


def csr_delta_meets_threshold(csr_duet: float, csr_gated: float) -> bool:
    """Compare the registered decimal CSR difference without binary-addition boundary error."""
    _finite_unit(csr_duet, "csr_duet")
    _finite_unit(csr_gated, "csr_gated")
    return Decimal(str(csr_duet)) - Decimal(str(csr_gated)) >= Decimal(str(CSR_DELTA_THRESHOLD))


def classify_verdict(
    evidence: object, preserved_blind_package_seal_anchor_sha256: str
) -> VerdictResult:
    """Derive all decision facts from bound evidence, then apply frozen verdict precedence."""
    if not isinstance(evidence, VerdictEvidence):
        return _invalid_verdict_result()
    try:
        inputs = _derive_verdict_metrics(evidence, preserved_blind_package_seal_anchor_sha256)
    except (TypeError, ValueError):
        return _invalid_verdict_result()
    csr_delta = inputs.csr_duet - inputs.csr_gated
    prompt_delta = inputs.prompt_adherence_duet - inputs.prompt_adherence_gated
    csr_superiority = csr_delta_meets_threshold(inputs.csr_duet, inputs.csr_gated)
    pairwise_superiority = (
        inputs.pairwise_duet >= PAIRWISE_THRESHOLD
        and inputs.pairwise_lower_95 > PAIRWISE_CI_LOWER_THRESHOLD
    )
    quality_superiority = csr_superiority or pairwise_superiority
    category_guardrail = sum(inputs.category_superiority) >= CATEGORY_DIRECTION_MINIMUM
    calibration_informative = (
        inputs.calibration_full_pairwise > PAIRWISE_CI_LOWER_THRESHOLD
        and inputs.csr_full > inputs.csr_recent
    )
    prompt_guardrail = (
        inputs.prompt_adherence_duet >= inputs.prompt_adherence_gated + PROMPT_ADHERENCE_DELTA_FLOOR
    )

    if not calibration_informative:
        verdict = Verdict.HUMAN_EVALUATION_UNINFORMATIVE
    elif inputs.systematic_defects:
        verdict = Verdict.SYSTEMATIC_DEFECT
    elif not prompt_guardrail:
        verdict = Verdict.PROMPT_ADHERENCE_FAILED
    elif quality_superiority and category_guardrail:
        verdict = Verdict.ADVANCE_QUALITY_FIRST
    elif inputs.csr_duet < inputs.csr_gated or inputs.pairwise_duet < 0.50:
        verdict = Verdict.CLOSE_LTX_NEGATIVE
    else:
        verdict = Verdict.CLOSE_LTX_TIE
    return VerdictResult(
        verdict=verdict,
        csr_delta=csr_delta,
        prompt_adherence_delta=prompt_delta,
        pairwise_duet=inputs.pairwise_duet,
        pairwise_lower_95=inputs.pairwise_lower_95,
        pairwise_upper_95=inputs.pairwise_upper_95,
        quality_superiority=quality_superiority,
        category_guardrail_pass=category_guardrail,
        systematic_defects=inputs.systematic_defects,
        category_superiority=inputs.category_superiority,
    ).validate()


@dataclass(frozen=True, slots=True)
class QualityProtocol(FrozenRecord):
    """The immutable confirmatory configuration consumed by all later gate stages."""

    format: str
    trainable_checkpoint_sha256: str
    categories: tuple[QualityCategory, ...]
    methods: tuple[Method, ...]
    cluster_count: int
    scene_count: int
    variants_per_cluster: int
    category_cluster_count: int
    seeds: tuple[int, ...]
    protected_slots: tuple[int, ...]
    fact_eligible_slots: tuple[int, ...]
    fact_absent_slots: tuple[int, ...]
    full_guide_count: int
    bounded_guide_count: int
    generation_cell_count: int
    review_pair_count: int
    complete_rater_count: int
    runtime: RuntimeCoordinate
    bootstrap: BootstrapSettings
    csr_delta_threshold: float
    pairwise_threshold: float
    pairwise_ci_lower_threshold: float
    prompt_adherence_delta_floor: float
    systematic_defect_cluster_count: int
    category_direction_minimum: int
    calibration_pairwise_threshold: float
    cell_state_chain: tuple[CellState, ...]
    review_phase_chain: tuple[ReviewPhase, ...]

    @classmethod
    def default(cls) -> Self:
        return cls(
            format=QUALITY_PROTOCOL_FORMAT,
            trainable_checkpoint_sha256=FROZEN_TRAINABLE_CHECKPOINT_SHA256,
            categories=tuple(QualityCategory),
            methods=tuple(Method),
            cluster_count=CONFIRMATORY_CLUSTER_COUNT,
            scene_count=CONFIRMATORY_SCENE_COUNT,
            variants_per_cluster=2,
            category_cluster_count=CATEGORY_CLUSTER_COUNT,
            seeds=MATCHED_SEEDS,
            protected_slots=PROTECTED_SLOTS,
            fact_eligible_slots=FACT_ELIGIBLE_SLOTS,
            fact_absent_slots=FACT_ABSENT_SLOTS,
            full_guide_count=9,
            bounded_guide_count=4,
            generation_cell_count=GENERATION_CELL_COUNT,
            review_pair_count=REVIEW_PAIR_COUNT,
            complete_rater_count=COMPLETE_RATER_COUNT,
            runtime=RuntimeCoordinate.default(),
            bootstrap=BootstrapSettings.default(),
            csr_delta_threshold=CSR_DELTA_THRESHOLD,
            pairwise_threshold=PAIRWISE_THRESHOLD,
            pairwise_ci_lower_threshold=PAIRWISE_CI_LOWER_THRESHOLD,
            prompt_adherence_delta_floor=PROMPT_ADHERENCE_DELTA_FLOOR,
            systematic_defect_cluster_count=SYSTEMATIC_DEFECT_CLUSTER_COUNT,
            category_direction_minimum=CATEGORY_DIRECTION_MINIMUM,
            calibration_pairwise_threshold=PAIRWISE_CI_LOWER_THRESHOLD,
            cell_state_chain=tuple(CellState),
            review_phase_chain=tuple(ReviewPhase),
        )

    def validate(self) -> Self:
        if self != type(self).default():
            raise ValueError("decoded-quality confirmatory configuration is frozen")
        self.runtime.validate()
        self.bootstrap.validate()
        return self


Category = QualityCategory

__all__ = (
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "CATEGORY_CLUSTER_COUNT",
    "CATEGORY_DIRECTION_MINIMUM",
    "COMPLETE_RATER_COUNT",
    "CONFIRMATORY_CLUSTER_COUNT",
    "CONFIRMATORY_SCENE_COUNT",
    "CSR_DELTA_THRESHOLD",
    "FACT_ABSENT_SLOTS",
    "FACT_ELIGIBLE_SLOTS",
    "FROZEN_TRAINABLE_CHECKPOINT_SHA256",
    "GENERATION_CELL_COUNT",
    "LTX_DEVELOPMENT_CHECKPOINT_SHA256",
    "LTX_SOURCE_COMMIT",
    "MATCHED_SEEDS",
    "METHOD_IDS",
    "PAIRWISE_CI_LOWER_THRESHOLD",
    "PAIRWISE_THRESHOLD",
    "PROMPT_ADHERENCE_DELTA_FLOOR",
    "PROTECTED_SLOTS",
    "QUALITY_PROTOCOL_FORMAT",
    "REVIEW_PAIR_COUNT",
    "SYSTEMATIC_DEFECT_CLUSTER_COUNT",
    "Ballot",
    "BlindPackageSealReceipt",
    "BlindPresentationSide",
    "BlindReviewPackage",
    "BootstrapSettings",
    "Category",
    "CellState",
    "ClusterAggregate",
    "CommittedGenerationCell",
    "CommittedGenerationMatrix",
    "CounterfactualCluster",
    "Defect",
    "GenerationApparatusFailure",
    "GenerationCellKey",
    "GenerationCellLifecycle",
    "GuideSource",
    "GuideSourceKind",
    "Method",
    "MethodGuideReceipt",
    "QualityCategory",
    "QualityMethod",
    "QualityProtocol",
    "ReviewComparison",
    "ReviewEvidenceBinding",
    "ReviewPhase",
    "ReviewPhaseHistory",
    "ReviewPresentationReceipt",
    "RuntimeCoordinate",
    "SceneVariant",
    "ScoredReviewPair",
    "SidePreference",
    "Verdict",
    "VerdictEvidence",
    "VerdictResult",
    "answer_key_sha256",
    "canonical_json",
    "canonical_sha256",
    "classify_verdict",
    "committed_generation_manifest_sha256",
    "csr_delta_meets_threshold",
    "derive_cluster_aggregates",
    "from_canonical_json",
    "scene_manifest_sha256",
    "scored_review_manifest_sha256",
    "side_assignment_commitment_sha256",
    "side_assignment_methods",
    "validate_confirmatory_clusters",
    "validate_generation_cell_keys",
)
