"""Strict, versioned configuration for the Duet-X Decision 1 canary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf


class Method(StrEnum):
    FULL_HISTORY_TEACHER = "full_history_teacher"
    TOPK_ONLY = "topk_only"
    RECENT_ANCHOR_TOPK = "recent_anchor_topk"
    MEAN_CORE_TOPK = "mean_core_topk"
    GATED_CORE_TOPK = "gated_core_topk"
    CENTRALIZED_RESAMPLER_TOPK = "centralized_resampler_topk"
    DUET_CORE_TOPK = "duet_core_topk"


class Selector(StrEnum):
    RANDOM = "random"
    RECENT = "recent"
    UNIFORM = "uniform"
    ENGINEERED = "engineered"
    LEARNED = "learned"
    ORACLE = "oracle"


@dataclass(frozen=True)
class ExperimentConfig:
    history_items: int
    protected_exceptions: int
    mandatory_guides: int
    methods: tuple[Method, ...]
    selectors: tuple[Selector, ...]


@dataclass(frozen=True)
class ModelConfig:
    operator_size: int
    salience_scale: int
    teacher_timesteps: int


@dataclass(frozen=True)
class TrainingConfig:
    optimizer_steps: int
    seeds: tuple[int, ...]
    synthetic_smoke: bool


@dataclass(frozen=True)
class ScoreConfig:
    teacher_similarity: float
    identity: float
    rare_detail: float
    continuity: float
    bootstrap_samples: int


@dataclass(frozen=True)
class StopGatesConfig:
    require_structural_gates: bool
    require_positive_paired_ci: bool
    require_exact_exceptions: bool
    require_fixed_guide_count: bool
    require_systems_advantage: bool
    stop_on_gated_loss: bool
    stop_on_resampler_loss: bool
    stop_on_oracle_barely_improves: bool


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int
    latent_channels: int
    require_equal_latent_shapes: bool
    accumulation_dtype: str
    model_dtype: str
    freeze_ltx: bool


@dataclass(frozen=True)
class Decision1Config:
    format: str
    experiment: ExperimentConfig
    model: ModelConfig
    training: TrainingConfig
    primary_score: ScoreConfig
    stop_gates: StopGatesConfig
    runtime: RuntimeConfig

    def validate(self) -> Decision1Config:
        """Validate the immutable Decision 1 axes and return this configuration."""
        if self.format != "duet-x-ltx-decision1-v1":
            raise ValueError(f"unsupported format: {self.format!r}")
        if self.experiment.history_items != 8:
            raise ValueError("Decision 1 fixes history_items to 8")
        if self.experiment.protected_exceptions != 2:
            raise ValueError("Decision 1 fixes protected_exceptions to 2")
        if self.experiment.mandatory_guides != 1:
            raise ValueError("Decision 1 fixes mandatory_guides to 1")
        if tuple(self.experiment.methods) != tuple(Method):
            raise ValueError("methods must contain every Decision 1 method in canonical order")
        if tuple(self.experiment.selectors) != tuple(Selector):
            raise ValueError("selectors must contain every Decision 1 selector in canonical order")
        if self.model.operator_size != 16 or self.model.salience_scale != 1_000_000:
            raise ValueError("Decision 1 fixes operator_size=16 and salience_scale=1000000")
        if self.model.teacher_timesteps != 3:
            raise ValueError("Decision 1 fixes teacher_timesteps to 3")
        if self.training.seeds != (101, 202, 303):
            raise ValueError("Decision 1 fixes seeds to 101, 202, 303")
        expected_steps = 2 if self.training.synthetic_smoke else 2_000
        if self.training.optimizer_steps != expected_steps:
            raise ValueError("synthetic smoke requires 2 optimizer steps; Decision 1 requires 2000")
        weights = (
            self.primary_score.teacher_similarity,
            self.primary_score.identity,
            self.primary_score.rare_detail,
            self.primary_score.continuity,
        )
        if weights != (0.35, 0.20, 0.30, 0.15):
            raise ValueError("primary score weights are fixed to 0.35/0.20/0.30/0.15")
        if self.primary_score.bootstrap_samples != 10_000:
            raise ValueError("Decision 1 fixes bootstrap_samples to 10000")
        if self.stop_gates != StopGatesConfig(True, True, True, True, True, True, True, True):
            raise ValueError("Decision 1 stop gates are locked")
        if self.runtime != RuntimeConfig(1, 128, True, "float32", "bfloat16", True):
            raise ValueError("Decision 1 runtime/data fields are locked")
        return self

    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of canonical JSON configuration."""
        payload = json.dumps(_jsonable(asdict(self)), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _reject_unknown(data: dict[str, Any], cls: type[Any], prefix: str = "") -> None:
    allowed = {field.name for field in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        name = sorted(unknown)[0]
        raise ValueError(f"unknown config key: {prefix}{name}")


def _tuple(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("config sequence must be a list")
    return tuple(value)


def _strict(value: Any, expected: type[Any], name: str) -> Any:
    if type(value) is not expected:
        raise ValueError(f"{name} must be a {expected.__name__}")
    return value


def load_config(path: Path, *, expected_fingerprint: str | None = None) -> Decision1Config:
    """Load and strictly validate a Decision 1 YAML configuration."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    root = cast(dict[str, Any], raw)
    _reject_unknown(root, Decision1Config)
    for key, cls in (
        ("experiment", ExperimentConfig),
        ("model", ModelConfig),
        ("training", TrainingConfig),
        ("primary_score", ScoreConfig),
        ("stop_gates", StopGatesConfig),
        ("runtime", RuntimeConfig),
    ):
        section = root.get(key)
        if not isinstance(section, dict):
            raise ValueError(f"config section {key!r} must be a mapping")
        _reject_unknown(section, cls, f"{key}.")
    experiment = root["experiment"]
    model = root["model"]
    training = root["training"]
    score = root["primary_score"]
    gates = root["stop_gates"]
    config = Decision1Config(
        format=_strict(root["format"], str, "format"),
        experiment=ExperimentConfig(
            history_items=_strict(experiment["history_items"], int, "history_items"),
            protected_exceptions=_strict(
                experiment["protected_exceptions"], int, "protected_exceptions"
            ),
            mandatory_guides=_strict(experiment["mandatory_guides"], int, "mandatory_guides"),
            methods=tuple(
                Method(_strict(item, str, "methods item")) for item in _tuple(experiment["methods"])
            ),
            selectors=tuple(
                Selector(_strict(item, str, "selectors item"))
                for item in _tuple(experiment["selectors"])
            ),
        ),
        model=ModelConfig(
            _strict(model["operator_size"], int, "operator_size"),
            _strict(model["salience_scale"], int, "salience_scale"),
            _strict(model["teacher_timesteps"], int, "teacher_timesteps"),
        ),
        training=TrainingConfig(
            _strict(training["optimizer_steps"], int, "optimizer_steps"),
            tuple(_strict(item, int, "seed") for item in _tuple(training["seeds"])),
            _strict(training["synthetic_smoke"], bool, "synthetic_smoke"),
        ),
        primary_score=ScoreConfig(
            _strict(score["teacher_similarity"], float, "teacher_similarity"),
            _strict(score["identity"], float, "identity"),
            _strict(score["rare_detail"], float, "rare_detail"),
            _strict(score["continuity"], float, "continuity"),
            _strict(score["bootstrap_samples"], int, "bootstrap_samples"),
        ),
        stop_gates=StopGatesConfig(
            **{key: _strict(value, bool, key) for key, value in gates.items()}
        ),
        runtime=RuntimeConfig(
            _strict(root["runtime"]["batch_size"], int, "batch_size"),
            _strict(root["runtime"]["latent_channels"], int, "latent_channels"),
            _strict(
                root["runtime"]["require_equal_latent_shapes"], bool, "require_equal_latent_shapes"
            ),
            _strict(root["runtime"]["accumulation_dtype"], str, "accumulation_dtype"),
            _strict(root["runtime"]["model_dtype"], str, "model_dtype"),
            _strict(root["runtime"]["freeze_ltx"], bool, "freeze_ltx"),
        ),
    )
    config.validate()
    digest = config.fingerprint()
    if expected_fingerprint is not None:
        if len(expected_fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in expected_fingerprint
        ):
            raise ValueError("expected_fingerprint must be a lowercase SHA-256 hex digest")
        if digest != expected_fingerprint:
            raise ValueError(f"configuration fingerprint mismatch: {digest}")
    return config
