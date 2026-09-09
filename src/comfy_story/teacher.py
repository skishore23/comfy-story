"""Deterministic offline teacher-influence labels for Duet-X Decision 1."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast

import torch
from torch.nn import functional

from comfy_story.contracts import SALIENCE_Q_MAX, SALIENCE_Q_MIN
from comfy_story.data import HistoryBatch

_TIMESTEPS = (0.2, 0.5, 0.8)
_HISTORY_ITEMS = 8
_SHA256_HEX = frozenset("0123456789abcdef")
LabelSplit = Literal["train", "validation"]


class _DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _bounded_score(value: torch.Tensor | float, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or not torch.is_floating_point(value):
            raise ValueError(f"{name} must be a floating strided tensor or float")
        if value.ndim > 0 and value.shape[0] != len(_TIMESTEPS):
            raise ValueError(f"{name} tensor must have a leading coordinate dimension of 3")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be finite")
        if not bool(((value >= 0) & (value <= 1)).all().item()):
            raise ValueError(f"{name} must be within [0,1]")
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise ValueError(f"{name} must be a finite scalar within [0,1]")


@dataclass(frozen=True, slots=True)
class TeacherOutput:
    """Teacher observations at the three locked diffusion coordinates."""

    velocity: torch.Tensor
    hidden: torch.Tensor
    marker_score: torch.Tensor | float
    structural_score: torch.Tensor | float

    def validate(self) -> Self:
        for name in ("velocity", "hidden"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or value.ndim < 2
                or value.shape[0] != len(_TIMESTEPS)
                or not torch.is_floating_point(value)
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(
                    f"teacher {name} must be a finite floating tensor with leading dimension 3"
                )
        _bounded_score(self.marker_score, "marker_score")
        _bounded_score(self.structural_score, "structural_score")
        return self


class TeacherBackend(Protocol):
    """Fakeable boundary around a frozen, externally supplied teacher artifact."""

    def evaluate(
        self,
        batch: HistoryBatch,
        *,
        timesteps: tuple[float, ...] = _TIMESTEPS,
        coordinate_seed: int,
        omitted_index: int | None,
    ) -> TeacherOutput: ...


def quantize_half_up(value: int | float | Decimal, scale: int) -> int:
    """Quantize with decimal half-away-from-zero ties into a signed int64."""
    if type(scale) is not int or scale <= 0:
        raise ValueError("scale must be a positive integer")
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("value must be a finite number")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("value must be a finite number") from error
    if not decimal_value.is_finite():
        raise ValueError("value must be a finite number")
    try:
        quantized = int((decimal_value * scale).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    except (InvalidOperation, OverflowError) as error:
        raise ValueError("quantized value exceeds signed int64") from error
    if not SALIENCE_Q_MIN <= quantized <= SALIENCE_Q_MAX:
        raise ValueError("quantized value exceeds signed int64")
    return quantized


def _weighted_score(velocity: float, hidden: float, marker: float, structural: float) -> Decimal:
    return (
        Decimal("0.40") * Decimal(str(velocity))
        + Decimal("0.20") * Decimal(str(hidden))
        + Decimal("0.25") * Decimal(str(marker))
        + Decimal("0.15") * Decimal(str(structural))
    )


@dataclass(frozen=True, slots=True)
class InfluenceRecord:
    """One primitive, content-addressed leave-one-out influence label."""

    sample_id: str
    split: LabelSplit
    item_index: int
    item_id: str
    item_sha256: str
    source_sha256: str
    media_sha256: str
    velocity: float
    hidden: float
    marker: float
    structural: float
    score_q: int
    teacher_artifact_sha256: str
    coordinate_seed: int
    timesteps: tuple[float, float, float]
    config_fingerprint: str
    source_fingerprint: str
    runtime_fingerprint: str
    parent_manifest_sha256: str

    def validate(self) -> Self:
        _nonempty(self.sample_id, "sample_id")
        _nonempty(self.item_id, "item_id")
        if self.split not in {"train", "validation"}:
            raise ValueError("test influence-label records cannot be created or written")
        if type(self.item_index) is not int or not 0 <= self.item_index < _HISTORY_ITEMS:
            raise ValueError("item_index must be in range [0,8)")
        for name in (
            "item_sha256",
            "source_sha256",
            "media_sha256",
            "teacher_artifact_sha256",
            "config_fingerprint",
            "source_fingerprint",
            "runtime_fingerprint",
            "parent_manifest_sha256",
        ):
            _sha256(getattr(self, name), name)
        for name in ("velocity", "hidden", "marker", "structural"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"{name} must be a finite number within [0,1]")
        expected_q = quantize_half_up(
            _weighted_score(self.velocity, self.hidden, self.marker, self.structural),
            1_000_000,
        )
        if type(self.score_q) is not int or self.score_q != expected_q:
            raise ValueError("score_q does not match the locked influence arithmetic")
        if type(self.coordinate_seed) is not int or self.coordinate_seed < 0:
            raise ValueError("coordinate_seed must be a nonnegative integer")
        if self.timesteps != _TIMESTEPS:
            raise ValueError("timesteps must be exactly (0.2,0.5,0.8)")
        return self

    def to_dict(self) -> dict[str, object]:
        """Return the strict primitive JSON schema for one label row."""
        self.validate()
        return {
            "config_fingerprint": self.config_fingerprint,
            "coordinate_seed": self.coordinate_seed,
            "hidden": float(self.hidden),
            "item_id": self.item_id,
            "item_index": self.item_index,
            "item_sha256": self.item_sha256,
            "marker": float(self.marker),
            "media_sha256": self.media_sha256,
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "runtime_fingerprint": self.runtime_fingerprint,
            "sample_id": self.sample_id,
            "score_q": self.score_q,
            "source_fingerprint": self.source_fingerprint,
            "source_sha256": self.source_sha256,
            "split": self.split,
            "structural": float(self.structural),
            "teacher_artifact_sha256": self.teacher_artifact_sha256,
            "timesteps": list(self.timesteps),
            "velocity": float(self.velocity),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Strictly parse one existing primitive shard row."""
        expected = {
            "config_fingerprint",
            "coordinate_seed",
            "hidden",
            "item_id",
            "item_index",
            "item_sha256",
            "marker",
            "media_sha256",
            "parent_manifest_sha256",
            "runtime_fingerprint",
            "sample_id",
            "score_q",
            "source_fingerprint",
            "source_sha256",
            "split",
            "structural",
            "teacher_artifact_sha256",
            "timesteps",
            "velocity",
        }
        if set(payload) != expected:
            raise ValueError("influence record has missing or unknown fields")
        timesteps = payload["timesteps"]
        if not isinstance(timesteps, list) or any(type(value) is not float for value in timesteps):
            raise ValueError("influence record timesteps must be a JSON float list")
        record = cls(
            sample_id=_strict(payload["sample_id"], str, "sample_id"),
            split=_label_split(payload["split"]),
            item_index=_strict(payload["item_index"], int, "item_index"),
            item_id=_strict(payload["item_id"], str, "item_id"),
            item_sha256=_strict(payload["item_sha256"], str, "item_sha256"),
            source_sha256=_strict(payload["source_sha256"], str, "source_sha256"),
            media_sha256=_strict(payload["media_sha256"], str, "media_sha256"),
            velocity=_number(payload["velocity"], "velocity"),
            hidden=_number(payload["hidden"], "hidden"),
            marker=_number(payload["marker"], "marker"),
            structural=_number(payload["structural"], "structural"),
            score_q=_strict(payload["score_q"], int, "score_q"),
            teacher_artifact_sha256=_strict(
                payload["teacher_artifact_sha256"], str, "teacher_artifact_sha256"
            ),
            coordinate_seed=_strict(payload["coordinate_seed"], int, "coordinate_seed"),
            timesteps=tuple(timesteps),
            config_fingerprint=_strict(payload["config_fingerprint"], str, "config_fingerprint"),
            source_fingerprint=_strict(payload["source_fingerprint"], str, "source_fingerprint"),
            runtime_fingerprint=_strict(payload["runtime_fingerprint"], str, "runtime_fingerprint"),
            parent_manifest_sha256=_strict(
                payload["parent_manifest_sha256"], str, "parent_manifest_sha256"
            ),
        )
        return record.validate()


def _strict(value: Any, expected: type[Any], name: str) -> Any:
    if type(value) is not expected:
        raise ValueError(f"{name} has the wrong primitive JSON type")
    return value


def _number(value: Any, name: str) -> float:
    if type(value) is not float:
        raise ValueError(f"{name} must be a JSON float")
    return value


def _label_split(value: Any) -> LabelSplit:
    if value not in {"train", "validation"}:
        raise ValueError("test influence-label records cannot be created or written")
    if not isinstance(value, str):
        raise ValueError("split has the wrong primitive JSON type")
    return cast(LabelSplit, value)


def _scalar(value: torch.Tensor) -> float:
    return float(value.clamp(0, 1).item())


def _components(full: TeacherOutput, leave_one_out: TeacherOutput) -> tuple[float, ...]:
    full.validate()
    leave_one_out.validate()
    if full.velocity.shape != leave_one_out.velocity.shape:
        raise ValueError("full and leave-one-out velocity shapes must match")
    if full.hidden.shape != leave_one_out.hidden.shape:
        raise ValueError("full and leave-one-out hidden shapes must match")
    if full.velocity.device != leave_one_out.velocity.device:
        raise ValueError("full and leave-one-out velocity devices must match")
    if full.hidden.device != leave_one_out.hidden.device:
        raise ValueError("full and leave-one-out hidden devices must match")

    velocity_full = full.velocity.to(torch.float64)
    velocity_loo = leave_one_out.velocity.to(torch.float64)
    velocity = _scalar(
        ((velocity_loo - velocity_full).square().mean()) / (velocity_full.square().mean() + 1e-8)
    )
    hidden_full = full.hidden.to(torch.float64).flatten(start_dim=1)
    hidden_loo = leave_one_out.hidden.to(torch.float64).flatten(start_dim=1)
    cosine = functional.cosine_similarity(hidden_loo, hidden_full, dim=1)
    hidden_distance = (1 - cosine) / 2
    both_zero = (hidden_full.square().sum(dim=1) == 0) & (hidden_loo.square().sum(dim=1) == 0)
    hidden = _scalar(hidden_distance.masked_fill(both_zero, 0).mean())

    marker_full = torch.as_tensor(full.marker_score, dtype=torch.float64)
    marker_loo = torch.as_tensor(leave_one_out.marker_score, dtype=torch.float64)
    structural_full = torch.as_tensor(full.structural_score, dtype=torch.float64)
    structural_loo = torch.as_tensor(leave_one_out.structural_score, dtype=torch.float64)
    if marker_full.device != marker_loo.device or structural_full.device != structural_loo.device:
        raise ValueError("full and leave-one-out score devices must match")
    if marker_full.shape != marker_loo.shape:
        raise ValueError("full and leave-one-out marker_score shapes must match exactly")
    if structural_full.shape != structural_loo.shape:
        raise ValueError("full and leave-one-out structural_score shapes must match exactly")
    marker = _scalar((marker_loo - marker_full).abs().mean())
    structural = _scalar((structural_loo - structural_full).abs().mean())
    return velocity, hidden, marker, structural


def label_influence(
    batch: HistoryBatch,
    backend: TeacherBackend,
    *,
    teacher_artifact_sha256: str,
    coordinate_seed: int,
    config_fingerprint: str,
    source_fingerprint: str,
    runtime_fingerprint: str,
) -> tuple[InfluenceRecord, ...]:
    """Run one full and eight leave-one-out calls at identical coordinates."""
    if not isinstance(batch, HistoryBatch):
        raise ValueError("batch must be a HistoryBatch")
    if batch.split not in {"train", "validation"}:
        raise ValueError("influence labels may be generated only for train or validation")
    batch.validate()
    if type(coordinate_seed) is not int or coordinate_seed < 0:
        raise ValueError("coordinate_seed must be a nonnegative integer")
    for value, name in (
        (teacher_artifact_sha256, "teacher_artifact_sha256"),
        (config_fingerprint, "config_fingerprint"),
        (source_fingerprint, "source_fingerprint"),
        (runtime_fingerprint, "runtime_fingerprint"),
        (batch.parent_manifest_sha256, "parent_manifest_sha256"),
    ):
        _sha256(value, name)

    full = backend.evaluate(
        batch,
        timesteps=_TIMESTEPS,
        coordinate_seed=coordinate_seed,
        omitted_index=None,
    )
    if not isinstance(full, TeacherOutput):
        raise ValueError("teacher backend must return TeacherOutput")
    full.validate()
    records: list[InfluenceRecord] = []
    for index in range(_HISTORY_ITEMS):
        leave_one_out = backend.evaluate(
            batch,
            timesteps=_TIMESTEPS,
            coordinate_seed=coordinate_seed,
            omitted_index=index,
        )
        if not isinstance(leave_one_out, TeacherOutput):
            raise ValueError("teacher backend must return TeacherOutput")
        velocity, hidden, marker, structural = _components(full, leave_one_out)
        record = InfluenceRecord(
            sample_id=batch.sample_id,
            split=_label_split(batch.split),
            item_index=index,
            item_id=batch.item_ids[index],
            item_sha256=batch.content_sha256[index],
            source_sha256=batch.source_sha256[index],
            media_sha256=batch.media_sha256[index],
            velocity=velocity,
            hidden=hidden,
            marker=marker,
            structural=structural,
            score_q=quantize_half_up(
                _weighted_score(velocity, hidden, marker, structural), 1_000_000
            ),
            teacher_artifact_sha256=teacher_artifact_sha256,
            coordinate_seed=coordinate_seed,
            timesteps=_TIMESTEPS,
            config_fingerprint=config_fingerprint,
            source_fingerprint=source_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            parent_manifest_sha256=batch.parent_manifest_sha256,
        )
        records.append(record.validate())
    return tuple(records)


def measure_sealed_teacher_influence(
    batch: HistoryBatch,
    backend: TeacherBackend,
    *,
    coordinate_seed: int,
) -> tuple[TeacherOutput, tuple[int, ...]]:
    """Measure full-plus-eight influence in memory without creating test label records."""
    if not isinstance(batch, HistoryBatch):
        raise ValueError("batch must be a HistoryBatch")
    batch.validate()
    if batch.split != "test":
        raise ValueError("sealed influence measurement is restricted to the test split")
    if type(coordinate_seed) is not int or coordinate_seed not in {101, 202, 303}:
        raise ValueError("sealed influence seed must be one of the three Decision 1 seeds")
    full = backend.evaluate(
        batch,
        timesteps=_TIMESTEPS,
        coordinate_seed=coordinate_seed,
        omitted_index=None,
    )
    if not isinstance(full, TeacherOutput):
        raise ValueError("teacher backend must return TeacherOutput")
    full.validate()
    scores = []
    for index in range(_HISTORY_ITEMS):
        leave_one_out = backend.evaluate(
            batch,
            timesteps=_TIMESTEPS,
            coordinate_seed=coordinate_seed,
            omitted_index=index,
        )
        if not isinstance(leave_one_out, TeacherOutput):
            raise ValueError("teacher backend must return TeacherOutput")
        scores.append(
            quantize_half_up(_weighted_score(*_components(full, leave_one_out)), 1_000_000)
        )
    return full, tuple(scores)


def validate_completed_influence_records(
    records: Sequence[InfluenceRecord],
) -> tuple[InfluenceRecord, ...]:
    """Validate and return canonical complete eight-record teacher-label groups."""
    validated = tuple(records)
    for record in validated:
        record.validate()
    if not validated or len(validated) % _HISTORY_ITEMS:
        raise ValueError("resume requires a validated completed-record set of eight items")
    shard_fingerprints = {
        (
            record.teacher_artifact_sha256,
            record.config_fingerprint,
            record.source_fingerprint,
            record.runtime_fingerprint,
            record.parent_manifest_sha256,
        )
        for record in validated
    }
    if len(shard_fingerprints) != 1:
        raise ValueError(
            "an influence shard cannot mix teacher, config, source, runtime, or parent fingerprints"
        )
    identities: set[tuple[str, str]] = set()
    sample_ids: set[str] = set()
    for start in range(0, len(validated), _HISTORY_ITEMS):
        group = validated[start : start + _HISTORY_ITEMS]
        if len({record.sample_id for record in group}) != 1:
            raise ValueError("each completed-record set must belong to one sample")
        sample_id = group[0].sample_id
        if sample_id in sample_ids:
            raise ValueError("each sample may have only one completed-record set")
        sample_ids.add(sample_id)
        if tuple(record.item_index for record in group) != tuple(range(_HISTORY_ITEMS)):
            raise ValueError("each completed-record set must contain canonical item indices 0-7")
        run_identities = {
            (
                record.split,
                record.teacher_artifact_sha256,
                record.coordinate_seed,
                record.timesteps,
                record.config_fingerprint,
                record.source_fingerprint,
                record.runtime_fingerprint,
                record.parent_manifest_sha256,
            )
            for record in group
        }
        if len(run_identities) != 1:
            raise ValueError("a completed-record set cannot mix run fingerprints or coordinates")
        for record in group:
            identity = (record.sample_id, record.item_id)
            if identity in identities:
                raise ValueError("influence shard contains a duplicate record identity")
            identities.add(identity)
    return validated


def _load_existing(path: Path) -> tuple[InfluenceRecord, ...]:
    return load_influence_shard(path)


def load_influence_shard(path: Path) -> tuple[InfluenceRecord, ...]:
    """Load canonical complete influence rows without accepting ambiguous JSON."""
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError("influence shard must be a regular non-symlink file")
    encoded = path.read_bytes()
    if not encoded.endswith(b"\n"):
        raise ValueError("influence shard must end with one final newline")
    if b"\r" in encoded:
        raise ValueError("influence shard must use LF newlines")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("influence shard must be UTF-8") from error
    records: list[InfluenceRecord] = []
    for line_number, line in enumerate(text[:-1].split("\n"), start=1):
        if not line:
            raise ValueError(f"influence shard contains a blank row at line {line_number}")
        try:
            payload = json.loads(line, object_pairs_hook=_unique_object)
        except (_DuplicateKeyError, json.JSONDecodeError) as error:
            raise ValueError(f"influence shard row {line_number}: {error}") from error
        if type(payload) is not dict:
            raise ValueError(f"influence shard row {line_number} must be an object")
        canonical = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        if line != canonical:
            raise ValueError(f"influence shard row {line_number} is not canonical JSON")
        records.append(InfluenceRecord.from_dict(payload))
    return validate_completed_influence_records(records)


def write_influence_shard(path: Path, records: Sequence[InfluenceRecord]) -> None:
    """Atomically write or resume a shard only from complete matching sample records."""
    if not isinstance(path, Path):
        raise ValueError("path must be a pathlib.Path")
    desired = tuple(records)
    validate_completed_influence_records(desired)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: tuple[InfluenceRecord, ...] = ()
    if path.exists():
        if not path.is_file():
            raise ValueError("influence shard path must be a regular file")
        try:
            existing = _load_existing(path)
        except ValueError as error:
            raise ValueError(
                f"existing influence shard is conflicting or tampered: {error}"
            ) from error
        validate_completed_influence_records(existing)
        if len(existing) > len(desired):
            raise ValueError("existing influence shard exceeds the expected record set")
        for current, expected in zip(existing, desired, strict=False):
            if current.to_dict() != expected.to_dict():
                raise ValueError("existing influence row is conflicting or tampered")
        if len(existing) == len(desired):
            return

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            for record in desired:
                handle.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


__all__ = (
    "InfluenceRecord",
    "TeacherBackend",
    "TeacherOutput",
    "label_influence",
    "measure_sealed_teacher_influence",
    "quantize_half_up",
    "validate_completed_influence_records",
    "write_influence_shard",
)
