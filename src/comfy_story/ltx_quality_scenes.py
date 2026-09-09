"""Pure, content-addressed scene authoring for the frozen decoded-quality gate.

This module deliberately has no LTX import.  The only model-facing operation is the injected
``materialize_rgb_frames`` boundary at the bottom of the file.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import numpy as np
import torch
from numpy.typing import NDArray

from comfy_story.contracts import tensor_sha256
from comfy_story.ltx_quality_protocol import (
    CATEGORY_CLUSTER_COUNT,
    CONFIRMATORY_CLUSTER_COUNT,
    FACT_ABSENT_SLOTS,
    FACT_ELIGIBLE_SLOTS,
    CounterfactualCluster,
    FrozenRecord,
    QualityCategory,
    SceneVariant,
    validate_confirmatory_clusters,
)

RGBFrame = NDArray[np.uint8]
VideoPreprocess = Callable[[RGBFrame], torch.Tensor]
ImageConditioner = Callable[[torch.Tensor], torch.Tensor]
VideoEncoder = Callable[[torch.Tensor], torch.Tensor]
BasePlateDecoder = Callable[[Path], RGBFrame]

_FRAME_COUNT = 9
_FRAME_SHAPE = (384, 384, 3)
_BASE_SHAPE = (1024, 1024, 3)
_LATENT_SHAPE = (1, 128, 1, 12, 12)
_SHA256_HEX = frozenset("0123456789abcdef")
_FORMAT = "duet-x-ltx-quality-scenes-v1"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty trimmed string")
    return value


def _tuple(value: object, name: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be an immutable tuple")
    return value


def _hashes(values: object, name: str) -> tuple[str, ...]:
    items = _tuple(values, name)
    if len(items) != _FRAME_COUNT:
        raise ValueError(f"{name} must contain exactly nine hashes")
    return tuple(_sha256(item, name) for item in items)


def _freeze_rgb(frame: object, shape: tuple[int, int, int], name: str) -> RGBFrame:
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.shape != shape
        or not frame.flags.c_contiguous
    ):
        raise ValueError(f"{name} must be a C-contiguous RGB uint8 {shape[0]}x{shape[1]} frame")
    frozen = np.array(frame, dtype=np.uint8, copy=True, order="C")
    frozen.flags.writeable = False
    return frozen


def rgb_frame_sha256(frame: RGBFrame) -> str:
    """Hash canonical RGB8 pixel bytes, never a PNG transport encoding."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] <= 0
        or frame.shape[1] <= 0
        or not frame.flags.c_contiguous
    ):
        raise ValueError("RGB frame must be a nonempty C-contiguous RGB uint8 array")
    rgb = _freeze_rgb(frame, cast(tuple[int, int, int], tuple(frame.shape)), "RGB frame")
    header = (
        b"duet-x-rgb8-raw-pixels-v1\0"
        + str(rgb.shape[0]).encode("ascii")
        + b"x"
        + str(rgb.shape[1]).encode("ascii")
        + b"x3\0C\0uint8\0"
    )
    return _digest(header + rgb.tobytes(order="C"))


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("base plate must be a regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("base plate must be a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BasePlateReceipt(FrozenRecord):
    """Content and workflow identity for one accepted 1024-square authored plate."""

    cluster_id: str
    archive_name: str
    source_file_sha256: str
    decoded_rgb_sha256: str
    decoded_shape: tuple[int, int, int]
    accepted_job_id: str
    executed_prompt_sha256: str
    workflow_sha256: str
    checkpoint_sha256: str
    authored_source_receipt_sha256: str

    def validate(self) -> Self:
        _nonempty(self.cluster_id, "cluster_id")
        if (
            not isinstance(self.archive_name, str)
            or not self.archive_name
            or self.archive_name.startswith("/")
            or ".." in Path(self.archive_name).parts
        ):
            raise ValueError("archive_name must be a relative archive path")
        for name in (
            "source_file_sha256",
            "decoded_rgb_sha256",
            "executed_prompt_sha256",
            "workflow_sha256",
            "checkpoint_sha256",
            "authored_source_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.decoded_shape != _BASE_SHAPE:
            raise ValueError("base plate decoded shape must be exactly 1024x1024 RGB")
        _nonempty(self.accepted_job_id, "accepted_job_id")
        return self


def _expected_cluster_ids() -> tuple[str, ...]:
    return tuple(
        [f"identity-{index:02d}" for index in range(1, 7)]
        + [f"object-{index:02d}" for index in range(1, 7)]
        + [f"temporal-{index:02d}" for index in range(1, 7)]
    )


@dataclass(frozen=True, slots=True)
class BasePlateManifest(FrozenRecord):
    format: str
    plates: tuple[BasePlateReceipt, ...]

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-base-plates-v1":
            raise ValueError("base plate manifest format changed")
        values = _tuple(self.plates, "plates")
        if len(values) != CONFIRMATORY_CLUSTER_COUNT:
            raise ValueError("base plate manifest requires exactly 18 plates")
        if not all(isinstance(value, BasePlateReceipt) for value in values):
            raise ValueError("base plate manifest contains an invalid receipt")
        receipts = cast(tuple[BasePlateReceipt, ...], values)
        for receipt in receipts:
            receipt.validate()
        if tuple(receipt.cluster_id for receipt in receipts) != _expected_cluster_ids():
            raise ValueError("base plate manifest must use the frozen sorted cluster roster")
        jobs = tuple(receipt.accepted_job_id for receipt in receipts)
        if len(set(jobs)) != len(jobs):
            raise ValueError("accepted source job may not be used twice")
        if len({receipt.archive_name for receipt in receipts}) != len(receipts):
            raise ValueError("base plate archive names must be unique")
        if len({receipt.source_file_sha256 for receipt in receipts}) != len(receipts):
            raise ValueError("base plate source-file hashes must be unique")
        if len({receipt.decoded_rgb_sha256 for receipt in receipts}) != len(receipts):
            raise ValueError("base plate decoded-pixel hashes must be unique")
        return self


@dataclass(frozen=True, slots=True)
class RightsProvenanceReceipt(FrozenRecord):
    """The canonical source-rights assertion bound into each final scene."""

    cluster_id: str
    accepted_job_id: str
    base_plate_sha256: str
    workflow_sha256: str
    checkpoint_sha256: str
    self_authored_sdxl_source: bool
    no_third_party_footage: bool
    no_third_party_audio: bool
    no_third_party_logos: bool
    no_private_person_media: bool

    def validate(self) -> Self:
        _nonempty(self.cluster_id, "cluster_id")
        _nonempty(self.accepted_job_id, "accepted_job_id")
        for name in ("base_plate_sha256", "workflow_sha256", "checkpoint_sha256"):
            _sha256(getattr(self, name), name)
        if not all(
            value is True
            for value in (
                self.self_authored_sdxl_source,
                self.no_third_party_footage,
                self.no_third_party_audio,
                self.no_third_party_logos,
                self.no_private_person_media,
            )
        ):
            raise ValueError("rights receipt must affirm every frozen source-rights condition")
        return self


@dataclass(frozen=True, slots=True)
class RejectedPredecessor(FrozenRecord):
    rejected_job_id: str
    source_file_sha256: str
    reason: str
    superseding_cluster_id: str

    def validate(self) -> Self:
        _nonempty(self.rejected_job_id, "rejected_job_id")
        _sha256(self.source_file_sha256, "rejected source file hash")
        _nonempty(self.reason, "rejected reason")
        _nonempty(self.superseding_cluster_id, "superseding_cluster_id")
        return self


@dataclass(frozen=True, slots=True)
class PredecessorExclusionLedger(FrozenRecord):
    format: str
    source_manifest_sha256: str
    cluster_ids: tuple[str, ...]
    rejected: tuple[RejectedPredecessor, ...]

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-predecessors-v1":
            raise ValueError("predecessor ledger format changed")
        _sha256(self.source_manifest_sha256, "source_manifest_sha256")
        if self.cluster_ids != _expected_cluster_ids():
            raise ValueError("predecessor ledger roster changed")
        values = _tuple(self.rejected, "rejected")
        if len(values) != 3 or not all(isinstance(item, RejectedPredecessor) for item in values):
            raise ValueError("predecessor ledger must bind exactly three rejected predecessors")
        records = cast(tuple[RejectedPredecessor, ...], values)
        for record in records:
            record.validate()
        if len({record.rejected_job_id for record in records}) != len(records):
            raise ValueError("rejected predecessor job IDs must be unique")
        if any(record.superseding_cluster_id not in self.cluster_ids for record in records):
            raise ValueError("predecessor replacement is absent from frozen roster")
        return self


@dataclass(frozen=True, slots=True)
class FactLayer(FrozenRecord):
    """A non-text pixel program.  ``kind`` names a geometric primitive, never rendered text."""

    kind: str
    rgb: tuple[int, int, int]

    def validate(self) -> Self:
        if self.kind not in {"stripes", "dots", "box", "disk", "arrow_left", "arrow_right"}:
            raise ValueError("fact layer kind is not frozen")
        if (
            type(self.rgb) is not tuple
            or len(self.rgb) != 3
            or any(type(channel) is not int or not 0 <= channel <= 255 for channel in self.rgb)
        ):
            raise ValueError("fact layer RGB must be three uint8 integers")
        return self


@dataclass(frozen=True, slots=True)
class FrozenClusterSpec(FrozenRecord):
    cluster_id: str
    category: QualityCategory
    base_plate_key: str
    fact_attribute: str
    accepted_values: tuple[str, str]
    forbidden_fact_terms: tuple[str, ...]
    fact_visible_slot: int
    scoring_region: tuple[int, int, int, int]
    reveal_interval: tuple[int, int]
    prompt: str
    negative_prompt: str
    objective_fact_rubric: str
    layers: tuple[FactLayer, FactLayer]

    def validate(self) -> Self:
        _nonempty(self.cluster_id, "cluster_id")
        if not isinstance(self.category, QualityCategory):
            raise ValueError("cluster category is invalid")
        if self.base_plate_key != self.cluster_id:
            raise ValueError("cluster base plate key must match cluster_id")
        _nonempty(self.fact_attribute, "fact_attribute")
        if (
            type(self.accepted_values) is not tuple
            or len(self.accepted_values) != 2
            or len(set(self.accepted_values)) != 2
            or any(not isinstance(value, str) or not value for value in self.accepted_values)
        ):
            raise ValueError("cluster requires two distinct fact values")
        if self.fact_visible_slot not in FACT_ELIGIBLE_SLOTS:
            raise ValueError("fact slot must be an old unprotected slot")
        left, top, right, bottom = self.scoring_region
        if not (0 <= left < right <= 384 and 0 <= top < bottom <= 384):
            raise ValueError("scoring region must be inside 384 square")
        if not (0 <= self.reveal_interval[0] < self.reveal_interval[1] <= 25):
            raise ValueError("reveal interval must be inside the frozen clip")
        if type(self.forbidden_fact_terms) is not tuple or not self.forbidden_fact_terms:
            raise ValueError("fact aliases must be a nonempty immutable tuple")
        terms = tuple(term.casefold() for term in self.forbidden_fact_terms)
        if len(set(terms)) != len(terms) or any(not term or term != term.strip() for term in terms):
            raise ValueError("fact aliases must be unique nonempty terms")
        if not {value.casefold() for value in self.accepted_values} <= set(terms):
            raise ValueError("fact aliases must cover both fact values")
        prompt_text = f"{self.prompt}\n{self.negative_prompt}".casefold()
        if any(term in prompt_text for term in terms):
            raise ValueError("generic prompt leaks a fact term or alias")
        if type(self.layers) is not tuple or len(self.layers) != 2:
            raise ValueError("cluster requires exactly two frozen fact layers")
        for layer in self.layers:
            layer.validate()
        if self.layers[0] == self.layers[1]:
            raise ValueError("counterfactual layers must be visibly distinct")
        _nonempty(self.objective_fact_rubric, "objective_fact_rubric")
        return self


def _spec(
    cluster_id: str,
    category: QualityCategory,
    values: tuple[str, str],
    aliases: tuple[str, ...],
    slot: int,
    layers: tuple[FactLayer, FactLayer],
) -> FrozenClusterSpec:
    prompt, negative, attribute, rubric = {
        QualityCategory.IDENTITY_DETAIL: (
            "Continue naturally and reveal the same marking shown earlier.",
            "text, lettering, logo, watermark",
            "garment marking",
            "The revealed marking agrees with the old visible reference.",
        ),
        QualityCategory.OBJECT_WORLD_STATE: (
            "Continue naturally and reveal what was stored behind the cover.",
            "text, lettering, logo, watermark",
            "covered state",
            "The revealed state agrees with the old visible reference.",
        ),
        QualityCategory.TEMPORAL_CONTINUITY: (
            "Continue naturally and resume the prior motion.",
            "text, lettering, logo, watermark",
            "motion direction",
            "The resumed motion agrees with the old visible reference.",
        ),
    }[category]
    return FrozenClusterSpec(
        cluster_id,
        category,
        cluster_id,
        attribute,
        values,
        aliases,
        slot,
        (80, 80, 304, 304),
        (12, 21),
        prompt,
        negative,
        rubric,
        layers,
    ).validate()


def frozen_cluster_specs() -> tuple[FrozenClusterSpec, ...]:
    """Return the sole canonical, ordered 18-row preregistered scene table."""
    amber = FactLayer("stripes", (245, 166, 35))
    cyan = FactLayer("dots", (25, 176, 207))
    box = FactLayer("box", (220, 55, 73))
    disk = FactLayer("disk", (47, 153, 87))
    left = FactLayer("arrow_left", (66, 105, 225))
    right = FactLayer("arrow_right", (230, 126, 34))
    specs = (
        _spec(
            "identity-01",
            QualityCategory.IDENTITY_DETAIL,
            ("amber stripe", "cyan dot"),
            ("amber stripe", "cyan dot", "stripe", "dot"),
            0,
            (amber, cyan),
        ),
        _spec(
            "identity-02",
            QualityCategory.IDENTITY_DETAIL,
            ("cyan dot", "amber stripe"),
            ("cyan dot", "amber stripe", "dot", "stripe"),
            1,
            (cyan, amber),
        ),
        _spec(
            "identity-03",
            QualityCategory.IDENTITY_DETAIL,
            ("amber stripe", "cyan dot"),
            ("amber stripe", "cyan dot", "stripe", "dot"),
            3,
            (amber, cyan),
        ),
        _spec(
            "identity-04",
            QualityCategory.IDENTITY_DETAIL,
            ("cyan dot", "amber stripe"),
            ("cyan dot", "amber stripe", "dot", "stripe"),
            4,
            (cyan, amber),
        ),
        _spec(
            "identity-05",
            QualityCategory.IDENTITY_DETAIL,
            ("amber stripe", "cyan dot"),
            ("amber stripe", "cyan dot", "stripe", "dot"),
            0,
            (amber, cyan),
        ),
        _spec(
            "identity-06",
            QualityCategory.IDENTITY_DETAIL,
            ("cyan dot", "amber stripe"),
            ("cyan dot", "amber stripe", "dot", "stripe"),
            1,
            (cyan, amber),
        ),
        _spec(
            "object-01",
            QualityCategory.OBJECT_WORLD_STATE,
            ("red box", "green disk"),
            ("red box", "green disk", "box", "disk"),
            3,
            (box, disk),
        ),
        _spec(
            "object-02",
            QualityCategory.OBJECT_WORLD_STATE,
            ("green disk", "red box"),
            ("green disk", "red box", "disk", "box"),
            4,
            (disk, box),
        ),
        _spec(
            "object-03",
            QualityCategory.OBJECT_WORLD_STATE,
            ("red box", "green disk"),
            ("red box", "green disk", "box", "disk"),
            0,
            (box, disk),
        ),
        _spec(
            "object-04",
            QualityCategory.OBJECT_WORLD_STATE,
            ("green disk", "red box"),
            ("green disk", "red box", "disk", "box"),
            1,
            (disk, box),
        ),
        _spec(
            "object-05",
            QualityCategory.OBJECT_WORLD_STATE,
            ("red box", "green disk"),
            ("red box", "green disk", "box", "disk"),
            3,
            (box, disk),
        ),
        _spec(
            "object-06",
            QualityCategory.OBJECT_WORLD_STATE,
            ("green disk", "red box"),
            ("green disk", "red box", "disk", "box"),
            4,
            (disk, box),
        ),
        _spec(
            "temporal-01",
            QualityCategory.TEMPORAL_CONTINUITY,
            ("left motion", "right motion"),
            ("left motion", "right motion", "leftward", "rightward"),
            0,
            (left, right),
        ),
        _spec(
            "temporal-02",
            QualityCategory.TEMPORAL_CONTINUITY,
            ("right motion", "left motion"),
            ("right motion", "left motion", "rightward", "leftward"),
            1,
            (right, left),
        ),
        _spec(
            "temporal-03",
            QualityCategory.TEMPORAL_CONTINUITY,
            ("left motion", "right motion"),
            ("left motion", "right motion", "leftward", "rightward"),
            3,
            (left, right),
        ),
        _spec(
            "temporal-04",
            QualityCategory.TEMPORAL_CONTINUITY,
            ("right motion", "left motion"),
            ("right motion", "left motion", "rightward", "leftward"),
            4,
            (right, left),
        ),
        _spec(
            "temporal-05",
            QualityCategory.TEMPORAL_CONTINUITY,
            ("left motion", "right motion"),
            ("left motion", "right motion", "leftward", "rightward"),
            0,
            (left, right),
        ),
        _spec(
            "temporal-06",
            QualityCategory.TEMPORAL_CONTINUITY,
            ("right motion", "left motion"),
            ("right motion", "left motion", "rightward", "leftward"),
            1,
            (right, left),
        ),
    )
    for spec in specs:
        spec.validate()
    if tuple(spec.cluster_id for spec in specs) != _expected_cluster_ids():
        raise RuntimeError("frozen cluster table changed")
    if any(
        sum(spec.category is category for spec in specs) != CATEGORY_CLUSTER_COUNT
        for category in QualityCategory
    ):
        raise RuntimeError("frozen category balance changed")
    return specs


def load_content_addressed_base_plate(
    path: Path, receipt: BasePlateReceipt, *, decoder: BasePlateDecoder
) -> RGBFrame:
    """Read an accepted file with before/after byte checks, then verify decoded RGB pixels."""
    receipt.validate()
    before = _file_sha256(path)
    if before != receipt.source_file_sha256:
        raise ValueError("base plate source file hash mismatch")
    decoded = _freeze_rgb(decoder(path), _BASE_SHAPE, "decoded base plate")
    after = _file_sha256(path)
    if before != after:
        raise ValueError("base plate source changed during decode")
    if rgb_frame_sha256(decoded) != receipt.decoded_rgb_sha256:
        raise ValueError("base plate decoded RGB hash mismatch")
    return decoded


def _resize_1024_to_384(plate: RGBFrame) -> RGBFrame:
    """Fixed nearest-neighbour integer index map; no Pillow or floating point arithmetic."""
    source = _freeze_rgb(plate, _BASE_SHAPE, "base plate")
    indices = (np.arange(384, dtype=np.int64) * 1024) // 384
    return _freeze_rgb(source[indices[:, None], indices[None, :], :], _FRAME_SHAPE, "resized plate")


def _layer_mask(region: tuple[int, int, int, int], layer: FactLayer) -> NDArray[np.bool_]:
    """Return the exact integer mask of one frozen non-text fact program."""
    layer.validate()
    left, top, right, bottom = region
    height, width = bottom - top, right - left
    y, x = np.indices((height, width), dtype=np.int32)
    if layer.kind == "stripes":
        return np.asarray(((x // 16) % 2) == 0)
    elif layer.kind == "dots":
        return np.asarray(((x - width // 2) ** 2 + (y - height // 2) ** 2) % 977 < 110)
    elif layer.kind == "box":
        return np.asarray(
            (x >= width // 4) & (x < 3 * width // 4) & (y >= height // 4) & (y < 3 * height // 4),
        )
    elif layer.kind == "disk":
        return np.asarray(
            (x - width // 2) ** 2 + (y - height // 2) ** 2 <= (min(width, height) // 3) ** 2,
        )
    point = x <= y if layer.kind == "arrow_left" else x >= width - 1 - y
    middle = (y >= height // 3) & (y < 2 * height // 3)
    shaft = (x >= width // 4) & (x < 3 * width // 4) & middle
    return np.asarray(point | shaft)


def _audit_fact_layer_absence(shared_frames: Sequence[RGBFrame], spec: FrozenClusterSpec) -> None:
    """Fail if either program's exact mask/palette cue is visible outside its old slot."""
    left, top, right, bottom = spec.scoring_region
    for slot, frame in enumerate(shared_frames):
        if slot == spec.fact_visible_slot:
            continue
        region = frame[top:bottom, left:right]
        for layer in spec.layers:
            mask = _layer_mask(spec.scoring_region, layer)
            if bool(np.any(np.all(region[mask] == np.asarray(layer.rgb, dtype=np.uint8), axis=1))):
                raise ValueError(
                    "fact layer palette cue must be absent from every non-visible slot"
                )


def _assert_rendered_fact_layer(
    neutral: RGBFrame, rendered: RGBFrame, spec: FrozenClusterSpec, layer: FactLayer
) -> None:
    """Ensure a selected layer creates the declared visible cue in its scoring region."""
    left, top, right, bottom = spec.scoring_region
    mask = _layer_mask(spec.scoring_region, layer)
    neutral_region = neutral[top:bottom, left:right]
    rendered_region = rendered[top:bottom, left:right]
    palette = np.asarray(layer.rgb, dtype=np.uint8)
    if not bool(np.all(rendered_region[mask] == palette)) or not bool(
        np.any(rendered_region[mask] != neutral_region[mask])
    ):
        raise ValueError("fact layer must create its declared cue in the scoring region")


def _paint(frame: RGBFrame, region: tuple[int, int, int, int], layer: FactLayer) -> RGBFrame:
    """Apply one fixed integer geometric layer in a region without text or alpha blending."""
    result = np.array(frame, copy=True, order="C")
    left, top, right, bottom = region
    mask = _layer_mask(region, layer)
    view = result[top:bottom, left:right]
    view[mask] = np.asarray(layer.rgb, dtype=np.uint8)
    return _freeze_rgb(result, _FRAME_SHAPE, "composed frame")


@dataclass(frozen=True, slots=True)
class ComposedFrames:
    frames: tuple[RGBFrame, ...]
    rgb_frame_sha256: tuple[str, ...]

    def validate(self) -> Self:
        values = _tuple(self.frames, "frames")
        if len(values) != _FRAME_COUNT:
            raise ValueError("composition requires exactly nine RGB frames")
        if not all(isinstance(frame, np.ndarray) for frame in values):
            raise ValueError("composition includes a non-array frame")
        frames = cast(tuple[RGBFrame, ...], values)
        expected = tuple(
            rgb_frame_sha256(_freeze_rgb(frame, _FRAME_SHAPE, "composed frame")) for frame in frames
        )
        if self.rgb_frame_sha256 != expected:
            raise ValueError("composed RGB frame hash drift")
        _hashes(self.rgb_frame_sha256, "rgb_frame_sha256")
        return self


@dataclass(frozen=True, slots=True)
class ComposedVariant:
    spec: FrozenClusterSpec
    variant_id: str
    frames: ComposedFrames

    @property
    def fact_value(self) -> str:
        return self.spec.accepted_values[0 if self.variant_id == "A" else 1]

    def validate(self) -> Self:
        self.spec.validate()
        if self.variant_id not in {"A", "B"}:
            raise ValueError("variant ID must be A or B")
        self.frames.validate()
        return self


@dataclass(frozen=True, slots=True)
class ComposedCluster:
    spec: FrozenClusterSpec
    base_plate_sha256: str
    variants: tuple[ComposedVariant, ComposedVariant]

    def validate(self) -> Self:
        self.spec.validate()
        _sha256(self.base_plate_sha256, "composed base plate hash")
        if type(self.variants) is not tuple or len(self.variants) != 2:
            raise ValueError("composed cluster must contain A/B variants")
        left, right = self.variants
        left.validate()
        right.validate()
        if (
            left.spec != self.spec
            or right.spec != self.spec
            or (left.variant_id, right.variant_id) != ("A", "B")
        ):
            raise ValueError("composed variants must be ordered A then B for one spec")
        differences = tuple(
            slot
            for slot, pair in enumerate(zip(left.frames.frames, right.frames.frames, strict=True))
            if not np.array_equal(*pair)
        )
        if differences != (self.spec.fact_visible_slot,):
            raise ValueError("A/B frames may differ only at the frozen fact-visible slot")
        return self


@dataclass(frozen=True, slots=True)
class RGBVariantManifest(FrozenRecord):
    variant_id: str
    rgb_frame_sha256: tuple[str, ...]

    def validate(self) -> Self:
        if self.variant_id not in {"A", "B"}:
            raise ValueError("RGB variant ID must be A or B")
        _hashes(self.rgb_frame_sha256, "rgb_frame_sha256")
        return self


@dataclass(frozen=True, slots=True)
class RGBClusterManifest(FrozenRecord):
    spec: FrozenClusterSpec
    base_plate_sha256: str
    variants: tuple[RGBVariantManifest, RGBVariantManifest]

    def validate(self) -> Self:
        self.spec.validate()
        _sha256(self.base_plate_sha256, "RGB cluster base plate hash")
        if type(self.variants) is not tuple or len(self.variants) != 2:
            raise ValueError("RGB cluster manifest requires an A/B pair")
        left, right = self.variants
        left.validate()
        right.validate()
        if (left.variant_id, right.variant_id) != ("A", "B"):
            raise ValueError("RGB cluster variants must be ordered A then B")
        for slot in range(_FRAME_COUNT):
            if slot == self.spec.fact_visible_slot:
                if left.rgb_frame_sha256[slot] == right.rgb_frame_sha256[slot]:
                    raise ValueError("RGB fact-visible slot must differ")
            elif left.rgb_frame_sha256[slot] != right.rgb_frame_sha256[slot]:
                raise ValueError("RGB variants may differ only at the fact-visible slot")
        return self


@dataclass(frozen=True, slots=True)
class RGBFrameManifest(FrozenRecord):
    format: str
    source_manifest_sha256: str
    rights_receipt_sha256: tuple[str, ...]
    predecessor_exclusion_ledger_sha256: str
    clusters: tuple[RGBClusterManifest, ...]

    def validate(self) -> Self:
        if self.format != _FORMAT:
            raise ValueError("RGB frame manifest format changed")
        _sha256(self.source_manifest_sha256, "source_manifest_sha256")
        if len(self.rights_receipt_sha256) != CONFIRMATORY_CLUSTER_COUNT:
            raise ValueError("RGB manifest must bind all rights receipts")
        for value in self.rights_receipt_sha256:
            _sha256(value, "rights receipt hash")
        _sha256(self.predecessor_exclusion_ledger_sha256, "predecessor ledger hash")
        if len(self.clusters) != CONFIRMATORY_CLUSTER_COUNT:
            raise ValueError("RGB frame manifest must contain 18 clusters")
        for cluster in self.clusters:
            cluster.validate()
        if tuple(cluster.spec.cluster_id for cluster in self.clusters) != _expected_cluster_ids():
            raise ValueError("RGB manifest cluster order changed")
        return self


@dataclass(frozen=True, slots=True)
class ComposedPopulationManifest(FrozenRecord):
    format: str
    source_manifest_sha256: str
    rights_receipt_sha256: tuple[str, ...]
    predecessor_exclusion_ledger_sha256: str
    rgb_frame_manifest_sha256: str

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-composed-population-v1":
            raise ValueError("composed population manifest format changed")
        _sha256(self.source_manifest_sha256, "source_manifest_sha256")
        if len(self.rights_receipt_sha256) != CONFIRMATORY_CLUSTER_COUNT:
            raise ValueError("composed population must bind 18 rights receipts")
        for receipt in self.rights_receipt_sha256:
            _sha256(receipt, "rights receipt hash")
        _sha256(self.predecessor_exclusion_ledger_sha256, "predecessor ledger hash")
        _sha256(self.rgb_frame_manifest_sha256, "rgb frame manifest hash")
        return self


@dataclass(frozen=True, slots=True)
class ComposedPopulation:
    source_manifest: BasePlateManifest
    rights_receipts: tuple[RightsProvenanceReceipt, ...]
    predecessor_ledger: PredecessorExclusionLedger
    clusters: tuple[ComposedCluster, ...]

    def validate(self) -> Self:
        self.source_manifest.validate()
        if self.predecessor_ledger.source_manifest_sha256 != self.source_manifest.fingerprint():
            raise ValueError("predecessor ledger source manifest drift")
        self.predecessor_ledger.validate()
        if (
            type(self.rights_receipts) is not tuple
            or len(self.rights_receipts) != CONFIRMATORY_CLUSTER_COUNT
        ):
            raise ValueError("exactly one rights receipt per cluster is required")
        if len(self.clusters) != CONFIRMATORY_CLUSTER_COUNT:
            raise ValueError("population requires 18 composed clusters")
        expected_ids = _expected_cluster_ids()
        if tuple(cluster.spec.cluster_id for cluster in self.clusters) != expected_ids:
            raise ValueError("composed population roster changed")
        receipt_by_id = {receipt.cluster_id: receipt for receipt in self.source_manifest.plates}
        rights_by_id = {receipt.cluster_id: receipt for receipt in self.rights_receipts}
        rejected_jobs = {record.rejected_job_id for record in self.predecessor_ledger.rejected}
        if any(receipt.accepted_job_id in rejected_jobs for receipt in receipt_by_id.values()):
            raise ValueError("rejected predecessor may not be used as an accepted base")
        rejected_digests = {
            record.source_file_sha256 for record in self.predecessor_ledger.rejected
        }
        if any(
            receipt.source_file_sha256 in rejected_digests for receipt in receipt_by_id.values()
        ):
            raise ValueError(
                "rejected predecessor source bytes may not be used as an accepted base"
            )
        if tuple(rights_by_id) != expected_ids:
            raise ValueError("rights receipt roster changed")
        if len(rights_by_id) != len(self.rights_receipts):
            raise ValueError("rights receipt cluster duplicated")
        for cluster in self.clusters:
            cluster.validate()
            source = receipt_by_id[cluster.spec.cluster_id]
            rights = rights_by_id[cluster.spec.cluster_id]
            rights.validate()
            if cluster.base_plate_sha256 != source.decoded_rgb_sha256:
                raise ValueError("composed base plate digest does not bind the accepted source")
            if (
                rights.accepted_job_id != source.accepted_job_id
                or rights.base_plate_sha256 != source.decoded_rgb_sha256
                or rights.workflow_sha256 != source.workflow_sha256
                or rights.checkpoint_sha256 != source.checkpoint_sha256
            ):
                raise ValueError("rights provenance does not bind the accepted source")
        return self

    def rgb_manifest(self) -> RGBFrameManifest:
        wire_clusters: list[RGBClusterManifest] = []
        for cluster in self.clusters:
            left, right = cluster.variants
            wire_clusters.append(
                RGBClusterManifest(
                    cluster.spec,
                    cluster.base_plate_sha256,
                    (
                        RGBVariantManifest(left.variant_id, left.frames.rgb_frame_sha256),
                        RGBVariantManifest(right.variant_id, right.frames.rgb_frame_sha256),
                    ),
                ).validate()
            )
        return RGBFrameManifest(
            _FORMAT,
            self.source_manifest.fingerprint(),
            tuple(receipt.fingerprint() for receipt in self.rights_receipts),
            self.predecessor_ledger.fingerprint(),
            tuple(wire_clusters),
        ).validate()

    def manifest(self) -> ComposedPopulationManifest:
        self.validate()
        return ComposedPopulationManifest(
            "duet-x-ltx-quality-composed-population-v1",
            self.source_manifest.fingerprint(),
            tuple(receipt.fingerprint() for receipt in self.rights_receipts),
            self.predecessor_ledger.fingerprint(),
            self.rgb_manifest().fingerprint(),
        ).validate()


def compose_population(
    source_manifest: BasePlateManifest,
    base_plates: Mapping[str, RGBFrame],
    rights_receipts: Sequence[RightsProvenanceReceipt],
    predecessor_ledger: PredecessorExclusionLedger,
) -> ComposedPopulation:
    """Compose the exact frozen population with integer RGB operations only."""
    source_manifest.validate()
    specs = frozen_cluster_specs()
    if set(base_plates) != set(_expected_cluster_ids()):
        raise ValueError("base plates must contain exactly the frozen source roster")
    receipts = {receipt.cluster_id: receipt for receipt in source_manifest.plates}
    clusters: list[ComposedCluster] = []
    for spec in specs:
        plate = _freeze_rgb(base_plates[spec.base_plate_key], _BASE_SHAPE, "base plate")
        if rgb_frame_sha256(plate) != receipts[spec.cluster_id].decoded_rgb_sha256:
            raise ValueError("base plate decoded pixel hash does not match source manifest")
        neutral = _resize_1024_to_384(plate)
        shared = tuple(
            _freeze_rgb(neutral, _FRAME_SHAPE, "shared frame") for _ in range(_FRAME_COUNT)
        )
        _audit_fact_layer_absence(shared, spec)
        variants: list[ComposedVariant] = []
        for index, variant_id in enumerate(("A", "B")):
            frames = list(shared)
            frames[spec.fact_visible_slot] = _paint(
                neutral, spec.scoring_region, spec.layers[index]
            )
            _assert_rendered_fact_layer(
                neutral, frames[spec.fact_visible_slot], spec, spec.layers[index]
            )
            fixed = tuple(_freeze_rgb(frame, _FRAME_SHAPE, "composed frame") for frame in frames)
            variants.append(
                ComposedVariant(
                    spec,
                    variant_id,
                    ComposedFrames(
                        fixed, tuple(rgb_frame_sha256(frame) for frame in fixed)
                    ).validate(),
                ).validate()
            )
        clusters.append(
            ComposedCluster(
                spec,
                receipts[spec.cluster_id].decoded_rgb_sha256,
                (variants[0], variants[1]),
            ).validate()
        )
    return ComposedPopulation(
        source_manifest,
        tuple(rights_receipts),
        predecessor_ledger,
        tuple(clusters),
    ).validate()


def _finite_tensor(value: object, shape: tuple[int, ...] | None, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or not value.is_floating_point()
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
        or (shape is not None and tuple(value.shape) != shape)
    ):
        suffix = "" if shape is None else f" {list(shape)}"
        raise ValueError(f"{name} must be a finite detached floating tensor{suffix}")
    return value


@dataclass(frozen=True, slots=True)
class MaterializationManifest(FrozenRecord):
    format: str
    rgb_frame_sha256: tuple[str, ...]
    preprocessed_tensor_sha256: tuple[str, ...]
    conditioner_tensor_sha256: tuple[str, ...]
    raw_encoder_tensor_sha256: tuple[str, ...]
    normalized_latent_sha256: tuple[str, ...]

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-materialization-v1":
            raise ValueError("materialization manifest format changed")
        for name in (
            "rgb_frame_sha256",
            "preprocessed_tensor_sha256",
            "conditioner_tensor_sha256",
            "raw_encoder_tensor_sha256",
            "normalized_latent_sha256",
        ):
            _hashes(getattr(self, name), name)
        return self


@dataclass(frozen=True, slots=True)
class MaterializedSceneGuides:
    latents: tuple[torch.Tensor, ...]
    manifest: MaterializationManifest

    def validate(self) -> Self:
        self.manifest.validate()
        if type(self.latents) is not tuple or len(self.latents) != _FRAME_COUNT:
            raise ValueError("materialization must return exactly nine latents")
        for latent, expected in zip(
            self.latents, self.manifest.normalized_latent_sha256, strict=True
        ):
            if (
                latent.layout != torch.strided
                or tuple(latent.shape) != _LATENT_SHAPE
                or latent.device.type != "cpu"
                or latent.dtype is not torch.float32
                or latent.requires_grad
                or not latent.is_contiguous()
                or not bool(torch.isfinite(latent).all().item())
            ):
                raise ValueError(
                    "latent must be finite detached contiguous CPU float32 [1,128,1,12,12]"
                )
            if tensor_sha256(latent) != expected:
                raise ValueError("normalized latent hash drift")
        return self


@dataclass(frozen=True, slots=True)
class FinalSceneReceipt(FrozenRecord):
    """One final-scene pointer to the complete, intermediate-aware guide receipt."""

    scene_id: str
    scene_variant_sha256: str
    materialization_manifest_sha256: str

    def validate(self) -> Self:
        _nonempty(self.scene_id, "scene_id")
        _sha256(self.scene_variant_sha256, "scene_variant_sha256")
        _sha256(self.materialization_manifest_sha256, "materialization_manifest_sha256")
        return self


@dataclass(frozen=True, slots=True)
class MaterializedPopulationManifest(FrozenRecord):
    """The final parent-linked evidence manifest retained alongside Task-1 scenes."""

    format: str
    source_manifest_sha256: str
    rgb_frame_manifest_sha256: str
    final_scene_receipts: tuple[FinalSceneReceipt, ...]

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-materialized-population-v1":
            raise ValueError("materialized population manifest format changed")
        _sha256(self.source_manifest_sha256, "source_manifest_sha256")
        _sha256(self.rgb_frame_manifest_sha256, "rgb_frame_manifest_sha256")
        if type(self.final_scene_receipts) is not tuple or len(self.final_scene_receipts) != 36:
            raise ValueError("final materialized manifest requires exactly 36 scene receipts")
        for receipt in self.final_scene_receipts:
            receipt.validate()
        if len({receipt.scene_id for receipt in self.final_scene_receipts}) != 36:
            raise ValueError("final materialized manifest scene IDs must be unique")
        return self


@dataclass(frozen=True, slots=True)
class MaterializedPopulation:
    """Validated Task-1 scenes plus the retained guide manifests that they cannot carry directly."""

    composition: ComposedPopulation
    clusters: tuple[CounterfactualCluster, ...]
    guides: tuple[MaterializedSceneGuides, ...]
    manifest: MaterializedPopulationManifest

    def validate(self) -> Self:
        self.composition.validate()
        clusters = validate_confirmatory_clusters(self.clusters)
        if type(self.guides) is not tuple or len(self.guides) != 36:
            raise ValueError("materialized population requires exactly 36 guide receipts")
        for guide in self.guides:
            guide.validate()
        self.manifest.validate()
        if self.manifest.source_manifest_sha256 != self.composition.source_manifest.fingerprint():
            raise ValueError("final materialized manifest source parent drift")
        if self.manifest.rgb_frame_manifest_sha256 != self.composition.rgb_manifest().fingerprint():
            raise ValueError("final materialized manifest RGB parent drift")
        expected = tuple(
            FinalSceneReceipt(scene.scene_id, scene.fingerprint(), guide.manifest.fingerprint())
            for scene, guide in zip(
                (scene for cluster in clusters for scene in cluster.variants),
                self.guides,
                strict=True,
            )
        )
        if self.manifest.final_scene_receipts != expected:
            raise ValueError("final materialized manifest intermediate receipt drift")
        return self


def materialize_rgb_frames(
    frames: Sequence[RGBFrame],
    *,
    video_preprocess: VideoPreprocess,
    image_conditioner: ImageConditioner,
    video_encoder: VideoEncoder,
) -> MaterializedSceneGuides:
    """Materialize nine guides through the injected official LTX boundary, one frame at a time."""
    if len(frames) != _FRAME_COUNT:
        raise ValueError("materializer requires exactly nine RGB frames")
    rgb_hashes: list[str] = []
    preprocessed_hashes: list[str] = []
    conditioner_hashes: list[str] = []
    raw_hashes: list[str] = []
    normalized_hashes: list[str] = []
    latents: list[torch.Tensor] = []
    for frame in frames:
        frozen = _freeze_rgb(frame, _FRAME_SHAPE, "materializer input")
        rgb_hashes.append(rgb_frame_sha256(frozen))
        preprocessed = _finite_tensor(
            video_preprocess(np.array(frozen, copy=True, order="C")),
            (3, 384, 384),
            "video_preprocess output",
        )
        preprocessed_hashes.append(tensor_sha256(preprocessed))
        conditioned = _finite_tensor(
            image_conditioner(preprocessed.detach().clone()), None, "ImageConditioner output"
        )
        conditioner_hashes.append(tensor_sha256(conditioned))
        raw = _finite_tensor(
            video_encoder(conditioned.detach().clone()), _LATENT_SHAPE, "VideoEncoder output"
        )
        raw_hashes.append(tensor_sha256(raw))
        normalized = raw.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        if not bool(torch.isfinite(normalized).all().item()):
            raise ValueError("normalized latent must be finite")
        latents.append(normalized)
        normalized_hashes.append(tensor_sha256(normalized))
    manifest = MaterializationManifest(
        "duet-x-ltx-quality-materialization-v1",
        tuple(rgb_hashes),
        tuple(preprocessed_hashes),
        tuple(conditioner_hashes),
        tuple(raw_hashes),
        tuple(normalized_hashes),
    ).validate()
    return MaterializedSceneGuides(tuple(latents), manifest).validate()


def _protocol_cluster_from_materialized(
    population: ComposedPopulation,
    cluster: ComposedCluster,
    materialized: tuple[MaterializedSceneGuides, MaterializedSceneGuides],
) -> CounterfactualCluster:
    """Internal conversion after the complete source/RGB/rights/ledger population is validated."""
    population.validate()
    cluster.validate()
    registered_cluster = next(
        (
            value
            for value in population.clusters
            if value.spec.cluster_id == cluster.spec.cluster_id
        ),
        None,
    )
    if registered_cluster is not cluster:
        raise ValueError("materialization cluster must be the registered population member")
    source_by_id = {receipt.cluster_id: receipt for receipt in population.source_manifest.plates}
    rights_by_id = {receipt.cluster_id: receipt for receipt in population.rights_receipts}
    source = source_by_id.get(cluster.spec.cluster_id)
    rights = rights_by_id.get(cluster.spec.cluster_id)
    if source is None or rights is None:
        raise ValueError("composed cluster is absent from its validated population context")
    source.validate()
    rights.validate()
    predecessor_ledger = population.predecessor_ledger
    predecessor_ledger.validate()
    if source.cluster_id != cluster.spec.cluster_id or rights.cluster_id != source.cluster_id:
        raise ValueError("source/rights receipts do not match composed cluster")
    if (
        rights.accepted_job_id != source.accepted_job_id
        or rights.base_plate_sha256 != source.decoded_rgb_sha256
        or rights.workflow_sha256 != source.workflow_sha256
        or rights.checkpoint_sha256 != source.checkpoint_sha256
    ):
        raise ValueError("rights provenance does not bind the accepted source")
    variants: list[SceneVariant] = []
    for composed, guides in zip(cluster.variants, materialized, strict=True):
        guides.validate()
        if guides.manifest.rgb_frame_sha256 != composed.frames.rgb_frame_sha256:
            raise ValueError("materialization RGB parent hash drift")
        variants.append(
            SceneVariant(
                scene_id=f"{cluster.spec.cluster_id}-{composed.variant_id.lower()}",
                cluster_id=cluster.spec.cluster_id,
                variant_id=composed.variant_id,
                category=cluster.spec.category,
                fact_attribute=cluster.spec.fact_attribute,
                fact_value=composed.fact_value,
                counterfactual_value=cluster.spec.accepted_values[
                    1 if composed.variant_id == "A" else 0
                ],
                accepted_values=cluster.spec.accepted_values,
                fact_visible_slot=cluster.spec.fact_visible_slot,
                fact_absent_slots=FACT_ABSENT_SLOTS,
                prompt=cluster.spec.prompt,
                negative_prompt=cluster.spec.negative_prompt,
                forbidden_fact_terms=cluster.spec.forbidden_fact_terms,
                base_plate_sha256=source.decoded_rgb_sha256,
                rgb_frame_sha256=composed.frames.rgb_frame_sha256,
                preprocessed_tensor_sha256=guides.manifest.preprocessed_tensor_sha256,
                vae_latent_sha256=guides.manifest.normalized_latent_sha256,
                scoring_region=cluster.spec.scoring_region,
                reveal_interval=cluster.spec.reveal_interval,
                objective_fact_rubric=cluster.spec.objective_fact_rubric,
                source_workflow_sha256=source.workflow_sha256,
                rights_receipt_sha256=rights.fingerprint(),
                predecessor_exclusion_ledger_sha256=predecessor_ledger.fingerprint(),
            ).validate()
        )
    return CounterfactualCluster(
        cluster.spec.cluster_id, cluster.spec.category, (variants[0], variants[1])
    ).validate()


def validate_materialized_population(
    population: ComposedPopulation,
    materialized: Mapping[str, tuple[MaterializedSceneGuides, MaterializedSceneGuides]],
) -> MaterializedPopulation:
    """Produce final scenes and retain all linked intermediate materialization receipts."""
    population.validate()
    if set(materialized) != set(_expected_cluster_ids()):
        raise ValueError("materialized guide roster drift")
    clusters = tuple(
        _protocol_cluster_from_materialized(
            population,
            cluster,
            materialized[cluster.spec.cluster_id],
        )
        for cluster in population.clusters
    )
    validated_clusters = validate_confirmatory_clusters(clusters)
    guides = tuple(
        guide for cluster in population.clusters for guide in materialized[cluster.spec.cluster_id]
    )
    manifest = MaterializedPopulationManifest(
        "duet-x-ltx-quality-materialized-population-v1",
        population.source_manifest.fingerprint(),
        population.rgb_manifest().fingerprint(),
        tuple(
            FinalSceneReceipt(scene.scene_id, scene.fingerprint(), guide.manifest.fingerprint())
            for scene, guide in zip(
                (scene for cluster in validated_clusters for scene in cluster.variants),
                guides,
                strict=True,
            )
        ),
    ).validate()
    return MaterializedPopulation(population, validated_clusters, guides, manifest).validate()


__all__ = [
    "BasePlateManifest",
    "BasePlateReceipt",
    "ComposedCluster",
    "ComposedFrames",
    "ComposedPopulation",
    "ComposedPopulationManifest",
    "ComposedVariant",
    "FactLayer",
    "FinalSceneReceipt",
    "FrozenClusterSpec",
    "ImageConditioner",
    "MaterializationManifest",
    "MaterializedPopulation",
    "MaterializedPopulationManifest",
    "MaterializedSceneGuides",
    "PredecessorExclusionLedger",
    "RGBClusterManifest",
    "RGBFrame",
    "RGBFrameManifest",
    "RGBVariantManifest",
    "RejectedPredecessor",
    "RightsProvenanceReceipt",
    "VideoEncoder",
    "VideoPreprocess",
    "compose_population",
    "frozen_cluster_specs",
    "load_content_addressed_base_plate",
    "materialize_rgb_frames",
    "rgb_frame_sha256",
    "validate_materialized_population",
]
