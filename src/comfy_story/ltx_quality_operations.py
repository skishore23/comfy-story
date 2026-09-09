"""Fail-closed operational foundations for decoded-quality development rehearsal.

This module deliberately contains no import of the external LTX checkout.  Production callers
inject the pinned APIs after an import-only preflight; unit tests use local lifecycle fakes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, Self, cast

import numpy as np
import torch
from PIL import Image

from comfy_story.contracts import tensor_sha256
from comfy_story.ltx_quality_generation import (
    GenerationStatusCode,
    PhaseInputSeal,
    _matches_frozen_target_device,
    load_phase_input_seal,
)
from comfy_story.ltx_quality_protocol import (
    CONFIRMATORY_SCENE_COUNT,
    CounterfactualCluster,
    FrozenRecord,
    Method,
    ReviewPhase,
    ReviewPhaseHistory,
    RuntimeCoordinate,
    canonical_json,
    canonical_sha256,
    scene_manifest_sha256,
    validate_confirmatory_clusters,
)
from comfy_story.ltx_quality_scenes import (
    BasePlateManifest,
    ComposedPopulation,
    ComposedPopulationManifest,
    MaterializationManifest,
    MaterializedSceneGuides,
    PredecessorExclusionLedger,
    RGBFrame,
    RGBFrameManifest,
    RightsProvenanceReceipt,
    compose_population,
    load_content_addressed_base_plate,
    rgb_frame_sha256,
    validate_materialized_population,
)
from comfy_story.media_probe import (
    DecodedMediaFacts,
    _run_linux_descriptor_stable_media_tool,
    probe_decoded_video,
)

_SHA256_HEX = frozenset("0123456789abcdef")
_FRAME_SHAPE = (384, 384, 3)
_PREPROCESSED_SHAPE = (1, 3, 1, 384, 384)
_GUIDE_SHAPE = (1, 128, 1, 12, 12)
_DECODED_FRAME_COUNT = 25
_DEVELOPMENT_PROMPT = "A video of a marching band."
_DEVELOPMENT_NEGATIVE_PROMPT = ""
_DEVELOPMENT_SOURCE_RELATIVE_PATH = "train/BandMarching/v_BandMarching_g06_c01.avi"
_DEVELOPMENT_SOURCE_VIDEO_SHA256 = (
    "56fc776b2ec31c16a11555a6bf9f1d3d9736f168f5fc20037e8d4ced55ee229d"
)
_DEVELOPMENT_SCENE_SHA256 = "74cba958d3b8f6ce654d198cdcd2175f97c8f3be24c3adc92478b25b73e8a107"
_DEVELOPMENT_BUNDLE_SHA256 = "fc94a4133f6fe5cfcfda7e948913827860cc8cabedbac28b94ee4ef8999c97e9"
_DEVELOPMENT_K2_SHA256 = "2ed0a9054f9357db43d5b24905d55adaf425ba0d8d79df187ce6520d9ed1c973"
_SOURCE_POPULATION_SHA256SUMS_SHA256 = (
    "151022349f38b787cc54408453838b5ddc72e26cfba8d64d034877479f4deb7d"
)
_SOURCE_POPULATION_AUDIT_SHA256 = "4b7917ce7b134e69ac5675b5058a59e5d7913cfe1395c8e25f4c65a5ec45f70e"
_BASE_PLATE_MANIFEST_SHA256 = "15d12d8e63c1b209155ae8234dcac6181f16ff19516e7d0f52f5640d684d7641"
_RGB_FRAME_MANIFEST_SHA256 = "8e214ae8e9723192edeba36ba158064f0deacee2066515e8b5535effe1d9359f"
_COMPOSED_POPULATION_SHA256 = "c951d723f96c179fa2b6b34b8a542b6038acac28bf3a4f5a71f856be3615d024"
_PREDECESSOR_LEDGER_SHA256 = "8c464fbf34af30e11c767f6c6f9990142c572d008984f736a61c79d29c4667b6"


class OperationalCommand(StrEnum):
    STATUS = "status"
    MATERIALIZE = "materialize"
    PILOT = "pilot"


class OperationalStatusCode(StrEnum):
    READY = "ready"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class PublicOperationalStatus(FrozenRecord):
    command: OperationalCommand
    phase: ReviewPhase
    status: OperationalStatusCode

    def validate(self) -> Self:
        if not isinstance(self.command, OperationalCommand):
            raise ValueError("operational command must be a frozen code")
        if not isinstance(self.phase, ReviewPhase):
            raise ValueError("operational phase must be a frozen code")
        if not isinstance(self.status, OperationalStatusCode):
            raise ValueError("operational status must be a frozen code")
        return self


@dataclass(frozen=True, slots=True)
class PhaseCleanupReceipt(FrozenRecord):
    format: str
    command: OperationalCommand
    source_commit: str
    source_archive_sha256: str
    phase_seal_sha256: str
    resource_close_completed: bool
    cuda_cleanup_completed: bool

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-phase-cleanup-v1" or self.command not in {
            OperationalCommand.MATERIALIZE,
            OperationalCommand.PILOT,
        }:
            raise ValueError("phase cleanup receipt identity changed")
        if len(self.source_commit) != 40 or any(
            character not in _SHA256_HEX for character in self.source_commit
        ):
            raise ValueError("phase cleanup source commit is malformed")
        _sha256(self.source_archive_sha256, "phase cleanup source archive")
        _sha256(self.phase_seal_sha256, "phase cleanup seal")
        if any(
            type(value) is not bool
            for value in (self.resource_close_completed, self.cuda_cleanup_completed)
        ):
            raise ValueError("phase cleanup outcomes must be explicit booleans")
        return self


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty trimmed string")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(_read_regular_bytes(path)).hexdigest()


def _read_regular_bytes(path: Path, name: str = "input") -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{name} is missing, symlinked, or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{name} must be a regular non-symlink file")
        if metadata.st_nlink != 1:
            raise ValueError(f"{name} must not be hardlinked")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_single_link_regular(path: Path, name: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{name} is missing, symlinked, or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"{name} must be a single-link regular file")
    finally:
        os.close(descriptor)


def _seal_development_media(path: Path) -> None:
    """Pin a successful engine artifact to the evidence store's private file mode."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("development media is missing, symlinked, or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("development media must be a single-link regular file")
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
            raise ValueError("development media mode could not be sealed to 0600")
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_new_file(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(data: bytes, *, newline: bool = False) -> dict[str, object]:
    try:
        raw = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evidence is not strict JSON") from error
    if type(raw) is not dict:
        raise ValueError("evidence must be a JSON object")
    expected = canonical_json(raw) + (b"\n" if newline else b"")
    if data != expected:
        raise ValueError("evidence is not canonical JSON")
    return cast(dict[str, object], raw)


def _exact_keys(raw: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(raw) != expected:
        raise ValueError(f"{name} has missing or unknown fields")


class _Encoder(Protocol):
    def __call__(self, value: torch.Tensor) -> torch.Tensor: ...


class _Conditioner(Protocol):
    def __call__(self, callback: Callable[[_Encoder], torch.Tensor]) -> torch.Tensor: ...


def _closed_materializer_dependency(*_args: object, **_kwargs: object) -> torch.Tensor:
    raise RuntimeError("pinned materializer is closed")


@dataclass(frozen=True, slots=True)
class FrameMaterializationReceipt(FrozenRecord):
    format: str
    rgb_frame_sha256: str
    preprocessed_tensor_sha256: str
    conditioner_lifecycle_sha256: str
    raw_encoder_tensor_sha256: str
    normalized_latent_sha256: str

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-frame-materialization-v1":
            raise ValueError("frame materialization receipt format changed")
        for name in (
            "rgb_frame_sha256",
            "preprocessed_tensor_sha256",
            "conditioner_lifecycle_sha256",
            "raw_encoder_tensor_sha256",
            "normalized_latent_sha256",
        ):
            _sha256(getattr(self, name), name)
        return self


@dataclass(frozen=True, slots=True)
class MaterializedFrameResult:
    latent: torch.Tensor
    receipt: FrameMaterializationReceipt
    raw_data_ptr: int


@dataclass(frozen=True, slots=True)
class PinnedLTXRGBMaterializer:
    """Adapt the official callback-owning ImageConditioner without retaining its encoder."""

    video_preprocess: Callable[..., torch.Tensor]
    image_conditioner: _Conditioner
    device: torch.device

    def __post_init__(self) -> None:
        if not callable(self.video_preprocess) or not callable(self.image_conditioner):
            raise ValueError("pinned materializer dependencies must be callable")
        if not isinstance(self.device, torch.device):
            raise ValueError("pinned materializer device must be a torch.device")

    def materialize_frame(self, frame: RGBFrame) -> MaterializedFrameResult:
        if self.image_conditioner is _closed_materializer_dependency:
            raise RuntimeError("pinned materializer is closed")
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or tuple(frame.shape) != _FRAME_SHAPE
            or not frame.flags.c_contiguous
        ):
            raise ValueError("materializer input must be C-contiguous RGB uint8 384x384")
        frozen = np.array(frame, dtype=np.uint8, copy=True, order="C")
        frame_tensor = torch.from_numpy(frozen).unsqueeze(0)
        preprocessed = self.video_preprocess(
            iter((frame_tensor,)),
            height=384,
            width=384,
            dtype=torch.bfloat16,
            device=self.device,
        )
        if (
            not isinstance(preprocessed, torch.Tensor)
            or preprocessed.layout != torch.strided
            or tuple(preprocessed.shape) != _PREPROCESSED_SHAPE
            or preprocessed.dtype is not torch.bfloat16
            or not _matches_frozen_target_device(preprocessed.device, self.device)
            or preprocessed.requires_grad
            or not bool(torch.isfinite(preprocessed).all().item())
        ):
            raise ValueError(
                "preprocessed tensor must be finite detached BF16 [1,3,1,384,384] on target device"
            )
        preprocessed_hash = tensor_sha256(preprocessed)

        def encode(encoder: _Encoder) -> torch.Tensor:
            if not callable(encoder):
                raise ValueError("conditioner callback did not receive a callable encoder")
            with torch.inference_mode():
                return encoder(preprocessed)

        raw = self.image_conditioner(encode)
        if (
            not isinstance(raw, torch.Tensor)
            or raw.layout != torch.strided
            or tuple(raw.shape) != _GUIDE_SHAPE
            or raw.dtype is not torch.bfloat16
            or not _matches_frozen_target_device(raw.device, self.device)
            or raw.requires_grad
            or not bool(torch.isfinite(raw).all().item())
        ):
            raise ValueError(
                "raw encoder tensor must be finite detached BF16 [1,128,1,12,12] on target device"
            )
        raw_hash = tensor_sha256(raw)
        raw_data_ptr = raw.data_ptr()
        normalized = raw.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        if (
            normalized.data_ptr() == raw_data_ptr
            or normalized.requires_grad
            or not normalized.is_contiguous()
            or not bool(torch.isfinite(normalized).all().item())
        ):
            raise ValueError("normalized latent must be a finite independent CPU float32 clone")
        receipt = FrameMaterializationReceipt(
            "duet-x-ltx-quality-frame-materialization-v1",
            rgb_frame_sha256(frozen),
            preprocessed_hash,
            canonical_sha256(
                {
                    "format": "duet-x-ltx-image-conditioner-lifecycle-v1",
                    "preprocessed_tensor_sha256": preprocessed_hash,
                    "raw_encoder_tensor_sha256": raw_hash,
                    "callback_scope_exited": True,
                }
            ),
            raw_hash,
            tensor_sha256(normalized),
        ).validate()
        return MaterializedFrameResult(normalized, receipt, raw_data_ptr)

    def materialize_frames(self, frames: Sequence[RGBFrame]) -> MaterializedSceneGuides:
        if len(frames) != 9:
            raise ValueError("materialization requires exactly nine RGB frames")
        results = tuple(self.materialize_frame(frame) for frame in frames)
        manifest = MaterializationManifest(
            "duet-x-ltx-quality-materialization-v1",
            tuple(result.receipt.rgb_frame_sha256 for result in results),
            tuple(result.receipt.preprocessed_tensor_sha256 for result in results),
            tuple(result.receipt.conditioner_lifecycle_sha256 for result in results),
            tuple(result.receipt.raw_encoder_tensor_sha256 for result in results),
            tuple(result.receipt.normalized_latent_sha256 for result in results),
        ).validate()
        return MaterializedSceneGuides(
            tuple(result.latent for result in results), manifest
        ).validate()

    def close(self) -> None:
        if self.image_conditioner is _closed_materializer_dependency:
            return
        close = getattr(self.image_conditioner, "close", None)
        try:
            if callable(close):
                close()
        finally:
            object.__setattr__(
                self,
                "image_conditioner",
                cast(_Conditioner, _closed_materializer_dependency),
            )
            object.__setattr__(self, "video_preprocess", _closed_materializer_dependency)


@dataclass(frozen=True, slots=True)
class AcceptedPopulationFingerprints(FrozenRecord):
    sha256s_sha256: str
    source_population_audit_sha256: str
    base_plate_manifest_sha256: str
    rgb_frame_manifest_sha256: str
    composed_population_sha256: str
    predecessor_ledger_sha256: str

    def validate(self) -> Self:
        for name in (
            "sha256s_sha256",
            "source_population_audit_sha256",
            "base_plate_manifest_sha256",
            "rgb_frame_manifest_sha256",
            "composed_population_sha256",
            "predecessor_ledger_sha256",
        ):
            _sha256(getattr(self, name), name)
        return self


@dataclass(frozen=True, slots=True)
class MaterializedBundleIdentity(FrozenRecord):
    runtime: RuntimeCoordinate
    runtime_identity_sha256: str
    model_identity_sha256: str
    ltx_source_commit: str

    def validate(self) -> Self:
        self.runtime.validate()
        _sha256(self.runtime_identity_sha256, "runtime identity")
        _sha256(self.model_identity_sha256, "model identity")
        if (
            type(self.ltx_source_commit) is not str
            or len(self.ltx_source_commit) != 40
            or any(character not in _SHA256_HEX for character in self.ltx_source_commit)
        ):
            raise ValueError("LTX source commit must be a lowercase 40-hex commit")
        return self


@dataclass(frozen=True, slots=True)
class MaterializedSceneRecord(FrozenRecord):
    scene_id: str
    scene_variant_sha256: str
    materialization_manifest_sha256: str
    ordered_latent_sha256: tuple[str, ...]

    def validate(self) -> Self:
        _nonempty(self.scene_id, "materialized scene ID")
        _sha256(self.scene_variant_sha256, "scene variant SHA-256")
        _sha256(self.materialization_manifest_sha256, "materialization manifest SHA-256")
        if type(self.ordered_latent_sha256) is not tuple or len(self.ordered_latent_sha256) != 9:
            raise ValueError("materialized scene must bind exactly nine ordered latents")
        for digest in self.ordered_latent_sha256:
            _sha256(digest, "ordered latent SHA-256")
        return self


@dataclass(frozen=True, slots=True)
class MaterializedBundleManifest(FrozenRecord):
    format: str
    source_seal_sha256: str
    scene_manifest_sha256: str
    population_fingerprints: AcceptedPopulationFingerprints
    identity: MaterializedBundleIdentity
    clusters: tuple[CounterfactualCluster, ...]
    scenes: tuple[MaterializedSceneRecord, ...]
    latent_bundle_sha256: str

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-materialized-bundle-v1":
            raise ValueError("materialized bundle format changed")
        _sha256(self.source_seal_sha256, "source seal SHA-256")
        _sha256(self.scene_manifest_sha256, "scene manifest SHA-256")
        self.population_fingerprints.validate()
        self.identity.validate()
        cluster_values = validate_confirmatory_clusters(self.clusters)
        if type(self.scenes) is not tuple or len(self.scenes) != CONFIRMATORY_SCENE_COUNT:
            raise ValueError("materialized bundle requires exactly 36 scenes")
        for scene in self.scenes:
            scene.validate()
        if len({scene.scene_id for scene in self.scenes}) != CONFIRMATORY_SCENE_COUNT:
            raise ValueError("materialized bundle scene IDs must be unique")
        variants = tuple(scene for cluster in cluster_values for scene in cluster.variants)
        if tuple((record.scene_id, record.scene_variant_sha256) for record in self.scenes) != tuple(
            (scene.scene_id, scene.fingerprint()) for scene in variants
        ):
            raise ValueError("materialized scene records changed from their bound clusters")
        _sha256(self.latent_bundle_sha256, "latent bundle file SHA-256")
        return self


@dataclass(frozen=True, slots=True)
class MaterializedBundleSeal(FrozenRecord):
    format: str
    source_seal_sha256: str
    manifest_file_sha256: str
    latent_bundle_sha256: str

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-materialized-seal-v1":
            raise ValueError("materialized seal format changed")
        for name in (
            "source_seal_sha256",
            "manifest_file_sha256",
            "latent_bundle_sha256",
        ):
            _sha256(getattr(self, name), name)
        return self


@dataclass(frozen=True, slots=True)
class LoadedMaterializedBundle:
    root: Path
    manifest: MaterializedBundleManifest
    seal: MaterializedBundleSeal
    latents_by_scene: Mapping[str, tuple[torch.Tensor, ...]]


def _materialized_manifest_from_raw(raw: Mapping[str, object]) -> MaterializedBundleManifest:
    _exact_keys(
        raw,
        {
            "format",
            "source_seal_sha256",
            "scene_manifest_sha256",
            "population_fingerprints",
            "identity",
            "clusters",
            "scenes",
            "latent_bundle_sha256",
        },
        "materialized manifest",
    )
    population = raw["population_fingerprints"]
    identity_raw = raw["identity"]
    scenes_raw = raw["scenes"]
    clusters_raw = raw["clusters"]
    if (
        type(population) is not dict
        or type(identity_raw) is not dict
        or type(scenes_raw) is not list
        or type(clusters_raw) is not list
    ):
        raise ValueError("materialized manifest nested schema is malformed")
    population_map = cast(dict[str, object], population)
    _exact_keys(
        population_map,
        {
            "sha256s_sha256",
            "source_population_audit_sha256",
            "base_plate_manifest_sha256",
            "rgb_frame_manifest_sha256",
            "composed_population_sha256",
            "predecessor_ledger_sha256",
        },
        "population fingerprints",
    )
    identity_map = cast(dict[str, object], identity_raw)
    _exact_keys(
        identity_map,
        {"runtime", "runtime_identity_sha256", "model_identity_sha256", "ltx_source_commit"},
        "materialized identity",
    )
    runtime_raw = identity_map["runtime"]
    if type(runtime_raw) is not dict:
        raise ValueError("runtime coordinate is malformed")
    runtime = RuntimeCoordinate.from_dict(runtime_raw)
    scene_records: list[MaterializedSceneRecord] = []
    for value in scenes_raw:
        if type(value) is not dict:
            raise ValueError("materialized scene record is malformed")
        row = cast(dict[str, object], value)
        _exact_keys(
            row,
            {
                "scene_id",
                "scene_variant_sha256",
                "materialization_manifest_sha256",
                "ordered_latent_sha256",
            },
            "materialized scene record",
        )
        hashes = row["ordered_latent_sha256"]
        if type(hashes) is not list:
            raise ValueError("ordered latent hashes must be an array")
        scene_records.append(
            MaterializedSceneRecord(
                cast(str, row["scene_id"]),
                cast(str, row["scene_variant_sha256"]),
                cast(str, row["materialization_manifest_sha256"]),
                tuple(cast(list[str], hashes)),
            ).validate()
        )
    return MaterializedBundleManifest(
        cast(str, raw["format"]),
        cast(str, raw["source_seal_sha256"]),
        cast(str, raw["scene_manifest_sha256"]),
        AcceptedPopulationFingerprints(
            cast(str, population_map["sha256s_sha256"]),
            cast(str, population_map["source_population_audit_sha256"]),
            cast(str, population_map["base_plate_manifest_sha256"]),
            cast(str, population_map["rgb_frame_manifest_sha256"]),
            cast(str, population_map["composed_population_sha256"]),
            cast(str, population_map["predecessor_ledger_sha256"]),
        ),
        MaterializedBundleIdentity(
            runtime,
            cast(str, identity_map["runtime_identity_sha256"]),
            cast(str, identity_map["model_identity_sha256"]),
            cast(str, identity_map["ltx_source_commit"]),
        ),
        tuple(
            CounterfactualCluster.from_dict(cast(dict[str, object], value)).validate()
            for value in clusters_raw
            if type(value) is dict
        ),
        tuple(scene_records),
        cast(str, raw["latent_bundle_sha256"]),
    ).validate()


def _seal_from_raw(raw: Mapping[str, object]) -> MaterializedBundleSeal:
    _exact_keys(
        raw,
        {"format", "source_seal_sha256", "manifest_file_sha256", "latent_bundle_sha256"},
        "materialized seal",
    )
    return MaterializedBundleSeal(
        cast(str, raw["format"]),
        cast(str, raw["source_seal_sha256"]),
        cast(str, raw["manifest_file_sha256"]),
        cast(str, raw["latent_bundle_sha256"]),
    ).validate()


def publish_materialized_bundle(
    destination: Path,
    *,
    source_seal: PhaseInputSeal,
    clusters: Sequence[Any],
    latents_by_scene: Mapping[str, Sequence[torch.Tensor]],
    materialization_manifest_by_scene: Mapping[str, str],
    population_fingerprints: AcceptedPopulationFingerprints,
    identity: MaterializedBundleIdentity,
) -> LoadedMaterializedBundle:
    """Atomically publish the complete 36x9 tensor bundle and canonical bindings."""
    source_seal.validate()
    if source_seal.next_command != "materialize":
        raise ValueError("materialized bundle requires a source-authored materialize seal")
    cluster_values = validate_confirmatory_clusters(clusters)
    population_fingerprints.validate()
    identity.validate()
    scenes = tuple(scene for cluster in cluster_values for scene in cluster.variants)
    if set(latents_by_scene) != {scene.scene_id for scene in scenes}:
        raise ValueError("materialized latent roster must contain exactly 36 scene IDs")
    if set(materialization_manifest_by_scene) != {scene.scene_id for scene in scenes}:
        raise ValueError("materialization receipt roster must contain exactly 36 scene IDs")
    ordered_stacks: list[torch.Tensor] = []
    records: list[MaterializedSceneRecord] = []
    for scene in scenes:
        values = tuple(latents_by_scene[scene.scene_id])
        if len(values) != 9:
            raise ValueError("every materialized scene requires exactly nine latents")
        hashes: list[str] = []
        fixed: list[torch.Tensor] = []
        for latent in values:
            if (
                not isinstance(latent, torch.Tensor)
                or latent.layout != torch.strided
                or tuple(latent.shape) != _GUIDE_SHAPE
                or latent.device.type != "cpu"
                or latent.dtype is not torch.float32
                or latent.requires_grad
                or not latent.is_contiguous()
                or not bool(torch.isfinite(latent).all().item())
            ):
                raise ValueError("bundle latent must be finite contiguous CPU float32 guide tensor")
            fixed.append(latent.detach().clone())
            hashes.append(tensor_sha256(latent))
        if tuple(hashes) != scene.vae_latent_sha256:
            raise ValueError("bundle latent hashes do not match the accepted scene")
        ordered_stacks.append(torch.stack(fixed))
        records.append(
            MaterializedSceneRecord(
                scene.scene_id,
                scene.fingerprint(),
                _sha256(
                    materialization_manifest_by_scene[scene.scene_id],
                    "materialization manifest SHA-256",
                ),
                tuple(hashes),
            ).validate()
        )
    if destination.exists() or destination.is_symlink():
        raise ValueError("materialized destination must be fresh")
    if not destination.is_absolute() or not destination.parent.is_dir():
        raise ValueError("materialized destination must be below an existing absolute directory")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    temporary.chmod(0o700)
    try:
        buffer = io.BytesIO()
        torch.save(
            {
                "format": "duet-x-ltx-quality-materialized-tensors-v1",
                "scene_ids": tuple(scene.scene_id for scene in scenes),
                "latents": torch.stack(ordered_stacks),
            },
            buffer,
        )
        tensor_bytes = buffer.getvalue()
        tensor_digest = hashlib.sha256(tensor_bytes).hexdigest()
        manifest = MaterializedBundleManifest(
            "duet-x-ltx-quality-materialized-bundle-v1",
            source_seal.fingerprint(),
            scene_manifest_sha256(cluster_values),
            population_fingerprints,
            identity,
            cluster_values,
            tuple(records),
            tensor_digest,
        ).validate()
        manifest_bytes = manifest.to_json()
        seal = MaterializedBundleSeal(
            "duet-x-ltx-quality-materialized-seal-v1",
            source_seal.fingerprint(),
            hashlib.sha256(manifest_bytes).hexdigest(),
            tensor_digest,
        ).validate()
        _write_new_file(temporary / "latents.pt", tensor_bytes)
        _write_new_file(temporary / "manifest.json", manifest_bytes)
        _write_new_file(temporary / "seal.json", seal.to_json())
        _fsync_directory(temporary)
        temporary.rename(destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return load_materialized_bundle(destination)


def load_materialized_bundle(root: Path) -> LoadedMaterializedBundle:
    if root.is_symlink() or not root.is_absolute() or not root.is_dir():
        raise ValueError("materialized bundle root must be an absolute non-symlink directory")
    if _mode(root) != 0o700:
        raise ValueError("materialized bundle root must be mode 0700")
    entries = tuple(root.iterdir())
    if {entry.name for entry in entries} != {"latents.pt", "manifest.json", "seal.json"}:
        raise ValueError("materialized bundle contains missing or extra files")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("materialized bundle contains symlinked or unsafe evidence")
    if any(_mode(entry) != 0o600 for entry in entries):
        raise ValueError("materialized bundle files must be mode 0600")
    tensor_bytes = _read_regular_bytes(root / "latents.pt", "materialized latent bundle")
    manifest_bytes = _read_regular_bytes(root / "manifest.json", "materialized manifest")
    seal_bytes = _read_regular_bytes(root / "seal.json", "materialized seal")
    manifest = _materialized_manifest_from_raw(_strict_json(manifest_bytes))
    seal = _seal_from_raw(_strict_json(seal_bytes))
    if (
        hashlib.sha256(tensor_bytes).hexdigest() != manifest.latent_bundle_sha256
        or seal.latent_bundle_sha256 != manifest.latent_bundle_sha256
        or hashlib.sha256(manifest_bytes).hexdigest() != seal.manifest_file_sha256
        or manifest.source_seal_sha256 != seal.source_seal_sha256
    ):
        raise ValueError("materialized bundle file hash or parent binding changed")
    try:
        payload = torch.load(io.BytesIO(tensor_bytes), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("materialized latent bundle cannot be loaded safely") from error
    if type(payload) is not dict or set(payload) != {"format", "scene_ids", "latents"}:
        raise ValueError("materialized latent envelope has missing or extra fields")
    if payload["format"] != "duet-x-ltx-quality-materialized-tensors-v1":
        raise ValueError("materialized latent envelope format changed")
    scene_ids = payload["scene_ids"]
    stack = payload["latents"]
    if (
        type(scene_ids) is not tuple
        or scene_ids != tuple(scene.scene_id for scene in manifest.scenes)
        or not isinstance(stack, torch.Tensor)
        or tuple(stack.shape) != (36, 9, 1, 128, 1, 12, 12)
        or stack.device.type != "cpu"
        or stack.dtype is not torch.float32
        or not stack.is_contiguous()
        or not bool(torch.isfinite(stack).all().item())
    ):
        raise ValueError("materialized latent tensor payload changed")
    by_scene: dict[str, tuple[torch.Tensor, ...]] = {}
    for scene_index, record in enumerate(manifest.scenes):
        values = tuple(stack[scene_index, slot].contiguous() for slot in range(9))
        if tuple(tensor_sha256(value) for value in values) != record.ordered_latent_sha256:
            raise ValueError("materialized ordered latent hash changed")
        by_scene[record.scene_id] = values
    return LoadedMaterializedBundle(root, manifest, seal, by_scene)


def finalize_materialized_bundle(root: Path) -> bytes:
    """Rebuild a deterministic inventory from disk only; this never has a media/model path."""
    loaded = load_materialized_bundle(root)
    return canonical_json(
        {
            "format": "duet-x-ltx-quality-materialized-finalization-v1",
            "latent_bundle_sha256": loaded.manifest.latent_bundle_sha256,
            "manifest_file_sha256": loaded.seal.manifest_file_sha256,
            "seal_file_sha256": _file_sha256(root / "seal.json"),
            "scene_ids": tuple(scene.scene_id for scene in loaded.manifest.scenes),
            "materialization_manifest_sha256": tuple(
                scene.materialization_manifest_sha256 for scene in loaded.manifest.scenes
            ),
            "ordered_latent_sha256": tuple(
                scene.ordered_latent_sha256 for scene in loaded.manifest.scenes
            ),
        }
    )


def _strict_json_sequence(data: bytes, name: str) -> list[object]:
    try:
        value = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if type(value) is not list or data != canonical_json(value) + b"\n":
        raise ValueError(f"{name} must be a canonical JSON array")
    return cast(list[object], value)


def _decode_base_plate(path: Path) -> RGBFrame:
    try:
        with Image.open(path) as image:
            image.load()
            decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as error:
        raise ValueError("accepted base plate cannot be decoded") from error
    return np.ascontiguousarray(decoded)


def _source_population_records(sums_bytes: bytes) -> tuple[tuple[str, str], ...]:
    try:
        sum_lines = sums_bytes.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("source-population SHA256SUMS is not ASCII") from error
    records: list[tuple[str, str]] = []
    for line in sum_lines:
        fields = line.split("  ")
        if len(fields) != 2:
            raise ValueError("source-population SHA256SUMS syntax changed")
        digest, relative_text = fields
        _sha256(digest, "source-population file SHA-256")
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative_text != relative.as_posix()
        ):
            raise ValueError("source-population SHA256SUMS path is unsafe")
        records.append((relative_text, digest))
    if not records or len(records) != len({path for path, _ in records}):
        raise ValueError("source-population SHA256SUMS roster is empty or duplicated")
    return tuple(records)


def load_accepted_source_population(
    root: Path,
) -> tuple[ComposedPopulation, AcceptedPopulationFingerprints]:
    """Authenticate the complete accepted source archive and reconstruct its RGB population."""
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
        or _mode(root) != 0o700
    ):
        raise ValueError("accepted source-population root must be canonical and mode 0700")
    sums_path = root / "SHA256SUMS"
    if sums_path.is_symlink() or _mode(sums_path) != 0o600:
        raise ValueError("source-population SHA256SUMS must be a mode 0600 file")
    sums_bytes = _read_regular_bytes(sums_path, "source-population SHA256SUMS")
    if hashlib.sha256(sums_bytes).hexdigest() != _SOURCE_POPULATION_SHA256SUMS_SHA256:
        raise ValueError("accepted source-population SHA256SUMS changed")
    records = _source_population_records(sums_bytes)
    expected_paths = {root / relative for relative, _ in records} | {sums_path}
    actual_paths: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("source-population archive contains a symlink")
        if path.is_dir():
            if _mode(path) != 0o700:
                raise ValueError("source-population directory must be mode 0700")
            continue
        if not path.is_file() or _mode(path) != 0o600:
            raise ValueError("source-population input must be a mode 0600 regular file")
        actual_paths.add(path)
    if actual_paths != expected_paths:
        raise ValueError("source-population archive contains missing or extra files")
    for relative_name, expected_digest in records:
        if (
            hashlib.sha256(
                _read_regular_bytes(root / relative_name, "source-population input")
            ).hexdigest()
            != expected_digest
        ):
            raise ValueError("source-population input differs from SHA256SUMS")
    audit_bytes = _read_regular_bytes(root / "audit.json", "source-population audit")
    if hashlib.sha256(audit_bytes).hexdigest() != _SOURCE_POPULATION_AUDIT_SHA256:
        raise ValueError("accepted source-population audit changed")
    _strict_json(audit_bytes, newline=True)
    base = BasePlateManifest.from_dict(
        _strict_json(
            _read_regular_bytes(root / "base-plate-manifest.json", "base-plate manifest"),
            newline=True,
        )
    ).validate()
    ledger = PredecessorExclusionLedger.from_dict(
        _strict_json(
            _read_regular_bytes(root / "predecessor-exclusion-ledger.json", "predecessor ledger"),
            newline=True,
        )
    ).validate()

    rights_raw = _strict_json_sequence(
        _read_regular_bytes(root / "rights-provenance-receipts.json", "rights-provenance receipts"),
        "rights-provenance receipts",
    )
    rights = tuple(
        RightsProvenanceReceipt.from_dict(cast(dict[str, object], value)).validate()
        for value in rights_raw
        if type(value) is dict
    )
    if len(rights) != len(rights_raw):
        raise ValueError("rights-provenance receipt roster is malformed")
    rgb_manifest = RGBFrameManifest.from_dict(
        _strict_json(
            _read_regular_bytes(root / "rgb-frame-manifest.json", "RGB frame manifest"),
            newline=True,
        )
    ).validate()
    composed_manifest = ComposedPopulationManifest.from_dict(
        _strict_json(
            _read_regular_bytes(
                root / "composed-population-manifest.json", "composed population manifest"
            ),
            newline=True,
        )
    ).validate()
    if (
        base.fingerprint() != _BASE_PLATE_MANIFEST_SHA256
        or rgb_manifest.fingerprint() != _RGB_FRAME_MANIFEST_SHA256
        or composed_manifest.fingerprint() != _COMPOSED_POPULATION_SHA256
        or ledger.fingerprint() != _PREDECESSOR_LEDGER_SHA256
    ):
        raise ValueError("accepted source-population manifest identity changed")
    plates = {
        receipt.cluster_id: load_content_addressed_base_plate(
            root / receipt.archive_name,
            receipt,
            decoder=_decode_base_plate,
        )
        for receipt in base.plates
    }
    population = compose_population(base, plates, rights, ledger)
    if population.rgb_manifest() != rgb_manifest or population.manifest() != composed_manifest:
        raise ValueError("accepted source-population reconstruction changed")
    return population, AcceptedPopulationFingerprints(
        _SOURCE_POPULATION_SHA256SUMS_SHA256,
        _SOURCE_POPULATION_AUDIT_SHA256,
        base.fingerprint(),
        rgb_manifest.fingerprint(),
        composed_manifest.fingerprint(),
        ledger.fingerprint(),
    ).validate()


def stage_source_population(
    source_root: Path, destination: Path
) -> tuple[ComposedPopulation, AcceptedPopulationFingerprints]:
    """Authenticate a read-only local population and publish a secure creator input copy."""
    if (
        not source_root.is_absolute()
        or source_root.is_symlink()
        or not source_root.is_dir()
        or source_root.resolve(strict=True) != source_root
        or _mode(source_root) & 0o022
    ):
        raise ValueError("source-population staging input must be canonical and not writable")
    _checked_local_output(destination, source_root)
    sums_path = source_root / "SHA256SUMS"
    sums_bytes = _read_regular_bytes(sums_path, "source-population staging SHA256SUMS")
    if hashlib.sha256(sums_bytes).hexdigest() != _SOURCE_POPULATION_SHA256SUMS_SHA256:
        raise ValueError("source-population staging SHA256SUMS changed")
    records = _source_population_records(sums_bytes)
    expected_paths = {source_root / relative for relative, _ in records} | {sums_path}
    contents: dict[str, bytes] = {"SHA256SUMS": sums_bytes}
    actual_paths: set[Path] = set()
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("source-population staging input contains a symlink")
        if path.is_dir():
            if _mode(path) & 0o022:
                raise ValueError("source-population staging directory is writable")
            continue
        if not path.is_file() or _mode(path) & 0o022:
            raise ValueError("source-population staging file is unsafe or writable")
        actual_paths.add(path)
    if actual_paths != expected_paths:
        raise ValueError("source-population staging input contains missing or extra files")
    for relative_name, expected_digest in records:
        encoded = _read_regular_bytes(
            source_root / relative_name, "source-population staging input"
        )
        if hashlib.sha256(encoded).hexdigest() != expected_digest:
            raise ValueError("source-population staging input differs from SHA256SUMS")
        contents[relative_name] = encoded
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.stage-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary.chmod(0o700)
        for relative_name, encoded in contents.items():
            output = temporary / relative_name
            output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            current = output.parent
            while current != temporary:
                current.chmod(0o700)
                current = current.parent
            _write_new_file(output, encoded)
        population, fingerprints = load_accepted_source_population(temporary.resolve())
        _fsync_directory(temporary)
        temporary.rename(destination)
        _fsync_directory(destination.parent)
    return population, fingerprints


def materialize_accepted_population(
    destination: Path,
    *,
    source_seal: PhaseInputSeal,
    population: ComposedPopulation,
    population_fingerprints: AcceptedPopulationFingerprints,
    identity: MaterializedBundleIdentity,
    materializer: PinnedLTXRGBMaterializer,
) -> LoadedMaterializedBundle:
    """Materialize the frozen 36x9 population and prove disk-only finalization stability."""
    population.validate()
    population_fingerprints.validate()
    expected_fingerprints = (
        _SOURCE_POPULATION_SHA256SUMS_SHA256,
        _SOURCE_POPULATION_AUDIT_SHA256,
        _BASE_PLATE_MANIFEST_SHA256,
        _RGB_FRAME_MANIFEST_SHA256,
        _COMPOSED_POPULATION_SHA256,
        _PREDECESSOR_LEDGER_SHA256,
    )
    if (
        population_fingerprints.sha256s_sha256,
        population_fingerprints.source_population_audit_sha256,
        population_fingerprints.base_plate_manifest_sha256,
        population_fingerprints.rgb_frame_manifest_sha256,
        population_fingerprints.composed_population_sha256,
        population_fingerprints.predecessor_ledger_sha256,
    ) != expected_fingerprints:
        raise ValueError("accepted source-population fingerprints changed")
    if (
        population.source_manifest.fingerprint() != _BASE_PLATE_MANIFEST_SHA256
        or population.rgb_manifest().fingerprint() != _RGB_FRAME_MANIFEST_SHA256
        or population.manifest().fingerprint() != _COMPOSED_POPULATION_SHA256
        or population.predecessor_ledger.fingerprint() != _PREDECESSOR_LEDGER_SHA256
    ):
        raise ValueError("accepted source-population records changed")
    materialized_by_cluster: dict[str, tuple[MaterializedSceneGuides, MaterializedSceneGuides]] = {}
    for cluster in population.clusters:
        left = materializer.materialize_frames(cluster.variants[0].frames.frames)
        right = materializer.materialize_frames(cluster.variants[1].frames.frames)
        materialized_by_cluster[cluster.spec.cluster_id] = (left, right)
    materialized = validate_materialized_population(population, materialized_by_cluster)
    scenes = tuple(scene for cluster in materialized.clusters for scene in cluster.variants)
    guides = tuple(
        guide
        for cluster in population.clusters
        for guide in materialized_by_cluster[cluster.spec.cluster_id]
    )
    if len(scenes) != 36 or len(guides) != 36:
        raise ValueError("materialization did not produce exactly 36 scene variants")
    loaded = publish_materialized_bundle(
        destination,
        source_seal=source_seal,
        clusters=materialized.clusters,
        latents_by_scene={
            scene.scene_id: guide.latents for scene, guide in zip(scenes, guides, strict=True)
        },
        materialization_manifest_by_scene={
            scene.scene_id: guide.manifest.fingerprint()
            for scene, guide in zip(scenes, guides, strict=True)
        },
        population_fingerprints=population_fingerprints,
        identity=identity,
    )
    if finalize_materialized_bundle(destination) != finalize_materialized_bundle(destination):
        raise ValueError("materialized disk finalization is not byte stable")
    return loaded


@dataclass(frozen=True, slots=True)
class DecodedRGB24:
    width: int
    height: int
    pixel_format: str
    frames: tuple[bytes, ...]

    def validate(self) -> Self:
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("decoded frame dimensions must be integers")
        if self.pixel_format != "rgb24":
            raise ValueError("decoded frame pixel format must be rgb24")
        if type(self.frames) is not tuple:
            raise ValueError("decoded frames must be an immutable tuple")
        expected = self.width * self.height * 3
        if any(type(frame) is not bytes or len(frame) != expected for frame in self.frames):
            raise ValueError("every decoded RGB24 frame must contain exact raw pixel bytes")
        return self


def decode_rgb24_frames(
    path: Path,
    *,
    ffmpeg_path: Path | None = None,
    ffmpeg_sha256: str | None = None,
    probe: Callable[[Path], DecodedMediaFacts] = probe_decoded_video,
    descriptor_stable: bool = False,
) -> DecodedRGB24:
    """Decode RGB24 with either the generic host tool or an exact preflighted executable."""
    facts = probe(path)
    if type(descriptor_stable) is not bool:
        raise ValueError("descriptor-stable ffmpeg mode must be boolean")
    if descriptor_stable:
        if ffmpeg_path is None or ffmpeg_sha256 is None:
            raise ValueError("descriptor-stable ffmpeg requires an exact path and digest")
        executable_digest = None
        media_digest = None
        executable = None
    elif ffmpeg_path is not None:
        if (
            not ffmpeg_path.is_absolute()
            or ffmpeg_sha256 is None
            or _file_sha256(ffmpeg_path) != ffmpeg_sha256
        ):
            raise ValueError("ffmpeg path and digest must name the preflighted executable")
        executable_digest = ffmpeg_sha256
        media_digest = _file_sha256(path)
        executable = str(ffmpeg_path)
    elif ffmpeg_sha256 is not None:
        raise ValueError("ffmpeg path and digest must be provided together")
    else:
        executable_digest = None
        media_digest = None
        executable = "ffmpeg"
    try:
        if descriptor_stable:
            completed = _run_linux_descriptor_stable_media_tool(
                executable_path=cast(Path, ffmpeg_path),
                executable_sha256=cast(str, ffmpeg_sha256),
                media_path=path,
                arguments=lambda media_fd_path: (
                    "-v",
                    "error",
                    "-i",
                    media_fd_path,
                    "-map",
                    "0:v:0",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ),
            )
        else:
            completed = subprocess.run(
                (
                    cast(str, executable),
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ),
                check=True,
                capture_output=True,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("media bytes could not be decoded to RGB24") from error
    if (
        not descriptor_stable
        and ffmpeg_path is not None
        and (_file_sha256(ffmpeg_path) != executable_digest or _file_sha256(path) != media_digest)
    ):
        raise ValueError("ffmpeg executable or decoded media changed during RGB24 decoding")
    frame_size = facts.width * facts.height * 3
    if frame_size <= 0 or len(completed.stdout) % frame_size:
        raise ValueError("decoded RGB24 byte stream is truncated")
    frames = tuple(
        completed.stdout[offset : offset + frame_size]
        for offset in range(0, len(completed.stdout), frame_size)
    )
    return DecodedRGB24(facts.width, facts.height, "rgb24", frames).validate()


def decoded_frames_sha256(
    path: Path,
    *,
    decoder: Callable[[Path], DecodedRGB24] = decode_rgb24_frames,
    probe: Callable[[Path], DecodedMediaFacts] = probe_decoded_video,
) -> str:
    """Hash the exact 25 canonical RGB24 frames, independent of container metadata."""
    try:
        facts = probe(path).validate()
    except Exception as error:
        raise ValueError("decoded media probe failed") from error
    if (facts.width, facts.height, facts.frames) != (384, 384, _DECODED_FRAME_COUNT):
        raise ValueError("decoded media must be exactly 384x384 with 25 frames")
    if not math.isclose(facts.fps, 24.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("decoded media must be exactly 24 FPS")
    try:
        decoded = decoder(path).validate()
    except Exception as error:
        raise ValueError("decoded media RGB24 conversion failed") from error
    if (decoded.width, decoded.height, len(decoded.frames)) != (384, 384, 25):
        raise ValueError("decoded RGB24 output must be exactly 384x384 with 25 frames")
    digest = hashlib.sha256()
    digest.update(b"duet-x-decoded-rgb24-frames-v1\0")
    digest.update(struct.pack(">III", decoded.width, decoded.height, len(decoded.frames)))
    digest.update(b"rgb24\0")
    for frame in decoded.frames:
        digest.update(struct.pack(">I", len(frame)))
        digest.update(frame)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DevelopmentInput:
    scene_id: str
    split: str
    prompt: str
    source_relative_path: str
    source_video_sha256: str
    protected_slots: tuple[int, int]
    current_slot: int
    latent_bundle_sha256: str
    latent_sha256: tuple[str, ...]
    latents: tuple[torch.Tensor, ...]
    scene_manifest_sha256: str
    k2_receipt_sha256: str

    def validate(self) -> Self:
        if self.scene_id != "development-00" or self.split != "development":
            raise ValueError("preserved development scene identity or split changed")
        if self.prompt != _DEVELOPMENT_PROMPT:
            raise ValueError("preserved development prompt changed")
        if self.source_relative_path != _DEVELOPMENT_SOURCE_RELATIVE_PATH:
            raise ValueError("preserved development source path changed")
        if self.source_video_sha256 != _DEVELOPMENT_SOURCE_VIDEO_SHA256:
            raise ValueError("preserved development source video changed")
        if self.protected_slots != (2, 5) or self.current_slot != 8:
            raise ValueError("development protected/current slot contract changed")
        if self.latent_bundle_sha256 != _DEVELOPMENT_BUNDLE_SHA256:
            raise ValueError("preserved development latent bundle changed")
        if type(self.latent_sha256) is not tuple or len(self.latent_sha256) != 9:
            raise ValueError("development input requires nine ordered latent hashes")
        if type(self.latents) is not tuple or len(self.latents) != 9:
            raise ValueError("development input requires nine ordered latent tensors")
        for latent, digest in zip(self.latents, self.latent_sha256, strict=True):
            _sha256(digest, "development latent SHA-256")
            if (
                latent.layout != torch.strided
                or tuple(latent.shape) != _GUIDE_SHAPE
                or latent.device.type != "cpu"
                or latent.dtype is not torch.float32
                or latent.requires_grad
                or not latent.is_contiguous()
                or not bool(torch.isfinite(latent).all().item())
                or tensor_sha256(latent) != digest
            ):
                raise ValueError("development latent tensor or ordered hash changed")
        if self.scene_manifest_sha256 != _DEVELOPMENT_SCENE_SHA256:
            raise ValueError("preserved development scene manifest changed")
        if self.k2_receipt_sha256 != _DEVELOPMENT_K2_SHA256:
            raise ValueError("preserved development K=2 receipt changed")
        return self


def _hash_list(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not list or len(value) != 9:
        raise ValueError(f"{name} must contain exactly nine hashes")
    return tuple(_sha256(item, name) for item in cast(list[object], value))


def _validate_k2_receipt(raw: Mapping[str, object]) -> None:
    _exact_keys(raw, {"format", "guides", "selected_ids", "source_order"}, "K=2 receipt")
    if raw["format"] != "duet-x-ltx-k2-binding-v1":
        raise ValueError("development K=2 receipt format changed")
    if raw["source_order"] != [2, 5, 8]:
        raise ValueError("development K=2 source order changed")
    guides = raw["guides"]
    selected = raw["selected_ids"]
    if type(guides) is not list or len(guides) != 3 or type(selected) is not list:
        raise ValueError("development K=2 receipt roster is malformed")
    ids: list[str] = []
    for index, value in enumerate(guides):
        if type(value) is not dict:
            raise ValueError("development K=2 guide is malformed")
        guide = cast(dict[str, object], value)
        _exact_keys(
            guide,
            {"guide_id", "materialized_bytes_hex", "role", "source_index", "source_sha256"},
            "development K=2 guide",
        )
        expected_role = "current" if index == 2 else "protected"
        expected_index = (2, 5, 8)[index]
        if guide["role"] != expected_role or guide["source_index"] != expected_index:
            raise ValueError("development K=2 guide role/order changed")
        if type(guide["guide_id"]) is not str:
            raise ValueError("development K=2 guide ID is malformed")
        ids.append(guide["guide_id"])
        encoded = guide["materialized_bytes_hex"]
        if type(encoded) is not str:
            raise ValueError("development K=2 guide bytes are malformed")
        try:
            materialized = bytes.fromhex(encoded)
        except ValueError as error:
            raise ValueError("development K=2 guide bytes are malformed") from error
        if hashlib.sha256(materialized).hexdigest() != guide["source_sha256"]:
            raise ValueError("development K=2 guide source hash changed")
    if selected != ids[:2]:
        raise ValueError("development K=2 selected IDs changed")


def load_development_input(
    scene_manifest_path: Path, latent_bundle_path: Path, k2_receipt_path: Path
) -> DevelopmentInput:
    """Strict-load the preserved real-probe development envelope with safe tensor semantics."""
    manifest_bytes = _read_regular_bytes(scene_manifest_path, "development scene manifest")
    tensor_bytes = _read_regular_bytes(latent_bundle_path, "development latent bundle")
    k2_bytes = _read_regular_bytes(k2_receipt_path, "development K=2 receipt")
    for path, name in (
        (scene_manifest_path, "development scene manifest"),
        (latent_bundle_path, "development latent bundle"),
        (k2_receipt_path, "development K=2 receipt"),
    ):
        if not path.is_absolute() or _mode(path) != 0o600:
            raise ValueError(f"{name} must be an absolute mode 0600 file")
    if hashlib.sha256(manifest_bytes).hexdigest() != _DEVELOPMENT_SCENE_SHA256:
        raise ValueError("preserved development scene manifest file changed")
    if hashlib.sha256(tensor_bytes).hexdigest() != _DEVELOPMENT_BUNDLE_SHA256:
        raise ValueError("preserved development latent bundle file changed")
    if hashlib.sha256(k2_bytes).hexdigest() != _DEVELOPMENT_K2_SHA256:
        raise ValueError("preserved development K=2 receipt file changed")
    manifest = _strict_json(manifest_bytes, newline=True)
    _exact_keys(
        manifest,
        {
            "format",
            "guide_semantics",
            "media",
            "prompt",
            "protected_slots",
            "scene_id",
            "split",
            "latent_bundle_sha256",
        },
        "development scene manifest",
    )
    if manifest["format"] != "duet-x-ltx-scene-v1":
        raise ValueError("development scene manifest format changed")
    expected_semantics = {
        "attention_strength": 1.0,
        "conditioning_type": "VideoConditionByReferenceLatent",
        "downscale_factor": 1,
        "strength": 1.0,
        "temporal_scale_factor": 1,
        "token_count_per_guide": 144,
    }
    if manifest["guide_semantics"] != expected_semantics:
        raise ValueError("development guide semantics changed")
    media = manifest["media"]
    if type(media) is not dict:
        raise ValueError("development media manifest is malformed")
    media_map = cast(dict[str, object], media)
    _exact_keys(
        media_map,
        {
            "format",
            "split",
            "scene_name",
            "relative_path",
            "video_sha256_before",
            "video_sha256_after",
            "decoded_frame_count",
            "selected_frame_indices",
            "rgb_frame_sha256s",
            "preprocessed_tensor_sha256s",
            "latent_sha256s",
        },
        "development media manifest",
    )
    if (
        media_map["format"] != "duet-x-ltx-ucf101-latents-v1"
        or media_map["split"] != "development"
        or type(media_map["decoded_frame_count"]) is not int
        or media_map["decoded_frame_count"] < 9
        or type(media_map["selected_frame_indices"]) is not list
        or len(cast(list[object], media_map["selected_frame_indices"])) != 9
    ):
        raise ValueError("development UCF101 media contract changed")
    _hash_list(media_map["rgb_frame_sha256s"], "development RGB hash")
    _hash_list(media_map["preprocessed_tensor_sha256s"], "development preprocessed hash")
    latent_hashes = _hash_list(media_map["latent_sha256s"], "development latent hash")
    video_hash = _sha256(media_map["video_sha256_before"], "development video hash")
    if media_map["video_sha256_after"] != video_hash:
        raise ValueError("development source video changed during original materialization")
    bundle_digest = hashlib.sha256(tensor_bytes).hexdigest()
    if manifest["latent_bundle_sha256"] != bundle_digest:
        raise ValueError("development latent bundle hash changed")
    try:
        payload = torch.load(io.BytesIO(tensor_bytes), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("development latent bundle cannot be safely loaded") from error
    if (
        type(payload) is not dict
        or set(payload) != {"format", "latents", "scene_manifest_body"}
        or payload["format"] != "duet-x-ltx-scene-latents-v2"
        or payload["scene_manifest_body"]
        != {key: value for key, value in manifest.items() if key != "latent_bundle_sha256"}
    ):
        raise ValueError("development latent envelope schema or manifest binding changed")
    stack = payload["latents"]
    if (
        not isinstance(stack, torch.Tensor)
        or tuple(stack.shape) != (9, 1, 128, 1, 12, 12)
        or stack.device.type != "cpu"
        or stack.dtype is not torch.float32
        or not stack.is_contiguous()
        or not bool(torch.isfinite(stack).all().item())
    ):
        raise ValueError("development latent envelope tensor changed")
    latents = tuple(stack[index].contiguous() for index in range(9))
    if tuple(tensor_sha256(latent) for latent in latents) != latent_hashes:
        raise ValueError("development ordered latent hashes changed")
    k2 = _strict_json(k2_bytes, newline=True)
    _validate_k2_receipt(k2)
    if manifest["protected_slots"] != [2, 5]:
        raise ValueError("development protected slots changed")
    return DevelopmentInput(
        cast(str, manifest["scene_id"]),
        cast(str, manifest["split"]),
        cast(str, manifest["prompt"]),
        cast(str, media_map["relative_path"]),
        video_hash,
        (2, 5),
        8,
        bundle_digest,
        latent_hashes,
        latents,
        hashlib.sha256(manifest_bytes).hexdigest(),
        hashlib.sha256(k2_bytes).hexdigest(),
    ).validate()


@dataclass(frozen=True, slots=True)
class DevelopmentDisjointnessReceipt(FrozenRecord):
    format: str
    scene_id: str
    split: str
    source_relative_path: str
    source_video_sha256: str
    development_latent_sha256: tuple[str, ...]
    confirmatory_scene_manifest_sha256: str
    disjoint: bool

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-development-disjointness-v1":
            raise ValueError("development disjointness receipt format changed")
        if self.scene_id != "development-00" or self.split != "development":
            raise ValueError("development disjointness identity changed")
        _nonempty(self.source_relative_path, "development source path")
        _sha256(self.source_video_sha256, "development source video SHA-256")
        if len(self.development_latent_sha256) != 9:
            raise ValueError("disjointness receipt requires nine development latent hashes")
        for digest in self.development_latent_sha256:
            _sha256(digest, "development latent SHA-256")
        _sha256(self.confirmatory_scene_manifest_sha256, "confirmatory scene manifest SHA-256")
        if self.disjoint is not True:
            raise ValueError("development disjointness must be proven true")
        return self


def _disjointness_from_raw(raw: Mapping[str, object]) -> DevelopmentDisjointnessReceipt:
    _exact_keys(
        raw,
        {
            "format",
            "scene_id",
            "split",
            "source_relative_path",
            "source_video_sha256",
            "development_latent_sha256",
            "confirmatory_scene_manifest_sha256",
            "disjoint",
        },
        "development disjointness receipt",
    )
    hashes = raw["development_latent_sha256"]
    if type(hashes) is not list:
        raise ValueError("development disjointness latent roster must be an array")
    return DevelopmentDisjointnessReceipt(
        cast(str, raw["format"]),
        cast(str, raw["scene_id"]),
        cast(str, raw["split"]),
        cast(str, raw["source_relative_path"]),
        cast(str, raw["source_video_sha256"]),
        tuple(cast(list[str], hashes)),
        cast(str, raw["confirmatory_scene_manifest_sha256"]),
        cast(bool, raw["disjoint"]),
    ).validate()


def persist_development_disjointness(root: Path, receipt: DevelopmentDisjointnessReceipt) -> Path:
    receipt.validate()
    directory = _prepare_development_directory(root, Path("development/receipts"))
    path = directory / "development-disjointness.json"
    _write_idempotent(path, receipt.to_json())
    _fsync_directory(directory)
    return path


def _all_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, FrozenRecord):
        return _all_strings(value.to_dict())
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(item for child in value.values() for item in _all_strings(child))
    if isinstance(value, (tuple, list)):
        return tuple(item for child in value for item in _all_strings(child))
    return ()


def validate_development_disjointness(
    development: DevelopmentInput,
    clusters: Sequence[Any],
    confirmatory_ancestry: Sequence[object],
) -> DevelopmentDisjointnessReceipt:
    """Prove the development scene and every content identity are absent from confirmation."""
    development.validate()
    cluster_values = validate_confirmatory_clusters(clusters)
    variants = tuple(scene for cluster in cluster_values for scene in cluster.variants)
    if development.scene_id in {scene.scene_id for scene in variants}:
        raise ValueError("development scene ID overlaps the confirmatory population")
    confirmatory_strings = set(_all_strings(tuple(scene.to_dict() for scene in variants)))
    if development.source_video_sha256 in confirmatory_strings:
        raise ValueError("development source-video hash overlaps confirmatory evidence")
    confirmatory_latents = {digest for scene in variants for digest in scene.vae_latent_sha256}
    if confirmatory_latents.intersection(development.latent_sha256):
        raise ValueError("development latent hash overlap with confirmatory evidence")
    forbidden = {
        development.scene_id,
        development.source_relative_path,
        development.source_video_sha256,
        development.latent_bundle_sha256,
        development.scene_manifest_sha256,
        development.k2_receipt_sha256,
        *development.latent_sha256,
    }
    for ancestry in confirmatory_ancestry:
        for value in _all_strings(ancestry):
            if value in forbidden or any(token in value for token in forbidden):
                raise ValueError("confirmatory ancestry contains development identity or output")
            if "development/" in value or "/development/" in value:
                raise ValueError("confirmatory ancestry contains a development path")
    return DevelopmentDisjointnessReceipt(
        "duet-x-ltx-development-disjointness-v1",
        development.scene_id,
        development.split,
        development.source_relative_path,
        development.source_video_sha256,
        development.latent_sha256,
        scene_manifest_sha256(cluster_values),
        True,
    ).validate()


def checked_development_output(root: Path, relative: Path) -> Path:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("development output root must be an absolute non-symlink directory")
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != "development"
        or ".." in relative.parts
        or "." in relative.parts
    ):
        raise ValueError("development writes must use the literal development/ namespace")
    candidate = root / relative
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and (
            current.is_symlink() or not current.is_dir() or _mode(current) != 0o700
        ):
            raise ValueError("development output path contains an unsafe or non-0700 component")
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError("development output escapes its root") from error
    return candidate


def _prepare_development_directory(root: Path, relative: Path) -> Path:
    checked_development_output(root, relative / ".permission-probe")
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir() or _mode(current) != 0o700:
                raise ValueError("development output directory must be mode 0700 and non-symlinked")
            continue
        current.mkdir(mode=0o700)
        _fsync_directory(current.parent)
    return current


class DevelopmentAttemptState(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _development_slot_ids(method: Method) -> tuple[str, ...]:
    if method is Method.FULL_HISTORY:
        return tuple(str(slot) for slot in range(9))
    if method is Method.RECENT_ANCHOR:
        return ("7", "2", "5", "8")
    if method in {Method.GATED_CORE, Method.DUET_CORE}:
        return ("method_core[0,1,3,4,6,7]", "2", "5", "8")
    raise ValueError("development method is outside the frozen four-method roster")


@dataclass(frozen=True, slots=True)
class DevelopmentForwardBinding(FrozenRecord):
    method: Method
    seed: int
    prompt: str
    negative_prompt: str
    ordered_slot_ids: tuple[str, ...]
    initial_noise_sha256: str
    method_core_sha256: str | None
    ordered_guide_sha256: tuple[str, ...]

    def validate(self) -> Self:
        if not isinstance(self.method, Method) or self.seed != 101:
            raise ValueError("development forward binding method or seed changed")
        if self.prompt != _DEVELOPMENT_PROMPT or self.negative_prompt != "":
            raise ValueError("development forward prompt or negative prompt changed")
        if self.ordered_slot_ids != _development_slot_ids(self.method):
            raise ValueError("development forward ordered slot IDs changed")
        _sha256(self.initial_noise_sha256, "development initial-noise SHA-256")
        expected_guides = 9 if self.method is Method.FULL_HISTORY else 4
        if len(self.ordered_guide_sha256) != expected_guides:
            raise ValueError("development forward ordered guide roster changed")
        for digest in self.ordered_guide_sha256:
            _sha256(digest, "development ordered guide SHA-256")
        if self.method in {Method.GATED_CORE, Method.DUET_CORE}:
            _sha256(self.method_core_sha256, "development method-core SHA-256")
        elif self.method_core_sha256 is not None:
            raise ValueError("raw-guide development method cannot claim a method core")
        return self


class DevelopmentForwardBindingLike(Protocol):
    method: Method
    seed: int
    prompt: str
    negative_prompt: str
    ordered_slot_ids: tuple[str, ...]
    initial_noise_sha256: str
    method_core_sha256: str | None
    ordered_guide_sha256: tuple[str, ...]

    def validate(self) -> Self: ...


@dataclass(frozen=True, slots=True)
class DevelopmentAttemptRecord(FrozenRecord):
    format: str
    attempt_id: str
    ordinal: int
    method: Method
    seed: int
    state: DevelopmentAttemptState
    wall_time_seconds: float
    cuda_peak_allocated_bytes: int
    cuda_peak_reserved_bytes: int
    final_latent_finite: bool
    initial_noise_sha256: str | None
    method_core_sha256: str | None
    ordered_guide_sha256: tuple[str, ...]
    final_latent_sha256: str | None
    decoded_frames_sha256: str | None
    media_sha256: str | None
    media_size_bytes: int
    media_width: int
    media_height: int
    media_frames: int
    media_fps: int
    engine_identity_sha256: str
    foundation_before_sha256: str
    foundation_after_sha256: str | None
    runtime_identity_sha256: str
    comfy_lifecycle_receipt_sha256: str
    failure_code: str | None
    prompt: str
    negative_prompt: str
    ordered_slot_ids: tuple[str, ...]
    started_receipt_sha256: str | None
    partial_media_sha256: str | None
    partial_media_size_bytes: int

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-development-attempt-v1":
            raise ValueError("development attempt format changed")
        _nonempty(self.attempt_id, "development attempt ID")
        if type(self.ordinal) is not int or self.ordinal not in range(5):
            raise ValueError("development attempt ordinal must be in the five-forward contract")
        if not isinstance(self.method, Method) or type(self.seed) is not int or self.seed != 101:
            raise ValueError("development attempt method/seed changed")
        if not isinstance(self.state, DevelopmentAttemptState):
            raise ValueError("development attempt state changed")
        if (
            type(self.wall_time_seconds) is not float
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds < 0.0
        ):
            raise ValueError("development wall time must be finite and nonnegative")
        for value, name in (
            (self.cuda_peak_allocated_bytes, "CUDA peak allocated bytes"),
            (self.cuda_peak_reserved_bytes, "CUDA peak reserved bytes"),
            (self.media_size_bytes, "development media size"),
            (self.partial_media_size_bytes, "development partial-media size"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.final_latent_finite) is not bool:
            raise ValueError("final latent finiteness must be explicit")
        for name in (
            "engine_identity_sha256",
            "foundation_before_sha256",
            "runtime_identity_sha256",
            "comfy_lifecycle_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.foundation_after_sha256 is not None:
            _sha256(self.foundation_after_sha256, "foundation_after_sha256")
        if self.prompt != _DEVELOPMENT_PROMPT or self.negative_prompt != "":
            raise ValueError("development attempt prompt or negative prompt changed")
        if self.ordered_slot_ids != _development_slot_ids(self.method):
            raise ValueError("development attempt ordered slot IDs changed")
        _sha256(self.initial_noise_sha256, "development initial-noise SHA-256")
        expected_guides = 9 if self.method is Method.FULL_HISTORY else 4
        if len(self.ordered_guide_sha256) != expected_guides:
            raise ValueError("development ordered guide roster changed")
        for digest in self.ordered_guide_sha256:
            _sha256(digest, "development ordered guide SHA-256")
        if self.method in {Method.GATED_CORE, Method.DUET_CORE}:
            _sha256(self.method_core_sha256, "development method-core SHA-256")
        elif self.method_core_sha256 is not None:
            raise ValueError("raw-guide development method cannot claim a method core")
        if self.state is DevelopmentAttemptState.SUCCEEDED:
            if not self.final_latent_finite or self.failure_code is not None:
                raise ValueError("successful development attempt has failure state")
            for successful_digest in (
                self.final_latent_sha256,
                self.decoded_frames_sha256,
                self.media_sha256,
                self.started_receipt_sha256,
            ):
                _sha256(successful_digest, "successful development attempt hash")
            if self.foundation_before_sha256 != self.foundation_after_sha256:
                raise ValueError("development model foundation changed during forward")
            if self.media_size_bytes <= 0 or self.wall_time_seconds <= 0.0:
                raise ValueError("successful development attempt requires media and positive time")
            if self.partial_media_sha256 is not None or self.partial_media_size_bytes != 0:
                raise ValueError("successful development attempt cannot claim partial media")
            if any(
                type(value) is not int
                for value in (
                    self.media_width,
                    self.media_height,
                    self.media_frames,
                    self.media_fps,
                )
            ) or (
                self.media_width,
                self.media_height,
                self.media_frames,
                self.media_fps,
            ) != (384, 384, 25, 24):
                raise ValueError("successful development media probe facts changed")
        elif self.state is DevelopmentAttemptState.FAILED:
            if (
                any(
                    digest is not None
                    for digest in (
                        self.final_latent_sha256,
                        self.decoded_frames_sha256,
                        self.media_sha256,
                    )
                )
                or self.media_size_bytes != 0
                or any(
                    value != 0
                    for value in (
                        self.media_width,
                        self.media_height,
                        self.media_frames,
                        self.media_fps,
                    )
                )
            ):
                raise ValueError("failed attempt cannot claim committed scientific output")
            _nonempty(self.failure_code, "development failure code")
            _sha256(self.started_receipt_sha256, "failed attempt started receipt SHA-256")
            if self.partial_media_sha256 is None:
                if self.partial_media_size_bytes != 0:
                    raise ValueError("failed attempt partial-media size lacks a hash")
            else:
                _sha256(self.partial_media_sha256, "failed attempt partial-media SHA-256")
                if self.partial_media_size_bytes <= 0:
                    raise ValueError("failed attempt partial media must have positive size")
        else:
            if (
                self.wall_time_seconds != 0.0
                or self.cuda_peak_allocated_bytes != 0
                or self.cuda_peak_reserved_bytes != 0
                or self.final_latent_finite
                or any(
                    value is not None
                    for value in (
                        self.final_latent_sha256,
                        self.decoded_frames_sha256,
                        self.media_sha256,
                        self.foundation_after_sha256,
                        self.failure_code,
                        self.started_receipt_sha256,
                    )
                )
                or self.media_size_bytes != 0
                or self.partial_media_sha256 is not None
                or self.partial_media_size_bytes != 0
                or any(
                    value != 0
                    for value in (
                        self.media_width,
                        self.media_height,
                        self.media_frames,
                        self.media_fps,
                    )
                )
            ):
                raise ValueError("started attempt cannot claim measured or terminal output")
        return self


def _attempt_from_raw(raw: Mapping[str, object]) -> DevelopmentAttemptRecord:
    expected = {
        "format",
        "attempt_id",
        "ordinal",
        "method",
        "seed",
        "state",
        "wall_time_seconds",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "final_latent_finite",
        "initial_noise_sha256",
        "method_core_sha256",
        "ordered_guide_sha256",
        "final_latent_sha256",
        "decoded_frames_sha256",
        "media_sha256",
        "media_size_bytes",
        "media_width",
        "media_height",
        "media_frames",
        "media_fps",
        "engine_identity_sha256",
        "foundation_before_sha256",
        "foundation_after_sha256",
        "runtime_identity_sha256",
        "comfy_lifecycle_receipt_sha256",
        "failure_code",
        "prompt",
        "negative_prompt",
        "ordered_slot_ids",
        "started_receipt_sha256",
        "partial_media_sha256",
        "partial_media_size_bytes",
    }
    _exact_keys(raw, expected, "development attempt")
    try:
        method = Method(cast(str, raw["method"]))
        state = DevelopmentAttemptState(cast(str, raw["state"]))
    except ValueError as error:
        raise ValueError("development attempt enum changed") from error
    guide_hashes = raw["ordered_guide_sha256"]
    slot_ids = raw["ordered_slot_ids"]
    if type(guide_hashes) is not list or type(slot_ids) is not list:
        raise ValueError("development ordered guide hashes and slot IDs must be arrays")
    return DevelopmentAttemptRecord(
        cast(str, raw["format"]),
        cast(str, raw["attempt_id"]),
        cast(int, raw["ordinal"]),
        method,
        cast(int, raw["seed"]),
        state,
        cast(float, raw["wall_time_seconds"]),
        cast(int, raw["cuda_peak_allocated_bytes"]),
        cast(int, raw["cuda_peak_reserved_bytes"]),
        cast(bool, raw["final_latent_finite"]),
        cast(str | None, raw["initial_noise_sha256"]),
        cast(str | None, raw["method_core_sha256"]),
        tuple(cast(list[str], guide_hashes)),
        cast(str | None, raw["final_latent_sha256"]),
        cast(str | None, raw["decoded_frames_sha256"]),
        cast(str | None, raw["media_sha256"]),
        cast(int, raw["media_size_bytes"]),
        cast(int, raw["media_width"]),
        cast(int, raw["media_height"]),
        cast(int, raw["media_frames"]),
        cast(int, raw["media_fps"]),
        cast(str, raw["engine_identity_sha256"]),
        cast(str, raw["foundation_before_sha256"]),
        cast(str | None, raw["foundation_after_sha256"]),
        cast(str, raw["runtime_identity_sha256"]),
        cast(str, raw["comfy_lifecycle_receipt_sha256"]),
        cast(str | None, raw["failure_code"]),
        cast(str, raw["prompt"]),
        cast(str, raw["negative_prompt"]),
        tuple(cast(list[str], slot_ids)),
        cast(str | None, raw["started_receipt_sha256"]),
        cast(str | None, raw["partial_media_sha256"]),
        cast(int, raw["partial_media_size_bytes"]),
    ).validate()


@dataclass(frozen=True, slots=True)
class DevelopmentAttemptStore:
    root: Path

    def validate(self) -> Self:
        if not self.root.is_absolute() or self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("development attempt root must be an absolute non-symlink directory")
        return self

    def persist(self, attempt: DevelopmentAttemptRecord) -> Path:
        self.validate()
        attempt.validate()
        if "/" in attempt.attempt_id or attempt.attempt_id in {".", ".."}:
            raise ValueError("development attempt ID is not path-safe")
        directory = _prepare_development_directory(self.root, Path("development/attempts"))
        suffix = ".started.json" if attempt.state is DevelopmentAttemptState.STARTED else ".json"
        path = directory / f"{attempt.attempt_id}{suffix}"
        encoded = attempt.to_json()
        if path.exists() or path.is_symlink():
            if _mode(path) != 0o600:
                raise ValueError("development attempt must be a mode 0600 file")
            if _read_regular_bytes(path, "development attempt") != encoded:
                raise ValueError("development attempt conflicts with persisted bytes")
            return path
        _write_new_file(path, encoded)
        _fsync_directory(directory)
        return path

    def load(self, attempt_id: str) -> DevelopmentAttemptRecord:
        _nonempty(attempt_id, "development attempt ID")
        path = checked_development_output(
            self.root, Path("development/attempts") / f"{attempt_id}.json"
        )
        if path.is_symlink() or _mode(path) != 0o600:
            raise ValueError("development attempt must be a mode 0600 non-symlink file")
        encoded = _read_regular_bytes(path, "development attempt")
        return _attempt_from_raw(_strict_json(encoded))

    def load_started(self, attempt_id: str) -> DevelopmentAttemptRecord:
        _nonempty(attempt_id, "development attempt ID")
        path = checked_development_output(
            self.root, Path("development/attempts") / f"{attempt_id}.started.json"
        )
        if path.is_symlink() or _mode(path) != 0o600:
            raise ValueError("development started attempt must be a mode 0600 non-symlink file")
        attempt = _attempt_from_raw(
            _strict_json(_read_regular_bytes(path, "development started attempt"))
        )
        if attempt.state is not DevelopmentAttemptState.STARTED:
            raise ValueError("development started attempt has a terminal state")
        return attempt


@dataclass(frozen=True, slots=True)
class DevelopmentPilotReceipt(FrozenRecord):
    format: str
    scene_id: str
    attempt: DevelopmentAttemptRecord

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-development-pilot-v1" or self.scene_id != "development-00":
            raise ValueError("development pilot identity changed")
        self.attempt.validate()
        if (
            self.attempt.ordinal != 0
            or self.attempt.method is not Method.DUET_CORE
            or self.attempt.state is not DevelopmentAttemptState.SUCCEEDED
        ):
            raise ValueError("pilot must be the standalone successful Duet forward")
        return self


@dataclass(frozen=True, slots=True)
class DevelopmentRehearsalReceipt(FrozenRecord):
    format: str
    scene_id: str
    attempts: tuple[DevelopmentAttemptRecord, ...]
    repeated_duet_final_latent_sha256: str
    repeated_duet_decoded_frames_sha256: str

    def validate(self) -> Self:
        if (
            self.format != "duet-x-ltx-development-rehearsal-v1"
            or self.scene_id != "development-00"
        ):
            raise ValueError("development rehearsal identity changed")
        if type(self.attempts) is not tuple or len(self.attempts) != 4:
            raise ValueError("development rehearsal requires four ordered forwards")
        expected = (
            Method.FULL_HISTORY,
            Method.RECENT_ANCHOR,
            Method.GATED_CORE,
            Method.DUET_CORE,
        )
        for ordinal, (attempt, method) in enumerate(zip(self.attempts, expected, strict=True), 1):
            attempt.validate()
            if (
                attempt.ordinal != ordinal
                or attempt.method is not method
                or attempt.seed != 101
                or attempt.state is not DevelopmentAttemptState.SUCCEEDED
            ):
                raise ValueError("development rehearsal order or seed changed")
        _sha256(self.repeated_duet_final_latent_sha256, "repeated Duet latent SHA-256")
        _sha256(self.repeated_duet_decoded_frames_sha256, "repeated Duet frames SHA-256")
        repeated = self.attempts[-1]
        if (
            repeated.final_latent_sha256 != self.repeated_duet_final_latent_sha256
            or repeated.decoded_frames_sha256 != self.repeated_duet_decoded_frames_sha256
        ):
            raise ValueError("repeated Duet equality check failed")
        identities = {
            (
                attempt.engine_identity_sha256,
                attempt.foundation_before_sha256,
                attempt.runtime_identity_sha256,
                attempt.comfy_lifecycle_receipt_sha256,
            )
            for attempt in self.attempts
        }
        if len(identities) != 1:
            raise ValueError("development rehearsal engine/foundation/lifecycle identity drifted")
        if len({attempt.initial_noise_sha256 for attempt in self.attempts}) != 1:
            raise ValueError("development rehearsal initial-noise identity drifted")
        return self


@dataclass(frozen=True, slots=True)
class DevelopmentRehearsedSeal(FrozenRecord):
    format: str
    phases: ReviewPhaseHistory
    source_seal_sha256: str
    pilot_receipt_sha256: str
    rehearsal_receipt_sha256: str
    disjointness_receipt_sha256: str
    nonconfirmatory: bool

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-development-rehearsed-seal-v1":
            raise ValueError("development-rehearsed seal format changed")
        self.phases.validate()
        if self.phases.phases != (
            ReviewPhase.SOURCE_AUTHORED,
            ReviewPhase.DEVELOPMENT_REHEARSED,
        ):
            raise ValueError("development-rehearsed phase transition changed")
        for name in (
            "source_seal_sha256",
            "pilot_receipt_sha256",
            "rehearsal_receipt_sha256",
            "disjointness_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.nonconfirmatory is not True:
            raise ValueError("development seal must remain explicitly nonconfirmatory")
        return self

    @classmethod
    def build(
        cls,
        *,
        source_seal: PhaseInputSeal,
        pilot: DevelopmentPilotReceipt,
        rehearsal: DevelopmentRehearsalReceipt,
        disjointness_sha256: str,
    ) -> Self:
        source_seal.validate()
        pilot.validate()
        rehearsal.validate()
        if source_seal.next_command != "pilot":
            raise ValueError("development rehearsal requires a source-authored pilot seal")
        repeated = rehearsal.attempts[-1]
        if (
            pilot.attempt.final_latent_sha256 != repeated.final_latent_sha256
            or pilot.attempt.decoded_frames_sha256 != repeated.decoded_frames_sha256
        ):
            raise ValueError("repeated Duet final-latent or decoded-frame equality failed")
        for name in (
            "engine_identity_sha256",
            "foundation_before_sha256",
            "runtime_identity_sha256",
            "comfy_lifecycle_receipt_sha256",
            "initial_noise_sha256",
            "method_core_sha256",
            "ordered_guide_sha256",
        ):
            if getattr(pilot.attempt, name) != getattr(repeated, name):
                raise ValueError("repeated Duet engine/foundation/lifecycle identity changed")
        return cls(
            "duet-x-ltx-development-rehearsed-seal-v1",
            ReviewPhaseHistory((ReviewPhase.SOURCE_AUTHORED, ReviewPhase.DEVELOPMENT_REHEARSED)),
            source_seal.fingerprint(),
            pilot.fingerprint(),
            rehearsal.fingerprint(),
            _sha256(disjointness_sha256, "disjointness receipt SHA-256"),
            True,
        ).validate()


@dataclass(frozen=True, slots=True)
class DevelopmentExecutionIdentity(FrozenRecord):
    engine_identity_sha256: str
    foundation_sha256: str
    runtime_identity_sha256: str

    def validate(self) -> Self:
        _sha256(self.engine_identity_sha256, "development engine identity")
        _sha256(self.foundation_sha256, "development foundation identity")
        _sha256(self.runtime_identity_sha256, "development runtime identity")
        return self


@dataclass(frozen=True, slots=True)
class DevelopmentForwardResult:
    path: Path
    final_latent: torch.Tensor
    initial_noise_sha256: str
    method_core_sha256: str | None
    ordered_guide_sha256: tuple[str, ...]
    decoded_frames_sha256: str
    media_facts: DecodedMediaFacts
    engine_identity_sha256: str
    foundation_before_sha256: str
    foundation_after_sha256: str
    runtime_identity_sha256: str

    def validate(self) -> Self:
        if (
            not self.path.is_absolute()
            or self.path.is_symlink()
            or not self.path.is_file()
            or self.path.stat().st_size <= 0
        ):
            raise ValueError("development forward media must be a nonempty regular file")
        if (
            not isinstance(self.final_latent, torch.Tensor)
            or self.final_latent.layout != torch.strided
            or tuple(self.final_latent.shape) != (1, 128, 4, 12, 12)
            or self.final_latent.device.type != "cpu"
            or self.final_latent.dtype is not torch.float32
            or self.final_latent.requires_grad
            or not self.final_latent.is_contiguous()
            or not bool(torch.isfinite(self.final_latent).all().item())
        ):
            raise ValueError("development final latent must be finite contiguous CPU float32")
        for name in (
            "initial_noise_sha256",
            "decoded_frames_sha256",
            "engine_identity_sha256",
            "foundation_before_sha256",
            "foundation_after_sha256",
            "runtime_identity_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.method_core_sha256 is not None:
            _sha256(self.method_core_sha256, "development method core SHA-256")
        if type(self.ordered_guide_sha256) is not tuple:
            raise ValueError("development ordered guide hashes must be immutable")
        for digest in self.ordered_guide_sha256:
            _sha256(digest, "development ordered guide SHA-256")
        self.media_facts.validate()
        if (
            self.media_facts.width,
            self.media_facts.height,
            self.media_facts.frames,
        ) != (384, 384, 25) or not math.isclose(
            self.media_facts.fps, 24.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("development decoded media facts changed")
        if self.foundation_before_sha256 != self.foundation_after_sha256:
            raise ValueError("development foundation changed during forward")
        return self


@dataclass(frozen=True, slots=True)
class DevelopmentRuntimeMetrics:
    synchronize: Callable[[], None]
    reset_peaks: Callable[[], None]
    peak_allocated_bytes: Callable[[], int]
    peak_reserved_bytes: Callable[[], int]

    def validate(self) -> Self:
        if not all(
            callable(value)
            for value in (
                self.synchronize,
                self.reset_peaks,
                self.peak_allocated_bytes,
                self.peak_reserved_bytes,
            )
        ):
            raise ValueError("development CUDA metric callbacks must be callable")
        return self


@dataclass(frozen=True, slots=True)
class DevelopmentRunArtifacts:
    pilot: DevelopmentPilotReceipt
    rehearsal: DevelopmentRehearsalReceipt
    seal: DevelopmentRehearsedSeal

    @property
    def all_attempts(self) -> tuple[DevelopmentAttemptRecord, ...]:
        return (self.pilot.attempt, *self.rehearsal.attempts)


def _write_idempotent(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or _mode(path) != 0o600
            or _read_regular_bytes(path) != data
        ):
            raise ValueError("development receipt conflicts with persisted bytes")
        return
    _write_new_file(path, data)


def _pilot_from_raw(raw: Mapping[str, object]) -> DevelopmentPilotReceipt:
    _exact_keys(raw, {"format", "scene_id", "attempt"}, "development pilot receipt")
    attempt = raw["attempt"]
    if type(attempt) is not dict:
        raise ValueError("development pilot attempt is malformed")
    return DevelopmentPilotReceipt(
        cast(str, raw["format"]),
        cast(str, raw["scene_id"]),
        _attempt_from_raw(cast(dict[str, object], attempt)),
    ).validate()


def _rehearsal_from_raw(raw: Mapping[str, object]) -> DevelopmentRehearsalReceipt:
    _exact_keys(
        raw,
        {
            "format",
            "scene_id",
            "attempts",
            "repeated_duet_final_latent_sha256",
            "repeated_duet_decoded_frames_sha256",
        },
        "development rehearsal receipt",
    )
    attempts = raw["attempts"]
    if type(attempts) is not list:
        raise ValueError("development rehearsal attempts are malformed")
    parsed = []
    for attempt in attempts:
        if type(attempt) is not dict:
            raise ValueError("development rehearsal attempt is malformed")
        parsed.append(_attempt_from_raw(cast(dict[str, object], attempt)))
    return DevelopmentRehearsalReceipt(
        cast(str, raw["format"]),
        cast(str, raw["scene_id"]),
        tuple(parsed),
        cast(str, raw["repeated_duet_final_latent_sha256"]),
        cast(str, raw["repeated_duet_decoded_frames_sha256"]),
    ).validate()


def _development_seal_from_raw(raw: Mapping[str, object]) -> DevelopmentRehearsedSeal:
    _exact_keys(
        raw,
        {
            "format",
            "phases",
            "source_seal_sha256",
            "pilot_receipt_sha256",
            "rehearsal_receipt_sha256",
            "disjointness_receipt_sha256",
            "nonconfirmatory",
        },
        "development-rehearsed seal",
    )
    phases = raw["phases"]
    if type(phases) is not dict:
        raise ValueError("development-rehearsed phase history is malformed")
    return DevelopmentRehearsedSeal(
        cast(str, raw["format"]),
        ReviewPhaseHistory.from_dict(phases),
        cast(str, raw["source_seal_sha256"]),
        cast(str, raw["pilot_receipt_sha256"]),
        cast(str, raw["rehearsal_receipt_sha256"]),
        cast(str, raw["disjointness_receipt_sha256"]),
        cast(bool, raw["nonconfirmatory"]),
    ).validate()


def persist_development_rehearsal(
    root: Path, artifacts: DevelopmentRunArtifacts
) -> tuple[Path, Path, Path]:
    """Publish the three completion records inside the literal development namespace."""
    artifacts.pilot.validate()
    artifacts.rehearsal.validate()
    artifacts.seal.validate()
    if (
        artifacts.seal.pilot_receipt_sha256 != artifacts.pilot.fingerprint()
        or artifacts.seal.rehearsal_receipt_sha256 != artifacts.rehearsal.fingerprint()
    ):
        raise ValueError("development completion seal parent binding changed")
    directory = _prepare_development_directory(root, Path("development/receipts"))
    paths = (
        directory / "development-pilot.json",
        directory / "development-rehearsal.json",
        directory / "development-rehearsed-seal.json",
    )
    for path, encoded in zip(
        paths,
        (
            artifacts.pilot.to_json(),
            artifacts.rehearsal.to_json(),
            artifacts.seal.to_json(),
        ),
        strict=True,
    ):
        _write_idempotent(path, encoded)
    _fsync_directory(directory)
    return paths


def load_development_rehearsal(root: Path) -> DevelopmentRunArtifacts:
    """Strict-load only the complete development receipts; unknown disk evidence aborts."""
    directory = checked_development_output(root, Path("development/receipts"))
    if directory.is_symlink() or not directory.is_dir() or _mode(directory) != 0o700:
        raise ValueError("development receipt directory must be a mode 0700 directory")
    expected = {
        "development-disjointness.json",
        "development-pilot.json",
        "development-rehearsal.json",
        "development-rehearsed-seal.json",
    }
    entries = tuple(directory.iterdir())
    if {entry.name for entry in entries} != expected:
        raise ValueError("development receipt directory contains missing or extra evidence")
    if any(entry.is_symlink() or not entry.is_file() or _mode(entry) != 0o600 for entry in entries):
        raise ValueError("development receipts must be mode 0600 non-symlink files")
    pilot = _pilot_from_raw(_strict_json(_read_regular_bytes(directory / "development-pilot.json")))
    rehearsal = _rehearsal_from_raw(
        _strict_json(_read_regular_bytes(directory / "development-rehearsal.json"))
    )
    seal = _development_seal_from_raw(
        _strict_json(_read_regular_bytes(directory / "development-rehearsed-seal.json"))
    )
    disjointness = _disjointness_from_raw(
        _strict_json(_read_regular_bytes(directory / "development-disjointness.json"))
    )
    artifacts = DevelopmentRunArtifacts(pilot, rehearsal, seal)
    if (
        seal.pilot_receipt_sha256 != pilot.fingerprint()
        or seal.rehearsal_receipt_sha256 != rehearsal.fingerprint()
        or seal.disjointness_receipt_sha256 != disjointness.fingerprint()
    ):
        raise ValueError("development receipt ancestry changed on disk")
    lifecycle_sha256 = _stopped_lifecycle_sha256(root)
    if any(
        attempt.comfy_lifecycle_receipt_sha256 != lifecycle_sha256
        for attempt in artifacts.all_attempts
    ):
        raise ValueError("development attempts changed their stopped-Comfy lifecycle parent")
    store = DevelopmentAttemptStore(root).validate()
    attempts_directory = checked_development_output(root, Path("development/attempts"))
    if (
        attempts_directory.is_symlink()
        or not attempts_directory.is_dir()
        or _mode(attempts_directory) != 0o700
    ):
        raise ValueError("development attempt directory must be a mode 0700 directory")
    expected_attempt_files = {
        f"attempt-{ordinal}{suffix}"
        for ordinal in range(5)
        for suffix in (".started.json", ".json")
    }
    if {entry.name for entry in attempts_directory.iterdir()} != expected_attempt_files:
        raise ValueError("development attempt directory contains missing or extra evidence")
    media_directory = checked_development_output(root, Path("development/media"))
    if (
        media_directory.is_symlink()
        or not media_directory.is_dir()
        or _mode(media_directory) != 0o700
    ):
        raise ValueError("development media directory must be a mode 0700 directory")
    if {entry.name for entry in media_directory.iterdir()} != {
        f"attempt-{ordinal}.mp4" for ordinal in range(5)
    }:
        raise ValueError("development media directory contains missing or extra evidence")
    for expected_ordinal, terminal in enumerate(artifacts.all_attempts):
        started = store.load_started(f"attempt-{expected_ordinal}")
        loaded_terminal = store.load(f"attempt-{expected_ordinal}")
        if loaded_terminal != terminal:
            raise ValueError("development terminal attempt differs from completion receipts")
        if terminal.started_receipt_sha256 != started.fingerprint():
            raise ValueError("development terminal attempt lost its started parent")
        if (
            started.ordinal != terminal.ordinal
            or started.attempt_id != terminal.attempt_id
            or started.method is not terminal.method
            or started.seed != terminal.seed
            or started.prompt != terminal.prompt
            or started.negative_prompt != terminal.negative_prompt
            or started.ordered_slot_ids != terminal.ordered_slot_ids
            or started.initial_noise_sha256 != terminal.initial_noise_sha256
            or started.method_core_sha256 != terminal.method_core_sha256
            or started.ordered_guide_sha256 != terminal.ordered_guide_sha256
            or started.engine_identity_sha256 != terminal.engine_identity_sha256
            or started.foundation_before_sha256 != terminal.foundation_before_sha256
            or started.runtime_identity_sha256 != terminal.runtime_identity_sha256
            or started.comfy_lifecycle_receipt_sha256 != terminal.comfy_lifecycle_receipt_sha256
        ):
            raise ValueError("development terminal attempt changed its started bindings")
        media_path = media_directory / f"attempt-{expected_ordinal}.mp4"
        if media_path.is_symlink() or not media_path.is_file() or _mode(media_path) != 0o600:
            raise ValueError("development media must be a mode 0600 non-symlink file")
        media_bytes = _read_regular_bytes(media_path, "development media")
        if (
            len(media_bytes) != terminal.media_size_bytes
            or hashlib.sha256(media_bytes).hexdigest() != terminal.media_sha256
        ):
            raise ValueError("development media differs from its terminal attempt")
    return artifacts


def finalize_development_rehearsal(root: Path) -> bytes:
    """Build one byte-stable, disk-only development receipt inventory."""
    artifacts = load_development_rehearsal(root)
    return canonical_json(
        {
            "format": "duet-x-ltx-development-finalization-v1",
            "attempt_lifecycle_sha256": tuple(
                {
                    "started": cast(str, attempt.started_receipt_sha256),
                    "terminal": attempt.fingerprint(),
                    "media": cast(str, attempt.media_sha256),
                }
                for attempt in artifacts.all_attempts
            ),
            "pilot_receipt_sha256": artifacts.pilot.fingerprint(),
            "rehearsal_receipt_sha256": artifacts.rehearsal.fingerprint(),
            "seal_sha256": artifacts.seal.fingerprint(),
            "disjointness_receipt_sha256": artifacts.seal.disjointness_receipt_sha256,
            "comfy_lifecycle_receipt_sha256": tuple(
                attempt.comfy_lifecycle_receipt_sha256 for attempt in artifacts.all_attempts
            ),
        }
    )


_DEVELOPMENT_FORWARD_ORDER = (
    Method.DUET_CORE,
    Method.FULL_HISTORY,
    Method.RECENT_ANCHOR,
    Method.GATED_CORE,
    Method.DUET_CORE,
)


def run_development_rehearsal(
    *,
    store: DevelopmentAttemptStore,
    source_seal: PhaseInputSeal,
    scene_id: str,
    disjointness_receipt: DevelopmentDisjointnessReceipt,
    comfy_lifecycle_receipt_sha256: str,
    identity: DevelopmentExecutionIdentity,
    bind_forward: Callable[[Method, int], DevelopmentForwardBindingLike],
    forward: Callable[[Method, int, Path], DevelopmentForwardResult],
    metrics: DevelopmentRuntimeMetrics,
) -> DevelopmentRunArtifacts:
    """Execute only the five development forwards, persisting every attempt before return."""
    store.validate()
    source_seal.validate()
    identity.validate()
    metrics.validate()
    if source_seal.next_command != "pilot" or scene_id != "development-00":
        raise ValueError("development runner requires the source-authored pilot boundary")
    disjointness_receipt.validate()
    persist_development_disjointness(store.root, disjointness_receipt)
    _sha256(comfy_lifecycle_receipt_sha256, "Comfy lifecycle receipt SHA-256")
    if _stopped_lifecycle_sha256(store.root) != comfy_lifecycle_receipt_sha256:
        raise ValueError("development runner stopped-Comfy lifecycle identity changed")
    attempts: list[DevelopmentAttemptRecord] = []
    media_directory = _prepare_development_directory(store.root, Path("development/media"))
    for ordinal, method in enumerate(_DEVELOPMENT_FORWARD_ORDER):
        output_path = media_directory / f"attempt-{ordinal}.mp4"
        binding = bind_forward(method, 101).validate()
        if binding.method is not method or binding.seed != 101:
            raise ValueError("development forward binding does not match the five-forward order")
        started_attempt = DevelopmentAttemptRecord(
            format="duet-x-ltx-development-attempt-v1",
            attempt_id=f"attempt-{ordinal}",
            ordinal=ordinal,
            method=method,
            seed=101,
            state=DevelopmentAttemptState.STARTED,
            wall_time_seconds=0.0,
            cuda_peak_allocated_bytes=0,
            cuda_peak_reserved_bytes=0,
            final_latent_finite=False,
            initial_noise_sha256=binding.initial_noise_sha256,
            method_core_sha256=binding.method_core_sha256,
            ordered_guide_sha256=binding.ordered_guide_sha256,
            final_latent_sha256=None,
            decoded_frames_sha256=None,
            media_sha256=None,
            media_size_bytes=0,
            media_width=0,
            media_height=0,
            media_frames=0,
            media_fps=0,
            engine_identity_sha256=identity.engine_identity_sha256,
            foundation_before_sha256=identity.foundation_sha256,
            foundation_after_sha256=None,
            runtime_identity_sha256=identity.runtime_identity_sha256,
            comfy_lifecycle_receipt_sha256=comfy_lifecycle_receipt_sha256,
            failure_code=None,
            prompt=binding.prompt,
            negative_prompt=binding.negative_prompt,
            ordered_slot_ids=binding.ordered_slot_ids,
            started_receipt_sha256=None,
            partial_media_sha256=None,
            partial_media_size_bytes=0,
        ).validate()
        store.persist(started_attempt)
        started = time.perf_counter()
        observed_result: DevelopmentForwardResult | None = None
        failure_stage = "initial_metrics"
        try:
            metrics.synchronize()
            metrics.reset_peaks()
            failure_stage = "forward"
            result = forward(method, 101, output_path).validate()
            observed_result = result
            failure_stage = "terminal_metrics"
            metrics.synchronize()
            elapsed = max(time.perf_counter() - started, float.fromhex("0x1p-52"))
            if result.path != output_path:
                raise ValueError("development forward wrote outside its authorized output path")
            _seal_development_media(result.path)
            if (
                result.initial_noise_sha256 != binding.initial_noise_sha256
                or result.method_core_sha256 != binding.method_core_sha256
                or result.ordered_guide_sha256 != binding.ordered_guide_sha256
            ):
                raise ValueError("development forward result changed its started bindings")
            if (
                result.engine_identity_sha256 != identity.engine_identity_sha256
                or result.foundation_before_sha256 != identity.foundation_sha256
                or result.foundation_after_sha256 != identity.foundation_sha256
                or result.runtime_identity_sha256 != identity.runtime_identity_sha256
            ):
                raise ValueError("development forward engine/foundation/runtime identity changed")
            expected_guides = 9 if method is Method.FULL_HISTORY else 4
            if len(result.ordered_guide_sha256) != expected_guides:
                raise ValueError("development forward ordered guide roster changed")
            if method in {Method.GATED_CORE, Method.DUET_CORE}:
                _sha256(result.method_core_sha256, "development method core SHA-256")
            elif result.method_core_sha256 is not None:
                raise ValueError("raw-guide development forward claimed a method core")
            allocated = metrics.peak_allocated_bytes()
            reserved = metrics.peak_reserved_bytes()
            attempt = DevelopmentAttemptRecord(
                "duet-x-ltx-development-attempt-v1",
                f"attempt-{ordinal}",
                ordinal,
                method,
                101,
                DevelopmentAttemptState.SUCCEEDED,
                elapsed,
                allocated,
                reserved,
                True,
                result.initial_noise_sha256,
                result.method_core_sha256,
                result.ordered_guide_sha256,
                tensor_sha256(result.final_latent),
                result.decoded_frames_sha256,
                _file_sha256(result.path),
                result.path.stat().st_size,
                result.media_facts.width,
                result.media_facts.height,
                result.media_facts.frames,
                round(result.media_facts.fps),
                result.engine_identity_sha256,
                result.foundation_before_sha256,
                result.foundation_after_sha256,
                result.runtime_identity_sha256,
                comfy_lifecycle_receipt_sha256,
                None,
                binding.prompt,
                binding.negative_prompt,
                binding.ordered_slot_ids,
                started_attempt.fingerprint(),
                None,
                0,
            ).validate()
            store.persist(attempt)
            attempts.append(attempt)
        except BaseException as error:
            with suppress(Exception):
                metrics.synchronize()
            elapsed = max(time.perf_counter() - started, float.fromhex("0x1p-52"))
            allocated = 0
            reserved = 0
            with suppress(Exception):
                allocated = metrics.peak_allocated_bytes()
            with suppress(Exception):
                reserved = metrics.peak_reserved_bytes()
            partial_sha256: str | None = None
            partial_size_bytes = 0
            if output_path.exists() or output_path.is_symlink():
                if output_path.is_symlink() or not output_path.is_file():
                    raise ValueError("failed development output is unsafe") from error
                partial = checked_development_output(
                    store.root,
                    Path("development/attempts") / f"attempt-{ordinal}.partial.mp4",
                )
                if partial.exists() or partial.is_symlink():
                    raise ValueError("failed development partial output conflicts") from error
                output_path.rename(partial)
                partial.chmod(0o600)
                partial_bytes = _read_regular_bytes(partial, "failed development partial media")
                partial_sha256 = hashlib.sha256(partial_bytes).hexdigest()
                partial_size_bytes = len(partial_bytes)
                _fsync_directory(partial.parent)
                _fsync_directory(output_path.parent)
            failure = DevelopmentAttemptRecord(
                "duet-x-ltx-development-attempt-v1",
                f"attempt-{ordinal}",
                ordinal,
                method,
                101,
                DevelopmentAttemptState.FAILED,
                elapsed,
                allocated,
                reserved,
                False,
                binding.initial_noise_sha256,
                binding.method_core_sha256,
                binding.ordered_guide_sha256,
                None,
                None,
                None,
                0,
                0,
                0,
                0,
                0,
                identity.engine_identity_sha256,
                identity.foundation_sha256,
                observed_result.foundation_after_sha256 if observed_result is not None else None,
                identity.runtime_identity_sha256,
                comfy_lifecycle_receipt_sha256,
                f"{failure_stage}_exception:{type(error).__name__}",
                binding.prompt,
                binding.negative_prompt,
                binding.ordered_slot_ids,
                started_attempt.fingerprint(),
                partial_sha256,
                partial_size_bytes,
            ).validate()
            store.persist(failure)
            raise error
    pilot = DevelopmentPilotReceipt(
        "duet-x-ltx-development-pilot-v1", scene_id, attempts[0]
    ).validate()
    standalone = attempts[0]
    rehearsal = DevelopmentRehearsalReceipt(
        "duet-x-ltx-development-rehearsal-v1",
        scene_id,
        tuple(attempts[1:]),
        cast(str, standalone.final_latent_sha256),
        cast(str, standalone.decoded_frames_sha256),
    ).validate()
    seal = DevelopmentRehearsedSeal.build(
        source_seal=source_seal,
        pilot=pilot,
        rehearsal=rehearsal,
        disjointness_sha256=disjointness_receipt.fingerprint(),
    )
    artifacts = DevelopmentRunArtifacts(pilot, rehearsal, seal)
    persist_development_rehearsal(store.root, artifacts)
    return artifacts


@dataclass(frozen=True, slots=True)
class OperationalInput(FrozenRecord):
    relative_path: str
    sha256: str
    size_bytes: int

    def validate(self) -> Self:
        if (
            type(self.relative_path) is not str
            or not self.relative_path
            or Path(self.relative_path).is_absolute()
            or ".." in Path(self.relative_path).parts
            or "." in Path(self.relative_path).parts
        ):
            raise ValueError("operational input must use a confined relative path")
        _sha256(self.sha256, "operational input SHA-256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("operational input size must be nonnegative")
        return self


_ENVIRONMENT_STAGES = frozenset(
    {"bootstrap", "confirmatory", "disk-authentication", "status", "materialize", "pilot"}
)


@dataclass(frozen=True, slots=True)
class EnvironmentReceipt(FrozenRecord):
    format: str
    stage: str
    allowed_names: tuple[str, ...]
    environment_sha256: str

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-environment-v1" or self.stage not in _ENVIRONMENT_STAGES:
            raise ValueError("environment receipt format or stage changed")
        if (
            type(self.allowed_names) is not tuple
            or not self.allowed_names
            or self.allowed_names != tuple(sorted(self.allowed_names))
            or len(self.allowed_names) != len(set(self.allowed_names))
            or any(type(name) is not str or not name for name in self.allowed_names)
        ):
            raise ValueError("environment receipt allowed-name roster changed")
        _sha256(self.environment_sha256, "environment receipt identity")
        return self

    @classmethod
    def capture(cls, stage: str, environment: Mapping[str, str]) -> EnvironmentReceipt:
        names = ("<none>",) if not environment else tuple(sorted(environment))
        identity = canonical_sha256(
            {
                name: hashlib.sha256(value.encode()).hexdigest()
                for name, value in sorted(environment.items())
            }
        )
        return cls("duet-x-ltx-environment-v1", stage, names, identity).validate()


def _environment_receipt_from_raw(raw: Mapping[str, object]) -> EnvironmentReceipt:
    _exact_keys(
        raw,
        {"format", "stage", "allowed_names", "environment_sha256"},
        "environment receipt",
    )
    names = raw["allowed_names"]
    if type(names) is not list or any(type(name) is not str for name in names):
        raise ValueError("environment receipt allowed names are malformed")
    return EnvironmentReceipt(
        cast(str, raw["format"]),
        cast(str, raw["stage"]),
        tuple(cast(list[str], names)),
        cast(str, raw["environment_sha256"]),
    ).validate()


def load_environment_receipt(path: Path) -> EnvironmentReceipt:
    if path.is_symlink() or _mode(path) != 0o600:
        raise ValueError("environment receipt must be a mode 0600 non-symlink file")
    encoded = _read_regular_bytes(path, "environment receipt")
    receipt = _environment_receipt_from_raw(_strict_json(encoded))
    if receipt.to_json() != encoded:
        raise ValueError("environment receipt must be canonical JSON")
    return receipt


def publish_environment_receipt(
    run_root: Path, stage: str, environment: Mapping[str, str]
) -> EnvironmentReceipt:
    receipt = EnvironmentReceipt.capture(stage, environment)
    root = run_root / "environment"
    if root.exists() or root.is_symlink():
        if root.is_symlink() or not root.is_dir() or _mode(root) != 0o700:
            raise ValueError("environment receipt root is unsafe")
    else:
        root.mkdir(mode=0o700)
        _fsync_directory(run_root)
    path = root / f"{stage}.json"
    if path.exists() or path.is_symlink():
        if load_environment_receipt(path) != receipt:
            raise ValueError("environment receipt identity changed")
    else:
        _write_new_file(path, receipt.to_json())
        _fsync_directory(root)
    return receipt


def _environment_receipts_sha256(run_root: Path) -> str:
    root = run_root / "environment"
    if not root.exists() and not root.is_symlink():
        return "0" * 64
    if root.is_symlink() or not root.is_dir() or _mode(root) != 0o700:
        raise ValueError("environment receipt root is unsafe")
    identities: dict[str, str] = {}
    for path in root.iterdir():
        if path.suffix != ".json" or path.stem not in _ENVIRONMENT_STAGES:
            raise ValueError("environment receipt roster changed")
        receipt = load_environment_receipt(path)
        if receipt.stage != path.stem or receipt.stage in identities:
            raise ValueError("environment receipt stage changed")
        identities[receipt.stage] = _file_sha256(path)
    if not identities:
        raise ValueError("environment receipt roster is empty")
    return canonical_sha256(identities)


@dataclass(frozen=True, slots=True)
class SourceArchiveEntry(FrozenRecord):
    relative_path: str
    git_mode: str
    archive_mode: int
    size_bytes: int
    sha256: str
    git_blob_sha1: str

    def validate(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if (
            type(self.relative_path) is not str
            or not self.relative_path
            or path.is_absolute()
            or path.as_posix() != self.relative_path
            or "." in path.parts
            or ".." in path.parts
            or self.relative_path.startswith("source/")
        ):
            raise ValueError("tracked source path is unsafe")
        if self.git_mode not in {"100644", "100755"}:
            raise ValueError("tracked source mode is not a regular Git file mode")
        expected_archive_mode = 0o755 if self.git_mode == "100755" else 0o644
        if type(self.archive_mode) is not int or self.archive_mode != expected_archive_mode:
            raise ValueError("tracked source archive mode changed")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("tracked source size is malformed")
        _sha256(self.sha256, "tracked source SHA-256")
        if (
            type(self.git_blob_sha1) is not str
            or len(self.git_blob_sha1) != 40
            or any(character not in _SHA256_HEX for character in self.git_blob_sha1)
        ):
            raise ValueError("tracked source Git blob identity is malformed")
        return self


@dataclass(frozen=True, slots=True)
class SourceArchiveReceipt(FrozenRecord):
    format: str
    source_commit: str
    source_tree_sha1: str
    source_archive_sha256: str
    bootstrap_sha256: str
    bootstrap_size_bytes: int
    entries: tuple[SourceArchiveEntry, ...]

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-source-archive-v1":
            raise ValueError("source archive receipt format changed")
        for value, name in (
            (self.source_commit, "source commit"),
            (self.source_tree_sha1, "source tree"),
        ):
            if (
                type(value) is not str
                or len(value) != 40
                or any(character not in _SHA256_HEX for character in value)
            ):
                raise ValueError(f"{name} Git identity is malformed")
        _sha256(self.source_archive_sha256, "source archive SHA-256")
        _sha256(self.bootstrap_sha256, "source bootstrap SHA-256")
        if type(self.bootstrap_size_bytes) is not int or self.bootstrap_size_bytes <= 0:
            raise ValueError("source bootstrap size is malformed")
        if type(self.entries) is not tuple or not self.entries:
            raise ValueError("source archive receipt requires tracked entries")
        for entry in self.entries:
            entry.validate()
        paths = tuple(entry.relative_path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("source archive entries must be unique and sorted")
        return self


@dataclass(frozen=True, slots=True)
class SourceExtractionReceipt(FrozenRecord):
    format: str
    source_receipt_sha256: str
    source_commit: str
    source_tree_sha1: str
    source_archive_sha256: str
    extracted_root: str
    environment_receipt_sha256: str

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-source-extraction-v1":
            raise ValueError("source extraction receipt format changed")
        _sha256(self.source_receipt_sha256, "source receipt SHA-256")
        SourceArchiveReceipt(
            "duet-x-ltx-source-archive-v1",
            self.source_commit,
            self.source_tree_sha1,
            self.source_archive_sha256,
            "0" * 64,
            1,
            (SourceArchiveEntry("placeholder", "100644", 0o644, 0, "0" * 64, "0" * 40),),
        ).validate()
        if self.extracted_root != "source-extraction/tree":
            raise ValueError("source extraction root changed")
        _sha256(self.environment_receipt_sha256, "source extraction environment receipt")
        return self


def _git_object_sha1(kind: str, encoded: bytes) -> str:
    header = f"{kind} {len(encoded)}\0".encode()
    return hashlib.sha1(header + encoded, usedforsecurity=False).hexdigest()


def _source_tree_sha1(entries: Sequence[SourceArchiveEntry]) -> str:
    by_path = {PurePosixPath(entry.relative_path): entry for entry in entries}

    def tree(prefix: PurePosixPath | None) -> str:
        children: dict[str, SourceArchiveEntry | None] = {}
        prefix_parts = () if prefix is None else prefix.parts
        for path, source_entry in by_path.items():
            if path.parts[: len(prefix_parts)] != prefix_parts:
                continue
            remainder = path.parts[len(prefix_parts) :]
            if not remainder:
                continue
            child = remainder[0]
            children[child] = source_entry if len(remainder) == 1 else None
        payload = bytearray()
        ordered = sorted(
            children.items(),
            key=lambda item: (item[0] + ("/" if item[1] is None else "")).encode(),
        )
        for name, child_entry in ordered:
            encoded_name = name.encode("utf-8")
            if child_entry is None:
                child_prefix = PurePosixPath(name) if prefix is None else prefix / name
                mode = "40000"
                object_sha1 = tree(child_prefix)
            else:
                mode = child_entry.git_mode
                object_sha1 = child_entry.git_blob_sha1
            payload.extend(f"{mode} ".encode() + encoded_name + b"\0")
            payload.extend(bytes.fromhex(object_sha1))
        return _git_object_sha1("tree", bytes(payload))

    return tree(None)


def _source_archive_receipt_from_raw(raw: Mapping[str, object]) -> SourceArchiveReceipt:
    _exact_keys(
        raw,
        {
            "format",
            "source_commit",
            "source_tree_sha1",
            "source_archive_sha256",
            "bootstrap_sha256",
            "bootstrap_size_bytes",
            "entries",
        },
        "source archive receipt",
    )
    entries_raw = raw["entries"]
    if type(entries_raw) is not list:
        raise ValueError("source archive entries must be an array")
    entries: list[SourceArchiveEntry] = []
    for value in entries_raw:
        if type(value) is not dict:
            raise ValueError("source archive entry is malformed")
        row = cast(dict[str, object], value)
        _exact_keys(
            row,
            {
                "relative_path",
                "git_mode",
                "archive_mode",
                "size_bytes",
                "sha256",
                "git_blob_sha1",
            },
            "source archive entry",
        )
        entries.append(
            SourceArchiveEntry(
                cast(str, row["relative_path"]),
                cast(str, row["git_mode"]),
                cast(int, row["archive_mode"]),
                cast(int, row["size_bytes"]),
                cast(str, row["sha256"]),
                cast(str, row["git_blob_sha1"]),
            ).validate()
        )
    return SourceArchiveReceipt(
        cast(str, raw["format"]),
        cast(str, raw["source_commit"]),
        cast(str, raw["source_tree_sha1"]),
        cast(str, raw["source_archive_sha256"]),
        cast(str, raw["bootstrap_sha256"]),
        cast(int, raw["bootstrap_size_bytes"]),
        tuple(entries),
    ).validate()


def load_source_archive_receipt(path: Path) -> SourceArchiveReceipt:
    if path.is_symlink() or not path.is_file() or _mode(path) != 0o600:
        raise ValueError("source archive receipt must be a mode 0600 regular file")
    encoded = _read_regular_bytes(path, "source archive receipt")
    receipt = _source_archive_receipt_from_raw(_strict_json(encoded))
    if receipt.to_json() != encoded:
        raise ValueError("source archive receipt must be canonical JSON")
    return receipt


def _checked_local_output(path: Path, repository: Path) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("source archive outputs must be new absolute paths")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise ValueError("source archive output parent must be a canonical real directory")
    try:
        path.relative_to(repository)
    except ValueError:
        return
    raise ValueError("source archive outputs must be outside the reviewed repository")


def _run_git(repository: Path, arguments: Sequence[str]) -> bytes:
    try:
        return subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("reviewed source Git query failed") from error


def create_source_archive(
    repository: Path,
    archive_output: Path,
    receipt_output: Path,
    bootstrap_output: Path,
    *,
    expected_source_commit: str,
) -> SourceArchiveReceipt:
    """Create a deterministic archive and receipt from one clean reviewed Git commit."""
    if (
        not repository.is_absolute()
        or repository.is_symlink()
        or not repository.is_dir()
        or repository.resolve(strict=True) != repository
    ):
        raise ValueError("reviewed source repository must be a canonical real directory")
    _checked_local_output(archive_output, repository)
    _checked_local_output(receipt_output, repository)
    _checked_local_output(bootstrap_output, repository)
    if len({archive_output, receipt_output, bootstrap_output}) != 3:
        raise ValueError("source archive, receipt, and bootstrap outputs must differ")
    status = _run_git(repository, ("status", "--porcelain=v1", "--untracked-files=all"))
    if status:
        raise ValueError("reviewed source checkout must be clean")
    observed_commit = _run_git(repository, ("rev-parse", "HEAD")).decode().strip()
    if observed_commit != expected_source_commit:
        raise ValueError("reviewed source commit changed")
    observed_tree = _run_git(repository, ("rev-parse", "HEAD^{tree}")).decode().strip()
    listed = _run_git(repository, ("ls-tree", "-r", "-z", "-l", "--full-tree", "HEAD"))
    entries: list[SourceArchiveEntry] = []
    contents: dict[str, bytes] = {}
    for record in listed.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode, kind, blob, encoded_size = metadata.split()
        if kind != b"blob":
            raise ValueError("reviewed source contains a non-blob tracked entry")
        try:
            relative_path = encoded_path.decode("utf-8")
            size = int(encoded_size)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("reviewed source inventory is malformed") from error
        content = _run_git(repository, ("show", f"HEAD:{relative_path}"))
        if len(content) != size or _git_object_sha1("blob", content) != blob.decode():
            raise ValueError("reviewed source blob changed during archive creation")
        git_mode = mode.decode()
        contents[relative_path] = content
        entries.append(
            SourceArchiveEntry(
                relative_path,
                git_mode,
                0o755 if git_mode == "100755" else 0o644,
                size,
                hashlib.sha256(content).hexdigest(),
                blob.decode(),
            ).validate()
        )
    entries.sort(key=lambda entry: entry.relative_path)
    if _source_tree_sha1(entries) != observed_tree:
        raise ValueError("reviewed source tree could not be reconstructed")
    archive = _run_git(
        repository,
        (
            "-c",
            "tar.umask=0022",
            "archive",
            "--format=tar",
            "--prefix=source/",
            observed_commit,
        ),
    )
    bootstrap_relative = "src/comfy_story/ltx_quality_bootstrap.py"
    bootstrap = contents.get(bootstrap_relative)
    bootstrap_entries = tuple(
        entry for entry in entries if entry.relative_path == bootstrap_relative
    )
    if (
        bootstrap is None
        or len(bootstrap_entries) != 1
        or bootstrap_entries[0].git_mode != "100644"
    ):
        raise ValueError("reviewed source must contain the regular standalone quality bootstrap")
    receipt = SourceArchiveReceipt(
        "duet-x-ltx-source-archive-v1",
        observed_commit,
        observed_tree,
        hashlib.sha256(archive).hexdigest(),
        hashlib.sha256(bootstrap).hexdigest(),
        len(bootstrap),
        tuple(entries),
    ).validate()
    _write_new_file(archive_output, archive)
    _write_new_file(receipt_output, receipt.to_json())
    _write_new_file(bootstrap_output, bootstrap)
    _fsync_directory(archive_output.parent)
    if receipt_output.parent != archive_output.parent:
        _fsync_directory(receipt_output.parent)
    if bootstrap_output.parent not in {archive_output.parent, receipt_output.parent}:
        _fsync_directory(bootstrap_output.parent)
    return receipt


@dataclass(frozen=True, slots=True)
class SourcePopulationArchiveEntry(FrozenRecord):
    relative_path: str
    size_bytes: int
    sha256: str

    def validate(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if (
            type(self.relative_path) is not str
            or not self.relative_path
            or path.is_absolute()
            or path.as_posix() != self.relative_path
            or "." in path.parts
            or ".." in path.parts
        ):
            raise ValueError("source-population archive path is unsafe")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("source-population archive size is malformed")
        _sha256(self.sha256, "source-population archive entry SHA-256")
        return self


@dataclass(frozen=True, slots=True)
class SourcePopulationArchiveReceipt(FrozenRecord):
    format: str
    source_population_archive_sha256: str
    population_fingerprints_sha256: str
    entries: tuple[SourcePopulationArchiveEntry, ...]

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-source-population-archive-v1":
            raise ValueError("source-population archive receipt format changed")
        _sha256(self.source_population_archive_sha256, "source-population archive SHA-256")
        _sha256(self.population_fingerprints_sha256, "source-population fingerprints")
        if type(self.entries) is not tuple or not self.entries:
            raise ValueError("source-population archive receipt requires entries")
        for entry in self.entries:
            entry.validate()
        paths = tuple(entry.relative_path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("source-population archive entries must be unique and sorted")
        return self


def _source_population_receipt_from_raw(
    raw: Mapping[str, object],
) -> SourcePopulationArchiveReceipt:
    _exact_keys(
        raw,
        {
            "format",
            "source_population_archive_sha256",
            "population_fingerprints_sha256",
            "entries",
        },
        "source-population archive receipt",
    )
    rows = raw["entries"]
    if type(rows) is not list:
        raise ValueError("source-population archive entries must be an array")
    entries: list[SourcePopulationArchiveEntry] = []
    for value in rows:
        if type(value) is not dict:
            raise ValueError("source-population archive entry is malformed")
        row = cast(dict[str, object], value)
        _exact_keys(row, {"relative_path", "size_bytes", "sha256"}, "source-population entry")
        entries.append(
            SourcePopulationArchiveEntry(
                cast(str, row["relative_path"]),
                cast(int, row["size_bytes"]),
                cast(str, row["sha256"]),
            ).validate()
        )
    return SourcePopulationArchiveReceipt(
        cast(str, raw["format"]),
        cast(str, raw["source_population_archive_sha256"]),
        cast(str, raw["population_fingerprints_sha256"]),
        tuple(entries),
    ).validate()


def load_source_population_archive_receipt(path: Path) -> SourcePopulationArchiveReceipt:
    if path.is_symlink() or not path.is_file() or _mode(path) != 0o600:
        raise ValueError("source-population archive receipt must be a mode 0600 regular file")
    encoded = _read_regular_bytes(path, "source-population archive receipt")
    receipt = _source_population_receipt_from_raw(_strict_json(encoded))
    if receipt.to_json() != encoded:
        raise ValueError("source-population archive receipt must be canonical JSON")
    return receipt


def _population_tar_info(name: str, *, directory: bool, size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.mode = 0o755 if directory else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    info.size = size
    return info


def create_source_population_archive(
    source_root: Path, archive_output: Path, receipt_output: Path
) -> SourcePopulationArchiveReceipt:
    """Validate and deterministically bundle the accepted source population."""
    if (
        not source_root.is_absolute()
        or source_root.is_symlink()
        or not source_root.is_dir()
        or source_root.resolve(strict=True) != source_root
        or _mode(source_root) != 0o700
    ):
        raise ValueError("source-population creator requires a canonical mode 0700 root")
    _checked_local_output(archive_output, source_root)
    _checked_local_output(receipt_output, source_root)
    if archive_output == receipt_output:
        raise ValueError("source-population archive and receipt outputs must differ")
    _, fingerprints = load_accepted_source_population(source_root)
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    for path in source_root.rglob("*"):
        relative = path.relative_to(source_root).as_posix()
        if path.is_symlink():
            raise ValueError("source-population creator rejects symlinks")
        if path.is_dir():
            if _mode(path) != 0o700:
                raise ValueError("source-population creator requires mode 0700 directories")
            directories.add(relative)
        else:
            if _mode(path) != 0o600:
                raise ValueError("source-population creator requires mode 0600 files")
            files[relative] = _read_regular_bytes(path, "source-population creator input")
    entries = tuple(
        SourcePopulationArchiveEntry(
            name, len(encoded), hashlib.sha256(encoded).hexdigest()
        ).validate()
        for name, encoded in sorted(files.items())
    )
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:", format=tarfile.USTAR_FORMAT) as bundle:
        bundle.addfile(_population_tar_info("source-population", directory=True))
        for directory in sorted(directories):
            bundle.addfile(_population_tar_info(f"source-population/{directory}", directory=True))
        for entry in entries:
            encoded = files[entry.relative_path]
            bundle.addfile(
                _population_tar_info(
                    f"source-population/{entry.relative_path}",
                    directory=False,
                    size=len(encoded),
                ),
                io.BytesIO(encoded),
            )
    archive = stream.getvalue()
    receipt = SourcePopulationArchiveReceipt(
        "duet-x-ltx-source-population-archive-v1",
        hashlib.sha256(archive).hexdigest(),
        fingerprints.fingerprint(),
        entries,
    ).validate()
    _write_new_file(archive_output, archive)
    _write_new_file(receipt_output, receipt.to_json())
    _fsync_directory(archive_output.parent)
    if receipt_output.parent != archive_output.parent:
        _fsync_directory(receipt_output.parent)
    return receipt


def _population_archive_members(
    archive: bytes, receipt: SourcePopulationArchiveReceipt
) -> dict[str, bytes]:
    expected = {f"source-population/{entry.relative_path}": entry for entry in receipt.entries}
    expected_directories = {"source-population"}
    for name in expected:
        path = PurePosixPath(name)
        expected_directories.update(
            parent.as_posix() for parent in path.parents if parent != PurePosixPath(".")
        )
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    try:
        bundle = tarfile.open(fileobj=io.BytesIO(archive), mode="r:")  # noqa: SIM115
    except tarfile.TarError as error:
        raise ValueError("source-population archive is not an uncompressed tar") from error
    with bundle:
        members = bundle.getmembers()
        if len({member.name for member in members}) != len(members):
            raise ValueError("source-population archive contains duplicate members")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or path.as_posix() != member.name
                or "." in path.parts
                or ".." in path.parts
                or member.linkname
                or member.uid != 0
                or member.gid != 0
                or member.uname != "root"
                or member.gname != "root"
                or member.mtime != 0
                or member.pax_headers
            ):
                raise ValueError("source-population archive contains unsafe metadata")
            if member.isdir():
                if member.name not in expected_directories or member.mode != 0o755 or member.size:
                    raise ValueError("source-population archive directory roster changed")
                directories.add(member.name)
                continue
            expected_entry = expected.get(member.name)
            if not member.isreg() or expected_entry is None or member.mode != 0o644:
                raise ValueError("source-population archive contains an unsafe or extra member")
            handle = bundle.extractfile(member)
            if handle is None:
                raise ValueError("source-population archive member is unreadable")
            encoded = handle.read()
            if (
                len(encoded) != expected_entry.size_bytes
                or hashlib.sha256(encoded).hexdigest() != expected_entry.sha256
            ):
                raise ValueError("source-population archive member identity changed")
            files[member.name] = encoded
    if set(files) != set(expected) or directories != expected_directories:
        raise ValueError("source-population archive member roster changed")
    return files


def _verify_extracted_source_population(
    destination: Path, receipt: SourcePopulationArchiveReceipt
) -> None:
    expected = {Path(entry.relative_path): entry for entry in receipt.entries}
    expected_directories = {
        parent for relative in expected for parent in relative.parents if parent != Path()
    }
    files: set[Path] = set()
    directories: set[Path] = set()
    if destination.is_symlink() or not destination.is_dir() or _mode(destination) != 0o700:
        raise ValueError("source-population extraction root is unsafe")
    for path in destination.rglob("*"):
        relative = path.relative_to(destination)
        if path.is_symlink():
            raise ValueError("source-population extraction contains a symlink")
        if path.is_dir():
            if _mode(path) != 0o700:
                raise ValueError("source-population extraction directory mode changed")
            directories.add(relative)
            continue
        expected_entry = expected.get(relative)
        if expected_entry is None or _mode(path) != 0o600:
            raise ValueError("source-population extraction contains an extra file or mode change")
        encoded = _read_regular_bytes(path, "source-population extraction")
        if (
            len(encoded) != expected_entry.size_bytes
            or hashlib.sha256(encoded).hexdigest() != expected_entry.sha256
        ):
            raise ValueError("source-population extraction identity changed")
        files.add(relative)
    if files != set(expected) or directories != expected_directories:
        raise ValueError("source-population extraction roster changed")


def extract_authenticated_source_population_archive(
    archive_path: Path, receipt_path: Path, destination: Path
) -> SourcePopulationArchiveReceipt:
    if (
        not destination.is_absolute()
        or destination.name != "source-population-preseal"
        or destination.parent.name != "incoming"
    ):
        raise ValueError("source-population extraction destination changed")
    if _mode(archive_path) != 0o600:
        raise ValueError("source-population archive must be mode 0600")
    archive = _read_regular_bytes(archive_path, "source-population archive")
    receipt = load_source_population_archive_receipt(receipt_path)
    if hashlib.sha256(archive).hexdigest() != receipt.source_population_archive_sha256:
        raise ValueError("source-population archive identity changed")
    members = _population_archive_members(archive, receipt)
    if destination.exists() or destination.is_symlink():
        _verify_extracted_source_population(destination, receipt)
        return receipt
    with tempfile.TemporaryDirectory(
        prefix=".source-population-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary.chmod(0o700)
        for entry in receipt.entries:
            target = temporary / entry.relative_path
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            current = target.parent
            while current != temporary:
                current.chmod(0o700)
                current = current.parent
            _write_new_file(target, members[f"source-population/{entry.relative_path}"])
        temporary.rename(destination)
        _fsync_directory(destination.parent)
    _verify_extracted_source_population(destination, receipt)
    return receipt


def _source_extraction_receipt_from_raw(raw: Mapping[str, object]) -> SourceExtractionReceipt:
    _exact_keys(
        raw,
        {
            "format",
            "source_receipt_sha256",
            "source_commit",
            "source_tree_sha1",
            "source_archive_sha256",
            "extracted_root",
            "environment_receipt_sha256",
        },
        "source extraction receipt",
    )
    return SourceExtractionReceipt(
        cast(str, raw["format"]),
        cast(str, raw["source_receipt_sha256"]),
        cast(str, raw["source_commit"]),
        cast(str, raw["source_tree_sha1"]),
        cast(str, raw["source_archive_sha256"]),
        cast(str, raw["extracted_root"]),
        cast(str, raw["environment_receipt_sha256"]),
    ).validate()


def _expected_source_directories(entries: Sequence[SourceArchiveEntry]) -> set[str]:
    directories = {"source"}
    for entry in entries:
        path = PurePosixPath("source") / entry.relative_path
        directories.update(
            parent.as_posix() for parent in path.parents if parent != PurePosixPath(".")
        )
    return directories


def _safe_source_members(archive: bytes, receipt: SourceArchiveReceipt) -> dict[str, bytes]:
    expected_files = {f"source/{entry.relative_path}": entry for entry in receipt.entries}
    expected_directories = _expected_source_directories(receipt.entries)
    observed_files: dict[str, bytes] = {}
    observed_directories: set[str] = set()
    try:
        # Opening is separated from the closing context only to normalize malformed-tar errors.
        bundle = tarfile.open(fileobj=io.BytesIO(archive), mode="r:")  # noqa: SIM115
    except tarfile.TarError as error:
        raise ValueError("source archive is not a valid uncompressed tar") from error
    with bundle:
        members = bundle.getmembers()
        names = tuple(member.name for member in members)
        if len(names) != len(set(names)):
            raise ValueError("source archive contains duplicate members")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not member.name
                or path.is_absolute()
                or path.as_posix() != member.name
                or "." in path.parts
                or ".." in path.parts
                or "\\" in member.name
                or member.linkname
            ):
                raise ValueError("source archive contains an unsafe path or link member")
            if (
                member.uid != 0
                or member.gid != 0
                or member.uname != "root"
                or member.gname != "root"
                or member.pax_headers != {"comment": receipt.source_commit}
            ):
                raise ValueError("source archive member ownership or commit metadata changed")
            if member.isdir():
                if member.name not in expected_directories or member.mode != 0o755 or member.size:
                    raise ValueError("source archive directory roster or mode changed")
                observed_directories.add(member.name)
                continue
            expected = expected_files.get(member.name)
            if not member.isreg() or expected is None or member.mode != expected.archive_mode:
                raise ValueError("source archive contains an unsafe or extra file member")
            handle = bundle.extractfile(member)
            if handle is None:
                raise ValueError("source archive regular member cannot be read")
            encoded = handle.read()
            if (
                len(encoded) != expected.size_bytes
                or hashlib.sha256(encoded).hexdigest() != expected.sha256
                or _git_object_sha1("blob", encoded) != expected.git_blob_sha1
            ):
                raise ValueError("source archive tracked member identity changed")
            observed_files[member.name] = encoded
    if set(observed_files) != set(expected_files) or observed_directories != expected_directories:
        raise ValueError("source archive member roster changed")
    observed_entries = tuple(
        SourceArchiveEntry(
            entry.relative_path,
            entry.git_mode,
            entry.archive_mode,
            entry.size_bytes,
            hashlib.sha256(observed_files[f"source/{entry.relative_path}"]).hexdigest(),
            _git_object_sha1("blob", observed_files[f"source/{entry.relative_path}"]),
        ).validate()
        for entry in receipt.entries
    )
    if _source_tree_sha1(observed_entries) != receipt.source_tree_sha1:
        raise ValueError("source archive tree identity changed")
    return observed_files


def _verify_extracted_source(
    extraction_root: Path, receipt: SourceArchiveReceipt, receipt_bytes: bytes
) -> SourceExtractionReceipt:
    if (
        extraction_root.is_symlink()
        or not extraction_root.is_dir()
        or _mode(extraction_root) != 0o700
        or {entry.name for entry in extraction_root.iterdir()}
        != {"tree", "receipt.json", "environment.json"}
    ):
        raise ValueError("source extraction namespace is unsafe")
    extraction_receipt_path = extraction_root / "receipt.json"
    if extraction_receipt_path.is_symlink() or _mode(extraction_receipt_path) != 0o600:
        raise ValueError("source extraction receipt is unsafe")
    extraction_encoded = _read_regular_bytes(extraction_receipt_path, "source extraction receipt")
    extraction = _source_extraction_receipt_from_raw(_strict_json(extraction_encoded))
    if extraction.to_json() != extraction_encoded:
        raise ValueError("source extraction receipt must be canonical JSON")
    environment_path = extraction_root / "environment.json"
    environment = load_environment_receipt(environment_path)
    if environment.stage not in {"bootstrap", "disk-authentication"}:
        raise ValueError("source extraction environment stage changed")
    expected_extraction = SourceExtractionReceipt(
        "duet-x-ltx-source-extraction-v1",
        hashlib.sha256(receipt_bytes).hexdigest(),
        receipt.source_commit,
        receipt.source_tree_sha1,
        receipt.source_archive_sha256,
        "source-extraction/tree",
        _file_sha256(environment_path),
    ).validate()
    if extraction != expected_extraction:
        raise ValueError("source extraction receipt identity changed")
    tree_root = extraction_root / "tree"
    if tree_root.is_symlink() or not tree_root.is_dir() or _mode(tree_root) != 0o700:
        raise ValueError("source extraction tree is unsafe")
    expected_files = {Path(entry.relative_path): entry for entry in receipt.entries}
    expected_directories = {
        parent for path in expected_files for parent in path.parents if parent != Path()
    }
    observed_files: set[Path] = set()
    observed_directories: set[Path] = set()
    observed_entries: list[SourceArchiveEntry] = []
    for path in tree_root.rglob("*"):
        relative = path.relative_to(tree_root)
        if path.is_symlink():
            raise ValueError("source extraction contains a symlink")
        if path.is_dir():
            if _mode(path) != 0o700:
                raise ValueError("source extraction directory mode changed")
            observed_directories.add(relative)
            continue
        expected = expected_files.get(relative)
        if expected is None or _mode(path) != (0o700 if expected.git_mode == "100755" else 0o600):
            raise ValueError("source extraction contains an extra file or mode change")
        encoded = _read_regular_bytes(path, "tracked source extraction")
        observed_files.add(relative)
        observed_entries.append(
            SourceArchiveEntry(
                expected.relative_path,
                expected.git_mode,
                expected.archive_mode,
                len(encoded),
                hashlib.sha256(encoded).hexdigest(),
                _git_object_sha1("blob", encoded),
            ).validate()
        )
    if observed_files != set(expected_files) or observed_directories != expected_directories:
        raise ValueError("source extraction roster changed")
    observed_entries.sort(key=lambda entry: entry.relative_path)
    if tuple(observed_entries) != receipt.entries or _source_tree_sha1(observed_entries) != (
        receipt.source_tree_sha1
    ):
        raise ValueError("tracked source extraction identity changed")
    return extraction


def extract_authenticated_source_archive(
    archive_path: Path, receipt_path: Path, extraction_root: Path
) -> SourceExtractionReceipt:
    """Strictly authenticate and extract one source tar into a fixed fresh root."""
    if not extraction_root.is_absolute() or extraction_root.name != "source-extraction":
        raise ValueError("source extraction must use the exact source-extraction root")
    if extraction_root.parent.is_symlink() or not extraction_root.parent.is_dir():
        raise ValueError("source extraction parent is unsafe")
    if _mode(archive_path) != 0o600:
        raise ValueError("source archive must be mode 0600")
    archive = _read_regular_bytes(archive_path, "source archive")
    receipt_bytes = _read_regular_bytes(receipt_path, "source archive receipt")
    receipt = load_source_archive_receipt(receipt_path)
    if hashlib.sha256(archive).hexdigest() != receipt.source_archive_sha256:
        raise ValueError("source archive identity changed")
    members = _safe_source_members(archive, receipt)
    if extraction_root.exists() or extraction_root.is_symlink():
        return _verify_extracted_source(extraction_root, receipt, receipt_bytes)
    with tempfile.TemporaryDirectory(
        prefix=".source-extraction-", dir=extraction_root.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary.chmod(0o700)
        tree_root = temporary / "tree"
        tree_root.mkdir(mode=0o700)
        for entry in receipt.entries:
            destination = tree_root / entry.relative_path
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            current = destination.parent
            while current != tree_root:
                current.chmod(0o700)
                current = current.parent
            _write_new_file(destination, members[f"source/{entry.relative_path}"])
            if entry.git_mode == "100755":
                destination.chmod(0o700)
        extraction = SourceExtractionReceipt(
            "duet-x-ltx-source-extraction-v1",
            hashlib.sha256(receipt_bytes).hexdigest(),
            receipt.source_commit,
            receipt.source_tree_sha1,
            receipt.source_archive_sha256,
            "source-extraction/tree",
            "0" * 64,
        ).validate()
        environment = EnvironmentReceipt.capture("disk-authentication", {})
        environment_path = temporary / "environment.json"
        _write_new_file(environment_path, environment.to_json())
        extraction = SourceExtractionReceipt(
            extraction.format,
            extraction.source_receipt_sha256,
            extraction.source_commit,
            extraction.source_tree_sha1,
            extraction.source_archive_sha256,
            extraction.extracted_root,
            _file_sha256(environment_path),
        ).validate()
        _write_new_file(temporary / "receipt.json", extraction.to_json())
        _fsync_directory(tree_root)
        _fsync_directory(temporary)
        temporary.rename(extraction_root)
        _fsync_directory(extraction_root.parent)
    return _verify_extracted_source(extraction_root, receipt, receipt_bytes)


@dataclass(frozen=True, slots=True)
class OperationalConfiguration(FrozenRecord):
    format: str
    source_commit: str
    source_archive_sha256: str
    phase_seal_sha256: str
    inputs: tuple[OperationalInput, ...]

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-operations-v1":
            raise ValueError("operational configuration format changed")
        if (
            type(self.source_commit) is not str
            or len(self.source_commit) != 40
            or any(character not in _SHA256_HEX for character in self.source_commit)
        ):
            raise ValueError("expected source commit identity is malformed")
        _sha256(self.source_archive_sha256, "source archive SHA-256")
        _sha256(self.phase_seal_sha256, "phase seal SHA-256")
        if type(self.inputs) is not tuple or not self.inputs:
            raise ValueError("operational configuration requires immutable inputs")
        for item in self.inputs:
            item.validate()
        paths = tuple(item.relative_path for item in self.inputs)
        if len(set(paths)) != len(paths):
            raise ValueError("operational input paths must be unique")
        return self


def _configuration_from_raw(raw: Mapping[str, object]) -> OperationalConfiguration:
    _exact_keys(
        raw,
        {"format", "source_commit", "source_archive_sha256", "phase_seal_sha256", "inputs"},
        "operational configuration",
    )
    inputs_raw = raw["inputs"]
    if type(inputs_raw) is not list:
        raise ValueError("operational configuration inputs must be an array")
    inputs = []
    for value in inputs_raw:
        if type(value) is not dict:
            raise ValueError("operational input record is malformed")
        row = cast(dict[str, object], value)
        _exact_keys(row, {"relative_path", "sha256", "size_bytes"}, "operational input")
        inputs.append(
            OperationalInput(
                cast(str, row["relative_path"]),
                cast(str, row["sha256"]),
                cast(int, row["size_bytes"]),
            ).validate()
        )
    return OperationalConfiguration(
        cast(str, raw["format"]),
        cast(str, raw["source_commit"]),
        cast(str, raw["source_archive_sha256"]),
        cast(str, raw["phase_seal_sha256"]),
        tuple(inputs),
    ).validate()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)


def _load_operational_configuration(
    run_root: Path, configuration_name: str
) -> OperationalConfiguration:
    if (
        not run_root.is_absolute()
        or run_root.is_symlink()
        or not run_root.is_dir()
        or run_root.resolve(strict=True) != run_root
        or _mode(run_root) != 0o700
    ):
        raise ValueError("operational run root must be real, absolute, and mode 0700")
    config_path = run_root / configuration_name
    if config_path.is_symlink() or _mode(config_path) != 0o600:
        raise ValueError("operational configuration must be a mode 0600 non-symlink file")
    configuration = _configuration_from_raw(
        _strict_json(_read_regular_bytes(config_path, "operational configuration"))
    )
    return configuration


def prepare_operational_run(
    run_root: Path,
    *,
    expected_source_commit: str,
    expected_source_archive_sha256: str,
) -> OperationalConfiguration:
    """Build the only canonical seal/configuration after standalone source bootstrap."""
    if (
        not run_root.is_absolute()
        or run_root.is_symlink()
        or not run_root.is_dir()
        or run_root.resolve(strict=True) != run_root
        or _mode(run_root) != 0o700
    ):
        raise ValueError("operational preparation requires a canonical mode 0700 run root")
    if (run_root / "operational-config.json").exists():
        raise ValueError("operational preparation requires a fresh configuration")
    incoming = run_root / "incoming"
    expected_files = {
        "source.tar",
        "source/source-receipt.json",
        "source/ltx-quality-bootstrap.py",
        "source-population-preseal.tar",
        "source-population/source-population-receipt.json",
        "development/development-00.json",
        "development/development-00.pt",
        "development/development-00-k2.json",
        "training/trainable-checkpoint.pt",
    }
    expected_directories = {
        "source",
        "source-population",
        "development",
        "training",
    }
    if incoming.is_symlink() or not incoming.is_dir() or _mode(incoming) != 0o700:
        raise ValueError("operational incoming root is missing or unsafe")
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in incoming.rglob("*"):
        relative = path.relative_to(incoming).as_posix()
        if path.is_symlink():
            raise ValueError("operational preparation input contains a symlink")
        if path.is_dir():
            if _mode(path) != 0o700:
                raise ValueError("operational preparation directories must be mode 0700")
            observed_directories.add(relative)
        else:
            if _mode(path) != 0o600:
                raise ValueError("operational preparation files must be mode 0600")
            _read_regular_bytes(path, "operational preparation input")
            observed_files.add(relative)
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("operational preparation input roster changed")
    source_archive = incoming / "source.tar"
    source_receipt_path = incoming / "source/source-receipt.json"
    source_receipt = load_source_archive_receipt(source_receipt_path)
    bootstrap = _read_regular_bytes(
        incoming / "source/ltx-quality-bootstrap.py", "source bootstrap"
    )
    if (
        source_receipt.source_commit != expected_source_commit
        or source_receipt.source_archive_sha256 != expected_source_archive_sha256
        or hashlib.sha256(_read_regular_bytes(source_archive, "source archive")).hexdigest()
        != expected_source_archive_sha256
        or len(bootstrap) != source_receipt.bootstrap_size_bytes
        or hashlib.sha256(bootstrap).hexdigest() != source_receipt.bootstrap_sha256
    ):
        raise ValueError("operational source/bootstrap identity mismatch")
    extract_authenticated_source_archive(
        source_archive,
        source_receipt_path,
        run_root / "source-extraction",
    )
    population_archive = incoming / "source-population-preseal.tar"
    population_receipt_path = incoming / "source-population/source-population-receipt.json"
    population_receipt = load_source_population_archive_receipt(population_receipt_path)
    extracted_population = incoming / "source-population-preseal"
    extracted_receipt = extract_authenticated_source_population_archive(
        population_archive,
        population_receipt_path,
        extracted_population,
    )
    _, population_fingerprints = load_accepted_source_population(extracted_population)
    if (
        extracted_receipt.population_fingerprints_sha256
        != population_receipt.population_fingerprints_sha256
        or population_fingerprints.fingerprint()
        != population_receipt.population_fingerprints_sha256
    ):
        raise ValueError("source-population receipt or reconstructed fingerprints changed")
    development_root = incoming / "development"
    load_development_input(
        development_root / "development-00.json",
        development_root / "development-00.pt",
        development_root / "development-00-k2.json",
    )
    from comfy_story.ltx_quality_runtime import (
        FrozenQualityRuntimeIdentity,
        capture_gemma_inventory,
    )

    identity = FrozenQualityRuntimeIdentity.default(
        source_commit=expected_source_commit,
        source_archive_sha256=expected_source_archive_sha256,
    ).validate()
    trainable = _read_regular_bytes(
        incoming / "training/trainable-checkpoint.pt", "trainable checkpoint"
    )
    if (
        len(trainable) != identity.trainable_checkpoint_size_bytes
        or hashlib.sha256(trainable).hexdigest() != identity.trainable_checkpoint_sha256
    ):
        raise ValueError("operational trainable checkpoint identity changed")
    inventory = capture_gemma_inventory(Path(identity.gemma_root))
    runtime_root = incoming / "runtime"
    seals_root = incoming / "seals"
    runtime_root.mkdir(mode=0o700)
    seals_root.mkdir(mode=0o700)
    inventory_path = runtime_root / "gemma-inventory.json"
    _write_new_file(inventory_path, inventory.to_json())
    source_seal = PhaseInputSeal(
        "duet-x-ltx-quality-phase-input-v1",
        "materialize",
        ReviewPhaseHistory((ReviewPhase.SOURCE_AUTHORED,)),
        canonical_sha256(
            {
                "source_receipt_sha256": _file_sha256(source_receipt_path),
                "population_fingerprints_sha256": population_fingerprints.fingerprint(),
            }
        ),
    ).validate()
    seal_path = seals_root / "source-authored.json"
    _write_new_file(seal_path, source_seal.to_json())
    input_paths = (
        *(incoming / relative for relative in sorted(expected_files)),
        inventory_path,
        seal_path,
    )
    configuration = OperationalConfiguration(
        "duet-x-ltx-quality-operations-v1",
        expected_source_commit,
        expected_source_archive_sha256,
        source_seal.fingerprint(),
        tuple(
            OperationalInput(
                path.relative_to(run_root).as_posix(),
                _file_sha256(path),
                path.stat().st_size,
            ).validate()
            for path in input_paths
        ),
    ).validate()
    _write_new_file(run_root / "operational-config.json", configuration.to_json())
    _fsync_directory(runtime_root)
    _fsync_directory(seals_root)
    _fsync_directory(run_root)
    return configuration


def _operational_preflight(
    run_root: Path, seal: PhaseInputSeal, configuration_name: str
) -> OperationalConfiguration:
    seal.validate()
    configuration = _load_operational_configuration(run_root, configuration_name)
    config_path = run_root / configuration_name
    if configuration.phase_seal_sha256 != seal.fingerprint():
        raise ValueError("operational phase seal identity mismatch")
    expected_paths: set[Path] = set()
    for item in configuration.inputs:
        item.validate()
        relative = Path(item.relative_path)
        current = run_root
        for index, part in enumerate(relative.parts):
            current = current / part
            if current.is_symlink():
                raise ValueError("operational input path contains a symlink")
            if index < len(relative.parts) - 1 and (
                not current.is_dir() or _mode(current) != 0o700
            ):
                raise ValueError("operational input directories must be mode 0700")
        try:
            resolved = current.resolve(strict=True)
            resolved.relative_to(run_root)
        except (OSError, ValueError) as error:
            raise ValueError("operational input is missing or out of run root") from error
        if not resolved.is_file() or _mode(resolved) != 0o600:
            raise ValueError("operational input must be a mode 0600 regular file")
        encoded = _read_regular_bytes(resolved, "operational input")
        if len(encoded) != item.size_bytes or hashlib.sha256(encoded).hexdigest() != item.sha256:
            raise ValueError("operational input size or identity mismatch")
        expected_paths.add(resolved)
    actual_paths: set[Path] = set()
    approved_outputs = {
        "operational",
        "scientific",
        "development",
        "environment",
        "source-extraction",
    }
    expected_directories = {
        run_root / parent
        for item in configuration.inputs
        for parent in Path(item.relative_path).parents
        if parent != Path()
    }
    for path in run_root.rglob("*"):
        if path == config_path:
            continue
        relative = path.relative_to(run_root)
        if relative.parts[0] == "source-extraction":
            # This generated subtree has its own exact receipt, Git-mode, tree, and
            # member-roster authentication.  Generic external-input modes would
            # incorrectly reject reviewed executable source files.
            continue
        if path.is_symlink():
            raise ValueError("operational input namespace contains a symlink")
        if path.is_dir():
            if _mode(path) != 0o700:
                raise ValueError("operational input directories must be mode 0700")
            generated_population = relative.is_relative_to(
                Path("incoming/source-population-preseal")
            )
            if (
                path not in expected_directories
                and relative.parts[0] not in approved_outputs
                and not generated_population
            ):
                raise ValueError("operational input namespace contains an extra directory")
        elif path.is_file():
            if _mode(path) != 0o600:
                raise ValueError("operational input must be a mode 0600 regular file")
            generated_population = relative.is_relative_to(
                Path("incoming/source-population-preseal")
            )
            if relative.parts[0] in approved_outputs or generated_population:
                _require_single_link_regular(path, "operational output evidence")
            else:
                actual_paths.add(path.resolve(strict=True))
        else:
            raise ValueError("operational input namespace contains unsafe evidence")
    if actual_paths != expected_paths:
        raise ValueError("operational input namespace contains missing or extra files")
    return configuration


def verify_operational_configuration_inputs(
    run_root: Path,
    expected: OperationalConfiguration,
) -> OperationalConfiguration:
    """Reauthenticate the complete configured input roster without model or CUDA work."""
    expected.validate()
    source_seal = load_phase_input_seal(run_root / "incoming/seals/source-authored.json")
    observed = _operational_preflight(run_root, source_seal, "operational-config.json")
    if observed != expected or observed.fingerprint() != expected.fingerprint():
        raise ValueError("operational configuration changed during complete input authentication")
    return observed


def _authenticate_operational_source(
    run_root: Path, configuration: OperationalConfiguration
) -> tuple[Path, SourceArchiveReceipt]:
    archive_inputs = tuple(
        item
        for item in configuration.inputs
        if item.relative_path == "incoming/source.tar"
        and item.sha256 == configuration.source_archive_sha256
    )
    receipt_inputs = tuple(
        item
        for item in configuration.inputs
        if item.relative_path == "incoming/source/source-receipt.json"
    )
    bootstrap_inputs = tuple(
        item
        for item in configuration.inputs
        if item.relative_path == "incoming/source/ltx-quality-bootstrap.py"
    )
    if len(archive_inputs) != 1 or len(receipt_inputs) != 1 or len(bootstrap_inputs) != 1:
        raise ValueError(
            "operational configuration must bind the exact source archive and receipt, "
            "plus bootstrap"
        )
    receipt_path = run_root / receipt_inputs[0].relative_path
    receipt = load_source_archive_receipt(receipt_path)
    if (
        receipt.source_commit != configuration.source_commit
        or receipt.source_archive_sha256 != configuration.source_archive_sha256
    ):
        raise ValueError("source receipt commit or archive identity changed")
    bootstrap_path = run_root / bootstrap_inputs[0].relative_path
    bootstrap = _read_regular_bytes(bootstrap_path, "source bootstrap")
    if (
        len(bootstrap) != receipt.bootstrap_size_bytes
        or hashlib.sha256(bootstrap).hexdigest() != receipt.bootstrap_sha256
    ):
        raise ValueError("source bootstrap identity changed")
    extraction_root = run_root / "source-extraction"
    extract_authenticated_source_archive(
        run_root / archive_inputs[0].relative_path,
        receipt_path,
        extraction_root,
    )
    return extraction_root / "tree", receipt


def _authenticate_operational_population(
    run_root: Path, configuration: OperationalConfiguration
) -> tuple[ComposedPopulation, AcceptedPopulationFingerprints]:
    archive_inputs = tuple(
        item
        for item in configuration.inputs
        if item.relative_path == "incoming/source-population-preseal.tar"
    )
    receipt_inputs = tuple(
        item
        for item in configuration.inputs
        if item.relative_path == "incoming/source-population/source-population-receipt.json"
    )
    if len(archive_inputs) != 1 or len(receipt_inputs) != 1:
        raise ValueError(
            "operational configuration must bind the exact source-population archive and receipt"
        )
    archive_path = run_root / archive_inputs[0].relative_path
    receipt_path = run_root / receipt_inputs[0].relative_path
    receipt = load_source_population_archive_receipt(receipt_path)
    if receipt.source_population_archive_sha256 != archive_inputs[0].sha256:
        raise ValueError("source-population archive identity changed")
    destination = run_root / "incoming/source-population-preseal"
    extracted = extract_authenticated_source_population_archive(
        archive_path, receipt_path, destination
    )
    population, fingerprints = load_accepted_source_population(destination)
    if extracted != receipt or fingerprints.fingerprint() != receipt.population_fingerprints_sha256:
        raise ValueError("source-population receipt or reconstructed fingerprints changed")
    return population, fingerprints


def verify_loaded_duet_module_origins(
    source_tree: Path,
    receipt: SourceArchiveReceipt,
    *,
    required_modules: tuple[str, ...],
    loaded_modules: Mapping[str, object] | None = None,
) -> None:
    """Bind every loaded Duet module, including transitive imports, to the extracted tree."""
    entries = {entry.relative_path: entry for entry in receipt.entries}
    observed: set[str] = set()
    modules = sys.modules if loaded_modules is None else loaded_modules
    for name, module in tuple(modules.items()):
        if name != "comfy_story" and not name.startswith("comfy_story."):
            continue
        origin_value = getattr(module, "__file__", None)
        if type(origin_value) is not str:
            raise ValueError("loaded Duet module has no authenticated source origin")
        origin = Path(origin_value)
        if origin.suffix != ".py" or origin.is_symlink():
            raise ValueError("loaded Duet module origin is not a source file")
        try:
            resolved = origin.resolve(strict=True)
            relative = resolved.relative_to(source_tree).as_posix()
        except (OSError, ValueError) as error:
            raise ValueError("loaded Duet module escaped the authenticated source tree") from error
        stem = f"src/{name.replace('.', '/')}"
        if relative not in {f"{stem}.py", f"{stem}/__init__.py"}:
            raise ValueError("loaded Duet module name and source origin disagree")
        expected = entries.get(relative)
        if expected is None:
            raise ValueError("loaded Duet module is absent from the tracked source receipt")
        encoded = _read_regular_bytes(resolved, "loaded Duet module")
        if (
            len(encoded) != expected.size_bytes
            or hashlib.sha256(encoded).hexdigest() != expected.sha256
            or _git_object_sha1("blob", encoded) != expected.git_blob_sha1
        ):
            raise ValueError("loaded Duet module source identity changed")
        observed.add(name)
    if not set(required_modules).issubset(observed):
        raise ValueError("required scientific Duet module origin was not authenticated")


def _verify_frozen_runtime_media_tools(source_commit: str, source_archive_sha256: str) -> None:
    from comfy_story.ltx_quality_runtime import (
        FrozenQualityRuntimeIdentity,
        verify_frozen_media_tools,
    )

    identity = FrozenQualityRuntimeIdentity.default(
        source_commit=source_commit,
        source_archive_sha256=source_archive_sha256,
    ).validate()
    verify_frozen_media_tools(identity)


def load_public_operational_status(
    run_root: Path,
    *,
    expected_source_commit: str,
    expected_source_archive_sha256: str,
) -> PublicOperationalStatus:
    """Strict disk-only import preflight; this function has no CUDA, model, or service path."""
    configuration = _load_operational_configuration(run_root, "operational-config.json")
    if (
        configuration.source_commit != expected_source_commit
        or configuration.source_archive_sha256 != expected_source_archive_sha256
    ):
        raise ValueError("operational source commit or archive identity mismatch")
    candidates = tuple(
        item for item in configuration.inputs if item.sha256 == configuration.phase_seal_sha256
    )
    if len(candidates) != 1:
        raise ValueError("operational configuration must bind exactly one phase seal input")
    seal_path = run_root / candidates[0].relative_path
    seal = load_phase_input_seal(seal_path)
    configuration = _operational_preflight(run_root, seal, "operational-config.json")
    _authenticate_operational_source(run_root, configuration)
    _authenticate_operational_population(run_root, configuration)
    _verify_frozen_runtime_media_tools(expected_source_commit, expected_source_archive_sha256)
    return PublicOperationalStatus(
        OperationalCommand.STATUS,
        seal.phases.phases[-1],
        OperationalStatusCode.READY,
    ).validate()


def _prepare_scientific_root(run_root: Path) -> Path:
    scientific = run_root / "scientific"
    if scientific.exists() or scientific.is_symlink():
        if scientific.is_symlink() or not scientific.is_dir() or _mode(scientific) != 0o700:
            raise ValueError("scientific output root must be a mode 0700 real directory")
    else:
        scientific.mkdir(mode=0o700)
        _fsync_directory(run_root)
    return scientific


def load_phase_cleanup_receipt(path: Path) -> PhaseCleanupReceipt:
    if path.is_symlink() or not path.is_file() or _mode(path) != 0o600:
        raise ValueError("phase cleanup receipt must be a mode 0600 regular file")
    raw = _strict_json(_read_regular_bytes(path, "phase cleanup receipt"))
    _exact_keys(
        raw,
        {
            "format",
            "command",
            "source_commit",
            "source_archive_sha256",
            "phase_seal_sha256",
            "resource_close_completed",
            "cuda_cleanup_completed",
        },
        "phase cleanup receipt",
    )
    try:
        receipt = PhaseCleanupReceipt(
            cast(str, raw["format"]),
            OperationalCommand(cast(str, raw["command"])),
            cast(str, raw["source_commit"]),
            cast(str, raw["source_archive_sha256"]),
            cast(str, raw["phase_seal_sha256"]),
            cast(bool, raw["resource_close_completed"]),
            cast(bool, raw["cuda_cleanup_completed"]),
        ).validate()
    except (TypeError, ValueError) as error:
        raise ValueError("phase cleanup receipt is malformed") from error
    if receipt.to_json() != _read_regular_bytes(path, "phase cleanup receipt"):
        raise ValueError("phase cleanup receipt must be canonical JSON")
    return receipt


def _persist_phase_cleanup(run_root: Path, receipt: PhaseCleanupReceipt) -> None:
    scientific = _prepare_scientific_root(run_root)
    directory = scientific / "cleanup"
    if directory.exists() or directory.is_symlink():
        if directory.is_symlink() or not directory.is_dir() or _mode(directory) != 0o700:
            raise ValueError("phase cleanup directory is unsafe")
    else:
        directory.mkdir(mode=0o700)
        _fsync_directory(scientific)
    path = directory / f"{receipt.command.value}.json"
    _write_new_file(path, receipt.validate().to_json())
    _fsync_directory(directory)


def verify_phase_cleanup_receipts(
    run_root: Path,
    commands: tuple[str, ...],
    source_commit: str,
    source_archive_sha256: str,
) -> None:
    if not commands or commands not in {("materialize",), ("materialize", "pilot")}:
        raise ValueError("phase cleanup command order changed")
    directory = run_root / "scientific/cleanup"
    if directory.is_symlink() or not directory.is_dir() or _mode(directory) != 0o700:
        raise ValueError("phase cleanup evidence directory is missing or unsafe")
    if {path.name for path in directory.iterdir()} != {f"{command}.json" for command in commands}:
        raise ValueError("phase cleanup evidence roster changed")
    seal_paths = {
        "materialize": run_root / "incoming/seals/source-authored.json",
        "pilot": run_root / "scientific/pilot-input.json",
    }
    for command in commands:
        receipt = load_phase_cleanup_receipt(directory / f"{command}.json")
        seal = load_phase_input_seal(seal_paths[command])
        if (
            receipt.command.value != command
            or receipt.source_commit != source_commit
            or receipt.source_archive_sha256 != source_archive_sha256
            or receipt.phase_seal_sha256 != seal.fingerprint()
            or not receipt.resource_close_completed
            or not receipt.cuda_cleanup_completed
        ):
            raise ValueError("phase cleanup receipt authentication failed")


def _release_phase_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _stopped_lifecycle_sha256(run_root: Path) -> str:
    path = run_root / "operational" / "comfy-stopped-seal.json"
    if path.is_symlink() or _mode(path) != 0o600:
        raise ValueError("pilot requires the mode 0600 stopped-Comfy lifecycle seal")
    raw = _strict_json(_read_regular_bytes(path, "stopped-Comfy lifecycle seal"))
    _exact_keys(
        raw,
        {
            "format",
            "original_status_sha256",
            "pre_run_queue_sha256",
            "stop_receipt_sha256",
            "stopped_status_sha256",
            "gpu_after_stop_sha256",
        },
        "stopped-Comfy lifecycle seal",
    )
    if raw["format"] != "duet-x-ltx-comfy-stopped-v1":
        raise ValueError("stopped-Comfy lifecycle format changed")
    for name, value in raw.items():
        if name != "format":
            _sha256(value, name)
    return hashlib.sha256(canonical_json(raw)).hexdigest()


def execute_operational_phase(
    command: str,
    run_root: Path,
    seal: PhaseInputSeal,
    *,
    expected_source_commit: str,
    expected_source_archive_sha256: str,
) -> PublicOperationalStatus:
    """Execute materialize or pilot through only the pinned reviewed runtime factories."""
    if command not in {"materialize", "pilot"} or seal.next_command != command:
        raise ValueError("scientific operational command does not match its phase seal")
    configuration = _load_operational_configuration(run_root, "operational-config.json")
    source_seal_inputs = tuple(
        item
        for item in configuration.inputs
        if item.relative_path == "incoming/seals/source-authored.json"
        and item.sha256 == configuration.phase_seal_sha256
    )
    if len(source_seal_inputs) != 1:
        raise ValueError("operational configuration must bind the exact source-authored phase seal")
    source_seal = load_phase_input_seal(run_root / "incoming/seals/source-authored.json")
    if source_seal.next_command != "materialize":
        raise ValueError("configured source-authored seal must authorize materialization")
    configuration = _operational_preflight(run_root, source_seal, "operational-config.json")
    if command == "materialize":
        if seal != source_seal:
            raise ValueError("materialize requires the configured source-authored seal")
    else:
        pilot_path = run_root / "scientific/pilot-input.json"
        if pilot_path.is_symlink() or not pilot_path.is_file() or _mode(pilot_path) != 0o600:
            raise ValueError("pilot requires the exact internal pilot-input seal")
        internal_pilot_seal = load_phase_input_seal(pilot_path)
        if seal != internal_pilot_seal:
            raise ValueError("pilot requires the exact authenticated internal phase transition")
    if (
        configuration.source_commit != expected_source_commit
        or configuration.source_archive_sha256 != expected_source_archive_sha256
    ):
        raise ValueError("operational source commit or archive identity mismatch")
    source_tree, source_receipt = _authenticate_operational_source(run_root, configuration)
    verify_loaded_duet_module_origins(
        source_tree,
        source_receipt,
        required_modules=(
            "comfy_story.contracts",
            "comfy_story.ltx_quality_generation",
            "comfy_story.ltx_quality_operations",
            "comfy_story.ltx_quality_protocol",
            "comfy_story.ltx_quality_scenes",
            "comfy_story.media_probe",
        ),
    )
    population, population_fingerprints = _authenticate_operational_population(
        run_root, configuration
    )
    scientific = _prepare_scientific_root(run_root)
    development_output = run_root / "development"
    if command == "materialize":
        if (
            tuple(scientific.iterdir())
            or development_output.exists()
            or development_output.is_symlink()
        ):
            raise ValueError("materialize requires fresh scientific and development outputs")
    elif (
        {entry.name for entry in scientific.iterdir()}
        != {
            "cleanup",
            "materialized",
            "materialized-finalization.json",
            "pilot-input.json",
        }
        or development_output.exists()
        or development_output.is_symlink()
    ):
        raise ValueError("pilot requires exact materialized parents and fresh development output")
    if command == "pilot":
        verify_phase_cleanup_receipts(
            run_root,
            ("materialize",),
            expected_source_commit,
            expected_source_archive_sha256,
        )
    comfy_lifecycle_sha256 = _stopped_lifecycle_sha256(run_root) if command == "pilot" else None

    from comfy_story.ltx_quality_runtime import (
        DevelopmentRuntimePaths,
        FrozenDevelopmentIdentity,
        FrozenQualityRuntimeIdentity,
        PinnedOfficialLTXFactory,
        PinnedQualityDevelopmentFactory,
        load_gemma_inventory_receipt,
        verify_frozen_media_tools,
        verify_gemma_inventory,
    )

    verify_loaded_duet_module_origins(
        source_tree,
        source_receipt,
        required_modules=(
            "comfy_story.ltx_media",
            "comfy_story.ltx_quality_runtime",
        ),
    )

    runtime_identity = FrozenQualityRuntimeIdentity.default(
        source_commit=expected_source_commit,
        source_archive_sha256=expected_source_archive_sha256,
    ).validate()
    verify_frozen_media_tools(runtime_identity)
    gemma_inputs = tuple(
        item
        for item in configuration.inputs
        if item.relative_path == "incoming/runtime/gemma-inventory.json"
    )
    if len(gemma_inputs) != 1:
        raise ValueError("operational configuration must bind exactly one Gemma inventory receipt")
    gemma_inventory = load_gemma_inventory_receipt(
        run_root / "incoming/runtime/gemma-inventory.json"
    )
    verify_gemma_inventory(Path(runtime_identity.gemma_root), gemma_inventory)
    official_factory = PinnedOfficialLTXFactory(
        checkout_root=Path(runtime_identity.ltx_checkout_root),
        pipelines_source_root=Path(runtime_identity.ltx_pipelines_source_root),
        core_source_root=Path(runtime_identity.ltx_core_source_root),
        dependency_source_root=Path(runtime_identity.dependency_source_root),
    )
    bundle_identity = MaterializedBundleIdentity(
        RuntimeCoordinate.default(),
        runtime_identity.fingerprint(),
        canonical_sha256(
            {
                "teacher_checkpoint_sha256": runtime_identity.teacher_checkpoint_sha256,
                "gemma_root": runtime_identity.gemma_root,
                "offload_mode": runtime_identity.offload_mode,
                "foundation_guard_sha256": runtime_identity.foundation_guard_sha256,
            }
        ),
        runtime_identity.ltx_source_commit,
    ).validate()
    if command == "materialize":
        api = official_factory.load()
        materializer = api.build_materializer(
            runtime_identity, device=torch.device(runtime_identity.device)
        )
        try:
            loaded = materialize_accepted_population(
                scientific / "materialized",
                source_seal=seal,
                population=population,
                population_fingerprints=population_fingerprints,
                identity=bundle_identity,
                materializer=materializer,
            )
            materialized_finalization = finalize_materialized_bundle(loaded.root)
            if materialized_finalization != finalize_materialized_bundle(loaded.root):
                raise ValueError("materialized disk finalization changed on repetition")
            _write_new_file(
                scientific / "materialized-finalization.json", materialized_finalization
            )
            pilot_seal = PhaseInputSeal(
                "duet-x-ltx-quality-phase-input-v1",
                "pilot",
                ReviewPhaseHistory((ReviewPhase.SOURCE_AUTHORED,)),
                loaded.seal.fingerprint(),
            ).validate()
            _write_new_file(scientific / "pilot-input.json", pilot_seal.to_json())
            _fsync_directory(scientific)
            return PublicOperationalStatus(
                OperationalCommand.MATERIALIZE,
                ReviewPhase.SOURCE_AUTHORED,
                OperationalStatusCode.COMPLETE,
            ).validate()
        finally:
            active_error = sys.exception()
            close_error: BaseException | None = None
            cuda_error: BaseException | None = None
            try:
                materializer.close()
            except BaseException as error:
                close_error = error
            try:
                _release_phase_cuda()
            except BaseException as error:
                cuda_error = error
            _persist_phase_cleanup(
                run_root,
                PhaseCleanupReceipt(
                    "duet-x-ltx-phase-cleanup-v1",
                    OperationalCommand.MATERIALIZE,
                    configuration.source_commit,
                    configuration.source_archive_sha256,
                    seal.fingerprint(),
                    close_error is None,
                    cuda_error is None,
                ).validate(),
            )
            if active_error is None:
                if close_error is not None:
                    raise close_error
                if cuda_error is not None:
                    raise cuda_error
    bundle = load_materialized_bundle(scientific / "materialized")
    materialized_finalization = _read_regular_bytes(
        scientific / "materialized-finalization.json", "materialized finalization"
    )
    if _mode(
        scientific / "materialized-finalization.json"
    ) != 0o600 or materialized_finalization != finalize_materialized_bundle(bundle.root):
        raise ValueError("pilot materialized disk finalization changed")
    if (
        bundle.seal.fingerprint() != seal.parent_sha256
        or bundle.manifest.population_fingerprints != population_fingerprints
        or bundle.manifest.identity != bundle_identity
    ):
        raise ValueError("pilot materialized/source/runtime ancestry changed")
    paths = DevelopmentRuntimePaths(
        teacher_checkpoint=Path(runtime_identity.teacher_checkpoint_path),
        gemma_root=Path(runtime_identity.gemma_root),
        scene_manifest=run_root / "incoming/development/development-00.json",
        latent_bundle=run_root / "incoming/development/development-00.pt",
        k2_receipt=run_root / "incoming/development/development-00-k2.json",
        trainable_checkpoint=run_root / "incoming/training/trainable-checkpoint.pt",
    )
    development = load_development_input(
        paths.scene_manifest, paths.latent_bundle, paths.k2_receipt
    )
    confirmatory_ancestry = (bundle.manifest, population.manifest())
    disjointness = validate_development_disjointness(
        development,
        bundle.manifest.clusters,
        confirmatory_ancestry,
    )
    persist_development_disjointness(run_root, disjointness)
    prepared = PinnedQualityDevelopmentFactory(
        runtime_identity,
        FrozenDevelopmentIdentity.default(),
        official_factory,
        expected_gemma_inventory=gemma_inventory,
    ).prepare(
        paths=paths,
        observed_source_commit=configuration.source_commit,
        observed_source_archive_sha256=configuration.source_archive_sha256,
        clusters=bundle.manifest.clusters,
        confirmatory_ancestry=confirmatory_ancestry,
    )
    if prepared.disjointness != disjointness:
        prepared.adapter.close()
        raise ValueError("development runtime changed the pre-model disjointness proof")
    adapter = prepared.adapter
    try:
        run_development_rehearsal(
            store=DevelopmentAttemptStore(run_root).validate(),
            source_seal=seal,
            scene_id="development-00",
            disjointness_receipt=disjointness,
            comfy_lifecycle_receipt_sha256=cast(str, comfy_lifecycle_sha256),
            identity=DevelopmentExecutionIdentity(
                adapter.engine_identity_sha256,
                adapter.foundation_sha256,
                adapter.runtime_identity_sha256,
            ).validate(),
            bind_forward=lambda method, seed: cast(
                DevelopmentForwardBindingLike, adapter.binding(method, seed)
            ),
            forward=adapter.forward,
            metrics=DevelopmentRuntimeMetrics(
                torch.cuda.synchronize,
                torch.cuda.reset_peak_memory_stats,
                torch.cuda.max_memory_allocated,
                torch.cuda.max_memory_reserved,
            ).validate(),
        )
        development_finalization = finalize_development_rehearsal(run_root)
        if development_finalization != finalize_development_rehearsal(run_root):
            raise ValueError("development disk finalization changed on repetition")
        finalization_path = checked_development_output(
            run_root, Path("development/development-finalization.json")
        )
        _write_new_file(finalization_path, development_finalization)
        _fsync_directory(finalization_path.parent)
    finally:
        active_error = sys.exception()
        close_error = None
        cuda_error = None
        try:
            adapter.close()
        except BaseException as error:
            close_error = error
        try:
            _release_phase_cuda()
        except BaseException as error:
            cuda_error = error
        _persist_phase_cleanup(
            run_root,
            PhaseCleanupReceipt(
                "duet-x-ltx-phase-cleanup-v1",
                OperationalCommand.PILOT,
                configuration.source_commit,
                configuration.source_archive_sha256,
                seal.fingerprint(),
                close_error is None,
                cuda_error is None,
            ).validate(),
        )
        if active_error is None:
            if close_error is not None:
                raise close_error
            if cuda_error is not None:
                raise cuda_error
    return PublicOperationalStatus(
        OperationalCommand.PILOT,
        ReviewPhase.DEVELOPMENT_REHEARSED,
        OperationalStatusCode.COMPLETE,
    ).validate()


def _noop_phase_preflight(
    _command: str,
    _run_root: Path,
    _seal: PhaseInputSeal,
    _configuration: OperationalConfiguration,
) -> None:
    return None


@dataclass(frozen=True, slots=True)
class OperationalHandler:
    """Strict handler boundary; CLI adaptation is intentionally owned by a later task."""

    model_loader: Callable[[], object]
    materialize: Callable[[object], GenerationStatusCode]
    pilot: Callable[[object], GenerationStatusCode]
    configuration_name: str = "operational-config.json"
    phase_preflight: (
        Callable[[str, Path, PhaseInputSeal, OperationalConfiguration], None] | None
    ) = None

    def __post_init__(self) -> None:
        if not all(
            callable(value)
            for value in (
                self.model_loader,
                self.materialize,
                self.pilot,
            )
        ):
            raise ValueError("operational callbacks must be callable")
        if self.phase_preflight is None or self.phase_preflight is _noop_phase_preflight:
            raise ValueError("operational phase preflight must be mandatory and non-noop")
        if self.configuration_name != "operational-config.json":
            raise ValueError("operational configuration filename changed")

    def dispatch(
        self, command: str, run_root: Path, seal: PhaseInputSeal
    ) -> PublicOperationalStatus:
        if command not in {"status", "materialize", "pilot"}:
            raise ValueError("operational command must be status, materialize, or pilot")
        if command != "status" and command != seal.next_command:
            raise ValueError("operational command does not match its authenticated seal command")
        configuration = _operational_preflight(run_root, seal, self.configuration_name)
        phase_preflight = self.phase_preflight
        if phase_preflight is None:
            raise RuntimeError("operational phase preflight is unavailable")
        phase_preflight(command, run_root, seal, configuration)
        if command == "status":
            return PublicOperationalStatus(
                OperationalCommand.STATUS,
                seal.phases.phases[-1],
                OperationalStatusCode.READY,
            ).validate()
        model = self.model_loader()
        status = self.materialize(model) if command == "materialize" else self.pilot(model)
        if status is not GenerationStatusCode.COMPLETE:
            raise ValueError("operational callback must return the complete internal status code")
        return PublicOperationalStatus(
            OperationalCommand(command),
            seal.phases.phases[-1],
            OperationalStatusCode.COMPLETE,
        ).validate()


class ForgeLifecycleClient(Protocol):
    def status(self) -> dict[str, object]: ...

    def queue(self) -> dict[str, object]: ...

    def stop(self) -> dict[str, object]: ...

    def start(self) -> dict[str, object]: ...

    def free_vram_mib(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ForgeWrapperReceipt(FrozenRecord):
    format: str
    import_only: bool
    scientific_success: bool
    restore_success: bool
    original_status_sha256: str
    pre_run_queue_sha256: str
    stop_receipt_sha256: str
    stopped_status_sha256: str
    gpu_after_stop_sha256: str
    cleanup_result_sha256: str
    restore_receipt_sha256: str
    final_status_sha256: str
    restored_queue_sha256: str
    gpu_after_restore_sha256: str
    restore_result_sha256: str
    environment_receipts_sha256: str

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-forge-wrapper-v1":
            raise ValueError("Forge wrapper receipt format changed")
        if any(
            type(value) is not bool
            for value in (self.import_only, self.scientific_success, self.restore_success)
        ):
            raise ValueError("Forge wrapper outcomes must be explicit booleans")
        for name in (
            "original_status_sha256",
            "pre_run_queue_sha256",
            "stop_receipt_sha256",
            "stopped_status_sha256",
            "gpu_after_stop_sha256",
            "cleanup_result_sha256",
            "restore_receipt_sha256",
            "final_status_sha256",
            "restored_queue_sha256",
            "gpu_after_restore_sha256",
            "restore_result_sha256",
            "environment_receipts_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.import_only and (self.scientific_success or not self.restore_success):
            raise ValueError("import-only wrapper receipt cannot claim scientific execution")
        return self


class ForgeRestoreError(RuntimeError):
    def __init__(self, receipt: ForgeWrapperReceipt) -> None:
        super().__init__("Comfy restoration was not verified; restore failure dominates")
        self.receipt = receipt


class ForgeEvidenceError(RuntimeError):
    def __init__(
        self,
        receipt: ForgeWrapperReceipt,
        evidence_error: BaseException,
        scientific_error: BaseException | None,
    ) -> None:
        super().__init__("Forge lifecycle evidence persistence failed")
        self.receipt = receipt
        self.evidence_error = evidence_error
        self.scientific_error = scientific_error


_FORGE_STATUS_KEYS = frozenset(
    {
        "healthy",
        "name",
        "ok",
        "path",
        "pid",
        "port",
        "running",
        "serve_url",
        "started_at",
        "status",
        "uptime_s",
    }
)
_FORGE_STOPPED_REQUIRED_KEYS = frozenset(
    {"healthy", "name", "ok", "path", "port", "running", "serve_url", "status"}
)
_FORGE_DOCUMENTED_STOPPED_KEYS = frozenset(
    {"ok", "pid", "port", "running", "serve_url", "tunnel_url"}
)
_FORGE_PRESERVED_STOPPED_KEYS = frozenset({"name", "ok", "path", "running", "status"})
_FORGE_TRANSITIONAL_RUNNING_KEYS = _FORGE_STATUS_KEYS - {"serve_url"}


def _forge_serve_url(host_id: str) -> str:
    if host_id not in {"forge1", "forge2"}:
        raise ValueError("Forge host identity must be forge1 or forge2")
    return f"https://{host_id}.fennec-typhon.ts.net:8188"


def _validated_forge_transitional_running_status(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _FORGE_TRANSITIONAL_RUNNING_KEYS:
        raise ValueError("Comfy status response is not exact")
    status = cast(dict[str, object], value)
    if (
        status["healthy"] is not False
        or status["ok"] is not True
        or status["name"] != "main"
        or status["path"] != "/mnt/data/comfy-runner/state/installations/main"
        or type(status["port"]) is not int
        or status["port"] != 8188
        or status["running"] is not True
        or status["status"] != "installed"
        or type(status["pid"]) is not int
        or status["pid"] <= 0
        or type(status["started_at"]) not in {int, float}
        or type(status["uptime_s"]) not in {int, float}
    ):
        raise ValueError("Comfy status response is not exact")
    started_at = cast(int | float, status["started_at"])
    uptime_s = cast(int | float, status["uptime_s"])
    if (
        not math.isfinite(started_at)
        or started_at <= 0
        or not math.isfinite(uptime_s)
        or uptime_s < 0
    ):
        raise ValueError("Comfy status response is not exact")
    return status


def _validated_forge_status(value: object, host_id: str = "forge1") -> dict[str, object]:
    serve_url = _forge_serve_url(host_id)
    if type(value) is not dict or type(value.get("running")) is not bool:
        raise ValueError("Comfy status response is not exact")
    status = cast(dict[str, object], value)
    running = cast(bool, status["running"])
    keys = set(status)
    documented_stopped = not running and keys == _FORGE_DOCUMENTED_STOPPED_KEYS
    preserved_stopped = not running and keys == _FORGE_PRESERVED_STOPPED_KEYS
    expanded_stopped = (
        not running and keys >= _FORGE_STOPPED_REQUIRED_KEYS and keys <= _FORGE_STATUS_KEYS
    )
    if running and keys != _FORGE_STATUS_KEYS:
        raise ValueError("Comfy status response is not exact")
    if not running and not documented_stopped and not preserved_stopped and not expanded_stopped:
        raise ValueError("Comfy status response is not exact")
    if preserved_stopped:
        if (
            status["name"] != "main"
            or status["ok"] is not True
            or status["path"] != "/mnt/data/comfy-runner/state/installations/main"
            or status["status"] != "installed"
        ):
            raise ValueError("Comfy status response is not exact")
        return status
    if documented_stopped:
        if (
            status["ok"] is not True
            or type(status["port"]) is not int
            or status["port"] != 8188
            or status["serve_url"] != serve_url
            or status["pid"] is not None
            or status["tunnel_url"] is not None
        ):
            raise ValueError("Comfy status response is not exact")
        return status
    if (
        type(status["healthy"]) is not bool
        or status["healthy"] is not running
        or status["ok"] is not True
        or status["name"] != "main"
        or status["path"] != "/mnt/data/comfy-runner/state/installations/main"
        or type(status["port"]) is not int
        or status["port"] != 8188
        or status["serve_url"] != serve_url
        or status["status"] != "installed"
    ):
        raise ValueError("Comfy status response is not exact")
    pid = status.get("pid")
    started_at = status.get("started_at")
    uptime_s = status.get("uptime_s")
    if running:
        if (
            type(pid) is not int
            or pid <= 0
            or type(started_at) not in {int, float}
            or type(uptime_s) not in {int, float}
        ):
            raise ValueError("Comfy status response is not exact")
        started_number = cast(int | float, started_at)
        uptime_number = cast(int | float, uptime_s)
        if (
            not math.isfinite(started_number)
            or started_number <= 0
            or not math.isfinite(uptime_number)
            or uptime_number < 0
        ):
            raise ValueError("Comfy status response is not exact")
    elif any(
        status.get(name) is not None for name in ("pid", "started_at", "uptime_s") if name in status
    ):
        raise ValueError("Comfy status response is not exact")
    return status


def _strict_status(
    value: object, expected_running: bool, host_id: str = "forge1"
) -> dict[str, object]:
    status = _validated_forge_status(value, host_id)
    if status["running"] is not expected_running:
        raise ValueError("Comfy status response is not exact")
    return status


def _validated_forge_poll_status(
    value: object, host_id: str = "forge1"
) -> tuple[dict[str, object], bool]:
    try:
        return _validated_forge_transitional_running_status(value), False
    except ValueError:
        return _validated_forge_status(value, host_id), True


_UNSAFE_RESPONSE_FRAGMENTS = (
    "access_key",
    "api_key",
    "authorization",
    "bearer",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_SAFE_JOB_ID_CHARACTERS = frozenset("0123456789abcdef")


def _unsafe_response_text(value: object) -> bool:
    if type(value) is str:
        encoded = value.encode()
        normalized = value.casefold()
        recognizable_credential = normalized.startswith(("hf_", "ghp_", "github_pat_")) or (
            value.startswith(("AKIA", "ASIA"))
            and len(value) == 20
            and all(character.isupper() or character.isdigit() for character in value)
        )
        return (
            len(encoded) > 8_192
            or recognizable_credential
            or any(fragment in normalized for fragment in _UNSAFE_RESPONSE_FRAGMENTS)
            or any(not character.isprintable() and character not in "\n\r\t" for character in value)
        )
    if type(value) is list:
        return len(value) > 64 or any(_unsafe_response_text(item) for item in value)
    if type(value) is dict:
        return any(
            _unsafe_response_text(key) or _unsafe_response_text(item) for key, item in value.items()
        )
    return False


def _redacted_controller_observation(
    encoded: bytes, *, endpoint: str, phase: str, reason: str
) -> dict[str, object]:
    return {
        "format": "duet-x-ltx-redacted-observation-v1",
        "endpoint": endpoint,
        "phase": phase,
        "redacted": True,
        "rejection_reason": reason,
        "response_sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }


def _persist_controller_response(
    evidence: Path,
    name: str,
    value: object,
    *,
    endpoint: str,
    phase: str,
    host_id: str = "forge1",
) -> tuple[str, ValueError | None]:
    encoded = canonical_json(value)
    if endpoint == "/main/status":
        raw_status = False
        if type(value) is dict:
            try:
                _validated_forge_poll_status(value, host_id)
            except ValueError:
                pass
            else:
                raw_status = True
        if not raw_status:
            redacted = _redacted_controller_observation(
                encoded,
                endpoint=endpoint,
                phase=phase,
                reason="unapproved-status-body",
            )
            return _write_wrapper_evidence(evidence, name, redacted), None
    if endpoint == "/main/stop" and type(value) is dict and "output" in value:
        output = value["output"]
        if type(output) is list and output:
            if (
                type(value.get("ok")) is bool
                and type(value.get("was_running")) is bool
                and all(type(item) is str for item in output)
            ):
                lines = cast(list[str], output)
                sanitized = {
                    "format": "duet-x-ltx-sanitized-stop-observation-v1",
                    "endpoint": endpoint,
                    "phase": phase,
                    "ok": value["ok"],
                    "was_running": value["was_running"],
                    "response_sha256": hashlib.sha256(encoded).hexdigest(),
                    "size_bytes": len(encoded),
                    "output_count": len(lines),
                    "output_lines": [
                        {
                            "sha256": hashlib.sha256(line.encode()).hexdigest(),
                            "size_bytes": len(line.encode()),
                        }
                        for line in lines
                    ],
                }
                return _write_wrapper_evidence(evidence, name, sanitized), None
            redacted = _redacted_controller_observation(
                encoded,
                endpoint=endpoint,
                phase=phase,
                reason="unsafe-response-body",
            )
            return (
                _write_wrapper_evidence(evidence, name, redacted),
                ValueError("controller response body is unsafe"),
            )
    if endpoint == "/main/start" and type(value) is dict:
        job_id = value.get("job_id")
        approved_job_id = (
            type(job_id) is str
            and len(job_id) == 12
            and all(character in _SAFE_JOB_ID_CHARACTERS for character in job_id)
        )
        if not approved_job_id:
            redacted = _redacted_controller_observation(
                encoded,
                endpoint=endpoint,
                phase=phase,
                reason="unapproved-start-job-id",
            )
            return (
                _write_wrapper_evidence(evidence, name, redacted),
                ValueError("Comfy start response is not exact"),
            )
    unsafe = len(encoded) > 262_144 or _unsafe_response_text(value)
    if not unsafe:
        return _write_wrapper_evidence(evidence, name, value), None
    redacted = _redacted_controller_observation(
        encoded,
        endpoint=endpoint,
        phase=phase,
        reason="unsafe-response-body",
    )
    return (
        _write_wrapper_evidence(evidence, name, redacted),
        ValueError("controller response body is unsafe"),
    )


def _strict_stop_response(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"ok", "was_running", "output"}:
        raise ValueError("Comfy stop response is not exact")
    response = cast(dict[str, object], value)
    output = response["output"]
    if (
        response["ok"] is not True
        or response["was_running"] is not True
        or type(output) is not list
        or len(output) > 64
        or any(type(item) is not str for item in output)
    ):
        raise ValueError("Comfy stop response is not exact")
    strings = cast(list[str], output)
    if sum(len(item.encode()) for item in strings) > 65_536 or any(
        len(item.encode()) > 8_192
        or any(not character.isprintable() and character not in "\n\r\t" for character in item)
        for item in strings
    ):
        raise ValueError("Comfy stop response is not exact")
    return response


def _strict_start_response(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"ok", "job_id", "async"}:
        raise ValueError("Comfy start response is not exact")
    response = cast(dict[str, object], value)
    job_id = response["job_id"]
    if (
        response["ok"] is not True
        or response["async"] is not True
        or type(job_id) is not str
        or len(job_id) != 12
        or any(character not in _SAFE_JOB_ID_CHARACTERS for character in job_id)
    ):
        raise ValueError("Comfy start response is not exact")
    return response


def _wait_status(
    client: ForgeLifecycleClient,
    expected_running: bool,
    sleeper: Callable[[float], None],
    observe: Callable[[int, object], object],
    host_id: str = "forge1",
) -> dict[str, object]:
    for attempt in range(121):
        observed = client.status()
        observe(attempt, observed)
        value, terminal = _validated_forge_poll_status(observed, host_id)
        if terminal and value["running"] is expected_running:
            return value
        if attempt < 120:
            sleeper(2.0)
    raise ValueError("Comfy status transition exceeded 240 seconds")


def _strict_empty_queue(value: object) -> dict[str, object]:
    expected: dict[str, object] = {"queue_pending": [], "queue_running": []}
    if type(value) is not dict or value != expected:
        raise ValueError("Comfy queue must be exactly empty")
    return cast(dict[str, object], value)


def _validate_forge_run_root(run_root: Path) -> None:
    try:
        parsed = uuid.UUID(run_root.name)
    except (AttributeError, ValueError) as error:
        raise ValueError("Forge run root must use a canonical UUID name") from error
    if (
        not run_root.is_absolute()
        or str(parsed) != run_root.name
        or run_root.is_symlink()
        or not run_root.is_dir()
        or run_root.resolve(strict=True) != run_root
        or _mode(run_root) != 0o700
    ):
        raise ValueError("Forge run root must be canonical, real, absolute, and mode 0700")


def _write_wrapper_evidence(directory: Path, name: str, value: object) -> str:
    if not name.endswith(".json") or "/" in name:
        raise ValueError("Forge wrapper evidence name is unsafe")
    encoded = canonical_json(value)
    _write_new_file(directory / name, encoded)
    _fsync_directory(directory)
    return hashlib.sha256(encoded).hexdigest()


def _forge_wrapper_receipt(
    *,
    import_only: bool,
    scientific_success: bool,
    restore_success: bool,
    hashes: Mapping[str, str],
    environment_receipts_sha256: str,
) -> ForgeWrapperReceipt:
    return ForgeWrapperReceipt(
        format="duet-x-ltx-forge-wrapper-v1",
        import_only=import_only,
        scientific_success=scientific_success,
        restore_success=restore_success,
        original_status_sha256=hashes["original"],
        pre_run_queue_sha256=hashes["queue"],
        stop_receipt_sha256=hashes["stop"],
        stopped_status_sha256=hashes["stopped"],
        gpu_after_stop_sha256=hashes["gpu_stop"],
        cleanup_result_sha256=hashes["cleanup"],
        restore_receipt_sha256=hashes["restore"],
        final_status_sha256=hashes["final"],
        restored_queue_sha256=hashes["restored_queue"],
        gpu_after_restore_sha256=hashes["gpu_restore"],
        restore_result_sha256=hashes["restore_result"],
        environment_receipts_sha256=environment_receipts_sha256,
    ).validate()


def run_persisted_forge_wrapper(
    run_root: Path,
    client: ForgeLifecycleClient | None,
    *,
    preflight: Callable[[], None],
    operation: Callable[[], None],
    dispose: Callable[[], None],
    cuda_cleanup: Callable[[], None],
    import_only: bool,
    sleeper: Callable[[float], None] = time.sleep,
    host_id: str = "forge1",
) -> ForgeWrapperReceipt:
    """Run one Forge lifecycle and append every observed response to canonical evidence."""
    _validate_forge_run_root(run_root)
    _forge_serve_url(host_id)
    _environment_receipts_sha256(run_root)
    if not import_only and client is None:
        raise ValueError("scientific Forge wrapper requires a lifecycle client")
    active_client = cast(ForgeLifecycleClient, client)
    evidence = run_root / "operational"
    if evidence.exists() or evidence.is_symlink():
        raise ValueError("Forge operational evidence namespace must be fresh")
    evidence.mkdir(mode=0o700)
    _fsync_directory(run_root)
    zero = "0" * 64
    hashes = {
        "original": zero,
        "queue": zero,
        "stop": zero,
        "stopped": zero,
        "gpu_stop": zero,
        "cleanup": zero,
        "restore": zero,
        "final": zero,
        "restored_queue": zero,
        "gpu_restore": zero,
        "restore_result": zero,
    }
    scientific_success = False
    restore_success = True
    stopped_path = False
    scientific_error: BaseException | None = None
    restore_error: BaseException | None = None
    evidence_error: BaseException | None = None
    try:
        preflight()
        if import_only:
            receipt = _forge_wrapper_receipt(
                import_only=True,
                scientific_success=False,
                restore_success=True,
                hashes=hashes,
                environment_receipts_sha256=_environment_receipts_sha256(run_root),
            )
            _write_wrapper_evidence(evidence, "wrapper-result.json", receipt.to_dict())
            return receipt
        observed_original = active_client.status()
        hashes["original"], _ = _persist_controller_response(
            evidence,
            "comfy-original-status.json",
            observed_original,
            endpoint="/main/status",
            phase="original",
            host_id=host_id,
        )
        _strict_status(observed_original, True, host_id)
        queue = _strict_empty_queue(active_client.queue())
        hashes["queue"] = _write_wrapper_evidence(evidence, "comfy-pre-run-queue.json", queue)
        stopped_path = True
        stop = active_client.stop()
        try:
            hashes["stop"], unsafe_stop = _persist_controller_response(
                evidence,
                "comfy-stop.json",
                stop,
                endpoint="/main/stop",
                phase="stop",
                host_id=host_id,
            )
        except BaseException as error:
            evidence_error = error
            raise
        if unsafe_stop is not None:
            raise unsafe_stop
        _strict_stop_response(stop)
        stopped = _wait_status(
            active_client,
            False,
            sleeper,
            lambda attempt, value: _persist_controller_response(
                evidence,
                f"comfy-stop-status-observed-{attempt:03d}.json",
                value,
                endpoint="/main/status",
                phase="stop-poll",
                host_id=host_id,
            ),
            host_id,
        )
        hashes["stopped"], _ = _persist_controller_response(
            evidence,
            "comfy-stop-status.json",
            stopped,
            endpoint="/main/status",
            phase="stopped",
            host_id=host_id,
        )
        free_vram = active_client.free_vram_mib()
        gpu_stop = {"free_vram_mib": free_vram}
        hashes["gpu_stop"] = _write_wrapper_evidence(evidence, "gpu-after-stop.json", gpu_stop)
        if type(free_vram) is not int or free_vram < 90_000:
            raise ValueError("free VRAM is below the 90,000 MiB floor")
        _write_wrapper_evidence(
            evidence,
            "comfy-stopped-seal.json",
            {
                "format": "duet-x-ltx-comfy-stopped-v1",
                "original_status_sha256": hashes["original"],
                "pre_run_queue_sha256": hashes["queue"],
                "stop_receipt_sha256": hashes["stop"],
                "stopped_status_sha256": hashes["stopped"],
                "gpu_after_stop_sha256": hashes["gpu_stop"],
            },
        )
        operation()
        scientific_success = True
    except BaseException as error:
        scientific_error = error
    finally:
        if stopped_path:
            dispose_completed = False
            cuda_cleanup_completed = False
            try:
                dispose()
                dispose_completed = True
            except BaseException as error:
                if scientific_error is None:
                    scientific_error = error
                    scientific_success = False
            try:
                cuda_cleanup()
                cuda_cleanup_completed = True
            except BaseException as error:
                if scientific_error is None:
                    scientific_error = error
                    scientific_success = False
            try:
                hashes["cleanup"] = _write_wrapper_evidence(
                    evidence,
                    "cleanup-result.json",
                    {
                        "dispose_completed": dispose_completed,
                        "cuda_cleanup_completed": cuda_cleanup_completed,
                    },
                )
            except BaseException as error:
                evidence_error = error
            try:
                restore = active_client.start()
                try:
                    hashes["restore"], unsafe_start = _persist_controller_response(
                        evidence,
                        "comfy-restore-job.json",
                        restore,
                        endpoint="/main/start",
                        phase="restore",
                        host_id=host_id,
                    )
                except BaseException as error:
                    evidence_error = evidence_error or error
                else:
                    if unsafe_start is not None:
                        evidence_error = evidence_error or unsafe_start
                    else:
                        try:
                            _strict_start_response(restore)
                        except BaseException as error:
                            evidence_error = evidence_error or error
            except BaseException as error:
                evidence_error = evidence_error or error
            try:
                final = _wait_status(
                    active_client,
                    True,
                    sleeper,
                    lambda attempt, value: _persist_controller_response(
                        evidence,
                        f"comfy-restore-status-observed-{attempt:03d}.json",
                        value,
                        endpoint="/main/status",
                        phase="restore-poll",
                        host_id=host_id,
                    ),
                    host_id,
                )
                try:
                    hashes["final"], _ = _persist_controller_response(
                        evidence,
                        "comfy-last-status.json",
                        final,
                        endpoint="/main/status",
                        phase="restored",
                        host_id=host_id,
                    )
                except BaseException as error:
                    evidence_error = evidence_error or error
                restored_queue = _strict_empty_queue(active_client.queue())
                try:
                    hashes["restored_queue"] = _write_wrapper_evidence(
                        evidence, "comfy-restored-queue.json", restored_queue
                    )
                except BaseException as error:
                    evidence_error = evidence_error or error
                restored_vram = active_client.free_vram_mib()
                if type(restored_vram) is not int or restored_vram < 0:
                    raise ValueError("restored GPU inventory is malformed")
                try:
                    hashes["gpu_restore"] = _write_wrapper_evidence(
                        evidence,
                        "gpu-after-restore.json",
                        {"free_vram_mib": restored_vram},
                    )
                except BaseException as error:
                    evidence_error = evidence_error or error
                restore_success = True
            except BaseException as error:
                restore_success = False
                restore_error = error
            try:
                hashes["restore_result"] = _write_wrapper_evidence(
                    evidence,
                    "comfy-restore-result.json",
                    {"restore_success": restore_success},
                )
            except BaseException as error:
                evidence_error = evidence_error or error
    receipt = _forge_wrapper_receipt(
        import_only=import_only,
        scientific_success=scientific_success,
        restore_success=restore_success,
        hashes=hashes,
        environment_receipts_sha256=_environment_receipts_sha256(run_root),
    )
    try:
        _write_wrapper_evidence(evidence, "wrapper-result.json", receipt.to_dict())
    except BaseException as error:
        evidence_error = evidence_error or error
    if restore_error is not None:
        raise ForgeRestoreError(receipt) from restore_error
    if evidence_error is not None:
        raise ForgeEvidenceError(receipt, evidence_error, scientific_error) from evidence_error
    if scientific_error is not None:
        raise scientific_error
    return receipt


def create_forge_run_root(parent: Path, run_id: str) -> Path:
    if not parent.is_absolute() or parent.is_symlink() or not parent.is_dir():
        raise ValueError("Forge run parent must be an absolute non-symlink directory")
    try:
        parsed = uuid.UUID(run_id)
    except (AttributeError, ValueError) as error:
        raise ValueError("Forge run ID must be a canonical UUID") from error
    if str(parsed) != run_id:
        raise ValueError("Forge run ID must be a canonical UUID")
    candidate = parent / run_id
    if candidate.exists() or candidate.is_symlink():
        raise ValueError("Forge run root must be fresh")
    candidate.mkdir(mode=0o700)
    candidate.chmod(0o700)
    _fsync_directory(parent)
    return candidate


__all__ = (
    "AcceptedPopulationFingerprints",
    "DecodedRGB24",
    "DevelopmentAttemptRecord",
    "DevelopmentAttemptState",
    "DevelopmentAttemptStore",
    "DevelopmentDisjointnessReceipt",
    "DevelopmentExecutionIdentity",
    "DevelopmentForwardBinding",
    "DevelopmentForwardResult",
    "DevelopmentInput",
    "DevelopmentPilotReceipt",
    "DevelopmentRehearsalReceipt",
    "DevelopmentRehearsedSeal",
    "DevelopmentRunArtifacts",
    "DevelopmentRuntimeMetrics",
    "EnvironmentReceipt",
    "ForgeEvidenceError",
    "ForgeRestoreError",
    "ForgeWrapperReceipt",
    "FrameMaterializationReceipt",
    "LoadedMaterializedBundle",
    "MaterializedBundleIdentity",
    "MaterializedBundleManifest",
    "MaterializedBundleSeal",
    "MaterializedFrameResult",
    "OperationalCommand",
    "OperationalConfiguration",
    "OperationalHandler",
    "OperationalInput",
    "OperationalStatusCode",
    "PhaseCleanupReceipt",
    "PinnedLTXRGBMaterializer",
    "PublicOperationalStatus",
    "SourceArchiveEntry",
    "SourceArchiveReceipt",
    "SourceExtractionReceipt",
    "SourcePopulationArchiveEntry",
    "SourcePopulationArchiveReceipt",
    "checked_development_output",
    "create_forge_run_root",
    "create_source_archive",
    "create_source_population_archive",
    "decode_rgb24_frames",
    "decoded_frames_sha256",
    "execute_operational_phase",
    "extract_authenticated_source_archive",
    "extract_authenticated_source_population_archive",
    "finalize_development_rehearsal",
    "finalize_materialized_bundle",
    "load_accepted_source_population",
    "load_development_input",
    "load_development_rehearsal",
    "load_environment_receipt",
    "load_materialized_bundle",
    "load_phase_cleanup_receipt",
    "load_public_operational_status",
    "load_source_archive_receipt",
    "load_source_population_archive_receipt",
    "materialize_accepted_population",
    "persist_development_disjointness",
    "persist_development_rehearsal",
    "prepare_operational_run",
    "publish_environment_receipt",
    "publish_materialized_bundle",
    "run_development_rehearsal",
    "run_persisted_forge_wrapper",
    "stage_source_population",
    "validate_development_disjointness",
    "verify_loaded_duet_module_origins",
    "verify_operational_configuration_inputs",
    "verify_phase_cleanup_receipts",
)
