"""Authenticated, deterministic disk boundary for tensor-native N=128 LTX memory.

The bundle has exactly two physical files: canonical ``manifest.json`` and a deterministic
ZIP_STORED ``tensors.npz`` containing seven non-pickle NPY arrays.  The streaming-v1 state does
not retain the independently bracketed balanced operator tensor after hashing it, so the manifest
honestly binds that audit hash and records the tensor as omitted instead of fabricating a duplicate
of the centralized operator.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import torch

from comfy_story.contracts import (
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    RawEvidencePointer,
    SpatialLocation,
    TimeFrameRange,
    tensor_sha256,
)
from comfy_story.ltx_n128_guide_memory import N128LTXGuideMemory
from comfy_story.streaming_long_context import (
    StreamingEquivalenceAudit,
    StreamingLongContextContractV1,
    StreamingLongContextMaterialization,
    StreamingRawException,
)

_FORMAT = "duet-x-n128-ltx-memory-bundle-v1"
_ARCHIVE_FORMAT = "deterministic-zip-stored-npy-v1"
_BALANCED_OMISSION_REASON = "streaming-long-context-v1-retains-only-the-bound-audit-hash"
_MANIFEST_FILENAME = "manifest.json"
_ARCHIVE_FILENAME = "tensors.npz"
_EXPECTED_FILES = frozenset({_MANIFEST_FILENAME, _ARCHIVE_FILENAME})
_TENSOR_NAMES = (
    "core_latent",
    "core_operator",
    "core_tokens",
    "exception_00_embedding",
    "exception_00_tokens",
    "exception_01_embedding",
    "exception_01_tokens",
)
_ENTRY_NAMES = tuple(f"{name}.npy" for name in _TENSOR_NAMES)
_SUPPORTED_GUIDE_SHAPES = frozenset({(1, 128, 1, 12, 12), (1, 128, 1, 14, 24)})
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
_SHA256_HEX = frozenset("0123456789abcdef")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _object(value: object, name: str, keys: frozenset[str] | None = None) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object")
    result = cast(dict[str, object], value)
    if keys is not None and set(result) != keys:
        raise ValueError(f"{name} fields changed")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _signed_integer(value: object, name: str) -> int:
    if type(value) is not int or not -(2**63) <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be a signed int64 integer")
    return value


def _regular_root(root: Path) -> Path:
    if (
        not isinstance(root, Path)
        or not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise ValueError("memory bundle root must be a configured absolute regular directory")
    return root


def _read_regular(path: Path, name: str, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{name} must be a regular non-symlink file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum:
            raise ValueError(f"{name} must be a bounded regular non-symlink file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            value = handle.read(maximum + 1)
        if len(value) != metadata.st_size:
            raise ValueError(f"{name} changed while it was read")
        return value
    finally:
        os.close(descriptor)


def _write_new(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _tensor_arrays(memory: N128LTXGuideMemory) -> dict[str, np.ndarray[Any, np.dtype[np.float32]]]:
    memory.validate()
    exceptions = memory.materialization.exceptions
    tensors = {
        "core_latent": memory.core_latent,
        "core_operator": memory.materialization.core_operator,
        "core_tokens": memory.materialization.core_tokens,
        "exception_00_embedding": exceptions[0].item.embedding,
        "exception_00_tokens": exceptions[0].raw_leaf,
        "exception_01_embedding": exceptions[1].item.embedding,
        "exception_01_tokens": exceptions[1].raw_leaf,
    }
    result: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
    for name in _TENSOR_NAMES:
        tensor = tensors[name]
        if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("bundle tensors must be finite float32")
        result[name] = tensor.detach().contiguous().cpu().numpy().copy(order="C")
    return result


def _npy_bytes(array: np.ndarray[Any, np.dtype[np.float32]]) -> bytes:
    target = io.BytesIO()
    # NumPy versions differ in NPY writer annotations; preserve its public call contract.
    write_array = cast(Callable[..., None], np.lib.format.write_array)
    write_array(target, array, version=(2, 0), allow_pickle=False)
    return target.getvalue()


def _archive(
    arrays: Mapping[str, np.ndarray[Any, np.dtype[np.float32]]],
) -> tuple[bytes, dict[str, dict[str, object]]]:
    target = io.BytesIO()
    descriptors: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True
    ) as file:
        for name, entry_name in zip(_TENSOR_NAMES, _ENTRY_NAMES, strict=True):
            array = arrays[name]
            encoded = _npy_bytes(array)
            info = zipfile.ZipInfo(entry_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            file.writestr(info, encoded, compress_type=zipfile.ZIP_STORED)
            tensor = torch.from_numpy(array)
            descriptors[name] = {
                "dtype": "float32",
                "entry": entry_name,
                "nbytes": array.nbytes,
                "npy_sha256": _sha256_bytes(encoded),
                "npy_size_bytes": len(encoded),
                "shape": list(array.shape),
                "tensor_sha256": tensor_sha256(tensor),
            }
    return target.getvalue(), descriptors


def _provenance_json(value: EvidenceProvenance) -> dict[str, object]:
    value.validate()
    return {
        "adapter_sha256": value.adapter_sha256,
        "event_type": value.event_type,
        "leaf": {
            "slot": value.leaf.slot,
            "source_id": value.leaf.source_id,
            "source_rank": value.leaf.source_rank,
        },
        "memory_contract_sha256": value.memory_contract_sha256,
        "preprocessing_sha256": value.preprocessing_sha256,
        "raw": {
            "byte_start": value.raw.byte_start,
            "byte_stop": value.raw.byte_stop,
            "content_sha256": value.raw.content_sha256,
            "locator": value.raw.locator,
        },
        "scorer_sha256": value.scorer_sha256,
        "source_registry_sha256": value.source_registry_sha256,
        "spatial": {
            "coordinate_system": value.spatial.coordinate_system,
            "coordinates_q16": list(value.spatial.coordinates_q16),
            "geometry": value.spatial.geometry,
        },
        "time": {
            "frame_start": value.time.frame_start,
            "frame_stop": value.time.frame_stop,
            "timestamp_start_ns": value.time.timestamp_start_ns,
            "timestamp_stop_ns": value.time.timestamp_stop_ns,
        },
        "vae_sha256": value.vae_sha256,
    }


def _audit_json(value: StreamingEquivalenceAudit) -> dict[str, object]:
    value.validate()
    return {
        "atol": value.atol,
        "balanced_operator_sha256": value.balanced_operator_sha256,
        "centralized_operator_sha256": value.centralized_operator_sha256,
        "equivalent": value.equivalent,
        "excluded_leaf_count": value.excluded_leaf_count,
        "format": value.format,
        "history_items": value.history_items,
        "max_abs_error": value.max_abs_error,
        "ordinary_leaf_count": value.ordinary_leaf_count,
        "rtol": value.rtol,
    }


def _contract_json(value: StreamingLongContextContractV1) -> dict[str, object]:
    value.validate()
    return {
        "dense_accumulation_dtype": "float32",
        "equivalence_atol": value.equivalence_atol,
        "equivalence_rtol": value.equivalence_rtol,
        "exception_capacity": value.sparse.capacity,
        "fingerprint": value.fingerprint(),
        "format": value.format,
        "history_items": value.history_items,
        "latent_channels": value.latent_channels,
        "source_contract_sha256": value.source_contract_sha256,
        "sparse_fingerprint": value.sparse.fingerprint(),
    }


def _manifest(
    memory: N128LTXGuideMemory,
    archive: bytes,
    tensors: dict[str, dict[str, object]],
) -> dict[str, object]:
    materialization = memory.materialization
    exceptions = materialization.exceptions
    return {
        "audit": _audit_json(materialization.audit),
        "balanced_operator": {
            "reason": _BALANCED_OMISSION_REASON,
            "sha256": materialization.audit.balanced_operator_sha256,
            "storage": "omitted",
        },
        "checkpoint_sha256": memory.checkpoint_sha256,
        "contract": _contract_json(materialization.contract),
        "core_item_ids": list(memory.core_item_ids),
        "core_modules_sha256": memory.core_modules_sha256,
        "exceptions": [
            {
                "embedding_tensor_sha256": tensor_sha256(exception.item.embedding),
                "item_id": exception.item.item_id,
                "pinned": exception.item.pinned,
                "provenance": _provenance_json(exception.item.provenance),
                "raw_leaf_sha256": exception.raw_leaf_sha256,
                "salience_q": exception.item.salience_q,
            }
            for exception in exceptions
        ],
        "format": _FORMAT,
        "tensor_archive": {
            "filename": _ARCHIVE_FILENAME,
            "format": _ARCHIVE_FORMAT,
            "sha256": _sha256_bytes(archive),
            "size_bytes": len(archive),
        },
        "tensors": tensors,
    }


@dataclass(frozen=True, slots=True)
class N128LTXMemoryBundleReceipt:
    """Canonical identity returned only after publish or authenticated load succeeds."""

    bundle_id: str
    manifest_bytes: bytes
    tensor_archive_sha256: str
    checkpoint_sha256: str
    core_modules_sha256: str

    def validate(self) -> Self:
        _sha256(self.bundle_id, "bundle_id")
        if _sha256_bytes(self.manifest_bytes) != self.bundle_id:
            raise ValueError("bundle receipt manifest does not match bundle_id")
        _sha256(self.tensor_archive_sha256, "tensor_archive_sha256")
        _sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _sha256(self.core_modules_sha256, "core_modules_sha256")
        return self

    def to_json(self) -> str:
        self.validate()
        return _canonical_json(
            {
                "bundle_id": self.bundle_id,
                "checkpoint_sha256": self.checkpoint_sha256,
                "core_modules_sha256": self.core_modules_sha256,
                "tensor_archive_sha256": self.tensor_archive_sha256,
            }
        ).decode()


def publish_n128_ltx_memory_bundle(
    root: Path, memory: N128LTXGuideMemory
) -> N128LTXMemoryBundleReceipt:
    """Atomically publish a fresh deterministic content-addressed two-file bundle."""
    bundle_root = _regular_root(root)
    arrays = _tensor_arrays(memory)
    archive, descriptors = _archive(arrays)
    if len(archive) > _MAX_ARCHIVE_BYTES:
        raise ValueError("tensor archive exceeds the fixed bundle size ceiling")
    manifest_bytes = _canonical_json(_manifest(memory, archive, descriptors))
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError("bundle manifest exceeds the fixed size ceiling")
    bundle_id = _sha256_bytes(manifest_bytes)
    destination = bundle_root / bundle_id
    if destination.exists() or destination.is_symlink():
        raise ValueError("memory bundle destination already exists")
    temporary = Path(tempfile.mkdtemp(prefix=".duetx-n128-", dir=bundle_root))
    temporary.chmod(0o700)
    try:
        _write_new(temporary / _MANIFEST_FILENAME, manifest_bytes)
        _write_new(temporary / _ARCHIVE_FILENAME, archive)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if destination.exists() or destination.is_symlink():
            raise ValueError("memory bundle destination already exists")
        temporary.rename(destination)
        root_descriptor = os.open(bundle_root, os.O_RDONLY)
        try:
            os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return N128LTXMemoryBundleReceipt(
        bundle_id,
        manifest_bytes,
        _sha256_bytes(archive),
        memory.checkpoint_sha256,
        memory.core_modules_sha256,
    ).validate()


def _parse_canonical_manifest(value: bytes, bundle_id: str) -> dict[str, object]:
    if _sha256_bytes(value) != bundle_id:
        raise ValueError("canonical manifest SHA-256 does not match trusted bundle_id")
    try:
        decoded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("bundle manifest must be canonical JSON") from error
    manifest = _object(
        decoded,
        "manifest",
        frozenset(
            {
                "audit",
                "balanced_operator",
                "checkpoint_sha256",
                "contract",
                "core_item_ids",
                "core_modules_sha256",
                "exceptions",
                "format",
                "tensor_archive",
                "tensors",
            }
        ),
    )
    if _canonical_json(manifest) != value or manifest["format"] != _FORMAT:
        raise ValueError("bundle manifest must use the canonical versioned format")
    return manifest


def _contract_from_json(value: object) -> StreamingLongContextContractV1:
    data = _object(
        value,
        "contract",
        frozenset(
            {
                "dense_accumulation_dtype",
                "equivalence_atol",
                "equivalence_rtol",
                "exception_capacity",
                "fingerprint",
                "format",
                "history_items",
                "latent_channels",
                "source_contract_sha256",
                "sparse_fingerprint",
            }
        ),
    )
    if (
        data["dense_accumulation_dtype"] != "float32"
        or data["history_items"] != 128
        or data["exception_capacity"] != 2
        or data["latent_channels"] != 128
    ):
        raise ValueError("bundle contract must be exact N=128/K=2/C=128 FP32")
    contract = replace(
        StreamingLongContextContractV1.default(
            source_contract_sha256=_sha256(
                data["source_contract_sha256"], "source_contract_sha256"
            ),
            history_items=128,
            exception_capacity=2,
            latent_channels=128,
        ),
        equivalence_atol=_float(data["equivalence_atol"], "equivalence_atol"),
        equivalence_rtol=_float(data["equivalence_rtol"], "equivalence_rtol"),
    ).validate()
    if (
        data["format"] != contract.format
        or data["fingerprint"] != contract.fingerprint()
        or data["sparse_fingerprint"] != contract.sparse.fingerprint()
    ):
        raise ValueError("bundle contract fingerprint binding changed")
    return contract


def _audit_from_json(value: object) -> StreamingEquivalenceAudit:
    data = _object(
        value,
        "audit",
        frozenset(
            {
                "atol",
                "balanced_operator_sha256",
                "centralized_operator_sha256",
                "equivalent",
                "excluded_leaf_count",
                "format",
                "history_items",
                "max_abs_error",
                "ordinary_leaf_count",
                "rtol",
            }
        ),
    )
    if type(data["equivalent"]) is not bool:
        raise ValueError("audit equivalent flag must be a bool")
    return StreamingEquivalenceAudit(
        cast(str, data["format"]),
        _integer(data["history_items"], "audit.history_items", minimum=1),
        _integer(data["ordinary_leaf_count"], "audit.ordinary_leaf_count", minimum=1),
        _integer(data["excluded_leaf_count"], "audit.excluded_leaf_count"),
        _sha256(data["centralized_operator_sha256"], "centralized_operator_sha256"),
        _sha256(data["balanced_operator_sha256"], "balanced_operator_sha256"),
        _float(data["atol"], "audit.atol"),
        _float(data["rtol"], "audit.rtol"),
        _float(data["max_abs_error"], "audit.max_abs_error"),
        data["equivalent"],
    ).validate()


def _provenance_from_json(value: object) -> EvidenceProvenance:
    data = _object(
        value,
        "provenance",
        frozenset(
            {
                "adapter_sha256",
                "event_type",
                "leaf",
                "memory_contract_sha256",
                "preprocessing_sha256",
                "raw",
                "scorer_sha256",
                "source_registry_sha256",
                "spatial",
                "time",
                "vae_sha256",
            }
        ),
    )
    leaf = _object(data["leaf"], "provenance.leaf", frozenset({"slot", "source_id", "source_rank"}))
    time = _object(
        data["time"],
        "provenance.time",
        frozenset({"frame_start", "frame_stop", "timestamp_start_ns", "timestamp_stop_ns"}),
    )
    spatial = _object(
        data["spatial"],
        "provenance.spatial",
        frozenset({"coordinate_system", "coordinates_q16", "geometry"}),
    )
    raw = _object(
        data["raw"],
        "provenance.raw",
        frozenset({"byte_start", "byte_stop", "content_sha256", "locator"}),
    )
    coordinates = spatial["coordinates_q16"]
    if type(coordinates) is not list or any(type(item) is not int for item in coordinates):
        raise ValueError("provenance spatial coordinates must be integer JSON values")
    return EvidenceProvenance(
        LeafKey(
            cast(str, leaf["source_id"]),
            _integer(leaf["source_rank"], "source_rank"),
            _integer(leaf["slot"], "slot"),
        ),
        TimeFrameRange(
            _integer(time["timestamp_start_ns"], "timestamp_start_ns"),
            _integer(time["timestamp_stop_ns"], "timestamp_stop_ns", minimum=1),
            _integer(time["frame_start"], "frame_start"),
            _integer(time["frame_stop"], "frame_stop", minimum=1),
        ),
        SpatialLocation(
            cast(str, spatial["coordinate_system"]),
            cast(str, spatial["geometry"]),
            tuple(coordinates),
        ),
        cast(str, data["event_type"]),
        RawEvidencePointer(
            cast(str, raw["locator"]),
            _sha256(raw["content_sha256"], "raw.content_sha256"),
            _integer(raw["byte_start"], "raw.byte_start"),
            _integer(raw["byte_stop"], "raw.byte_stop", minimum=1),
        ),
        _sha256(data["source_registry_sha256"], "source_registry_sha256"),
        _sha256(data["preprocessing_sha256"], "preprocessing_sha256"),
        _sha256(data["vae_sha256"], "vae_sha256"),
        _sha256(data["adapter_sha256"], "adapter_sha256"),
        _sha256(data["scorer_sha256"], "scorer_sha256"),
        _sha256(data["memory_contract_sha256"], "memory_contract_sha256"),
    ).validate()


def _load_arrays(archive_bytes: bytes, descriptors_value: object) -> dict[str, torch.Tensor]:
    descriptors = _object(descriptors_value, "tensors", frozenset(_TENSOR_NAMES))
    result: dict[str, torch.Tensor] = {}
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes), "r")
    except zipfile.BadZipFile as error:
        raise ValueError("tensor archive is not a valid deterministic ZIP") from error
    with archive:
        infos = archive.infolist()
        if [info.filename for info in infos] != list(_ENTRY_NAMES) or len(
            {info.filename for info in infos}
        ) != len(_ENTRY_NAMES):
            raise ValueError("tensor archive entry roster or order changed")
        if any(
            info.compress_type != zipfile.ZIP_STORED
            or info.flag_bits & 0x1
            or info.is_dir()
            or info.filename != Path(info.filename).name
            for info in infos
        ):
            raise ValueError("tensor archive entries must be unencrypted flat ZIP_STORED files")
        for name, info in zip(_TENSOR_NAMES, infos, strict=True):
            descriptor = _object(
                descriptors[name],
                f"tensors.{name}",
                frozenset(
                    {
                        "dtype",
                        "entry",
                        "nbytes",
                        "npy_sha256",
                        "npy_size_bytes",
                        "shape",
                        "tensor_sha256",
                    }
                ),
            )
            if descriptor["entry"] != info.filename or descriptor["dtype"] != "float32":
                raise ValueError("tensor descriptor entry or dtype changed")
            npy_size = _integer(descriptor["npy_size_bytes"], "npy_size_bytes", minimum=1)
            if info.file_size != npy_size or info.compress_size != npy_size:
                raise ValueError("tensor NPY entry size does not match its manifest")
            encoded = archive.read(info)
            if _sha256_bytes(encoded) != _sha256(descriptor["npy_sha256"], "npy_sha256"):
                raise ValueError("tensor NPY entry SHA-256 changed")
            try:
                array = np.load(io.BytesIO(encoded), allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ValueError("tensor NPY entry is invalid or requires pickle") from error
            if not isinstance(array, np.ndarray) or array.dtype != np.dtype("<f4"):
                raise ValueError("bundle tensor must be a non-object little-endian float32 array")
            shape = descriptor["shape"]
            if type(shape) is not list or any(type(dimension) is not int for dimension in shape):
                raise ValueError("tensor descriptor shape must contain integers")
            if (
                list(array.shape) != shape
                or array.nbytes != _integer(descriptor["nbytes"], "tensor nbytes", minimum=1)
                or not array.flags.c_contiguous
                or not bool(np.isfinite(array).all())
            ):
                raise ValueError("bundle tensor shape, size, layout, or finiteness changed")
            tensor = torch.from_numpy(array.copy(order="C"))
            if tensor_sha256(tensor) != _sha256(descriptor["tensor_sha256"], "tensor_sha256"):
                raise ValueError("bundle tensor SHA-256 changed")
            result[name] = tensor
    return result


def _validate_tensor_shapes(tensors: Mapping[str, torch.Tensor]) -> None:
    latent_shape = tuple(tensors["core_latent"].shape)
    if latent_shape not in _SUPPORTED_GUIDE_SHAPES:
        raise ValueError("core latent shape is outside the fixed supported N=128 profiles")
    _, channels, frames, height, width = latent_shape
    token_count = frames * height * width
    expected = {
        "core_latent": latent_shape,
        "core_operator": (1, token_count, 16, 16),
        "core_tokens": (1, token_count, channels),
        "exception_00_embedding": (128,),
        "exception_00_tokens": (1, token_count, channels),
        "exception_01_embedding": (128,),
        "exception_01_tokens": (1, token_count, channels),
    }
    if any(tuple(tensors[name].shape) != shape for name, shape in expected.items()):
        raise ValueError("bundle tensor roster does not match exact N=128/K=2/C=128/O=16 shapes")


def load_n128_ltx_memory_bundle(
    root: Path,
    bundle_id: str,
    *,
    device: torch.device,
) -> tuple[N128LTXGuideMemory, N128LTXMemoryBundleReceipt]:
    """Load one trusted content ID from a configured root after complete authentication."""
    bundle_root = _regular_root(root)
    trusted_id = _sha256(bundle_id, "bundle_id")
    if not isinstance(device, torch.device):
        raise ValueError("bundle target device must be a torch.device")
    directory = bundle_root / trusted_id
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("memory bundle must be a regular non-symlink directory")
    children = tuple(directory.iterdir())
    if {child.name for child in children} != _EXPECTED_FILES or len(children) != 2:
        raise ValueError("memory bundle must contain exactly manifest.json and tensors.npz")
    manifest_bytes = _read_regular(
        directory / _MANIFEST_FILENAME, "bundle manifest", _MAX_MANIFEST_BYTES
    )
    manifest = _parse_canonical_manifest(manifest_bytes, trusted_id)
    archive_meta = _object(
        manifest["tensor_archive"],
        "tensor_archive",
        frozenset({"filename", "format", "sha256", "size_bytes"}),
    )
    if archive_meta["filename"] != _ARCHIVE_FILENAME or archive_meta["format"] != _ARCHIVE_FORMAT:
        raise ValueError("tensor archive identity changed")
    expected_archive_size = _integer(archive_meta["size_bytes"], "tensor archive size", minimum=1)
    if expected_archive_size > _MAX_ARCHIVE_BYTES:
        raise ValueError("tensor archive exceeds the fixed bundle size ceiling")
    archive_bytes = _read_regular(
        directory / _ARCHIVE_FILENAME, "tensor archive", _MAX_ARCHIVE_BYTES
    )
    archive_sha256 = _sha256(archive_meta["sha256"], "tensor archive SHA-256")
    if (
        len(archive_bytes) != expected_archive_size
        or _sha256_bytes(archive_bytes) != archive_sha256
    ):
        raise ValueError("raw tensor archive SHA-256 or size changed")

    contract = _contract_from_json(manifest["contract"])
    audit = _audit_from_json(manifest["audit"])
    balanced = _object(
        manifest["balanced_operator"],
        "balanced_operator",
        frozenset({"reason", "sha256", "storage"}),
    )
    if balanced != {
        "reason": _BALANCED_OMISSION_REASON,
        "sha256": audit.balanced_operator_sha256,
        "storage": "omitted",
    }:
        raise ValueError("balanced operator omission must match the bound audit hash")
    tensors = _load_arrays(archive_bytes, manifest["tensors"])
    _validate_tensor_shapes(tensors)
    if tensor_sha256(tensors["core_operator"]) != audit.centralized_operator_sha256:
        raise ValueError("core operator does not match the streaming audit binding")

    exceptions_value = manifest["exceptions"]
    if type(exceptions_value) is not list or len(exceptions_value) != 2:
        raise ValueError("bundle must contain exactly two exception records")
    exceptions: list[StreamingRawException] = []
    for ordinal, raw_value in enumerate(exceptions_value):
        data = _object(
            raw_value,
            f"exceptions[{ordinal}]",
            frozenset(
                {
                    "embedding_tensor_sha256",
                    "item_id",
                    "pinned",
                    "provenance",
                    "raw_leaf_sha256",
                    "salience_q",
                }
            ),
        )
        if data["pinned"] is not True or not isinstance(data["item_id"], str):
            raise ValueError("bundle exception must be explicitly hard-pinned and identified")
        embedding = tensors[f"exception_{ordinal:02d}_embedding"]
        tokens = tensors[f"exception_{ordinal:02d}_tokens"]
        if tensor_sha256(embedding) != _sha256(
            data["embedding_tensor_sha256"], "embedding_tensor_sha256"
        ):
            raise ValueError("exception embedding does not match its manifest")
        if tensor_sha256(tokens) != _sha256(data["raw_leaf_sha256"], "raw_leaf_sha256"):
            raise ValueError("raw exception tensor does not match its manifest")
        item = ExceptionItem(
            data["item_id"],
            _provenance_from_json(data["provenance"]),
            embedding,
            _signed_integer(data["salience_q"], "salience_q"),
            pinned=True,
        )
        item.validate(contract.sparse)
        exceptions.append(StreamingRawException(item, tokens))
    core_tokens = tensors["core_tokens"].to(device=device)
    core_operator = tensors["core_operator"].to(device=device)
    materialization = StreamingLongContextMaterialization(
        contract,
        core_tokens,
        core_operator,
        tuple(exceptions),
        audit,
    )
    core_item_ids_value = manifest["core_item_ids"]
    if type(core_item_ids_value) is not list or any(
        not isinstance(item_id, str) for item_id in core_item_ids_value
    ):
        raise ValueError("core item IDs must be a JSON string list")
    checkpoint_sha256 = _sha256(manifest["checkpoint_sha256"], "checkpoint_sha256")
    core_modules_sha256 = _sha256(manifest["core_modules_sha256"], "core_modules_sha256")
    memory = N128LTXGuideMemory(
        materialization,
        tensors["core_latent"].to(device=device),
        tuple(core_item_ids_value),
        checkpoint_sha256,
        core_modules_sha256,
    )
    receipt = N128LTXMemoryBundleReceipt(
        trusted_id,
        manifest_bytes,
        archive_sha256,
        checkpoint_sha256,
        core_modules_sha256,
    ).validate()
    return memory, receipt


__all__ = (
    "N128LTXMemoryBundleReceipt",
    "load_n128_ltx_memory_bundle",
    "publish_n128_ltx_memory_bundle",
)
