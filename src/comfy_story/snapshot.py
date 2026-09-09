"""Strict, atomic runtime snapshots for the Duet-X Decision 1 product tree."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from comfy_story.cache import AuthoritativeLeaf, DuetXProductTree
from comfy_story.contracts import (
    SALIENCE_Q_MAX,
    SALIENCE_Q_MIN,
    Coverage,
    DuetXContract,
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    RawEvidencePointer,
    SparseMemoryContract,
    SpatialLocation,
    TimeFrameRange,
    tensor_sha256,
)

_SNAPSHOT_FORMAT = "duet-x-runtime-snapshot-v1"
_SHA256_HEX = frozenset("0123456789abcdef")
_TOP_LEVEL_KEYS = frozenset(
    {
        "format",
        "contract",
        "contract_fingerprint",
        "model_checkpoint_sha256",
        "tree",
        "root",
        "tensor_specs",
        "tensors",
        "manifest_sha256",
    }
)
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _mapping(value: object, name: str, keys: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a plain mapping")
    raw = value
    if any(type(key) is not str for key in raw):
        raise ValueError(f"{name} keys must be strings")
    result = cast(dict[str, Any], raw)
    if set(result) != keys:
        raise ValueError(f"{name} fields do not match the snapshot schema")
    return result


def _open_mapping(value: object, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a plain mapping")
    raw = value
    if any(type(key) is not str for key in raw):
        raise ValueError(f"{name} keys must be strings")
    return cast(dict[str, Any], raw)


def _list(value: object, name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a list")
    return value


def _string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = f"[{minimum}, {maximum}]" if maximum is not None else f"[{minimum}, infinity)"
        raise ValueError(f"{name} must be an integer in {bounds}")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _optional_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name)


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise ValueError(f"unsupported snapshot tensor dtype: {dtype}")
    return name


def _dtype(value: object, name: str) -> torch.dtype:
    dtype_name = _string(value, name)
    try:
        return _DTYPES[dtype_name]
    except KeyError as error:
        raise ValueError(f"{name} is not a supported floating dtype") from error


def _canonical_manifest_value(value: object, name: str = "manifest") -> object:
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is list:
        return [
            _canonical_manifest_value(item, f"{name}[{index}]") for index, item in enumerate(value)
        ]
    if type(value) is dict:
        mapping = _open_mapping(value, name)
        return {
            key: _canonical_manifest_value(item, f"{name}.{key}") for key, item in mapping.items()
        }
    raise ValueError(f"{name} contains a non-primitive value")


def _manifest_sha256(payload: Mapping[str, Any]) -> str:
    manifest = {
        key: _canonical_manifest_value(value, key)
        for key, value in payload.items()
        if key not in {"manifest_sha256", "tensors"}
    }
    encoded = json.dumps(
        manifest,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contract_data(contract: DuetXContract) -> dict[str, object]:
    contract.validate()
    return {
        "format": contract.format,
        "decision1_fingerprint": contract.decision1_fingerprint,
        "sparse": {
            "capacity": contract.sparse.capacity,
            "embedding_dim": contract.sparse.embedding_dim,
            "embedding_dtype": _dtype_name(contract.sparse.embedding_dtype),
            "max_pinned": contract.sparse.max_pinned,
        },
        "history_items": contract.history_items,
        "latent_channels": contract.latent_channels,
        "dense_accumulation_dtype": _dtype_name(contract.dense_accumulation_dtype),
    }


def _contract_from_data(value: object) -> DuetXContract:
    data = _mapping(
        value,
        "contract",
        frozenset(
            {
                "format",
                "decision1_fingerprint",
                "sparse",
                "history_items",
                "latent_channels",
                "dense_accumulation_dtype",
            }
        ),
    )
    sparse_data = _mapping(
        data["sparse"],
        "contract.sparse",
        frozenset({"capacity", "embedding_dim", "embedding_dtype", "max_pinned"}),
    )
    contract = DuetXContract(
        format=_string(data["format"], "contract.format"),
        decision1_fingerprint=_sha256(
            data["decision1_fingerprint"], "contract.decision1_fingerprint"
        ),
        sparse=SparseMemoryContract(
            capacity=_integer(sparse_data["capacity"], "contract.sparse.capacity", minimum=1),
            embedding_dim=_integer(
                sparse_data["embedding_dim"], "contract.sparse.embedding_dim", minimum=1
            ),
            embedding_dtype=_dtype(
                sparse_data["embedding_dtype"], "contract.sparse.embedding_dtype"
            ),
            max_pinned=_optional_integer(sparse_data["max_pinned"], "contract.sparse.max_pinned"),
        ),
        history_items=_integer(data["history_items"], "contract.history_items", minimum=1),
        latent_channels=_integer(data["latent_channels"], "contract.latent_channels", minimum=1),
        dense_accumulation_dtype=_dtype(
            data["dense_accumulation_dtype"], "contract.dense_accumulation_dtype"
        ),
    )
    return contract.validate()


def _leaf_key_data(key: LeafKey) -> dict[str, object]:
    key.validate()
    return {"source_id": key.source_id, "source_rank": key.source_rank, "slot": key.slot}


def _leaf_key_from_data(value: object, name: str) -> LeafKey:
    data = _mapping(value, name, frozenset({"source_id", "source_rank", "slot"}))
    return LeafKey(
        _string(data["source_id"], f"{name}.source_id"),
        _integer(data["source_rank"], f"{name}.source_rank"),
        _integer(data["slot"], f"{name}.slot"),
    ).validate()


def _provenance_data(provenance: EvidenceProvenance) -> dict[str, object]:
    provenance.validate()
    return {
        "leaf": _leaf_key_data(provenance.leaf),
        "time": {
            "timestamp_start_ns": provenance.time.timestamp_start_ns,
            "timestamp_stop_ns": provenance.time.timestamp_stop_ns,
            "frame_start": provenance.time.frame_start,
            "frame_stop": provenance.time.frame_stop,
        },
        "spatial": {
            "coordinate_system": provenance.spatial.coordinate_system,
            "geometry": provenance.spatial.geometry,
            "coordinates_q16": list(provenance.spatial.coordinates_q16),
        },
        "event_type": provenance.event_type,
        "raw": {
            "locator": provenance.raw.locator,
            "content_sha256": provenance.raw.content_sha256,
            "byte_start": provenance.raw.byte_start,
            "byte_stop": provenance.raw.byte_stop,
        },
        "source_registry_sha256": provenance.source_registry_sha256,
        "preprocessing_sha256": provenance.preprocessing_sha256,
        "vae_sha256": provenance.vae_sha256,
        "adapter_sha256": provenance.adapter_sha256,
        "scorer_sha256": provenance.scorer_sha256,
        "memory_contract_sha256": provenance.memory_contract_sha256,
    }


def _provenance_from_data(value: object, name: str) -> EvidenceProvenance:
    data = _mapping(
        value,
        name,
        frozenset(
            {
                "leaf",
                "time",
                "spatial",
                "event_type",
                "raw",
                "source_registry_sha256",
                "preprocessing_sha256",
                "vae_sha256",
                "adapter_sha256",
                "scorer_sha256",
                "memory_contract_sha256",
            }
        ),
    )
    time = _mapping(
        data["time"],
        f"{name}.time",
        frozenset({"timestamp_start_ns", "timestamp_stop_ns", "frame_start", "frame_stop"}),
    )
    spatial = _mapping(
        data["spatial"],
        f"{name}.spatial",
        frozenset({"coordinate_system", "geometry", "coordinates_q16"}),
    )
    coordinates = _list(spatial["coordinates_q16"], f"{name}.spatial.coordinates_q16")
    raw = _mapping(
        data["raw"],
        f"{name}.raw",
        frozenset({"locator", "content_sha256", "byte_start", "byte_stop"}),
    )
    provenance = EvidenceProvenance(
        leaf=_leaf_key_from_data(data["leaf"], f"{name}.leaf"),
        time=TimeFrameRange(
            _integer(time["timestamp_start_ns"], f"{name}.time.timestamp_start_ns"),
            _integer(time["timestamp_stop_ns"], f"{name}.time.timestamp_stop_ns"),
            _integer(time["frame_start"], f"{name}.time.frame_start"),
            _integer(time["frame_stop"], f"{name}.time.frame_stop"),
        ),
        spatial=SpatialLocation(
            _string(spatial["coordinate_system"], f"{name}.spatial.coordinate_system"),
            _string(spatial["geometry"], f"{name}.spatial.geometry"),
            tuple(
                _integer(coordinate, f"{name}.spatial.coordinates_q16[{index}]")
                for index, coordinate in enumerate(coordinates)
            ),
        ),
        event_type=_string(data["event_type"], f"{name}.event_type"),
        raw=RawEvidencePointer(
            _string(raw["locator"], f"{name}.raw.locator"),
            _sha256(raw["content_sha256"], f"{name}.raw.content_sha256"),
            _integer(raw["byte_start"], f"{name}.raw.byte_start"),
            _integer(raw["byte_stop"], f"{name}.raw.byte_stop"),
        ),
        source_registry_sha256=_sha256(
            data["source_registry_sha256"], f"{name}.source_registry_sha256"
        ),
        preprocessing_sha256=_sha256(data["preprocessing_sha256"], f"{name}.preprocessing_sha256"),
        vae_sha256=_sha256(data["vae_sha256"], f"{name}.vae_sha256"),
        adapter_sha256=_sha256(data["adapter_sha256"], f"{name}.adapter_sha256"),
        scorer_sha256=_sha256(data["scorer_sha256"], f"{name}.scorer_sha256"),
        memory_contract_sha256=_sha256(
            data["memory_contract_sha256"], f"{name}.memory_contract_sha256"
        ),
    )
    return provenance.validate()


def _add_tensor(
    tensors: dict[str, torch.Tensor],
    specs: dict[str, dict[str, object]],
    name: str,
    tensor: torch.Tensor,
) -> None:
    if type(tensor) is not torch.Tensor or tensor.layout != torch.strided:
        raise ValueError(f"{name} must be a plain strided tensor")
    canonical = tensor.detach().contiguous().cpu().clone()
    if not bool(torch.isfinite(canonical).all().item()):
        raise ValueError(f"{name} must be finite")
    tensors[name] = canonical
    specs[name] = {
        "dtype": _dtype_name(canonical.dtype),
        "shape": list(canonical.shape),
        "sha256": tensor_sha256(canonical),
    }


def _item_data(
    item: ExceptionItem,
    tensor_name: str,
    tensors: dict[str, torch.Tensor],
    specs: dict[str, dict[str, object]],
    contract: SparseMemoryContract,
) -> dict[str, object]:
    item.validate(contract)
    _add_tensor(tensors, specs, tensor_name, item.embedding)
    return {
        "item_id": item.item_id,
        "provenance": _provenance_data(item.provenance),
        "embedding_tensor": tensor_name,
        "salience_q": item.salience_q,
        "pinned": item.pinned,
        "fingerprint": item.fingerprint(),
    }


def _root_data(tree: DuetXProductTree) -> dict[str, object]:
    root = tree.root
    return {
        "coverage": {
            "start_slot": root.coverage.start_slot,
            "stop_slot": root.coverage.stop_slot,
        },
        "dense_operator_sha256": tensor_sha256(root.dense_operator),
        "exception_item_ids": [item.item_id for item in root.exceptions],
        "exception_fingerprints": [item.fingerprint() for item in root.exceptions],
    }


def _build_payload(tree: DuetXProductTree, model_checkpoint_sha256: str) -> dict[str, Any]:
    if not isinstance(tree, DuetXProductTree):
        raise ValueError("tree must be a DuetXProductTree")
    tree.contract.validate()
    model_digest = _sha256(model_checkpoint_sha256, "model_checkpoint_sha256")
    cold_root = tree.full_rebuild_root()
    if tensor_sha256(cold_root.dense_operator) != tensor_sha256(tree.root.dense_operator):
        raise ValueError("cached root dense operator does not match a cold rebuild")
    if [item.fingerprint() for item in cold_root.exceptions] != [
        item.fingerprint() for item in tree.root.exceptions
    ]:
        raise ValueError("cached root exceptions do not match a cold rebuild")

    tensors: dict[str, torch.Tensor] = {}
    specs: dict[str, dict[str, object]] = {}
    leaves: list[dict[str, object] | None] = []
    for slot, leaf in enumerate(tree.leaves):
        if leaf is None:
            leaves.append(None)
            continue
        dense_name = f"leaf.{slot}.dense_operator"
        _add_tensor(tensors, specs, dense_name, leaf.dense_operator)
        candidates = [
            _item_data(
                item,
                f"leaf.{slot}.candidate.{index}.embedding",
                tensors,
                specs,
                tree.contract.sparse,
            )
            for index, item in enumerate(leaf.candidates)
        ]
        leaves.append(
            {
                "key": _leaf_key_data(leaf.key),
                "source_fingerprint": leaf.source_fingerprint,
                "revision": leaf.revision,
                "dense_operator_tensor": dense_name,
                "candidates": candidates,
            }
        )
    payload: dict[str, Any] = {
        "format": _SNAPSHOT_FORMAT,
        "contract": _contract_data(tree.contract),
        "contract_fingerprint": tree.contract.fingerprint(),
        "model_checkpoint_sha256": model_digest,
        "tree": {
            "dense_steps": tree.root.dense_operator.shape[1],
            "operator_size": tree.root.dense_operator.shape[-1],
            "leaf_count": tree.leaf_count,
            "leaves": leaves,
        },
        "root": _root_data(tree),
        "tensor_specs": specs,
        "tensors": tensors,
    }
    payload["manifest_sha256"] = _manifest_sha256(payload)
    return payload


def save_runtime_snapshot(
    path: Path,
    tree: DuetXProductTree,
    *,
    model_checkpoint_sha256: str,
) -> Path:
    """Atomically persist validated primitive metadata and hashed CPU tensors."""
    destination = Path(path)
    payload = _build_payload(tree, model_checkpoint_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def _expected_tensor_names(leaf_data: list[Any]) -> frozenset[str]:
    names: list[str] = []
    occupied_slots: set[int] = set()
    leaf_keys = frozenset(
        {"key", "source_fingerprint", "revision", "dense_operator_tensor", "candidates"}
    )
    item_keys = frozenset(
        {
            "item_id",
            "provenance",
            "embedding_tensor",
            "salience_q",
            "pinned",
            "fingerprint",
        }
    )
    for position, value in enumerate(leaf_data):
        if value is None:
            continue
        leaf = _mapping(value, f"tree.leaves[{position}]", leaf_keys)
        key = _leaf_key_from_data(leaf["key"], f"tree.leaves[{position}].key")
        if key.slot >= len(leaf_data):
            raise ValueError("snapshot leaf slot is out of range")
        if key.slot in occupied_slots:
            raise ValueError("snapshot contains a duplicate occupied leaf slot")
        occupied_slots.add(key.slot)
        if key.slot != position:
            raise ValueError("snapshot leaf slot does not match its fixed list position")
        dense_name = f"leaf.{position}.dense_operator"
        if leaf["dense_operator_tensor"] != dense_name:
            raise ValueError("snapshot dense tensor reference is not canonical")
        names.append(dense_name)
        candidates = _list(leaf["candidates"], f"tree.leaves[{position}].candidates")
        for candidate_position, candidate_value in enumerate(candidates):
            candidate = _mapping(
                candidate_value,
                f"tree.leaves[{position}].candidates[{candidate_position}]",
                item_keys,
            )
            embedding_name = f"leaf.{position}.candidate.{candidate_position}.embedding"
            if candidate["embedding_tensor"] != embedding_name:
                raise ValueError("snapshot embedding tensor reference is not canonical")
            names.append(embedding_name)
    if len(names) != len(set(names)):
        raise ValueError("snapshot tensor references must be unique")
    return frozenset(names)


def _validated_tensors(
    payload: Mapping[str, Any], expected_names: frozenset[str]
) -> dict[str, torch.Tensor]:
    specs = _open_mapping(payload["tensor_specs"], "tensor_specs")
    tensors_data = _open_mapping(payload["tensors"], "tensors")
    if set(specs) != expected_names:
        raise ValueError("snapshot tensor_specs do not match canonical tensor references")
    if set(tensors_data) != expected_names:
        raise ValueError("snapshot tensor payloads do not match canonical tensor references")
    tensors: dict[str, torch.Tensor] = {}
    for name, raw_spec in specs.items():
        spec = _mapping(
            raw_spec,
            f"tensor_specs.{name}",
            frozenset({"dtype", "shape", "sha256"}),
        )
        tensor = tensors_data[name]
        if type(tensor) is not torch.Tensor or tensor.layout != torch.strided:
            raise ValueError(f"snapshot tensor {name!r} must be a plain strided tensor")
        shape_data = _list(spec["shape"], f"tensor_specs.{name}.shape")
        shape = tuple(
            _integer(item, f"tensor_specs.{name}.shape[{index}]")
            for index, item in enumerate(shape_data)
        )
        expected_dtype = _dtype(spec["dtype"], f"tensor_specs.{name}.dtype")
        if tensor.dtype != expected_dtype:
            raise ValueError(f"snapshot tensor {name!r} dtype does not match its record")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"snapshot tensor {name!r} shape does not match its record")
        if tensor.device.type != "cpu":
            raise ValueError(f"snapshot tensor {name!r} must load on CPU")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"snapshot tensor {name!r} must be finite")
        expected_hash = _sha256(spec["sha256"], f"tensor_specs.{name}.sha256")
        if tensor_sha256(tensor) != expected_hash:
            raise ValueError(f"snapshot tensor {name!r} hash mismatch")
        tensors[name] = tensor.detach().contiguous().cpu()
    return tensors


def _tensor_reference(
    value: object, tensors: Mapping[str, torch.Tensor], name: str
) -> torch.Tensor:
    reference = _string(value, name)
    try:
        return tensors[reference]
    except KeyError as error:
        raise ValueError(f"{name} refers to an unknown snapshot tensor") from error


def _item_from_data(
    value: object,
    tensors: Mapping[str, torch.Tensor],
    contract: SparseMemoryContract,
    name: str,
) -> ExceptionItem:
    data = _mapping(
        value,
        name,
        frozenset(
            {
                "item_id",
                "provenance",
                "embedding_tensor",
                "salience_q",
                "pinned",
                "fingerprint",
            }
        ),
    )
    item = ExceptionItem(
        _string(data["item_id"], f"{name}.item_id"),
        _provenance_from_data(data["provenance"], f"{name}.provenance"),
        _tensor_reference(data["embedding_tensor"], tensors, f"{name}.embedding_tensor"),
        _integer(
            data["salience_q"],
            f"{name}.salience_q signed int64",
            minimum=SALIENCE_Q_MIN,
            maximum=SALIENCE_Q_MAX,
        ),
        _boolean(data["pinned"], f"{name}.pinned"),
    )
    item.validate(contract)
    expected_fingerprint = _sha256(data["fingerprint"], f"{name}.fingerprint")
    if item.fingerprint() != expected_fingerprint:
        raise ValueError(f"{name} fingerprint mismatch")
    return item


def _leaf_from_data(
    value: object,
    tensors: Mapping[str, torch.Tensor],
    contract: SparseMemoryContract,
    name: str,
) -> AuthoritativeLeaf:
    data = _mapping(
        value,
        name,
        frozenset(
            {
                "key",
                "source_fingerprint",
                "revision",
                "dense_operator_tensor",
                "candidates",
            }
        ),
    )
    candidate_data = _list(data["candidates"], f"{name}.candidates")
    candidates = tuple(
        _item_from_data(item, tensors, contract, f"{name}.candidates[{index}]")
        for index, item in enumerate(candidate_data)
    )
    return AuthoritativeLeaf(
        key=_leaf_key_from_data(data["key"], f"{name}.key"),
        source_fingerprint=_sha256(data["source_fingerprint"], f"{name}.source_fingerprint"),
        revision=_integer(data["revision"], f"{name}.revision"),
        dense_operator=_tensor_reference(
            data["dense_operator_tensor"], tensors, f"{name}.dense_operator_tensor"
        ),
        candidates=candidates,
    )


def _verify_root(tree: DuetXProductTree, value: object) -> None:
    data = _mapping(
        value,
        "root",
        frozenset(
            {
                "coverage",
                "dense_operator_sha256",
                "exception_item_ids",
                "exception_fingerprints",
            }
        ),
    )
    coverage_data = _mapping(
        data["coverage"], "root.coverage", frozenset({"start_slot", "stop_slot"})
    )
    recorded_coverage = Coverage(
        _integer(coverage_data["start_slot"], "root.coverage.start_slot"),
        _integer(coverage_data["stop_slot"], "root.coverage.stop_slot"),
    ).validate()
    if recorded_coverage != tree.root.coverage:
        raise ValueError("snapshot root coverage mismatch")
    recorded_dense_hash = _sha256(data["dense_operator_sha256"], "root.dense_operator_sha256")
    if tensor_sha256(tree.root.dense_operator) != recorded_dense_hash:
        raise ValueError("snapshot root dense operator hash mismatch")
    recorded_ids = tuple(
        _string(item, f"root.exception_item_ids[{index}]")
        for index, item in enumerate(_list(data["exception_item_ids"], "root.exception_item_ids"))
    )
    actual_ids = tuple(item.item_id for item in tree.root.exceptions)
    if recorded_ids != actual_ids:
        raise ValueError("snapshot root exception item IDs mismatch")
    recorded_fingerprints = tuple(
        _sha256(item, f"root.exception_fingerprints[{index}]")
        for index, item in enumerate(
            _list(data["exception_fingerprints"], "root.exception_fingerprints")
        )
    )
    actual_fingerprints = tuple(item.fingerprint() for item in tree.root.exceptions)
    if recorded_fingerprints != actual_fingerprints:
        raise ValueError("snapshot root exception fingerprints mismatch")


def load_runtime_snapshot(
    path: Path,
    *,
    expected_contract: DuetXContract,
    expected_model_checkpoint_sha256: str,
) -> DuetXProductTree:
    """Safely restore and fully verify a Duet-X product-tree runtime snapshot."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not isinstance(expected_contract, DuetXContract):
        raise ValueError("expected_contract must be a DuetXContract")
    expected_contract.validate()
    expected_model = _sha256(expected_model_checkpoint_sha256, "expected_model_checkpoint_sha256")
    try:
        raw = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("runtime snapshot payload could not be loaded safely") from error
    payload = _mapping(raw, "runtime snapshot payload", _TOP_LEVEL_KEYS)
    recorded_manifest = _sha256(payload["manifest_sha256"], "manifest_sha256")
    if _manifest_sha256(payload) != recorded_manifest:
        raise ValueError("runtime snapshot manifest hash mismatch")
    if payload["format"] != _SNAPSHOT_FORMAT:
        raise ValueError("unsupported runtime snapshot format")

    contract = _contract_from_data(payload["contract"])
    recorded_contract = _sha256(payload["contract_fingerprint"], "contract_fingerprint")
    if contract.fingerprint() != recorded_contract:
        raise ValueError("snapshot contract fingerprint does not match its metadata")
    if contract != expected_contract or recorded_contract != expected_contract.fingerprint():
        raise ValueError("snapshot contract does not match expected contract")
    recorded_model = _sha256(payload["model_checkpoint_sha256"], "model_checkpoint_sha256")
    if recorded_model != expected_model:
        raise ValueError("snapshot model checkpoint does not match expected model")

    tree_data = _mapping(
        payload["tree"],
        "tree",
        frozenset({"dense_steps", "operator_size", "leaf_count", "leaves"}),
    )
    dense_steps = _integer(tree_data["dense_steps"], "tree.dense_steps", minimum=1)
    operator_size = _integer(tree_data["operator_size"], "tree.operator_size", minimum=1)
    leaf_count = _integer(tree_data["leaf_count"], "tree.leaf_count", minimum=1)
    leaf_data = _list(tree_data["leaves"], "tree.leaves")
    if len(leaf_data) != leaf_count:
        raise ValueError("tree leaves length does not match leaf_count")
    expected_tensor_names = _expected_tensor_names(leaf_data)
    tensors = _validated_tensors(payload, expected_tensor_names)
    leaves = tuple(
        _leaf_from_data(item, tensors, contract.sparse, f"tree.leaves[{index}]")
        for index, item in enumerate(leaf_data)
        if item is not None
    )
    tree = DuetXProductTree(
        contract,
        dense_steps,
        operator_size,
        leaves,
        leaf_count=leaf_count,
    )
    _verify_root(tree, payload["root"])
    return tree
