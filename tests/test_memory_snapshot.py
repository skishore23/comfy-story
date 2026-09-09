from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from comfy_story.memory.cache import AuthoritativeLeaf, MemoryProductTree
from comfy_story.memory.contracts import (
    SALIENCE_Q_MAX,
    SALIENCE_Q_MIN,
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    MemoryContract,
    RawEvidencePointer,
    SparseMemoryContract,
    SpatialLocation,
    TimeFrameRange,
    tensor_sha256,
)
from comfy_story.memory.snapshot import load_runtime_snapshot, save_runtime_snapshot


class _UnsafePayload:
    pass


@pytest.fixture
def contract() -> MemoryContract:
    return MemoryContract(
        format="comfy-story-memory-minimax-h3-v1",
        adapter_fingerprint="1" * 64,
        sparse=SparseMemoryContract(2, 24, torch.float32, 2),
        history_items=8,
        latent_channels=24,
        dense_accumulation_dtype=torch.float32,
    )


def _item(
    contract: MemoryContract,
    key: LeafKey,
    *,
    item_id: str,
    salience_q: int,
    locator: str | None = None,
) -> ExceptionItem:
    provenance = EvidenceProvenance(
        leaf=key,
        time=TimeFrameRange(100 + key.slot, 101 + key.slot, key.slot, key.slot + 1),
        spatial=SpatialLocation("normalized", "bbox", (0, 0, 8, 8)),
        event_type="rare-detail-v1",
        raw=RawEvidencePointer(
            locator or f"comfy-evidence://registry/sha256/{f'{key.slot:x}' * 64}",
            f"{key.slot:x}" * 64,
            key.slot,
            key.slot + 1,
        ),
        source_registry_sha256="1" * 64,
        preprocessing_sha256="2" * 64,
        vae_sha256="3" * 64,
        adapter_sha256="4" * 64,
        scorer_sha256="5" * 64,
        memory_contract_sha256=contract.sparse.fingerprint(),
    )
    return ExceptionItem(
        item_id,
        provenance,
        torch.arange(24, dtype=torch.float32).add_(salience_q),
        salience_q,
        key.slot in {2, 5},
    )


def _leaf(
    contract: MemoryContract,
    slot: int,
    *,
    locator: str | None = None,
) -> AuthoritativeLeaf:
    key = LeafKey(f"camera-{slot}", slot, slot)
    return AuthoritativeLeaf(
        key=key,
        source_fingerprint=f"{slot:x}" * 64,
        revision=slot,
        dense_operator=torch.tensor(
            [[[[1.0, float(slot + 1)], [float(slot % 2), 1.0]]]], dtype=torch.float32
        ),
        candidates=(
            _item(
                contract,
                key,
                item_id=f"item-{slot}",
                salience_q=slot,
                locator=locator,
            ),
        ),
    )


@pytest.fixture
def tree(contract: MemoryContract) -> MemoryProductTree:
    return MemoryProductTree(
        contract,
        dense_steps=1,
        operator_size=2,
        leaves=tuple(_leaf(contract, slot) for slot in range(8)),
    )


def _assert_tree_equal(actual: MemoryProductTree, expected: MemoryProductTree) -> None:
    assert actual.contract == expected.contract
    assert actual.leaf_count == expected.leaf_count
    assert actual.leaves == expected.leaves
    assert torch.equal(actual.root.dense_operator, expected.root.dense_operator)
    assert tuple(item.item_id for item in actual.root.exceptions) == tuple(
        item.item_id for item in expected.root.exceptions
    )
    assert tuple(item.fingerprint() for item in actual.root.exceptions) == tuple(
        item.fingerprint() for item in expected.root.exceptions
    )


def _read(path: Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(raw, dict)
    return raw


def _write_with_manifest(path: Path, payload: dict[str, Any]) -> None:
    manifest = {
        key: value for key, value in payload.items() if key not in {"manifest_sha256", "tensors"}
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    torch.save(payload, path)


def _first_tensor_name(payload: dict[str, Any]) -> str:
    specs = payload["tensor_specs"]
    assert isinstance(specs, dict)
    name = sorted(specs)[0]
    assert isinstance(name, str)
    return name


def test_snapshot_round_trip(
    tmp_path: Path, tree: MemoryProductTree, contract: MemoryContract
) -> None:
    path = tmp_path / "runtime.pt"
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)

    restored = load_runtime_snapshot(
        path,
        expected_contract=contract,
        expected_model_checkpoint_sha256="0" * 64,
    )

    _assert_tree_equal(restored, tree)


def test_snapshot_payload_is_primitive_metadata_plus_tensors(
    tmp_path: Path, tree: MemoryProductTree
) -> None:
    path = tmp_path / "runtime.pt"
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)

    payload = _read(path)

    def assert_safe(value: object, *, tensors: bool = False) -> None:
        if tensors:
            assert type(value) is dict
            assert all(type(key) is str for key in value)
            assert all(type(item) is torch.Tensor for item in value.values())
        elif type(value) is dict:
            assert all(type(key) is str for key in value)
            for key, item in value.items():
                assert_safe(item, tensors=key == "tensors")
        elif type(value) is list:
            for item in value:
                assert_safe(item)
        else:
            assert value is None or type(value) in {str, int, bool}

    assert_safe(payload)
    assert b"raw-media" not in path.read_bytes()


@pytest.mark.parametrize("field", ["format", "contract_fingerprint", "model_checkpoint_sha256"])
def test_snapshot_rejects_format_contract_or_model_metadata_tampering(
    tmp_path: Path,
    tree: MemoryProductTree,
    contract: MemoryContract,
    field: str,
) -> None:
    path = tmp_path / "runtime.pt"
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)
    payload = _read(path)
    payload[field] = "f" * 64 if field != "format" else "unknown-runtime-format"
    _write_with_manifest(path, payload)

    with pytest.raises(ValueError, match=r"format|contract|model"):
        load_runtime_snapshot(
            path,
            expected_contract=contract,
            expected_model_checkpoint_sha256="0" * 64,
        )


def test_snapshot_rejects_expected_contract_or_model_mismatch(
    tmp_path: Path, tree: MemoryProductTree, contract: MemoryContract
) -> None:
    path = tmp_path / "runtime.pt"
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)

    with pytest.raises(ValueError, match="contract"):
        load_runtime_snapshot(
            path,
            expected_contract=replace(contract, adapter_fingerprint="2" * 64),
            expected_model_checkpoint_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="model"):
        load_runtime_snapshot(
            path,
            expected_contract=contract,
            expected_model_checkpoint_sha256="9" * 64,
        )


def test_snapshot_rejects_other_adapter(tmp_path: Path) -> None:
    minimax = MemoryContract.minimax_h3(adapter_fingerprint="a" * 64)
    tree = MemoryProductTree(minimax, dense_steps=1, operator_size=2)
    path = tmp_path / "minimax-runtime.pt"
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="b" * 64)

    with pytest.raises(ValueError, match="contract"):
        load_runtime_snapshot(
            path,
            expected_contract=MemoryContract.minimax_h3(adapter_fingerprint="c" * 64),
            expected_model_checkpoint_sha256="b" * 64,
        )


def test_snapshot_rejects_manifest_tampering(
    tmp_path: Path, tree: MemoryProductTree, contract: MemoryContract
) -> None:
    path = tmp_path / "runtime.pt"
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)
    payload = _read(path)
    payload["tree"]["dense_steps"] = 2
    torch.save(payload, path)

    with pytest.raises(ValueError, match="manifest"):
        load_runtime_snapshot(
            path,
            expected_contract=contract,
            expected_model_checkpoint_sha256="0" * 64,
        )


def test_snapshot_rejects_tensor_hash_drift(
    tmp_path: Path, tree: MemoryProductTree, contract: MemoryContract
) -> None:
    path = tmp_path / "runtime.pt"
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)
    payload = _read(path)
    name = _first_tensor_name(payload)
    payload["tensors"][name] = payload["tensors"][name].clone().add_(1)
    torch.save(payload, path)

    with pytest.raises(ValueError, match=r"tensor.*hash"):
        load_runtime_snapshot(
            path,
            expected_contract=contract,
            expected_model_checkpoint_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "tamper",
    ["extra", "payload-only", "missing", "alias", "wrong-name"],
)
def test_snapshot_rejects_noncanonical_tensor_reference_sets(
    tmp_path: Path,
    tree: MemoryProductTree,
    contract: MemoryContract,
    tamper: str,
) -> None:
    path = tmp_path / "runtime.pt"
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)
    payload = _read(path)
    canonical = "leaf.0.candidate.0.embedding"
    other = "leaf.1.candidate.0.embedding"
    if tamper == "extra":
        payload["tensors"]["unreferenced"] = torch.ones(1)
        payload["tensor_specs"]["unreferenced"] = {
            "dtype": "float32",
            "shape": [1],
            "sha256": tensor_sha256(torch.ones(1)),
        }
    elif tamper == "payload-only":
        payload["tensors"]["unknown-payload"] = torch.ones(1)
    elif tamper == "missing":
        del payload["tensors"][canonical]
        del payload["tensor_specs"][canonical]
    elif tamper == "alias":
        payload["tree"]["leaves"][1]["candidates"][0]["embedding_tensor"] = canonical
    else:
        payload["tree"]["leaves"][0]["candidates"][0]["embedding_tensor"] = "renamed"
        payload["tensors"]["renamed"] = payload["tensors"].pop(canonical)
        payload["tensor_specs"]["renamed"] = payload["tensor_specs"].pop(canonical)
    assert other in payload["tensors"]
    _write_with_manifest(path, payload)

    with pytest.raises(ValueError, match=r"tensor|reference|canonical|unknown"):
        load_runtime_snapshot(
            path,
            expected_contract=contract,
            expected_model_checkpoint_sha256="0" * 64,
        )


def test_snapshot_rejects_leaf_record_moved_from_its_fixed_position(
    tmp_path: Path, tree: MemoryProductTree, contract: MemoryContract
) -> None:
    path = tmp_path / "runtime.pt"
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)
    payload = _read(path)
    leaves = payload["tree"]["leaves"]
    leaves[0], leaves[1] = leaves[1], leaves[0]
    _write_with_manifest(path, payload)

    with pytest.raises(ValueError, match=r"position|slot"):
        load_runtime_snapshot(
            path,
            expected_contract=contract,
            expected_model_checkpoint_sha256="0" * 64,
        )


@pytest.mark.parametrize("salience_q", [SALIENCE_Q_MIN, SALIENCE_Q_MAX])
def test_snapshot_round_trips_signed_int64_salience_boundaries(
    tmp_path: Path, contract: MemoryContract, salience_q: int
) -> None:
    key = LeafKey("camera-0", 0, 0)
    leaf = AuthoritativeLeaf(
        key=key,
        source_fingerprint="0" * 64,
        revision=0,
        dense_operator=torch.eye(2).reshape(1, 1, 2, 2),
        candidates=(_item(contract, key, item_id="boundary", salience_q=salience_q),),
    )
    tree = MemoryProductTree(contract, 1, 2, leaves=(leaf,))
    path = tmp_path / "runtime.pt"

    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)
    restored = load_runtime_snapshot(
        path,
        expected_contract=contract,
        expected_model_checkpoint_sha256="0" * 64,
    )

    assert restored.leaves[0] is not None
    assert restored.leaves[0].candidates[0].salience_q == salience_q


@pytest.mark.parametrize("salience_q", [SALIENCE_Q_MIN - 1, SALIENCE_Q_MAX + 1])
def test_snapshot_rejects_salience_outside_signed_int64(
    tmp_path: Path,
    tree: MemoryProductTree,
    contract: MemoryContract,
    salience_q: int,
) -> None:
    path = tmp_path / "runtime.pt"
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)
    payload = _read(path)
    payload["tree"]["leaves"][0]["candidates"][0]["salience_q"] = salience_q
    _write_with_manifest(path, payload)

    with pytest.raises(ValueError, match="signed int64"):
        load_runtime_snapshot(
            path,
            expected_contract=contract,
            expected_model_checkpoint_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("kind", "message"), [("shape", "shape"), ("dtype", "dtype"), ("nan", "finite")]
)
def test_snapshot_rejects_invalid_tensor_boundaries(
    tmp_path: Path,
    tree: MemoryProductTree,
    contract: MemoryContract,
    kind: str,
    message: str,
) -> None:
    path = tmp_path / "runtime.pt"
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)
    payload = _read(path)
    name = _first_tensor_name(payload)
    tensor = payload["tensors"][name]
    if kind == "shape":
        tensor = tensor.unsqueeze(0)
    elif kind == "dtype":
        tensor = tensor.double()
    else:
        tensor = tensor.clone()
        tensor.reshape(-1)[0] = float("nan")
    payload["tensors"][name] = tensor
    payload["tensor_specs"][name]["sha256"] = tensor_sha256(tensor)
    _write_with_manifest(path, payload)

    with pytest.raises(ValueError, match=message):
        load_runtime_snapshot(
            path,
            expected_contract=contract,
            expected_model_checkpoint_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("invalid-item", "salience"),
        ("duplicate", "duplicate"),
        ("range", "range"),
        ("root", "root"),
    ],
)
def test_snapshot_rejects_invalid_reconstructed_state(
    tmp_path: Path,
    tree: MemoryProductTree,
    contract: MemoryContract,
    kind: str,
    message: str,
) -> None:
    path = tmp_path / "runtime.pt"
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)
    payload = _read(path)
    leaves = payload["tree"]["leaves"]
    if kind == "invalid-item":
        leaves[0]["candidates"][0]["salience_q"] = "high"
    elif kind == "duplicate":
        leaves[1]["key"]["slot"] = 0
        leaves[1]["candidates"] = []
    elif kind == "range":
        leaves[0]["key"]["slot"] = 8
        leaves[0]["candidates"] = []
    else:
        payload["root"]["dense_operator_sha256"] = "f" * 64
    _write_with_manifest(path, payload)

    with pytest.raises(ValueError, match=message):
        load_runtime_snapshot(
            path,
            expected_contract=contract,
            expected_model_checkpoint_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "locator",
    [
        "data:video/mp4;base64,cmF3LW1lZGlh",
        "https://user:secret@example.invalid/video.mp4",
        "https://example.invalid/video.mp4?X-Amz-Signature=secret",
        "comfy-evidence://registry/token/secret",
        "comfy-evidence://registry/source%2Fsecret",
        f"comfy-evidence://registry/bearer/jwt/{'0' * 64}",
        f"comfy-evidence://registry/private-key/{'0' * 64}",
        f"comfy-evidence://registry/token/{'0' * 64}",
        f"comfy-evidence://registry/signature/{'0' * 64}",
        f"comfy-evidence://registry/sha256/{'1' * 64}",
        f"comfy-evidence://registry/sha256/{'A' * 64}",
        f"comfy-evidence://user:secret@registry/sha256/{'0' * 64}",
        f"comfy-evidence://registry:443/sha256/{'0' * 64}",
        f"comfy-evidence://./sha256/{'0' * 64}",
        f"comfy-evidence://../sha256/{'0' * 64}",
        f"comfy-evidence://registry/sha256/{'0' * 64}/extra",
        "comfy-evidence://registry/sha256",
    ],
)
def test_snapshot_never_serializes_inline_media_credentials_or_signed_urls(
    tmp_path: Path, contract: MemoryContract, locator: str
) -> None:
    path = tmp_path / "runtime.pt"
    tree = MemoryProductTree(contract, 1, 2, leaves=(_leaf(contract, 0),))
    save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)
    payload = _read(path)
    payload["tree"]["leaves"][0]["candidates"][0]["provenance"]["raw"]["locator"] = locator
    _write_with_manifest(path, payload)

    with pytest.raises(ValueError, match="locator"):
        load_runtime_snapshot(
            path,
            expected_contract=contract,
            expected_model_checkpoint_sha256="0" * 64,
        )


def test_snapshot_load_rejects_arbitrary_custom_objects(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.pt"
    torch.save(_UnsafePayload(), path)

    with pytest.raises(ValueError, match=r"safe|payload"):
        load_runtime_snapshot(
            path,
            expected_contract=MemoryContract.minimax_h3(adapter_fingerprint="2" * 64),
            expected_model_checkpoint_sha256="0" * 64,
        )


def test_failed_save_preserves_existing_destination(
    tmp_path: Path,
    tree: MemoryProductTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "runtime.pt"
    path.write_bytes(b"existing-snapshot")

    def fail_save(payload: object, destination: object) -> None:
        if hasattr(destination, "write"):
            destination.write(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr("comfy_story.memory.snapshot.torch.save", fail_save)

    with pytest.raises(OSError, match="simulated"):
        save_runtime_snapshot(path, tree, model_checkpoint_sha256="0" * 64)
    assert path.read_bytes() == b"existing-snapshot"
    assert list(tmp_path.iterdir()) == [path]
