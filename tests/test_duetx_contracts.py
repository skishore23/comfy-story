from __future__ import annotations

from dataclasses import fields, replace
from typing import Protocol, cast

import pytest
import torch

from duet.duetx.contracts import (
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


class _Validatable(Protocol):
    def validate(self) -> object: ...


@pytest.fixture
def contract() -> DuetXContract:
    return DuetXContract.default(decision1_fingerprint="1" * 64)


@pytest.fixture
def sparse_memory() -> SparseMemoryContract:
    return SparseMemoryContract(2, 4, torch.float32, 2)


@pytest.fixture
def provenance(sparse_memory: SparseMemoryContract) -> EvidenceProvenance:
    return EvidenceProvenance(
        leaf=LeafKey(source_id="camera-a", source_rank=3, slot=2),
        time=TimeFrameRange(100, 200, 10, 11),
        spatial=SpatialLocation("normalized-image", "bbox", (0, 0, 65536, 65536)),
        event_type="rare-detail-v1",
        raw=RawEvidencePointer(f"duet-evidence://registry/sha256/{'0' * 64}", "0" * 64, 12, 64),
        source_registry_sha256="1" * 64,
        preprocessing_sha256="2" * 64,
        vae_sha256="3" * 64,
        adapter_sha256="4" * 64,
        scorer_sha256="5" * 64,
        memory_contract_sha256=sparse_memory.fingerprint(),
    )


@pytest.fixture
def item(provenance: EvidenceProvenance) -> ExceptionItem:
    return ExceptionItem("item-a", provenance, torch.arange(4, dtype=torch.float32), 10)


def _replace_item(item: ExceptionItem, **changes: object) -> ExceptionItem:
    return ExceptionItem(
        cast(str, changes.get("item_id", item.item_id)),
        cast(EvidenceProvenance, changes.get("provenance", item.provenance)),
        cast(torch.Tensor, changes.get("embedding", item.embedding)),
        cast(int, changes.get("salience_q", item.salience_q)),
        cast(bool, changes.get("pinned", item.pinned)),
    )


def test_valid_contracts_and_fingerprints(
    contract: DuetXContract,
    sparse_memory: SparseMemoryContract,
    provenance: EvidenceProvenance,
    item: ExceptionItem,
) -> None:
    assert contract.validate() is contract
    assert provenance.validate() is provenance
    item.validate(sparse_memory)
    assert len(item.fingerprint()) == 64
    assert tensor_sha256(item.embedding) == tensor_sha256(item.embedding.clone())


def test_exception_fingerprint_covers_embedding(item: ExceptionItem) -> None:
    changed = _replace_item(item, embedding=item.embedding.clone().add_(1))
    assert item.fingerprint() != changed.fingerprint()


def test_exception_fingerprint_has_a_canonical_golden_value(item: ExceptionItem) -> None:
    assert item.fingerprint() == "3af46d85ad30bdf0c3f15b46382dfe77b620634ef39d3227cc1fd11cc235fd08"


@pytest.mark.parametrize("salience_q", [SALIENCE_Q_MIN, SALIENCE_Q_MAX])
def test_exception_accepts_signed_int64_salience_boundaries(
    item: ExceptionItem, sparse_memory: SparseMemoryContract, salience_q: int
) -> None:
    boundary = _replace_item(item, salience_q=salience_q)
    boundary.validate(sparse_memory)
    assert boundary.salience_q == salience_q


@pytest.mark.parametrize("salience_q", [SALIENCE_Q_MIN - 1, SALIENCE_Q_MAX + 1])
def test_exception_rejects_salience_outside_signed_int64(
    item: ExceptionItem, sparse_memory: SparseMemoryContract, salience_q: int
) -> None:
    with pytest.raises(ValueError, match="signed int64"):
        _replace_item(item, salience_q=salience_q).validate(sparse_memory)


def test_raw_evidence_pointer_accepts_canonical_durable_locator() -> None:
    pointer = RawEvidencePointer(f"duet-evidence://registry_1/sha256/{'0' * 64}", "0" * 64, 0, 1)
    assert pointer.validate() is pointer


@pytest.mark.parametrize(
    "locator",
    [
        "",
        "artifact://registry/source",
        "file:///private/video.mp4",
        "https://example.invalid/video.mp4",
        "arbitrary:registry/source",
        "duet-evidence://user:secret@registry/source",
        f"duet-evidence://registry:443/sha256/{'0' * 64}",
        f"duet-evidence://./sha256/{'0' * 64}",
        f"duet-evidence://../sha256/{'0' * 64}",
        f"duet-evidence://registry/sha256/{'0' * 64}?token=secret",
        f"duet-evidence://registry/sha256/{'0' * 64}#fragment",
        f"duet-evidence://registry/sha256/{'0' * 62}%30%30",
        f"duet-evidence://registry//sha256/{'0' * 64}",
        f"duet-evidence://registry/bearer/jwt/{'0' * 64}",
        f"duet-evidence://registry/private-key/{'0' * 64}",
        f"duet-evidence://registry/token/{'0' * 64}",
        f"duet-evidence://registry/signature/{'0' * 64}",
        f"duet-evidence://registry/sha256/{'0' * 64}/extra",
        "duet-evidence://registry/sha256",
        f"duet-evidence://registry/{'0' * 64}",
        f"duet-evidence://registry/sha256/{'A' * 64}",
        f"duet-evidence://registry/sha256/{'1' * 64}",
    ],
)
def test_raw_evidence_pointer_rejects_noncanonical_or_credential_locator(
    locator: str,
) -> None:
    with pytest.raises(ValueError, match="locator"):
        RawEvidencePointer(locator, "0" * 64, 0, 1).validate()


@pytest.mark.parametrize("bad_hash", ["", "A" * 64, "0" * 63])
def test_provenance_rejects_invalid_sha256(bad_hash: str, provenance: EvidenceProvenance) -> None:
    raw = replace(provenance.raw, content_sha256=bad_hash)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(provenance, raw=raw).validate()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (LeafKey("", 0, 0), "source_id"),
        (LeafKey("source", -1, 0), "source_rank"),
        (LeafKey("source", 0, -1), "slot"),
        (TimeFrameRange(1, 1, 0, 1), "timestamp"),
        (TimeFrameRange(1, 2, 1, 1), "frame"),
        (RawEvidencePointer("", "0" * 64, 0, 1), "locator"),
        (
            RawEvidencePointer(f"duet-evidence://registry/sha256/{'0' * 64}", "0" * 64, 2, 2),
            "byte",
        ),
        (SpatialLocation("normalized", "bbox", (0, 0, 2, 2, 3)), "bbox"),
        (SpatialLocation("normalized", "polygon", (0, 0, 2, 2)), "polygon"),
        (SpatialLocation("normalized", "bbox", (-1, 0, 2, 2)), "q16"),
    ],
)
def test_component_contracts_reject_invalid_values(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cast(_Validatable, value).validate()


@pytest.mark.parametrize(
    "coordinates",
    [
        (0, 0, 0, 0, 0, 0),
        (0, 0, 2, 2, 4, 4),
    ],
)
def test_spatial_location_rejects_degenerate_polygons(coordinates: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="polygon"):
        SpatialLocation("normalized", "polygon", coordinates).validate()


def test_exception_and_memory_reject_invalid_embeddings(
    item: ExceptionItem, sparse_memory: SparseMemoryContract
) -> None:
    with pytest.raises(ValueError, match="dtype"):
        _replace_item(item, embedding=item.embedding.to(torch.float64)).validate(sparse_memory)
    with pytest.raises(ValueError, match="finite"):
        _replace_item(item, embedding=torch.tensor([0.0, float("nan"), 2.0, 3.0])).validate(
            sparse_memory
        )
    with pytest.raises(ValueError, match="shape"):
        _replace_item(item, embedding=torch.zeros(2, 2)).validate(sparse_memory)
    with pytest.raises(ValueError, match="capacity"):
        SparseMemoryContract(0, 4, torch.float32).validate()
    with pytest.raises(ValueError, match="pinned"):
        SparseMemoryContract(2, 4, torch.float32, max_pinned=3).validate()


def test_exception_requires_matching_memory_contract_fingerprint(
    item: ExceptionItem, sparse_memory: SparseMemoryContract
) -> None:
    item.validate(sparse_memory)
    mismatched = _replace_item(
        item,
        provenance=replace(item.provenance, memory_contract_sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="memory contract fingerprint"):
        mismatched.validate(sparse_memory)


def test_exception_embedding_is_semantically_immutable(
    provenance: EvidenceProvenance, sparse_memory: SparseMemoryContract
) -> None:
    caller_embedding = torch.arange(4, dtype=torch.float32)
    item = ExceptionItem("item-a", provenance, caller_embedding, 10)
    original_fingerprint = item.fingerprint()
    caller_embedding.add_(10)
    item.embedding.add_(20)
    assert item.fingerprint() == original_fingerprint
    torch.testing.assert_close(item.embedding, torch.arange(4, dtype=torch.float32))
    assert all(not isinstance(getattr(item, field.name), torch.Tensor) for field in fields(item))
    item.validate(sparse_memory)


def test_exception_replace_and_equality_include_embedding(item: ExceptionItem) -> None:
    same = ExceptionItem(
        item.item_id,
        item.provenance,
        item.embedding.clone(),
        item.salience_q,
        item.pinned,
    )
    changed = replace(item, embedding=item.embedding.clone().add_(1))
    assert same == item
    assert changed != item
    assert changed.fingerprint() != item.fingerprint()


def test_duetx_contract_locks_dense_boundary_and_expected_config(
    contract: DuetXContract,
) -> None:
    assert contract.validate("1" * 64) is contract
    assert replace(contract, sparse=SparseMemoryContract(2, 4, torch.float32, 2)).validate()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        contract.validate("2" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        contract.validate("A" * 64)
    with pytest.raises(ValueError, match="latent_channels"):
        replace(contract, latent_channels=127).validate()
    with pytest.raises(ValueError, match="dense_accumulation_dtype"):
        replace(contract, dense_accumulation_dtype=torch.float64).validate()


def test_minimax_h3_contract_has_an_exact_native_latent_identity() -> None:
    contract = DuetXContract.minimax_h3(adapter_fingerprint="a" * 64)

    assert contract.validate() is contract
    assert contract.format == "duet-x-minimax-h3-v1"
    assert contract.history_items == 8
    assert contract.latent_channels == 24
    assert contract.sparse == SparseMemoryContract(2, 24, torch.float32, 2)
    assert contract.dense_accumulation_dtype is torch.float32
    assert contract.label() == (
        '{"decision1_fingerprint":"' + "a" * 64 + '","dense_accumulation_dtype":"float32",'
        '"format":"duet-x-minimax-h3-v1","history_items":8,'
        '"latent_channels":24,'
        '"sparse":"76ca04c34d721312cb4c8df56ef7ffd7064dbc07ef098ec86110cbb4e18d72c1"}'
    )


def test_minimax_h3_contract_rejects_cross_backend_dimensions() -> None:
    contract = DuetXContract.minimax_h3(adapter_fingerprint="a" * 64)

    with pytest.raises(ValueError, match="latent_channels"):
        replace(contract, latent_channels=128).validate()
    with pytest.raises(ValueError, match="embedding_dim"):
        replace(
            contract,
            sparse=SparseMemoryContract(2, 128, torch.float32, 2),
        ).validate()


def test_ltx_default_contract_label_remains_byte_compatible() -> None:
    contract = DuetXContract.default(decision1_fingerprint="0" * 64)

    assert contract.label() == (
        '{"decision1_fingerprint":"' + "0" * 64 + '","dense_accumulation_dtype":"float32",'
        '"format":"duet-x-ltx-decision1-v1","history_items":8,'
        '"latent_channels":128,'
        '"sparse":"8976e792e6ac87773f5a29922a91458ddaf804cdffe6d1539ec24f2a432d1c66"}'
    )


def test_coverage_rejects_negative_or_reversed_ranges() -> None:
    assert Coverage(0, 1).validate().stop_slot == 1
    assert Coverage(1, 1).validate().start_slot == 1
    with pytest.raises(ValueError, match="coverage"):
        Coverage(-1, 1).validate()
    with pytest.raises(ValueError, match="coverage"):
        Coverage(2, 1).validate()


def test_exception_fingerprint_changes_for_every_field(item: ExceptionItem) -> None:
    provenance = item.provenance
    fields = (
        _replace_item(item, item_id="item-b"),
        _replace_item(item, salience_q=11),
        _replace_item(item, pinned=True),
        _replace_item(item, provenance=replace(provenance, event_type="other-v1")),
        _replace_item(
            item,
            provenance=replace(provenance, leaf=replace(provenance.leaf, source_id="camera-b")),
        ),
        _replace_item(
            item,
            provenance=replace(provenance, leaf=replace(provenance.leaf, source_rank=4)),
        ),
        _replace_item(item, provenance=replace(provenance, leaf=replace(provenance.leaf, slot=3))),
        _replace_item(item, provenance=replace(provenance, time=TimeFrameRange(101, 200, 10, 11))),
        _replace_item(item, provenance=replace(provenance, time=TimeFrameRange(100, 201, 10, 11))),
        _replace_item(item, provenance=replace(provenance, time=TimeFrameRange(100, 200, 9, 11))),
        _replace_item(item, provenance=replace(provenance, time=TimeFrameRange(100, 200, 10, 12))),
        _replace_item(
            item,
            provenance=replace(
                provenance,
                spatial=SpatialLocation("other-system", "bbox", (0, 0, 65536, 65536)),
            ),
        ),
        _replace_item(
            item,
            provenance=replace(
                provenance,
                spatial=SpatialLocation("normalized-image", "polygon", (0, 0, 65536, 0, 0, 65536)),
            ),
        ),
        _replace_item(
            item,
            provenance=replace(
                provenance,
                spatial=SpatialLocation("normalized-image", "bbox", (1, 0, 65536, 65536)),
            ),
        ),
        _replace_item(
            item,
            provenance=replace(
                provenance,
                raw=replace(
                    provenance.raw,
                    locator=f"duet-evidence://other-registry/sha256/{'0' * 64}",
                ),
            ),
        ),
        _replace_item(
            item,
            provenance=replace(
                provenance,
                raw=replace(provenance.raw, content_sha256="7" * 64),
            ),
        ),
        _replace_item(
            item,
            provenance=replace(provenance, raw=replace(provenance.raw, byte_start=11)),
        ),
        _replace_item(
            item,
            provenance=replace(provenance, raw=replace(provenance.raw, byte_stop=65)),
        ),
        _replace_item(item, provenance=replace(provenance, source_registry_sha256="7" * 64)),
        _replace_item(item, provenance=replace(provenance, preprocessing_sha256="7" * 64)),
        _replace_item(item, provenance=replace(provenance, vae_sha256="7" * 64)),
        _replace_item(item, provenance=replace(provenance, adapter_sha256="7" * 64)),
        _replace_item(item, provenance=replace(provenance, scorer_sha256="7" * 64)),
        _replace_item(item, provenance=replace(provenance, memory_contract_sha256="7" * 64)),
    )
    assert all(changed.fingerprint() != item.fingerprint() for changed in fields)
