from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
import torch

from duet.duetx.contracts import (
    EvidenceProvenance,
    LeafKey,
    RawEvidencePointer,
    SpatialLocation,
    TimeFrameRange,
    tensor_sha256,
)
from duet.duetx.data import HistoryBatch, HistoryItem, HistorySample
from duet.duetx.teacher import (
    InfluenceRecord,
    TeacherBackend,
    TeacherOutput,
    label_influence,
    load_influence_shard,
    quantize_half_up,
    write_influence_shard,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _batch(split: str = "train", *, sample_offset: int = 0) -> HistoryBatch:
    history = []
    for slot in range(8):
        digest = _digest(100 + sample_offset * 20 + slot)
        end_time_ns = 10 + slot
        evidence = EvidenceProvenance(
            LeafKey(f"source-{sample_offset}-{slot}", slot, slot),
            TimeFrameRange(end_time_ns - 1, end_time_ns, slot, slot + 1),
            SpatialLocation("normalized-v1", "bbox", (0, 0, 65_536, 65_536)),
            "history-v1",
            RawEvidencePointer(f"duet-evidence://test/sha256/{digest}", digest, 0, 1),
            "a" * 64,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            "e" * 64,
            "f" * 64,
        )
        latent = torch.full((1, 128, 1, 1, 1), float(slot))
        history.append(
            HistoryItem(
                item_id=f"item-{sample_offset}-{slot}",
                content_sha256=tensor_sha256(latent),
                source_sha256=_digest(2_000 + sample_offset * 20 + slot),
                media_sha256=_digest(3_000 + sample_offset * 20 + slot),
                end_time_ns=end_time_ns,
                latent=latent,
                evidence=evidence,
            )
        )
    sample = HistorySample(
        sample_id=f"sample-{sample_offset}",
        split=split,  # type: ignore[arg-type]
        performer_id=f"performer-{sample_offset}",
        room_id=f"room-{sample_offset}",
        prop_ids=(f"prop-{sample_offset}",),
        glyph_family=f"glyph-{sample_offset}",
        license_id=f"license-{sample_offset}",
        consent_id=f"consent-{sample_offset}",
        target_cutoff_ns=20,
        history=tuple(history),
        parent_manifest_sha256="9" * 64,
    )
    return HistoryBatch.from_sample(sample)


class FakeTeacher:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[float, ...], int, int | None]] = []

    def evaluate(
        self,
        batch: HistoryBatch,
        *,
        timesteps: tuple[float, ...] = (0.2, 0.5, 0.8),
        coordinate_seed: int,
        omitted_index: int | None,
    ) -> TeacherOutput:
        del batch
        self.calls.append((timesteps, coordinate_seed, omitted_index))
        if omitted_index is None:
            return TeacherOutput(
                velocity=torch.full((3, 2), 2.0),
                hidden=torch.tensor([[1.0, 0.0]]).repeat(3, 1),
                marker_score=torch.zeros(3),
                structural_score=0.0,
            )
        offset = (omitted_index + 1) / 10
        return TeacherOutput(
            velocity=torch.full((3, 2), 3.0),
            hidden=torch.tensor([[0.0, 1.0]]).repeat(3, 1),
            marker_score=torch.full((3,), offset),
            structural_score=offset / 2,
        )


class ZeroTeacher:
    def evaluate(
        self,
        batch: HistoryBatch,
        *,
        timesteps: tuple[float, ...] = (0.2, 0.5, 0.8),
        coordinate_seed: int,
        omitted_index: int | None,
    ) -> TeacherOutput:
        del batch, timesteps, coordinate_seed, omitted_index
        return TeacherOutput(
            velocity=torch.zeros(3, 2),
            hidden=torch.zeros(3, 2),
            marker_score=torch.zeros(3),
            structural_score=torch.zeros(3),
        )


class BroadcastShapeTeacher:
    def __init__(self, mismatch: str) -> None:
        self.mismatch = mismatch

    def evaluate(
        self,
        batch: HistoryBatch,
        *,
        timesteps: tuple[float, ...] = (0.2, 0.5, 0.8),
        coordinate_seed: int,
        omitted_index: int | None,
    ) -> TeacherOutput:
        del batch, timesteps, coordinate_seed
        score = torch.tensor([0.0, 0.5, 1.0])
        changed = score if omitted_index is None else score[:, None]
        return TeacherOutput(
            velocity=torch.zeros(3, 2),
            hidden=torch.ones(3, 2),
            marker_score=changed if self.mismatch == "marker" else score,
            structural_score=changed if self.mismatch == "structural" else score,
        )


def _label(
    batch: HistoryBatch, backend: TeacherBackend, *, coordinate_seed: int = 42
) -> tuple[InfluenceRecord, ...]:
    return label_influence(
        batch,
        backend,
        teacher_artifact_sha256="1" * 64,
        coordinate_seed=coordinate_seed,
        config_fingerprint="2" * 64,
        source_fingerprint="3" * 64,
        runtime_fingerprint="4" * 64,
    )


def test_label_influence_uses_one_full_and_eight_identically_coordinated_calls() -> None:
    backend = FakeTeacher()

    records = _label(_batch(), backend)

    assert backend.calls == [
        ((0.2, 0.5, 0.8), 42, None),
        *[((0.2, 0.5, 0.8), 42, index) for index in range(8)],
    ]
    assert tuple(record.item_index for record in records) == tuple(range(8))


def test_label_influence_uses_locked_component_math_and_half_up_quantization() -> None:
    records = _label(_batch(), FakeTeacher())

    first = records[0]
    assert first.velocity == pytest.approx(0.249999999375)
    assert first.hidden == 0.5
    assert first.marker == pytest.approx(0.1)
    assert first.structural == pytest.approx(0.05)
    assert first.score_q == 232_500
    assert quantize_half_up(0.0000005, 1_000_000) == 1
    assert quantize_half_up(-0.0000005, 1_000_000) == -1
    with pytest.raises(ValueError, match="signed int64"):
        quantize_half_up(Decimal("1e100"), 1)


def test_identical_zero_teacher_outputs_have_zero_influence() -> None:
    records = _label(_batch(), ZeroTeacher())

    assert all(record.hidden == 0.0 for record in records)
    assert all(record.score_q == 0 for record in records)


@pytest.mark.parametrize("score_name", ["marker", "structural"])
def test_teacher_score_shapes_must_match_without_coordinate_broadcasting(
    score_name: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{score_name}_score shapes must match"):
        _label(_batch(), BroadcastShapeTeacher(score_name))


def test_label_records_are_primitive_and_include_every_identity_fingerprint() -> None:
    batch = _batch()
    record = _label(batch, FakeTeacher())[0]

    payload = record.to_dict()

    assert payload == {
        "config_fingerprint": "2" * 64,
        "coordinate_seed": 42,
        "hidden": 0.5,
        "item_id": "item-0-0",
        "item_index": 0,
        "item_sha256": batch.content_sha256[0],
        "marker": pytest.approx(0.1),
        "media_sha256": _digest(3_000),
        "parent_manifest_sha256": "9" * 64,
        "runtime_fingerprint": "4" * 64,
        "sample_id": "sample-0",
        "score_q": 232_500,
        "source_fingerprint": "3" * 64,
        "source_sha256": _digest(2_000),
        "split": "train",
        "structural": pytest.approx(0.05),
        "teacher_artifact_sha256": "1" * 64,
        "timesteps": [0.2, 0.5, 0.8],
        "velocity": pytest.approx(0.249999999375),
    }
    assert json.loads(json.dumps(payload)) == payload


def test_test_label_generation_and_writes_are_impossible(tmp_path: Path) -> None:
    backend = FakeTeacher()
    with pytest.raises(ValueError, match="train or validation"):
        _label(_batch("test"), backend)
    assert backend.calls == []

    record = _label(_batch(), FakeTeacher())[0]
    with pytest.raises(ValueError, match="test"):
        write_influence_shard(
            tmp_path / "labels.jsonl",
            (replace(record, split="test"),),  # type: ignore[arg-type]
        )
    assert not (tmp_path / "labels.jsonl").exists()


def test_shard_resume_accepts_only_complete_validated_record_sets(tmp_path: Path) -> None:
    first = _label(_batch(sample_offset=0), FakeTeacher())
    second = _label(_batch("validation", sample_offset=1), FakeTeacher())
    path = tmp_path / "labels.jsonl"

    write_influence_shard(path, first)
    first_bytes = path.read_bytes()
    write_influence_shard(path, first)
    assert path.read_bytes() == first_bytes

    write_influence_shard(path, (*first, *second))
    assert len(path.read_text().splitlines()) == 16


def test_public_shard_loader_round_trips_and_rejects_duplicate_keys(tmp_path: Path) -> None:
    records = _label(_batch(), FakeTeacher())
    path = tmp_path / "labels.jsonl"
    write_influence_shard(path, records)

    assert load_influence_shard(path) == records

    row = path.read_text().splitlines()[0]
    path.write_text(row[:-1] + ',"sample_id":"duplicate"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_influence_shard(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda encoded: encoded.rstrip(b"\n"), "final newline"),
        (lambda encoded: encoded.replace(b"\n", b"\r\n"), "LF newlines"),
    ],
)
def test_public_shard_loader_requires_canonical_line_endings(
    tmp_path: Path, mutate: Callable[[bytes], bytes], message: str
) -> None:
    records = _label(_batch(), FakeTeacher())
    path = tmp_path / "labels.jsonl"
    write_influence_shard(path, records)
    path.write_bytes(mutate(path.read_bytes()))

    with pytest.raises(ValueError, match=message):
        load_influence_shard(path)


def test_shard_resume_rejects_duplicate_keys_even_when_values_match(tmp_path: Path) -> None:
    records = _label(_batch(), FakeTeacher())
    path = tmp_path / "labels.jsonl"
    write_influence_shard(path, records)
    rows = path.read_text().splitlines()
    rows[0] = rows[0][:-1] + ',"sample_id":"sample-0"}'
    path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="duplicate"):
        write_influence_shard(path, records)


def test_shard_rejects_mixed_run_fingerprints_within_a_completed_set(tmp_path: Path) -> None:
    records = _label(_batch(), FakeTeacher())
    mixed = (*records[:4], replace(records[4], runtime_fingerprint="8" * 64), *records[5:])

    with pytest.raises(ValueError, match="fingerprints"):
        write_influence_shard(tmp_path / "labels.jsonl", mixed)


def test_shard_enforces_run_fingerprints_across_completed_sets(tmp_path: Path) -> None:
    first = _label(_batch(sample_offset=0), FakeTeacher())
    second = _label(_batch("validation", sample_offset=1), FakeTeacher(), coordinate_seed=99)
    mixed_runtime = tuple(replace(record, runtime_fingerprint="8" * 64) for record in second)

    with pytest.raises(ValueError, match=r"shard.*fingerprints"):
        write_influence_shard(tmp_path / "mixed.jsonl", (*first, *mixed_runtime))

    write_influence_shard(tmp_path / "allowed.jsonl", (*first, *second))


@pytest.mark.parametrize("existing_count", [1, 7])
def test_shard_resume_rejects_partial_or_tampered_existing_rows(
    tmp_path: Path, existing_count: int
) -> None:
    records = _label(_batch(), FakeTeacher())
    path = tmp_path / "labels.jsonl"
    rows = [record.to_dict() for record in records[:existing_count]]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )

    with pytest.raises(ValueError, match="completed-record set"):
        write_influence_shard(path, records)

    rows = [record.to_dict() for record in records]
    rows[0]["score_q"] = 0
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match=r"conflicting|tampered"):
        write_influence_shard(path, records)
