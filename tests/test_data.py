from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from comfy_story.contracts import (
    EvidenceProvenance,
    LeafKey,
    RawEvidencePointer,
    SpatialLocation,
    TimeFrameRange,
    tensor_sha256,
)
from comfy_story.data import (
    HistoryBatch,
    HistoryItem,
    HistorySample,
    Split,
    audit_decision1_procedural_canary,
    audit_manifest,
)
from comfy_story.guide_boundary import LTXGuideProvenance, LTXLatentGuide


def _digest(value: int) -> str:
    return f"{value:064x}"


def _evidence(slot: int, *, sample_offset: int = 0) -> EvidenceProvenance:
    digest = _digest(100 + sample_offset + slot)
    end_time_ns = 10 + slot
    return EvidenceProvenance(
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


def _sample(
    split: Split,
    sample_offset: int,
    *,
    performer_id: str | None = None,
    room_id: str | None = None,
    prop_ids: tuple[str, ...] | None = None,
    glyph_family: str | None = None,
) -> HistorySample:
    items_list: list[HistoryItem] = []
    for slot in range(8):
        latent = torch.full((1, 128, 1, 2, 2), float(slot + sample_offset))
        items_list.append(
            HistoryItem(
                item_id=f"item-{sample_offset}-{slot}",
                content_sha256=tensor_sha256(latent),
                source_sha256=_digest(2_000 + sample_offset * 20 + slot),
                media_sha256=_digest(3_000 + sample_offset * 20 + slot),
                end_time_ns=10 + slot,
                latent=latent,
                evidence=_evidence(slot, sample_offset=sample_offset * 20),
            )
        )
    items = tuple(items_list)
    return HistorySample(
        sample_id=f"sample-{sample_offset}",
        split=split,
        performer_id=performer_id or f"performer-{sample_offset}",
        room_id=room_id or f"room-{sample_offset}",
        prop_ids=prop_ids or (f"prop-{sample_offset}",),
        glyph_family=glyph_family or f"glyph-{sample_offset}",
        license_id=f"license-{sample_offset}",
        consent_id=f"consent-{sample_offset}",
        target_cutoff_ns=20,
        history=items,
        parent_manifest_sha256="9" * 64,
    )


def test_audit_accepts_isolated_declared_pre_cutoff_histories() -> None:
    samples = (_sample("train", 0), _sample("validation", 1), _sample("test", 2))

    audited = audit_manifest(
        samples,
        license_rows={sample.license_id for sample in samples},
        consent_rows={sample.consent_id for sample in samples},
    )

    assert audited == samples


def _procedural_canary() -> tuple[HistorySample, ...]:
    samples: list[HistorySample] = []
    splits: tuple[Split, ...] = ("train", "validation", "test")
    for split_index, split in enumerate(splits):
        for scene_index in range(8):
            sample = _sample(split, split_index * 100 + scene_index)
            history = tuple(
                replace(
                    item,
                    evidence=replace(
                        item.evidence,
                        event_type=(
                            f"rare-marker:slot-{slot}" if slot in {2, 5} else "common-motion"
                        ),
                    ),
                )
                for slot, item in enumerate(sample.history)
            )
            samples.append(replace(sample, history=history))
    return tuple(samples)


def test_procedural_canary_accepts_eight_isolated_scenes_per_split() -> None:
    samples = _procedural_canary()

    audited = audit_decision1_procedural_canary(
        samples,
        license_rows={sample.license_id for sample in samples},
        consent_rows={sample.consent_id for sample in samples},
    )

    assert len(audited) == 24
    assert {
        split: sum(sample.split == split for sample in audited)
        for split in ("train", "validation", "test")
    } == {"train": 8, "validation": 8, "test": 8}


@pytest.mark.parametrize("change", ["count", "rare_slot", "scene_cluster"])
def test_procedural_canary_rejects_underpowered_or_ambiguous_design(change: str) -> None:
    samples = list(_procedural_canary())
    if change == "count":
        samples.pop()
        message = "eight scenes per split"
    elif change == "rare_slot":
        sample = samples[0]
        history = list(sample.history)
        history[2] = replace(
            history[2], evidence=replace(history[2].evidence, event_type="common-motion")
        )
        samples[0] = replace(sample, history=tuple(history))
        message = "rare markers"
    else:
        samples[1] = replace(
            samples[1],
            performer_id=samples[0].performer_id,
            room_id=samples[0].room_id,
            glyph_family=samples[0].glyph_family,
        )
        message = "scene clusters"

    with pytest.raises(ValueError, match=message):
        audit_decision1_procedural_canary(
            samples,
            license_rows={sample.license_id for sample in samples},
            consent_rows={sample.consent_id for sample in samples},
        )


@pytest.mark.parametrize("field", ["performer_id", "room_id", "prop_ids", "glyph_family"])
def test_audit_rejects_group_leakage_across_splits(field: str) -> None:
    train = _sample("train", 0)
    validation = _sample("validation", 1)
    leaked = replace(validation, **{field: getattr(train, field)})

    with pytest.raises(ValueError, match=field):
        audit_manifest(
            (train, leaked),
            license_rows={train.license_id, leaked.license_id},
            consent_rows={train.consent_id, leaked.consent_id},
        )


@pytest.mark.parametrize("field", ["source_sha256", "media_sha256"])
def test_audit_rejects_duplicate_source_or_media_hashes(field: str) -> None:
    train = _sample("train", 0)
    validation = _sample("validation", 1)
    duplicate = replace(validation.history[0], **{field: getattr(train.history[0], field)})
    validation = replace(validation, history=(duplicate, *validation.history[1:]))

    with pytest.raises(ValueError, match=field):
        audit_manifest(
            (train, validation),
            license_rows={train.license_id, validation.license_id},
            consent_rows={train.consent_id, validation.consent_id},
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("history_count", "exactly eight"),
        ("future", "cutoff"),
        ("shape", "equal latent shapes"),
        ("license", "license"),
        ("consent", "consent"),
        ("test_labels", "sealed"),
    ],
)
def test_audit_rejects_any_boundary_or_declaration_violation(change: str, message: str) -> None:
    sample = _sample("test", 0)
    licenses: set[str] = {sample.license_id}
    consents: set[str] = {sample.consent_id}
    if change == "history_count":
        sample = replace(sample, history=sample.history[:-1])
    elif change == "future":
        future = replace(sample.history[-1], end_time_ns=sample.target_cutoff_ns)
        sample = replace(sample, history=(*sample.history[:-1], future))
    elif change == "shape":
        latent = torch.zeros(1, 128, 2, 2, 2)
        different = replace(sample.history[-1], latent=latent, content_sha256=tensor_sha256(latent))
        sample = replace(sample, history=(*sample.history[:-1], different))
    elif change == "license":
        licenses.clear()
    elif change == "consent":
        consents.clear()
    else:
        sample = replace(sample, influence_labels=(1,) * 8)

    with pytest.raises(ValueError, match=message):
        audit_manifest((sample,), license_rows=licenses, consent_rows=consents)


def test_history_batch_adapter_preserves_canonical_slots_without_reordering() -> None:
    sample = _sample("train", 0)
    batch = HistoryBatch.from_sample(sample)
    mandatory = LTXLatentGuide(torch.ones(1, 128, 1, 2, 2), 0.75, 0)
    mandatory_provenance = LTXGuideProvenance("mandatory", 0, ("current",))

    method_batch = batch.to_method_batch((2, 7), mandatory, mandatory_provenance)

    assert method_batch.item_ids == tuple(item.item_id for item in sample.history)
    assert method_batch.evidence == tuple(item.evidence for item in sample.history)
    assert tuple(row.leaf.slot for row in method_batch.evidence) == tuple(range(8))
    assert torch.equal(method_batch.history[:, 0], sample.history[0].latent)
    assert torch.equal(method_batch.history[:, 7], sample.history[7].latent)
    assert method_batch.selected_indices == (2, 7)


def test_history_batch_rejects_metadata_that_does_not_cover_all_eight_slots() -> None:
    batch = HistoryBatch.from_sample(_sample("train", 0))

    with pytest.raises(ValueError, match="eight"):
        replace(batch, source_sha256=batch.source_sha256[:-1]).validate()


def test_audit_rejects_latent_mutation_after_content_hashing() -> None:
    sample = _sample("train", 0)
    sample = replace(
        sample,
        history=tuple(
            replace(item, content_sha256=tensor_sha256(item.latent)) for item in sample.history
        ),
    )
    sample.history[0].latent.add_(1.0)

    with pytest.raises(ValueError, match="content_sha256"):
        audit_manifest(
            (sample,),
            license_rows={sample.license_id},
            consent_rows={sample.consent_id},
        )
