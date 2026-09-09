from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import pytest
import torch

from comfy_story.config import Method
from comfy_story.contracts import (
    DuetXContract,
    EvidenceProvenance,
    LeafKey,
    RawEvidencePointer,
    SpatialLocation,
    TimeFrameRange,
)
from comfy_story.guide_boundary import (
    GuideRole,
    LTXGuideProvenance,
    LTXLatentGuide,
    apply_history_guides,
)
from comfy_story.methods import MethodBatch, build_method, run_method


def _evidence(slot: int) -> EvidenceProvenance:
    digest = f"{slot + 1:064x}"
    memory_contract = DuetXContract.default(decision1_fingerprint="0" * 64).sparse.fingerprint()
    return EvidenceProvenance(
        LeafKey(f"camera-{slot}", slot, slot),
        TimeFrameRange(slot * 10 + 1, slot * 10 + 2, slot, slot + 1),
        SpatialLocation("normalized-v1", "bbox", (0, 0, 65_536, 65_536)),
        "history-v1",
        RawEvidencePointer(f"duet-evidence://test/sha256/{digest}", digest, 0, 1),
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        memory_contract,
    )


@pytest.fixture
def batch() -> MethodBatch:
    history = torch.zeros(1, 8, 128, 1, 1, 1)
    values = (0.0, 100.0, 2.0, 3.0, 4.0, 5.0, 600.0, 7.0)
    for index, value in enumerate(values):
        history[:, index].fill_(value)
    return MethodBatch(
        history=history,
        item_ids=tuple(f"history-{index}" for index in range(8)),
        evidence=tuple(_evidence(index) for index in range(8)),
        selected_indices=(1, 6),
        mandatory=LTXLatentGuide(torch.full((1, 128, 1, 1, 1), 9.0), 0.75, 0),
        mandatory_provenance=LTXGuideProvenance("mandatory", 0, ("current-guide",)),
    )


def test_full_history_teacher_returns_all_eight_raw_guides(batch: MethodBatch) -> None:
    output = run_method(build_method(Method.FULL_HISTORY_TEACHER), batch)

    assert output.mode == "full_history_teacher"
    assert output.core is None
    assert len(output.exceptions) == 8
    for index, guide in enumerate(output.exceptions):
        assert torch.equal(guide.latent, batch.history[:, index])
    assert output.to_payload().materialized()[-1] == ("mandatory", batch.mandatory)


def test_topk_only_has_no_synthesized_core(batch: MethodBatch) -> None:
    output = run_method(build_method(Method.TOPK_ONLY), batch)

    assert output.mode == "topk_only"
    assert output.core is None
    assert tuple(guide.ordinal for guide in output.exceptions) == (0, 1)
    assert tuple(sidecar.item_ids for sidecar in output.provenance[:-1]) == (
        ("history-1",),
        ("history-6",),
    )
    assert len(output.to_payload().materialized()) == 3


def test_method_output_rejects_a_method_label_from_another_payload_mode(
    batch: MethodBatch,
) -> None:
    output = run_method(build_method(Method.FULL_HISTORY_TEACHER), batch)

    with pytest.raises(ValueError, match=r"method.*mode"):
        replace(output, method=Method.TOPK_ONLY)


def test_recent_and_mean_cores_use_only_the_six_unselected_items(batch: MethodBatch) -> None:
    recent = run_method(build_method(Method.RECENT_ANCHOR_TOPK), batch)
    mean = run_method(build_method(Method.MEAN_CORE_TOPK), batch)

    assert recent.core is not None
    assert mean.core is not None
    assert torch.equal(recent.core.latent, torch.full_like(recent.core.latent, 7.0))
    assert torch.equal(mean.core.latent, torch.full_like(mean.core.latent, 3.5))
    expected_core_ids = (
        "history-0",
        "history-2",
        "history-3",
        "history-4",
        "history-5",
        "history-7",
    )
    assert recent.provenance[0].item_ids == expected_core_ids
    assert mean.provenance[0].item_ids == expected_core_ids


@pytest.mark.parametrize(
    "method",
    [
        Method.GATED_CORE_TOPK,
        Method.CENTRALIZED_RESAMPLER_TOPK,
        Method.DUET_CORE_TOPK,
    ],
)
def test_trainable_cores_cannot_read_selected_items(method: Method, batch: MethodBatch) -> None:
    torch.manual_seed(31)
    module = build_method(method)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.02)
    changed_history = batch.history.clone()
    changed_history[:, 1].copy_(torch.randn_like(changed_history[:, 1]) * 10_000)
    changed_history[:, 6].copy_(torch.randn_like(changed_history[:, 6]) * 10_000)

    original = run_method(module, batch)
    changed = run_method(module, replace(batch, history=changed_history))

    assert original.core is not None
    assert changed.core is not None
    assert torch.equal(original.core.latent, changed.core.latent)


def test_duet_quality_path_cannot_fall_back_to_central_six_item_compress(
    monkeypatch: pytest.MonkeyPatch, batch: MethodBatch
) -> None:
    module = build_method(Method.DUET_CORE_TOPK)

    def forbidden(_history: torch.Tensor) -> torch.Tensor:
        raise AssertionError("central six-item Duet path was called")

    monkeypatch.setattr(module, "compress", forbidden)

    output = run_method(module, batch)

    assert output.core is not None


@pytest.mark.parametrize("method", tuple(Method)[1:])
def test_bounded_method_guide_budget(method: Method, batch: MethodBatch) -> None:
    output = run_method(build_method(method), batch)

    assert len(output.exceptions) == 2
    assert int(output.core is not None) + len(output.exceptions) <= 3
    assert len(output.to_payload().materialized()) in {3, 4}


def test_trainable_method_parameter_counts_are_matched_without_padding() -> None:
    modules = tuple(
        build_method(method)
        for method in (
            Method.GATED_CORE_TOPK,
            Method.CENTRALIZED_RESAMPLER_TOPK,
            Method.DUET_CORE_TOPK,
        )
    )

    counts = tuple(module.trainable_parameter_count() for module in modules)

    assert all(count > 0 for count in counts)
    assert (max(counts) - min(counts)) / max(counts) <= 0.05
    assert counts == tuple(
        sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        for module in modules
    )


class RecordingBackend:
    def __init__(self) -> None:
        self.roles: list[GuideRole] = []
        self.latents: list[torch.Tensor] = []

    def append_guide(
        self,
        positive: Any,
        negative: Any,
        latent: Mapping[str, object],
        guide: LTXLatentGuide,
        *,
        role: GuideRole,
    ) -> tuple[Any, Any, Mapping[str, object]]:
        self.roles.append(role)
        self.latents.append(guide.latent)
        return positive, negative, latent


@pytest.mark.parametrize("method", tuple(Method))
def test_every_method_preserves_protected_bytes_and_append_order(
    method: Method, batch: MethodBatch
) -> None:
    before = tuple(batch.history[:, index].clone() for index in batch.selected_indices)
    output = run_method(build_method(method), batch)
    backend = RecordingBackend()

    receipt = apply_history_guides(
        [],
        [],
        {"samples": torch.zeros_like(batch.mandatory.latent)},
        output.to_payload(),
        backend,
    )

    if method is Method.FULL_HISTORY_TEACHER:
        assert backend.roles == [*(["exception"] * 8), "mandatory"]
        assert receipt.guide_count == 9
    elif method is Method.TOPK_ONLY:
        assert backend.roles == ["exception", "exception", "mandatory"]
        assert receipt.guide_count == 3
    else:
        assert backend.roles == ["core", "exception", "exception", "mandatory"]
        assert receipt.guide_count == 4
    if method is not Method.FULL_HISTORY_TEACHER:
        for expected, protected in zip(before, output.exceptions, strict=True):
            assert torch.equal(protected.latent, expected)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"history": torch.zeros(1, 7, 128, 1, 1, 1)}, "history"),
        ({"item_ids": tuple("duplicate" for _ in range(8))}, "item_ids"),
        ({"selected_indices": (6, 1)}, "canonical"),
        ({"selected_indices": (1, 1)}, "distinct"),
        ({"mandatory": LTXLatentGuide(torch.zeros(1, 128, 1, 1, 2), 1.0, 0)}, "shape"),
        (
            {"mandatory_provenance": LTXGuideProvenance("mandatory", 0, ("history-4",))},
            "disjoint",
        ),
    ],
)
def test_method_batch_rejects_invalid_locked_contracts(
    batch: MethodBatch, change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(batch, **cast(Any, change))
