from __future__ import annotations

import itertools
from dataclasses import replace
from typing import cast

import pytest
import torch

from duet.duetx.config import Selector
from duet.duetx.contracts import (
    SALIENCE_Q_MAX,
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    RawEvidencePointer,
    SparseMemoryContract,
    SpatialLocation,
    TimeFrameRange,
)
from duet.duetx.selection import (
    SelectorFeatures,
    canonicalize_items,
    item_rank,
    merge_top_k,
    select_candidates,
    select_top_k,
)


@pytest.fixture
def contract() -> SparseMemoryContract:
    return SparseMemoryContract(2, 4, torch.float32, 2)


@pytest.fixture
def items(contract: SparseMemoryContract) -> tuple[ExceptionItem, ...]:
    result: list[ExceptionItem] = []
    for index in range(5):
        provenance = EvidenceProvenance(
            leaf=LeafKey(f"camera-{index}", index + 1, index),
            time=TimeFrameRange(100 + index, 200 + index, index, index + 1),
            spatial=SpatialLocation("normalized", "bbox", (index, 0, index + 1, 1)),
            event_type=f"event-{index}",
            raw=RawEvidencePointer(
                f"duet-evidence://registry/sha256/{f'{index:x}' * 64}",
                f"{index:x}" * 64,
                0,
                1,
            ),
            source_registry_sha256="1" * 64,
            preprocessing_sha256="2" * 64,
            vae_sha256="3" * 64,
            adapter_sha256="4" * 64,
            scorer_sha256="5" * 64,
            memory_contract_sha256=contract.fingerprint(),
        )
        result.append(
            ExceptionItem(f"item-{index}", provenance, torch.full((4,), float(index)), 10 + index)
        )
    return tuple(result)


def _replace_item(item: ExceptionItem, **changes: object) -> ExceptionItem:
    return ExceptionItem(
        cast(str, changes.get("item_id", item.item_id)),
        cast(EvidenceProvenance, changes.get("provenance", item.provenance)),
        cast(torch.Tensor, changes.get("embedding", item.embedding)),
        cast(int, changes.get("salience_q", item.salience_q)),
        cast(bool, changes.get("pinned", item.pinned)),
    )


def _select(
    selector: Selector,
    candidates: tuple[ExceptionItem, ...],
    contract: SparseMemoryContract,
    features: dict[str, SelectorFeatures],
    scores: dict[str, int],
) -> tuple[ExceptionItem, ...]:
    if selector is Selector.ENGINEERED:
        return select_candidates(
            selector, candidates, contract, seed=17, engineered_features=features
        )
    if selector is Selector.LEARNED:
        return select_candidates(selector, candidates, contract, seed=17, learned_scores_q=scores)
    if selector is Selector.ORACLE:
        return select_candidates(selector, candidates, contract, seed=17, oracle_scores_q=scores)
    return select_candidates(selector, candidates, contract, seed=17)


def test_hierarchical_topk_equals_centralized(
    items: tuple[ExceptionItem, ...], contract: SparseMemoryContract
) -> None:
    expected = select_top_k(items, contract)
    expected_ids = tuple(item.item_id for item in expected)
    for permutation in itertools.permutations(items):
        left = merge_top_k(permutation[:2], permutation[2:4], contract)
        actual = merge_top_k(left, permutation[4:], contract)
        assert tuple(item.item_id for item in actual) == expected_ids


def test_pin_overflow_fails(
    items: tuple[ExceptionItem, ...], contract: SparseMemoryContract
) -> None:
    pinned = tuple(_replace_item(item, pinned=True) for item in items[:3])
    with pytest.raises(ValueError, match="pinned"):
        select_top_k(pinned, contract)


def test_canonicalize_deduplicates_identical_item_id_and_rejects_conflicts(
    items: tuple[ExceptionItem, ...],
) -> None:
    assert canonicalize_items((items[1], items[1], items[0])) == (items[1], items[0])
    conflicting = _replace_item(items[1], embedding=items[1].embedding.add(1))
    with pytest.raises(ValueError, match="conflicting"):
        canonicalize_items((items[1], conflicting))


def test_item_rank_has_pins_then_salience_and_stable_ties(items: tuple[ExceptionItem, ...]) -> None:
    pinned = _replace_item(items[0], pinned=True, salience_q=-100)
    tied = _replace_item(items[1], salience_q=10)
    assert item_rank(pinned) < item_rank(items[-1])
    assert item_rank(items[0]) < item_rank(tied)


@pytest.mark.parametrize("selector", tuple(Selector))
def test_every_selector_is_permutation_independent_and_preserves_inputs(
    selector: Selector, items: tuple[ExceptionItem, ...], contract: SparseMemoryContract
) -> None:
    features = {
        item.item_id: SelectorFeatures(
            100_000 + index, 200_000 + index, 300_000 + index, 400_000 + index
        )
        for index, item in enumerate(items)
    }
    scores = {item.item_id: 1_000_000 - index for index, item in enumerate(items)}
    fingerprints = tuple(item.fingerprint() for item in items)
    expected = _select(selector, items, contract, features, scores)
    actual = _select(selector, tuple(reversed(items)), contract, features, scores)
    assert tuple(item.item_id for item in actual) == tuple(item.item_id for item in expected)
    assert tuple(item.fingerprint() for item in items) == fingerprints


def test_random_uses_seed_and_not_process_rng(
    items: tuple[ExceptionItem, ...], contract: SparseMemoryContract
) -> None:
    selected_a = select_candidates(Selector.RANDOM, items, contract, seed=17)
    selected_b = select_candidates(Selector.RANDOM, items, contract, seed=17)
    selected_c = select_candidates(Selector.RANDOM, items, contract, seed=18)
    assert selected_a == selected_b
    assert tuple(item.item_id for item in selected_a) != tuple(item.item_id for item in selected_c)
    assert all(0 <= item.salience_q <= SALIENCE_Q_MAX for item in selected_a)


def test_recent_selector_rejects_timestamp_outside_signed_int64(
    items: tuple[ExceptionItem, ...], contract: SparseMemoryContract
) -> None:
    provenance = replace(
        items[0].provenance,
        time=TimeFrameRange(SALIENCE_Q_MAX + 1, SALIENCE_Q_MAX + 2, 0, 1),
    )
    outside = _replace_item(items[0], provenance=provenance)

    with pytest.raises(ValueError, match="signed int64"):
        select_candidates(Selector.RECENT, (outside,), contract, seed=1)


def test_learned_and_oracle_require_complete_valid_score_maps(
    items: tuple[ExceptionItem, ...], contract: SparseMemoryContract
) -> None:
    with pytest.raises(ValueError, match="complete"):
        select_candidates(Selector.LEARNED, items, contract, seed=1)
    with pytest.raises(ValueError, match="complete"):
        select_candidates(Selector.ORACLE, items, contract, seed=1)
    invalid = {item.item_id: 0 for item in items}
    invalid[items[0].item_id] = 1_000_001
    with pytest.raises(ValueError, match="range"):
        select_candidates(Selector.LEARNED, items, contract, seed=1, learned_scores_q=invalid)
    with pytest.raises(ValueError, match="range"):
        select_candidates(Selector.ORACLE, items, contract, seed=1, oracle_scores_q=invalid)


def test_engineered_requires_complete_valid_features(
    items: tuple[ExceptionItem, ...], contract: SparseMemoryContract
) -> None:
    with pytest.raises(ValueError, match="complete"):
        select_candidates(Selector.ENGINEERED, items, contract, seed=1)
    invalid = {item.item_id: SelectorFeatures(0, 0, 0, 0) for item in items}
    invalid[items[0].item_id] = SelectorFeatures(1_000_001, 0, 0, 0)
    with pytest.raises(ValueError, match="range"):
        select_candidates(Selector.ENGINEERED, items, contract, seed=1, engineered_features=invalid)
