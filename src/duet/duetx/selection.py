"""Deterministic, mergeable bounded selection for Duet-X exceptions."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TypeAlias

from duet.duetx.config import Selector
from duet.duetx.contracts import SALIENCE_Q_MAX, SALIENCE_Q_MIN, ExceptionItem, SparseMemoryContract

RankKey: TypeAlias = tuple[int, int, int, int, int, str, tuple[int, ...], str, str]

_SCORE_SCALE = 1_000_000
_UNIFORM_SOURCE_RANKS = frozenset((2, 5))
_ENGINEERED_WEIGHTS = (200_000, 350_000, 300_000, 150_000)


@dataclass(frozen=True, slots=True)
class SelectorFeatures:
    """Fixed-point item-local signals for the hand-engineered selector."""

    marker_confidence_q: int
    object_confidence_q: int
    edge_density_q: int
    event_priority_q: int

    def validate(self) -> SelectorFeatures:
        for name in (
            "marker_confidence_q",
            "object_confidence_q",
            "edge_density_q",
            "event_priority_q",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= _SCORE_SCALE:
                raise ValueError(f"{name} must be an integer in range [0, {_SCORE_SCALE}]")
        return self


def item_rank(item: ExceptionItem) -> RankKey:
    """Return the versioned total retention order, best item first."""
    provenance = item.provenance
    return (
        0 if item.pinned else 1,
        -item.salience_q,
        provenance.time.timestamp_start_ns,
        provenance.time.frame_start,
        provenance.source_rank,
        provenance.event_type,
        provenance.spatial.coordinates_q16,
        item.item_id,
        item.fingerprint(),
    )


def canonicalize_items(items: tuple[ExceptionItem, ...]) -> tuple[ExceptionItem, ...]:
    """Deduplicate idempotent IDs and fail closed on conflicting logical IDs."""
    by_id: dict[str, ExceptionItem] = {}
    for item in items:
        previous = by_id.get(item.item_id)
        if previous is None:
            by_id[item.item_id] = item
        elif previous.fingerprint() != item.fingerprint():
            raise ValueError(f"conflicting exception item_id: {item.item_id}")
    return tuple(sorted(by_id.values(), key=item_rank))


def select_top_k(
    items: tuple[ExceptionItem, ...], contract: SparseMemoryContract
) -> tuple[ExceptionItem, ...]:
    """Select bounded winners in retention order, preserving every valid pin."""
    contract.validate()
    canonical = canonicalize_items(items)
    for item in canonical:
        item.validate(contract)
    pins = sum(item.pinned for item in canonical)
    allowed_pins = contract.capacity if contract.max_pinned is None else contract.max_pinned
    if pins > allowed_pins:
        raise ValueError("pinned exception count exceeds the memory contract")
    return canonical[: contract.capacity]


def merge_top_k(
    left: tuple[ExceptionItem, ...],
    right: tuple[ExceptionItem, ...],
    contract: SparseMemoryContract,
) -> tuple[ExceptionItem, ...]:
    """Merge bounded partial selections using the same Top-K homomorphism."""
    return select_top_k(left + right, contract)


def _scored_copy(item: ExceptionItem, score_q: int) -> ExceptionItem:
    if type(score_q) is not int or not SALIENCE_Q_MIN <= score_q <= SALIENCE_Q_MAX:
        raise ValueError("selector score must be a signed int64 integer")
    return replace(item, embedding=item.embedding, salience_q=score_q)


def _validate_seed(seed: int) -> None:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")


def _validated_scores(
    scores: Mapping[str, int] | None, item_ids: frozenset[str], name: str
) -> Mapping[str, int]:
    if scores is None or set(scores) != item_ids:
        raise ValueError(f"{name} must be complete with no extra item IDs")
    for item_id, score in scores.items():
        if type(score) is not int:
            raise ValueError(f"{name} score for {item_id!r} must be an integer")
        if not 0 <= score <= _SCORE_SCALE:
            raise ValueError(f"{name} score for {item_id!r} must be in range [0, {_SCORE_SCALE}]")
    return scores


def _validated_features(
    features: Mapping[str, SelectorFeatures] | None, item_ids: frozenset[str]
) -> Mapping[str, SelectorFeatures]:
    if features is None or set(features) != item_ids:
        raise ValueError("engineered_features must be complete with no extra item IDs")
    for item_id, feature in features.items():
        if not isinstance(feature, SelectorFeatures):
            raise ValueError(f"engineered feature for {item_id!r} must be SelectorFeatures")
        feature.validate()
    return features


def _engineered_score(features: SelectorFeatures) -> int:
    values = (
        features.marker_confidence_q,
        features.object_confidence_q,
        features.edge_density_q,
        features.event_priority_q,
    )
    return (
        sum(weight * value for weight, value in zip(_ENGINEERED_WEIGHTS, values, strict=True))
        // _SCORE_SCALE
    )


def select_candidates(
    selector: Selector,
    candidates: tuple[ExceptionItem, ...],
    contract: SparseMemoryContract,
    seed: int,
    *,
    engineered_features: Mapping[str, SelectorFeatures] | None = None,
    learned_scores_q: Mapping[str, int] | None = None,
    oracle_scores_q: Mapping[str, int] | None = None,
) -> tuple[ExceptionItem, ...]:
    """Score immutable candidates item-locally, then apply the common Top-K selector."""
    _validate_seed(seed)
    canonical = canonicalize_items(candidates)
    item_ids = frozenset(item.item_id for item in canonical)
    if selector is Selector.RANDOM:
        scored = tuple(
            _scored_copy(
                item,
                int.from_bytes(
                    hashlib.sha256(f"{seed}|{item.item_id}".encode()).digest()[:8], "big"
                )
                & SALIENCE_Q_MAX,
            )
            for item in canonical
        )
    elif selector is Selector.RECENT:
        scored = tuple(
            _scored_copy(item, item.provenance.time.timestamp_start_ns) for item in canonical
        )
    elif selector is Selector.UNIFORM:
        scored = tuple(
            _scored_copy(
                item, _SCORE_SCALE if item.provenance.source_rank in _UNIFORM_SOURCE_RANKS else 0
            )
            for item in canonical
        )
    elif selector is Selector.ENGINEERED:
        features = _validated_features(engineered_features, item_ids)
        scored = tuple(
            _scored_copy(item, _engineered_score(features[item.item_id])) for item in canonical
        )
    elif selector is Selector.LEARNED:
        scores = _validated_scores(learned_scores_q, item_ids, "learned_scores_q")
        scored = tuple(_scored_copy(item, scores[item.item_id]) for item in canonical)
    elif selector is Selector.ORACLE:
        scores = _validated_scores(oracle_scores_q, item_ids, "oracle_scores_q")
        scored = tuple(_scored_copy(item, scores[item.item_id]) for item in canonical)
    else:
        raise ValueError(f"unsupported selector: {selector!r}")
    return select_top_k(scored, contract)
