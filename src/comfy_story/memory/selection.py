"""Deterministic bounded selection of exact evidence."""

from __future__ import annotations

from typing import TypeAlias

from comfy_story.memory.contracts import ExceptionItem, SparseMemoryContract

RankKey: TypeAlias = tuple[int, int, int, int, int, str, tuple[int, ...], str, str]


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
