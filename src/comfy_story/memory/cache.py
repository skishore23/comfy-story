"""Fixed-slot logarithmic updates over authoritative Associative memory history leaves."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field

import torch

from comfy_story.memory.contracts import Coverage, ExceptionItem, LeafKey, MemoryContract
from comfy_story.memory.selection import canonicalize_items
from comfy_story.memory.state import MemoryState, identity_state, leaf_state, merge_states


class _AuthoritativeDenseOperator:
    _dense_operator_bytes: bytes
    _dense_operator_dtype: torch.dtype
    _dense_operator_shape: tuple[int, ...]

    @property
    def dense_operator(self) -> torch.Tensor:
        """Reconstruct a fresh CPU tensor from immutable canonical operator bytes."""
        return torch.frombuffer(
            bytearray(self._dense_operator_bytes), dtype=self._dense_operator_dtype
        ).reshape(self._dense_operator_shape)


@dataclass(frozen=True, slots=True)
class AuthoritativeLeaf(_AuthoritativeDenseOperator):
    """One complete, revisioned source manifest for a fixed history slot.

    ``candidates`` deliberately remains unbounded.  The tree derives a bounded state from this
    manifest at its leaf, so replacing a leaf can expose a candidate that was previously below K.
    """

    key: LeafKey
    source_fingerprint: str
    revision: int
    dense_operator: InitVar[torch.Tensor] = field()
    candidates: tuple[ExceptionItem, ...] = ()
    _dense_operator_bytes: bytes = field(init=False, repr=False)
    _dense_operator_dtype: torch.dtype = field(init=False, repr=False)
    _dense_operator_shape: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self, dense_operator: torch.Tensor) -> None:
        if not isinstance(self.key, LeafKey):
            raise ValueError("key must be a LeafKey")
        self.key.validate()
        _validate_sha256(self.source_fingerprint, "source_fingerprint")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be a nonnegative integer")
        if not isinstance(dense_operator, torch.Tensor) or dense_operator.layout != torch.strided:
            raise ValueError("dense_operator must be a strided tensor")
        canonical_operator = dense_operator.detach().contiguous().cpu()
        object.__setattr__(
            self,
            "_dense_operator_bytes",
            canonical_operator.view(torch.uint8).numpy().tobytes(),
        )
        object.__setattr__(self, "_dense_operator_dtype", canonical_operator.dtype)
        object.__setattr__(self, "_dense_operator_shape", tuple(canonical_operator.shape))
        if not isinstance(self.candidates, tuple):
            raise ValueError("candidates must be a tuple")
        canonical = canonicalize_items(self.candidates)
        if canonical != self.candidates:
            raise ValueError("candidates must be canonical retention order")


@dataclass(frozen=True, slots=True)
class CacheUpdate:
    """The bounded work performed while changing a single fixed slot."""

    slot: int
    products_recomputed: int
    sparse_merges_recomputed: int


class MemoryProductTree:
    """Cache an ordered, padded power-of-two reduction of authoritative history leaves."""

    def __init__(
        self,
        contract: MemoryContract,
        dense_steps: int,
        operator_size: int,
        leaves: Sequence[AuthoritativeLeaf] = (),
        *,
        leaf_count: int | None = None,
    ) -> None:
        if not isinstance(contract, MemoryContract):
            raise ValueError("contract must be a MemoryContract")
        contract.validate()
        _validate_positive_int(dense_steps, "dense_steps")
        _validate_positive_int(operator_size, "operator_size")
        count = contract.history_items if leaf_count is None else leaf_count
        if type(count) is not int or count <= 0 or count & (count - 1):
            raise ValueError("leaf_count must be a positive power-of-two integer")
        if count > contract.history_items:
            raise ValueError("leaf_count exceeds contract history_items")
        if not isinstance(leaves, Sequence):
            raise ValueError("leaves must be a sequence")

        self._contract = contract
        self._dense_steps = dense_steps
        self._operator_size = operator_size
        self._leaf_count = count
        self._leaves: list[AuthoritativeLeaf | None] = [None] * count
        for leaf in leaves:
            self._insert_initial(leaf)
        self._levels: list[list[MemoryState]] = [
            [self._state_for_slot(slot) for slot in range(self._leaf_count)]
        ]
        while len(self._levels[-1]) > 1:
            children = self._levels[-1]
            self._levels.append(
                [
                    merge_states(children[index], children[index + 1])
                    for index in range(0, len(children), 2)
                ]
            )

    @property
    def contract(self) -> MemoryContract:
        """Return the immutable state contract shared by every tree node."""
        return self._contract

    @property
    def leaf_count(self) -> int:
        """Return the fixed padded slot count."""
        return self._leaf_count

    @property
    def leaves(self) -> tuple[AuthoritativeLeaf | None, ...]:
        """Return authoritative leaves by fixed slot; the tuple cannot mutate the cache."""
        return tuple(self._leaves)

    @property
    def root(self) -> MemoryState:
        """Return the bounded cached root state; edits never mutate it directly."""
        return self._levels[-1][0]

    def insert(self, leaf: AuthoritativeLeaf) -> CacheUpdate:
        """Install a new authoritative leaf in an empty fixed slot."""
        slot = self._validate_leaf(leaf)
        if self._leaves[slot] is not None:
            raise ValueError("cannot insert into an occupied slot")
        self._ensure_unique_source(leaf)
        self._leaves[slot] = leaf
        return self._recompute_path(slot)

    def replace(self, leaf: AuthoritativeLeaf) -> CacheUpdate:
        """Replace an occupied leaf only with a newer complete revision."""
        slot = self._validate_leaf(leaf)
        previous = self._leaves[slot]
        if previous is None or previous.key != leaf.key:
            raise ValueError("cannot replace a missing leaf")
        if leaf.source_fingerprint != previous.source_fingerprint:
            raise ValueError("replacement source_fingerprint must remain unchanged")
        if leaf.revision <= previous.revision:
            raise ValueError("replacement revision must be strictly increasing")
        self._leaves[slot] = leaf
        return self._recompute_path(slot)

    def delete(self, key: LeafKey, *, expected_revision: int | None = None) -> CacheUpdate:
        """Clear one authoritative leaf, optionally guarding against a stale revision."""
        slot = self._validate_key(key)
        previous = self._leaves[slot]
        if previous is None or previous.key != key:
            raise ValueError("cannot delete a missing leaf")
        if expected_revision is not None:
            if type(expected_revision) is not int or expected_revision < 0:
                raise ValueError("expected_revision must be a nonnegative integer")
            if previous.revision != expected_revision:
                raise ValueError("expected revision does not match the occupied leaf")
        self._leaves[slot] = None
        return self._recompute_path(slot)

    def full_rebuild_root(self) -> MemoryState:
        """Cold-rebuild the root from complete leaves and exact fixed-slot identities."""
        states = [self._state_for_slot(slot) for slot in range(self._leaf_count)]
        while len(states) > 1:
            states = [
                merge_states(states[index], states[index + 1]) for index in range(0, len(states), 2)
            ]
        return states[0]

    def _insert_initial(self, leaf: AuthoritativeLeaf) -> None:
        slot = self._validate_leaf(leaf)
        if self._leaves[slot] is not None:
            raise ValueError("duplicate occupied slot")
        self._ensure_unique_source(leaf)
        self._leaves[slot] = leaf

    def _validate_leaf(self, leaf: AuthoritativeLeaf) -> int:
        if not isinstance(leaf, AuthoritativeLeaf):
            raise ValueError("leaf must be an AuthoritativeLeaf")
        slot = self._validate_key(leaf.key)
        candidate_source_ids = {item.provenance.leaf.source_id for item in leaf.candidates}
        if candidate_source_ids and candidate_source_ids != {leaf.key.source_id}:
            raise ValueError("authoritative candidates must share the leaf source_id")
        for item in leaf.candidates:
            item.validate(self._contract.sparse)
            if item.provenance.leaf != leaf.key:
                raise ValueError("authoritative candidates must match the leaf key")
        leaf_state(self._contract, slot, leaf.dense_operator, leaf.candidates)
        return slot

    def _validate_key(self, key: LeafKey) -> int:
        if not isinstance(key, LeafKey):
            raise ValueError("key must be a LeafKey")
        key.validate()
        if key.slot >= self._leaf_count:
            raise ValueError("leaf key slot is out of range")
        return key.slot

    def _ensure_unique_source(self, leaf: AuthoritativeLeaf) -> None:
        if any(
            existing is not None
            and existing.key.source_id == leaf.key.source_id
            and existing.key.slot != leaf.key.slot
            for existing in self._leaves
        ):
            raise ValueError("duplicate source_id in authoritative leaves")
        if any(
            existing is not None
            and existing.source_fingerprint == leaf.source_fingerprint
            and existing.key.slot != leaf.key.slot
            for existing in self._leaves
        ):
            raise ValueError("duplicate source_fingerprint in authoritative leaves")

    def _state_for_slot(self, slot: int) -> MemoryState:
        leaf = self._leaves[slot]
        if leaf is None:
            return identity_state(
                self._contract,
                Coverage(slot, slot + 1),
                self._dense_steps,
                self._operator_size,
            )
        return leaf_state(self._contract, slot, leaf.dense_operator, leaf.candidates)

    def _recompute_path(self, slot: int) -> CacheUpdate:
        self._levels[0][slot] = self._state_for_slot(slot)
        products_recomputed = 0
        child_index = slot
        for depth in range(1, len(self._levels)):
            parent_index = child_index // 2
            children = self._levels[depth - 1]
            self._levels[depth][parent_index] = merge_states(
                children[2 * parent_index], children[2 * parent_index + 1]
            )
            child_index = parent_index
            products_recomputed += 1
        return CacheUpdate(slot, products_recomputed, products_recomputed)


def _validate_positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
