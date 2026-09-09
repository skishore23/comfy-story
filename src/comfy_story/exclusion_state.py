"""Mergeable ordered Duet state with bounded whole-leaf exception exclusion.

For fixed exception capacity ``K``, each state retains local deterministic Top-K candidates and
``2**K`` (or fewer) ordered dense variants.  A variant identifies whole retained leaves excluded
from its product.  Parent Top-K winners are necessarily present in the corresponding child Top-K,
so a merge can construct every parent variant from child variants without revisiting raw leaves.

The construction is deliberately bounded to the Decision 1 ``K=2`` canary.  Its exponential
variant storage is not an acceptable implementation for a future ``K=16`` product.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import torch

from comfy_story.cache import AuthoritativeLeaf, CacheUpdate
from comfy_story.contracts import Coverage, DuetXContract, ExceptionItem, LeafKey
from comfy_story.fusion import MatrixSemigroupFusion
from comfy_story.selection import merge_top_k
from comfy_story.state import MaterializedDuetXMemory, _presentation_key, _validate_dense_operator


def _leaf_key(value: LeafKey) -> tuple[int, int, str]:
    return (value.slot, value.source_rank, value.source_id)


@dataclass(frozen=True, slots=True)
class ExclusionKey:
    """Content-bound identity of one whole retained leaf candidate."""

    leaf: LeafKey
    item_id: str
    item_fingerprint: str

    @classmethod
    def from_item(cls, item: ExceptionItem) -> ExclusionKey:
        item.provenance.leaf.validate()
        return cls(item.provenance.leaf, item.item_id, item.fingerprint())


def _exclusion_key(value: ExclusionKey) -> tuple[int, int, str, str, str]:
    return (*_leaf_key(value.leaf), value.item_id, value.item_fingerprint)


def _powerset(values: tuple[ExclusionKey, ...]) -> tuple[tuple[ExclusionKey, ...], ...]:
    return tuple(subset for size in range(len(values) + 1) for subset in combinations(values, size))


@dataclass(frozen=True, slots=True)
class ExclusionVariant:
    """One ordered dense product with an exact content-bound leaf subset absent."""

    excluded: tuple[ExclusionKey, ...]
    operator: torch.Tensor


def _candidate_keys(exceptions: tuple[ExceptionItem, ...]) -> tuple[ExclusionKey, ...]:
    keys = tuple(sorted((ExclusionKey.from_item(item) for item in exceptions), key=_exclusion_key))
    if len({key.leaf for key in keys}) != len(keys):
        raise ValueError("exclusion state permits at most one whole-leaf candidate per LeafKey")
    return keys


@dataclass(frozen=True, slots=True)
class ExclusionDuetXState:
    """Top-K candidates plus all ordered dense products needed to exclude their leaves."""

    contract: DuetXContract
    coverage: Coverage
    exceptions: tuple[ExceptionItem, ...]
    variants: tuple[ExclusionVariant, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.contract, DuetXContract):
            raise ValueError("contract must be a DuetXContract")
        self.contract.validate()
        if not isinstance(self.coverage, Coverage):
            raise ValueError("coverage must be a Coverage")
        self.coverage.validate()
        if self.coverage.stop_slot > self.contract.history_items:
            raise ValueError("coverage exceeds contract history_items")
        if not isinstance(self.exceptions, tuple):
            raise ValueError("exceptions must be a tuple")
        canonical = merge_top_k((), self.exceptions, self.contract.sparse)
        if canonical != self.exceptions:
            raise ValueError("exceptions must be canonical bounded Top-K retention order")
        keys = _candidate_keys(self.exceptions)
        for item in self.exceptions:
            slot = item.provenance.leaf.slot
            if not self.coverage.start_slot <= slot < self.coverage.stop_slot:
                raise ValueError("exception LeafKey.slot must be within state coverage")
        if not isinstance(self.variants, tuple) or not self.variants:
            raise ValueError("exclusion variants must be a nonempty tuple")
        expected_subsets = _powerset(keys)
        if tuple(variant.excluded for variant in self.variants) != expected_subsets:
            raise ValueError("exclusion variants must cover every canonical retained-leaf subset")
        shape: tuple[int, ...] | None = None
        for variant in self.variants:
            if not isinstance(variant, ExclusionVariant):
                raise ValueError("variants must contain ExclusionVariant values")
            _validate_dense_operator(variant.operator, self.contract)
            if shape is None:
                shape = tuple(variant.operator.shape)
            elif tuple(variant.operator.shape) != shape:
                raise ValueError("all exclusion variant operators must have the same shape")

    @property
    def candidate_keys(self) -> tuple[ExclusionKey, ...]:
        return _candidate_keys(self.exceptions)

    @property
    def excluded_leaf_keys(self) -> tuple[LeafKey, ...]:
        """Return the selected whole leaves absent from the materialized root core."""
        return tuple(sorted((key.leaf for key in self.candidate_keys), key=_leaf_key))

    @property
    def full_operator(self) -> torch.Tensor:
        """Return the ordered product with no retained leaf excluded."""
        return self.variants[0].operator

    @property
    def excluded_operator(self) -> torch.Tensor:
        """Return the ordered product excluding every retained exception leaf."""
        return self.variants[-1].operator

    def operator_for(self, excluded: tuple[ExclusionKey, ...]) -> torch.Tensor:
        """Look up one canonical content-bound exclusion subset."""
        canonical = tuple(sorted(excluded, key=_exclusion_key))
        for variant in self.variants:
            if variant.excluded == canonical:
                return variant.operator
        raise ValueError("requested exclusion subset is unavailable in this bounded state")


def _identity_operator(
    contract: DuetXContract,
    *,
    dense_steps: int,
    operator_size: int,
    device: torch.device | None,
) -> torch.Tensor:
    if type(dense_steps) is not int or dense_steps <= 0:
        raise ValueError("dense_steps must be a positive integer")
    if type(operator_size) is not int or operator_size <= 0:
        raise ValueError("operator_size must be a positive integer")
    identity = torch.eye(operator_size, dtype=contract.dense_accumulation_dtype, device=device)
    return (
        identity.reshape(1, 1, operator_size, operator_size)
        .expand(1, dense_steps, operator_size, operator_size)
        .clone()
    )


def exclusion_identity_state(
    contract: DuetXContract,
    coverage: Coverage,
    dense_steps: int,
    operator_size: int,
    *,
    device: torch.device | None = None,
) -> ExclusionDuetXState:
    """Build an absent fixed-interval leaf whose sole variant is dense identity."""
    identity = _identity_operator(
        contract,
        dense_steps=dense_steps,
        operator_size=operator_size,
        device=device,
    )
    return ExclusionDuetXState(contract, coverage, (), (ExclusionVariant((), identity),))


def exclusion_leaf_state(
    contract: DuetXContract,
    slot: int,
    dense_operator: torch.Tensor,
    candidates: tuple[ExceptionItem, ...] = (),
) -> ExclusionDuetXState:
    """Build one whole-leaf state and its operator/identity exclusion variants."""
    if type(slot) is not int or not 0 <= slot < contract.history_items:
        raise ValueError("leaf slot must be within contract history_items")
    if not isinstance(candidates, tuple):
        raise ValueError("candidates must be a tuple")
    if len(candidates) > 1:
        raise ValueError("exclusion leaf accepts exactly zero or one whole-leaf candidate")
    selected = merge_top_k((), candidates, contract.sparse)
    for item in selected:
        if item.provenance.leaf.slot != slot:
            raise ValueError("whole-leaf candidate must match the encoded leaf slot")
    _validate_dense_operator(dense_operator, contract)
    keys = _candidate_keys(selected)
    identity = _identity_operator(
        contract,
        dense_steps=dense_operator.shape[1],
        operator_size=dense_operator.shape[-1],
        device=dense_operator.device,
    )
    variants = tuple(
        ExclusionVariant(subset, dense_operator if not subset else identity)
        for subset in _powerset(keys)
    )
    return ExclusionDuetXState(contract, Coverage(slot, slot + 1), selected, variants)


def merge_exclusion_states(
    left: ExclusionDuetXState, right: ExclusionDuetXState
) -> ExclusionDuetXState:
    """Merge adjacent ranges and construct every parent exclusion as ``right @ left``."""
    if not isinstance(left, ExclusionDuetXState) or not isinstance(right, ExclusionDuetXState):
        raise ValueError("merge requires ExclusionDuetXState operands")
    if left.contract != right.contract:
        raise ValueError("cannot merge exclusion states with different contracts")
    if left.full_operator.shape != right.full_operator.shape:
        raise ValueError("cannot merge exclusion states with different dense shapes")
    if left.coverage.stop_slot != right.coverage.start_slot:
        raise ValueError("exclusion state coverage must be adjacent without gaps or overlaps")
    exceptions = merge_top_k(left.exceptions, right.exceptions, left.contract.sparse)
    parent_keys = _candidate_keys(exceptions)
    left_keys = frozenset(left.candidate_keys)
    right_keys = frozenset(right.candidate_keys)
    variants = []
    for subset in _powerset(parent_keys):
        left_subset = tuple(key for key in subset if key in left_keys)
        right_subset = tuple(key for key in subset if key in right_keys)
        variants.append(
            ExclusionVariant(
                subset,
                right.operator_for(right_subset) @ left.operator_for(left_subset),
            )
        )
    return ExclusionDuetXState(
        left.contract,
        Coverage(left.coverage.start_slot, right.coverage.stop_slot),
        exceptions,
        tuple(variants),
    )


def materialize_exclusion_state(
    state: ExclusionDuetXState, fusion: MatrixSemigroupFusion
) -> MaterializedDuetXMemory:
    """Decode the core with every retained whole-leaf exception absent."""
    if not isinstance(state, ExclusionDuetXState):
        raise ValueError("materialization requires ExclusionDuetXState")
    if not isinstance(fusion, MatrixSemigroupFusion):
        raise ValueError("materialization requires MatrixSemigroupFusion")
    if state.excluded_operator.shape[-1] != fusion.operator_size:
        raise ValueError("dense operator_size does not match MatrixSemigroupFusion")
    if fusion.channels != state.contract.latent_channels:
        raise ValueError("MatrixSemigroupFusion channels do not match the state contract")
    dense_tokens = fusion.decode_operator(state.excluded_operator)
    ordered = tuple(sorted(state.exceptions, key=_presentation_key))
    embeddings = (
        torch.stack(tuple(item.embedding for item in ordered))
        if ordered
        else torch.empty(
            (0, state.contract.sparse.embedding_dim),
            dtype=state.contract.sparse.embedding_dtype,
        )
    )
    return MaterializedDuetXMemory(
        dense_tokens,
        embeddings,
        tuple(item.item_id for item in ordered),
        tuple(item.provenance for item in ordered),
    )


class ExclusionDuetXProductTree:
    """Fixed-slot cached tree for the mergeable exclusion state."""

    def __init__(
        self,
        contract: DuetXContract,
        dense_steps: int,
        operator_size: int,
        leaves: Sequence[AuthoritativeLeaf] = (),
        *,
        leaf_count: int | None = None,
    ) -> None:
        contract.validate()
        count = contract.history_items if leaf_count is None else leaf_count
        if type(count) is not int or count <= 0 or count & (count - 1):
            raise ValueError("leaf_count must be a positive power-of-two integer")
        if count > contract.history_items:
            raise ValueError("leaf_count exceeds contract history_items")
        self._contract = contract
        self._dense_steps = dense_steps
        self._operator_size = operator_size
        self._leaf_count = count
        self._leaves: list[AuthoritativeLeaf | None] = [None] * count
        for leaf in leaves:
            slot = self._validate_leaf(leaf)
            if self._leaves[slot] is not None:
                raise ValueError("duplicate occupied slot")
            self._leaves[slot] = leaf
        self._levels = self._build_levels()

    @classmethod
    def from_states(cls, states: Sequence[ExclusionDuetXState]) -> ExclusionDuetXProductTree:
        """Build an immutable reduction tree directly from adjacent leaf states."""
        values = tuple(states)
        if not values or len(values) & (len(values) - 1):
            raise ValueError("state count must be a positive power of two")
        first = values[0]
        instance = object.__new__(cls)
        instance._contract = first.contract
        instance._dense_steps = first.full_operator.shape[1]
        instance._operator_size = first.full_operator.shape[-1]
        instance._leaf_count = len(values)
        instance._leaves = []
        for index, state in enumerate(values):
            if state.coverage != Coverage(index, index + 1) or state.contract != first.contract:
                raise ValueError("direct leaf states must be canonical adjacent single slots")
        instance._levels = instance._reduce_levels(list(values))
        return instance

    @property
    def root(self) -> ExclusionDuetXState:
        return self._levels[-1][0]

    def replace(self, leaf: AuthoritativeLeaf) -> CacheUpdate:
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
        key.validate()
        if key.slot >= self._leaf_count:
            raise ValueError("leaf key slot is out of range")
        previous = self._leaves[key.slot]
        if previous is None or previous.key != key:
            raise ValueError("cannot delete a missing leaf")
        if expected_revision is not None and previous.revision != expected_revision:
            raise ValueError("expected revision does not match the occupied leaf")
        self._leaves[key.slot] = None
        return self._recompute_path(key.slot)

    def full_rebuild_root(self) -> ExclusionDuetXState:
        return self._reduce_levels(
            [self._state_for_slot(slot) for slot in range(self._leaf_count)]
        )[-1][0]

    def _validate_leaf(self, leaf: AuthoritativeLeaf) -> int:
        if not isinstance(leaf, AuthoritativeLeaf):
            raise ValueError("leaf must be an AuthoritativeLeaf")
        slot = leaf.key.slot
        if slot >= self._leaf_count:
            raise ValueError("leaf key slot is out of range")
        if len(leaf.candidates) > 1:
            raise ValueError("exclusion tree permits one whole-leaf candidate per LeafKey")
        exclusion_leaf_state(
            self._contract,
            slot,
            leaf.dense_operator,
            leaf.candidates,
        )
        return slot

    def _state_for_slot(self, slot: int) -> ExclusionDuetXState:
        leaf = self._leaves[slot]
        if leaf is None:
            return exclusion_identity_state(
                self._contract,
                Coverage(slot, slot + 1),
                self._dense_steps,
                self._operator_size,
            )
        return exclusion_leaf_state(
            self._contract,
            slot,
            leaf.dense_operator,
            leaf.candidates,
        )

    @staticmethod
    def _reduce_levels(
        leaves: list[ExclusionDuetXState],
    ) -> list[list[ExclusionDuetXState]]:
        levels = [leaves]
        while len(levels[-1]) > 1:
            children = levels[-1]
            levels.append(
                [
                    merge_exclusion_states(children[index], children[index + 1])
                    for index in range(0, len(children), 2)
                ]
            )
        return levels

    def _build_levels(self) -> list[list[ExclusionDuetXState]]:
        return self._reduce_levels([self._state_for_slot(slot) for slot in range(self._leaf_count)])

    def _recompute_path(self, slot: int) -> CacheUpdate:
        self._levels[0][slot] = self._state_for_slot(slot)
        child_index = slot
        products = 0
        for depth in range(1, len(self._levels)):
            parent_index = child_index // 2
            children = self._levels[depth - 1]
            self._levels[depth][parent_index] = merge_exclusion_states(
                children[parent_index * 2], children[parent_index * 2 + 1]
            )
            child_index = parent_index
            products += 1
        return CacheUpdate(slot, products, products)


__all__ = (
    "ExclusionDuetXProductTree",
    "ExclusionDuetXState",
    "ExclusionKey",
    "ExclusionVariant",
    "exclusion_identity_state",
    "exclusion_leaf_state",
    "materialize_exclusion_state",
    "merge_exclusion_states",
)
