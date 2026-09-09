"""Hierarchical 128-shot memory built from sixteen validated Associative memory blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from comfy_story.memory.cache import AuthoritativeLeaf, MemoryProductTree
from comfy_story.memory.contracts import ExceptionItem, LeafKey, MemoryContract
from comfy_story.memory.exclusion import ExclusionKey, ExclusionMemoryProductTree
from comfy_story.memory.fusion import MatrixSemigroupFusion
from comfy_story.memory.selection import canonicalize_items, merge_top_k
from comfy_story.story_product_contracts import StoryMemoryPolicy

_BLOCK_COUNT = 16
_BLOCK_SIZE = 8
_SHOT_CAPACITY = _BLOCK_COUNT * _BLOCK_SIZE


@dataclass(frozen=True, slots=True)
class StoryMemoryRoot:
    """One exclusion-aware ordered root over all occupied story shots."""

    core_operator: torch.Tensor
    exceptions: tuple[ExceptionItem, ...]
    excluded_leaf_keys: tuple[LeafKey, ...]
    occupied_shots: int


def empty_story_blocks(
    contract: MemoryContract,
    *,
    dense_steps: int,
    operator_size: int,
) -> tuple[MemoryProductTree, ...]:
    """Create sixteen independent identity-filled eight-slot snapshot trees."""
    contract.validate()
    return tuple(
        MemoryProductTree(contract, dense_steps, operator_size) for _ in range(_BLOCK_COUNT)
    )


def _validate_blocks(
    blocks: tuple[MemoryProductTree, ...],
) -> tuple[MemoryContract, int, int, int]:
    if not isinstance(blocks, tuple) or len(blocks) != _BLOCK_COUNT:
        raise ValueError("story memory requires exactly sixteen Associative memory blocks")
    if any(not isinstance(block, MemoryProductTree) for block in blocks):
        raise ValueError("story memory blocks must be MemoryProductTree values")
    first = blocks[0]
    contract = first.contract
    dense_steps = first.root.dense_operator.shape[1]
    operator_size = first.root.dense_operator.shape[-1]
    occupied: list[int] = []
    for block_index, block in enumerate(blocks):
        if block.contract != contract or block.leaf_count != _BLOCK_SIZE:
            raise ValueError("story memory blocks must share the eight-slot contract")
        if tuple(block.root.dense_operator.shape[1:]) != (
            dense_steps,
            operator_size,
            operator_size,
        ):
            raise ValueError("story memory blocks must share one operator boundary")
        for local_slot, leaf in enumerate(block.leaves):
            if leaf is None:
                continue
            global_slot = block_index * _BLOCK_SIZE + local_slot
            if leaf.key.slot != local_slot or leaf.key.source_rank != global_slot:
                raise ValueError("story leaf key does not match its global ordered slot")
            occupied.append(global_slot)
    if occupied != list(range(len(occupied))):
        raise ValueError("story memory occupied shots must form one contiguous prefix")
    return contract, dense_steps, operator_size, len(occupied)


def append_story_leaf(
    blocks: tuple[MemoryProductTree, ...],
    *,
    shot_index: int,
    leaf: AuthoritativeLeaf,
) -> tuple[MemoryProductTree, ...]:
    """Return a new block tuple with one leaf appended at the next global slot."""
    contract, dense_steps, operator_size, occupied = _validate_blocks(blocks)
    if type(shot_index) is not int or not 0 <= shot_index < _SHOT_CAPACITY:
        raise ValueError("story memory has a 128-shot capacity")
    if shot_index != occupied:
        raise ValueError("shot_index must be the next contiguous story slot")
    if not isinstance(leaf, AuthoritativeLeaf):
        raise ValueError("story append requires an AuthoritativeLeaf")
    local_slot = shot_index % _BLOCK_SIZE
    if leaf.key.slot != local_slot or leaf.key.source_rank != shot_index:
        raise ValueError("story leaf key must bind the local slot and global source rank")
    block_index = shot_index // _BLOCK_SIZE
    original = blocks[block_index]
    replacement = MemoryProductTree(
        contract,
        dense_steps,
        operator_size,
        tuple(value for value in original.leaves if value is not None),
    )
    replacement.insert(leaf)
    return (*blocks[:block_index], replacement, *blocks[block_index + 1 :])


def compose_story_blocks(
    blocks: tuple[MemoryProductTree, ...],
    fusion: MatrixSemigroupFusion,
) -> StoryMemoryRoot:
    """Select global Top-K and combine each block's matching exclusion variant."""
    contract, _dense_steps, operator_size, occupied = _validate_blocks(blocks)
    if not isinstance(fusion, MatrixSemigroupFusion):
        raise ValueError("story composition requires MatrixSemigroupFusion")
    if fusion.channels != contract.latent_channels or fusion.operator_size != operator_size:
        raise ValueError("story blocks do not match the fusion boundary")
    roots = tuple(
        ExclusionMemoryProductTree(
            contract,
            block.root.dense_operator.shape[1],
            operator_size,
            tuple(leaf for leaf in block.leaves if leaf is not None),
        ).root
        for block in blocks
    )
    exceptions: tuple[ExceptionItem, ...] = ()
    for root in roots:
        exceptions = merge_top_k(exceptions, root.exceptions, contract.sparse)
    selected = frozenset(ExclusionKey.from_item(item) for item in exceptions)
    operators = tuple(
        root.operator_for(tuple(key for key in root.candidate_keys if key in selected))
        for root in roots
    )
    combined = fusion.combine_ordered_operators(operators)
    return StoryMemoryRoot(
        core_operator=combined,
        exceptions=exceptions,
        excluded_leaf_keys=tuple(item.provenance.leaf for item in exceptions),
        occupied_shots=occupied,
    )


def compose_inclusive_story_core(
    blocks: tuple[MemoryProductTree, ...],
    fusion: MatrixSemigroupFusion,
) -> torch.Tensor:
    """Compose all occupied history into the product core without exclusions."""
    contract, _dense_steps, operator_size, _occupied = _validate_blocks(blocks)
    if not isinstance(fusion, MatrixSemigroupFusion):
        raise ValueError("story composition requires MatrixSemigroupFusion")
    if fusion.channels != contract.latent_channels or fusion.operator_size != operator_size:
        raise ValueError("story blocks do not match the fusion boundary")
    return fusion.combine_ordered_operators(tuple(block.root.dense_operator for block in blocks))


def story_evidence_reservoir(
    blocks: tuple[MemoryProductTree, ...],
) -> tuple[ExceptionItem, ...]:
    """Expose every block-local Top-K in chronological block order."""
    _validate_blocks(blocks)
    return tuple(item for block in blocks for item in block.root.exceptions)


def _copy_candidate(item: ExceptionItem, *, pinned: bool) -> ExceptionItem:
    return ExceptionItem(
        item.item_id,
        item.provenance,
        item.embedding,
        item.salience_q,
        pinned,
    )


def apply_story_memory_policy(
    blocks: tuple[MemoryProductTree, ...],
    previous_policy: StoryMemoryPolicy,
    next_policy: StoryMemoryPolicy,
    *,
    excluded_source_ranks: frozenset[int] = frozenset(),
) -> tuple[MemoryProductTree, ...]:
    """Apply durable evidence policy, omitting forgotten leaves from dense context.

    An observation cannot be removed from an already fused leaf without its source
    operators. Conservatively replace that leaf's dense contribution by identity,
    preserving its chronological slot and all other exact evidence. Source ranks
    also repair older snapshots that removed candidates but retained their operator.
    """
    contract, dense_steps, operator_size, _occupied = _validate_blocks(blocks)
    previous_policy.validate()
    next_policy.validate()
    all_ids = {
        item.item_id
        for block in blocks
        for leaf in block.leaves
        if leaf is not None
        for item in leaf.candidates
    }
    requested = set(next_policy.pinned_evidence_ids) | set(next_policy.tombstoned_evidence_ids)
    if not isinstance(excluded_source_ranks, frozenset) or any(
        type(rank) is not int or not 0 <= rank < _SHOT_CAPACITY for rank in excluded_source_ranks
    ):
        raise ValueError("excluded source ranks must name valid story slots")
    previous_tombstones = set(previous_policy.tombstoned_evidence_ids)
    if not previous_tombstones <= set(next_policy.tombstoned_evidence_ids):
        raise ValueError("forgotten evidence cannot be resurrected")
    unknown = requested - all_ids - previous_tombstones
    if unknown:
        raise ValueError(f"memory policy names unknown evidence: {min(unknown)}")
    changed = (set(previous_policy.pinned_evidence_ids) ^ set(next_policy.pinned_evidence_ids)) | (
        set(previous_policy.tombstoned_evidence_ids) ^ set(next_policy.tombstoned_evidence_ids)
    )
    if not changed and not excluded_source_ranks:
        return blocks
    pinned_ids = set(next_policy.pinned_evidence_ids)
    tombstoned_ids = set(next_policy.tombstoned_evidence_ids)
    replacements: list[MemoryProductTree] = []
    for block in blocks:
        block_ids = {
            item.item_id for leaf in block.leaves if leaf is not None for item in leaf.candidates
        }
        excludes_dense = any(
            leaf is not None and leaf.key.source_rank in excluded_source_ranks
            for leaf in block.leaves
        )
        if not block_ids & changed and not excludes_dense:
            replacements.append(block)
            continue
        leaves: list[AuthoritativeLeaf] = []
        # Durable retention belongs to policy/assets; the bounded matrix tree may
        # expose only two pinned candidates. Other approved evidence stays recallable.
        retained = sorted(block_ids & pinned_ids)[: contract.sparse.capacity]
        for leaf in block.leaves:
            if leaf is None:
                continue
            candidates = tuple(
                _copy_candidate(item, pinned=item.item_id in retained)
                for item in leaf.candidates
                if item.item_id not in tombstoned_ids
            )
            forgotten = leaf.key.source_rank in excluded_source_ranks or any(
                item.item_id in tombstoned_ids for item in leaf.candidates
            )
            operator = leaf.dense_operator
            if forgotten:
                operator = (
                    torch.eye(operator_size, dtype=operator.dtype, device=operator.device)
                    .expand_as(operator)
                    .clone()
                )
            leaves.append(
                AuthoritativeLeaf(
                    leaf.key,
                    leaf.source_fingerprint,
                    leaf.revision,
                    operator,
                    canonicalize_items(candidates),
                )
            )
        replacements.append(
            MemoryProductTree(
                contract,
                dense_steps,
                operator_size,
                tuple(leaves),
            )
        )
    return tuple(replacements)


__all__ = (
    "StoryMemoryRoot",
    "append_story_leaf",
    "apply_story_memory_policy",
    "compose_inclusive_story_core",
    "compose_story_blocks",
    "empty_story_blocks",
    "story_evidence_reservoir",
)
