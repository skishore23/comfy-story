"""Attach authenticated associative history to the existing RGB evidence archive."""

from __future__ import annotations

import hashlib
import io
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch
from PIL import Image

from comfy_story.memory.cache import AuthoritativeLeaf, MemoryProductTree
from comfy_story.memory.contracts import (
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    MemoryContract,
    RawEvidencePointer,
    SpatialLocation,
    TimeFrameRange,
)
from comfy_story.memory.history import (
    _validate_blocks,
    append_story_leaf,
    apply_story_memory_policy,
    compose_inclusive_story_core,
    empty_story_blocks,
)
from comfy_story.memory.runtime import MiniMaxH3StoryRuntime
from comfy_story.memory.selection import canonicalize_items
from comfy_story.memory.settings import InspectionCodec, MemoryConfiguration
from comfy_story.memory.snapshot import decode_runtime_snapshot, encode_runtime_snapshot
from comfy_story.story_observations import normalize_story_frame
from comfy_story.story_product_contracts import GuideBinding, MemoryAction, StoryObservationPacket
from comfy_story.story_store import StoryProjectStore
from comfy_story.tensors import tensor_sha256

if TYPE_CHECKING:
    from comfy_story.story_native_archive import NativeArchiveRevision
    from comfy_story.story_service import PreparedStoryGeneration


def _parent_blocks(
    parent: NativeArchiveRevision,
    config: MemoryConfiguration,
    store: StoryProjectStore,
    contract: MemoryContract,
) -> tuple[MemoryProductTree, ...]:
    data = parent.shot_metadata.get("associative_memory")
    if not isinstance(data, dict) or set(data) != {"configuration", "blocks", "context_sha256"}:
        raise ValueError("parent has no complete associative memory; start a new story")
    if data["configuration"] != config.binding():
        raise ValueError(
            "associative memory checkpoint or model identity changed; start a new story"
        )
    digests = data["blocks"]
    if (
        not isinstance(digests, list)
        or len(digests) != 16
        or any(not isinstance(x, str) for x in digests)
    ):
        raise ValueError("associative memory requires sixteen block snapshots")
    blocks = tuple(
        decode_runtime_snapshot(
            store.load_asset(digest),
            expected_contract=contract,
            expected_model_checkpoint_sha256=config.checkpoint_sha256,
        )
        for digest in digests
    )
    if _validate_blocks(blocks)[3] != parent.state.shot_count:
        raise ValueError("associative memory shot count does not match its parent")
    return blocks


def prepare_memory_guides(
    prepared: PreparedStoryGeneration, store: StoryProjectStore
) -> PreparedStoryGeneration:
    config, parent = prepared.request.associative_memory, prepared.parent
    if config is None:
        if parent is not None and "associative_memory" in parent.shot_metadata:
            raise ValueError("this story requires its associative memory configuration")
        return prepared
    if not isinstance(config, MemoryConfiguration):
        raise ValueError("associative_memory must be a MemoryConfiguration")
    runtime = config.load(InspectionCodec())
    try:
        if parent is None:
            return prepared
        _parent_blocks(parent, config, store, runtime.identity.contract)
        data = cast(dict[str, object], parent.shot_metadata["associative_memory"])
        context = data["context_sha256"]
        if not isinstance(context, str):
            raise ValueError("associative context digest is missing")
        encoded = store.load_asset(context)
        # Avoid replaying an inherited scene across cuts or using a stale image
        # while a forget command removes its dense contribution at commit.
        if (
            prepared.request.composition != "Continue frame"
            or prepared.request.render_profile != "Reference shot"
            or any(
                command.action is MemoryAction.FORGET
                for command in prepared.request.memory_commands
            )
        ):
            return prepared
        from comfy_story.story_service import _comfy_from_png

        guide = _comfy_from_png(encoded)
        if len(prepared.visual_guides) >= 9:
            raise ValueError("associative history needs one of H3's nine image slots")
        guides = (*prepared.visual_guides, guide)
        roles = (*prepared.visual_roles, "associative-history")
        decision = prepared.recall_decision
        if decision is None:
            raise ValueError("associative memory requires a recall decision")
        decision = replace(
            decision,
            guide_bindings=(
                *decision.guide_bindings,
                GuideBinding(roles[-1], tensor_sha256(guide)),
            ),
        ).validate()
        return replace(
            prepared,
            visual_guides=guides,
            visual_roles=roles,
            recall_decision=decision,
            resolved_prompt=prepared.resolved_prompt
            + f" <Picture {len(guides)}> supplies historical visual context. "
            "Preserve relevant continuity; approved exact evidence "
            "and the new direction take precedence.",
        )
    finally:
        runtime.close()


@torch.inference_mode()
def append_associative_memory(
    prepared: PreparedStoryGeneration,
    packet: StoryObservationPacket,
    images: torch.Tensor,
    store: StoryProjectStore,
    runtime: MiniMaxH3StoryRuntime,
) -> dict[str, object]:
    config = prepared.request.associative_memory
    if config is None or prepared.product_state is None:
        raise ValueError("associative memory preparation is missing")
    if (
        runtime.checkpoint.checkpoint_sha256 != config.checkpoint_sha256
        or runtime.vae_sha256 != config.vae_sha256
        or runtime.checkpoint.foundation_sha256 != config.foundation_sha256
        or runtime.checkpoint.model_configuration_sha256 != config.model_configuration_sha256
    ):
        raise ValueError("associative memory runtime does not match the prepared request")
    contract = runtime.identity.contract
    blocks = (
        None
        if prepared.parent is None
        else _parent_blocks(prepared.parent, config, store, contract)
    )
    if blocks is not None and prepared.parent is not None:
        product = prepared.product_state
        excluded = frozenset(
            record.source_shot_index
            for record in product.evidence_records
            if record.evidence_id in product.policy.tombstoned_evidence_ids
            and record.source_shot_index is not None
        )
        blocks = apply_story_memory_policy(
            blocks,
            prepared.parent.product_state.policy,
            product.policy,
            excluded_source_ranks=excluded,
        )
    key = LeafKey(f"shot-{packet.shot_index:03d}", packet.shot_index, packet.shot_index % 8)
    operators = []
    candidates = []
    last_latent = None
    for index, observation in enumerate(packet.observations):
        frame = runtime.materialize_frame(
            normalize_story_frame(images[observation.frame_index : observation.frame_index + 1])
        )
        latent = frame.latent
        last_latent = latent
        operators.append(runtime.bridge.encode_operators(latent.unsqueeze(1))[:, 0])
        asset = store.load_asset(observation.asset_sha256)
        provenance = EvidenceProvenance(
            key,
            TimeFrameRange(
                observation.timestamp_start_ns,
                observation.timestamp_stop_ns,
                observation.frame_index,
                observation.frame_index + 1,
            ),
            SpatialLocation("image", "bbox", (0, 0, 65536, 65536)),
            observation.kind.value,
            RawEvidencePointer(
                f"comfy-evidence://story/sha256/{observation.asset_sha256}",
                observation.asset_sha256,
                0,
                len(asset),
            ),
            hashlib.sha256(prepared.library.to_json()).hexdigest(),
            frame.preprocessing_sha256,
            frame.vae_sha256,
            frame.adapter_sha256,
            frame.scorer_sha256,
            contract.sparse.fingerprint(),
        )
        score = runtime.score(latent, index, observation.timestamp_start_ns / 1e9)
        candidates.append(
            ExceptionItem(
                observation.evidence_id,
                provenance,
                latent.mean(dim=(2, 3, 4))[0],
                max(-(2**63), min(2**63 - 1, round(score * 1_000_000))),
                False,
            )
        )
    if last_latent is None:
        raise ValueError("associative memory requires shot observations")
    operator = runtime.bridge.fusion.combine_ordered_operators(tuple(operators))
    leaf = AuthoritativeLeaf(
        key, packet.source_video_sha256, 0, operator, canonicalize_items(tuple(candidates))
    )
    if blocks is None:
        blocks = empty_story_blocks(
            contract, dense_steps=operator.shape[1], operator_size=operator.shape[-1]
        )
    blocks = append_story_leaf(blocks, shot_index=packet.shot_index, leaf=leaf)
    core = compose_inclusive_story_core(blocks, runtime.bridge.fusion)
    latent = runtime.bridge.materialize_operator(core, anchor=last_latent)
    context = io.BytesIO()
    Image.fromarray(runtime.decode(latent)).save(context, format="PNG")
    return {
        "configuration": config.binding(),
        "context_sha256": store.put_asset(context.getvalue()),
        "blocks": [
            store.put_asset(
                encode_runtime_snapshot(block, model_checkpoint_sha256=config.checkpoint_sha256)
            )
            for block in blocks
        ],
    }


def verify_memory_revision(
    revision: NativeArchiveRevision, config: MemoryConfiguration | None, store: StoryProjectStore
) -> None:
    """Validate recovered memory assets even when no new shot will be sampled."""
    if config is None:
        if "associative_memory" in revision.shot_metadata:
            raise ValueError("recovered story requires its associative memory configuration")
        return
    runtime = config.load(InspectionCodec())
    try:
        _parent_blocks(revision, config, store, runtime.identity.contract)
        data = cast(dict[str, object], revision.shot_metadata["associative_memory"])
        digest = data["context_sha256"]
        if not isinstance(digest, str):
            raise ValueError("associative context digest is missing")
        store.load_asset(digest)
    finally:
        runtime.close()
