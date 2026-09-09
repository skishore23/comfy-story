"""Capture native H3 evidence without running a learned memory sidecar."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from comfy_story.memory.runtime import MiniMaxH3StoryRuntime
from comfy_story.memory.service import append_associative_memory
from comfy_story.story_contracts import ComfyStoryStateRef, canonical_story_json
from comfy_story.story_native_archive import (
    NativeArchiveRevision,
    NativeReferenceArchive,
    NativeReferenceReceipt,
)
from comfy_story.story_observations import extract_story_observation_frames, timeline_start_ns
from comfy_story.story_product_contracts import (
    NativeRGBObservation,
    StoryEvidenceRecord,
    StoryObservationPacket,
)
from comfy_story.story_service import (
    StoryCommitRequest,
    _digest,
    _guide_bindings,
    _scene_entity_ids,
    _sha256,
    append_product_observation,
    validate_comfy_images,
)
from comfy_story.story_store import StoryProjectStore

NATIVE_RGB_CAPTURE_POLICY_SHA256 = _sha256(
    canonical_story_json(
        {
            "format": "comfy-story-native-rgb-capture-v2",
            "frame_selection": "opening-maximum-change-384-square-closing",
            "normalization": "RGB-u8-native-resolution-and-aspect",
            "salience": "not-predicted-zero",
            "vae": None,
            "latent": None,
        }
    )
)


def _native_frame_png(image: torch.Tensor) -> bytes:
    """Keep the selected decoded frame's geometry; compact RGB is for selection only."""
    rgb = image.detach().cpu().mul(255).round().to(torch.uint8).numpy()
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass(frozen=True, slots=True)
class NativeStoryCommitResult:
    state: ComfyStoryStateRef
    last_frame: torch.Tensor
    loaded_revision: NativeArchiveRevision


def commit_native_story_generation(
    request: StoryCommitRequest,
    *,
    store: StoryProjectStore,
    memory_runtime: MiniMaxH3StoryRuntime | None = None,
) -> NativeStoryCommitResult:
    prepared = request.prepared
    settings = prepared.request
    product, decision = prepared.product_state, prepared.recall_decision
    if product is None or decision is None:
        raise ValueError("Native archive preparation is incomplete")
    if _guide_bindings(prepared.visual_roles, prepared.visual_guides) != decision.guide_bindings:
        raise ValueError("Native guide roster changed after preparation")
    execution = _digest(settings.execution_sha256, "execution_sha256")
    video = _digest(request.saved_video_sha256, "video_sha256")
    if request.saved_video_locator != f"comfy-evidence://story/sha256/{video}":
        raise ValueError("saved video locator must match its SHA-256")
    images = validate_comfy_images(request.decoded_images, field="decoded_images")
    last_frame = images[-1:].clone().contiguous()
    shot_index = prepared.parent_shot_count
    timeline_start = timeline_start_ns(product.observation_packets)
    duration_ns = round(images.shape[0] * 1_000_000_000 / 24)
    entity_ids = _scene_entity_ids(settings, decision.explicit_entity_ids)
    observations = []
    for frame in extract_story_observation_frames(
        images, timeline_start_ns=timeline_start, shot_duration_ns=duration_ns
    ):
        digest = store.put_asset(_native_frame_png(images[frame.frame_index]))
        observations.append(
            NativeRGBObservation(
                f"shot-{shot_index:03d}-{frame.kind.value}-{frame.frame_index:03d}-{digest[:12]}",
                frame.kind,
                frame.frame_index,
                frame.timestamp_start_ns,
                frame.timestamp_stop_ns,
                (0, 0, 65536, 65536),
                digest,
                NATIVE_RGB_CAPTURE_POLICY_SHA256,
                0,
                frame.change_score_q,
                entity_ids,
            ).validate()
        )
    packet = StoryObservationPacket(
        f"native-shot-{shot_index:03d}-{video[:12]}",
        shot_index,
        images.shape[0],
        timeline_start,
        timeline_start + duration_ns,
        video,
        request.saved_video_locator,
        NATIVE_RGB_CAPTURE_POLICY_SHA256,
        entity_ids,
        tuple(observations),
    ).validate()
    records = tuple(StoryEvidenceRecord.from_observation(packet, o) for o in observations)
    next_product = append_product_observation(product, packet, records, entity_ids, video)
    full_png = _native_frame_png(last_frame[0])
    tensor_file = io.BytesIO()
    np.save(tensor_file, last_frame.detach().cpu().numpy(), allow_pickle=False)
    metadata: dict[str, object] = {
        "active_references": list(prepared.active_reference_names),
        "intent": settings.intent.value,
        "memory_backend": "minimax-h3-native-reference",
        "prompt_sha256": _sha256(settings.prompt.strip().encode()),
        "sampler": settings.sampler.value,
        "seed": settings.variation,
        "shot_length_seconds": settings.shot_length_seconds,
        "video_sha256": video,
        "execution_sha256": execution,
        "last_frame_full_sha256": store.put_asset(full_png),
        "last_frame_tensor_sha256": store.put_asset(tensor_file.getvalue()),
        "decoded_frame_count": images.shape[0],
        "fps": 24,
        "decoded_duration_ns": duration_ns,
        "declared_scene_entities": list(entity_ids),
        "render_profile": settings.render_profile,
    }
    if settings.associative_memory is not None:
        if memory_runtime is None:
            raise ValueError("associative memory runtime is required before publishing a shot")
        metadata["memory_backend"] = "minimax-h3-associative"
        metadata["associative_memory"] = append_associative_memory(
            prepared, packet, images, store, memory_runtime
        )
    elif memory_runtime is not None:
        raise ValueError("native memory must not receive an associative runtime")
    receipt = NativeReferenceReceipt(
        execution,
        _sha256(settings.prompt.strip().encode()),
        decision.selected_evidence_ids,
        decision.guide_bindings,
    )
    archive = NativeReferenceArchive(store)
    state = archive.publish(
        project_id=settings.project_id,
        branch_id=settings.branch_id,
        parent=None if prepared.parent is None else prepared.parent.state,
        library=prepared.library,
        product_state=next_product,
        model_configuration_sha256=settings.model_configuration_sha256,
        last_frame_png=full_png,
        shot_metadata=metadata,
        receipt=receipt,
    )
    return NativeStoryCommitResult(
        state,
        last_frame,
        archive.load(state, model_configuration_sha256=settings.model_configuration_sha256),
    )
