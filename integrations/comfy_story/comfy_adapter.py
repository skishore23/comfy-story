"""Comfy-specific media, configuration, and transaction adapters for Comfy Story."""

from __future__ import annotations

import hashlib
import io as python_io
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import folder_paths
import numpy as np
import torch
from PIL import Image

from comfy_story.h3_acceleration import CACHE_CONFIGURATION, NVFP4_MODEL
from comfy_story.h3_prompt import H3_PROMPT_FORMAT, H3_PROMPT_PROTOCOL, compile_h3_prompt
from comfy_story.h3_quality import FULL_HD_CONFIGURATION, UPSCALER_MODEL
from comfy_story.h3_sol_attention import SOL_CONFIGURATION
from comfy_story.memory.settings import configured_memory
from comfy_story.samplers import StorySampler
from comfy_story.story_attempts import StoryAttemptIndex
from comfy_story.story_contracts import (
    ComfyStoryStateRef,
    ReferenceRole,
    ShotIntent,
    StoryLibrary,
    StoryReference,
    canonical_story_json,
)
from comfy_story.story_language import AuthoredStoryAudio, prepare_authored_audio
from comfy_story.story_native_archive import (
    NATIVE_REFERENCE_RUNTIME_SHA256,
    NativeArchiveRevision,
    NativeReferenceArchive,
)
from comfy_story.story_native_service import (
    NATIVE_RGB_CAPTURE_POLICY_SHA256,
    NativeStoryCommitResult,
    commit_native_story_generation,
)
from comfy_story.story_product_contracts import StoryMemoryCommand
from comfy_story.story_prompt import PROMPT_FORMATS, structured_reference_prompt
from comfy_story.story_recall import parse_shot_state_evidence
from comfy_story.story_service import (
    PreparedStoryGeneration,
    StoryCommitRequest,
    StoryGenerationRequest,
    prepare_story_generation,
    validate_comfy_images,
)
from comfy_story.story_store import StoryProjectStore
from comfy_story.story_video_canvas import H3_VIDEO_HEIGHT, H3_VIDEO_WIDTH
from comfy_story.tensors import tensor_sha256

_IDENTIFIER = re.compile(r"[^A-Za-z0-9._-]+")
_FRAME_COUNTS = {"5 seconds": 124, "10 seconds": 243, "15 seconds": 362}
_MODEL_CONFIGURATION = {
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "height": H3_VIDEO_HEIGHT,
    "model": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "sampler": "res_multistep",
    "scheduler": "beta",
    "steps": 20,
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "width": H3_VIDEO_WIDTH,
}
TURBO_CONFIGURATION = {
    **_MODEL_CONFIGURATION,
    "sampler": "euler",
    "scheduler": "simple",
    "steps": 4,
    "ref_image_size": "match",
    "shift_video": 12.0,
    "shift_audio": 3.0,
    "lora": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    "lora_strength": 1.0,
}
REFERENCE_TURBO_8_CONFIGURATION = {
    **TURBO_CONFIGURATION,
    "steps": 8,
    "lora": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
}
ANIMATE_CONFIGURATION = {
    **_MODEL_CONFIGURATION,
    "model": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "first_frame_fit": "center_crop",
}
ANIMATE_TURBO_CONFIGURATION = {
    **TURBO_CONFIGURATION,
    "model": ANIMATE_CONFIGURATION["model"],
    "first_frame_fit": "center_crop",
    "shift_video": 6.0,
    "lora": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
}

ANIMATE_TURBO_8_CONFIGURATION = {
    **ANIMATE_CONFIGURATION,
    "scheduler": "simple",
    "steps": 8,
    "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "lora_strength": 1.0,
}


def render_configuration(profile: str, sampler: StorySampler) -> dict[str, Any]:
    if profile not in {"Reference shot", "Animate frame"}:
        raise ValueError("Render profile must be Reference shot or Animate frame")
    nvfp4 = {
        StorySampler.NVFP4_EXACT: NVFP4_CONFIGURATION,
        StorySampler.NVFP4_BALANCED: NVFP4_BALANCED_CONFIGURATION,
        StorySampler.NVFP4_ULTRA_FAST: NVFP4_ULTRA_CONFIGURATION,
        StorySampler.NVFP4_TURBO: NVFP4_TURBO_CONFIGURATION,
    }
    if sampler in nvfp4:
        if profile != "Reference shot":
            raise ValueError("NVFP4 sampling requires Reference shot")
        return nvfp4[sampler]
    if sampler is StorySampler.FULL_HD_2PASS:
        base = (
            ANIMATE_TURBO_8_CONFIGURATION
            if profile == "Animate frame"
            else REFERENCE_TURBO_8_CONFIGURATION
        )
        return {
            **base,
            **FULL_HD_CONFIGURATION,
            "width": 960,
            "height": 544,
            "steps": 6,
            "sampler": "euler",
            "scheduler": "simple",
            "shift_video": 6.0,
            "shift_audio": 3.0,
        }
    if sampler is StorySampler.TURBO_8STEP:
        return (
            ANIMATE_TURBO_8_CONFIGURATION
            if profile == "Animate frame"
            else REFERENCE_TURBO_8_CONFIGURATION
        )
    if profile == "Animate frame":
        if sampler is StorySampler.SPEED_EULER_2STAGE:
            raise ValueError("Animate frame supports Native or Turbo sampling")
        return (
            ANIMATE_TURBO_CONFIGURATION
            if sampler is StorySampler.TURBO_4STEP
            else ANIMATE_CONFIGURATION
        )
    return TURBO_CONFIGURATION if sampler is StorySampler.TURBO_4STEP else _MODEL_CONFIGURATION


NVFP4_CONFIGURATION = {
    **_MODEL_CONFIGURATION,
    "model": NVFP4_MODEL,
    "sampler": "euler",
    "scheduler": "simple",
    "steps": 27,
    "shift_video": 12.0,
    "shift_audio": 3.0,
    "attention": "native-comfy-dense",
}
NVFP4_BALANCED_CONFIGURATION = {**NVFP4_CONFIGURATION, "cache": CACHE_CONFIGURATION}
NVFP4_ULTRA_CONFIGURATION = {**NVFP4_BALANCED_CONFIGURATION, "attention": SOL_CONFIGURATION}
NVFP4_TURBO_CONFIGURATION = {**TURBO_CONFIGURATION, "model": NVFP4_MODEL}
MINIMAX_MODEL_CONFIGURATION_SHA256 = hashlib.sha256(
    canonical_story_json({"format": "comfy-story-h3-v1", "model": _MODEL_CONFIGURATION})
).hexdigest()
_SAMPLERS = {
    "Native res_multistep": StorySampler.NATIVE_RES_MULTISTEP,
    "SPEED Euler 2-stage": StorySampler.SPEED_EULER_2STAGE,
    "Turbo 4-step": StorySampler.TURBO_4STEP,
    "Turbo 8-step": StorySampler.TURBO_8STEP,
    "Full HD 2-pass": StorySampler.FULL_HD_2PASS,
    "NVFP4 Exact": StorySampler.NVFP4_EXACT,
    "NVFP4 Balanced": StorySampler.NVFP4_BALANCED,
    "NVFP4 Ultra Fast": StorySampler.NVFP4_ULTRA_FAST,
    "NVFP4 Turbo 4-step": StorySampler.NVFP4_TURBO,
}
_SPEED_INPUT_TYPES = {
    "Tolerance (Delta)": "FLOAT",
    "guider": "GUIDER",
    "latent_image": "LATENT",
    "noise": "NOISE",
    "noise_amplitude": "FLOAT",
    "noise_decay_exponent": "FLOAT",
    "seed_offset": "INT",
    "sigmas": "SIGMAS",
    "stages": "INT",
}
_MAX_MEMORY_ACTION_BYTES = 256 * 1024
_MAX_MEMORY_ACTIONS = 64


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate Story Library field: {key}")
        result[key] = value
    return result


def parse_memory_actions(encoded: object) -> tuple[StoryMemoryCommand, ...]:
    """Decode the panel's bounded canonical creator-command roster."""
    text = "" if encoded is None else str(encoded)
    if not text:
        return ()
    payload = text.encode("utf-8")
    if len(payload) > _MAX_MEMORY_ACTION_BYTES:
        raise ValueError("Memory actions exceed the 256 KiB limit")
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Memory actions must be valid JSON") from error
    if not isinstance(value, list) or len(value) > _MAX_MEMORY_ACTIONS:
        raise ValueError("Memory actions must be an array of at most 64 commands")
    if canonical_story_json(value) != payload:
        raise ValueError("Memory actions must use canonical JSON")
    return tuple(StoryMemoryCommand.from_mapping(item) for item in value)


def story_inspector_summary(
    loaded: NativeArchiveRevision, *, owner_node_id: str
) -> dict[str, object]:
    """Build a bounded, path-free UI projection from an authenticated v3 revision."""
    if not isinstance(loaded, NativeArchiveRevision):
        raise ValueError("inspector summary requires a loaded Story revision")
    if not isinstance(owner_node_id, str) or not owner_node_id or len(owner_node_id) > 128:
        raise ValueError("owner_node_id must be a bounded Comfy node identifier")
    product = loaded.require_product_state()
    receipt = loaded.generation_receipt
    if receipt is None:
        raise ValueError("Living Canon revision is missing its generation receipt")
    selected = set(receipt.selected_evidence_ids)
    canon_rows = [
        {
            "entity_id": entity.entity_id,
            "presence": entity.presence.value,
            "reference_name": entity.reference_name,
            "state_note": entity.state_note,
            "supporting_evidence_ids": list(entity.supporting_evidence_ids),
        }
        for entity in product.canon.entities
    ]
    pending_rows = [
        {
            "entity_id": entity.entity_id,
            "evidence_id": item.evidence_id,
            "source_revision_sha256": item.source_revision_sha256,
            "state_note": item.state_note,
        }
        for entity in product.canon.entities
        for item in entity.pending
    ]
    forgotten = set(product.policy.tombstoned_evidence_ids)
    pinned = set(product.policy.pinned_evidence_ids)
    moments = [
        {
            "asset_sha256": record.asset_sha256,
            "entity_ids": list(record.entity_ids),
            "evidence_id": record.evidence_id,
            "frame_index": record.frame_index,
            "kind": record.kind.value,
            "remembered": record.evidence_id not in forgotten,
            "retained": record.evidence_id in pinned,
            "salience_q": record.salience_q,
            "selected": record.evidence_id in selected,
            "shot_index": record.source_shot_index,
        }
        for record in product.evidence_records
    ]
    summary: dict[str, object] = {
        "canon": canon_rows,
        "important_moments": moments,
        "owner_node_id": owner_node_id,
        "pending": pending_rows,
        "revision_sha256": loaded.state.revision_sha256,
        "parent_revision_sha256": loaded.state.parent_revision_sha256,
        "shot_count": loaded.state.shot_count,
        "used_for_this_shot": [
            {"asset_sha256": binding.asset_sha256, "role": binding.role}
            for binding in receipt.guide_bindings
        ],
    }
    video_digest = loaded.shot_metadata.get("video_sha256")
    if isinstance(video_digest, str):
        summary["video_sha256"] = video_digest
    if len(canonical_story_json(summary)) > _MAX_MEMORY_ACTION_BYTES:
        raise ValueError("Story inspector summary exceeds the 256 KiB UI boundary")
    return summary


def _read_regular(path: Path, field: str, maximum: int = 4 * 1024**3) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{field} is unavailable or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError(f"{field} must be a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            value = handle.read(maximum + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ) or len(value) != after.st_size:
            raise ValueError(f"{field} changed while it was read")
        return value
    finally:
        os.close(descriptor)


def _configured_path(name: str) -> Path:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} must name an explicit local path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute local path")
    return path.resolve()


def _story_root() -> Path:
    if "COMFY_STORY_ROOT" in os.environ:
        return _configured_path("COMFY_STORY_ROOT")
    return (Path(folder_paths.get_user_directory()) / "comfy_story").resolve()


def _story_sampler(value: object) -> StorySampler:
    try:
        return _SAMPLERS[str(value)]
    except KeyError as error:
        raise ValueError(f"Sampler must be one of: {', '.join(_SAMPLERS)}") from error


def require_speed_sampler(node_classes: Mapping[str, object]) -> None:
    """Reject absent or drifted third-party SPEED nodes before graph expansion."""
    node = node_classes.get("MiniMaxH3SPEEDSampler")
    if node is None:
        raise ValueError("MiniMaxH3SPEEDSampler is not installed")
    input_types = getattr(node, "INPUT_TYPES", None)
    if not callable(input_types):
        raise ValueError("MiniMaxH3SPEEDSampler schema is incompatible")
    try:
        schema = input_types()
        required = schema["required"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("MiniMaxH3SPEEDSampler schema is incompatible") from error
    if not isinstance(required, dict):
        raise ValueError("MiniMaxH3SPEEDSampler schema is incompatible")
    for name, expected_type in _SPEED_INPUT_TYPES.items():
        descriptor = required.get(name)
        if (
            not isinstance(descriptor, (tuple, list))
            or not descriptor
            or descriptor[0] != expected_type
        ):
            raise ValueError("MiniMaxH3SPEEDSampler schema is incompatible")
    noise_policy = required.get("noise_policy")
    if (
        not isinstance(noise_policy, (tuple, list))
        or not noise_policy
        or not isinstance(noise_policy[0], (tuple, list))
        or "direct_coarse" not in noise_policy[0]
    ):
        raise ValueError("MiniMaxH3SPEEDSampler schema is incompatible")


def _portable_project_id(name: str) -> str:
    value = _IDENTIFIER.sub("-", name.strip()).strip("-._").lower()
    if not value:
        raise ValueError("Story Library project name must produce a usable project ID")
    return value[:80]


def _input_path(filename: str) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
        raise ValueError("Story Library file must be a Comfy input filename")
    root = Path(folder_paths.get_input_directory()).resolve()
    candidate = (root / filename).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError("Story Library file must stay inside the Comfy input directory")
    return candidate


def _image_tensor(value: bytes, field: str) -> torch.Tensor:
    try:
        import io

        with Image.open(io.BytesIO(value)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as error:
        raise ValueError(f"{field} is not a decodable RGB image") from error
    return torch.from_numpy(np.array(array, copy=True)).to(torch.float32).div(255).unsqueeze(0)


def _optional_uploaded_image(value: object, field: str) -> torch.Tensor | None:
    if value in (None, "", "None"):
        return None
    if isinstance(value, torch.Tensor):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an uploaded image or IMAGE connection")
    encoded = _read_regular(_input_path(value), field)
    return _image_tensor(encoded, field)


def _library_from_editor(
    encoded: str, store: StoryProjectStore
) -> tuple[StoryLibrary, dict[str, torch.Tensor]]:
    try:
        raw = json.loads(encoded, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Story Library must be valid JSON") from error
    if not isinstance(raw, dict) or set(raw) != {"project_name", "references"}:
        raise ValueError("Story Library must contain project_name and references")
    project_name = raw["project_name"]
    rows = raw["references"]
    if not isinstance(project_name, str) or not isinstance(rows, list) or not rows:
        raise ValueError("Story Library needs a project name and at least one reference")
    references: list[StoryReference] = []
    images: dict[str, torch.Tensor] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"file", "name", "note", "role"}:
            raise ValueError(f"Story Library reference {index + 1} is malformed")
        filename, name, note, role = (row[key] for key in ("file", "name", "note", "role"))
        if not all(isinstance(value, str) for value in (filename, name, note, role)):
            raise ValueError(f"Story Library reference {index + 1} fields must be text")
        source = _read_regular(_input_path(cast(str, filename)), f"@{name} source")
        digest = store.put_asset(source)
        image = _image_tensor(source, f"@{name} source")
        preprocessing = hashlib.sha256(
            canonical_story_json({"color": "RGB", "decoder": "Pillow", "source_sha256": digest})
        ).hexdigest()
        reference = StoryReference(
            cast(str, name),
            ReferenceRole(cast(str, role)),
            cast(str, note),
            digest,
            f"comfy-story://assets/sha256/{digest}",
            (digest,),
            preprocessing,
        ).validate()
        references.append(reference)
        images[reference.name] = image
    return StoryLibrary(project_name, tuple(references)).validate(), images


def _reference_images_from_store(
    library: StoryLibrary, store: StoryProjectStore
) -> dict[str, torch.Tensor]:
    images: dict[str, torch.Tensor] = {}
    for reference in library.references:
        path = store.root / "assets" / "sha256" / reference.keyframe_sha256s[0]
        value = _read_regular(path, f"@{reference.name} asset")
        if hashlib.sha256(value).hexdigest() != reference.keyframe_sha256s[0]:
            raise ValueError(f"@{reference.name} asset SHA-256 changed")
        images[reference.name] = _image_tensor(value, f"@{reference.name} asset")
    return images


def _state_from_inputs(inputs: dict[str, Any]) -> ComfyStoryStateRef | None:
    connected = inputs.get("Previous Story")
    if connected is not None and not isinstance(connected, ComfyStoryStateRef):
        raise ValueError("Previous Story must come from a Comfy Story node")
    recovery_text = str(inputs.get("Story revision", "")).strip()
    recovery = None if not recovery_text else ComfyStoryStateRef.from_json(recovery_text.encode())
    if connected is not None and recovery is not None and connected != recovery:
        raise ValueError("Connected Previous Story does not match the recovery revision")
    return connected if connected is not None else recovery


@dataclass(frozen=True, slots=True)
class PreparedNodeGeneration:
    prepared: PreparedStoryGeneration
    visual_guides: tuple[torch.Tensor, ...]
    resolved_prompt: str
    frame_count: int
    variation: int
    sampler: StorySampler
    motion_reference: torch.Tensor | None = None
    authored_audio: AuthoredStoryAudio | None = None
    output_duration_ms: int = 0
    ending_frame: torch.Tensor | None = None
    render_profile: str = "Reference shot"


@lru_cache(maxsize=32)
def _model_file_digest(path: str, signature: tuple[int, int, int, int]) -> str:
    del signature
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _generation_asset_digest(folder: str, name: str) -> str:
    location = folder_paths.get_full_path(folder, name)
    if location is None:
        raise ValueError(f"Comfy Story requires the installed MiniMax H3 model: {name}")
    path = Path(location).resolve(strict=True)
    before = path.stat()
    signature = (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    digest = _model_file_digest(str(path), signature)
    after = path.stat()
    if (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != signature:
        raise ValueError(f"MiniMax H3 model changed during preflight: {name}")
    return digest


def _generation_assets(configuration: dict[str, Any] | None = None) -> dict[str, str]:
    configuration = _MODEL_CONFIGURATION if configuration is None else configuration
    result: dict[str, str] = {}
    for field, folder in (
        ("model", "diffusion_models"),
        ("clip", "text_encoders"),
        ("video_vae", "vae"),
        ("audio_vae", "vae"),
    ):
        name = str(configuration[field])
        result[field] = _generation_asset_digest(folder, name)
    if "upscaler" in configuration:
        result["upscaler"] = _generation_asset_digest("latent_upscale_models", UPSCALER_MODEL)
    return result


def _bind_execution(
    prepared: PreparedStoryGeneration,
    motion: torch.Tensor | None,
    authored_audio: AuthoredStoryAudio | None = None,
    output_duration_ms: int = 0,
    ending_frame: torch.Tensor | None = None,
    prompt_protocol: str | None = None,
) -> PreparedStoryGeneration:
    request = prepared.request
    configuration = render_configuration(request.render_profile, request.sampler)
    assets = _generation_assets(configuration)
    if "lora" in configuration:
        assets["lora"] = _generation_asset_digest("loras", str(configuration["lora"]))
    memory = request.associative_memory
    if memory is not None and (
        assets["model"] != memory.foundation_sha256 or assets["video_vae"] != memory.vae_sha256
    ):
        raise ValueError(
            "associative checkpoint foundation/VAE does not match the generation models"
        )
    binding = {
        "format": "comfy-story-execution-v1",
        "generation_configuration": configuration,
        "model_files": assets,
        "project_id": request.project_id,
        "branch_id": request.branch_id,
        "parent": request.previous_story,
        "library": prepared.library,
        "intent": request.intent.value,
        "prompt": prepared.resolved_prompt,
        "frames": _FRAME_COUNTS[f"{request.shot_length_seconds} seconds"],
        "variation": request.variation,
        "sampler": request.sampler.value,
        "memory_configuration": request.model_configuration_sha256,
        "memory_actions": request.memory_commands,
        "reference_policy": request.reference_policy,
        "guides": [
            (role, tensor_sha256(guide))
            for role, guide in zip(prepared.visual_roles, prepared.visual_guides, strict=True)
        ],
        "motion": None if motion is None else tensor_sha256(motion),
        "memory_runtime": NATIVE_REFERENCE_RUNTIME_SHA256,
        "native_capture_policy": NATIVE_RGB_CAPTURE_POLICY_SHA256,
    }
    if memory is not None:
        binding["associative_memory"] = memory.binding()
    if prompt_protocol is not None:
        binding["prompt_compiler"] = prompt_protocol
    if request.shot_state_evidence:
        binding["shot_state_evidence"] = request.shot_state_evidence
    if request.composition != "Continue frame":
        binding["composition"] = request.composition
    if request.render_profile != "Reference shot":
        binding["render_profile"] = request.render_profile
    if output_duration_ms:
        binding["output_duration_ms"] = output_duration_ms
    if ending_frame is not None:
        binding["ending_frame"] = tensor_sha256(ending_frame)
    if request.scene_entity_names is not None:
        binding["declared_scene_entities"] = [
            name.casefold() for name in request.scene_entity_names
        ]
    if authored_audio is not None:
        binding["authored_audio"] = authored_audio.binding
    digest = StoryProjectStore(_story_root()).put_asset(canonical_story_json(binding))
    return replace(prepared, request=replace(request, execution_sha256=digest))


def recover_node_generation(
    prepared: PreparedNodeGeneration,
) -> tuple[NativeArchiveRevision, Path, torch.Tensor] | None:
    """Return authenticated completed outputs before building any GPU graph."""
    if not isinstance(prepared, PreparedNodeGeneration):
        return None
    request = prepared.prepared.request
    if request.execution_sha256 is None:
        return None
    store = StoryProjectStore(_story_root())
    state = StoryAttemptIndex(store.root).load(request.execution_sha256)
    if state is None:
        return None
    loaded = NativeReferenceArchive(store).load(
        state,
        model_configuration_sha256=request.model_configuration_sha256,
    )
    from comfy_story.memory.service import verify_memory_revision

    verify_memory_revision(loaded, request.associative_memory, store)
    if loaded.shot_metadata.get("execution_sha256") != request.execution_sha256:
        raise ValueError("Completed shot does not match this generation request")
    video_digest = str(loaded.shot_metadata["video_sha256"])
    store.load_asset(video_digest)
    frame_digest = str(loaded.shot_metadata["last_frame_tensor_sha256"])
    frame = torch.from_numpy(
        np.load(python_io.BytesIO(store.load_asset(frame_digest)), allow_pickle=False)
    )
    return (
        loaded,
        store.root / "assets" / "sha256" / video_digest,
        validate_comfy_images(frame, field="completed last frame"),
    )


def _scene_entities(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, str) or len(value.encode()) > 64 * 1024:
        raise ValueError("Scene entities must be a bounded JSON list of reference names")
    if not value.strip():
        return None
    try:
        names = json.loads(value)
    except ValueError as error:
        raise ValueError("Scene entities must be a JSON list of reference names") from error
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Scene entities must be a JSON list of reference names")
    return tuple(names)


def prepare_node_generation(inputs: dict[str, Any]) -> PreparedNodeGeneration:
    """Convert customer widgets to a validated, model-free generation request."""
    store = StoryProjectStore(_story_root())
    state = _state_from_inputs(inputs)
    sampler = _story_sampler(inputs.get("Sampler", "Native res_multistep"))
    render_profile = str(inputs.get("Render profile", "Reference shot"))
    render_configuration(render_profile, sampler)
    motion_value = inputs.get("Motion reference")
    if render_profile == "Animate frame" and (motion_value is not None):
        raise ValueError("Animate frame requires Native context and no motion reference")
    ending_value = inputs.get("Ending frame")
    ending_frame = (
        None if ending_value is None else validate_comfy_images(ending_value, field="Ending frame")
    )
    if ending_frame is not None and ending_frame.shape[0] != 1:
        raise ValueError("Ending frame must contain exactly one image")
    motion_reference = (
        None
        if motion_value is None
        else validate_comfy_images(motion_value, field="Motion reference")
    )
    starting_value = inputs.get("Starting image")
    starting_image = (
        None
        if starting_value is None
        else validate_comfy_images(starting_value, field="Starting image")
    )
    if starting_image is not None and starting_image.shape[0] != 1:
        raise ValueError("Starting image must contain exactly one image")
    model_configuration_sha256 = MINIMAX_MODEL_CONFIGURATION_SHA256
    intent = ShotIntent(str(inputs["Create"]))
    duration = str(inputs["Shot length"])
    if duration not in _FRAME_COUNTS:
        raise ValueError("Shot length must be 5 seconds, 10 seconds, or 15 seconds")
    output_duration_ms = inputs.get("Output duration (ms)", 0)
    if (
        type(output_duration_ms) is not int
        or not 0 <= output_duration_ms <= int(duration.split()[0]) * 1000
        or output_duration_ms * 24 % 1000
    ):
        raise ValueError("Output duration must be exact 24 fps milliseconds within Shot length")
    authored_audio = prepare_authored_audio(
        inputs.get("Authored audio"), frame_count=_FRAME_COUNTS[duration]
    )
    if state is None:
        library, reference_images = _library_from_editor(str(inputs["Story Library"]), store)
        project_id = _portable_project_id(library.project_name)
        branch_id = "main"
    else:
        parent = NativeReferenceArchive(store).load(
            state,
            model_configuration_sha256=model_configuration_sha256,
        )
        library = parent.library
        reference_images = _reference_images_from_store(library, store)
        project_id = state.project_id
        branch_id = state.branch_id
    request = StoryGenerationRequest(
        associative_memory=configured_memory(),
        intent=intent,
        project_id=project_id,
        branch_id=branch_id,
        previous_story=state,
        previous_frame=inputs.get("Previous Frame"),
        world_frame=starting_image
        if starting_image is not None
        else _optional_uploaded_image(
            inputs.get("World / starting frame"), "World / starting frame"
        ),
        library=library,
        reference_images=reference_images,
        prompt=str(inputs["What happens next?"]),
        shot_length_seconds=int(duration.split()[0]),
        variation=int(inputs["Variation"]),
        model_configuration_sha256=model_configuration_sha256,
        reference_policy=str(inputs["Reference policy"]),
        sampler=sampler,
        shot_state_evidence=parse_shot_state_evidence(inputs.get("Shot state evidence", "{}")),
        memory_commands=parse_memory_actions(inputs.get("Memory actions", "")),
        composition=str(inputs.get("Composition", "Continue frame")),
        scene_entity_names=_scene_entities(inputs.get("Scene entities", "")),
        render_profile=render_profile,
    )
    prepared = prepare_story_generation(request, store=store)
    prompt_format = inputs.get("Prompt format", "Current")
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError("Prompt format is unsupported")
    if sampler is StorySampler.FULL_HD_2PASS and prompt_format != H3_PROMPT_FORMAT:
        raise ValueError("Full HD 2-pass requires H3 automatic v1 prompting")
    if prompt_format == H3_PROMPT_FORMAT:
        if (
            len(prepared.visual_guides)
            + int(ending_frame is not None)
            + int(motion_reference is not None)
            > 9
        ):
            raise ValueError(
                "H3 automatic prompting supports at most nine sources including the ending image"
            )
        prepared = replace(
            prepared,
            resolved_prompt=compile_h3_prompt(
                prepared,
                duration_ms=output_duration_ms or _FRAME_COUNTS[duration] * 1000 / 24,
                ending_frame=ending_frame is not None,
                motion_reference=motion_reference is not None,
                authored_audio=authored_audio is not None,
            ),
        )
    elif prompt_format != "Current":
        prepared = replace(prepared, resolved_prompt=structured_reference_prompt(prepared))
    if len(prepared.visual_guides) + int(motion_reference is not None) > 9:
        raise ValueError("MiniMax H3 Story supports at most nine selected visual sources")
    prepared = _bind_execution(
        prepared,
        motion_reference,
        authored_audio,
        output_duration_ms,
        ending_frame,
        prompt_protocol=H3_PROMPT_PROTOCOL if prompt_format == H3_PROMPT_FORMAT else None,
    )
    return PreparedNodeGeneration(
        prepared=prepared,
        visual_guides=prepared.visual_guides,
        resolved_prompt=prepared.resolved_prompt,
        frame_count=_FRAME_COUNTS[duration],
        variation=request.variation,
        sampler=request.sampler,
        motion_reference=motion_reference,
        authored_audio=authored_audio,
        output_duration_ms=output_duration_ms,
        ending_frame=ending_frame,
        render_profile=render_profile,
    )


def _saved_video_path(saved_video: object, filename_prefix: str | None = None) -> Path:
    if isinstance(saved_video, (str, Path)):
        candidate = Path(saved_video)
    elif isinstance(saved_video, dict):
        filename = saved_video.get("filename")
        subfolder = saved_video.get("subfolder", "")
        if not isinstance(filename, str) or not isinstance(subfolder, str):
            raise ValueError("saved video result does not expose a filename")
        output_root = Path(folder_paths.get_output_directory()).resolve()
        candidate = output_root / subfolder / filename
    elif filename_prefix is not None and callable(
        dimensions := getattr(saved_video, "get_dimensions", None)
    ):
        width, height = dimensions()
        output_root = Path(folder_paths.get_output_directory()).resolve()
        folder, filename, next_counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, str(output_root), width, height
        )
        if next_counter <= 1:
            raise ValueError("saved video is unavailable after SaveVideo")
        candidate = Path(folder) / f"{filename}_{next_counter - 1:05}_.mp4"
    else:
        candidate = Path(str(getattr(saved_video, "path", "")))
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError("saved video is unavailable after SaveVideo")
    return resolved


def commit_node_generation(
    *,
    prepared: PreparedStoryGeneration,
    decoded_images: torch.Tensor,
    saved_video: object,
    filename_prefix: str = "comfy_story/shot",
    memory_vae: object | None = None,
) -> NativeStoryCommitResult:
    """Publish shot evidence after MiniMax decoding and saving complete."""
    video = _read_regular(
        _saved_video_path(saved_video, filename_prefix),
        "saved video",
    )
    digest = hashlib.sha256(video).hexdigest()
    store = StoryProjectStore(_story_root())
    store.put_asset(video)
    memory = prepared.request.associative_memory
    runtime = None
    try:
        if memory is not None:
            from .memory_adapter import ComfyMiniMaxH3Codec

            # Recheck the pinned checkpoint at commit; a mid-render file change
            # cannot publish a revision under the earlier execution identity.
            runtime = memory.load(ComfyMiniMaxH3Codec(memory_vae))
        native_result = commit_native_story_generation(
            StoryCommitRequest(
                prepared, decoded_images, digest, f"comfy-evidence://story/sha256/{digest}"
            ),
            store=store,
            memory_runtime=runtime,
        )
    finally:
        if runtime is not None:
            runtime.close()
    if prepared.request.execution_sha256 is not None:
        StoryAttemptIndex(store.root).publish(
            prepared.request.execution_sha256, native_result.state
        )
    return native_result


__all__ = (
    "MINIMAX_MODEL_CONFIGURATION_SHA256",
    "PreparedNodeGeneration",
    "commit_node_generation",
    "parse_memory_actions",
    "prepare_node_generation",
    "require_speed_sampler",
    "story_inspector_summary",
)
