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

from duet.duetx.contracts import DuetXContract, tensor_sha256
from duet.duetx.h3_acceleration import CACHE_CONFIGURATION, NVFP4_MODEL
from duet.duetx.h3_prompt import H3_PROMPT_FORMAT, H3_PROMPT_PROTOCOL, compile_h3_prompt
from duet.duetx.h3_quality import FULL_HD_CONFIGURATION, UPSCALER_MODEL
from duet.duetx.h3_reference_cache import CompiledReferenceCache
from duet.duetx.h3_reference_checkpoint import load_h3_reference_compiler_checkpoint
from duet.duetx.h3_reference_compressors import H3ReferenceCompiler
from duet.duetx.h3_reference_contracts import (
    H3ReferenceBudget,
    H3ReferenceKind,
    H3ReferenceMethod,
    H3SelectedSource,
)
from duet.duetx.h3_sol_attention import SOL_CONFIGURATION
from duet.duetx.ltx_quality_protocol import FROZEN_TRAINABLE_CHECKPOINT_SHA256
from duet.duetx.minimax_h3_memory import (
    MiniMaxH3StoryRuntime,
    inspect_minimax_h3_checkpoint,
)
from duet.duetx.minimax_h3_training import MiniMaxH3TrainingProtocol
from duet.duetx.story_attempts import StoryAttemptIndex
from duet.duetx.story_contracts import (
    DuetStoryStateRef,
    ReferenceRole,
    ShotIntent,
    StoryLibrary,
    StoryReference,
    canonical_story_json,
)
from duet.duetx.story_language import AuthoredStoryAudio, prepare_authored_audio
from duet.duetx.story_memory_backend import StoryMemoryBackend, StorySampler
from duet.duetx.story_native_archive import (
    NATIVE_REFERENCE_RUNTIME_SHA256,
    NativeArchiveRevision,
    NativeReferenceArchive,
)
from duet.duetx.story_native_service import (
    NATIVE_RGB_CAPTURE_POLICY_SHA256,
    NativeStoryCommitResult,
    commit_native_story_generation,
)
from duet.duetx.story_product_contracts import StoryMemoryCommand
from duet.duetx.story_prompt import PROMPT_FORMATS, structured_reference_prompt
from duet.duetx.story_recall import parse_shot_state_evidence
from duet.duetx.story_service import (
    PreparedStoryGeneration,
    StoryCommitRequest,
    StoryCommitResult,
    StoryGenerationRequest,
    StoryLTXRuntime,
    commit_story_generation,
    prepare_story_generation,
    validate_comfy_images,
)
from duet.duetx.story_store import LoadedStoryRevision, StoryProjectStore
from duet.duetx.story_video_canvas import H3_VIDEO_HEIGHT, H3_VIDEO_WIDTH

from .h3_reference_node import PreparedReferenceCompile

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
MODEL_CONFIGURATION_SHA256 = hashlib.sha256(canonical_story_json(_MODEL_CONFIGURATION)).hexdigest()
_MINIMAX_MEMORY_CONFIGURATION = {
    "backend": "minimax-h3",
    "format": "duet-story-minimax-h3-memory-v1",
    "history_items": 8,
    "latent_channels": 24,
    "protected_exceptions": 2,
    "video_vae": _MODEL_CONFIGURATION["video_vae"],
}
MINIMAX_MODEL_CONFIGURATION_SHA256 = hashlib.sha256(
    canonical_story_json(_MINIMAX_MEMORY_CONFIGURATION)
).hexdigest()
_BACKENDS = {
    "LTX": StoryMemoryBackend.LTX,
    "MiniMax H3": StoryMemoryBackend.MINIMAX_H3,
}
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
_REFERENCE_CONTEXTS = {"Native", "Compiled preview"}
_INTERNAL_COMPILER_METHODS = {method.value: method for method in H3ReferenceMethod}
_STORY_CONTEXT_EXPERIMENTS = {"core_plus_recall", "retrieval_only"}
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
    loaded: LoadedStoryRevision | NativeArchiveRevision, *, owner_node_id: str
) -> dict[str, object]:
    """Build a bounded, path-free UI projection from an authenticated v3 revision."""
    if not isinstance(loaded, (LoadedStoryRevision, NativeArchiveRevision)):
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
    if "DUET_STORY_ROOT" in os.environ:
        return _configured_path("DUET_STORY_ROOT")
    return (Path(folder_paths.get_user_directory()) / "duet_story").resolve()


def _checkpoint() -> Path:
    path = _configured_path("DUET_STORY_CHECKPOINT")
    if not path.is_file():
        raise ValueError("DUET_STORY_CHECKPOINT must name an existing local checkpoint")
    return path


def _ltx_source_identity() -> tuple[str, str]:
    commit = os.environ.get("DUET_STORY_SOURCE_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or commit == "0" * 40:
        raise ValueError("DUET_STORY_SOURCE_COMMIT must pin the actual deployed source revision")
    archive = _required_digest_environment("DUET_STORY_SOURCE_ARCHIVE_SHA256")
    if archive == "0" * 64:
        raise ValueError("DUET_STORY_SOURCE_ARCHIVE_SHA256 must pin the actual source archive")
    return commit, archive


def _ltx_contract() -> DuetXContract:
    return DuetXContract.default(decision1_fingerprint="0" * 64).validate()


def _required_digest_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise ValueError(f"{name} must pin the local memory runtime")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class _MiniMaxRuntimeSettings:
    checkpoint: Path
    checkpoint_sha256: str
    foundation_sha256: str
    vae_sha256: str
    protocol_sha256: str


def _minimax_runtime_settings() -> _MiniMaxRuntimeSettings:
    checkpoint_value = os.environ.get("DUET_STORY_MINIMAX_CHECKPOINT")
    if checkpoint_value is None:
        raise ValueError(
            "DUET_STORY_MINIMAX_CHECKPOINT must name the trained local memory checkpoint"
        )
    checkpoint = Path(checkpoint_value).resolve()
    return _MiniMaxRuntimeSettings(
        checkpoint,
        _required_digest_environment("DUET_STORY_MINIMAX_CHECKPOINT_SHA256"),
        _required_digest_environment("DUET_STORY_MINIMAX_FOUNDATION_SHA256"),
        _required_digest_environment("DUET_STORY_MINIMAX_VAE_SHA256"),
        MiniMaxH3TrainingProtocol.default().fingerprint(),
    )


def _memory_backend(value: object) -> StoryMemoryBackend:
    try:
        return _BACKENDS[str(value)]
    except KeyError as error:
        raise ValueError("Memory backend must be MiniMax H3 or LTX") from error


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


class ComfyMiniMaxH3Codec:
    """Narrow adapter around the MiniMax video VAE already owned by Comfy."""

    def __init__(self, vae: object) -> None:
        if not callable(getattr(vae, "encode", None)) or not callable(getattr(vae, "decode", None)):
            raise ValueError("memory_vae must be a loaded Comfy VAE")
        self._vae = vae

    def encode_frame(self, frame: np.ndarray[Any, Any]) -> torch.Tensor:
        if frame.dtype != np.uint8 or frame.shape != (384, 384, 3):
            raise ValueError("MiniMax H3 codec requires RGB uint8 384x384")
        images = (
            torch.from_numpy(np.array(frame, copy=True)).to(torch.float32).div(255).unsqueeze(0)
        )
        latent = self._vae.encode(images)  # type: ignore[attr-defined]
        if (
            not isinstance(latent, torch.Tensor)
            or latent.ndim != 5
            or latent.shape[0] != 1
            or latent.shape[1] != 24
            or any(dimension <= 0 for dimension in latent.shape[2:])
            or not torch.is_floating_point(latent)
            or not bool(torch.isfinite(latent).all().item())
        ):
            raise ValueError("Comfy MiniMax H3 VAE returned an invalid 24-channel latent")
        return latent.detach()

    def decode_frame(self, latent: torch.Tensor) -> np.ndarray[Any, Any]:
        decoded = self._vae.decode(latent)  # type: ignore[attr-defined]
        if not isinstance(decoded, torch.Tensor) or not torch.is_floating_point(decoded):
            raise ValueError("Comfy MiniMax H3 VAE returned an invalid image tensor")
        if decoded.ndim == 5 and decoded.shape[0] == 1:
            image = decoded[0, -1]
        elif decoded.ndim == 4:
            image = decoded[-1]
        else:
            raise ValueError("Comfy MiniMax H3 VAE returned an unexpected image boundary")
        if image.shape != (384, 384, 3) or not bool(torch.isfinite(image).all().item()):
            raise ValueError("Comfy MiniMax H3 VAE returned an unexpected RGB frame")
        value = image.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1)
        return np.ascontiguousarray(value.mul(255).round().to(torch.uint8).numpy())

    def close(self) -> None:
        """The VAE lifecycle remains owned by Comfy's graph executor."""


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
            f"duet-story://assets/sha256/{digest}",
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


def _state_from_inputs(inputs: dict[str, Any]) -> DuetStoryStateRef | None:
    connected = inputs.get("Previous Story")
    if connected is not None and not isinstance(connected, DuetStoryStateRef):
        raise ValueError("Previous Story must come from a Comfy Story node")
    recovery_text = str(inputs.get("Story revision", "")).strip()
    recovery = None if not recovery_text else DuetStoryStateRef.from_json(recovery_text.encode())
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
    reference_compile: PreparedReferenceCompile | None = None
    telemetry_enabled: bool = False
    benchmark_phase: str = ""
    benchmark_workload: str = ""
    benchmark_method: str = ""
    benchmark_cell: str = ""
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
    compiled: PreparedReferenceCompile | None,
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
    binding = {
        "format": "duet-story-execution-v1",
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
        "memory_checkpoint": request.checkpoint_sha256,
        "memory_configuration": request.model_configuration_sha256,
        "memory_actions": request.memory_commands,
        "reference_policy": request.reference_policy,
        "guides": [
            (role, tensor_sha256(guide))
            for role, guide in zip(prepared.visual_roles, prepared.visual_guides, strict=True)
        ],
        "motion": None if motion is None else tensor_sha256(motion),
        "compiler": None
        if compiled is None
        else {
            "method": compiled.compiler.method.value,
            "checkpoint": compiled.compiler.checkpoint_sha256,
            "budget": compiled.budget,
            "sources": compiled.sources,
        },
    }
    if prompt_protocol is not None:
        binding["prompt_compiler"] = prompt_protocol
    if request.shot_state_evidence:
        binding["shot_state_evidence"] = request.shot_state_evidence
    if request.native_reference_archive:
        del binding["memory_checkpoint"]
        binding["memory_runtime"] = NATIVE_REFERENCE_RUNTIME_SHA256
        binding["native_capture_policy"] = NATIVE_RGB_CAPTURE_POLICY_SHA256
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
) -> tuple[LoadedStoryRevision | NativeArchiveRevision, Path, torch.Tensor] | None:
    """Return authenticated completed outputs before building any GPU graph."""
    if not isinstance(prepared, PreparedNodeGeneration) or prepared.telemetry_enabled:
        return None
    request = prepared.prepared.request
    if request.execution_sha256 is None:
        return None
    store = StoryProjectStore(_story_root())
    state = StoryAttemptIndex(store.root).load(request.execution_sha256)
    if state is None:
        return None
    loaded: LoadedStoryRevision | NativeArchiveRevision
    if request.native_reference_archive:
        loaded = NativeReferenceArchive(store).load(
            state, model_configuration_sha256=request.model_configuration_sha256
        )
    else:
        contract = DuetXContract.minimax_h3(adapter_fingerprint=_minimax_identity().adapter_sha256)
        loaded = store.load(state, contract, request.checkpoint_sha256)
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


def _native_reference_runtime() -> bool:
    """Keep configured trained installations explicit; clean installs use RGB archives."""
    mode = os.environ.get("DUET_STORY_MEMORY_RUNTIME")
    if mode is None:
        return "DUET_STORY_MINIMAX_CHECKPOINT" not in os.environ
    if mode not in ("native-reference", "trained"):
        raise ValueError("DUET_STORY_MEMORY_RUNTIME must be native-reference or trained")
    return mode == "native-reference"


def _minimax_identity() -> Any:
    settings = _minimax_runtime_settings()
    return inspect_minimax_h3_checkpoint(
        settings.checkpoint,
        expected_checkpoint_sha256=settings.checkpoint_sha256,
        expected_foundation_sha256=settings.foundation_sha256,
        expected_protocol_sha256=settings.protocol_sha256,
        model_configuration_sha256=MINIMAX_MODEL_CONFIGURATION_SHA256,
    )


def _protected_reference_names(value: object) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    names: list[str] = []
    folded: set[str] = set()
    for item in str(value).split(","):
        name = item.strip().removeprefix("@").strip()
        if not name:
            continue
        key = name.casefold()
        if key not in folded:
            names.append(name)
            folded.add(key)
    if len(names) > 2:
        raise ValueError("Keep this detail supports at most two selected @references")
    return tuple(names)


def _reference_context(value: object) -> str:
    resolved = str(value or "Native")
    if resolved not in _REFERENCE_CONTEXTS:
        raise ValueError("Reference context must be Native or Compiled preview")
    return resolved


def _compiler_method(inputs: dict[str, Any], *, internal_test: bool) -> H3ReferenceMethod:
    if not internal_test:
        return H3ReferenceMethod.DUET_X
    value = str(
        inputs.get(
            "Compiler experiment",
            os.environ.get("DUET_H3_COMPILER_METHOD", H3ReferenceMethod.DUET_X.value),
        )
    )
    try:
        return _INTERNAL_COMPILER_METHODS[value]
    except KeyError as error:
        raise ValueError("DUET_H3_COMPILER_METHOD is unsupported") from error


def _story_context_experiment(inputs: dict[str, Any], *, internal_test: bool) -> str:
    if not internal_test:
        return "core_plus_recall"
    value = str(inputs.get("Story context experiment", "core_plus_recall"))
    if value not in _STORY_CONTEXT_EXPERIMENTS:
        raise ValueError("Story context experiment is unsupported")
    return value


def _prepare_reference_compile(
    prepared: PreparedStoryGeneration,
    motion_reference: torch.Tensor | None,
    *,
    method: H3ReferenceMethod,
) -> PreparedReferenceCompile:
    sources = prepared.compiler_sources
    if motion_reference is not None:
        sources = (
            *sources,
            H3SelectedSource(
                "input:motion-reference",
                H3ReferenceKind.VIDEO,
                len(sources),
                False,
            ).validate(),
        )
    if len(sources) < 2:
        raise ValueError("Compiled preview requires at least two selected visual sources")
    row_budget_text = os.environ.get("DUET_H3_COMPILER_MAX_VISUAL_ROWS", "16384")
    try:
        row_budget = int(row_budget_text)
    except ValueError as error:
        raise ValueError("DUET_H3_COMPILER_MAX_VISUAL_ROWS must be an integer") from error
    budget = H3ReferenceBudget(row_budget, 2).validate()
    learned = method in {
        H3ReferenceMethod.GATED,
        H3ReferenceMethod.RESAMPLER,
        H3ReferenceMethod.DUET,
        H3ReferenceMethod.DUET_X,
    }
    if learned:
        checkpoint_value = os.environ.get("DUET_H3_COMPILER_CHECKPOINT")
        if checkpoint_value is None:
            raise ValueError("DUET_H3_COMPILER_CHECKPOINT must name the local compiler checkpoint")
        checkpoint = Path(checkpoint_value).resolve()
        checkpoint_sha256 = _required_digest_environment("DUET_H3_COMPILER_CHECKPOINT_SHA256")
        loaded = load_h3_reference_compiler_checkpoint(
            checkpoint,
            expected_sha256=checkpoint_sha256,
            expected_model_configuration_sha256=MINIMAX_MODEL_CONFIGURATION_SHA256,
            device=torch.device("cpu"),
        )
        compiler = H3ReferenceCompiler(
            method,
            budget,
            checkpoint_sha256=loaded.identity.checkpoint_sha256,
            compressor=loaded.compressor_for(method),
        )
    else:
        compiler = H3ReferenceCompiler(method, budget)
    cache_root = Path(
        os.environ.get("DUET_H3_COMPILER_CACHE", str(_story_root() / "compiled-reference-cache"))
    ).resolve()
    return PreparedReferenceCompile(
        sources,
        budget,
        compiler,
        CompiledReferenceCache(cache_root),
    ).validate()


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


def prepare_node_generation(
    inputs: dict[str, Any], *, living_canon: bool = False
) -> PreparedNodeGeneration:
    """Convert customer widgets to a validated, model-free generation request."""
    store = StoryProjectStore(_story_root())
    state = _state_from_inputs(inputs)
    internal_test = os.environ.get("DUET_H3_COMPILER_INTERNAL_TEST") == "1"
    story_context_experiment = _story_context_experiment(inputs, internal_test=internal_test)
    memory_backend = (
        StoryMemoryBackend.MINIMAX_H3
        if living_canon
        else _memory_backend(inputs.get("Memory backend", "MiniMax H3"))
    )
    sampler = _story_sampler(inputs.get("Sampler", "Native res_multistep"))
    reference_context = _reference_context(inputs.get("Reference context", "Native"))
    render_profile = str(inputs.get("Render profile", "Reference shot"))
    render_configuration(render_profile, sampler)
    motion_value = inputs.get("Motion reference")
    if render_profile == "Animate frame" and (
        reference_context != "Native" or motion_value is not None
    ):
        raise ValueError("Animate frame requires Native context and no motion reference")
    ending_value = inputs.get("Ending frame")
    ending_frame = (
        None if ending_value is None else validate_comfy_images(ending_value, field="Ending frame")
    )
    if ending_frame is not None:
        if not living_canon:
            raise ValueError("Ending frames require the public Comfy Story generation path")
        if ending_frame.shape[0] != 1:
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
    protected_names = _protected_reference_names(inputs.get("Keep this detail", ""))
    native_archive = (
        living_canon
        and memory_backend is StoryMemoryBackend.MINIMAX_H3
        and _native_reference_runtime()
    )
    contract: DuetXContract | None
    checkpoint_sha256: str | None
    if native_archive:
        if reference_context != "Native" or internal_test:
            raise ValueError(
                "Native reference archives require Native context without compiler experiments"
            )
        contract = None
        checkpoint_sha256 = None
        model_configuration_sha256 = MINIMAX_MODEL_CONFIGURATION_SHA256
    elif memory_backend is StoryMemoryBackend.MINIMAX_H3:
        settings = _minimax_runtime_settings()
        identity = inspect_minimax_h3_checkpoint(
            settings.checkpoint,
            expected_checkpoint_sha256=settings.checkpoint_sha256,
            expected_foundation_sha256=settings.foundation_sha256,
            expected_protocol_sha256=settings.protocol_sha256,
            model_configuration_sha256=MINIMAX_MODEL_CONFIGURATION_SHA256,
        )
        contract = DuetXContract.minimax_h3(adapter_fingerprint=identity.adapter_sha256).validate()
        checkpoint_sha256 = identity.checkpoint_sha256
        model_configuration_sha256 = identity.model_configuration_sha256
    else:
        _checkpoint()
        _ltx_source_identity()
        contract = _ltx_contract()
        checkpoint_sha256 = FROZEN_TRAINABLE_CHECKPOINT_SHA256
        model_configuration_sha256 = MODEL_CONFIGURATION_SHA256
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
        parent: LoadedStoryRevision | NativeArchiveRevision
        if native_archive:
            parent = NativeReferenceArchive(store).load(
                state, model_configuration_sha256=model_configuration_sha256
            )
        else:
            if contract is None:
                raise ValueError("Trained memory requires a checkpoint contract")
            parent = store.load(state, contract, checkpoint_sha256)
        library = parent.library
        reference_images = _reference_images_from_store(library, store)
        project_id = state.project_id
        branch_id = state.branch_id
    request = StoryGenerationRequest(
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
        checkpoint_sha256=checkpoint_sha256,
        model_configuration_sha256=model_configuration_sha256,
        reference_policy=str(inputs["Reference policy"]),
        memory_backend=memory_backend,
        sampler=sampler,
        shot_state_evidence=parse_shot_state_evidence(inputs.get("Shot state evidence", "{}")),
        memory_commands=(
            parse_memory_actions(inputs.get("Memory actions", "")) if living_canon else ()
        ),
        living_canon=living_canon,
        native_reference_archive=native_archive,
        protected_reference_names=protected_names,
        # An inherited scene image can replay old actions across a cut. Keep
        # stored memory and selected evidence, but reserve this image guide for
        # frame continuation or an explicitly enabled internal experiment.
        include_associative_core=(
            not native_archive
            and story_context_experiment == "core_plus_recall"
            and (internal_test or inputs.get("Composition", "Continue frame") == "Continue frame")
        ),
        composition=str(inputs.get("Composition", "Continue frame")),
        separate_evidence_images=reference_context == "Native" and not internal_test,
        scene_entity_names=_scene_entities(inputs.get("Scene entities", "")),
        render_profile=render_profile,
    )
    prepared = prepare_story_generation(request, store=store, contract=contract)
    prompt_format = inputs.get("Prompt format", "Current")
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError("Prompt format is unsupported")
    if sampler is StorySampler.FULL_HD_2PASS and prompt_format != H3_PROMPT_FORMAT:
        raise ValueError("Full HD 2-pass requires H3 automatic v1 prompting")
    if prompt_format == H3_PROMPT_FORMAT:
        if memory_backend is not StoryMemoryBackend.MINIMAX_H3 or reference_context != "Native":
            raise ValueError("H3 automatic prompting requires native MiniMax H3 references")
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
        if memory_backend is not StoryMemoryBackend.MINIMAX_H3:
            raise ValueError("Structured reference prompts require MiniMax H3")
        prepared = replace(prepared, resolved_prompt=structured_reference_prompt(prepared))
    if (
        memory_backend is StoryMemoryBackend.MINIMAX_H3
        and len(prepared.compiler_sources) + int(motion_reference is not None) > 9
    ):
        raise ValueError("MiniMax H3 Story supports at most nine selected visual sources")
    if (
        reference_context == "Compiled preview"
        and memory_backend is not StoryMemoryBackend.MINIMAX_H3
    ):
        raise ValueError("Compiled preview currently requires the MiniMax H3 memory backend")
    compiler_method = _compiler_method(inputs, internal_test=internal_test)
    reference_compile = (
        _prepare_reference_compile(prepared, motion_reference, method=compiler_method)
        if reference_context == "Compiled preview"
        else None
    )
    if living_canon:
        prepared = _bind_execution(
            prepared,
            reference_compile,
            motion_reference,
            authored_audio,
            output_duration_ms,
            ending_frame,
            prompt_protocol=H3_PROMPT_PROTOCOL if prompt_format == H3_PROMPT_FORMAT else None,
        )
    return PreparedNodeGeneration(
        prepared,
        prepared.visual_guides,
        prepared.resolved_prompt,
        _FRAME_COUNTS[duration],
        request.variation,
        request.sampler,
        motion_reference,
        reference_compile,
        internal_test,
        str(inputs.get("Benchmark phase", "")).strip() if internal_test else "",
        str(inputs.get("Benchmark workload", "")).strip() if internal_test else "",
        compiler_method.value if internal_test else "",
        str(inputs.get("Benchmark cell", "")).strip() if internal_test else "",
        authored_audio,
        output_duration_ms,
        ending_frame,
        render_profile,
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
    filename_prefix: str = "duet_story/shot",
    memory_vae: object | None = None,
) -> StoryCommitResult | NativeStoryCommitResult:
    """Run the GPU memory transaction after MiniMax decoding and saving complete."""
    video = _read_regular(
        _saved_video_path(saved_video, filename_prefix),
        "saved video",
    )
    digest = hashlib.sha256(video).hexdigest()
    if prepared.request.native_reference_archive:
        store = StoryProjectStore(_story_root())
        store.put_asset(video)
        native_result = commit_native_story_generation(
            StoryCommitRequest(
                prepared, decoded_images, digest, f"duet-evidence://story/sha256/{digest}"
            ),
            store=store,
        )
        if prepared.request.execution_sha256 is not None:
            StoryAttemptIndex(store.root).publish(
                prepared.request.execution_sha256, native_result.state
            )
        return native_result
    if prepared.request.memory_backend is StoryMemoryBackend.MINIMAX_H3:
        settings = _minimax_runtime_settings()
        if settings.checkpoint_sha256 != prepared.request.checkpoint_sha256:
            raise ValueError("MiniMax H3 memory checkpoint changed after preparation")
        if memory_vae is None:
            raise ValueError("MiniMax H3 memory commit requires the loaded video VAE")
        runtime = MiniMaxH3StoryRuntime.load_pinned(
            settings.checkpoint,
            expected_checkpoint_sha256=settings.checkpoint_sha256,
            expected_foundation_sha256=settings.foundation_sha256,
            expected_protocol_sha256=settings.protocol_sha256,
            model_configuration_sha256=MINIMAX_MODEL_CONFIGURATION_SHA256,
            vae_sha256=settings.vae_sha256,
            codec=ComfyMiniMaxH3Codec(memory_vae),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    else:
        source_commit, source_archive = _ltx_source_identity()
        runtime = StoryLTXRuntime.load_pinned(
            _checkpoint(),
            source_commit=source_commit,
            source_archive_sha256=source_archive,
            contract=_ltx_contract(),
            model_configuration_sha256=MODEL_CONFIGURATION_SHA256,
        )
    try:
        store = StoryProjectStore(_story_root())
        store.put_asset(video)
        result = commit_story_generation(
            StoryCommitRequest(
                prepared,
                decoded_images,
                digest,
                f"duet-evidence://story/sha256/{digest}",
            ),
            store=store,
            runtime=runtime,
        )
        if prepared.request.execution_sha256 is not None:
            StoryAttemptIndex(store.root).publish(prepared.request.execution_sha256, result.state)
        return result
    finally:
        runtime.close()


__all__ = (
    "MINIMAX_MODEL_CONFIGURATION_SHA256",
    "MODEL_CONFIGURATION_SHA256",
    "ComfyMiniMaxH3Codec",
    "PreparedNodeGeneration",
    "commit_node_generation",
    "parse_memory_actions",
    "prepare_node_generation",
    "require_speed_sampler",
    "story_inspector_summary",
)
