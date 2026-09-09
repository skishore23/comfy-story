"""Customer-facing preparation and post-decode commit lifecycle for Duet Story."""

from __future__ import annotations

import gc
import hashlib
import io
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from comfy_story.cache import AuthoritativeLeaf, DuetXProductTree
from comfy_story.checkpoint import TrainingModules
from comfy_story.contracts import (
    DuetXContract,
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    RawEvidencePointer,
    SpatialLocation,
    TimeFrameRange,
    tensor_sha256,
)
from comfy_story.h3_reference_contracts import H3ReferenceKind, H3SelectedSource
from comfy_story.latent_bridge import LatentHistoryBridge
from comfy_story.ltx_bridge import LTXLatentHistoryBridge
from comfy_story.ltx_n128_runtime import OfficialLatentDecoder
from comfy_story.ltx_quality_operations import PinnedLTXRGBMaterializer
from comfy_story.ltx_quality_protocol import FROZEN_TRAINABLE_CHECKPOINT_SHA256, canonical_json
from comfy_story.ltx_quality_runtime import FrozenQualityRuntimeIdentity, PinnedOfficialLTXFactory
from comfy_story.selection import canonicalize_items
from comfy_story.story_contracts import (
    DuetStoryStateRef,
    ReferenceRole,
    ShotIntent,
    StoryLibrary,
    StoryReference,
    canonical_story_json,
)
from comfy_story.story_memory import (
    append_story_leaf,
    apply_story_memory_policy,
    compose_inclusive_story_core,
    compose_story_blocks,
    empty_story_blocks,
    story_evidence_reservoir,
)
from comfy_story.story_memory_backend import (
    EncodedStoryFrame,
    StoryMemoryBackend,
    StoryMemoryIdentity,
    StoryMemoryRuntime,
    StorySampler,
)
from comfy_story.story_native_archive import (
    NativeArchiveRevision,
    NativeReferenceArchive,
)
from comfy_story.story_observations import (
    extract_story_observation_frames,
    timeline_start_ns,
)
from comfy_story.story_product_contracts import (
    CanonEntity,
    CanonPresence,
    GuideBinding,
    PendingCanonObservation,
    StoryCanon,
    StoryEvidenceRecord,
    StoryGenerationReceipt,
    StoryMemoryCommand,
    StoryMemoryPolicy,
    StoryObservation,
    StoryObservationPacket,
    StoryProductState,
    StoryRecallDecision,
)
from comfy_story.story_recall import (
    apply_story_memory_commands,
    render_story_evidence_packet,
    select_story_recall,
)
from comfy_story.story_store import (
    LoadedStoryRevision,
    StoryCommitPayload,
    StoryGuideBundle,
    StoryProjectStore,
    migrate_minimax_v2_product_state,
)
from comfy_story.training import build_training_modules

_FRAME_SHAPE = (384, 384, 3)
_INTENT_EVENT_TYPE = {
    ShotIntent.START_STORY: 0,
    ShotIntent.CONTINUE_THIS_SHOT: 1,
    ShotIntent.NEXT_SHOT: 2,
    ShotIntent.NEW_SCENE: 3,
}
_SHA256_HEX = frozenset("0123456789abcdef")


def _digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_comfy_images(value: torch.Tensor, *, field: str = "IMAGE") -> torch.Tensor:
    """Require a finite Comfy IMAGE batch without changing its pixels."""
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.ndim != 4
        or value.shape[0] < 1
        or value.shape[1] < 1
        or value.shape[2] < 1
        or value.shape[3] != 3
        or not torch.is_floating_point(value)
        or not bool(torch.isfinite(value).all().item())
        or bool((value < 0).any().item())
        or bool((value > 1).any().item())
    ):
        raise ValueError(f"{field} must be a finite RGB Comfy IMAGE tensor in [0, 1]")
    return value.contiguous()


def normalize_story_frame(images: torch.Tensor) -> np.ndarray[Any, Any]:
    """Normalize the final Comfy frame to the pinned LTX RGB boundary."""
    final = validate_comfy_images(images)[-1]
    array = (
        final.detach()
        .to(device="cpu", dtype=torch.float32)
        .mul(255)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    image = Image.fromarray(np.ascontiguousarray(array), mode="RGB")
    resized = image.resize((384, 384), resample=Image.Resampling.LANCZOS)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def _png_from_rgb(value: np.ndarray[Any, Any]) -> bytes:
    if value.dtype != np.uint8 or tuple(value.shape) != _FRAME_SHAPE:
        raise ValueError("story context must be RGB uint8 384x384")
    output = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(value), mode="RGB").save(
        output, format="PNG", compress_level=9
    )
    return output.getvalue()


def _comfy_from_png(value: bytes) -> torch.Tensor:
    try:
        with Image.open(io.BytesIO(value)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as error:
        raise ValueError("authenticated story context is not a decodable image") from error
    return torch.from_numpy(np.array(rgb, copy=True)).to(torch.float32).div(255).unsqueeze(0)


def _rgb_from_png(value: bytes) -> np.ndarray[Any, Any]:
    try:
        with Image.open(io.BytesIO(value)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as error:
        raise ValueError("authenticated Story frame is not a decodable image") from error
    if tuple(rgb.shape) != _FRAME_SHAPE:
        raise ValueError("authenticated Story frame must be RGB uint8 384x384")
    return np.ascontiguousarray(rgb)


@dataclass(frozen=True, slots=True)
class StoryGenerationRequest:
    """Everything required to validate and prepare one visible story node."""

    intent: ShotIntent
    project_id: str
    branch_id: str
    previous_story: DuetStoryStateRef | None
    previous_frame: torch.Tensor | None
    world_frame: torch.Tensor | None
    library: StoryLibrary | None
    reference_images: Mapping[str, torch.Tensor]
    prompt: str
    shot_length_seconds: int
    variation: int
    checkpoint_sha256: str | None
    model_configuration_sha256: str
    reference_policy: str = "Automatic"
    memory_backend: StoryMemoryBackend = StoryMemoryBackend.LTX
    sampler: StorySampler = StorySampler.NATIVE_RES_MULTISTEP
    memory_commands: tuple[StoryMemoryCommand, ...] = ()
    living_canon: bool = False
    protected_reference_names: tuple[str, ...] = ()
    include_associative_core: bool = True
    execution_sha256: str | None = None
    composition: str = "Continue frame"
    separate_evidence_images: bool = False
    scene_entity_names: tuple[str, ...] | None = None
    render_profile: str = "Reference shot"
    native_reference_archive: bool = False
    shot_state_evidence: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedStoryGeneration:
    """Validated inputs and the exact ordered model-reference roster."""

    request: StoryGenerationRequest
    parent: LoadedStoryRevision | NativeArchiveRevision | None
    library: StoryLibrary
    active_reference_names: tuple[str, ...]
    visual_roles: tuple[str, ...]
    visual_guides: tuple[torch.Tensor, ...]
    resolved_prompt: str
    product_state: StoryProductState | None = None
    effective_blocks: tuple[DuetXProductTree, ...] | None = None
    recall_decision: StoryRecallDecision | None = None
    compiler_sources: tuple[H3SelectedSource, ...] = ()

    @property
    def parent_shot_count(self) -> int:
        return 0 if self.parent is None else self.parent.state.shot_count


def _reference_only_composition(request: StoryGenerationRequest) -> bool:
    """Only the public native reference path can start without a scene image."""
    return (
        request.native_reference_archive is True
        and request.living_canon
        and request.memory_backend is StoryMemoryBackend.MINIMAX_H3
        and request.render_profile == "Reference shot"
        and request.composition == "New composition"
    )


def _validate_intent(request: StoryGenerationRequest) -> ShotIntent:
    try:
        intent = ShotIntent(request.intent)
    except ValueError as error:
        raise ValueError("Create must be a supported story intent") from error
    if intent is ShotIntent.START_STORY:
        if request.previous_story is not None:
            raise ValueError("Start Story must not receive Previous Story")
        if request.world_frame is None and not _reference_only_composition(request):
            raise ValueError("Start Story requires World / starting frame")
    else:
        if request.previous_story is None:
            raise ValueError(f"{intent.value} requires Previous Story")
        if intent is ShotIntent.CONTINUE_THIS_SHOT and request.previous_frame is None:
            raise ValueError("Continue This Shot requires Previous Frame")
        if (
            intent is ShotIntent.NEW_SCENE
            and request.world_frame is None
            and not _reference_only_composition(request)
        ):
            raise ValueError("New Scene requires World / starting frame")
    return intent


def _baseline_product_state(library: StoryLibrary) -> StoryProductState:
    entities = tuple(
        sorted(
            (
                CanonEntity(
                    reference.name.casefold(),
                    reference.name,
                    reference.media_sha256,
                    reference.note,
                    CanonPresence.UNKNOWN,
                    (),
                    None,
                    (),
                ).validate()
                for reference in library.references
            ),
            key=lambda entity: entity.entity_id,
        )
    )
    return StoryProductState((), (), StoryCanon(entities), StoryMemoryPolicy()).validate()


def _guide_bindings(
    roles: tuple[str, ...], guides: tuple[torch.Tensor, ...]
) -> tuple[GuideBinding, ...]:
    if len(roles) != len(guides):
        raise ValueError("Story guide roles and tensors must align")
    return tuple(
        GuideBinding(role, tensor_sha256(validate_comfy_images(guide))).validate()
        for role, guide in zip(roles, guides, strict=True)
    )


def _baseline_reference_clause(picture: int, reference: StoryReference) -> str:
    scope = (
        "Preserve its recognizable architecture and materials; incidental people, props and "
        "actions are not current story state."
        if reference.role is ReferenceRole.LOCATION
        else "Preserve its recognizable design; incidental background, pose, other objects and "
        "actions are not current story state."
    )
    return (
        f"<Picture {picture}> is a baseline {reference.role.value.lower()} reference "
        f"for @{reference.name}. {scope} Use the shot direction for current events and state."
    )


def _scene_entity_ids(
    request: StoryGenerationRequest, reference_entity_ids: tuple[str, ...]
) -> tuple[str, ...]:
    """Declared cast labels pending evidence; it does not prove rendered presence."""
    return (
        reference_entity_ids
        if request.scene_entity_names is None
        else tuple(name.casefold() for name in request.scene_entity_names)
    )


def _prepare_living_minimax(
    request: StoryGenerationRequest,
    *,
    store: StoryProjectStore,
    parent: LoadedStoryRevision | NativeArchiveRevision | None,
    library: StoryLibrary,
    current: torch.Tensor | None,
) -> PreparedStoryGeneration:
    if request.memory_backend is not StoryMemoryBackend.MINIMAX_H3:
        raise ValueError("Living Canon currently requires the MiniMax H3 memory backend")
    reservoir: tuple[ExceptionItem, ...] = ()
    effective_blocks: tuple[DuetXProductTree, ...] | None = None
    if parent is None:
        if request.memory_commands or request.shot_state_evidence:
            raise ValueError("Start Story cannot apply commands without a parent revision")
        product = _baseline_product_state(library)
        parent_revision = "0" * 64
        reservoir = ()
    else:
        if isinstance(parent, NativeArchiveRevision):
            product = parent.product_state
        else:
            product = (
                parent.product_state
                if parent.product_state is not None
                else migrate_minimax_v2_product_state(parent)
            )
        parent_revision = parent.state.revision_sha256
    applied = apply_story_memory_commands(
        product,
        library,
        request.memory_commands,
        parent_revision_sha256=parent_revision,
    )
    if isinstance(parent, LoadedStoryRevision):
        tombstones = set(applied.product_state.policy.tombstoned_evidence_ids)
        effective_blocks = apply_story_memory_policy(
            parent.blocks,
            product.policy,
            applied.product_state.policy,
            excluded_source_ranks=frozenset(
                record.source_shot_index
                for record in product.evidence_records
                if record.evidence_id in tombstones and record.source_shot_index is not None
            ),
        )
        reservoir = story_evidence_reservoir(effective_blocks)
    decision = select_story_recall(
        applied,
        library,
        reservoir,
        prompt=request.prompt,
        reference_policy=request.reference_policy,
        image_references=request.render_profile == "Reference shot",
        shot_state_evidence=request.shot_state_evidence,
    )
    scene_entity_ids = _scene_entity_ids(request, decision.explicit_entity_ids)
    image_entity_ids = (
        decision.explicit_entity_ids if request.render_profile == "Reference shot" else ()
    )
    # A cut must not silently reintroduce the previous scene (and its cast).
    # Start Story and New Scene still honor their explicitly supplied world image.
    inherited_cut = (
        request.composition == "New composition"
        and request.previous_frame is not None
        and request.intent in {ShotIntent.NEXT_SHOT, ShotIntent.CONTINUE_THIS_SHOT}
    )
    omit_scene = inherited_cut or current is None
    role_list = [] if omit_scene else ["current"]
    guide_list: list[torch.Tensor] = [] if omit_scene or current is None else [current]
    core_invalidated = (
        isinstance(parent, LoadedStoryRevision)
        and effective_blocks is not None
        and any(
            not torch.equal(before.root.dense_operator, after.root.dense_operator)
            for before, after in zip(parent.blocks, effective_blocks, strict=True)
        )
    )
    if (
        isinstance(parent, LoadedStoryRevision)
        and request.include_associative_core
        and not inherited_cut
        and not core_invalidated
        and request.render_profile == "Reference shot"
    ):
        role_list.append("inclusive-core")
        guide_list.append(_comfy_from_png(parent.story_context_png))
    reference_tensors = {
        name.casefold(): validate_comfy_images(image, field=f"@{name}")[:1]
        for name, image in request.reference_images.items()
    }
    shot_state_entities = dict(request.shot_state_evidence)
    for spec in decision.packet_specs:
        if request.separate_evidence_images and len(spec.source_sha256s) == 2:
            # Preserve the packet's historical contract, but never show its
            # baseline/state contact sheet to a native video generator.
            # An explicitly selected native state replaces the old appearance
            # in render conditioning. Supplying both can ask the model to animate
            # the old-to-new transition again. The original identity stays in
            # the library and authenticated packet for independent verification.
            if not (request.native_reference_archive and spec.entity_id in shot_state_entities):
                role_list.append(f"identity-{spec.entity_id}")
                try:
                    guide_list.append(reference_tensors[spec.entity_id])
                except KeyError as error:
                    raise ValueError(
                        f"Story Library image is unavailable for @{spec.entity_id}"
                    ) from error
            role_list.append(f"evidence-{spec.entity_id}")
            guide_list.append(_comfy_from_png(store.load_asset(spec.source_sha256s[-1])))
            continue
        role_list.append(f"evidence-{spec.entity_id}")
        if spec.state_evidence_id is None:
            try:
                guide_list.append(reference_tensors[spec.entity_id])
            except KeyError as error:
                raise ValueError(
                    f"Story Library image is unavailable for @{spec.entity_id}"
                ) from error
        else:
            guide_list.append(
                _comfy_from_png(render_story_evidence_packet(spec, load_asset=store.load_asset))
            )
    exact_entity_ids = {spec.entity_id for spec in decision.packet_specs}
    for entity_id in image_entity_ids:
        if entity_id in exact_entity_ids:
            continue
        role_list.append(f"reference-{entity_id}")
        try:
            guide_list.append(reference_tensors[entity_id])
        except KeyError as error:
            raise ValueError(f"Story Library image is unavailable for @{entity_id}") from error
    if not guide_list:
        raise ValueError("New composition requires a selected visual reference for this cut")
    roles = tuple(role_list)
    guides = tuple(guide_list)
    if len(guides) > 9:
        raise ValueError("MiniMax H3 Story supports at most nine semantic guides")
    bindings = _guide_bindings(roles, guides)
    inclusive_core_index = next(
        (index for index, role in enumerate(roles) if role == "inclusive-core"), None
    )
    decision = replace(
        decision,
        inclusive_core_sha256=(
            None if inclusive_core_index is None else tensor_sha256(guides[inclusive_core_index])
        ),
        guide_bindings=bindings,
    ).validate()
    clauses: list[str] = []
    baseline_entities = {
        spec.entity_id for spec in decision.packet_specs if spec.state_evidence_id is None
    }
    references_by_id = {reference.name.casefold(): reference for reference in library.references}
    for picture, role in enumerate(roles, start=1):
        if role == "current":
            clauses.append(
                f"Use <Picture {picture}> as the current scene and composition anchor."
                if request.composition == "Continue frame"
                else f"<Picture {picture}> supplies prior visual context only. Compose a new "
                "camera angle and scene according to the user direction; do not copy its framing."
            )
        elif role == "inclusive-core":
            clauses.append(
                f"<Picture {picture}> is inclusive chronological Story Context; preserve "
                "durable identity and events without copying its composition."
            )
        elif role.startswith("identity-"):
            entity_id = role.removeprefix("identity-")
            clauses.append(
                f"<Picture {picture}> is the original identity reference for @{entity_id}. "
                "Use its recognizable design only; its old action, location, and mutable state "
                "are not the current event. The separate current-evidence picture takes "
                "precedence for the selected appearance in this shot; "
                "it does not update creator canon."
                if entity_id in shot_state_entities
                else f"<Picture {picture}> is the original identity reference for @{entity_id}. "
                "Use its recognizable design only; its old action, location, and mutable state "
                "are not the current event. The separate current-evidence picture takes "
                "precedence for confirmed state."
            )
        elif role.startswith("evidence-"):
            entity_id = role.removeprefix("evidence-")
            clauses.append(
                _baseline_reference_clause(picture, references_by_id[entity_id])
                if entity_id in baseline_entities
                else f"<Picture {picture}> is selected earlier visual evidence for @{entity_id}, "
                "not creator-approved canon. Use its relevant appearance for this shot without "
                "replaying its old action or composition."
                if entity_id in shot_state_entities
                else f"<Picture {picture}> is exact current evidence for @{entity_id}; preserve "
                "its identity and confirmed state."
            )
        else:
            entity_id = role.removeprefix("reference-")
            clauses.append(_baseline_reference_clause(picture, references_by_id[entity_id]))
    has_baseline_notes = False
    for entity in applied.product_state.canon.entities:
        if entity.entity_id in scene_entity_ids:
            if entity.state_note.strip():
                if entity.confirmed_revision_sha256 is None:
                    has_baseline_notes = True
                    note = entity.state_note.strip()
                    ending = "" if note.endswith((".", "!", "?")) else "."
                    clauses.append(
                        f"Baseline reference notes for @{entity.reference_name} "
                        f"(not approved current state): {note}{ending}"
                    )
                else:
                    clauses.append(
                        f"Earlier approved state of @{entity.reference_name}: "
                        f"{entity.state_note.strip()}. "
                        "Selected visual evidence supplies this shot's appearance "
                        "without changing that approval."
                        if entity.entity_id in shot_state_entities
                        else f"Approved state of @{entity.reference_name}: "
                        f"{entity.state_note.strip()}."
                    )
        elif entity.presence is CanonPresence.OFF_SCREEN:
            clauses.append(
                f"@{entity.reference_name} is off screen in this shot; do not depict them."
            )
    if has_baseline_notes:
        clauses.append(
            "Do not enact future events mentioned in reference notes. "
            "The shot direction determines current events and state."
        )
    clauses.append(f"User direction: {request.prompt.strip()}")
    names_by_id = {reference.name.casefold(): reference.name for reference in library.references}
    compiler_sources = tuple(
        H3SelectedSource(
            source_id=f"story:{role}",
            kind=H3ReferenceKind.IMAGE,
            ordinal=ordinal,
            protected=role.startswith("evidence-"),
        ).validate()
        for ordinal, role in enumerate(roles)
    )
    return PreparedStoryGeneration(
        request=request,
        parent=parent,
        library=library,
        active_reference_names=tuple(names_by_id[item] for item in image_entity_ids),
        visual_roles=roles,
        visual_guides=guides,
        resolved_prompt=" ".join(clauses),
        product_state=applied.product_state,
        effective_blocks=effective_blocks,
        recall_decision=decision,
        compiler_sources=compiler_sources,
    )


def prepare_story_generation(
    request: StoryGenerationRequest,
    *,
    store: StoryProjectStore,
    contract: DuetXContract | None,
) -> PreparedStoryGeneration:
    """Fail-fast, load authenticated state, and construct the ordered H3 guide roster."""
    if request.shot_state_evidence and not request.living_canon:
        raise ValueError("shot state evidence requires public Story reference generation")
    if request.render_profile not in {"Reference shot", "Animate frame"}:
        raise ValueError("Render profile must be Reference shot or Animate frame")
    if request.render_profile == "Animate frame" and (
        not request.living_canon
        or request.memory_backend is not StoryMemoryBackend.MINIMAX_H3
        or request.composition != "Continue frame"
        or request.sampler is StorySampler.SPEED_EULER_2STAGE
    ):
        raise ValueError("Animate frame requires public H3, Continue frame, and Native or Turbo")
    intent = _validate_intent(request)
    if type(request.native_reference_archive) is not bool:
        raise ValueError("native_reference_archive must be a bool")
    if request.native_reference_archive:
        if (
            not request.living_canon
            or request.memory_backend is not StoryMemoryBackend.MINIMAX_H3
            or request.checkpoint_sha256 is not None
            or contract is not None
            or request.include_associative_core
        ):
            raise ValueError(
                "Native reference archives require public H3 with no trained checkpoint, "
                "fusion contract or dense core"
            )
    else:
        if contract is None:
            raise ValueError("Trained Story generation requires its fusion contract")
        contract.validate()
    if not isinstance(request.memory_backend, StoryMemoryBackend):
        raise ValueError("memory_backend must be a StoryMemoryBackend")
    if not isinstance(request.sampler, StorySampler):
        raise ValueError("sampler must be a StorySampler")
    expected_format = {
        StoryMemoryBackend.LTX: "duet-x-ltx-decision1-v1",
        StoryMemoryBackend.MINIMAX_H3: "duet-x-minimax-h3-v1",
    }[request.memory_backend]
    if contract is not None and contract.format != expected_format:
        raise ValueError("memory backend does not match the requested contract")
    if not request.native_reference_archive:
        _digest(request.checkpoint_sha256, "checkpoint_sha256")
    _digest(request.model_configuration_sha256, "model_configuration_sha256")
    if request.shot_length_seconds not in {5, 10, 15}:
        raise ValueError("shot length must be 5, 10, or 15 seconds")
    if type(request.variation) is not int or not 0 <= request.variation <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Variation must be an unsigned 64-bit integer")
    if request.reference_policy not in {"Automatic", "Prompt mentions only"}:
        raise ValueError("Reference policy is unsupported")
    if type(request.include_associative_core) is not bool:
        raise ValueError("include_associative_core must be a bool")
    if type(request.separate_evidence_images) is not bool:
        raise ValueError("separate_evidence_images must be a bool")
    if request.composition not in {"Continue frame", "New composition"}:
        raise ValueError("Composition is unsupported")
    if request.composition != "Continue frame" and request.memory_backend is StoryMemoryBackend.LTX:
        raise ValueError("New composition currently requires MiniMax H3")
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ValueError("What happens next? must be nonempty")

    parent: LoadedStoryRevision | NativeArchiveRevision | None = None
    if request.previous_story is not None:
        if request.native_reference_archive:
            parent = NativeReferenceArchive(store).load(
                request.previous_story,
                model_configuration_sha256=request.model_configuration_sha256,
            )
        else:
            assert contract is not None
            parent = store.load(
                request.previous_story,
                contract,
                _digest(request.checkpoint_sha256, "checkpoint_sha256"),
            )
    if parent is not None:
        if parent.state.shot_count >= 128:
            raise ValueError("Story has reached its 128-shot capacity; start a new story")
        if parent.state.model_configuration_sha256 != request.model_configuration_sha256:
            raise ValueError("Previous Story uses a different model configuration")
        if parent.state.project_id != request.project_id:
            raise ValueError("Previous Story belongs to a different project")
        if request.library is not None and request.library != parent.library:
            raise ValueError("Story Library does not match the inherited project library")
        library = parent.library
    else:
        if request.library is None:
            raise ValueError("Start Story requires a Story Library")
        library = request.library.validate()
    if request.scene_entity_names is not None:
        names = request.scene_entity_names
        if not request.living_canon:
            raise ValueError("Declared scene entities require the public Story generation path")
        if not isinstance(names, tuple) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("scene_entity_names must be a tuple of nonempty names")
        folded = tuple(name.casefold() for name in names)
        if len(folded) != len(set(folded)):
            raise ValueError("Declared scene entities must be unique, ignoring case")
        if not set(folded) <= {reference.name.casefold() for reference in library.references}:
            raise ValueError("Declared scene entities must belong to the Story Library")
    if not isinstance(request.protected_reference_names, tuple) or any(
        not isinstance(name, str) or not name for name in request.protected_reference_names
    ):
        raise ValueError("protected_reference_names must be a tuple of nonempty strings")
    protected_folded = tuple(name.casefold() for name in request.protected_reference_names)
    if len(protected_folded) != len(set(protected_folded)):
        raise ValueError("protected reference names must be unique, ignoring case")
    if len(protected_folded) > 2:
        raise ValueError("at most two protected reference names are supported")
    active: tuple[StoryReference, ...]
    if request.living_canon:
        if request.protected_reference_names:
            raise ValueError("Living Canon uses memory actions instead of protected references")
        active = ()
    elif request.memory_backend is StoryMemoryBackend.MINIMAX_H3:
        inherited_count = 0
        if parent is not None:
            if not isinstance(parent, LoadedStoryRevision):
                raise ValueError("Native archives require Living Canon")
            inherited_bundle = parent.require_guide_bundle()
            inherited_count = int(inherited_bundle.core_png is not None) + len(
                inherited_bundle.exception_pngs
            )
        active = library.resolve_for_h3(request.prompt, limit=8 - inherited_count)
    else:
        active = library.resolve_mentions(request.prompt)
    active_folded = {reference.name.casefold() for reference in active}
    if any(name.casefold() not in active_folded for name in request.protected_reference_names):
        raise ValueError("protected references must be selected by @mention")
    lookup = {name.casefold(): image for name, image in request.reference_images.items()}
    selected: list[torch.Tensor] = []
    for reference in active:
        image = lookup.get(reference.name.casefold())
        if image is None:
            raise ValueError(f"Story Library image is unavailable for @{reference.name}")
        selected.append(validate_comfy_images(image, field=f"@{reference.name}")[:1])
    if intent is ShotIntent.NEW_SCENE or request.previous_frame is None:
        current_source = request.world_frame
    else:
        current_source = request.previous_frame
    if current_source is None and not _reference_only_composition(request):
        raise ValueError(f"{intent.value} requires a usable current or world frame")
    current = (
        None
        if current_source is None
        else validate_comfy_images(current_source, field="current-or-starting-frame")[:1]
    )
    if request.living_canon:
        return _prepare_living_minimax(
            request,
            store=store,
            parent=parent,
            library=library,
            current=current,
        )
    assert current is not None
    if parent is not None and not isinstance(parent, LoadedStoryRevision):
        raise ValueError("Native archives require Living Canon")
    names = tuple(reference.name for reference in active)
    roles: tuple[str, ...]
    guides: tuple[torch.Tensor, ...]
    if request.memory_backend is StoryMemoryBackend.LTX:
        if len(selected) == 1:
            selected.append(selected[0].clone())
        context = (
            torch.zeros_like(current)
            if parent is None
            else _comfy_from_png(parent.story_context_png)
        )
        roles = (
            "current-or-starting-frame",
            "duet-x-story-context",
            f"exact-reference:{names[0]}",
            f"exact-reference:{names[-1]}",
        )
        guides = (current, context, selected[0], selected[1])
        resolved = " ".join(
            (
                "Use <Picture 1> as the current scene and composition anchor.",
                (
                    "<Picture 2> carries no prior story history."
                    if parent is None
                    else "<Picture 2> is chronological Story Context from earlier shots; "
                    "preserve its durable identity and events without copying its composition."
                ),
                f"<Picture 3> is @{names[0]}, which must remain recognizable.",
                f"<Picture 4> is @{names[-1]}, which must remain recognizable.",
                f"User direction: {request.prompt.strip()}",
            )
        )
    else:
        role_list = ["current-or-starting-frame"]
        guide_list = [current]
        if parent is not None:
            bundle = parent.require_guide_bundle()
            if bundle.core_png is not None:
                role_list.append("duet-x-core")
                guide_list.append(_comfy_from_png(bundle.core_png))
            for ordinal, value in enumerate(bundle.exception_pngs):
                role_list.append(f"duet-x-exception:{ordinal}")
                guide_list.append(_comfy_from_png(value))
        for reference, image in zip(active, selected, strict=True):
            role_list.append(f"exact-reference:{reference.name}")
            guide_list.append(image)
        if len(role_list) > 9:
            raise ValueError("MiniMax H3 Story supports at most nine ordered image references")
        roles = tuple(role_list)
        guides = tuple(guide_list)
        clauses: list[str] = []
        for picture, role in enumerate(roles, start=1):
            if role == "current-or-starting-frame":
                clause = (
                    f"Use <Picture {picture}> as the current scene and composition anchor."
                    if request.composition == "Continue frame"
                    else f"<Picture {picture}> supplies prior visual context only. Compose a new "
                    "camera angle and scene according to the user direction; "
                    "do not copy its framing."
                )
            elif role == "duet-x-core":
                clause = (
                    f"<Picture {picture}> is chronological Duet Story core memory; preserve "
                    "its durable identity and events without copying its composition."
                )
            elif role.startswith("duet-x-exception:"):
                ordinal = int(role.rsplit(":", 1)[1])
                clause = (
                    f"<Picture {picture}> is exact protected Story event {ordinal + 1}; "
                    "preserve its specific identity and rare details."
                )
            else:
                reference_name = role.removeprefix("exact-reference:")
                clause = (
                    f"<Picture {picture}> is @{reference_name}, which must remain recognizable."
                )
            clauses.append(clause)
        clauses.append(f"User direction: {request.prompt.strip()}")
        resolved = " ".join(clauses)
    protected_set = frozenset(protected_folded)
    compiler_sources = (
        tuple(
            H3SelectedSource(
                source_id=(
                    f"library:{role.removeprefix('exact-reference:')}"
                    if role.startswith("exact-reference:")
                    else f"story:{role}"
                ),
                kind=H3ReferenceKind.IMAGE,
                ordinal=ordinal,
                protected=(
                    role.startswith("exact-reference:")
                    and role.removeprefix("exact-reference:").casefold() in protected_set
                ),
            ).validate()
            for ordinal, role in enumerate(roles)
        )
        if request.memory_backend is StoryMemoryBackend.MINIMAX_H3
        else ()
    )
    return PreparedStoryGeneration(
        request=request,
        parent=parent,
        library=library,
        active_reference_names=names,
        visual_roles=roles,
        visual_guides=guides,
        resolved_prompt=resolved,
        compiler_sources=compiler_sources,
    )


@dataclass(frozen=True, slots=True)
class StoryCommitRequest:
    prepared: PreparedStoryGeneration
    decoded_images: torch.Tensor
    saved_video_sha256: str
    saved_video_locator: str


@dataclass(frozen=True, slots=True)
class StoryCommitResult:
    state: DuetStoryStateRef
    last_frame: torch.Tensor
    loaded_revision: LoadedStoryRevision


def _story_leaf(
    request: StoryCommitRequest,
    encoded: EncodedStoryFrame,
    frame_png: bytes,
    bridge: LatentHistoryBridge,
    contract: DuetXContract,
    runtime: StoryMemoryRuntime,
) -> AuthoritativeLeaf:
    prepared = request.prepared
    shot_index = prepared.parent_shot_count
    local_slot = shot_index % 8
    key = LeafKey(
        f"{prepared.request.project_id}-{prepared.request.branch_id}-shot-{shot_index:03d}",
        shot_index,
        local_slot,
    ).validate()
    device = bridge.fusion.input_norm.weight.device
    latent = encoded.latent.to(device=device)
    tokens = latent.permute(0, 2, 3, 4, 1).flatten(1, 3).unsqueeze(1)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        operator = bridge.fusion.encode_stream(tokens.to(bridge.fusion.input_norm.weight.dtype))[
            :, 0
        ]
    event_type = _INTENT_EVENT_TYPE[ShotIntent(prepared.request.intent)]
    timestamp = shot_index / 127.0
    raw_score = runtime.score(encoded.latent, event_type, timestamp)
    if not isinstance(raw_score, (int, float)) or not math.isfinite(float(raw_score)):
        raise ValueError("trained story salience score must be finite")
    salience_q = round(torch.sigmoid(torch.tensor(float(raw_score))).item() * 1_000_000)
    source_manifest: dict[str, object] = {
        "active_references": list(prepared.active_reference_names),
        "branch_id": prepared.request.branch_id,
        "frame_latent_sha256": tensor_sha256(encoded.latent),
        "intent": prepared.request.intent.value,
        "model_configuration_sha256": prepared.request.model_configuration_sha256,
        "parent_revision_sha256": (
            None if prepared.parent is None else prepared.parent.state.revision_sha256
        ),
        "project_id": prepared.request.project_id,
        "prompt": prepared.request.prompt.strip(),
        "seed": prepared.request.variation,
        "shot_index": shot_index,
        "video_sha256": request.saved_video_sha256,
    }
    if runtime.identity.backend is StoryMemoryBackend.MINIMAX_H3:
        source_manifest["frame_png_sha256"] = _sha256(frame_png)
    source_fingerprint = _sha256(canonical_story_json(source_manifest))
    item_id = f"story-event-{source_fingerprint[:24]}"
    if runtime.identity.backend is StoryMemoryBackend.MINIMAX_H3:
        raw_sha256 = _sha256(frame_png)
        raw_locator = f"duet-evidence://story-frame/sha256/{raw_sha256}"
        raw_byte_stop = len(frame_png)
    else:
        raw_sha256 = request.saved_video_sha256
        raw_locator = request.saved_video_locator
        raw_byte_stop = 1
    provenance = EvidenceProvenance(
        key,
        TimeFrameRange(
            shot_index * prepared.request.shot_length_seconds * 1_000_000_000,
            (shot_index + 1) * prepared.request.shot_length_seconds * 1_000_000_000,
            0,
            prepared.request.shot_length_seconds * 24,
        ),
        SpatialLocation("normalized", "bbox", (0, 0, 65_536, 65_536)),
        f"duet-story:{prepared.request.intent.value}",
        RawEvidencePointer(
            raw_locator,
            raw_sha256,
            0,
            raw_byte_stop,
        ),
        _sha256(canonical_story_json(source_manifest)),
        encoded.preprocessing_sha256,
        encoded.vae_sha256,
        encoded.adapter_sha256,
        encoded.scorer_sha256,
        contract.sparse.fingerprint(),
    ).validate()
    embedding = encoded.latent.mean(dim=(0, 2, 3, 4)).contiguous()
    item = ExceptionItem(item_id, provenance, embedding, salience_q, False)
    item.validate(contract.sparse)
    return AuthoritativeLeaf(key, source_fingerprint, 0, operator, (item,))


def _living_observation_leaf(
    request: StoryCommitRequest,
    images: torch.Tensor,
    *,
    store: StoryProjectStore,
    runtime: StoryMemoryRuntime,
) -> tuple[
    AuthoritativeLeaf,
    StoryObservationPacket,
    tuple[StoryEvidenceRecord, ...],
    EncodedStoryFrame,
    bytes,
]:
    prepared = request.prepared
    product = prepared.product_state
    decision = prepared.recall_decision
    if product is None or decision is None:
        raise ValueError("Living Canon commit requires prepared product state and recall decision")
    shot_index = prepared.parent_shot_count
    timeline_start = timeline_start_ns(product.observation_packets)
    duration_ns = round(images.shape[0] * 1_000_000_000 / 24)
    extracted = extract_story_observation_frames(
        images,
        timeline_start_ns=timeline_start,
        shot_duration_ns=duration_ns,
    )
    contract = runtime.identity.contract
    bridge = runtime.bridge
    device = bridge.fusion.input_norm.weight.device
    encoded_frames: list[EncodedStoryFrame] = []
    frame_pngs: list[bytes] = []
    asset_digests: list[str] = []
    saliences: list[int] = []
    operators: list[torch.Tensor] = []
    event_type = _INTENT_EVENT_TYPE[ShotIntent(prepared.request.intent)]
    for frame in extracted:
        encoded = runtime.materialize_frame(frame.rgb).validate(contract)
        frame_png = _png_from_rgb(frame.rgb)
        asset_digest = store.put_asset(frame_png)
        raw_score = runtime.score(encoded.latent, event_type, shot_index / 127.0)
        if not isinstance(raw_score, (int, float)) or not math.isfinite(float(raw_score)):
            raise ValueError("trained story salience score must be finite")
        salience = round(torch.sigmoid(torch.tensor(float(raw_score))).item() * 1_000_000)
        latent = encoded.latent.to(device=device)
        tokens = latent.permute(0, 2, 3, 4, 1).flatten(1, 3).unsqueeze(1)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            operator = bridge.fusion.encode_stream(
                tokens.to(bridge.fusion.input_norm.weight.dtype)
            )[:, 0]
        encoded_frames.append(encoded)
        frame_pngs.append(frame_png)
        asset_digests.append(asset_digest)
        saliences.append(salience)
        operators.append(operator)
    key = LeafKey(
        f"{prepared.request.project_id}-{prepared.request.branch_id}-shot-{shot_index:03d}",
        shot_index,
        shot_index % 8,
    ).validate()
    source_manifest: dict[str, object] = {
        "asset_sha256s": asset_digests,
        "branch_id": prepared.request.branch_id,
        "frame_indexes": [frame.frame_index for frame in extracted],
        "intent": prepared.request.intent.value,
        "model_configuration_sha256": prepared.request.model_configuration_sha256,
        "parent_revision_sha256": (
            None if prepared.parent is None else prepared.parent.state.revision_sha256
        ),
        "project_id": prepared.request.project_id,
        "prompt_sha256": _sha256(prepared.request.prompt.strip().encode()),
        "seed": prepared.request.variation,
        "shot_index": shot_index,
        "video_sha256": request.saved_video_sha256,
    }
    if prepared.request.scene_entity_names is not None:
        source_manifest["declared_scene_entities"] = _scene_entity_ids(
            prepared.request, decision.explicit_entity_ids
        )
    source_registry_sha256 = _sha256(canonical_story_json(source_manifest))
    observations: list[StoryObservation] = []
    candidates: list[ExceptionItem] = []
    entity_ids = _scene_entity_ids(prepared.request, decision.explicit_entity_ids)
    for frame, encoded, frame_png, asset_digest, salience in zip(
        extracted,
        encoded_frames,
        frame_pngs,
        asset_digests,
        saliences,
        strict=True,
    ):
        evidence_id = (
            f"shot-{shot_index:03d}-{frame.kind.value}-{frame.frame_index:03d}-{asset_digest[:12]}"
        )
        observation = StoryObservation(
            evidence_id,
            frame.kind,
            frame.frame_index,
            frame.timestamp_start_ns,
            frame.timestamp_stop_ns,
            (0, 0, 65_536, 65_536),
            asset_digest,
            encoded.preprocessing_sha256,
            encoded.vae_sha256,
            tensor_sha256(encoded.latent),
            salience,
            frame.change_score_q,
            entity_ids,
        ).validate()
        provenance = EvidenceProvenance(
            key,
            TimeFrameRange(
                frame.timestamp_start_ns,
                frame.timestamp_stop_ns,
                frame.frame_index,
                frame.frame_index + 1,
            ),
            SpatialLocation("normalized", "bbox", (0, 0, 65_536, 65_536)),
            f"duet-story-observation:{frame.kind.value}",
            RawEvidencePointer(
                f"duet-evidence://story-frame/sha256/{asset_digest}",
                asset_digest,
                0,
                len(frame_png),
            ),
            source_registry_sha256,
            encoded.preprocessing_sha256,
            encoded.vae_sha256,
            encoded.adapter_sha256,
            encoded.scorer_sha256,
            contract.sparse.fingerprint(),
        ).validate()
        item = ExceptionItem(
            evidence_id,
            provenance,
            encoded.latent.mean(dim=(0, 2, 3, 4)).contiguous(),
            salience,
            False,
        )
        item.validate(contract.sparse)
        observations.append(observation)
        candidates.append(item)
    packet = StoryObservationPacket(
        f"shot-{shot_index:03d}",
        shot_index,
        len(images),
        timeline_start,
        timeline_start + duration_ns,
        request.saved_video_sha256,
        request.saved_video_locator,
        _sha256(
            canonical_story_json(
                {
                    "format": "duet-story-observation-extractor-v1",
                    "geometry": "full-frame-q16",
                    "selection": "opening-max-adjacent-change-closing",
                }
            )
        ),
        entity_ids,
        tuple(observations),
    ).validate()
    records = tuple(
        StoryEvidenceRecord.from_observation(packet, observation)
        for observation in packet.observations
    )
    leaf = AuthoritativeLeaf(
        key,
        source_registry_sha256,
        0,
        bridge.fusion.combine_ordered_operators(tuple(operators)),
        canonicalize_items(tuple(candidates)),
    )
    return leaf, packet, records, encoded_frames[-1], frame_pngs[-1]


def append_product_observation(
    product: StoryProductState,
    packet: StoryObservationPacket,
    records: tuple[StoryEvidenceRecord, ...],
    entity_ids: tuple[str, ...],
    video_sha256: str,
) -> StoryProductState:
    """Append reviewable evidence without automatically approving rendered state."""
    closing_evidence_id = packet.observations[-1].evidence_id
    entities: list[CanonEntity] = []
    explicit = set(entity_ids)
    for entity in product.canon.entities:
        pending = entity.pending
        if entity.entity_id in explicit:
            pending = (
                *pending,
                PendingCanonObservation(
                    closing_evidence_id,
                    f"Review generated state from shot {packet.shot_index + 1}",
                    video_sha256,
                ).validate(),
            )
        entities.append(
            CanonEntity(
                entity.entity_id,
                entity.reference_name,
                entity.baseline_sha256,
                entity.state_note,
                entity.presence,
                entity.supporting_evidence_ids,
                entity.confirmed_revision_sha256,
                pending,
            ).validate()
        )
    return StoryProductState(
        (*product.observation_packets, packet),
        tuple(sorted((*product.evidence_records, *records), key=lambda item: item.evidence_id)),
        StoryCanon(tuple(entities)),
        product.policy,
    ).validate()


def _commit_living_minimax(
    request: StoryCommitRequest,
    *,
    store: StoryProjectStore,
    runtime: StoryMemoryRuntime,
    images: torch.Tensor,
    last_frame: torch.Tensor,
) -> StoryCommitResult:
    prepared = request.prepared
    product = prepared.product_state
    decision = prepared.recall_decision
    if product is None or decision is None:
        raise ValueError("Living Canon preparation is incomplete")
    if _guide_bindings(prepared.visual_roles, prepared.visual_guides) != decision.guide_bindings:
        raise ValueError("prepared Story guide roster changed before commit")
    leaf, packet, records, closing_encoded, last_png = _living_observation_leaf(
        request,
        images,
        store=store,
        runtime=runtime,
    )
    contract = runtime.identity.contract
    bridge = runtime.bridge
    blocks = (
        empty_story_blocks(
            contract,
            dense_steps=closing_encoded.latent.shape[2]
            * closing_encoded.latent.shape[3]
            * closing_encoded.latent.shape[4],
            operator_size=bridge.fusion.operator_size,
        )
        if prepared.effective_blocks is None
        else prepared.effective_blocks
    )
    blocks = append_story_leaf(blocks, shot_index=prepared.parent_shot_count, leaf=leaf)
    inclusive = compose_inclusive_story_core(blocks, bridge.fusion)
    device = bridge.fusion.input_norm.weight.device
    context_latent = bridge.materialize_operator(
        inclusive.to(device=device),
        anchor=closing_encoded.latent.to(device=device),
    )
    context_png = _png_from_rgb(runtime.decode(context_latent))
    next_product = append_product_observation(
        product,
        packet,
        records,
        _scene_entity_ids(prepared.request, decision.explicit_entity_ids),
        request.saved_video_sha256,
    )
    receipt = StoryGenerationReceipt(
        _sha256(prepared.request.prompt.strip().encode()),
        None if prepared.parent is None else prepared.parent.state.revision_sha256,
        StoryMemoryBackend.MINIMAX_H3.value,
        _digest(prepared.request.checkpoint_sha256, "checkpoint_sha256"),
        prepared.request.sampler.value,
        decision.selected_evidence_ids,
        decision.guide_bindings,
        _sha256(decision.to_json()),
    ).validate()
    metadata: dict[str, object] = {
        "active_references": list(prepared.active_reference_names),
        "intent": prepared.request.intent.value,
        "memory_backend": StoryMemoryBackend.MINIMAX_H3.value,
        "memory_contract_sha256": contract.fingerprint(),
        "prompt_sha256": receipt.prompt_sha256,
        "sampler": prepared.request.sampler.value,
        "seed": prepared.request.variation,
        "shot_length_seconds": prepared.request.shot_length_seconds,
        "video_sha256": request.saved_video_sha256,
    }
    if prepared.request.scene_entity_names is not None:
        metadata["declared_scene_entities"] = list(
            _scene_entity_ids(prepared.request, decision.explicit_entity_ids)
        )
    if prepared.request.render_profile != "Reference shot":
        metadata["render_profile"] = prepared.request.render_profile
    if prepared.request.execution_sha256 is not None:
        metadata["execution_sha256"] = _digest(
            prepared.request.execution_sha256, "execution_sha256"
        )
    full_frame = last_frame[0].detach().cpu().mul(255).round().to(torch.uint8).numpy()
    full_png = io.BytesIO()
    Image.fromarray(full_frame, mode="RGB").save(full_png, format="PNG")
    metadata["last_frame_full_sha256"] = store.put_asset(full_png.getvalue())
    tensor_bytes = io.BytesIO()
    np.save(tensor_bytes, last_frame.detach().cpu().numpy(), allow_pickle=False)
    metadata["last_frame_tensor_sha256"] = store.put_asset(tensor_bytes.getvalue())
    metadata["decoded_frame_count"] = images.shape[0]
    metadata["fps"] = 24
    metadata["decoded_duration_ns"] = round(images.shape[0] * 1_000_000_000 / 24)
    state = store.publish(
        StoryCommitPayload(
            project_id=prepared.request.project_id,
            branch_id=prepared.request.branch_id,
            parent=None if prepared.parent is None else prepared.parent.state,
            library=prepared.library,
            blocks=blocks,
            checkpoint_sha256=_digest(prepared.request.checkpoint_sha256, "checkpoint_sha256"),
            model_configuration_sha256=prepared.request.model_configuration_sha256,
            last_frame_png=last_png,
            story_context_png=context_png,
            shot_metadata=metadata,
            product_state=next_product,
            generation_receipt=receipt,
        )
    )
    loaded = store.load(
        state, contract, _digest(prepared.request.checkpoint_sha256, "checkpoint_sha256")
    )
    return StoryCommitResult(state, last_frame, loaded)


def commit_story_generation(
    request: StoryCommitRequest,
    *,
    store: StoryProjectStore,
    runtime: StoryMemoryRuntime,
) -> StoryCommitResult:
    """Append one generated shot and publish only after all model work succeeds."""
    if not isinstance(runtime, StoryMemoryRuntime):
        raise ValueError("runtime must satisfy the StoryMemoryRuntime contract")
    identity = runtime.identity.validate()
    prepared_request = request.prepared.request
    if identity.backend is not prepared_request.memory_backend:
        raise ValueError("runtime memory backend does not match the prepared request")
    if identity.checkpoint_sha256 != prepared_request.checkpoint_sha256:
        raise ValueError("runtime checkpoint does not match the prepared request")
    if identity.model_configuration_sha256 != prepared_request.model_configuration_sha256:
        raise ValueError("runtime model configuration does not match the prepared request")
    bridge = runtime.bridge
    contract = identity.contract
    if bridge.fusion.channels != contract.latent_channels:
        raise ValueError("runtime bridge channels do not match its memory contract")
    _digest(request.saved_video_sha256, "saved_video_sha256")
    expected_locator = f"duet-evidence://story/sha256/{request.saved_video_sha256}"
    if request.saved_video_locator != expected_locator:
        raise ValueError("saved video locator must match its SHA-256")
    images = validate_comfy_images(request.decoded_images, field="decoded_images")
    last_frame = images[-1:].clone().contiguous()
    if prepared_request.living_canon:
        return _commit_living_minimax(
            request,
            store=store,
            runtime=runtime,
            images=images,
            last_frame=last_frame,
        )
    normalized = normalize_story_frame(last_frame)
    encoded = runtime.materialize_frame(normalized).validate(contract)
    last_png = _png_from_rgb(normalized)
    leaf = _story_leaf(request, encoded, last_png, bridge, contract, runtime)
    parent = request.prepared.parent
    if parent is not None and not isinstance(parent, LoadedStoryRevision):
        raise ValueError("Trained runtime cannot append to a native archive")
    blocks = (
        empty_story_blocks(
            contract,
            dense_steps=encoded.latent.shape[2] * encoded.latent.shape[3] * encoded.latent.shape[4],
            operator_size=bridge.fusion.operator_size,
        )
        if parent is None
        else parent.blocks
    )
    blocks = append_story_leaf(
        blocks,
        shot_index=request.prepared.parent_shot_count,
        leaf=leaf,
    )
    root = compose_story_blocks(blocks, bridge.fusion)
    device = bridge.fusion.input_norm.weight.device
    guide_bundle: StoryGuideBundle | None = None
    if identity.backend is StoryMemoryBackend.MINIMAX_H3:
        current_frame_sha256 = _sha256(last_png)
        exception_pngs = tuple(
            (
                last_png
                if item.provenance.raw.content_sha256 == current_frame_sha256
                else store.load_asset(item.provenance.raw.content_sha256)
            )
            for item in root.exceptions
        )
        core_png: bytes | None = None
        if root.occupied_shots > len(root.exceptions):
            excluded = frozenset(root.excluded_leaf_keys)
            nonexception_leaves = tuple(
                leaf
                for block in blocks
                for leaf in block.leaves
                if leaf is not None and leaf.key not in excluded
            )
            if not nonexception_leaves:
                raise ValueError("MiniMax H3 core has no nonexception anchor leaf")
            anchor_leaf = max(nonexception_leaves, key=lambda value: value.key.source_rank)
            if len(anchor_leaf.candidates) != 1:
                raise ValueError("MiniMax H3 core anchor must bind one frame candidate")
            anchor_sha256 = anchor_leaf.candidates[0].provenance.raw.content_sha256
            anchor_latent = (
                encoded.latent
                if anchor_sha256 == current_frame_sha256
                else runtime.materialize_frame(_rgb_from_png(store.load_asset(anchor_sha256)))
                .validate(contract)
                .latent
            )
            context_latent = bridge.materialize_operator(
                root.core_operator.to(device=device),
                anchor=anchor_latent.to(device=device),
            )
            core_png = _png_from_rgb(runtime.decode(context_latent))
        guide_bundle = StoryGuideBundle(
            core_png,
            exception_pngs,
            tuple(item.fingerprint() for item in root.exceptions),
        ).validate()
        context_png = last_png if core_png is None else core_png
    else:
        context_latent = bridge.materialize_operator(
            root.core_operator.to(device=device), anchor=encoded.latent.to(device=device)
        )
        context_rgb = runtime.decode(context_latent)
        context_png = _png_from_rgb(context_rgb)
    metadata: dict[str, object] = {
        "active_references": list(request.prepared.active_reference_names),
        "intent": request.prepared.request.intent.value,
        "prompt_sha256": _sha256(request.prepared.request.prompt.strip().encode()),
        "seed": request.prepared.request.variation,
        "shot_length_seconds": request.prepared.request.shot_length_seconds,
        "video_sha256": request.saved_video_sha256,
    }
    if identity.backend is StoryMemoryBackend.MINIMAX_H3:
        metadata.update(
            {
                "memory_backend": identity.backend.value,
                "memory_contract_sha256": contract.fingerprint(),
                "sampler": prepared_request.sampler.value,
            }
        )
    state = store.publish(
        StoryCommitPayload(
            request.prepared.request.project_id,
            request.prepared.request.branch_id,
            None if parent is None else parent.state,
            request.prepared.library,
            blocks,
            _digest(request.prepared.request.checkpoint_sha256, "checkpoint_sha256"),
            request.prepared.request.model_configuration_sha256,
            last_png,
            context_png,
            metadata,
            guide_bundle,
        )
    )
    loaded = store.load(
        state, contract, _digest(request.prepared.request.checkpoint_sha256, "checkpoint_sha256")
    )
    return StoryCommitResult(state, last_frame, loaded)


def _complete_checkpoint(path: Path, device: torch.device) -> TrainingModules:
    encoded = path.read_bytes()
    if _sha256(encoded) != FROZEN_TRAINABLE_CHECKPOINT_SHA256:
        raise ValueError("frozen trainable checkpoint content SHA-256 mismatch")
    try:
        raw = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("frozen trainable checkpoint could not be loaded safely") from error
    if not isinstance(raw, dict) or set(raw) != {
        "format",
        "foundation_sha256",
        "module_state_sha256",
        "modules",
        "optimizer_steps",
        "protocol_sha256",
        "training_trace",
        "training_trace_sha256",
    }:
        raise ValueError("frozen trainable checkpoint envelope is malformed")
    states = raw["modules"]
    if not isinstance(states, dict) or set(states) != {"bridge", "gated", "resampler", "scorer"}:
        raise ValueError("frozen trainable checkpoint module roster changed")
    modules = build_training_modules().validate()
    for name, module in modules.named().items():
        state = states.get(name)
        if not isinstance(state, dict):
            raise ValueError(f"frozen trainable checkpoint module {name} is malformed")
        module.load_state_dict(state, strict=True)
        module.to(device).eval().requires_grad_(False)
    rows: list[list[str]] = []
    for module_name, module in sorted(modules.named().items()):
        for tensor_name, tensor in sorted(module.state_dict().items()):
            rows.append([module_name, tensor_name, tensor_sha256(tensor)])
    observed = hashlib.sha256(
        b"duet-x-ltx-probe-modules-v1\0" + canonical_json(rows) + b"\n"
    ).hexdigest()
    if observed != raw["module_state_sha256"]:
        raise ValueError("frozen trainable checkpoint tensor digest changed")
    return modules


class StoryLTXRuntime:
    """Own the official LTX conditioner/decoder and all trained Duet-X modules."""

    def __init__(
        self,
        modules: TrainingModules,
        materializer: PinnedLTXRGBMaterializer,
        decoder: OfficialLatentDecoder,
        *,
        vae_sha256: str,
        identity: StoryMemoryIdentity,
    ) -> None:
        self.modules = modules.validate()
        if not isinstance(self.modules.bridge, LTXLatentHistoryBridge):
            raise ValueError("story runtime bridge architecture changed")
        self.materializer = materializer
        self.decoder = decoder
        self._identity = identity.validate()
        if self._identity.backend is not StoryMemoryBackend.LTX:
            raise ValueError("StoryLTXRuntime requires the LTX memory backend")
        if self._identity.contract.latent_channels != self.modules.bridge.fusion.channels:
            raise ValueError("LTX runtime contract channels changed")
        self.vae_sha256 = _digest(vae_sha256, "vae_sha256")
        self.adapter_sha256 = hashlib.sha256(
            canonical_json(
                [
                    [name, tensor_sha256(tensor)]
                    for name, tensor in sorted(self.modules.bridge.state_dict().items())
                ]
            )
        ).hexdigest()
        self.scorer_sha256 = hashlib.sha256(
            canonical_json(
                [
                    [name, tensor_sha256(tensor)]
                    for name, tensor in sorted(self.modules.scorer.state_dict().items())
                ]
            )
        ).hexdigest()

    @property
    def bridge(self) -> LTXLatentHistoryBridge:
        return cast(LTXLatentHistoryBridge, self.modules.bridge)

    @property
    def identity(self) -> StoryMemoryIdentity:
        return self._identity

    @classmethod
    def load_pinned(
        cls,
        checkpoint: Path,
        *,
        source_commit: str,
        source_archive_sha256: str,
        contract: DuetXContract,
        model_configuration_sha256: str,
    ) -> StoryLTXRuntime:
        identity = FrozenQualityRuntimeIdentity.default(
            source_commit=source_commit,
            source_archive_sha256=source_archive_sha256,
        ).validate()
        device = torch.device(identity.device)
        modules = _complete_checkpoint(checkpoint, device)
        factory = PinnedOfficialLTXFactory(
            Path(identity.ltx_checkout_root),
            Path(identity.ltx_pipelines_source_root),
            Path(identity.ltx_core_source_root),
            Path(identity.dependency_source_root),
        )
        api = factory.load()
        materializer = api.build_materializer(identity, device=device)
        decoder = OfficialLatentDecoder(Path(identity.teacher_checkpoint_path), device)
        return cls(
            modules,
            materializer,
            decoder,
            vae_sha256=identity.teacher_checkpoint_sha256,
            identity=StoryMemoryIdentity(
                StoryMemoryBackend.LTX,
                contract,
                FROZEN_TRAINABLE_CHECKPOINT_SHA256,
                model_configuration_sha256,
            ).validate(),
        )

    def materialize_frame(self, frame: np.ndarray[Any, Any]) -> EncodedStoryFrame:
        result = self.materializer.materialize_frame(frame)
        return EncodedStoryFrame(
            result.latent,
            result.receipt.preprocessed_tensor_sha256,
            self.vae_sha256,
            self.adapter_sha256,
            self.scorer_sha256,
        ).validate(self.identity.contract)

    @torch.inference_mode()
    def score(self, latent: torch.Tensor, event_type: int, timestamp: float) -> float:
        device = next(self.modules.scorer.parameters()).device
        score = self.modules.scorer(
            latent.to(device=device),
            event_types=torch.tensor([event_type], device=device, dtype=torch.int64),
            timestamps=torch.tensor([timestamp], device=device, dtype=torch.float32),
        )
        if not isinstance(score, torch.Tensor) or tuple(score.shape) != (1,):
            raise ValueError("trained story scorer returned an unexpected boundary")
        return float(score.item())

    def decode(self, latent: torch.Tensor) -> np.ndarray[Any, Any]:
        value = self.decoder(latent)
        if value.dtype != np.uint8 or tuple(value.shape) != _FRAME_SHAPE:
            raise ValueError("official LTX decoder returned an unexpected RGB boundary")
        return np.ascontiguousarray(value)

    def close(self) -> None:
        try:
            self.materializer.close()
        finally:
            self.decoder.close()
            for module in self.modules.named().values():
                module.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


__all__ = (
    "EncodedStoryFrame",
    "PreparedStoryGeneration",
    "StoryCommitRequest",
    "StoryCommitResult",
    "StoryGenerationRequest",
    "StoryLTXRuntime",
    "commit_story_generation",
    "normalize_story_frame",
    "prepare_story_generation",
    "validate_comfy_images",
)
