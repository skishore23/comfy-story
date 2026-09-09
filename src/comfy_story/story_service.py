"""Native H3 shot preparation and creator-controlled story evidence."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import torch
from PIL import Image

from comfy_story.memory.settings import MemoryConfiguration
from comfy_story.samplers import (
    StorySampler,
)
from comfy_story.story_contracts import (
    ComfyStoryStateRef,
    ReferenceRole,
    ShotIntent,
    StoryLibrary,
    StoryReference,
)
from comfy_story.story_native_archive import (
    NativeArchiveRevision,
    NativeReferenceArchive,
)
from comfy_story.story_product_contracts import (
    CanonEntity,
    CanonPresence,
    GuideBinding,
    PendingCanonObservation,
    StoryCanon,
    StoryEvidenceRecord,
    StoryMemoryCommand,
    StoryMemoryPolicy,
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
    StoryProjectStore,
)
from comfy_story.tensors import (
    tensor_sha256,
)

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


def _comfy_from_png(value: bytes) -> torch.Tensor:
    try:
        with Image.open(io.BytesIO(value)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as error:
        raise ValueError("authenticated story context is not a decodable image") from error
    return torch.from_numpy(np.array(rgb, copy=True)).to(torch.float32).div(255).unsqueeze(0)


@dataclass(frozen=True, slots=True)
class StoryGenerationRequest:
    """Inputs for one native H3 shot and its persistent story state."""

    intent: ShotIntent
    project_id: str
    branch_id: str
    previous_story: ComfyStoryStateRef | None
    previous_frame: torch.Tensor | None
    world_frame: torch.Tensor | None
    library: StoryLibrary | None
    reference_images: Mapping[str, torch.Tensor]
    prompt: str
    shot_length_seconds: int
    variation: int
    model_configuration_sha256: str
    associative_memory: MemoryConfiguration | None = None
    reference_policy: str = "Automatic"
    sampler: StorySampler = StorySampler.NATIVE_RES_MULTISTEP
    memory_commands: tuple[StoryMemoryCommand, ...] = ()
    execution_sha256: str | None = None
    composition: str = "Continue frame"
    scene_entity_names: tuple[str, ...] | None = None
    render_profile: str = "Reference shot"
    shot_state_evidence: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedStoryGeneration:
    """Validated inputs and the exact ordered reference roster."""

    request: StoryGenerationRequest
    parent: NativeArchiveRevision | None
    library: StoryLibrary
    active_reference_names: tuple[str, ...]
    visual_roles: tuple[str, ...]
    visual_guides: tuple[torch.Tensor, ...]
    resolved_prompt: str
    product_state: StoryProductState | None = None
    recall_decision: StoryRecallDecision | None = None

    @property
    def parent_shot_count(self) -> int:
        return 0 if self.parent is None else self.parent.state.shot_count


def _reference_only_composition(request: StoryGenerationRequest) -> bool:
    return request.render_profile == "Reference shot" and request.composition == "New composition"


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


def _prepare_native_references(
    request: StoryGenerationRequest,
    *,
    store: StoryProjectStore,
    parent: NativeArchiveRevision | None,
    library: StoryLibrary,
    current: torch.Tensor | None,
) -> PreparedStoryGeneration:
    if parent is None:
        if request.memory_commands or request.shot_state_evidence:
            raise ValueError("Start Story cannot apply commands without a parent revision")
        product = _baseline_product_state(library)
        parent_revision = "0" * 64
    else:
        product = parent.product_state
        parent_revision = parent.state.revision_sha256
    applied = apply_story_memory_commands(
        product,
        library,
        request.memory_commands,
        parent_revision_sha256=parent_revision,
    )
    decision = select_story_recall(
        applied,
        library,
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
    reference_tensors = {
        name.casefold(): validate_comfy_images(image, field=f"@{name}")[:1]
        for name, image in request.reference_images.items()
    }
    shot_state_entities = dict(request.shot_state_evidence)
    for spec in decision.packet_specs:
        if len(spec.source_sha256s) == 2:
            # Preserve the packet's historical contract, but never show its
            # baseline/state contact sheet to a native video generator.
            # An explicitly selected native state replaces the old appearance
            # in render conditioning. Supplying both can ask the model to animate
            # the old-to-new transition again. The original identity stays in
            # the library and authenticated packet for independent verification.
            if spec.entity_id not in shot_state_entities:
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
    decision = replace(decision, guide_bindings=bindings).validate()
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
    return PreparedStoryGeneration(
        request=request,
        parent=parent,
        library=library,
        active_reference_names=tuple(names_by_id[item] for item in image_entity_ids),
        visual_roles=roles,
        visual_guides=guides,
        resolved_prompt=" ".join(clauses),
        product_state=applied.product_state,
        recall_decision=decision,
    )


def prepare_story_generation(
    request: StoryGenerationRequest,
    *,
    store: StoryProjectStore,
) -> PreparedStoryGeneration:
    """Validate a shot, authenticate its parent, and select native visual references."""
    if request.render_profile not in {"Reference shot", "Animate frame"}:
        raise ValueError("Render profile must be Reference shot or Animate frame")
    if request.render_profile == "Animate frame" and (
        request.composition != "Continue frame"
        or request.sampler is StorySampler.SPEED_EULER_2STAGE
    ):
        raise ValueError("Animate frame requires Continue frame and Native or Turbo")
    intent = _validate_intent(request)
    if not isinstance(request.sampler, StorySampler):
        raise ValueError("sampler must be a StorySampler")
    _digest(request.model_configuration_sha256, "model_configuration_sha256")
    if request.shot_length_seconds not in {5, 10, 15}:
        raise ValueError("shot length must be 5, 10, or 15 seconds")
    if type(request.variation) is not int or not 0 <= request.variation <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Variation must be an unsigned 64-bit integer")
    if request.reference_policy not in {"Automatic", "Prompt mentions only"}:
        raise ValueError("Reference policy is unsupported")
    if request.composition not in {"Continue frame", "New composition"}:
        raise ValueError("Composition is unsupported")
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ValueError("What happens next? must be nonempty")
    parent = (
        None
        if request.previous_story is None
        else NativeReferenceArchive(store).load(
            request.previous_story,
            model_configuration_sha256=request.model_configuration_sha256,
        )
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
        if not isinstance(names, tuple) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("scene_entity_names must be a tuple of nonempty names")
        folded = tuple(name.casefold() for name in names)
        if len(folded) != len(set(folded)):
            raise ValueError("Declared scene entities must be unique, ignoring case")
        if not set(folded) <= {reference.name.casefold() for reference in library.references}:
            raise ValueError("Declared scene entities must belong to the Story Library")
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
    from comfy_story.memory.service import prepare_memory_guides

    prepared = _prepare_native_references(
        request, store=store, parent=parent, library=library, current=current
    )
    return prepare_memory_guides(prepared, store)


@dataclass(frozen=True, slots=True)
class StoryCommitRequest:
    prepared: PreparedStoryGeneration
    decoded_images: torch.Tensor
    saved_video_sha256: str
    saved_video_locator: str


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
