"""Versioned H3 prompting matched to the actual encoder roster and render task.

The legacy formatters remain unchanged. This compiler preserves authored action and
language; it does not invent story events or certify generated continuity.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from comfy_story.story_product_contracts import CanonPresence
from comfy_story.story_prompt import _mentions

if TYPE_CHECKING:
    from comfy_story.story_service import PreparedStoryGeneration

H3_PROMPT_FORMAT = "H3 automatic v1"
H3_PROMPT_PROTOCOL = "duet-h3-mode-aware-v1"


def compile_h3_prompt(
    prepared: PreparedStoryGeneration,
    *,
    duration_ms: float,
    ending_frame: bool = False,
    motion_reference: bool = False,
    authored_audio: bool = False,
) -> str:
    """Compile a native H3 prompt without changing seeds, facts or approval state."""
    if (
        isinstance(duration_ms, bool)
        or not math.isfinite(duration_ms)
        or duration_ms <= 0
        or abs(duration_ms * 24 / 1000 - round(duration_ms * 24 / 1000)) > 1e-6
    ):
        raise ValueError("H3 prompt duration must select exact 24 fps frames")
    if len(prepared.visual_roles) != len(prepared.visual_guides):
        raise ValueError("H3 prompt roster does not match supplied images")
    action = prepared.request.prompt.strip()
    if not action:
        raise ValueError("H3 prompt needs authored action")
    # Film compilation explicitly separates speech/music into authored export tracks.
    silent = authored_audio or action.endswith("The generated clip is completely silent.")
    sound = (
        "N/A"
        if silent
        else (
            "Only the ambience, physical sounds and vocal events explicitly described in the "
            "shot are audible. Unspecified speakers and narration are absent."
        )
    )
    music = (
        "N/A"
        if silent or action.endswith("physical sound effects; background music is absent.")
        else (
            "Only the audience-only music explicitly described in the shot is audible; "
            "otherwise there is no background score."
        )
    )
    duration = f"{duration_ms / 1000:.2f}"
    # Preserve scene exclusions even when no image is selected for that entity.
    scene_names = prepared.request.scene_entity_names
    scene_entities = {
        name.casefold()
        for name in (scene_names if scene_names is not None else prepared.active_reference_names)
    }
    offscreen = [
        f"{entity.reference_name} remains off screen in this shot."
        for entity in (prepared.product_state.canon.entities if prepared.product_state else ())
        if entity.presence is CanonPresence.OFF_SCREEN and entity.entity_id not in scene_entities
    ]
    if prepared.request.render_profile == "Animate frame":
        if len(prepared.visual_guides) != 1 or motion_reference:
            raise ValueError("H3 frame prompting requires exactly one opening image")
        if ending_frame:
            alignment = (
                "How the reference pictures align with the target video — Picture 1 "
                "(from Shot 1) aligns with the 0.00-second mark of the target video; "
                f"Picture 2 (from Shot 1) aligns with the {duration}-second mark "
                "of the target video."
            )
            path = (
                "Begin in the appearance, pose, composition and setting of Picture 1. "
                "Develop the described action continuously, with visible intermediate changes. "
                "Progressively approach the pose, object arrangement and composition of "
                "Picture 2, reaching it at the end of the shot."
            )
        else:
            alignment = (
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced."
            )
            path = (
                "Begin from the subjects, appearance, composition and setting in <Picture 1>. "
                "Develop the described action from that starting state through its visible "
                "result; appearance changes follow the authored action."
            )
        return "\n\n".join(
            (
                alignment,
                f"integrated_multimodal_description: [Shot 1] {path}\n"
                + "\n".join(offscreen)
                + "\n"
                + action,
                f"overall_soundscape: {sound}",
                f"non_diegetic_music: {music}",
            )
        )
    if prepared.request.render_profile != "Reference shot":
        raise ValueError("Unsupported H3 prompt render profile")

    references = {ref.name.casefold(): ref for ref in prepared.library.references}
    sources: dict[str, list[tuple[int, str]]] = {}
    definitions: list[str] = []
    retention: list[str] = []
    anchors: list[str] = []
    task = ["reference generation"]
    for index, role in enumerate(prepared.visual_roles, 1):
        picture = f"<Picture {index}>"
        if role == "current":
            if prepared.request.composition == "Continue frame":
                definitions.append(f"{picture} is the opening frame and composition of [Shot 1].")
                retention.append(f"{picture} ([Shot 1] opening): fully_preserved - opening layout.")
                anchors.append(
                    f"Begin from {picture} and develop the action from its visible state."
                )
                task.append("keyframe completion")
            else:
                definitions.append(f"{picture} supplies earlier scene context.")
                retention.append(f"{picture}: weak_reference - scene context, not the old action.")
        elif role == "inclusive-core":
            definitions.append(f"{picture} supplies historical visual context.")
            retention.append(f"{picture}: weak_reference - recognizable details, not old events.")
        else:
            kind, _, entity = role.partition("-")
            if kind not in {"identity", "evidence", "reference"} or entity not in references:
                raise ValueError("H3 prompting requires a supported semantic guide roster")
            sources.setdefault(entity, []).append((index, kind))
    labels = {entity: f"<Subject {index}>" for index, entity in enumerate(sources, 1)}
    selected = dict(prepared.request.shot_state_evidence)
    baseline = (
        {
            spec.entity_id
            for spec in prepared.recall_decision.packet_specs
            if spec.state_evidence_id is None
        }
        if prepared.recall_decision is not None
        else set(sources)
    )
    canon = (
        {entity.entity_id: entity for entity in prepared.product_state.canon.entities}
        if prepared.product_state is not None
        else {}
    )
    for entity, images in sources.items():
        ref = references[entity]
        label = labels[entity]
        definitions.append(
            f"{label} is @{ref.name}, the {ref.role.value.lower()} from "
            + ", ".join(f"<Picture {i}>" for i, _ in images)
            + "."
        )
        if ref.note.strip():
            definitions.append(f"Baseline appearance of {label}: {ref.note.strip()}")
        if ref.role.value == "Location":
            retention.append(
                f"{label} ([Shot 1]): partially_preserved - preserve the scene architecture, "
                "spatial layout, time of day, light directions, light colors, exposure and "
                "white balance described by this location reference and its notes. "
                "Camera position and subject action may change; lighting changes only when "
                "explicitly requested in this shot. Do not copy people or old actions from "
                "the location image."
            )
        else:
            retention.append(
                f"{label} ([Shot 1]): partially_preserved - retain recognizable identity or "
                "the reference's stated visual role; pose, action, location and mutable "
                "appearance follow this shot's direction, including explicit transformations."
            )
        evidence = [i for i, kind in images if kind == "evidence" and entity not in baseline]
        if evidence:
            retention.append(
                f"{label}: use the selected appearance in "
                + ", ".join(f"<Picture {i}>" for i in evidence)
                + " for this shot, without replaying its old action."
            )
        entity_canon = canon.get(entity)
        if (
            entity_canon is not None
            and entity_canon.confirmed_revision_sha256 is not None
            and entity_canon.state_note.strip()
            and entity not in selected
        ):
            retention.append(f"{label}: retained appearance: {entity_canon.state_note.strip()}")
    if motion_reference:
        definitions.append("<Video 1> supplies motion and camera rhythm, not new story events.")
        retention.append("<Video 1>: weak_reference - movement and rhythm only.")
    if ending_frame:
        picture = f"<Picture {len(prepared.visual_guides) + 1}>"
        definitions.append(f"{picture} is the final frame of [Shot 1] at {duration} seconds.")
        retention.append(
            f"{picture} ([Shot 1] ending): fully_preserved - final visual arrangement."
        )
        anchors.append(
            f"Develop the authored action through visible intermediate states and gradually "
            f"converge to {picture} at the end, rather than displaying that endpoint early."
        )
        if "keyframe completion" not in task:
            task.append("keyframe completion")
    retention.extend(offscreen)
    if sources:
        retention.append(
            "Reference notes describe reusable appearance, not events to perform. "
            "Only this shot's authored action determines what happens and when."
        )
    direction = _mentions(action, labels)
    return "\n\n".join(
        (
            "subject_definitions:\n" + "\n".join(definitions),
            f"summary:\n[{' + '.join(task)}] A {duration}-second shot using the defined "
            "references for their stated roles and the action below.",
            "retention_analysis:\n" + "\n".join(retention),
            "detailed_description:\n[Shot 1] " + " ".join(anchors) + "\n" + direction,
            f"overall_soundscape: {sound}",
            f"non_diegetic_music: {music}",
        )
    )


def film_h3_direction(plan: object, shot_index: int, *, generated_audio: bool = False) -> str:
    """Turn typed film requirements into visible instructions, without parsing prose."""
    from comfy_story.film_plan import FilmPlan, planned_shot_conditions

    if not isinstance(plan, FilmPlan):
        raise TypeError("H3 film direction requires a FilmPlan")
    shot = plan.shots[shot_index]
    meanings = {(item.key, item.value): item.definition for item in plan.state_definitions}

    def visible(key: str, value: str) -> str:
        return meanings.get(
            (key, value), f"{key.replace('.', ' ').replace('_', ' ')} is {value.replace('_', ' ')}"
        )

    changes = {item.key: item.value for item in shot.effects}
    opening = planned_shot_conditions(plan, shot_index)
    lines = []
    for key, value in opening.items():
        prefix = "At the opening" if key in changes else "Throughout the shot"
        lines.append(f"{prefix}, {visible(key, value)}.")
    lines.append(shot.action.strip())
    for key, value in changes.items():
        lines.append(f"By the end, {visible(key, value)}.")
    if shot.camera_policy == "Locked frame":
        lines.append("The camera holds a static shot with fixed framing, scale and position.")
    elif shot.camera_policy == "Single continuous shot":
        lines.append("One continuous take follows the described camera movement and action.")
    for name in shot.visible_throughout:
        lines.append(f"{name} remains visible throughout the shot.")
    for name in shot.fully_visible_throughout:
        lines.append(f"{name} remains fully in frame and unobscured throughout the shot.")
    for count in shot.ending_counts:
        lines.append(f"The final frame contains exactly {count.count} {count.category}.")
    if shot.border_policy == "Preserve opening borders":
        lines.append("The picture area and margins match the opening frame throughout.")
    for name in shot.absent:
        lines.append(f"{name} remains off screen throughout this shot.")
    text = "\n".join(lines)
    # Prose may mention entities without adding their images to the conditioning roster.
    if shot.reference_names is not None:
        names_in_text = re.findall(r"@([A-Za-z][A-Za-z0-9_-]*)", text)
        text = _mentions(text, {name.casefold(): name for name in names_in_text})
    names = shot.reference_names if shot.reference_names is not None else shot.present
    if names:
        text += "\nThe visible reference subjects are " + ", ".join(f"@{n}" for n in names) + "."
    languages = {"en": "English", "ja": "Japanese", "hi": "Hindi", "zh": "Chinese"}
    allowed = ", ".join(languages.get(code, code) for code in plan.languages)
    text += (
        f"\nFilm speech languages: {allowed} only. The visual style and location do not "
        "change the spoken language. Background conversations, narration and invented "
        "spoken words are absent."
    )
    if generated_audio:
        if not shot.dialogue:
            text += (
                "\nThis shot contains no dialogue: no spoken words from any foreground or "
                "background character. Only explicitly described nonverbal vocalizations "
                "and physical sound effects may occur."
            )
        else:
            text += (
                "\nSpeak each scripted line exactly once, only inside its specified window. "
                "Between dialogue windows there are no spoken words. Do not repeat, "
                "translate or add lines."
            )
        speakers = sorted(
            {cue.speaker for item in plan.shots for cue in item.dialogue if cue.speaker}
        )
        for cue in shot.dialogue:
            language = languages.get(cue.language, cue.language)
            voice = f"(S{speakers.index(cue.speaker) + 1}) " if cue.speaker else ""
            text += (
                f"\nAt {cue.start_ms / 1000:.2f}-{cue.end_ms / 1000:.2f} seconds, "
                f"{voice}{cue.speaker} says <d>[{language}]{cue.text}</d>."
            )
        return (
            text
            + "\nUse the described dialogue and physical sound effects; background music is absent."
        )
    return text + "\nThe generated clip is completely silent."
