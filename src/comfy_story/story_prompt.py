"""Opt-in reference prompt composition; it does not invent or rewrite story events."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from comfy_story.story_product_contracts import CanonPresence

if TYPE_CHECKING:
    from comfy_story.story_service import PreparedStoryGeneration

PROMPT_FORMATS = ("Current", "Structured reference (experimental)", "H3 automatic v1")


def _mentions(text: str, labels: dict[str, str]) -> str:
    # Preserve authored dialogue and quoted visible text exactly, including literal @names.
    def substitute(match: re.Match[str]) -> str:
        return labels.get(match[1].casefold(), match[0])

    parts = re.split(r'(<d>.*?</d>|"(?:[^"\\]|\\.)*")', text, flags=re.DOTALL)
    for index in range(0, len(parts), 2):
        parts[index] = re.sub(
            r"@([A-Za-z][A-Za-z0-9_-]*)",
            substitute,
            parts[index],
        )
    return "".join(parts)


def structured_reference_prompt(prepared: PreparedStoryGeneration) -> str:
    """Separate reusable subjects, frame anchors, retained state and requested action.

    The six-section organization follows MiniMax's reference prompt guide. This formatter
    preserves the creator's direction and exact source order; it is not a screenplay model.
    Its quality advantage is experimental and must be measured against the current format.
    """
    references = {ref.name.casefold(): ref for ref in prepared.library.references}
    sources: dict[str, list[tuple[int, str]]] = {}
    anchors = []
    retention = []
    for picture, role in enumerate(prepared.visual_roles, 1):
        label = f"<Picture {picture}>"
        if role == "current":
            if prepared.request.composition == "Continue frame":
                anchors.append(f"{label} is the target's opening composition anchor.")
                retention.append(
                    f"{label}: fully_preserved - use the opening composition, "
                    "then enact the direction."
                )
            else:
                anchors.append(f"{label} supplies prior visual context for a new composition.")
                retention.append(
                    f"{label}: partially_preserved - retain context, not its old framing or action."
                )
        elif role in {"inclusive-core", "associative-history"}:
            anchors.append(f"{label} supplies historical visual context, not a requested shot.")
            retention.append(
                f"{label}: partially_preserved - retain identity; "
                "current direction determines events."
            )
        else:
            prefix, separator, entity = role.partition("-")
            if (
                not separator
                or prefix not in {"identity", "evidence", "reference"}
                or entity not in references
            ):
                raise ValueError("structured prompt requires a supported semantic guide roster")
            sources.setdefault(entity, []).append((picture, prefix))
    labels = {entity: f"<Subject {index}>" for index, entity in enumerate(sources, 1)}
    definitions = list(anchors)
    canon = (
        {entity.entity_id: entity for entity in prepared.product_state.canon.entities}
        if prepared.product_state is not None
        else {}
    )
    baseline = (
        {
            spec.entity_id
            for spec in prepared.recall_decision.packet_specs
            if spec.state_evidence_id is None
        }
        if prepared.recall_decision is not None
        else set()
    )
    for entity, pictures in sources.items():
        ref = references[entity]
        label = labels[entity]
        image_labels = ", ".join(f"<Picture {picture}>" for picture, _ in pictures)
        definitions.append(
            f"{label} is @{ref.name}, the {ref.role.value.lower()} referenced by {image_labels}. "
            f"Baseline appearance notes: {ref.note.strip()}"
        )
        retention.append(
            f"{label}: fully_preserved - preserve identity. Reference backgrounds, "
            "incidental objects "
            "and old actions do not establish the current event."
        )
        evidence = [
            picture for picture, role in pictures if role == "evidence" and entity not in baseline
        ]
        if evidence:
            retention.append(
                f"{label}: confirmed current evidence in "
                + ", ".join(f"<Picture {picture}>" for picture in evidence)
                + " takes precedence over baseline mutable state."
            )
        current = canon.get(entity)
        if (
            current is not None
            and current.confirmed_revision_sha256 is not None
            and current.state_note.strip()
        ):
            retention.append(f"{label}: approved state: {current.state_note.strip()}")
    for canon_entity in canon.values():
        if (
            canon_entity.entity_id not in labels
            and canon_entity.presence is CanonPresence.OFF_SCREEN
        ):
            retention.append(
                f"@{canon_entity.reference_name} remains off screen; do not depict them."
            )
    direction = _mentions(prepared.request.prompt.strip(), labels)
    if not direction.startswith("[Shot 1]"):
        direction = "[Shot 1] " + direction
    return "\n\n".join(
        (
            "subject_definitions:\n" + "\n".join(definitions),
            "summary:\n[reference generation] Generate the requested "
            f"{prepared.request.shot_length_seconds}-second video using the defined subjects "
            "and anchors.",
            "retention_analysis:\n" + "\n".join(retention),
            "detailed_description:\n" + direction,
            "overall_soundscape:\nFollow the authored sound direction. Do not invent speech.",
            "non_diegetic_music:\nOnly music explicitly requested in the direction; "
            "otherwise none.",
        )
    )
