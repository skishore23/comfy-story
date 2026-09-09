"""Compile a planned film into one ordinary, unattended ComfyUI Story graph.

All shots use the public node and its actual Story State wire. No generated shot is accepted,
no canon approval is fabricated, and no per-shot repair policy is hidden in this compiler.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from comfy_story.film_plan import FilmPlan, compile_candidate_prompt
from comfy_story.h3_prompt import H3_PROMPT_FORMAT, film_h3_direction
from comfy_story.story_prompt import PROMPT_FORMATS
from comfy_story.story_recall import parse_shot_state_evidence


@dataclass(frozen=True, slots=True)
class FilmShotInput:
    """Customer inputs for one shot; files are names inside Comfy's input directory."""

    world: str | None
    variation: int
    audio_file: str | None = None
    intent: str = "New Scene"
    sampler: str = "Native res_multistep"
    ending_frame: str | None = None
    render_profile: str = "Reference shot"
    shot_state_evidence: str = "{}"
    opening_prompt: str = ""
    opening_attempts: int = 1
    opening_use_previous_scene: bool = False
    opening_mode: str = "Compose"

    prompt_format: str = "Current"


def _input_file(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value == "None"
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("film assets must be relative Comfy input filenames")


def compile_film_workflow(
    plan: FilmPlan,
    library: Mapping[str, object],
    inputs: tuple[FilmShotInput, ...],
    *,
    allow_pending_staging: bool = False,
    generated_audio: bool = False,
) -> dict[str, object]:
    """Compose repeated public Story-node fragments with deterministic graph identities.

    A fresh Start Story is followed by the creator's scene or continuation transitions.
    Reference-composed scene changes may use selected references without a world image.
    Next Shot and Continue This Shot carry the actual
    prior Last Frame as well as Story State. Authored audio uses the standard LoadAudio node.
    The graph can be queued once or opened directly by Comfy's API-workflow importer.
    """
    plan.validate()
    if type(generated_audio) is not bool:
        raise ValueError("generated_audio must be a boolean")
    if generated_audio and any(
        item.prompt_format != H3_PROMPT_FORMAT or item.audio_file for item in inputs
    ):
        raise ValueError(
            "Generated film audio requires H3 automatic prompts and no authored shot audio"
        )
    if len(inputs) != len(plan.shots):
        raise ValueError("film requires inputs for every planned shot")
    if not isinstance(library.get("project_name"), str) or not library["project_name"]:
        raise ValueError("film requires a named Story Library")
    references = library.get("references")
    if not isinstance(references, list):
        raise ValueError("film requires a Story Library reference list")
    names: set[str] = set()
    for row in references:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("film reference requires a name")
        name = row["name"]
        if name in names:
            raise ValueError("film references must have unique names")
        names.add(name)
        filename = row.get("file")
        if not isinstance(filename, str):
            raise ValueError("film reference requires a Comfy input filename")
        _input_file(filename)
    library_json = json.dumps(dict(library), sort_keys=True)
    graph: dict[str, object] = {}
    for index, (shot, settings) in enumerate(zip(plan.shots, inputs, strict=True)):
        if shot.duration_ms * 24 % 1000:
            raise ValueError("film shot durations must select exact 24 fps frames")
        if not set(shot.present) <= names:
            raise ValueError("film shot names are missing from the Story Library")
        if settings.intent not in {"New Scene", "Next Shot", "Continue This Shot"}:
            raise ValueError("film transition must be a supported public Story intent")
        if settings.sampler not in {
            "Native res_multistep",
            "SPEED Euler 2-stage",
            "Turbo 4-step",
            "Turbo 8-step",
            "Full HD 2-pass",
            "NVFP4 Exact",
            "NVFP4 Balanced",
            "NVFP4 Ultra Fast",
            "NVFP4 Turbo 4-step",
        }:
            raise ValueError("film sampler must be a supported public Story sampler")
        if settings.prompt_format not in PROMPT_FORMATS:
            raise ValueError("film prompt format is unsupported")
        if settings.render_profile not in {"Reference shot", "Animate frame"}:
            raise ValueError("film render profile must be Reference shot or Animate frame")
        if settings.render_profile == "Animate frame" and (
            shot.composition != "Continue frame"
            or settings.sampler
            not in {"Native res_multistep", "Turbo 4-step", "Turbo 8-step", "Full HD 2-pass"}
        ):
            raise ValueError("Animate frame requires Continue frame and Native or Turbo sampling")
        if (
            not isinstance(settings.opening_prompt, str)
            or len(settings.opening_prompt) > 8000
            or "\x00" in settings.opening_prompt
        ):
            raise ValueError("opening prompt must be bounded text")
        if type(settings.opening_attempts) is not int or not 1 <= settings.opening_attempts <= 4:
            raise ValueError("opening attempts must be an integer from 1 to 4")
        if type(settings.opening_use_previous_scene) is not bool:
            raise ValueError("opening_use_previous_scene must be a boolean")
        staging = bool(settings.opening_prompt.strip())
        if settings.opening_mode not in ("Compose", "Refine"):
            raise ValueError("opening mode must be Compose or Refine")
        if settings.opening_mode == "Refine" and (
            not staging or not (settings.world or settings.opening_use_previous_scene)
        ):
            raise ValueError("Refine opening mode requires staging and an existing scene image")
        if staging:
            if not allow_pending_staging:
                raise ValueError(
                    "opening staging requires verified film production before queueing video"
                )
            if settings.render_profile != "Animate frame" or settings.intent != "New Scene":
                raise ValueError("opening staging requires Animate frame and New Scene")
            if settings.shot_state_evidence != "{}":
                raise ValueError("opening staging selects its own authenticated state evidence")
            if settings.opening_use_previous_scene and (index == 0 or settings.world is not None):
                raise ValueError(
                    "previous opening scene requires an earlier shot and no uploaded world"
                )
            if not settings.world and not shot.present and not settings.opening_use_previous_scene:
                raise ValueError("opening staging requires a world or a present reference")
            if len(shot.present) + bool(settings.world or settings.opening_use_previous_scene) > 3:
                raise ValueError(
                    "opening staging supports at most three source images; split the shot"
                )
        if settings.world is None and not staging:
            reference_only = (
                settings.render_profile == "Reference shot"
                and shot.composition == "New composition"
                and bool(shot.reference_names if shot.reference_names is not None else shot.present)
            )
            if (index == 0 or settings.intent == "New Scene") and not reference_only:
                raise ValueError("a new story or scene requires an uploaded world")
        elif settings.world is not None:
            _input_file(settings.world)
        if type(settings.variation) is not int or not 0 <= settings.variation <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("film variation must fit the public node's seed range")
        seconds = next((value for value in (5, 10, 15) if shot.duration_ms <= value * 1000), None)
        if seconds is None:
            raise ValueError("split film shots longer than 15 seconds before generation")
        node_id = str((index + 1) * 10)
        node_inputs: dict[str, object] = {
            "Create": "Start Story" if index == 0 else settings.intent,
            "Story Library": library_json,
            "World / starting frame": settings.world or "None",
            "What happens next?": (
                film_h3_direction(plan, index, generated_audio=generated_audio)
                if settings.prompt_format == H3_PROMPT_FORMAT
                else compile_candidate_prompt(plan, index)
            ),
            "Shot length": f"{seconds} seconds",
            "Variation": settings.variation,
            "Story revision": "",
            "Reference policy": "Prompt mentions only",
            "Sampler": settings.sampler,
            "Memory actions": "[]",
            "Composition": shot.composition,
            "Output duration (ms)": shot.duration_ms,
            "Scene entities": json.dumps(list(shot.present)),
        }
        evidence = parse_shot_state_evidence(settings.shot_state_evidence)
        if evidence:
            if index == 0 or settings.render_profile != "Reference shot":
                raise ValueError(
                    "shot state evidence requires a preceding Story and Reference shot"
                )
            node_inputs["Shot state evidence"] = settings.shot_state_evidence
        if settings.render_profile != "Reference shot":
            node_inputs["Render profile"] = settings.render_profile

        if settings.prompt_format != "Current":
            node_inputs["Prompt format"] = settings.prompt_format
        if index:
            node_inputs["Previous Story"] = [str(index * 10), 2]
            if settings.intent != "New Scene":
                node_inputs["Previous Frame"] = [str(index * 10), 1]
        if settings.audio_file is not None:
            _input_file(settings.audio_file)
            audio_id = str((index + 1) * 10 + 1)
            graph[audio_id] = {
                "class_type": "LoadAudio",
                "inputs": {"audio": settings.audio_file},
            }
            node_inputs["Authored audio"] = [audio_id, 0]
        if settings.ending_frame is not None:
            _input_file(settings.ending_frame)
            ending_id = str((index + 1) * 10 + 2)
            graph[ending_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": settings.ending_frame},
            }
            node_inputs["Ending frame"] = [ending_id, 0]
        graph[node_id] = {
            "class_type": "ComfyStory",
            "inputs": node_inputs,
            "_meta": {"title": f"{index + 1:02} · {shot.purpose}"},
        }
    return graph
