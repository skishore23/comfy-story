"""Comfy Story public runtime API."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "StoryGenerationRequest": ("comfy_story.story_service", "StoryGenerationRequest"),
    "PreparedStoryGeneration": ("comfy_story.story_service", "PreparedStoryGeneration"),
    "prepare_story_generation": ("comfy_story.story_service", "prepare_story_generation"),
    "NativeReferenceArchive": ("comfy_story.story_native_archive", "NativeReferenceArchive"),
    "commit_native_story_generation": (
        "comfy_story.story_native_service",
        "commit_native_story_generation",
    ),
    "StorySampler": ("comfy_story.samplers", "StorySampler"),
    "ComfyStoryStateRef": ("comfy_story.story_contracts", "ComfyStoryStateRef"),
    "StoryLibrary": ("comfy_story.story_contracts", "StoryLibrary"),
    "StoryReference": ("comfy_story.story_contracts", "StoryReference"),
    "ShotIntent": ("comfy_story.story_contracts", "ShotIntent"),
}

__all__ = tuple(sorted(_EXPORTS))


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
