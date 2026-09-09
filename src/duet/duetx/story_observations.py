"""Deterministic bounded frame selection for one generated story shot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from duet.duetx.story_product_contracts import ObservationKind, StoryObservationPacket


@dataclass(frozen=True, slots=True)
class ExtractedStoryFrame:
    kind: ObservationKind
    frame_index: int
    timestamp_start_ns: int
    timestamp_stop_ns: int
    rgb: np.ndarray[Any, Any]
    change_score_q: int


def timeline_start_ns(parent_packets: tuple[StoryObservationPacket, ...]) -> int:
    """Return the exact cumulative boundary for the next generated shot."""
    if not isinstance(parent_packets, tuple):
        raise ValueError("parent_packets must be a tuple")
    if not parent_packets:
        return 0
    value = parent_packets[-1].timeline_stop_ns
    if type(value) is not int or value < 0:
        raise ValueError("parent packet timeline is invalid")
    return value


def _normalized_frames(images: torch.Tensor) -> tuple[np.ndarray[Any, Any], ...]:
    # Kept local to avoid making the service's public image boundary depend on this module.
    from duet.duetx.story_service import normalize_story_frame, validate_comfy_images

    validated = validate_comfy_images(images)
    return tuple(
        normalize_story_frame(validated[index : index + 1]) for index in range(len(validated))
    )


def extract_story_observation_frames(
    images: torch.Tensor,
    *,
    timeline_start_ns: int,
    shot_duration_ns: int,
) -> tuple[ExtractedStoryFrame, ...]:
    """Select distinct opening, maximum-change, and closing frames."""
    if type(timeline_start_ns) is not int or timeline_start_ns < 0:
        raise ValueError("timeline_start_ns must be a nonnegative integer")
    if type(shot_duration_ns) is not int or shot_duration_ns <= 0:
        raise ValueError("shot_duration_ns must be a positive integer")
    frames = _normalized_frames(images)
    frame_count = len(frames)
    selected: dict[int, tuple[ObservationKind, int]] = {0: (ObservationKind.OPENING, 0)}
    if frame_count > 1:
        selected[frame_count - 1] = (ObservationKind.CLOSING, 0)
    if frame_count > 2:
        scores = tuple(
            int(
                np.abs(frames[index].astype(np.int16) - frames[index - 1].astype(np.int16)).sum(
                    dtype=np.int64
                )
            )
            for index in range(1, frame_count - 1)
        )
        maximum = max(scores)
        if maximum > 0:
            change_index = scores.index(maximum) + 1
            selected[change_index] = (ObservationKind.CHANGE, maximum)
    result: list[ExtractedStoryFrame] = []
    for frame_index in sorted(selected):
        kind, score = selected[frame_index]
        start = timeline_start_ns + shot_duration_ns * frame_index // frame_count
        stop = timeline_start_ns + shot_duration_ns * (frame_index + 1) // frame_count
        if stop <= start:
            stop = start + 1
        stop = min(stop, timeline_start_ns + shot_duration_ns)
        result.append(
            ExtractedStoryFrame(
                kind,
                frame_index,
                start,
                stop,
                frames[frame_index],
                score,
            )
        )
    return tuple(result)


__all__ = (
    "ExtractedStoryFrame",
    "extract_story_observation_frames",
    "timeline_start_ns",
)
