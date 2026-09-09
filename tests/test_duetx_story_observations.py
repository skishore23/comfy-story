from __future__ import annotations

import torch

from duet.duetx.story_observations import (
    extract_story_observation_frames,
    timeline_start_ns,
)
from duet.duetx.story_product_contracts import ObservationKind


def test_extractor_selects_opening_change_and_closing_in_frame_order() -> None:
    images = torch.zeros(5, 8, 8, 3)
    images[2:] = 1.0

    result = extract_story_observation_frames(
        images,
        timeline_start_ns=5_000_000_000,
        shot_duration_ns=10_000_000_000,
    )

    assert tuple((item.kind, item.frame_index) for item in result) == (
        (ObservationKind.OPENING, 0),
        (ObservationKind.CHANGE, 2),
        (ObservationKind.CLOSING, 4),
    )
    assert result[0].timestamp_start_ns == 5_000_000_000
    assert result[-1].timestamp_stop_ns == 15_000_000_000
    assert result[1].change_score_q > 0
    assert all(item.rgb.shape == (384, 384, 3) for item in result)


def test_static_two_frame_clip_does_not_fabricate_change() -> None:
    result = extract_story_observation_frames(
        torch.zeros(2, 4, 4, 3),
        timeline_start_ns=0,
        shot_duration_ns=5_000_000_000,
    )

    assert tuple(item.kind for item in result) == (
        ObservationKind.OPENING,
        ObservationKind.CLOSING,
    )


def test_equal_change_scores_choose_the_earliest_candidate() -> None:
    images = torch.zeros(5, 4, 4, 3)
    images[1] = 1.0
    images[2] = 0.0
    images[3] = 1.0

    result = extract_story_observation_frames(
        images, timeline_start_ns=0, shot_duration_ns=5_000_000_000
    )

    assert tuple((item.kind, item.frame_index) for item in result) == (
        (ObservationKind.OPENING, 0),
        (ObservationKind.CHANGE, 1),
        (ObservationKind.CLOSING, 4),
    )


def test_timeline_start_uses_cumulative_mixed_shot_durations() -> None:
    class Packet:
        timeline_stop_ns = 20_000_000_000

    assert timeline_start_ns((Packet(),)) == 20_000_000_000  # type: ignore[arg-type]
    assert timeline_start_ns(()) == 0
