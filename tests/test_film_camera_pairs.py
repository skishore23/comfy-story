"""Camera comparisons retain per-pair uncertainty and failure evidence."""

import json

import pytest

from comfy_story.film_audit import parse_camera_pairs


def _pair(frame: int, framing: str) -> dict[str, object]:
    return {
        "frames": [0, frame],
        "response": json.dumps(
            {"observed_framing": {str(frame): {"framing": framing, "evidence": "Background."}}}
        ),
    }


def test_invalid_pair_does_not_erase_another_pairs_failure() -> None:
    pairs = [_pair(29, "changed"), {"frames": [0, 59], "response": "not JSON"}]
    observed = parse_camera_pairs(json.dumps(pairs), (0, 29, 59))
    assert observed == {"29": {"framing": "changed", "evidence": "Background."}}


def test_pair_cannot_supply_a_different_frames_pass() -> None:
    pair = _pair(59, "unchanged")
    pair["frames"] = [0, 29]
    assert parse_camera_pairs(json.dumps([pair]), (0, 29)) == {}


@pytest.mark.parametrize("framing", ["changed", "unchanged", "uncertain"])
def test_pair_labels_are_preserved(framing: str) -> None:
    observed = parse_camera_pairs(json.dumps([_pair(29, framing)]), (0, 29))
    assert observed["29"]["framing"] == framing


@pytest.mark.parametrize(
    "pairs",
    [
        [],
        [_pair(29, "unchanged"), _pair(29, "unchanged")],
        [{"frames": [1, 29], "response": "{}"}],
        [{"frames": [29, 0], "response": "{}"}],
        [{"frames": [0, 29], "response": {}}],
        [{"frames": [0, 29], "response": "{}", "extra": True}],
        {"observed_framing": {}},
    ],
)
def test_changed_or_missing_pair_bindings_are_rejected(pairs: object) -> None:
    with pytest.raises(ValueError, match="camera pair assessment"):
        parse_camera_pairs(json.dumps(pairs), (0, 29))


def test_malformed_or_oversized_envelope_is_rejected() -> None:
    for value in ("not JSON", "x" * (1024 * 1024 + 1)):
        with pytest.raises(ValueError, match="camera pair assessment"):
            parse_camera_pairs(value, (0, 29))
