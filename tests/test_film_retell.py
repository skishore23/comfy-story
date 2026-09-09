from __future__ import annotations

import json
from typing import Any

import pytest

from comfy_story.film_retell import parse_film_retell


def _retell() -> dict[str, Any]:
    return {
        "summary": "A lantern is carried through a tunnel.",
        "protagonist_goal": "Find the exit.",
        "decision": "Follow the light.",
        "outcome": "The carrier reaches daylight.",
        "contradictions": [],
        "coherence": "coherent",
    }


@pytest.mark.parametrize("coherence", ["coherent", "incoherent", "uncertain"])
def test_retelling_preserves_negative_and_uncertain_assessments(coherence: str) -> None:
    value = _retell()
    value["coherence"] = coherence
    value["contradictions"] = ["At 10s the lantern disappears."]
    parsed = parse_film_retell("```json\n" + json.dumps(value) + "\n```")
    assert parsed.coherence == coherence
    assert parsed.contradictions == ("At 10s the lantern disappears.",)
    assert parsed.outcome == value["outcome"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary", ""),
        ("protagonist_goal", "  "),
        ("decision", None),
        ("outcome", ["Success"]),
        ("summary", "x" * 2_001),
        ("contradictions", "none"),
        ("contradictions", [""]),
        ("contradictions", [{}]),
        ("contradictions", ["x"] * 33),
        ("coherence", True),
        ("coherence", "pass"),
        ("unrequested", "extra"),
    ],
)
def test_retelling_rejects_incomplete_or_invalid_evidence(field: str, value: Any) -> None:
    payload = _retell()
    payload[field] = value
    with pytest.raises(ValueError, match="film retelling"):
        parse_film_retell(json.dumps(payload))


@pytest.mark.parametrize("raw", ["[]", "null", "not json", " " * 16_385])
def test_retelling_rejects_invalid_envelope(raw: str) -> None:
    with pytest.raises(ValueError, match="film retelling"):
        parse_film_retell(raw)


def test_blind_retell_distinguishes_cuts_without_waiving_story_evidence() -> None:
    from comfy_story.film_retell import blind_film_retell_prompt

    prompt = blind_film_retell_prompt()
    assert "editorial shot boundaries" in prompt
    assert "offscreen establishes no change" in prompt
    assert "mark coherence uncertain" in prompt
    assert "do not excuse an incompatible visible state" in prompt
    assert "You have not been given the intended screenplay" in prompt
    assert "lantern" not in prompt.casefold()
    assert "elephant" not in prompt.casefold()
