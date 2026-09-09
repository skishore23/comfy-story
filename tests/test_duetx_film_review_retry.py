from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from duet.duetx.film_audit_cli import _review_with_format_retry


def test_malformed_review_retries_same_images_and_preserves_responses(tmp_path: Path) -> None:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Observe these frames."},
        {"type": "image", "image": "/frames/frame-01.png"},
    ]
    calls: list[list[dict[str, Any]]] = []
    responses = iter(['{"checks":', '{"checks":{"action":"fail"}}'])

    def review(value: list[dict[str, Any]]) -> str:
        calls.append(value)
        return next(responses)

    raw = _review_with_format_retry(review, content, json.loads, tmp_path)
    assert json.loads(raw) == {"checks": {"action": "fail"}}
    assert len(calls) == 2
    assert calls[0] == content
    assert calls[1][:-1] == content
    assert len(content) == 2
    assert (tmp_path / "model-response-attempt-01.txt").read_text() == '{"checks":'
    assert (tmp_path / "model-response-attempt-02.txt").read_text() == raw


@pytest.mark.parametrize("status", ["fail", "uncertain", "pass"])
def test_valid_assessment_is_not_retried_to_obtain_a_pass(tmp_path: Path, status: str) -> None:
    calls = 0

    def review(_: list[dict[str, Any]]) -> str:
        nonlocal calls
        calls += 1
        return json.dumps({"status": status})

    assert json.loads(_review_with_format_retry(review, [], json.loads, tmp_path)) == {
        "status": status
    }
    assert calls == 1
    assert not (tmp_path / "model-response-attempt-02.txt").exists()


def test_repeated_malformed_output_stays_invalid_after_two_attempts(tmp_path: Path) -> None:
    calls = 0

    def review(_: list[dict[str, Any]]) -> str:
        nonlocal calls
        calls += 1
        return "truncated response"

    raw = _review_with_format_retry(review, [], json.loads, tmp_path)
    assert calls == 2
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    assert len(list(tmp_path.glob("model-response-attempt-*.txt"))) == 2


def test_reviewer_runtime_error_does_not_trigger_format_retry(tmp_path: Path) -> None:
    calls = 0

    def review(_: list[dict[str, Any]]) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        _review_with_format_retry(review, [], json.loads, tmp_path)
    assert calls == 1
