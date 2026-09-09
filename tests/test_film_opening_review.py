from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from comfy_story.film_opening_review import opening_candidates


def _fixture(
    root: Path, *, legacy: bool = False, version: int = 2
) -> tuple[Path, Path, dict[str, Any]]:
    run, inputs = root / "run", root / "input"
    stage = run / "production/shot-001/opening/attempt-01"
    stage.mkdir(parents=True)
    inputs.mkdir()
    (run / "plan.json").write_text(json.dumps({"shots": [{"shot_id": "leaf"}]}))
    files = {"fitted.png": b"fitted fixture", "raw.png": b"raw fixture"}
    for name, data in files.items():
        (inputs / name).write_bytes(data)
    request = json.dumps(
        {"format": f"comfy-film-opening-stage-v{1 if legacy else version}"}
    ).encode()
    history = b'{"status":"success"}'
    (stage / "request.json").write_bytes(request)
    (stage / "history.json").write_bytes(history)
    receipt = {
        "file": "fitted.png",
        "sha256": hashlib.sha256(files["fitted.png"]).hexdigest(),
        "request_sha256": hashlib.sha256(request).hexdigest(),
        "history_sha256": hashlib.sha256(history).hexdigest(),
    }
    if not legacy:
        receipt.update(raw_file="raw.png", raw_sha256=hashlib.sha256(files["raw.png"]).hexdigest())
    (stage / "result.json").write_text(json.dumps(receipt))
    (stage / "assessment.json").write_text(
        json.dumps(
            {
                "state": {
                    "status": "fail",
                    "conditions": {
                        "Leaf.color": {
                            "expected": "green",
                            "observed": "brown",
                            "evidence": "Brown surface.",
                        }
                    },
                },
                "visibility": {"status": "pass", "observations": {}},
            }
        )
    )
    return run, inputs, {"shot_id": "leaf"}


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("version", [2, 3])
def test_failed_opening_exposes_original_observation_without_approval(
    tmp_path: Path, legacy: bool, version: int
) -> None:
    run, inputs, production = _fixture(tmp_path, legacy=legacy, version=version)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    rows = opening_candidates(run, inputs, production)
    assert len(rows) == 1
    assert rows[0]["state"]["conditions"]["Leaf.color"]["observed"] == "brown"
    assert rows[0]["image"] == "fitted.png"
    assert rows[0]["raw_image"] == (None if legacy else "raw.png")
    assert rows[0]["fitted"] is not legacy
    assert rows[0]["approval_created"] is False
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert opening_candidates(run, inputs, {"shot_id": "unrelated"}) == []


@pytest.mark.parametrize(
    "target", ["fitted", "raw", "request", "history", "assessment", "outside", "symlink"]
)
def test_changed_or_missing_opening_evidence_does_not_expose_a_preview(
    tmp_path: Path, target: str
) -> None:
    run, inputs, production = _fixture(tmp_path)
    stage = run / "production/shot-001/opening/attempt-01"
    if target in {"fitted", "raw"}:
        (inputs / f"{target}.png").write_bytes(b"changed")
    elif target in {"request", "history"}:
        (stage / f"{target}.json").write_text("{}")
    elif target == "assessment":
        (stage / "assessment.json").unlink()
    else:
        outside = tmp_path / "outside.png"
        outside.write_bytes((inputs / "fitted.png").read_bytes())
        if target == "symlink":
            (inputs / "fitted.png").unlink()
            (inputs / "fitted.png").symlink_to(outside)
        else:
            receipt = json.loads((stage / "result.json").read_text())
            receipt["file"] = "../outside.png"
            (stage / "result.json").write_text(json.dumps(receipt))
    row = opening_candidates(run, inputs, production)[0]
    assert "unavailable" in row
    assert "image" not in row
    assert "state" not in row
    assert row["approval_created"] is False


def _starting_fixture(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    from comfy_story.story_contracts import canonical_story_json

    run, inputs = root / "run", root / "input"
    stage = run / "production/starting-state"
    stage.mkdir(parents=True)
    inputs.mkdir()
    image = inputs / "checked-crop.png"
    image.write_bytes(b"the assessed crop")
    request = canonical_story_json(
        {
            "format": "comfy-film-starting-state-v2",
            "content": [
                {"type": "text", "text": "CURRENT STARTING FRAME"},
                {"type": "image", "image": str(image)},
            ],
            "image_sha256s": {str(image): hashlib.sha256(image.read_bytes()).hexdigest()},
        }
    )
    response = b"original observation"
    report = {
        "format": "comfy-film-starting-state-v2",
        "status": "needs_review",
        "conditions": {
            "Parcel.state": {
                "expected": "sealed",
                "observed": None,
                "evidence": "Seal is obscured.",
            }
        },
        "request_sha256": hashlib.sha256(request).hexdigest(),
        "response_sha256": hashlib.sha256(response).hexdigest(),
    }
    encoded = canonical_story_json(report)
    for name, data in [
        ("request.json", request),
        ("response.txt", response),
        ("report.json", encoded),
    ]:
        (stage / name).write_bytes(data)
    return (
        run,
        inputs,
        {"starting_state": report, "starting_state_sha256": hashlib.sha256(encoded).hexdigest()},
    )


def test_uploaded_opening_review_preserves_uncertainty_and_exact_checked_image(
    tmp_path: Path,
) -> None:
    from comfy_story.film_opening_review import starting_state_review

    run, inputs, production = _starting_fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    row = starting_state_review(run, inputs, production)
    assert row is not None
    assert row["image"] == "checked-crop.png"
    assert row["state"]["conditions"]["Parcel.state"]["observed"] is None
    assert row["approval_created"] is False
    assert row["kind"] == "uploaded"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert starting_state_review(run, inputs, {}) is None
    assert starting_state_review(run, inputs, {"starting_state": {"status": "pass"}}) is None


@pytest.mark.parametrize(
    "target", ["report", "request", "response", "image", "missing", "symlink", "binding"]
)
def test_uploaded_opening_changed_evidence_remains_unavailable(tmp_path: Path, target: str) -> None:
    from comfy_story.film_opening_review import starting_state_review

    run, inputs, production = _starting_fixture(tmp_path)
    stage = run / "production/starting-state"
    if target in {"report", "request", "response"}:
        (stage / (target + (".txt" if target == "response" else ".json"))).write_text("{}")
    elif target == "missing":
        (stage / "report.json").unlink()
    elif target == "binding":
        production["starting_state_sha256"] = "f" * 64
    elif target == "symlink":
        outside = tmp_path / "outside.png"
        outside.write_bytes((inputs / "checked-crop.png").read_bytes())
        (inputs / "checked-crop.png").unlink()
        (inputs / "checked-crop.png").symlink_to(outside)
    else:
        (inputs / "checked-crop.png").write_bytes(b"different picture")
    row = starting_state_review(run, inputs, production)
    assert row is not None
    assert "unavailable" in row
    assert "image" not in row
    assert "state" not in row
    assert row["approval_created"] is False
