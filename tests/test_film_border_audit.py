from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from comfy_story.film_audit import ShotVisualAudit
from comfy_story.film_border_audit import apply_border_policy, black_borders
from comfy_story.film_io import film_plan_from_json
from comfy_story.film_plan import FilmPlan, FilmShot, compile_candidate_prompt
from comfy_story.story_contracts import canonical_story_json


def _image(path: Path, edge: str = "", *, color: str = "steelblue") -> Path:
    image = Image.new("RGB", (200, 100), color)
    regions = {
        "top": (0, 0, 199, 9),
        "bottom": (0, 90, 199, 99),
        "left": (0, 0, 19, 99),
        "right": (180, 0, 199, 99),
        "object": (40, 0, 90, 30),
    }
    if edge:
        ImageDraw.Draw(image).rectangle(regions[edge], fill="black")
    image.save(path)
    return path


def _audit(status: str = "machine_pass") -> ShotVisualAudit:
    return ShotVisualAudit("one", "a" * 64, "b" * 64, "c" * 64, (), (), status)


@pytest.mark.parametrize("edge", ["top", "bottom", "left", "right"])
def test_added_or_removed_borders_override_a_model_pass(tmp_path: Path, edge: str) -> None:
    opening = _image(tmp_path / "opening.png")
    changed = _image(tmp_path / "changed.png", edge)
    for frames in ([opening, changed], [changed, opening]):
        result = apply_border_policy(_audit(), frames, (0, 119))
        assert result.status == "fail"
        assert result.findings[-1].frame_indices == (0, 119)
        assert "Measured pixels only" in result.findings[-1].evidence


@pytest.mark.parametrize("edge", ["", "top", "bottom", "left", "right"])
def test_unchanged_borders_allow_different_scene_colors(tmp_path: Path, edge: str) -> None:
    opening = _image(tmp_path / "opening.png", edge)
    later = _image(tmp_path / "later.png", edge, color="coral")
    assert apply_border_policy(_audit(), [opening, later], (0, 119)).status == "machine_pass"


def test_black_subject_is_not_a_border_and_existing_failures_survive(tmp_path: Path) -> None:
    opening = _image(tmp_path / "opening.png")
    later = _image(tmp_path / "later.png", "object")
    assert black_borders(later) == (0, 0, 0, 0)
    for status in ("machine_pass", "needs_review", "fail"):
        assert apply_border_policy(_audit(status), [opening, later], (0, 119)).status == status


def test_nearly_black_frame_is_uncertain_not_an_invented_border(tmp_path: Path) -> None:
    opening = _image(tmp_path / "opening.png")
    dark = _image(tmp_path / "dark.png", color="black")
    assert black_borders(dark) is None
    assert apply_border_policy(_audit(), [opening, dark], (0, 119)).status == "needs_review"


def test_border_contract_is_opt_in_serialized_and_changes_generation_identity() -> None:
    shot = FilmShot("one", 5000, "Hold", "A blue object rests", ())
    plan = FilmPlan("p", "P", 5000, (), (shot,))
    assert "border_policy" not in canonical_story_json(plan).decode()
    edited = replace(plan, shots=(replace(shot, border_policy="Preserve opening borders"),))
    assert film_plan_from_json(canonical_story_json(edited).decode()) == edited
    assert edited.shots[0].digest != shot.digest
    prompt = compile_candidate_prompt(edited, 0)
    canvas = next(line for line in prompt.splitlines() if line.startswith("Canvas:"))
    assert "match the opening frame's picture area and margins exactly" in canvas
    assert not any(word in canvas.lower() for word in ("black", "letterbox", "pillarbox"))
    assert "Canvas:" not in compile_candidate_prompt(plan, 0)
    with pytest.raises(ValueError, match="border policy"):
        replace(plan, shots=(replace(shot, border_policy="invented"),)).validate()


def test_border_frames_must_align(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aligned"):
        apply_border_policy(_audit(), [_image(tmp_path / "one.png")], (0, 119))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Uses actual encoded output frames")
def test_export_audit_detects_encoded_bars_despite_model_pass(tmp_path: Path) -> None:
    from comfy_story.film_audit_cli import audit_export
    from comfy_story.film_export import file_digest
    from comfy_story.film_plan import RenderedTake

    run = tmp_path / "run"
    export = run / "export"
    export.mkdir(parents=True)
    (run / "run.json").write_text('{"export_attempt":1}')
    video = export / "film.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=24:d=0.5",
            "-vf",
            "drawbox=x=0:y=0:w=iw:h=12:color=black:t=fill:enable='gte(t,0.2)'",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    digest = file_digest(video)
    (run / "sources").mkdir()
    shutil.copyfile(video, run / "sources" / (digest + ".mp4"))
    shot = FilmShot("one", 500, "Hold", "Stay blue", (), border_policy="Preserve opening borders")
    plan = FilmPlan("audit", "A fixture", 500, (), (shot,))
    take = RenderedTake("one", shot.digest, "a" * 64, digest, None, 500)
    (export / "film-plan.json").write_bytes(canonical_story_json(plan))
    (export / "rendered-takes.json").write_text(json.dumps([asdict(take)]))
    (export / "export-receipt.json").write_text(
        json.dumps(
            {
                "film_sha256": digest,
                "plan_sha256": hashlib.sha256(canonical_story_json(plan)).hexdigest(),
                "rendered_take_sha256s": [take.digest],
            }
        )
    )
    inputs = tmp_path / "inputs.json"
    inputs.write_text('{"library":{"references":[]}}')

    def reviewer(content: list[dict[str, Any]]) -> str:
        assert "intended_action" in content[0]["text"]
        return json.dumps(
            {
                "checks": {
                    name: {"status": "pass", "evidence": "Injected pass", "frame_indices": [0]}
                    for name in (
                        "action",
                        "location",
                        "continuity",
                        "temporal_stability",
                        "unwanted_text",
                    )
                },
                "observed_effects": {},
            }
        )

    report = json.loads(
        audit_export(
            run,
            inputs,
            tmp_path,
            tmp_path,
            tmp_path / "audit",
            reviewer=reviewer,
            model_sha256="b" * 64,
            shot_ids=("one",),
        ).read_text()
    )
    assert report["shots"][0]["status"] == "fail"
    assert any(
        "black top/bottom/left/right" in row["evidence"]
        for row in report["shots"][0]["findings"]
        if row["status"] == "fail"
    )
    assert report["approval_created"] is False
