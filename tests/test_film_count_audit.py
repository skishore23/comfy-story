from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from comfy_story.film_audit import ShotVisualAudit
from comfy_story.film_count_audit import (
    CountObservation,
    apply_ending_counts,
    category_count_prompt,
    parse_category_counts,
)
from comfy_story.film_io import film_plan_from_json
from comfy_story.film_plan import FilmCount, FilmPlan, FilmShot, compile_candidate_prompt
from comfy_story.story_contracts import canonical_story_json


def test_optional_count_contract_roundtrips_and_preserves_historical_identity() -> None:
    shot = FilmShot("one", 5000, "Wait", "Mugs rest on a table", ())
    legacy = asdict(shot)
    for key in (
        "ending_counts",
        "fully_visible_throughout",
        "camera_policy",
        "direction_version",
        "border_policy",
        "visible_throughout",
        "reference_names",
    ):
        legacy.pop(key)
    assert shot.digest == hashlib.sha256(canonical_story_json(legacy)).hexdigest()
    plan = FilmPlan("mugs", "Mugs", 5000, (), (shot,))
    legacy_plan = asdict(plan)
    legacy_plan.pop("state_definitions")
    legacy_plan.pop("narrative")  # Historical plans predate the optional intent contract.
    for row in legacy_plan["shots"]:
        row.pop("ending_counts")
        row.pop("camera_policy")
        row.pop("direction_version")
        row.pop("border_policy")
        row.pop("fully_visible_throughout")
    assert canonical_story_json(plan) == canonical_story_json(legacy_plan)
    # Frozen from the preceding 8f55bb8 implementation, before state definitions existed.
    assert hashlib.sha256(canonical_story_json(plan)).hexdigest() == (
        "3b3cd0a1f4726a3308c2f8fa20eec67b2aa0f518ff8aab1bbe2a642b50e289b8"
    )
    edited = replace(plan, shots=(replace(shot, ending_counts=(FilmCount("red mugs", 2),)),))
    assert film_plan_from_json(canonical_story_json(edited).decode()) == edited
    assert edited.shots[0].digest != shot.digest
    assert "red mugs = 2" in compile_candidate_prompt(edited, 0)
    assert "Depict these quantities in the final frame" in compile_candidate_prompt(edited, 0)
    assert "Count separate physical instances" not in compile_candidate_prompt(edited, 0)
    assert "Visible counts" not in compile_candidate_prompt(plan, 0)
    assert '"red mugs"' in category_count_prompt(("red mugs",))
    assert "= 2" not in category_count_prompt(("red mugs",))


@pytest.mark.parametrize("count", [True, -1, 65, 1.5, "2", None])
def test_plan_rejects_invalid_counts(count: object) -> None:
    payload = asdict(FilmPlan("p", "P", 5000, (), (FilmShot("s", 5000, "P", "A", ()),)))
    payload["shots"][0]["ending_counts"] = [{"category": "mugs", "count": count}]
    with pytest.raises(ValueError, match="ending counts"):
        film_plan_from_json(json.dumps(payload))


@pytest.mark.parametrize("categories", [("mugs", "MUGS"), (" ",), ("x" * 121,), ("a\nb",)])
def test_plan_rejects_invalid_count_categories(categories: tuple[str, ...]) -> None:
    shot = FilmShot(
        "s", 5000, "P", "A", (), ending_counts=tuple(FilmCount(x, 1) for x in categories)
    )
    with pytest.raises(ValueError, match="ending counts"):
        FilmPlan("p", "P", 5000, (), (shot,)).validate()


def _response(count: object, instances: object) -> str:
    return json.dumps(
        {
            "observed_counts": {
                "mugs": {"count": count, "instances": instances, "uncertainty": "Visible"}
            }
        }
    )


@pytest.mark.parametrize(
    ("count", "instances"),
    [(True, ["one"]), (2, ["one"]), (-1, []), (65, []), (1, [""]), (1, "one")],
)
def test_count_parser_rejects_unsupported_or_contradictory_evidence(
    count: object, instances: object
) -> None:
    with pytest.raises(ValueError, match="bounded instance evidence"):
        parse_category_counts(_response(count, instances), ("mugs",))


def test_count_parser_requires_exact_categories_and_retains_ambiguity() -> None:
    with pytest.raises(ValueError, match="every requested category"):
        parse_category_counts(_response(1, ["left mug"]), ("plates",))
    parsed = parse_category_counts(_response(None, ["possible left mug"]), ("mugs",))
    assert parsed["mugs"].count is None
    assert parsed["mugs"].instances == ("possible left mug",)


@pytest.mark.parametrize(
    ("count", "status"), [(2, "machine_pass"), (3, "fail"), (None, "needs_review")]
)
def test_ending_count_overrides_an_incorrect_full_audit_pass(
    count: int | None, status: str
) -> None:
    audit = ShotVisualAudit("s", "a" * 64, "b" * 64, "c" * 64, (), (), "machine_pass")
    observed = {"mugs": CountObservation(count, ("observed objects",), "")}
    result = apply_ending_counts(audit, observed, (FilmCount("mugs", 2),), 239)
    assert result.status == status
    assert result.findings[-1].frame_indices == (239,)
    assert result.video_sha256 == audit.video_sha256
    assert not hasattr(result, "reviewer")
    for prior in ("fail", "needs_review"):
        result = apply_ending_counts(
            replace(audit, status=prior),
            {"mugs": CountObservation(2, ("left", "right"), "")},
            (FilmCount("mugs", 2),),
            239,
        )
        assert result.status == prior
    assert apply_ending_counts(audit, {}, (FilmCount("mugs", 2),), 239).status == "needs_review"


@pytest.mark.parametrize(
    ("count", "status"),
    [(2, "machine_pass"), (3, "fail"), (None, "needs_review"), ("malformed", "needs_review")],
)
def test_export_counts_only_the_final_frame_and_binds_its_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int | str | None, status: str
) -> None:
    from comfy_story import film_audit_cli
    from comfy_story.film_plan import RenderedTake

    shot = FilmShot("one", 5000, "Hold", "Keep the mugs", (), ending_counts=(FilmCount("mugs", 2),))
    plan = FilmPlan("p", "P", 5000, (), (shot,))
    run = tmp_path / "run"
    export = run / "export"
    export.mkdir(parents=True)
    (run / "sources").mkdir()
    source = b"bound test video; decoder is replaced by exact frame fixtures"
    digest = hashlib.sha256(source).hexdigest()
    (export / "film.mp4").write_bytes(source)
    (run / "sources" / (digest + ".mp4")).write_bytes(source)
    (run / "run.json").write_text('{"export_attempt":1}')
    take = RenderedTake("one", shot.digest, "a" * 64, digest, None, 5000)
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
    calls = []

    def frames(
        source: Path, directory: Path, indices: tuple[int, ...], source_in_ms: int
    ) -> list[Path]:
        directory.mkdir(parents=True)
        paths = []
        for index in indices:
            path = directory / f"frame-{index}.png"
            path.write_bytes(str(index).encode())
            paths.append(path)
        return paths

    def reviewer(content: list[dict[str, Any]]) -> str:
        calls.append(content)
        if "observed_counts" in content[0]["text"]:
            assert len([item for item in content if item["type"] == "image"]) == 1
            assert content[1]["image"].endswith("frame-119.png")
            assert "Keep the mugs" not in content[0]["text"]
            assert "= 2" not in content[0]["text"]
            if count == "malformed":
                return "not JSON"
            assert count is None or isinstance(count, int)
            return _response(count, [f"mug {i}" for i in range(count or 0)])
        return json.dumps(
            {
                "checks": {
                    name: {"status": "pass", "evidence": "fixture", "frame_indices": [0, 119]}
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

    monkeypatch.setattr(film_audit_cli, "_frames", frames)
    report_path = film_audit_cli.audit_export(
        run,
        inputs,
        tmp_path,
        tmp_path,
        tmp_path / "audit",
        reviewer=reviewer,
        model_sha256="b" * 64,
        shot_ids=("one",),
    )
    report = json.loads(report_path.read_text())
    row = report["shots"][0]
    assert row["status"] == status
    assert row["ending_counts_source_frame_index"] == 119
    assert row["ending_counts_source_frame_sha256"] == hashlib.sha256(b"119").hexdigest()
    response = report_path.parent / "one/ending-counts/model-response.txt"
    assert row["ending_counts_response_sha256"] == hashlib.sha256(response.read_bytes()).hexdigest()
    assert report["approval_created"] is False
    assert len(calls) == (3 if count == "malformed" else 2)
