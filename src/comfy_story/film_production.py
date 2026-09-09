"""Bounded, resumable visual verification around the public Story film command.

Machine checks control scheduling, never creator approval. Each attempt queues only the
selected prefix and its candidate. Earlier nodes recover their exact stored outputs; rejected
candidates cannot become the parent of the next story event.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from comfy_story.film_audit import (
    VISUAL_AUDIT_PROTOCOL,
    ShotVisualAudit,
    apply_locked_camera,
    apply_opening_conditions,
    condition_scopes,
    parse_camera_pairs,
    parse_opening_conditions,
    parse_visual_audit,
    sample_frame_indices,
    visual_audit_prompt,
)
from comfy_story.film_audit_cli import _local_reviewer, _model_identity, audit_export
from comfy_story.film_border_audit import apply_border_policy
from comfy_story.film_count_audit import apply_ending_counts, parse_category_counts
from comfy_story.film_edit_artifacts import apply_edit_artifacts, parse_edit_artifact_pairs
from comfy_story.film_export import file_digest
from comfy_story.film_narrative import (
    NARRATIVE_PROTOCOL,
    FilmNarrativeContract,
    narrative_request,
    parse_narrative_findings,
)
from comfy_story.film_plan import FilmPlan, planned_shot_conditions
from comfy_story.film_retell import FilmRetell, parse_film_retell
from comfy_story.film_runner import _preflight, _write, run_film
from comfy_story.film_selected_state import (
    SelectedStateSource,
    ShotStateBinding,
    select_film_state_evidence,
)
from comfy_story.film_staging import (
    fit_authored_opening,
    stage_opening,
    validate_staging_configuration,
)
from comfy_story.film_staging_workflow import STAGING_PROTOCOL
from comfy_story.film_starting_state import STARTING_STATE_PROTOCOL, check_starting_state
from comfy_story.film_state_audit import (
    apply_ending_states,
    ending_conditions,
    parse_ending_states,
    state_definitions,
    state_vocabularies,
)
from comfy_story.film_visibility_audit import apply_full_visibility, parse_visibility_report
from comfy_story.story_contracts import canonical_story_json
from comfy_story.story_native_archive import NativeReferenceArchive
from comfy_story.story_store import StoryProjectStore

Reviewer = Callable[[list[dict[str, Any]]], str]


class ProductionStopped(RuntimeError):
    """Production needs attention; previous attempts and resumable state remain intact."""


class ProductionPaused(ProductionStopped):
    """The customer paused at a shot boundary; existing work remains resumable."""


def _pause_if_requested(directory: Path) -> None:
    if (directory / "pause.requested").exists():
        raise ProductionPaused("Paused at a shot boundary; no later shots queued")


def _write_selected_state(path: Path, value: dict[str, object]) -> None:
    if path.exists() and path.read_bytes() != canonical_story_json(value):
        raise ValueError("recorded selected state evidence changed")
    _write(path, value)


def _prefix_inputs(
    settings: dict[str, Any], plan: FilmPlan, end: int, base: Path
) -> dict[str, Any]:
    prefix = {**settings, "shots": [dict(row) for row in settings["shots"][:end]]}
    length = sum(shot.duration_ms for shot in plan.shots[:end])
    cues = {cue.cue_id for shot in plan.shots[:end] for cue in shot.dialogue}
    prefix["audio"] = [
        {
            **row,
            "path": str((base / row["path"]).resolve()),
            "duration_ms": min(row["duration_ms"], length - row["start_ms"]),
        }
        for row in settings.get("audio", [])
        if row["start_ms"] < length and (not row.get("cue_id") or row["cue_id"] in cues)
    ]
    prefix["repair_notes"] = {
        key: value
        for key, value in settings.get("repair_notes", {}).items()
        if key in {shot.shot_id for shot in plan.shots[:end]}
    }
    return prefix


def _report(
    run: Path,
    inputs: Path,
    root: Path,
    model: Path,
    comfy_input: Path,
    reviewer: Reviewer,
    identity: str,
    shot_ids: tuple[str, ...] | None,
    condition_vocabularies: dict[str, list[str]],
) -> Path:
    # A crash during review does not consume another render attempt or overwrite its evidence.
    for number in range(1, 101):
        directory = root / f"audit-{number:03}"
        result = directory / "report.json"
        if result.exists():
            return result
        if not directory.exists():
            return audit_export(
                run,
                inputs,
                comfy_input,
                model,
                directory,
                reviewer=reviewer,
                model_sha256=identity,
                shot_ids=shot_ids,
                condition_vocabularies=condition_vocabularies,
            )
    raise ValueError("too many interrupted audits; inspect the preserved attempts")


def _checked_audits(
    report_path: Path,
    run: Path,
    plan: FilmPlan,
    settings: dict[str, Any],
    identity: str,
    indices: tuple[int, ...],
    condition_vocabularies: dict[str, list[str]],
) -> tuple[ShotVisualAudit, ...]:
    report = json.loads(report_path.read_text())
    run_state = json.loads((run / "run.json").read_text())
    number = run_state.get("export_attempt", 1)
    export = run / ("export" if number <= 1 else f"export-{number:03}")
    receipt = json.loads((export / "export-receipt.json").read_text())
    takes = json.loads((export / "rendered-takes.json").read_text())
    expected_ids = [plan.shots[index].shot_id for index in indices]
    if (
        report.get("format") != VISUAL_AUDIT_PROTOCOL
        or report.get("film_sha256") != file_digest(export / "film.mp4")
        or report.get("plan_sha256") != hashlib.sha256(canonical_story_json(plan)).hexdigest()
        or report.get("state_vocabularies_sha256")
        != hashlib.sha256(canonical_story_json(condition_vocabularies)).hexdigest()
        or report.get("state_definitions") != state_definitions(plan)
        or report.get("model_sha256") != identity
        or report.get("audited_shot_ids") != expected_ids
        or report.get("approval_created") is not False
        or report.get("references_bound_to_submission") is not True
        or receipt.get("review_status") != "unreviewed"
        or len(report.get("shots", [])) != len(indices)
    ):
        raise ValueError("visual assessment is stale or not bound to this production")
    uploaded = json.loads((run / "input-sha256s.json").read_text())
    references = settings["library"]["references"]
    if report.get("reference_sha256s") != {
        row["name"]: uploaded[row["file"]] for row in references
    }:
        raise ValueError("assessment references do not match the submitted images")
    roles = {row["name"]: row["role"] for row in references}
    audits = []
    for index, row in zip(indices, report["shots"], strict=True):
        shot = plan.shots[index]
        take = takes[index]
        if row.get("video_sha256") != take["video_sha256"] or row.get("shot_sha256") != shot.digest:
            raise ValueError("assessment refers to a different take or shot")
        source = run / "sources" / (take["video_sha256"] + ".mp4")
        if file_digest(source) != take["video_sha256"]:
            raise ValueError("assessed source video changed")
        frames = sample_frame_indices(shot.duration_ms)
        conditions = planned_shot_conditions(plan, index)
        opening_conditions, continuing_conditions = condition_scopes(shot, conditions)
        _, checks = visual_audit_prompt(
            shot, frames, roles, expected_conditions=continuing_conditions
        )
        evidence_root = report_path.parent / shot.shot_id
        response = evidence_root / "model-response.txt"
        if row.get("response_sha256") != file_digest(response) or row.get("sampled_frames") != [
            {"frame_index": frame, "sha256": file_digest(evidence_root / f"frame-{number:02}.png")}
            for number, frame in enumerate(frames, 1)
        ]:
            raise ValueError("assessment response or sampled frame evidence changed")
        raw = response.read_text()
        try:
            parsed = parse_visual_audit(
                raw,
                shot,
                video_sha256=take["video_sha256"],
                model_sha256=identity,
                sampled_frames=frames,
                required_checks=checks,
                expected_conditions=continuing_conditions,
            )
        except ValueError as error:
            raise ProductionStopped(
                f"Visual checker could not assess {shot.shot_id}: {error}"
            ) from error
        if opening_conditions:
            opening_response = evidence_root / "opening-state" / "model-response.txt"
            if (
                not opening_response.is_file()
                or row.get("opening_response_sha256") != file_digest(opening_response)
                or row.get("opening_source_frame_index") != frames[0]
            ):
                raise ValueError("opening assessment or frame binding changed")
            try:
                opening_observed = parse_opening_conditions(
                    opening_response.read_text(), opening_conditions
                )
            except ValueError:
                opening_observed = {}
            parsed = apply_opening_conditions(parsed, opening_observed, opening_conditions)
        if shot.fully_visible_throughout:
            visibility_response = evidence_root / "full-visibility" / "model-response.txt"
            if not visibility_response.is_file() or row.get(
                "full_visibility_response_sha256"
            ) != file_digest(visibility_response):
                raise ValueError("full visibility assessment changed")
            try:
                visible = parse_visibility_report(
                    visibility_response.read_text(), shot.fully_visible_throughout, frames
                )
            except ValueError:
                visible = {}
            parsed = apply_full_visibility(parsed, visible, shot.fully_visible_throughout, frames)
        if shot.camera_policy == "Locked frame":
            camera_response = evidence_root / "camera" / "model-response.txt"
            if not camera_response.is_file() or row.get("camera_response_sha256") != file_digest(
                camera_response
            ):
                raise ValueError("camera assessment changed")
            try:
                camera_observed = parse_camera_pairs(camera_response.read_text(), frames)
            except ValueError:
                camera_observed = {}
            parsed = apply_locked_camera(parsed, camera_observed, frames)
        if shot.camera_policy == "Single continuous shot":
            edit_response = evidence_root / "edit-artifacts" / "model-response.txt"
            if not edit_response.is_file() or row.get(
                "edit_artifacts_response_sha256"
            ) != file_digest(edit_response):
                raise ValueError("edit artifact assessment changed")
            try:
                edit_observed = parse_edit_artifact_pairs(edit_response.read_text(), frames)
            except ValueError:
                edit_observed = {}
            parsed = apply_edit_artifacts(parsed, edit_observed, frames)
        frame_paths = [
            evidence_root / f"frame-{number:02}.png" for number in range(1, len(frames) + 1)
        ]
        if shot.ending_counts:
            count_response = evidence_root / "ending-counts" / "model-response.txt"
            if (
                not count_response.is_file()
                or row.get("ending_counts_response_sha256") != file_digest(count_response)
                or row.get("ending_counts_source_frame_index") != frames[-1]
                or row.get("ending_counts_source_frame_sha256") != file_digest(frame_paths[-1])
            ):
                raise ValueError("ending count assessment or frame binding changed")
            try:
                counts = parse_category_counts(
                    count_response.read_text(), tuple(item.category for item in shot.ending_counts)
                )
            except ValueError:
                counts = {}
            parsed = apply_ending_counts(parsed, counts, shot.ending_counts, frames[-1])
        if shot.border_policy == "Preserve opening borders":
            parsed = apply_border_policy(parsed, frame_paths, frames)
        expected_ending = ending_conditions(plan, index)
        if expected_ending:
            ending_response = evidence_root / "ending-state" / "model-response.txt"
            if (
                not ending_response.is_file()
                or row.get("ending_state_response_sha256") != file_digest(ending_response)
                or row.get("ending_state_source_frame_index") != frames[-1]
                or row.get("ending_state_source_frame_sha256") != file_digest(frame_paths[-1])
            ):
                raise ValueError("ending state assessment or frame binding changed")
            try:
                observed_ending = parse_ending_states(
                    ending_response.read_text(),
                    {key: condition_vocabularies[key] for key in expected_ending},
                )
            except ValueError:
                observed_ending = {}
            parsed = apply_ending_states(parsed, observed_ending, expected_ending, frames[-1])
        if parsed.status != row.get("status"):
            raise ValueError("assessment summary disagrees with its actual observations")
        audits.append(parsed)
    return tuple(audits)


def _failure_summary(audit: ShotVisualAudit, plan: FilmPlan, index: int) -> str:
    failures = [f"{row.check}: {row.evidence}" for row in audit.findings if row.status != "pass"]
    for label, wanted, observed in [
        ("condition", planned_shot_conditions(plan, index), dict(audit.observed_conditions)),
        (
            "ending",
            {fact.key: fact.value for fact in plan.shots[index].effects},
            dict(audit.observed_effects),
        ),
    ]:
        failures.extend(
            f"Required {label} {key}={value}; previous take showed {observed.get(key, 'unknown')}."
            for key, value in wanted.items()
            if observed.get(key) != value
        )
    # Finish on the desired state, rather than leaving the rejected state as the last instruction.
    targets = "; ".join(f"{fact.key}={fact.value}" for fact in plan.shots[index].effects)
    correction = " ".join(failures)[:1200]
    if targets:
        correction += " This attempt must visibly end with: " + targets[:700] + "."
    # Keep the customer-facing diagnostic bounded and free of reference activation syntax.
    return correction.replace("@", "").replace("\x00", "")[:2000]


def _repair(plan: FilmPlan, index: int) -> str:
    """Reinforce authored targets without turning rejected observations into instructions."""
    shot = plan.shots[index]
    lines = ["Render the saved shot action and its required ending visibly."]
    # Whole clauses only: cutting a long clause can invert its meaning. The complete
    # original requirements are already present in the compiled shot prompt.
    targets = [f"Required ending: {fact.key} = {fact.value}." for fact in shot.effects]
    targets.append(f"Action to complete visibly: {shot.action}")
    for target in targets:
        target = target.replace("@", "").replace("\x00", "")
        if len("\n".join((*lines, target))) <= 2000:
            lines.append(target)
    return "\n".join(lines)


def _check_narrative_assessment(
    report: Path,
    contract: FilmNarrativeContract,
    retell: FilmRetell,
    model_sha256: str,
) -> None:
    directory = report.parent / "narrative"
    summary_path = directory / "report.json"
    parent = json.loads(report.read_text())
    if not summary_path.is_file() or parent.get("narrative_assessment_sha256") != file_digest(
        summary_path
    ):
        raise ValueError("intended narrative assessment is missing or changed")
    request_path, response_path = directory / "request.json", directory / "model-response.txt"
    if not request_path.is_file() or not response_path.is_file():
        raise ValueError("intended narrative observation evidence is missing")
    summary = json.loads(summary_path.read_text())
    expected_request = narrative_request(
        contract, retell, file_digest(report.parent / "blind-story-retell.txt"), model_sha256
    )
    if (
        summary.get("format") != NARRATIVE_PROTOCOL
        or summary.get("request_sha256") != file_digest(request_path)
        or summary.get("response_sha256") != file_digest(response_path)
        or request_path.read_bytes() != canonical_story_json(expected_request)
    ):
        raise ValueError("intended narrative assessment binding changed")
    findings = parse_narrative_findings(response_path.read_text(), retell)
    supported = all(row.status == "supported" for row in findings)
    if (
        summary.get("findings") != [asdict(row) for row in findings]
        or summary.get("supported") is not supported
    ):
        raise ValueError("intended narrative summary disagrees with its observations")
    if not supported:
        missing = ", ".join(row.field for row in findings if row.status != "supported")
        detail = " ".join(
            f"{row.field}: {row.explanation}" for row in findings if row.status != "supported"
        )[:500]
        raise ProductionStopped(
            f"The blind retelling does not establish the intended narrative: {missing}. "
            f"{detail} Review the preserved film and observations."
        )


def run_production(
    plan_path: Path,
    inputs_path: Path,
    server: str,
    directory: Path,
    *,
    model: Path,
    comfy_input: Path,
    device: str = "cpu",
    max_attempts: int = 2,
    reviewer: Reviewer | None = None,
    model_sha256: str | None = None,
    story_root: Path | None = None,
    staging_identity: dict[str, Any] | None = None,
) -> Path:
    """Render/check/retry from fixed parents, then run a whole-film diagnostic.

    Historical one-queue drafts remain available through run_film. This opt-in customer mode
    has a finite render budget and halts on uncertainty. It never writes AcceptedTake or canon
    approval records, and machine success is only readiness for creator review.
    """
    if type(max_attempts) is not int or not 1 <= max_attempts <= 4:
        raise ValueError("max_attempts must be an integer from 1 to 4")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "production.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("verified production is already running") from error
        return _produce(
            plan_path,
            inputs_path,
            server,
            directory,
            model,
            comfy_input,
            device,
            max_attempts,
            reviewer,
            model_sha256,
            story_root,
            staging_identity,
        )


def _produce(
    plan_path: Path,
    inputs_path: Path,
    server: str,
    directory: Path,
    model: Path,
    comfy_input: Path,
    device: str,
    max_attempts: int,
    reviewer: Reviewer | None,
    model_sha256: str | None,
    story_root: Path | None,
    staging_identity: dict[str, Any] | None,
) -> Path:
    plan, settings, _, _, _, server = _preflight(
        plan_path, inputs_path, server, allow_pending_staging=True
    )
    has_staging = any(row.get("opening_prompt", "").strip() for row in settings["shots"])
    if has_staging and staging_identity is None:
        raise ValueError("opening staging requires a verified host model identity")
    if has_staging and staging_identity is not None:
        validate_staging_configuration(server, staging_identity)
    recall_enabled = settings.get("recall_selected_state", False)
    if type(recall_enabled) is not bool:
        raise ValueError("recall_selected_state must be a boolean")
    archive_needed = recall_enabled or has_staging
    if archive_needed and story_root is None:
        raise ValueError("selected state recall requires the Comfy host's native Story archive")
    archive = (
        NativeReferenceArchive(StoryProjectStore(story_root.resolve()))
        if archive_needed and story_root
        else None
    )
    if recall_enabled and any(
        row.get("shot_state_evidence", "{}") != "{}" for row in settings["shots"]
    ):
        raise ValueError(
            "automatic selected state recall cannot mix with manually supplied shot evidence"
        )
    vocabularies = state_vocabularies(plan)
    if len(settings["shots"]) != len(plan.shots):
        raise ValueError("verified production requires inputs for every shot")
    model_id = _model_identity(model) if reviewer is None else model_sha256
    if (
        model_id is None
        or len(model_id) != 64
        or any(c not in "0123456789abcdef" for c in model_id)
    ):
        raise ValueError("verified production requires a model SHA-256 identity")
    identity = hashlib.sha256(
        canonical_story_json(
            {
                "format": "duet-verified-production-v1",
                "visual_audit_protocol": VISUAL_AUDIT_PROTOCOL,
                "starting_state_protocol": STARTING_STATE_PROTOCOL,
                "plan": plan,
                "inputs": settings,
                **(
                    {"staging_identity": staging_identity, "staging_protocol": STAGING_PROTOCOL}
                    if has_staging
                    else {}
                ),
                **(
                    {"selected_state_archive": str(story_root.resolve())}
                    if archive_needed and story_root
                    else {}
                ),
                "server": server,
                "model_sha256": model_id,
                "max_attempts": max_attempts,
                "comfy_input": str(comfy_input.resolve()),
            }
        )
    ).hexdigest()
    path = directory / "production.json"
    state: dict[str, Any] = (
        json.loads(path.read_text()) if path.exists() else {"identity": identity, "decisions": {}}
    )
    if state.get("identity") != identity:
        raise ValueError("production inputs or verification policy changed; use a new directory")
    state.pop("reason", None)
    state.update(status="running", phase="checking_verifier", approval_created=False, selected=[])
    _write(path, state)
    loaded = reviewer

    def review(content: list[dict[str, Any]]) -> str:
        nonlocal loaded
        if loaded is None:
            loaded = _local_reviewer(model, device)
        return loaded(content)

    selected: list[dict[str, Any]] = []
    selected_sources: list[SelectedStateSource] = []
    selected_inputs: list[dict[str, Any]] = []
    repairs: dict[str, str] = dict(settings.get("repair_notes", {}))
    last_run = directory
    last_inputs = inputs_path
    final = directory / "film.mp4"
    try:
        _pause_if_requested(directory)
        # Validate optional vision dependencies and load the checker before spending render time.
        # This also makes a CUDA memory reservation visible to Comfy's model manager up front.
        if loaded is None:
            loaded = _local_reviewer(model, device)
        state["phase"] = "checking_inputs"
        _write(path, state)
        first = settings["shots"][0]
        if (
            planned_shot_conditions(plan, 0)
            and not first.get("opening_prompt", "").strip()
            and first.get("world") not in (None, "None", "")
        ):
            first["world"] = fit_authored_opening(
                server, directory / "starting-canvas", comfy_input, first["world"]
            )
        starting: dict[str, Any] = (
            {"format": STARTING_STATE_PROTOCOL, "status": "deferred_to_opening_stage"}
            if settings["shots"][0].get("opening_prompt", "").strip()
            else check_starting_state(
                plan, settings, comfy_input, directory / "starting-state", review, model_id
            )
        )
        starting_sha = hashlib.sha256(canonical_story_json(starting)).hexdigest()
        if state.get("starting_state_sha256", starting_sha) != starting_sha:
            raise ValueError("recorded starting-state assessment changed")
        state.update(starting_state_sha256=starting_sha, starting_state=starting)
        _write(path, state)
        if starting["status"] in ("fail", "needs_review"):
            differences = "; ".join(
                f"{key}: expected {row['expected']}, observed {row['observed']}"
                for key, row in starting["conditions"].items()
                if row["observed"] != row["expected"]
            )
            raise ProductionStopped(
                "Opening image does not establish the story's initial conditions: "
                + differences[:500]
                + ". No video shots queued. Review the opening image and initial conditions, "
                "then save a new plan. The original assessment is preserved."
            )
        for index, shot in enumerate(plan.shots):
            prefix_plan = replace(
                plan,
                shots=plan.shots[: index + 1],
                target_duration_ms=sum(s.duration_ms for s in plan.shots[: index + 1]),
            ).validate()
            staged_world = None
            if settings["shots"][index].get("opening_prompt", "").strip():
                if archive is None or staging_identity is None:
                    raise ValueError("opening staging archive or model identity is missing")
                stage_bindings = (
                    select_film_state_evidence(plan, index, tuple(selected_sources), archive)
                    if index
                    else ()
                )

                def progress(phase: str, image_attempt: int, shot_id: str = shot.shot_id) -> None:
                    state.update(shot_id=shot_id, phase=phase, opening_attempt=image_attempt)
                    _write(path, state)

                stage_options: dict[str, Any] = {}
                if settings["shots"][index].get("opening_use_previous_scene", False):
                    # select_film_state_evidence above authenticated this exact selected prefix.
                    previous = selected_sources[-1]
                    parent = archive.load_revision(previous.revision_sha256)
                    stage_options["scene_source"] = {
                        "source_revision_sha256": previous.revision_sha256,
                        "source_video_sha256": previous.audit.video_sha256,
                        "assessment_sha256": previous.assessment_sha256,
                        "asset_sha256": parent.state.last_frame_sha256,
                    }
                staged_world = stage_opening(
                    plan,
                    index,
                    settings,
                    comfy_input,
                    directory / f"shot-{index + 1:03}" / "opening",
                    server,
                    review,
                    model_id,
                    staging_identity,
                    stage_bindings,
                    archive,
                    lambda: _pause_if_requested(directory),
                    progress,
                    **stage_options,
                )
            for attempt in range(max_attempts):
                _pause_if_requested(directory)
                attempt_root = directory / f"shot-{index + 1:03}" / f"attempt-{attempt + 1:02}"
                attempt_root.mkdir(parents=True, exist_ok=True)
                prefix = _prefix_inputs(settings, plan, index + 1, inputs_path.parent)
                for prior, chosen_input in enumerate(selected_inputs):
                    prefix["shots"][prior] = dict(chosen_input)
                if staged_world is not None:
                    prefix["shots"][index].update(world=staged_world, opening_prompt="")
                    prefix["shots"][index].pop("opening_use_previous_scene", None)
                    prefix["shots"][index].pop("opening_mode", None)
                variation = (settings["shots"][index]["variation"] + attempt) % (1 << 64)
                prefix["shots"][index]["variation"] = variation
                prefix["repair_notes"] = {
                    key: value
                    for key, value in repairs.items()
                    if key in {s.shot_id for s in prefix_plan.shots}
                }
                if archive is not None:
                    bindings: tuple[ShotStateBinding, ...] = ()
                    if (
                        recall_enabled
                        and settings["shots"][index].get("render_profile", "Reference shot")
                        == "Reference shot"
                    ):
                        bindings = select_film_state_evidence(
                            plan, index, tuple(selected_sources), archive
                        )
                    if bindings:
                        prefix["shots"][index]["shot_state_evidence"] = canonical_story_json(
                            {item.entity_id: item.evidence_id for item in bindings}
                        ).decode()
                    _write_selected_state(
                        attempt_root / "selected-state.json",
                        {
                            "format": "duet-film-selected-state-v1",
                            "creator_approval": False,
                            "bindings": bindings,
                            "mode": "reference_evidence"
                            if settings["shots"][index].get("render_profile", "Reference shot")
                            == "Reference shot"
                            else "starting_frame_only",
                        },
                    )
                candidate_plan = attempt_root / "plan.json"
                candidate_inputs = attempt_root / "inputs.json"
                _write(candidate_plan, prefix_plan)
                _write(candidate_inputs, prefix)
                run = attempt_root / "run"
                state.update(shot_id=shot.shot_id, attempt=attempt + 1, phase="rendering_shot")
                _write(path, state)
                print(f"{shot.shot_id}: attempt {attempt + 1}/{max_attempts}", flush=True)
                output = run_film(candidate_plan, candidate_inputs, server, run)
                # This is the rendered prefix with the declared soundtrack, not an approval.
                # Preserve it before checking so a stopped customer can inspect the candidate.
                state["preview"] = {
                    "film": str(output),
                    "film_sha256": file_digest(output),
                    "shot_id": shot.shot_id,
                    "attempt": attempt + 1,
                    "duration_ms": prefix_plan.target_duration_ms,
                }
                _write(path, state)
                takes = json.loads((output.parent / "rendered-takes.json").read_text())
                if any(
                    takes[i]["video_sha256"] != prior["video_sha256"]
                    or takes[i]["revision_sha256"] != prior["revision_sha256"]
                    for i, prior in enumerate(selected)
                ):
                    raise ValueError("a previously selected parent changed during production")
                state["phase"] = "checking_shot"
                _write(path, state)
                report = _report(
                    run,
                    candidate_inputs,
                    attempt_root,
                    model,
                    comfy_input,
                    review,
                    model_id,
                    (shot.shot_id,),
                    vocabularies,
                )
                key = f"{index + 1}:{attempt + 1}"
                digest = file_digest(report)
                if key in state["decisions"] and state["decisions"][key] != digest:
                    raise ValueError("a recorded production assessment changed")
                state["decisions"][key] = digest
                audits = _checked_audits(
                    report, run, prefix_plan, prefix, model_id, (index,), vocabularies
                )
                audit = audits[0]
                state.update(shot_id=shot.shot_id, attempt=attempt + 1, last_assessment=str(report))
                _write(path, state)
                print(f"{shot.shot_id}: {audit.status}", flush=True)
                if audit.status == "machine_pass":
                    selected_sources.append(
                        SelectedStateSource(takes[index]["revision_sha256"], digest, audit)
                    )
                    selected_inputs.append(dict(prefix["shots"][index]))
                    selected.append(
                        {
                            "shot_id": shot.shot_id,
                            "variation": variation,
                            "video_sha256": audit.video_sha256,
                            "revision_sha256": takes[index]["revision_sha256"],
                            "assessment_sha256": digest,
                        }
                    )
                    state["selected"] = list(selected)
                    _write(path, state)
                    last_run, last_inputs, final = run, candidate_inputs, output
                    break
                if audit.status == "needs_review":
                    raise ProductionStopped(
                        f"{shot.shot_id}: visual evidence is uncertain; automatic retries stopped"
                    )
                if attempt + 1 == max_attempts:
                    raise ProductionStopped(
                        f"{shot.shot_id}: failed after {max_attempts} attempts; "
                        "no downstream shots queued. "
                        f"{_failure_summary(audit, prefix_plan, index)[:500]}"
                    )
                repairs[shot.shot_id] = _repair(prefix_plan, index)
        _pause_if_requested(directory)
        full_root = directory / "whole-film"
        full_root.mkdir(exist_ok=True)
        state["phase"] = "checking_film"
        _write(path, state)
        report = _report(
            last_run,
            last_inputs,
            full_root,
            model,
            comfy_input,
            review,
            model_id,
            None,
            vocabularies,
        )
        final_settings = json.loads(last_inputs.read_text())
        audits = _checked_audits(
            report,
            last_run,
            plan,
            final_settings,
            model_id,
            tuple(range(len(plan.shots))),
            vocabularies,
        )
        if any(audit.status != "machine_pass" for audit in audits):
            raise ProductionStopped(
                "Whole-film review found a remaining defect; completed candidates are preserved"
            )
        report_data = json.loads(report.read_text())
        retell_path = report.parent / "blind-story-retell.txt"
        if report_data.get("full_film_audit") is not True or report_data.get(
            "retell_sha256"
        ) != file_digest(retell_path):
            raise ValueError("whole-film retelling is missing or changed")
        digest = file_digest(report)
        if state["decisions"].get("whole-film", digest) != digest:
            raise ValueError("recorded whole-film assessment changed")
        state["decisions"]["whole-film"] = digest
        try:
            retell = parse_film_retell(retell_path.read_text())
        except ValueError as error:
            raise ProductionStopped("Whole-film retelling could not be assessed") from error
        if retell.coherence != "coherent" or retell.contradictions:
            raise ProductionStopped(
                "Whole-film retelling is inconsistent or uncertain; review is required"
            )
        if plan.narrative is not None:
            _check_narrative_assessment(report, plan.narrative, retell, model_id)
        state.update(
            status="ready_for_review",
            phase="complete",
            film=str(final),
            whole_film_assessment=str(report),
            film_sha256=file_digest(final),
            selected=selected,
        )
        _write(path, state)
        return final
    except (ProductionStopped, ValueError, RuntimeError, OSError) as error:
        state.update(
            status="paused" if isinstance(error, ProductionPaused) else "needs_attention",
            reason=str(error),
            selected=selected,
        )
        _write(path, state)
        if isinstance(error, ProductionStopped):
            raise type(error)(f"{error}. Recovery record: {path}") from error
        raise
