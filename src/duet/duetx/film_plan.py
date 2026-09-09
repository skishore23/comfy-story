"""Film-level causality, take acceptance, and exact editorial timing.

These sidecar contracts reference Story revisions without changing historical memory formats.
Acceptance records are review decisions, not assertions inferred from the generation prompt.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Self

from duet.duetx.film_narrative import FilmNarrativeContract
from duet.duetx.story_contracts import canonical_story_json

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("film identifiers must be portable nonempty strings")


@dataclass(frozen=True, slots=True)
class FilmFact:
    """One value for a named fact, e.g. key.owner=Mira or Mira.location=workshop."""

    key: str
    value: str


@dataclass(frozen=True, slots=True)
class FilmStateDefinition:
    """Creator-authored visible evidence that distinguishes one state alternative."""

    key: str
    value: str
    definition: str


def _facts(items: tuple[FilmFact, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        _identifier(item.key)
        if not isinstance(item.value, str) or not item.value.strip():
            raise ValueError("film facts require nonempty values")
        if item.key in result:
            raise ValueError(f"duplicate film fact: {item.key}")
        result[item.key] = item.value
    return result


@dataclass(frozen=True, slots=True)
class FilmCue:
    """An exact utterance or visible string at integer millisecond offsets within a shot."""

    cue_id: str
    start_ms: int
    end_ms: int
    text: str
    language: str
    speaker: str = ""

    def validate(self, duration_ms: int) -> Self:
        _identifier(self.cue_id)
        if (
            type(self.start_ms) is not int
            or type(self.end_ms) is not int
            or not 0 <= self.start_ms < self.end_ms <= duration_ms
        ):
            raise ValueError("film cue must fit inside its shot")
        if not self.text.strip() or "\x00" in self.text or "-->" in self.text:
            raise ValueError("film cue requires plain nonempty text")
        if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", self.language):
            raise ValueError("film cue requires an explicit language tag")
        if self.speaker:
            _identifier(self.speaker)
        return self


@dataclass(frozen=True, slots=True)
class FilmCount:
    """Creator-declared visible category count at the final edited frame."""

    category: str
    count: int


@dataclass(frozen=True, slots=True)
class FilmShot:
    shot_id: str
    duration_ms: int
    purpose: str
    action: str
    present: tuple[str, ...]
    absent: tuple[str, ...] = ()
    requires: tuple[FilmFact, ...] = ()
    effects: tuple[FilmFact, ...] = ()
    depends_on: tuple[str, ...] = ()
    dialogue: tuple[FilmCue, ...] = ()
    text: tuple[FilmCue, ...] = ()
    composition: str = "New composition"
    reference_names: tuple[str, ...] | None = None
    visible_throughout: tuple[str, ...] = ()
    camera_policy: str = field(default="Follow prompt", metadata={"omit_wire_default": True})
    ending_counts: tuple[FilmCount, ...] = field(default=(), metadata={"omit_wire_default": True})
    border_policy: str = field(default="Follow prompt", metadata={"omit_wire_default": True})
    fully_visible_throughout: tuple[str, ...] = field(
        default=(), metadata={"omit_wire_default": True}
    )
    direction_version: int = field(default=1, metadata={"omit_wire_default": True})

    @property
    def digest(self) -> str:
        payload = asdict(self)
        if self.direction_version == 1:
            payload.pop("direction_version")
        if not self.fully_visible_throughout:
            payload.pop("fully_visible_throughout")
        if self.border_policy == "Follow prompt":
            payload.pop("border_policy")
        if not self.ending_counts:
            payload.pop("ending_counts")
        if self.camera_policy == "Follow prompt":
            payload.pop("camera_policy")
        if not self.visible_throughout:
            payload.pop("visible_throughout")
        if self.reference_names is None:
            # Preserve existing take identities when no reference scope is set.
            payload.pop("reference_names")
        return hashlib.sha256(canonical_story_json(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class FilmPlan:
    project_id: str
    title: str
    target_duration_ms: int
    initial_facts: tuple[FilmFact, ...]
    shots: tuple[FilmShot, ...]
    languages: tuple[str, ...] = ("en",)
    version: int = 1
    state_definitions: tuple[FilmStateDefinition, ...] = field(
        default=(), metadata={"omit_wire_default": True}
    )
    narrative: FilmNarrativeContract | None = field(
        default=None, metadata={"omit_wire_default": True}
    )

    def validate(self) -> Self:
        if self.narrative is not None:
            if not isinstance(self.narrative, FilmNarrativeContract):
                raise ValueError("film narrative must be an intended-story contract")
            self.narrative.validate()
        _identifier(self.project_id)
        if self.version != 1 or not self.title.strip():
            raise ValueError("unsupported film version or empty title")
        if type(self.target_duration_ms) is not int or self.target_duration_ms <= 0:
            raise ValueError("film duration must be positive integer milliseconds")
        if not self.shots or not self.languages or len(set(self.languages)) != len(self.languages):
            raise ValueError("film requires shots and unique languages")
        state = _facts(self.initial_facts)
        seen: set[str] = set()
        cues: set[str] = set()
        for shot in self.shots:
            _identifier(shot.shot_id)
            if shot.shot_id in seen:
                raise ValueError("duplicate film shot")
            if type(shot.duration_ms) is not int or shot.duration_ms <= 0:
                raise ValueError("shot duration must be positive integer milliseconds")
            if not shot.purpose.strip() or not shot.action.strip():
                raise ValueError("every shot needs a purpose and action")
            if shot.composition not in {"Continue frame", "New composition"}:
                raise ValueError("unsupported shot composition")
            if not isinstance(shot.camera_policy, str) or shot.camera_policy not in {
                "Follow prompt",
                "Locked frame",
                "Single continuous shot",
            }:
                raise ValueError("unsupported film camera policy")
            if type(shot.direction_version) is not int or shot.direction_version not in {
                1,
                2,
            }:
                raise ValueError("unsupported film shot direction version")
            if not isinstance(shot.border_policy, str) or shot.border_policy not in {
                "Follow prompt",
                "Preserve opening borders",
            }:
                raise ValueError("unsupported film border policy")
            roster = shot.present + shot.absent
            if not isinstance(shot.ending_counts, tuple) or any(
                not isinstance(item, FilmCount)
                or not isinstance(item.category, str)
                or not 1 <= len(item.category.strip()) <= 120
                or item.category != item.category.strip()
                or "\n" in item.category
                or type(item.count) is not int
                or not 0 <= item.count <= 64
                for item in shot.ending_counts
            ):
                raise ValueError("ending counts require bounded categories and integer counts")
            if len(shot.ending_counts) > 8 or len(
                {item.category.casefold() for item in shot.ending_counts}
            ) != len(shot.ending_counts):
                raise ValueError("ending counts require at most eight unique categories")
            for entity in roster:
                _identifier(entity)
            if len(set(roster)) != len(roster):
                raise ValueError("shot roster duplicates or conflicts with absence")
            if shot.reference_names is not None and (
                not isinstance(shot.reference_names, tuple)
                or len(set(shot.reference_names)) != len(shot.reference_names)
                or not set(shot.reference_names) <= set(shot.present)
            ):
                raise ValueError(
                    "active film references must be a unique subset of present entities"
                )
            for roster in (shot.visible_throughout, shot.fully_visible_throughout):
                if (
                    not isinstance(roster, tuple)
                    or any(not isinstance(name, str) for name in roster)
                    or len(set(roster)) != len(roster)
                    or not set(roster) <= set(shot.present)
                ):
                    raise ValueError(
                        "continuous visibility must be a unique subset of present entities"
                    )
            if len(set(shot.depends_on)) != len(shot.depends_on):
                raise ValueError("duplicate shot dependency")
            if not set(shot.depends_on) <= seen:
                raise ValueError("shot dependencies must precede the shot")
            _check_requires(shot, state)
            state.update(_facts(shot.effects))
            for cue in shot.dialogue + shot.text:
                cue.validate(shot.duration_ms)
                if cue.language not in self.languages or cue.cue_id in cues:
                    raise ValueError("film cue has an unapproved language or duplicate ID")
                cues.add(cue.cue_id)
            for cue in shot.dialogue:
                if not cue.speaker:
                    raise ValueError("dialogue requires an explicit speaker")
            seen.add(shot.shot_id)
        if sum(shot.duration_ms for shot in self.shots) != self.target_duration_ms:
            raise ValueError("shot durations must sum to the exact film duration")
        if not isinstance(self.state_definitions, tuple) or len(self.state_definitions) > 256:
            raise ValueError("state definitions require at most 256 records")
        meanings: dict[str, set[str]] = {}
        for item in self.state_definitions:
            if not isinstance(item, FilmStateDefinition):
                raise ValueError("state definitions require typed records")
            _identifier(item.key)
            if any(
                not isinstance(text, str)
                or not text.strip()
                or len(text) > 2000
                or "\x00" in text
                or "@" in text
                for text in (item.value, item.definition)
            ):
                raise ValueError(
                    "state definitions require bounded text without reference mentions"
                )
            alternatives = meanings.setdefault(item.key, set())
            if item.value in alternatives:
                raise ValueError("duplicate state definition key and value")
            alternatives.add(item.value)
        for fact in (
            *self.initial_facts,
            *(fact for shot in self.shots for fact in (*shot.requires, *shot.effects)),
        ):
            if fact.key in meanings and fact.value not in meanings[fact.key]:
                raise ValueError("define every authored alternative of a defined state key")
        return self


def _check_requires(shot: FilmShot, state: dict[str, str]) -> None:
    for key, expected in _facts(shot.requires).items():
        if state.get(key) != expected:
            raise ValueError(f"{shot.shot_id} requires {key}={expected}; found {state.get(key)!r}")


@dataclass(frozen=True, slots=True)
class AcceptedTake:
    """A review bound to an exact shot contract, parent acceptance, and rendered asset."""

    shot_id: str
    shot_sha256: str
    revision_sha256: str
    video_sha256: str
    parent_acceptance_sha256: str
    observed_facts: tuple[FilmFact, ...]
    reviewer: str
    review_note: str
    source_duration_ms: int
    source_in_ms: int = 0

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_story_json(self)).hexdigest()


@dataclass(frozen=True, slots=True)
class RenderedTake:
    """An unreviewed render, with generation ancestry rather than invented acceptance."""

    shot_id: str
    shot_sha256: str
    revision_sha256: str
    video_sha256: str
    parent_revision_sha256: str | None
    source_duration_ms: int
    source_in_ms: int = 0

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_story_json(self)).hexdigest()


def validate_rendered_takes(plan: FilmPlan, takes: tuple[RenderedTake, ...]) -> None:
    """Check a complete generated sequence without advancing any reviewed story facts."""
    plan.validate()
    if len(takes) != len(plan.shots):
        raise ValueError("draft export requires a rendered take for every shot")
    parent: str | None = None
    revisions: set[str] = set()
    for shot, take in zip(plan.shots, takes, strict=True):
        if not isinstance(take, RenderedTake):
            raise ValueError("draft export requires rendered takes, not acceptance records")
        for digest in (take.shot_sha256, take.revision_sha256, take.video_sha256):
            if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
                raise ValueError("rendered take requires SHA-256 identities")
        if take.shot_id != shot.shot_id or take.shot_sha256 != shot.digest:
            raise ValueError("rendered take is out of order or its shot contract is stale")
        if take.parent_revision_sha256 != parent or take.revision_sha256 in revisions:
            raise ValueError("rendered takes must follow one uninterrupted generation ancestry")
        if (
            type(take.source_in_ms) is not int
            or type(take.source_duration_ms) is not int
            or take.source_in_ms < 0
            or take.source_in_ms + shot.duration_ms > take.source_duration_ms
        ):
            raise ValueError("rendered source is too short for the intended trim")
        revisions.add(take.revision_sha256)
        parent = take.revision_sha256


def initial_acceptance_digest(plan: FilmPlan) -> str:
    """Bind authored meanings into new acceptance chains; preserve historical empty defaults."""
    initial: object = plan.initial_facts
    if plan.state_definitions:
        initial = {"initial_facts": plan.initial_facts, "state_definitions": plan.state_definitions}
    return hashlib.sha256(canonical_story_json(initial)).hexdigest()


def accepted_state(plan: FilmPlan, takes: tuple[AcceptedTake, ...]) -> dict[str, str]:
    """Replay only reviewed takes. Missing or stale takes cannot advance narrative facts."""
    plan.validate()
    if len(takes) > len(plan.shots):
        raise ValueError("more accepted takes than planned shots")
    state = _facts(plan.initial_facts)
    parent = initial_acceptance_digest(plan)
    for shot, take in zip(plan.shots[: len(takes)], takes, strict=True):
        if not isinstance(take, AcceptedTake):
            raise ValueError("accepted state requires reviewed acceptance records")
        for digest in (
            take.shot_sha256,
            take.revision_sha256,
            take.video_sha256,
            take.parent_acceptance_sha256,
        ):
            if not _DIGEST.fullmatch(digest):
                raise ValueError("accepted take requires SHA-256 identities")
        if take.shot_id != shot.shot_id or take.shot_sha256 != shot.digest:
            raise ValueError("accepted take is out of order or its shot contract is stale")
        if take.parent_acceptance_sha256 != parent:
            raise ValueError("accepted take has a stale parent; review the downstream shot")
        if not take.reviewer.strip() or not take.review_note.strip():
            raise ValueError("acceptance requires a reviewer and evidence-based review note")
        if (
            type(take.source_in_ms) is not int
            or type(take.source_duration_ms) is not int
            or take.source_in_ms < 0
            or take.source_in_ms + shot.duration_ms > take.source_duration_ms
        ):
            raise ValueError("accepted source is too short for the intended trim")
        _check_requires(shot, state)
        observed = _facts(take.observed_facts)
        if observed != _facts(shot.effects):
            raise ValueError("reviewed facts must match intended effects before acceptance")
        state.update(observed)
        parent = take.digest
    return state


def accept_take(
    plan: FilmPlan, previous: tuple[AcceptedTake, ...], candidate: AcceptedTake
) -> tuple[AcceptedTake, ...]:
    """Append a caller-reviewed candidate; never infer facts or approval from a prompt."""
    accepted = (*previous, candidate)
    accepted_state(plan, accepted)
    return accepted


def compile_shot_prompt(plan: FilmPlan, shot_index: int, takes: tuple[AcceptedTake, ...]) -> str:
    """Preflight actual accepted history before authoring the next render direction."""
    state = accepted_state(plan, takes)
    if shot_index != len(takes) or not 0 <= shot_index < len(plan.shots):
        raise ValueError("compile the next shot after the accepted prefix")
    shot = plan.shots[shot_index]
    _check_requires(shot, state)
    return _shot_prompt(shot, state, reviewed=True, definitions=plan.state_definitions)


def compile_candidate_prompt(plan: FilmPlan, shot_index: int) -> str:
    """Prepare an unreviewed storyboard candidate, including for out-of-order production.

    The plan's intended preconditions guide rendering, but this does not accept a take or advance
    actual state. Acceptance still replays the reviewed prefix and verifies every required event.
    """
    state = _planned_prefix_state(plan, shot_index)
    return _shot_prompt(
        plan.shots[shot_index], state, reviewed=False, definitions=plan.state_definitions
    )


def _planned_prefix_state(plan: FilmPlan, shot_index: int) -> dict[str, str]:
    plan.validate()
    if not 0 <= shot_index < len(plan.shots):
        raise ValueError("shot index out of range")
    state = _facts(plan.initial_facts)
    for previous in plan.shots[:shot_index]:
        state.update(_facts(previous.effects))
    return state


def _shot_conditions(shot: FilmShot, state: dict[str, str]) -> dict[str, str]:
    required = _facts(shot.requires)
    effects = _facts(shot.effects)
    namespaces = {name.casefold() for name in shot.present}
    namespaces.update(key.split(".", 1)[0].casefold() for key in required | effects)
    return {
        key: value
        for key, value in state.items()
        if key in required or (key.split(".", 1)[0].casefold() in namespaces and key not in effects)
    }


def planned_shot_conditions(plan: FilmPlan, shot_index: int) -> dict[str, str]:
    """Return scoped rendering requirements, never observations or accepted state.

    Conditions changed by this shot's effects apply at the opening only. Other conditions
    should persist throughout the shot. The audit uses the same scope as the renderer.
    """
    state = _planned_prefix_state(plan, shot_index)
    return _shot_conditions(plan.shots[shot_index], state)


def _shot_prompt(
    shot: FilmShot,
    state: dict[str, str],
    *,
    reviewed: bool,
    definitions: tuple[FilmStateDefinition, ...] = (),
) -> str:
    lines = [f"Shot purpose: {shot.purpose}", f"Action: {shot.action}"]
    if shot.ending_counts:
        lines.append(
            "Depict these quantities in the final frame: "
            + "; ".join(f"{item.category} = {item.count}" for item in shot.ending_counts)
            + "."
        )
    if shot.border_policy == "Preserve opening borders":
        # Describe the desired canvas. Naming unwanted artifacts in a negative instruction
        # can itself condition the video model to draw them; pixel checks remain separate.
        lines.append(
            "Canvas: match the opening frame's picture area and margins exactly for the full shot."
        )
    if shot.camera_policy == "Locked frame":
        lines.append(
            "Camera: fixed position and framing throughout. Keep background scale and screen "
            "position stable; no zoom, pan, tilt, shake or cut. Subjects may move within the frame."
        )
    if shot.camera_policy == "Single continuous shot":
        lines.append(
            "Camera: one continuous take with natural motion. "
            + (
                "The same view and solid, opaque subjects persist throughout the shot."
                if shot.direction_version == 2
                else "No internal cut, dissolve, crossfade, wipe, or superimposed alternate view."
            )
        )
    if shot.present:
        lines.append(
            "Present references: " + ", ".join(f"@{name}" for name in shot.present)
            if shot.reference_names is None
            else "Visible entities: " + ", ".join(shot.present)
        )
    if shot.visible_throughout:
        lines.append(
            "Keep these entities visible throughout, including the final frame: "
            + ", ".join(shot.visible_throughout)
        )
    if shot.fully_visible_throughout:
        lines.append(
            "Keep each of these subjects fully in frame and unobscured by other subjects or "
            "objects throughout, including the final frame: "
            + ", ".join(shot.fully_visible_throughout)
            + ". Ordinary self-occlusion from a natural viewpoint is allowed."
        )
    if shot.absent and shot.direction_version == 1:
        # Version 2 keeps excluded entities in the review contract without naming them in
        # positive generation conditioning. It does not relax staging or acceptance checks.
        # Absence names must not become active @mentions during reference selection.
        lines.append("Do not depict: " + ", ".join(shot.absent))
    # Carry the visible entities' state, plus the explicitly scoped fact namespaces.
    # Unrelated offscreen history must not become additional image conditioning.
    effects = _facts(shot.effects)
    evidence = "Reviewed" if reviewed else "Planned, unverified"
    for key, value in _shot_conditions(shot, state).items():
        if key in effects:
            # A changed fact is a transition, never an instruction to keep the old value.
            # Only explicit preconditions constrain the opening of this shot.
            lines.append(f"{evidence} starting condition: {key} = {value}.")
        else:
            lines.append(f"{evidence} continuity to preserve: {key} = {value}.")
    lines.extend(
        f"Required outcome by the end of this shot (rendering goal): {key} = {value}."
        for key, value in effects.items()
    )
    relevant = set(_shot_conditions(shot, state).items()) | set(effects.items())
    lines.extend(
        f"Observable meaning of {item.key} = {item.value}: {item.definition}"
        for item in definitions
        if (item.key, item.value) in relevant
    )
    lines.append(
        "No generated captions, lettering, narration, or spoken dialogue; authored in edit."
    )
    if shot.reference_names is not None:
        # Prose and facts can mention an entity without activating its image.
        # Only the explicit conditioning roster contributes @mentions.
        lines = [
            re.sub(r"(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_-]*)", r"\1", line) for line in lines
        ]
        if shot.reference_names:
            lines.append(
                "Active visual references: "
                + ", ".join(f"@{name}" for name in shot.reference_names)
            )
    return "\n".join(lines)


def replace_shot(plan: FilmPlan, index: int, replacement: FilmShot) -> FilmPlan:
    """Create a revised plan; previous take digests will fail until reviewed again."""
    if not 0 <= index < len(plan.shots):
        raise ValueError("shot index out of range")
    shots = (*plan.shots[:index], replacement, *plan.shots[index + 1 :])
    return replace(plan, shots=shots).validate()


def subtitle_srt(plan: FilmPlan) -> str:
    """Produce exact, monotonic subtitle timings from authored dialogue, without translation."""
    plan.validate()
    rows: list[tuple[int, int, str]] = []
    offset = 0
    for shot in plan.shots:
        for cue in shot.dialogue:
            rows.append((offset + cue.start_ms, offset + cue.end_ms, cue.text))
        offset += shot.duration_ms
    rows.sort()

    def timestamp(ms: int) -> str:
        seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    return "\n".join(
        f"{index}\n{timestamp(start)} --> {timestamp(end)}\n{text}\n"
        for index, (start, end, text) in enumerate(rows, 1)
    )
