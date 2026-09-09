"""Immutable public contracts for Comfy Story projects and Comfy state links."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from typing import Self, cast

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_PROJECT_NAME = re.compile(r"^[^/\\\x00-\x1f]{1,96}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_MENTION = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_-]{0,63})")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ASSET_LOCATOR = re.compile(r"^comfy-story://assets/sha256/([0-9a-f]{64})$")


class ShotIntent(StrEnum):
    """Value-oriented action performed by one Comfy Story node."""

    START_STORY = "Start Story"
    CONTINUE_THIS_SHOT = "Continue This Shot"
    NEXT_SHOT = "Next Shot"
    NEW_SCENE = "New Scene"


class ReferenceRole(StrEnum):
    """Customer-facing role for one named reference asset."""

    CHARACTER = "Character"
    PRODUCT = "Product"
    PROP = "Prop"
    LOCATION = "Location"
    COSTUME = "Costume"
    STYLE = "Style"


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a portable identifier")
    return value


def _wire(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _wire(getattr(value, field.name))
            for field in fields(value)
            # New optional sidecar fields can preserve historical canonical byte identities.
            if not field.metadata.get("omit_wire_default", False)
            or getattr(value, field.name) != field.default
        }
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Comfy Story canonical mappings require string keys")
        return {key: _wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_wire(item) for item in value]
    if isinstance(value, tuple):
        return [_wire(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported Comfy Story canonical value: {type(value).__name__}")


def canonical_story_json(value: object) -> bytes:
    """Encode one story value as canonical UTF-8 JSON."""
    return json.dumps(
        _wire(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def decode_canonical_story_object(
    encoded: bytes, *, field: str = "story payload"
) -> dict[str, object]:
    if not isinstance(encoded, bytes) or not encoded:
        raise ValueError(f"{field} must be nonempty UTF-8 bytes")
    try:
        value = json.loads(encoded, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} must be valid JSON") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    if canonical_story_json(value) != encoded:
        raise ValueError(f"{field} must use canonical JSON")
    return cast(dict[str, object], value)


def _exact_fields(value: dict[str, object], expected: frozenset[str], field: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} has missing or unknown fields")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be an array of strings")
    return tuple(cast(list[str], value))


@dataclass(frozen=True, slots=True)
class StoryReference:
    """One immutable, named image or video reference in a story library."""

    name: str
    role: ReferenceRole
    note: str
    media_sha256: str
    evidence_locator: str
    keyframe_sha256s: tuple[str, ...]
    preprocessing_sha256: str

    def validate(self) -> Self:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError("reference name must start with a letter and be portable")
        if not isinstance(self.role, ReferenceRole):
            raise ValueError("reference role is unsupported")
        if not isinstance(self.note, str) or len(self.note) > 500:
            raise ValueError("reference note must contain at most 500 characters")
        digest = _sha256(self.media_sha256, "media_sha256")
        if (
            not isinstance(self.evidence_locator, str)
            or (match := _ASSET_LOCATOR.fullmatch(self.evidence_locator)) is None
            or match.group(1) != digest
        ):
            raise ValueError("evidence_locator must bind the reference media digest")
        if not isinstance(self.keyframe_sha256s, tuple) or not self.keyframe_sha256s:
            raise ValueError("reference must contain at least one ordered keyframe digest")
        for keyframe in self.keyframe_sha256s:
            _sha256(keyframe, "keyframe_sha256")
        _sha256(self.preprocessing_sha256, "preprocessing_sha256")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> StoryReference:
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ValueError("StoryReference must be a JSON object")
        data = cast(dict[str, object], value)
        _exact_fields(
            data,
            frozenset(
                {
                    "name",
                    "role",
                    "note",
                    "media_sha256",
                    "evidence_locator",
                    "keyframe_sha256s",
                    "preprocessing_sha256",
                }
            ),
            "StoryReference",
        )
        try:
            role = ReferenceRole(_string(data["role"], "role"))
        except ValueError as error:
            raise ValueError("reference role is unsupported") from error
        return cls(
            _string(data["name"], "name"),
            role,
            _string(data["note"], "note"),
            _string(data["media_sha256"], "media_sha256"),
            _string(data["evidence_locator"], "evidence_locator"),
            _string_tuple(data["keyframe_sha256s"], "keyframe_sha256s"),
            _string(data["preprocessing_sha256"], "preprocessing_sha256"),
        ).validate()


@dataclass(frozen=True, slots=True)
class StoryLibrary:
    """Ordered, project-wide collection of named reference assets."""

    project_name: str
    references: tuple[StoryReference, ...]

    def validate(self) -> Self:
        if (
            not isinstance(self.project_name, str)
            or _PROJECT_NAME.fullmatch(self.project_name) is None
        ):
            raise ValueError(
                "project_name must be 1-96 printable characters without path separators"
            )
        if not isinstance(self.references, tuple) or not self.references:
            raise ValueError("Story Library must contain at least one reference")
        for reference in self.references:
            if not isinstance(reference, StoryReference):
                raise ValueError("Story Library entries must be StoryReference values")
            reference.validate()
        folded = tuple(reference.name.casefold() for reference in self.references)
        if len(folded) != len(set(folded)):
            raise ValueError("Story Library reference names must be unique, ignoring case")
        return self

    def resolve_references(
        self,
        prompt: str,
        *,
        limit: int = 9,
        require_mentions: bool = False,
    ) -> tuple[StoryReference, ...]:
        """Resolve unique names in prompt order within the requested image budget."""
        self.validate()
        if type(limit) is not int or not 1 <= limit <= 9:
            raise ValueError("H3 reference limit must be in range [1,9]")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("What happens next? must be nonempty")
        lookup = {reference.name.casefold(): reference for reference in self.references}
        selected: dict[str, StoryReference] = {}
        for match in _MENTION.finditer(prompt):
            name = match.group(1).casefold()
            if name not in lookup:
                raise ValueError(f"Unknown Story Library reference: @{match.group(1)}")
            selected.setdefault(name, lookup[name])
        if not selected and require_mentions:
            raise ValueError("Mention at least one Story Library reference with @name")
        if len(selected) > limit:
            raise ValueError(f"MiniMax H3 supports at most {limit} named references")
        return tuple(selected.values())

    def to_json(self) -> bytes:
        self.validate()
        return canonical_story_json(self)

    @classmethod
    def from_json(cls, encoded: bytes) -> StoryLibrary:
        data = decode_canonical_story_object(encoded, field="StoryLibrary")
        _exact_fields(data, frozenset({"project_name", "references"}), "StoryLibrary")
        references = data["references"]
        if not isinstance(references, list):
            raise ValueError("StoryLibrary.references must be an array")
        return cls(
            _string(data["project_name"], "project_name"),
            tuple(StoryReference.from_mapping(item) for item in references),
        ).validate()


@dataclass(frozen=True, slots=True)
class ComfyStoryStateRef:
    """Opaque, content-bound state passed through a ``COMFY_STORY`` socket."""

    project_id: str
    branch_id: str
    parent_revision_sha256: str | None
    revision_sha256: str
    library_sha256: str
    memory_manifest_sha256: str
    shot_count: int
    next_history_slot: int
    last_frame_sha256: str
    model_configuration_sha256: str

    def validate(self) -> Self:
        _identifier(self.project_id, "project_id")
        _identifier(self.branch_id, "branch_id")
        if self.parent_revision_sha256 is not None:
            _sha256(self.parent_revision_sha256, "parent_revision_sha256")
        for field in (
            "revision_sha256",
            "library_sha256",
            "memory_manifest_sha256",
            "last_frame_sha256",
            "model_configuration_sha256",
        ):
            _sha256(getattr(self, field), field)
        if type(self.shot_count) is not int or not 0 <= self.shot_count <= 128:
            raise ValueError("shot_count must be an integer in [0, 128]")
        if type(self.next_history_slot) is not int or self.next_history_slot != self.shot_count:
            raise ValueError("next_history_slot must equal shot_count")
        return self

    def to_json(self) -> bytes:
        self.validate()
        return canonical_story_json(self)

    @classmethod
    def from_json(cls, encoded: bytes) -> ComfyStoryStateRef:
        data = decode_canonical_story_object(encoded, field="ComfyStoryStateRef")
        expected = frozenset(field.name for field in fields(cls))
        _exact_fields(data, expected, "ComfyStoryStateRef")
        parent = data["parent_revision_sha256"]
        if parent is not None and not isinstance(parent, str):
            raise ValueError("parent_revision_sha256 must be a digest or null")
        shot_count = data["shot_count"]
        next_slot = data["next_history_slot"]
        if type(shot_count) is not int or type(next_slot) is not int:
            raise ValueError("story coverage fields must be integers")
        return cls(
            _string(data["project_id"], "project_id"),
            _string(data["branch_id"], "branch_id"),
            parent,
            _string(data["revision_sha256"], "revision_sha256"),
            _string(data["library_sha256"], "library_sha256"),
            _string(data["memory_manifest_sha256"], "memory_manifest_sha256"),
            shot_count,
            next_slot,
            _string(data["last_frame_sha256"], "last_frame_sha256"),
            _string(data["model_configuration_sha256"], "model_configuration_sha256"),
        ).validate()


__all__ = (
    "ComfyStoryStateRef",
    "ReferenceRole",
    "ShotIntent",
    "StoryLibrary",
    "StoryReference",
    "canonical_story_json",
    "decode_canonical_story_object",
)
