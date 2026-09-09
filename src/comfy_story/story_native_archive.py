"""Immutable reference archives, shot evidence, and generation receipts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Self

from comfy_story.story_contracts import (
    ComfyStoryStateRef,
    StoryLibrary,
    canonical_story_json,
    decode_canonical_story_object,
)
from comfy_story.story_product_contracts import (
    GuideBinding,
    NativeRGBObservation,
    StoryEvidenceRecord,
    StoryProductState,
)
from comfy_story.story_store import StoryProjectStore, _read_regular

_FORMAT = "comfy-story-native-reference-revision-v1"
_MEMORY_FORMAT = "comfy-story-native-reference-memory-v1"
NATIVE_REFERENCE_RUNTIME_SHA256 = hashlib.sha256(
    canonical_story_json(
        {
            "format": _FORMAT,
            "observations": "comfy-story-rgb-observation-v1",
            "capacity": 128,
            "recall": "declared-exact-evidence",
        }
    )
).hexdigest()
_MAX_JSON_BYTES = 2 * 1024 * 1024


def _digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class NativeReferenceReceipt:
    """Generation provenance bound to the configured runtime."""

    execution_sha256: str
    prompt_sha256: str
    selected_evidence_ids: tuple[str, ...]
    guide_bindings: tuple[GuideBinding, ...]
    runtime_sha256: str = NATIVE_REFERENCE_RUNTIME_SHA256

    def validate(self) -> Self:
        _digest(self.execution_sha256, "execution_sha256")
        _digest(self.prompt_sha256, "prompt_sha256")
        if self.runtime_sha256 != NATIVE_REFERENCE_RUNTIME_SHA256:
            raise ValueError("native reference runtime identity changed")
        if (
            not isinstance(self.selected_evidence_ids, tuple)
            or len(self.selected_evidence_ids) > 2
            or any(not isinstance(x, str) or not x for x in self.selected_evidence_ids)
            or len(set(self.selected_evidence_ids)) != len(self.selected_evidence_ids)
        ):
            raise ValueError("native recall requires at most two distinct evidence IDs")
        if not isinstance(self.guide_bindings, tuple) or not 1 <= len(self.guide_bindings) <= 9:
            raise ValueError("native receipt requires one to nine guide bindings")
        for binding in self.guide_bindings:
            binding.validate()
            if binding.role == "inclusive-core":
                raise ValueError("native reference archives do not contain a dense scene core")
        if len({b.role for b in self.guide_bindings}) != len(self.guide_bindings):
            raise ValueError("native receipt guide roles must be unique")
        return self

    def to_json(self) -> bytes:
        return canonical_story_json(self.validate())

    @classmethod
    def from_json(cls, encoded: bytes) -> NativeReferenceReceipt:
        data = decode_canonical_story_object(encoded, field="native reference receipt")
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("native receipt has missing or unknown fields")
        selected = data["selected_evidence_ids"]
        bindings = data["guide_bindings"]
        if not isinstance(selected, list) or any(not isinstance(x, str) for x in selected):
            raise ValueError("native selected evidence must be an array of strings")
        if not isinstance(bindings, list):
            raise ValueError("native guides must be an array")
        return cls(
            _digest(data["execution_sha256"], "execution_sha256"),
            _digest(data["prompt_sha256"], "prompt_sha256"),
            tuple(selected),
            tuple(GuideBinding.from_mapping(x) for x in bindings),
            _digest(data["runtime_sha256"], "runtime_sha256"),
        ).validate()


@dataclass(frozen=True, slots=True)
class NativeArchiveRevision:
    state: ComfyStoryStateRef
    library: StoryLibrary
    product_state: StoryProductState
    last_frame_png: bytes
    shot_metadata: dict[str, object]
    generation_receipt: NativeReferenceReceipt

    def require_product_state(self) -> StoryProductState:
        return self.product_state.validate()


class NativeReferenceArchive:
    """Publish immutable native revisions using the existing atomic asset store."""

    def __init__(self, assets: StoryProjectStore) -> None:
        self.assets = assets

    def _json_asset(self, digest: object, field: str) -> bytes:
        expected = _digest(digest, field)
        encoded = _read_regular(
            self.assets.root / "assets" / "sha256" / expected, field, _MAX_JSON_BYTES
        )
        if hashlib.sha256(encoded).hexdigest() != expected:
            raise ValueError(f"{field} SHA-256 changed")
        return encoded

    def _manifest(self, state: ComfyStoryStateRef) -> dict[str, object]:
        state.validate()
        data = decode_canonical_story_object(
            self._json_asset(state.revision_sha256, "native revision"), field="native revision"
        )
        if set(data) != {
            "format",
            "runtime_sha256",
            "project_id",
            "branch_id",
            "parent",
            "library_sha256",
            "memory_manifest_sha256",
            "last_frame_sha256",
            "model_configuration_sha256",
            "shot_count",
            "receipt_sha256",
            "shot_metadata",
        }:
            raise ValueError("native revision has missing or unknown fields")
        if data["format"] != _FORMAT or data["runtime_sha256"] != NATIVE_REFERENCE_RUNTIME_SHA256:
            raise ValueError("revision is not compatible with the native reference runtime")
        parent = data["parent"]
        parent_ref = (
            None if parent is None else ComfyStoryStateRef.from_json(canonical_story_json(parent))
        )
        expected = {
            "project_id": state.project_id,
            "branch_id": state.branch_id,
            "library_sha256": state.library_sha256,
            "memory_manifest_sha256": state.memory_manifest_sha256,
            "last_frame_sha256": state.last_frame_sha256,
            "model_configuration_sha256": state.model_configuration_sha256,
            "shot_count": state.shot_count,
        }
        if any(data[k] != v for k, v in expected.items()):
            raise ValueError("native revision does not match its state reference")
        if (
            None if parent_ref is None else parent_ref.revision_sha256
        ) != state.parent_revision_sha256 or state.shot_count != (
            1 if parent_ref is None else parent_ref.shot_count + 1
        ):
            raise ValueError("native revision has inconsistent parent coverage")
        if parent_ref is not None and (
            parent_ref.project_id != state.project_id
            or parent_ref.library_sha256 != state.library_sha256
            or parent_ref.model_configuration_sha256 != state.model_configuration_sha256
        ):
            raise ValueError("native revision changed its inherited project identity")
        return data

    def load_revision(self, revision_sha256: str) -> NativeArchiveRevision:
        """Resolve an exact native revision digest, then authenticate its full state and lineage."""
        data = decode_canonical_story_object(
            self._json_asset(revision_sha256, "native revision"), field="native revision"
        )
        project, branch, count = (
            data.get("project_id"),
            data.get("branch_id"),
            data.get("shot_count"),
        )
        if not isinstance(project, str) or not isinstance(branch, str) or type(count) is not int:
            raise ValueError("native revision requires project, branch and integer shot count")
        parent = data.get("parent")
        parent_ref = (
            None if parent is None else ComfyStoryStateRef.from_json(canonical_story_json(parent))
        )
        configuration = _digest(data.get("model_configuration_sha256"), "model configuration")
        state = ComfyStoryStateRef(
            project,
            branch,
            None if parent_ref is None else parent_ref.revision_sha256,
            _digest(revision_sha256, "revision"),
            _digest(data.get("library_sha256"), "library"),
            _digest(data.get("memory_manifest_sha256"), "memory"),
            count,
            count,
            _digest(data.get("last_frame_sha256"), "last frame"),
            configuration,
        ).validate()
        return self.load(state, model_configuration_sha256=configuration)

    def load(
        self, state: ComfyStoryStateRef, *, model_configuration_sha256: str
    ) -> NativeArchiveRevision:
        if state.model_configuration_sha256 != _digest(
            model_configuration_sha256, "model_configuration_sha256"
        ):
            raise ValueError("native revision uses a different model configuration")
        data = self._manifest(state)
        # Authenticate every ancestor manifest without recursively decoding all videos.
        cursor = data
        for _ in range(state.shot_count - 1):
            parent = ComfyStoryStateRef.from_json(canonical_story_json(cursor["parent"]))
            cursor = self._manifest(parent)
        if cursor["parent"] is not None:
            raise ValueError("native revision ancestry exceeds its shot count")
        library = StoryLibrary.from_json(self._json_asset(state.library_sha256, "native library"))
        memory = decode_canonical_story_object(
            self._json_asset(state.memory_manifest_sha256, "native memory"), field="native memory"
        )
        if set(memory) != {"format", "product_sha256"} or memory["format"] != _MEMORY_FORMAT:
            raise ValueError("native memory manifest is unsupported")
        product = StoryProductState.from_json(self._json_asset(memory["product_sha256"], "canon"))
        self._validate_product(product, library, state.shot_count)
        receipt = NativeReferenceReceipt.from_json(
            self._json_asset(data["receipt_sha256"], "native receipt")
        )
        metadata = data["shot_metadata"]
        if not isinstance(metadata, dict) or any(not isinstance(k, str) for k in metadata):
            raise ValueError("native shot metadata must be a string-keyed object")
        self._validate_metadata(metadata, receipt, product)
        for digest in {ref.media_sha256 for ref in library.references} | {
            record.asset_sha256 for record in product.evidence_records
        }:
            self.assets.load_asset(digest)
        if metadata["last_frame_full_sha256"] != state.last_frame_sha256:
            raise ValueError("native last frame does not match its metadata")
        prior_ids = {
            e.evidence_id
            for e in product.evidence_records
            if e.source_shot_index is not None and e.source_shot_index < state.shot_count - 1
        }
        if not set(receipt.selected_evidence_ids) <= prior_ids:
            raise ValueError("native receipt recalls evidence unavailable before this shot")
        last_frame = self.assets.load_asset(state.last_frame_sha256)
        return NativeArchiveRevision(state, library, product, last_frame, metadata, receipt)

    @staticmethod
    def _validate_product(product: StoryProductState, library: StoryLibrary, count: int) -> None:
        product.validate()
        library.validate()
        if not 1 <= count <= 128 or tuple(
            p.shot_index for p in product.observation_packets
        ) != tuple(range(count)):
            raise ValueError("native observations must cover each shot exactly once")
        if product.observation_packets[0].timeline_start_ns != 0:
            raise ValueError("native observation timeline must start at zero")
        if any(
            not isinstance(o, NativeRGBObservation)
            for p in product.observation_packets
            for o in p.observations
        ):
            raise ValueError("Story archives require native RGB observations")
        records = {
            o.evidence_id: StoryEvidenceRecord.from_observation(p, o)
            for p in product.observation_packets
            for o in p.observations
        }
        if records != {e.evidence_id: e for e in product.evidence_records}:
            raise ValueError("native evidence must exactly match its RGB observations")
        expected = {r.name.casefold(): (r.name, r.media_sha256) for r in library.references}
        actual = {
            e.entity_id: (e.reference_name, e.baseline_sha256) for e in product.canon.entities
        }
        if actual != expected:
            raise ValueError("native canon does not match its immutable library")

    @staticmethod
    def _validate_metadata(
        metadata: dict[str, object], receipt: NativeReferenceReceipt, product: StoryProductState
    ) -> None:
        for key in ("video_sha256", "last_frame_tensor_sha256", "last_frame_full_sha256"):
            _digest(metadata.get(key), key)
        if metadata.get("execution_sha256") != receipt.execution_sha256:
            raise ValueError("native metadata does not bind its generation receipt")
        if metadata["video_sha256"] != product.observation_packets[-1].source_video_sha256:
            raise ValueError("native closing packet does not bind its video")

    def publish(
        self,
        *,
        project_id: str,
        branch_id: str,
        parent: ComfyStoryStateRef | None,
        library: StoryLibrary,
        product_state: StoryProductState,
        model_configuration_sha256: str,
        last_frame_png: bytes,
        shot_metadata: dict[str, object],
        receipt: NativeReferenceReceipt,
    ) -> ComfyStoryStateRef:
        if parent is not None:
            parent.validate()
        count = 1 if parent is None else parent.shot_count + 1
        self._validate_product(product_state, library, count)
        receipt.validate()
        self._validate_metadata(shot_metadata, receipt, product_state)
        if parent is not None:
            previous = self.load(parent, model_configuration_sha256=model_configuration_sha256)
            if previous.library != library or parent.project_id != project_id:
                raise ValueError("native append changed its project or library")
            if product_state.observation_packets[:-1] != previous.product_state.observation_packets:
                raise ValueError("native append changed historical observations")
            prior = {e.evidence_id: e for e in product_state.evidence_records}
            if any(prior.get(e.evidence_id) != e for e in previous.product_state.evidence_records):
                raise ValueError("native append changed historical evidence")
        selected = set(receipt.selected_evidence_ids)
        prior_ids = (
            set()
            if parent is None
            else {e.evidence_id for e in previous.product_state.evidence_records}
        )
        if not selected <= prior_ids:
            raise ValueError("native receipt recalls evidence unavailable before this shot")
        for key in (
            "video_sha256",
            "last_frame_tensor_sha256",
            "last_frame_full_sha256",
            "execution_sha256",
        ):
            self.assets.load_asset(_digest(shot_metadata[key], key))
        if hashlib.sha256(last_frame_png).hexdigest() != shot_metadata["last_frame_full_sha256"]:
            raise ValueError("native last frame does not match its metadata")
        for digest in {r.media_sha256 for r in library.references} | {
            e.asset_sha256 for e in product_state.evidence_records
        }:
            self.assets.load_asset(digest)
        library_json, product_json = library.to_json(), product_state.to_json()
        if max(len(library_json), len(product_json)) > _MAX_JSON_BYTES:
            raise ValueError("native library or evidence metadata exceeds its byte limit")
        library_sha = self.assets.put_asset(library_json)
        product_sha = self.assets.put_asset(product_json)
        memory_sha = self.assets.put_asset(
            canonical_story_json({"format": _MEMORY_FORMAT, "product_sha256": product_sha})
        )
        last_sha = self.assets.put_asset(last_frame_png)
        receipt_sha = self.assets.put_asset(receipt.to_json())
        data = {
            "format": _FORMAT,
            "runtime_sha256": NATIVE_REFERENCE_RUNTIME_SHA256,
            "project_id": project_id,
            "branch_id": branch_id,
            "parent": parent,
            "library_sha256": library_sha,
            "memory_manifest_sha256": memory_sha,
            "last_frame_sha256": last_sha,
            "model_configuration_sha256": model_configuration_sha256,
            "shot_count": count,
            "receipt_sha256": receipt_sha,
            "shot_metadata": shot_metadata,
        }
        encoded = canonical_story_json(data)
        if len(encoded) > _MAX_JSON_BYTES:
            raise ValueError("native revision metadata exceeds its byte limit")
        revision_sha = self.assets.put_asset(encoded)
        state = ComfyStoryStateRef(
            project_id,
            branch_id,
            None if parent is None else parent.revision_sha256,
            revision_sha,
            library_sha,
            memory_sha,
            count,
            count,
            last_sha,
            model_configuration_sha256,
        ).validate()
        self.load(state, model_configuration_sha256=model_configuration_sha256)
        return state
