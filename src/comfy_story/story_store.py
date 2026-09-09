"""Authenticated content-addressed storage for immutable Duet Story revisions."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

from comfy_story.cache import DuetXProductTree
from comfy_story.contracts import DuetXContract, tensor_sha256
from comfy_story.fusion import FusionSpec, MatrixSemigroupFusion
from comfy_story.snapshot import load_runtime_snapshot, save_runtime_snapshot
from comfy_story.story_contracts import (
    DuetStoryStateRef,
    StoryLibrary,
    canonical_story_json,
)
from comfy_story.story_memory import (
    StoryMemoryRoot,
    compose_inclusive_story_core,
    compose_story_blocks,
    story_evidence_reservoir,
)
from comfy_story.story_product_contracts import (
    CanonEntity,
    CanonPresence,
    ObservationKind,
    StoryCanon,
    StoryEvidenceRecord,
    StoryGenerationReceipt,
    StoryMemoryPolicy,
    StoryProductState,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION_FORMAT = "duet-story-revision-v1"
_REVISION_FORMAT_V2 = "duet-story-revision-v2"
_REVISION_FORMAT_V3 = "duet-story-revision-v3"
_MEMORY_FORMAT = "duet-story-memory-v1"
_MAX_JSON_BYTES = 1024 * 1024
_MAX_ASSET_BYTES = 4 * 1024 * 1024 * 1024
_MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a portable identifier")
    return value


def _mapping(value: object, field: str, expected: frozenset[str]) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    result = cast(dict[str, object], value)
    if set(result) != expected:
        raise ValueError(f"{field} has missing or unknown fields")
    return result


def _canonical_mapping(
    encoded: bytes, field: str, maximum: int = _MAX_JSON_BYTES
) -> dict[str, object]:
    if not encoded or len(encoded) > maximum:
        raise ValueError(f"{field} must be nonempty bounded JSON")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} must be valid JSON") from error
    if not isinstance(value, dict) or canonical_story_json(value) != encoded:
        raise ValueError(f"{field} must use canonical JSON")
    return cast(dict[str, object], value)


def _read_regular(path: Path, field: str, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{field} is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= maximum
            or before.st_nlink != 1
        ):
            raise ValueError(f"{field} must be a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            value = handle.read(maximum + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ) or len(value) != after.st_size:
            raise ValueError(f"{field} changed while it was read")
        return value
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        result = library.renamex_np(os.fsencode(source), os.fsencode(destination), 0x00000004)
    elif sys.platform.startswith("linux"):
        result = library.renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace directory publication unsupported")
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, os.strerror(number), destination)
        raise OSError(number, os.strerror(number), destination)


def _product_memory_root(
    blocks: tuple[DuetXProductTree, ...], fusion: MatrixSemigroupFusion
) -> StoryMemoryRoot:
    return StoryMemoryRoot(
        compose_inclusive_story_core(blocks, fusion),
        story_evidence_reservoir(blocks),
        (),
        sum(leaf is not None for block in blocks for leaf in block.leaves),
    )


@dataclass(frozen=True, slots=True)
class StoryGuideBundle:
    """Materialized core and exact exception images for one memory root."""

    core_png: bytes | None
    exception_pngs: tuple[bytes, ...]
    exception_fingerprints: tuple[str, ...]

    def validate(self) -> Self:
        if self.core_png is not None and (
            not isinstance(self.core_png, bytes) or not 0 < len(self.core_png) <= _MAX_ASSET_BYTES
        ):
            raise ValueError("guide core must be bounded nonempty bytes")
        if (
            not isinstance(self.exception_pngs, tuple)
            or not isinstance(self.exception_fingerprints, tuple)
            or not 1 <= len(self.exception_pngs) <= 2
            or len(self.exception_pngs) != len(self.exception_fingerprints)
        ):
            raise ValueError("guide exceptions must align with one or two fingerprints")
        for value in self.exception_pngs:
            if not isinstance(value, bytes) or not 0 < len(value) <= _MAX_ASSET_BYTES:
                raise ValueError("guide exception must be bounded nonempty bytes")
        for fingerprint in self.exception_fingerprints:
            _digest(fingerprint, "exception fingerprint")
        return self


@dataclass(frozen=True, slots=True)
class StoryCommitPayload:
    """Complete state required to publish one post-decode story revision."""

    project_id: str
    branch_id: str
    parent: DuetStoryStateRef | None
    library: StoryLibrary
    blocks: tuple[DuetXProductTree, ...]
    checkpoint_sha256: str
    model_configuration_sha256: str
    last_frame_png: bytes
    story_context_png: bytes
    shot_metadata: dict[str, object]
    guide_bundle: StoryGuideBundle | None = None
    product_state: StoryProductState | None = None
    generation_receipt: StoryGenerationReceipt | None = None

    def validate(self) -> Self:
        _identifier(self.project_id, "project_id")
        _identifier(self.branch_id, "branch_id")
        if self.parent is not None:
            self.parent.validate()
            if self.parent.project_id != self.project_id:
                raise ValueError("parent project_id does not match the new revision")
        self.library.validate()
        _digest(self.checkpoint_sha256, "checkpoint_sha256")
        _digest(self.model_configuration_sha256, "model_configuration_sha256")
        for value, field in (
            (self.last_frame_png, "last_frame_png"),
            (self.story_context_png, "story_context_png"),
        ):
            if not isinstance(value, bytes) or not 0 < len(value) <= _MAX_ASSET_BYTES:
                raise ValueError(f"{field} must be nonempty bounded bytes")
        if not isinstance(self.shot_metadata, dict) or any(
            not isinstance(key, str) for key in self.shot_metadata
        ):
            raise ValueError("shot_metadata must be a string-keyed mapping")
        canonical_story_json(self.shot_metadata)
        if self.guide_bundle is not None:
            self.guide_bundle.validate()
        if (self.product_state is None) is not (self.generation_receipt is None):
            raise ValueError("product state and generation receipt must be published together")
        if self.product_state is not None and self.generation_receipt is not None:
            self.product_state.validate()
            self.generation_receipt.validate()
            if self.guide_bundle is not None:
                raise ValueError("v3 product revisions replace the legacy guide bundle")
        return self


@dataclass(frozen=True, slots=True)
class LoadedStoryRevision:
    """A fully authenticated revision ready for story generation."""

    state: DuetStoryStateRef
    library: StoryLibrary
    blocks: tuple[DuetXProductTree, ...]
    memory: StoryMemoryRoot
    last_frame_png: bytes
    story_context_png: bytes
    shot_metadata: dict[str, object]
    guide_bundle: StoryGuideBundle | None = None
    product_state: StoryProductState | None = None
    generation_receipt: StoryGenerationReceipt | None = None

    def require_guide_bundle(self) -> StoryGuideBundle:
        """Return the authenticated guide bundle required by native MiniMax memory."""
        if self.guide_bundle is None:
            raise ValueError("story revision does not contain a materialized guide bundle")
        return self.guide_bundle.validate()

    def require_product_state(self) -> StoryProductState:
        """Return authenticated selective-recall state from a v3 revision."""
        if self.product_state is None:
            raise ValueError("story revision does not contain Living Canon product state")
        return self.product_state.validate()


class StoryProjectStore:
    """Persist story assets and revisions beneath one configured regular root."""

    def __init__(self, root: Path) -> None:
        if sys.platform != "darwin" and not sys.platform.startswith("linux"):
            raise ValueError(
                "Comfy Story currently requires Linux or macOS for atomic revision storage"
            )
        if not isinstance(root, Path) or not root.is_absolute() or root.is_symlink():
            raise ValueError("story project root must be an absolute regular directory")
        root.mkdir(parents=True, exist_ok=True)
        if root.resolve() != root or not root.is_dir():
            raise ValueError("story project root must be an absolute regular directory")
        self._root = root
        self._assets = root / "assets" / "sha256"
        self._snapshots = root / "snapshots" / "sha256"
        self._memories = root / "memories" / "sha256"
        self._libraries = root / "libraries" / "sha256"
        self._products = root / "products" / "sha256"
        self._receipts = root / "receipts" / "sha256"
        self._revisions = root / "revisions" / "sha256"
        self._staging = root / "staging"
        for directory in (
            self._assets,
            self._snapshots,
            self._memories,
            self._libraries,
            self._products,
            self._receipts,
            self._revisions,
            self._staging,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _put_bytes(self, directory: Path, value: bytes, *, suffix: str = "") -> str:
        digest = _sha256_bytes(value)
        target = directory / f"{digest}{suffix}"
        if target.exists():
            if _read_regular(target, "content-addressed object", len(value)) != value:
                raise ValueError("refusing conflicting content-addressed object")
            return digest
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if _read_regular(target, "content-addressed object", len(value)) != value:
                    raise ValueError("refusing conflicting content-addressed object") from None
            _fsync_directory(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return digest

    def put_asset(self, value: bytes) -> str:
        """Publish one immutable asset and return its content digest."""
        if not isinstance(value, bytes) or not 0 < len(value) <= _MAX_ASSET_BYTES:
            raise ValueError("story asset must be nonempty bounded bytes")
        return self._put_bytes(self._assets, value)

    def load_asset(self, digest: str) -> bytes:
        """Load and authenticate one public content-addressed Story asset."""
        return self._asset(digest, "story asset")

    def verified_asset_path(self, digest: str) -> Path:
        """Authenticate immutable media in bounded memory before HTTP range serving."""
        expected = _digest(digest, "story asset")
        path = self._assets / expected
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise ValueError("story asset is unavailable") from error
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or not 0 < before.st_size <= _MAX_ASSET_BYTES
                or before.st_nlink != 1
            ):
                raise ValueError("story asset must be a bounded regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                observed = hashlib.file_digest(handle, "sha256").hexdigest()
            after = os.fstat(descriptor)
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, field) != getattr(after, field) for field in fields):
                raise ValueError("story asset changed while it was read")
            if observed != expected:
                raise ValueError("story asset SHA-256 changed")
            return path
        finally:
            os.close(descriptor)

    def _asset(self, digest: str, field: str) -> bytes:
        expected = _digest(digest, field)
        value = _read_regular(self._assets / expected, field, _MAX_ASSET_BYTES)
        if _sha256_bytes(value) != expected:
            raise ValueError(f"{field} SHA-256 changed")
        return value

    def _snapshot_roster(
        self,
        staging: Path,
        payload: StoryCommitPayload,
    ) -> tuple[str, ...]:
        digests: list[str] = []
        for index, block in enumerate(payload.blocks):
            temporary = staging / f"block-{index:02d}.pt"
            save_runtime_snapshot(
                temporary,
                block,
                model_checkpoint_sha256=payload.checkpoint_sha256,
            )
            encoded = _read_regular(temporary, "staged story snapshot", _MAX_SNAPSHOT_BYTES)
            digest = self._put_bytes(self._snapshots, encoded, suffix=".pt")
            restored = load_runtime_snapshot(
                self._snapshots / f"{digest}.pt",
                expected_contract=block.contract,
                expected_model_checkpoint_sha256=payload.checkpoint_sha256,
            )
            if restored.leaves != block.leaves:
                raise ValueError("story snapshot round trip changed authoritative leaves")
            digests.append(digest)
        return tuple(digests)

    def publish(self, payload: StoryCommitPayload) -> DuetStoryStateRef:
        """Atomically publish one verified child revision or return its identical existing state."""
        payload.validate()
        if len(payload.blocks) != 16:
            raise ValueError("story commit requires exactly sixteen blocks")
        if payload.parent is not None:
            parent = self.load(
                payload.parent, payload.blocks[0].contract, payload.checkpoint_sha256
            )
            if parent.state.model_configuration_sha256 != payload.model_configuration_sha256:
                raise ValueError("parent model configuration does not match the child")
        else:
            parent = None
        operator_size = payload.blocks[0].root.dense_operator.shape[-1]
        fusion = MatrixSemigroupFusion(
            payload.blocks[0].contract.latent_channels,
            FusionSpec(),
            operator_size=operator_size,
        )
        memory = (
            compose_story_blocks(payload.blocks, fusion)
            if payload.product_state is None
            else _product_memory_root(payload.blocks, fusion)
        )
        expected_shots = 1 if parent is None else parent.state.shot_count + 1
        if memory.occupied_shots != expected_shots:
            raise ValueError("story commit must append exactly one shot to its parent")
        guide_bundle = payload.guide_bundle
        if guide_bundle is not None:
            guide_bundle.validate()
            if payload.blocks[0].contract.format != "duet-x-minimax-h3-v1":
                raise ValueError("materialized guide bundles require MiniMax H3 memory")
            expected_fingerprints = tuple(item.fingerprint() for item in memory.exceptions)
            if guide_bundle.exception_fingerprints != expected_fingerprints:
                raise ValueError("guide exception fingerprints do not match memory Top-K")
            needs_core = memory.occupied_shots > len(memory.exceptions)
            if (guide_bundle.core_png is not None) is not needs_core:
                raise ValueError("guide core presence does not match memory occupancy")
            for image, item in zip(
                guide_bundle.exception_pngs,
                memory.exceptions,
                strict=True,
            ):
                if _sha256_bytes(image) != item.provenance.raw.content_sha256:
                    raise ValueError("guide exception asset does not match raw evidence")

        staging = Path(tempfile.mkdtemp(prefix=".duet-story-", dir=self._staging))
        try:
            snapshot_digests = self._snapshot_roster(staging, payload)
            library_bytes = payload.library.to_json()
            library_sha256 = self._put_bytes(self._libraries, library_bytes, suffix=".json")
            for reference in payload.library.references:
                self._asset(reference.media_sha256, "reference media")
                for digest in reference.keyframe_sha256s:
                    self._asset(digest, "reference keyframe")
            last_frame_sha256 = self.put_asset(payload.last_frame_png)
            context_sha256 = self.put_asset(payload.story_context_png)
            guide_core_sha256 = (
                None
                if guide_bundle is None or guide_bundle.core_png is None
                else self.put_asset(guide_bundle.core_png)
            )
            guide_exception_sha256s = (
                []
                if guide_bundle is None
                else [self.put_asset(value) for value in guide_bundle.exception_pngs]
            )
            product_state_sha256 = (
                None
                if payload.product_state is None
                else self._put_bytes(
                    self._products,
                    payload.product_state.to_json(),
                    suffix=".json",
                )
            )
            generation_receipt_sha256 = (
                None
                if payload.generation_receipt is None
                else self._put_bytes(
                    self._receipts,
                    payload.generation_receipt.to_json(),
                    suffix=".json",
                )
            )
            memory_data: dict[str, object] = {
                "block_snapshot_sha256s": list(snapshot_digests),
                "core_operator_sha256": tensor_sha256(memory.core_operator),
                "exception_fingerprints": [item.fingerprint() for item in memory.exceptions],
                "format": _MEMORY_FORMAT,
                "occupied_shots": memory.occupied_shots,
            }
            memory_sha256 = self._put_bytes(
                self._memories,
                canonical_story_json(memory_data),
                suffix=".json",
            )
            revision_data: dict[str, object] = {
                "branch_id": payload.branch_id,
                "checkpoint_sha256": payload.checkpoint_sha256,
                "format": _REVISION_FORMAT,
                "last_frame_sha256": last_frame_sha256,
                "library_sha256": library_sha256,
                "memory_manifest_sha256": memory_sha256,
                "model_configuration_sha256": payload.model_configuration_sha256,
                "next_history_slot": memory.occupied_shots,
                "parent_revision_sha256": (
                    None if payload.parent is None else payload.parent.revision_sha256
                ),
                "project_id": payload.project_id,
                "shot_count": memory.occupied_shots,
                "shot_metadata": payload.shot_metadata,
                "story_context_sha256": context_sha256,
            }
            if guide_bundle is not None:
                revision_data.update(
                    {
                        "format": _REVISION_FORMAT_V2,
                        "guide_core_sha256": guide_core_sha256,
                        "guide_exception_fingerprints": list(guide_bundle.exception_fingerprints),
                        "guide_exception_sha256s": guide_exception_sha256s,
                    }
                )
            if product_state_sha256 is not None and generation_receipt_sha256 is not None:
                revision_data.update(
                    {
                        "format": _REVISION_FORMAT_V3,
                        "generation_receipt_sha256": generation_receipt_sha256,
                        "product_state_sha256": product_state_sha256,
                    }
                )
            manifest = canonical_story_json(revision_data)
            revision_sha256 = _sha256_bytes(manifest)
            state = DuetStoryStateRef(
                payload.project_id,
                payload.branch_id,
                None if payload.parent is None else payload.parent.revision_sha256,
                revision_sha256,
                library_sha256,
                memory_sha256,
                memory.occupied_shots,
                memory.occupied_shots,
                last_frame_sha256,
                payload.model_configuration_sha256,
            ).validate()
            (staging / "manifest.json").write_bytes(manifest)
            with (staging / "manifest.json").open("rb") as handle:
                os.fsync(handle.fileno())
            _fsync_directory(staging)
            target = self._revisions / revision_sha256
            try:
                _rename_directory_noreplace(staging, target)
            except FileExistsError:
                existing = _read_regular(
                    target / "manifest.json", "story revision manifest", _MAX_JSON_BYTES
                )
                if existing != manifest:
                    raise ValueError("refusing conflicting story revision bytes") from None
            _fsync_directory(self._revisions)
            self.load(state, payload.blocks[0].contract, payload.checkpoint_sha256)
            return state
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def load(
        self,
        state: DuetStoryStateRef,
        contract: DuetXContract,
        checkpoint_sha256: str,
    ) -> LoadedStoryRevision:
        """Authenticate every object reachable from one opaque story state."""
        state.validate()
        contract.validate()
        _digest(checkpoint_sha256, "checkpoint_sha256")
        revision_bytes = _read_regular(
            self._revisions / state.revision_sha256 / "manifest.json",
            "story revision manifest",
            _MAX_JSON_BYTES,
        )
        if _sha256_bytes(revision_bytes) != state.revision_sha256:
            raise ValueError("story revision manifest SHA-256 changed")
        revision_mapping = _canonical_mapping(revision_bytes, "story revision manifest")
        revision_format = revision_mapping.get("format")
        revision_fields = {
            "branch_id",
            "checkpoint_sha256",
            "format",
            "last_frame_sha256",
            "library_sha256",
            "memory_manifest_sha256",
            "model_configuration_sha256",
            "next_history_slot",
            "parent_revision_sha256",
            "project_id",
            "shot_count",
            "shot_metadata",
            "story_context_sha256",
        }
        if revision_format == _REVISION_FORMAT_V2:
            revision_fields.update(
                {
                    "guide_core_sha256",
                    "guide_exception_fingerprints",
                    "guide_exception_sha256s",
                }
            )
        if revision_format == _REVISION_FORMAT_V3:
            revision_fields.update(
                {
                    "generation_receipt_sha256",
                    "product_state_sha256",
                }
            )
        revision = _mapping(
            revision_mapping,
            "story revision manifest",
            frozenset(revision_fields),
        )
        if (
            revision["format"] not in {_REVISION_FORMAT, _REVISION_FORMAT_V2, _REVISION_FORMAT_V3}
            or revision["checkpoint_sha256"] != checkpoint_sha256
        ):
            raise ValueError("story revision runtime contract changed")
        shot_count = revision["shot_count"]
        next_slot = revision["next_history_slot"]
        if type(shot_count) is not int or type(next_slot) is not int:
            raise ValueError("story revision coverage must be integral")
        observed = DuetStoryStateRef(
            cast(str, revision["project_id"]),
            cast(str, revision["branch_id"]),
            cast(str | None, revision["parent_revision_sha256"]),
            state.revision_sha256,
            cast(str, revision["library_sha256"]),
            cast(str, revision["memory_manifest_sha256"]),
            shot_count,
            next_slot,
            cast(str, revision["last_frame_sha256"]),
            cast(str, revision["model_configuration_sha256"]),
        ).validate()
        if observed != state:
            raise ValueError("story state does not match its authenticated revision")
        if (
            state.parent_revision_sha256 is not None
            and not (self._revisions / state.parent_revision_sha256 / "manifest.json").is_file()
        ):
            raise ValueError("story parent revision is unavailable")

        library_bytes = _read_regular(
            self._libraries / f"{state.library_sha256}.json",
            "Story Library manifest",
            _MAX_JSON_BYTES,
        )
        if _sha256_bytes(library_bytes) != state.library_sha256:
            raise ValueError("Story Library manifest SHA-256 changed")
        library = StoryLibrary.from_json(library_bytes)
        for reference in library.references:
            self._asset(reference.media_sha256, "reference media")
            for digest in reference.keyframe_sha256s:
                self._asset(digest, "reference keyframe")

        memory_bytes = _read_regular(
            self._memories / f"{state.memory_manifest_sha256}.json",
            "story memory manifest",
            _MAX_JSON_BYTES,
        )
        if _sha256_bytes(memory_bytes) != state.memory_manifest_sha256:
            raise ValueError("story memory manifest SHA-256 changed")
        memory_data = _mapping(
            _canonical_mapping(memory_bytes, "story memory manifest"),
            "story memory manifest",
            frozenset(
                {
                    "block_snapshot_sha256s",
                    "core_operator_sha256",
                    "exception_fingerprints",
                    "format",
                    "occupied_shots",
                }
            ),
        )
        snapshots = memory_data["block_snapshot_sha256s"]
        if (
            memory_data["format"] != _MEMORY_FORMAT
            or not isinstance(snapshots, list)
            or len(snapshots) != 16
        ):
            raise ValueError("story memory manifest block roster changed")
        blocks = tuple(
            load_runtime_snapshot(
                self._snapshots / f"{_digest(digest, 'snapshot digest')}.pt",
                expected_contract=contract,
                expected_model_checkpoint_sha256=checkpoint_sha256,
            )
            for digest in snapshots
        )
        operator_size = blocks[0].root.dense_operator.shape[-1]
        fusion = MatrixSemigroupFusion(
            contract.latent_channels,
            FusionSpec(),
            operator_size=operator_size,
        )
        memory = (
            _product_memory_root(blocks, fusion)
            if revision["format"] == _REVISION_FORMAT_V3
            else compose_story_blocks(blocks, fusion)
        )
        if (
            tensor_sha256(memory.core_operator) != memory_data["core_operator_sha256"]
            or [item.fingerprint() for item in memory.exceptions]
            != memory_data["exception_fingerprints"]
            or memory.occupied_shots != memory_data["occupied_shots"]
            or memory.occupied_shots != state.shot_count
        ):
            raise ValueError("story memory manifest does not match its snapshots")
        metadata = revision["shot_metadata"]
        if not isinstance(metadata, dict) or any(not isinstance(key, str) for key in metadata):
            raise ValueError("story shot metadata is malformed")
        guide_bundle: StoryGuideBundle | None = None
        if revision["format"] == _REVISION_FORMAT_V2:
            if contract.format != "duet-x-minimax-h3-v1":
                raise ValueError("materialized guide revision requires MiniMax H3 memory")
            core_digest = revision["guide_core_sha256"]
            exception_digests = revision["guide_exception_sha256s"]
            fingerprints = revision["guide_exception_fingerprints"]
            if core_digest is not None and not isinstance(core_digest, str):
                raise ValueError("guide core digest must be null or a SHA-256 string")
            if (
                not isinstance(exception_digests, list)
                or any(not isinstance(value, str) for value in exception_digests)
                or not isinstance(fingerprints, list)
                or any(not isinstance(value, str) for value in fingerprints)
            ):
                raise ValueError("guide exception manifest is malformed")
            guide_bundle = StoryGuideBundle(
                None if core_digest is None else self._asset(core_digest, "guide core"),
                tuple(self._asset(value, "guide exception") for value in exception_digests),
                tuple(fingerprints),
            ).validate()
            expected_fingerprints = tuple(item.fingerprint() for item in memory.exceptions)
            if guide_bundle.exception_fingerprints != expected_fingerprints:
                raise ValueError("guide exception fingerprints do not match memory Top-K")
            needs_core = memory.occupied_shots > len(memory.exceptions)
            if (guide_bundle.core_png is not None) is not needs_core:
                raise ValueError("guide core presence does not match memory occupancy")
            for image, item in zip(
                guide_bundle.exception_pngs,
                memory.exceptions,
                strict=True,
            ):
                if _sha256_bytes(image) != item.provenance.raw.content_sha256:
                    raise ValueError("guide exception asset does not match raw evidence")
        product_state: StoryProductState | None = None
        generation_receipt: StoryGenerationReceipt | None = None
        if revision["format"] == _REVISION_FORMAT_V3:
            product_digest = _digest(revision["product_state_sha256"], "product state digest")
            receipt_digest = _digest(
                revision["generation_receipt_sha256"], "generation receipt digest"
            )
            product_bytes = _read_regular(
                self._products / f"{product_digest}.json",
                "story product state",
                _MAX_JSON_BYTES,
            )
            if _sha256_bytes(product_bytes) != product_digest:
                raise ValueError("story product state SHA-256 changed")
            receipt_bytes = _read_regular(
                self._receipts / f"{receipt_digest}.json",
                "story generation receipt",
                _MAX_JSON_BYTES,
            )
            if _sha256_bytes(receipt_bytes) != receipt_digest:
                raise ValueError("story generation receipt SHA-256 changed")
            product_state = StoryProductState.from_json(product_bytes)
            generation_receipt = StoryGenerationReceipt.from_json(receipt_bytes)
            expected_backend = "minimax-h3" if contract.format == "duet-x-minimax-h3-v1" else "ltx"
            if (
                generation_receipt.parent_revision_sha256 != state.parent_revision_sha256
                or generation_receipt.checkpoint_sha256 != checkpoint_sha256
                or generation_receipt.memory_backend != expected_backend
            ):
                raise ValueError("story generation receipt does not match its revision")
            evidence_ids = {item.evidence_id for item in product_state.evidence_records}
            if not set(generation_receipt.selected_evidence_ids) <= evidence_ids:
                raise ValueError("story generation receipt names unavailable evidence")
        return LoadedStoryRevision(
            state,
            library,
            blocks,
            memory,
            self._asset(state.last_frame_sha256, "last frame"),
            self._asset(cast(str, revision["story_context_sha256"]), "story context"),
            cast(dict[str, object], metadata),
            guide_bundle,
            product_state,
            generation_receipt,
        )


def migrate_minimax_v2_product_state(parent: LoadedStoryRevision) -> StoryProductState:
    """Represent authenticated v2 evidence without inventing missing time or entity facts."""
    if not isinstance(parent, LoadedStoryRevision) or parent.guide_bundle is None:
        raise ValueError("MiniMax v2 migration requires a materialized guide revision")
    bundle = parent.require_guide_bundle()
    by_fingerprint = {item.fingerprint(): item for item in parent.memory.exceptions}
    records: list[StoryEvidenceRecord] = []
    for image, fingerprint in zip(
        bundle.exception_pngs, bundle.exception_fingerprints, strict=True
    ):
        item = by_fingerprint.get(fingerprint)
        if item is None:
            raise ValueError("legacy guide fingerprint is unavailable from memory")
        records.append(
            StoryEvidenceRecord(
                item.item_id,
                None,
                ObservationKind.LEGACY,
                (),
                _sha256_bytes(image),
                None,
                None,
                None,
                None,
                None,
                item.salience_q,
                fingerprint,
            ).validate()
        )
    entities = tuple(
        sorted(
            (
                CanonEntity(
                    reference.name.casefold(),
                    reference.name,
                    reference.media_sha256,
                    "",
                    CanonPresence.UNKNOWN,
                    (),
                    None,
                    (),
                ).validate()
                for reference in parent.library.references
            ),
            key=lambda entity: entity.entity_id,
        )
    )
    return StoryProductState(
        (),
        tuple(sorted(records, key=lambda record: record.evidence_id)),
        StoryCanon(entities),
        StoryMemoryPolicy(),
    ).validate()


__all__ = (
    "LoadedStoryRevision",
    "StoryCommitPayload",
    "StoryGuideBundle",
    "StoryProjectStore",
    "migrate_minimax_v2_product_state",
)
