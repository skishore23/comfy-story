"""Pinned official-LTX construction for the decoded-quality development rehearsal.

The external LTX packages remain lazily imported.  This module first authenticates the frozen
source, runtime, and public API surface; model-owning objects are constructed only by explicit
post-preflight methods.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import inspect
import os
import platform
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol, Self, cast

import torch

from comfy_story.contracts import tensor_sha256
from comfy_story.ltx_media import (
    PINNED_LTX_OFFLOAD_MODE,
    STREAMING_FOUNDATION_GUARD_SHA256,
    OneStageMediaGenerationResult,
    PinnedLTXOneStageMediaEngine,
)
from comfy_story.ltx_quality_generation import (
    BuiltMethodGuides,
    FrozenMethodModules,
    build_method_guides,
    load_frozen_trainable_checkpoint,
)
from comfy_story.ltx_quality_operations import (
    DevelopmentDisjointnessReceipt,
    DevelopmentForwardResult,
    DevelopmentInput,
    PinnedLTXRGBMaterializer,
    decode_rgb24_frames,
    decoded_frames_sha256,
    load_development_input,
    validate_development_disjointness,
)
from comfy_story.ltx_quality_protocol import (
    FROZEN_TRAINABLE_CHECKPOINT_SHA256,
    LTX_DEVELOPMENT_CHECKPOINT_SHA256,
    LTX_SOURCE_COMMIT,
    FrozenRecord,
    GenerationCellKey,
    Method,
    RuntimeCoordinate,
    canonical_json,
    canonical_sha256,
    from_canonical_json,
)
from comfy_story.media_probe import DecodedMediaFacts, probe_decoded_video

_SHA256_HEX = frozenset("0123456789abcdef")
_TEACHER_SIZE_BYTES = 46_149_344_974
_TRAINABLE_SIZE_BYTES = 1_081_496
_CPU_STREAM_FOUNDATION_SHA256 = "c6cf979445daeae6672c6d565cec48fbf6a7212fb8e0a4ed58cd5f4ba42df65f"
_PYTHON = "/mnt/data/duet-x/decision1/runtime/py312-torch28/bin/python"
_CHECKOUT = "/mnt/data/duet-x/decision1/runtime/LTX-2"
_CORE_SOURCE = f"{_CHECKOUT}/packages/ltx-core/src"
_PIPELINES_SOURCE = f"{_CHECKOUT}/packages/ltx-pipelines/src"
_DEPENDENCY_SOURCE = "/mnt/data/duet-x/poc-runs/replication-r1-6f1dded-6ef499bf/deps"
_GEMMA_ROOT = "/mnt/data/duet-x/decision1/runtime/models/gemma-3-12b-it-qat-q4_0-unquantized"
_TEACHER = "/mnt/data/duet-x/decision1/models/ltx-2.3-22b-dev.safetensors"
_FFPROBE = "/usr/bin/ffprobe"
_FFPROBE_SIZE_BYTES = 178_832
_FFPROBE_SHA256 = "d4f3ef9c12be756793cad83dd2004d89f49c1c4094053bfbbe7e28925c8fa4fd"
_FFMPEG = "/usr/bin/ffmpeg"
_FFMPEG_SIZE_BYTES = 301_544
_FFMPEG_SHA256 = "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
_PINNED_API_ERROR = "pinned LTX quality materialization API is unavailable or has drifted"
_DEVELOPMENT_SCENE_SHA256 = "74cba958d3b8f6ce654d198cdcd2175f97c8f3be24c3adc92478b25b73e8a107"
_DEVELOPMENT_BUNDLE_SHA256 = "fc94a4133f6fe5cfcfda7e948913827860cc8cabedbac28b94ee4ef8999c97e9"
_DEVELOPMENT_K2_SHA256 = "2ed0a9054f9357db43d5b24905d55adaf425ba0d8d79df187ce6520d9ed1c973"
_DEVELOPMENT_SOURCE_SHA256 = "56fc776b2ec31c16a11555a6bf9f1d3d9736f168f5fc20037e8d4ced55ee229d"
_DEVELOPMENT_PROMPT = "A video of a marching band."
_DEVELOPMENT_SOURCE = "train/BandMarching/v_BandMarching_g06_c01.avi"
_DEVELOPMENT_FILE_SIZES = (2_752, 668_201, 443_085)
_DEVELOPMENT_LATENT_SHA256 = (
    "0e7235f394ff145b43fd489662b51c9f134c2ba5faccca8110fe332407ae72ca",
    "a886ddd51ea9936a250cdec92b967a9efdbdd5761614ac595c741fc526b59348",
    "f81774c32e62a0c0a05ff46a6981b3e578794e3e800d0f6b7b49b755d164f14a",
    "bfcd2610c93b5e83a5e8b5705c75be17e86deeb310542bf8d42d3fe9a5b5ae24",
    "5bafa674620c6287183b0407969fc257d82a410bbfc33c5ea3e29d348498765c",
    "c05e2aa1efe46ccdd4fbe9edf147bfdb60b169bd2c93e6f9f6457b952a21a83e",
    "38569366efc051f39590b9a049c9e9e406d9c147dd6246def8e0b16a8c77fe43",
    "14555a50994f6f91ac53eb36db26dcb3fbb163767004ff6aa6624ee3e032115c",
    "7574104505b67b37e334ec2bae4aac68d0a27d8efa8b0dbab99fd7ab6a5805f4",
)
_DEVELOPMENT_PREPROCESSED_SHA256 = (
    "1d1444e1f7608f846b65f08f101b08a85b4d514219e485bd4c3f8a6355ad1843",
    "22bdf43293c491b38578be41c807966104a942afb16e394b8a5b7d39e096d765",
    "d2e0b4b60ca1878bf462f25a2a59ef46248b1c774b2ca941fa6e18d80d173c8c",
    "8a3a9b224470d1d72a2a42524ec666db9c5e0692ce3b1e14812969e297850828",
    "0027e79cc8c6bce966701d26b3ac647a1cfee5b187e0a1b3dc3c091e16d6b1b6",
    "2bcd351b2777a3c8fbdefdf4e5a3e5bd97cfbd976c8c9caca5ffb6eee6e67db6",
    "5b1a8e19c25e7231c16152a36759b8df0ceac3e24b94bbf82e4163a7a926b72d",
    "3eecebd19f15920010923e8ad1a0a2224184057a3845b1d44faf19f2caf9d202",
    "e03bfb50980956e9f70892d0dac9c538dc16ee09132eee7c93b3169b9798c116",
)
_CORE_SLOTS = (0, 1, 3, 4, 6, 7)
_PROTECTED_SLOTS = (2, 5)
_CURRENT_SLOT = 8


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _commit(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 40-hex commit")
    return value


@dataclass(frozen=True, slots=True)
class FrozenQualityRuntimeIdentity(FrozenRecord):
    """Every immutable runtime/source/model identity in the Task 6 rehearsal."""

    source_commit: str
    source_archive_sha256: str
    python_executable: str
    python_version: str
    torch_version: str
    device: str
    ltx_checkout_root: str
    ltx_core_source_root: str
    ltx_pipelines_source_root: str
    dependency_source_root: str
    ltx_source_commit: str
    teacher_checkpoint_path: str
    teacher_checkpoint_sha256: str
    teacher_checkpoint_size_bytes: int
    ffprobe_path: str
    ffprobe_sha256: str
    ffprobe_size_bytes: int
    ffmpeg_path: str
    ffmpeg_sha256: str
    ffmpeg_size_bytes: int
    gemma_root: str
    trainable_checkpoint_sha256: str
    trainable_checkpoint_size_bytes: int
    expected_foundation_sha256: str
    offload_mode: str
    foundation_guard_sha256: str
    runtime_coordinate_sha256: str

    @classmethod
    def default(cls, *, source_commit: str, source_archive_sha256: str) -> Self:
        return cls(
            source_commit=source_commit,
            source_archive_sha256=source_archive_sha256,
            python_executable=_PYTHON,
            python_version="3.12.11",
            torch_version="2.8.0+cu128",
            device="cuda",
            ltx_checkout_root=_CHECKOUT,
            ltx_core_source_root=_CORE_SOURCE,
            ltx_pipelines_source_root=_PIPELINES_SOURCE,
            dependency_source_root=_DEPENDENCY_SOURCE,
            ltx_source_commit=LTX_SOURCE_COMMIT,
            teacher_checkpoint_path=_TEACHER,
            teacher_checkpoint_sha256=LTX_DEVELOPMENT_CHECKPOINT_SHA256,
            teacher_checkpoint_size_bytes=_TEACHER_SIZE_BYTES,
            ffprobe_path=_FFPROBE,
            ffprobe_sha256=_FFPROBE_SHA256,
            ffprobe_size_bytes=_FFPROBE_SIZE_BYTES,
            ffmpeg_path=_FFMPEG,
            ffmpeg_sha256=_FFMPEG_SHA256,
            ffmpeg_size_bytes=_FFMPEG_SIZE_BYTES,
            gemma_root=_GEMMA_ROOT,
            trainable_checkpoint_sha256=FROZEN_TRAINABLE_CHECKPOINT_SHA256,
            trainable_checkpoint_size_bytes=_TRAINABLE_SIZE_BYTES,
            expected_foundation_sha256=_CPU_STREAM_FOUNDATION_SHA256,
            offload_mode=PINNED_LTX_OFFLOAD_MODE,
            foundation_guard_sha256=STREAMING_FOUNDATION_GUARD_SHA256,
            runtime_coordinate_sha256=RuntimeCoordinate.default().fingerprint(),
        )

    def validate(self) -> Self:
        _commit(self.source_commit, "source commit")
        _sha256(self.source_archive_sha256, "source archive SHA-256")
        expected = type(self).default(
            source_commit=self.source_commit,
            source_archive_sha256=self.source_archive_sha256,
        )
        if self != expected:
            raise ValueError("frozen quality runtime identity changed")
        return self


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentIdentity(FrozenRecord):
    """Exact preserved development input identities, including ordered latent ancestry."""

    scene_manifest_sha256: str
    scene_manifest_size_bytes: int
    latent_bundle_sha256: str
    latent_bundle_size_bytes: int
    k2_receipt_sha256: str
    k2_receipt_size_bytes: int
    scene_id: str
    split: str
    prompt: str
    negative_prompt: str
    source_relative_path: str
    source_video_sha256: str
    protected_slots: tuple[int, int]
    current_slot: int
    core_slots: tuple[int, ...]
    ordered_latent_sha256: tuple[str, ...]
    ordered_preprocessed_sha256: tuple[str, ...]

    @classmethod
    def default(cls) -> Self:
        return cls(
            _DEVELOPMENT_SCENE_SHA256,
            _DEVELOPMENT_FILE_SIZES[0],
            _DEVELOPMENT_BUNDLE_SHA256,
            _DEVELOPMENT_FILE_SIZES[1],
            _DEVELOPMENT_K2_SHA256,
            _DEVELOPMENT_FILE_SIZES[2],
            "development-00",
            "development",
            _DEVELOPMENT_PROMPT,
            "",
            _DEVELOPMENT_SOURCE,
            _DEVELOPMENT_SOURCE_SHA256,
            _PROTECTED_SLOTS,
            _CURRENT_SLOT,
            _CORE_SLOTS,
            _DEVELOPMENT_LATENT_SHA256,
            _DEVELOPMENT_PREPROCESSED_SHA256,
        )

    def validate(self) -> Self:
        if self != type(self).default():
            raise ValueError("frozen development identity changed")
        return self


@dataclass(frozen=True, slots=True)
class FileIdentity(FrozenRecord):
    size_bytes: int
    sha256: str

    def validate(self) -> Self:
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("file identity size must be nonnegative")
        _sha256(self.sha256, "file identity SHA-256")
        return self


@dataclass(frozen=True, slots=True)
class ObservedRuntimeFacts(FrozenRecord):
    python_executable: str
    python_version: str
    torch_version: str
    device: str

    def validate(self) -> Self:
        if any(
            type(value) is not str or not value
            for value in (
                self.python_executable,
                self.python_version,
                self.torch_version,
                self.device,
            )
        ):
            raise ValueError("observed runtime facts must be nonempty strings")
        return self


@dataclass(frozen=True, slots=True)
class DevelopmentRuntimePaths:
    teacher_checkpoint: Path
    gemma_root: Path
    scene_manifest: Path
    latent_bundle: Path
    k2_receipt: Path
    trainable_checkpoint: Path


@dataclass(frozen=True, slots=True)
class CheckoutState(FrozenRecord):
    """Authenticated state of the pinned LTX checkout."""

    commit: str
    clean: bool

    def validate(self) -> Self:
        _commit(self.commit, "pinned LTX checkout commit")
        if type(self.clean) is not bool or not self.clean:
            raise ValueError("pinned LTX checkout must be clean")
        return self


@dataclass(frozen=True, slots=True)
class GemmaInventoryEntry(FrozenRecord):
    relative_path: str
    size_bytes: int
    sha256: str

    def validate(self) -> Self:
        path = Path(self.relative_path)
        if (
            type(self.relative_path) is not str
            or not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or type(self.size_bytes) is not int
            or self.size_bytes < 0
        ):
            raise ValueError("Gemma inventory entry is unsafe")
        _sha256(self.sha256, "Gemma inventory entry SHA-256")
        return self


@dataclass(frozen=True, slots=True)
class GemmaInventoryReceipt(FrozenRecord):
    """Strict, immutable inventory boundary for the local Gemma directory."""

    root: str
    entries: tuple[GemmaInventoryEntry, ...]
    inventory_sha256: str

    @classmethod
    def capture(cls, root: Path, entries: Sequence[GemmaInventoryEntry]) -> Self:
        validated = tuple(entry.validate() for entry in entries)
        if tuple(entry.relative_path for entry in validated) != tuple(
            sorted(entry.relative_path for entry in validated)
        ):
            raise ValueError("Gemma inventory entries must be sorted")
        payload = {
            "format": "duet-x-gemma-inventory-v1",
            "root": str(root),
            "entries": tuple(
                (entry.relative_path, entry.size_bytes, entry.sha256) for entry in validated
            ),
        }
        return cls(str(root), validated, canonical_sha256(payload)).validate()

    def validate(self) -> Self:
        if type(self.root) is not str or not Path(self.root).is_absolute() or not self.entries:
            raise ValueError("Gemma inventory receipt is incomplete")
        validated = tuple(entry.validate() for entry in self.entries)
        names = tuple(entry.relative_path for entry in validated)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("Gemma inventory entries must be unique and sorted")
        expected = canonical_sha256(
            {
                "format": "duet-x-gemma-inventory-v1",
                "root": self.root,
                "entries": tuple(
                    (entry.relative_path, entry.size_bytes, entry.sha256) for entry in validated
                ),
            }
        )
        if self.inventory_sha256 != expected:
            raise ValueError("Gemma inventory receipt changed")
        return self


@dataclass(frozen=True, slots=True)
class OfficialLTXMaterializationAPI:
    """The exact three official symbols used by source materialization."""

    video_preprocess: Callable[..., torch.Tensor]
    image_conditioner_type: type[Any]
    model_paths_type: type[Any]

    def build_materializer(
        self,
        identity: FrozenQualityRuntimeIdentity,
        *,
        device: torch.device,
        file_identity: Callable[[Path], FileIdentity] | None = None,
        directory_check: Callable[[Path], bool] | None = None,
    ) -> PinnedLTXRGBMaterializer:
        """Construct only the exact official ModelPaths/ImageConditioner lifecycle owner."""
        identity.validate()
        if device != torch.device(identity.device) or device.type != "cuda":
            raise ValueError("pinned materializer requires the frozen cuda device")
        file_reader = file_identity or _file_identity
        directory_reader = directory_check or _directory_check
        teacher_path = Path(identity.teacher_checkpoint_path)
        if (
            file_reader(teacher_path).validate()
            != FileIdentity(
                identity.teacher_checkpoint_size_bytes,
                identity.teacher_checkpoint_sha256,
            ).validate()
        ):
            raise ValueError("pinned materializer teacher identity changed")
        if not directory_reader(Path(identity.gemma_root)):
            raise ValueError("pinned materializer Gemma identity changed")
        factory = getattr(self.model_paths_type, "from_monolith", None)
        if not callable(factory):
            raise RuntimeError(_PINNED_API_ERROR)
        paths = factory(identity.teacher_checkpoint_path, identity.gemma_root)
        if type(paths) is not self.model_paths_type:
            raise RuntimeError(_PINNED_API_ERROR)
        video_vae = getattr(paths, "video_vae", None)
        if not callable(video_vae):
            raise ValueError("pinned LTX ModelPaths video VAE identity changed")
        video_vae_checkpoint = video_vae()
        if type(video_vae_checkpoint) is not str or not video_vae_checkpoint:
            raise ValueError("pinned LTX ModelPaths video VAE identity changed")
        conditioner = self.image_conditioner_type(
            checkpoint_path=video_vae_checkpoint,
            dtype=torch.bfloat16,
            device=device,
        )
        if type(conditioner) is not self.image_conditioner_type:
            raise RuntimeError(_PINNED_API_ERROR)
        return PinnedLTXRGBMaterializer(self.video_preprocess, cast(Any, conditioner), device)


def _git_state(root: Path) -> CheckoutState:
    try:
        commit = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ("git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("pinned LTX checkout state is unavailable") from error
    return CheckoutState(commit.stdout.strip(), not bool(status.stdout.strip())).validate()


def _symbol_origin(symbol: object) -> Path:
    inspected = cast(Any, symbol)
    value = inspect.getsourcefile(inspected) or inspect.getfile(inspected)
    return Path(value)


def _parameter_shape(value: Callable[..., object]) -> tuple[tuple[str, str, bool], ...]:
    return tuple(
        (name, parameter.kind.name, parameter.default is not inspect.Parameter.empty)
        for name, parameter in inspect.signature(value).parameters.items()
    )


_VIDEO_SIGNATURE = (
    ("frames", "POSITIONAL_OR_KEYWORD", False),
    ("height", "POSITIONAL_OR_KEYWORD", False),
    ("width", "POSITIONAL_OR_KEYWORD", False),
    ("dtype", "POSITIONAL_OR_KEYWORD", False),
    ("device", "POSITIONAL_OR_KEYWORD", False),
)
_CONDITIONER_SIGNATURE = (
    ("checkpoint_path", "POSITIONAL_OR_KEYWORD", False),
    ("dtype", "POSITIONAL_OR_KEYWORD", False),
    ("device", "POSITIONAL_OR_KEYWORD", False),
    ("registry", "POSITIONAL_OR_KEYWORD", True),
    ("alloc_trim_strategy", "POSITIONAL_OR_KEYWORD", True),
)
_CONDITIONER_CALL_SIGNATURE = (
    ("self", "POSITIONAL_OR_KEYWORD", False),
    ("fn", "POSITIONAL_OR_KEYWORD", False),
)
_MONOLITH_SIGNATURE = (
    ("checkpoint_path", "POSITIONAL_OR_KEYWORD", False),
    ("gemma_root", "POSITIONAL_OR_KEYWORD", True),
    ("video_vae_path", "KEYWORD_ONLY", True),
)
_VIDEO_VAE_SIGNATURE = (("self", "POSITIONAL_OR_KEYWORD", False),)

_PIPELINE_SIGNATURE = (
    ("model_paths", "POSITIONAL_OR_KEYWORD", False),
    ("loras", "POSITIONAL_OR_KEYWORD", False),
    ("device", "POSITIONAL_OR_KEYWORD", True),
    ("quantization", "POSITIONAL_OR_KEYWORD", True),
    ("registry", "POSITIONAL_OR_KEYWORD", True),
    ("compilation_config", "POSITIONAL_OR_KEYWORD", True),
    ("offload_mode", "POSITIONAL_OR_KEYWORD", True),
    ("alloc_trim_strategy", "POSITIONAL_OR_KEYWORD", True),
    ("prompt_enhancer_gemma_root", "POSITIONAL_OR_KEYWORD", True),
    ("diffvae_optimization", "POSITIONAL_OR_KEYWORD", True),
)
_REFERENCE_SIGNATURE = (
    ("latent", "POSITIONAL_OR_KEYWORD", False),
    ("downscale_factor", "POSITIONAL_OR_KEYWORD", True),
    ("temporal_scale_factor", "POSITIONAL_OR_KEYWORD", True),
    ("strength", "POSITIONAL_OR_KEYWORD", True),
)
_ATTENTION_SIGNATURE = (
    ("conditioning", "POSITIONAL_OR_KEYWORD", False),
    ("attention_mask", "POSITIONAL_OR_KEYWORD", False),
)
_GUIDER_SIGNATURE = (
    ("cfg_scale", "POSITIONAL_OR_KEYWORD", True),
    ("stg_scale", "POSITIONAL_OR_KEYWORD", True),
    ("stg_blocks", "POSITIONAL_OR_KEYWORD", True),
    ("rescale_scale", "POSITIONAL_OR_KEYWORD", True),
    ("modality_scale", "POSITIONAL_OR_KEYWORD", True),
    ("skip_step", "POSITIONAL_OR_KEYWORD", True),
)
_DETECT_SIGNATURE = (("checkpoint_path", "POSITIONAL_OR_KEYWORD", False),)
_CHUNKS_SIGNATURE = (
    ("num_frames", "POSITIONAL_OR_KEYWORD", False),
    ("tiling_config", "POSITIONAL_OR_KEYWORD", True),
)
_ENCODE_SIGNATURE = (
    ("video", "POSITIONAL_OR_KEYWORD", False),
    ("fps", "POSITIONAL_OR_KEYWORD", False),
    ("audio", "POSITIONAL_OR_KEYWORD", False),
    ("output_path", "POSITIONAL_OR_KEYWORD", False),
    ("video_chunks_number", "POSITIONAL_OR_KEYWORD", False),
    ("frame_converter", "POSITIONAL_OR_KEYWORD", True),
    ("crf", "POSITIONAL_OR_KEYWORD", True),
    ("preset", "POSITIONAL_OR_KEYWORD", True),
    ("thread_count", "POSITIONAL_OR_KEYWORD", True),
    ("color_space", "KEYWORD_ONLY", True),
)


def _directory_check(path: Path) -> bool:
    try:
        metadata = path.stat(follow_symlinks=False)
        return stat.S_ISDIR(metadata.st_mode) and not path.is_symlink()
    except OSError:
        return False


@dataclass(frozen=True, slots=True)
class PinnedOfficialLTXFactory:
    """Freshly authenticate every official symbol imported by materialization or generation."""

    checkout_root: Path
    pipelines_source_root: Path
    core_source_root: Path
    dependency_source_root: Path
    importer: Callable[[str], object] = importlib.import_module
    symbol_origin: Callable[[object], Path] = _symbol_origin
    checkout_state_reader: Callable[[Path], CheckoutState] = _git_state
    directory_check: Callable[[Path], bool] = _directory_check

    def _expected_module_paths(self) -> dict[str, Path]:
        package = self.pipelines_source_root / "ltx_pipelines/utils"
        core = self.core_source_root / "ltx_core"
        return {
            "ltx_pipelines": self.pipelines_source_root / "ltx_pipelines/__init__.py",
            "ltx_pipelines.utils.blocks": package / "blocks.py",
            "ltx_pipelines.utils.constants": package / "constants.py",
            "ltx_pipelines.utils.media_io": package / "media_io/__init__.py",
            "ltx_pipelines.utils.model_paths": package / "model_paths.py",
            "ltx_pipelines.utils.types": package / "types.py",
            "ltx_core.conditioning": self.core_source_root / "ltx_core/conditioning/__init__.py",
            "ltx_core.components.guiders": self.core_source_root / "ltx_core/components/guiders.py",
            "ltx_core.model.video_vae": self.core_source_root
            / "ltx_core/model/video_vae/__init__.py",
            "ltx_core.model.transformer": core / "model/transformer/__init__.py",
            "ltx_core.block_streaming": core / "block_streaming/__init__.py",
            "ltx_core.block_streaming.provider": core / "block_streaming/provider.py",
            "ltx_core.block_streaming.source": core / "block_streaming/source.py",
            "ltx_core.block_streaming.utils": core / "block_streaming/utils.py",
        }

    def _validate_checkout(self) -> None:
        state = self.checkout_state_reader(self.checkout_root).validate()
        if state.commit != LTX_SOURCE_COMMIT:
            raise ValueError("pinned LTX checkout commit changed")
        expected_roots = (
            self.checkout_root / "packages/ltx-pipelines/src",
            self.checkout_root / "packages/ltx-core/src",
        )
        if (self.pipelines_source_root, self.core_source_root) != expected_roots:
            raise ValueError("pinned LTX source roots changed")
        if not all(
            self.directory_check(path)
            for path in (
                self.checkout_root,
                self.pipelines_source_root,
                self.core_source_root,
                self.dependency_source_root,
            )
        ):
            raise ValueError("pinned LTX or dependency source root is unavailable")

    def _symbol_contracts(
        self,
    ) -> dict[str, tuple[tuple[str, str, Path, tuple[tuple[str, str, bool], ...]], ...]]:
        pipeline = self.pipelines_source_root / "ltx_pipelines"
        core = self.core_source_root / "ltx_core"
        return {
            "ltx_pipelines": (
                (
                    "TI2VidOneStagePipeline",
                    "ltx_pipelines.ti2vid_one_stage",
                    pipeline / "ti2vid_one_stage.py",
                    _PIPELINE_SIGNATURE,
                ),
            ),
            "ltx_pipelines.utils.blocks": (
                (
                    "ImageConditioner",
                    "ltx_pipelines.utils.blocks",
                    pipeline / "utils/blocks.py",
                    _CONDITIONER_SIGNATURE,
                ),
            ),
            "ltx_pipelines.utils.constants": (
                (
                    "detect_params",
                    "ltx_pipelines.utils.constants",
                    pipeline / "utils/constants.py",
                    _DETECT_SIGNATURE,
                ),
            ),
            "ltx_pipelines.utils.media_io": (
                (
                    "video_preprocess",
                    "ltx_pipelines.utils.media_io.decode",
                    pipeline / "utils/media_io/decode.py",
                    _VIDEO_SIGNATURE,
                ),
                (
                    "encode_video",
                    "ltx_pipelines.utils.media_io.encode",
                    pipeline / "utils/media_io/encode.py",
                    _ENCODE_SIGNATURE,
                ),
            ),
            "ltx_pipelines.utils.model_paths": (
                (
                    "ModelPaths",
                    "ltx_pipelines.utils.model_paths",
                    pipeline / "utils/model_paths.py",
                    (),
                ),
            ),
            "ltx_pipelines.utils.types": (
                (
                    "OffloadMode",
                    "ltx_pipelines.utils.types",
                    pipeline / "utils/types.py",
                    (),
                ),
            ),
            "ltx_core.conditioning": (
                (
                    "VideoConditionByReferenceLatent",
                    "ltx_core.conditioning.types.reference_video_cond",
                    core / "conditioning/types/reference_video_cond.py",
                    _REFERENCE_SIGNATURE,
                ),
                (
                    "ConditioningItemAttentionStrengthWrapper",
                    "ltx_core.conditioning.types.attention_strength_wrapper",
                    core / "conditioning/types/attention_strength_wrapper.py",
                    _ATTENTION_SIGNATURE,
                ),
            ),
            "ltx_core.components.guiders": (
                (
                    "MultiModalGuiderParams",
                    "ltx_core.components.guiders",
                    core / "components/guiders.py",
                    _GUIDER_SIGNATURE,
                ),
            ),
            "ltx_core.model.video_vae": (
                (
                    "get_video_chunks_number",
                    "ltx_core.model.video_vae.video_vae",
                    core / "model/video_vae/video_vae.py",
                    _CHUNKS_SIGNATURE,
                ),
            ),
            "ltx_core.model.transformer": (
                (
                    "X0Model",
                    "ltx_core.model.transformer.model",
                    core / "model/transformer/model.py",
                    (),
                ),
            ),
            "ltx_core.block_streaming": (
                (
                    "BlockStreamingWrapper",
                    "ltx_core.block_streaming.wrapper",
                    core / "block_streaming/wrapper.py",
                    (),
                ),
            ),
            "ltx_core.block_streaming.provider": (
                (
                    "WeightsProvider",
                    "ltx_core.block_streaming.provider",
                    core / "block_streaming/provider.py",
                    (),
                ),
            ),
            "ltx_core.block_streaming.source": (
                (
                    "PinnedWeightSource",
                    "ltx_core.block_streaming.source",
                    core / "block_streaming/source.py",
                    (),
                ),
                (
                    "DiskWeightSource",
                    "ltx_core.block_streaming.source",
                    core / "block_streaming/source.py",
                    (),
                ),
            ),
            "ltx_core.block_streaming.utils": (
                (
                    "carve_buffer",
                    "ltx_core.block_streaming.utils",
                    core / "block_streaming/utils.py",
                    (
                        ("buffer", "POSITIONAL_OR_KEYWORD", False),
                        ("layout", "POSITIONAL_OR_KEYWORD", False),
                    ),
                ),
                (
                    "layout_nbytes",
                    "ltx_core.block_streaming.utils",
                    core / "block_streaming/utils.py",
                    (("layout", "POSITIONAL_OR_KEYWORD", False),),
                ),
            ),
        }

    def _checked_import_without_checkout(self, name: str) -> object:
        expected_modules = self._expected_module_paths()
        if name not in expected_modules:
            raise ValueError("unpinned LTX module import was requested")
        try:
            module = self.importer(name)
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(_PINNED_API_ERROR) from error
        module_file = getattr(module, "__file__", None)
        if (
            getattr(module, "__name__", None) != name
            or type(module_file) is not str
            or Path(module_file) != expected_modules[name]
        ):
            raise ValueError("pinned LTX module origin changed")
        for symbol_name, symbol_module, origin, signature in self._symbol_contracts()[name]:
            symbol = getattr(module, symbol_name, None)
            if not callable(symbol):
                raise RuntimeError(_PINNED_API_ERROR)
            if getattr(symbol, "__module__", None) != symbol_module:
                raise ValueError("pinned LTX public symbol module changed")
            if self.symbol_origin(symbol) != origin:
                raise ValueError("pinned LTX symbol origin changed")
            if signature and _parameter_shape(cast(Callable[..., object], symbol)) != signature:
                raise RuntimeError(_PINNED_API_ERROR)
        if name == "ltx_pipelines.utils.types":
            offload_type = getattr(module, "OffloadMode", None)
            offload_cpu = getattr(offload_type, "CPU", None)
            if getattr(offload_cpu, "value", None) != PINNED_LTX_OFFLOAD_MODE:
                raise ValueError("pinned LTX CPU offload identity changed")
        return module

    def checked_importer(self, name: str) -> object:
        self._validate_checkout()
        return self._checked_import_without_checkout(name)

    def reauthenticate_engine_api(self) -> str:
        self._validate_checkout()
        names = (
            "ltx_pipelines",
            "ltx_pipelines.utils.model_paths",
            "ltx_core.conditioning",
            "ltx_core.components.guiders",
            "ltx_pipelines.utils.constants",
            "ltx_core.model.video_vae",
            "ltx_pipelines.utils.media_io",
            "ltx_pipelines.utils.types",
            "ltx_core.model.transformer",
            "ltx_core.block_streaming",
            "ltx_core.block_streaming.provider",
            "ltx_core.block_streaming.source",
            "ltx_core.block_streaming.utils",
        )
        for name in names:
            self._checked_import_without_checkout(name)
        return canonical_sha256(
            {
                "format": "duet-x-pinned-ltx-imports-v1",
                "commit": LTX_SOURCE_COMMIT,
                "modules": names,
                "pipelines_source_root": str(self.pipelines_source_root),
                "core_source_root": str(self.core_source_root),
                "dependency_source_root": str(self.dependency_source_root),
            }
        )

    def load(self) -> OfficialLTXMaterializationAPI:
        self._validate_checkout()
        modules = {
            name: self._checked_import_without_checkout(name)
            for name in (
                "ltx_pipelines.utils.blocks",
                "ltx_pipelines.utils.media_io",
                "ltx_pipelines.utils.model_paths",
            )
        }
        conditioner = getattr(modules["ltx_pipelines.utils.blocks"], "ImageConditioner", None)
        preprocess = getattr(modules["ltx_pipelines.utils.media_io"], "video_preprocess", None)
        model_paths = getattr(modules["ltx_pipelines.utils.model_paths"], "ModelPaths", None)
        if (
            not isinstance(conditioner, type)
            or not callable(preprocess)
            or not isinstance(model_paths, type)
        ):
            raise RuntimeError(_PINNED_API_ERROR)
        if (
            conditioner.__module__ != "ltx_pipelines.utils.blocks"
            or getattr(preprocess, "__module__", None) != "ltx_pipelines.utils.media_io.decode"
            or model_paths.__module__ != "ltx_pipelines.utils.model_paths"
        ):
            raise ValueError("pinned LTX public symbol module changed")
        monolith = getattr(model_paths, "from_monolith", None)
        video_vae = getattr(model_paths, "video_vae", None)
        conditioner_call = vars(conditioner).get("__call__")
        if not callable(monolith) or not callable(video_vae) or not callable(conditioner_call):
            raise RuntimeError(_PINNED_API_ERROR)
        signatures = (
            (_parameter_shape(cast(Callable[..., object], preprocess)), _VIDEO_SIGNATURE),
            (_parameter_shape(conditioner), _CONDITIONER_SIGNATURE),
            (
                _parameter_shape(cast(Callable[..., object], conditioner_call)),
                _CONDITIONER_CALL_SIGNATURE,
            ),
            (_parameter_shape(cast(Callable[..., object], monolith)), _MONOLITH_SIGNATURE),
            (_parameter_shape(cast(Callable[..., object], video_vae)), _VIDEO_VAE_SIGNATURE),
        )
        if any(observed != expected for observed, expected in signatures):
            raise RuntimeError(_PINNED_API_ERROR)
        return OfficialLTXMaterializationAPI(
            cast(Callable[..., torch.Tensor], preprocess), conditioner, model_paths
        )


def _file_identity(path: Path) -> FileIdentity:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("frozen runtime file is missing or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("frozen runtime file must be a single-link regular file")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while block := handle.read(1024 * 1024):
                digest.update(block)
        return FileIdentity(metadata.st_size, digest.hexdigest()).validate()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _executable_file_identity(path: Path) -> FileIdentity:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("frozen runtime executable path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("frozen runtime executable is missing or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o111 == 0
        ):
            raise ValueError("frozen runtime executable must be executable and single-link")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while block := handle.read(1024 * 1024):
                digest.update(block)
        return FileIdentity(metadata.st_size, digest.hexdigest()).validate()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_frozen_media_tools(
    identity: FrozenQualityRuntimeIdentity,
    *,
    executable_identity: Callable[[Path], FileIdentity] = _executable_file_identity,
) -> dict[Path, FileIdentity]:
    """Authenticate the exact provisioned ffprobe/ffmpeg executables before runtime work."""
    frozen = identity.validate()
    expected = {
        Path(frozen.ffprobe_path): FileIdentity(
            frozen.ffprobe_size_bytes, frozen.ffprobe_sha256
        ).validate(),
        Path(frozen.ffmpeg_path): FileIdentity(
            frozen.ffmpeg_size_bytes, frozen.ffmpeg_sha256
        ).validate(),
    }
    observed: dict[Path, FileIdentity] = {}
    for path, name in (
        (Path(frozen.ffprobe_path), "ffprobe executable"),
        (Path(frozen.ffmpeg_path), "ffmpeg executable"),
    ):
        if not path.is_absolute():
            raise ValueError(f"frozen {name} path must be absolute")
        try:
            actual = executable_identity(path).validate()
        except (OSError, ValueError) as error:
            raise ValueError(f"frozen {name} is missing or unsafe") from error
        if actual != expected[path]:
            raise ValueError(f"frozen {name} identity changed")
        observed[path] = actual
    return observed


def _frozen_media_callables(
    identity: FrozenQualityRuntimeIdentity,
) -> tuple[Callable[[Path], DecodedMediaFacts], Callable[[Path], str]]:
    frozen = identity.validate()
    media_probe: Callable[[Path], DecodedMediaFacts] = partial(
        probe_decoded_video,
        ffprobe_path=Path(frozen.ffprobe_path),
        ffprobe_sha256=frozen.ffprobe_sha256,
        descriptor_stable=True,
    )
    rgb24_decoder = partial(
        decode_rgb24_frames,
        ffmpeg_path=Path(frozen.ffmpeg_path),
        ffmpeg_sha256=frozen.ffmpeg_sha256,
        probe=media_probe,
        descriptor_stable=True,
    )
    decoded_frames_digest: Callable[[Path], str] = partial(
        decoded_frames_sha256,
        decoder=rgb24_decoder,
        probe=media_probe,
    )
    return media_probe, decoded_frames_digest


def _gemma_inventory(root: Path) -> GemmaInventoryReceipt:
    if not root.is_absolute() or not _directory_check(root):
        raise ValueError("Gemma inventory root is unavailable")
    entries: list[GemmaInventoryEntry] = []
    for path in sorted(root.rglob("*")):
        try:
            metadata = path.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError("Gemma inventory contains an unsafe entry") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("Gemma inventory contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            identity = _file_identity(path)
            entries.append(
                GemmaInventoryEntry(
                    path.relative_to(root).as_posix(), identity.size_bytes, identity.sha256
                ).validate()
            )
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Gemma inventory contains a non-file entry")
    return GemmaInventoryReceipt.capture(root, entries)


def capture_gemma_inventory(root: Path) -> GemmaInventoryReceipt:
    """Hash the complete exact Gemma tree, rejecting links and unsafe entries."""
    try:
        return _gemma_inventory(root)
    except (OSError, ValueError) as error:
        raise ValueError("Gemma inventory disk is missing or unsafe") from error


def load_gemma_inventory_receipt(path: Path) -> GemmaInventoryReceipt:
    """Strict-load one canonical mode-0600, non-linked Gemma inventory receipt."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("Gemma inventory receipt is missing or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ValueError("Gemma inventory receipt must be a mode 0600 single-link regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            data = handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        receipt = from_canonical_json(data, GemmaInventoryReceipt).validate()
    except (TypeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("Gemma inventory receipt is not strict canonical JSON") from error
    if data != canonical_json(receipt):
        raise ValueError("Gemma inventory receipt is not strict canonical JSON")
    return receipt


def verify_gemma_inventory(root: Path, expected: GemmaInventoryReceipt) -> GemmaInventoryReceipt:
    """Exact-compare the safe on-disk Gemma tree with an authenticated receipt."""
    receipt = expected.validate()
    if root != Path(receipt.root):
        raise ValueError("Gemma inventory root changed")
    observed = capture_gemma_inventory(root)
    if observed != receipt:
        raise ValueError("Gemma inventory disk content changed")
    return observed


def _runtime_facts() -> ObservedRuntimeFacts:
    return ObservedRuntimeFacts(
        str(Path(sys.executable)),
        platform.python_version(),
        str(torch.__version__),
        "cuda" if torch.cuda.is_available() else "unavailable",
    ).validate()


def _engine_factory(**kwargs: object) -> PinnedLTXOneStageMediaEngine:
    return PinnedLTXOneStageMediaEngine(**cast(dict[str, Any], kwargs))


class _DevelopmentLike(Protocol):
    @property
    def scene_id(self) -> str: ...

    @property
    def prompt(self) -> str: ...

    @property
    def latent_sha256(self) -> tuple[str, ...]: ...

    @property
    def latents(self) -> tuple[torch.Tensor, ...]: ...

    def validate(self) -> object: ...


@dataclass(frozen=True, slots=True)
class _DevelopmentGuideScene:
    """Nonserializable scene facade limited to fields read by the reviewed guide builder."""

    scene_id: str
    prompt: str
    negative_prompt: str
    vae_latent_sha256: tuple[str, ...]
    preprocessed_tensor_sha256: tuple[str, ...]
    _fingerprint: str

    @classmethod
    def build(
        cls,
        *,
        development: _DevelopmentLike,
        preprocessed_sha256: tuple[str, ...],
    ) -> Self:
        development.validate()
        value = cls(
            development.scene_id,
            development.prompt,
            "",
            development.latent_sha256,
            preprocessed_sha256,
            canonical_sha256(
                {
                    "format": "duet-x-ltx-development-guide-scene-v1",
                    "scene_id": development.scene_id,
                    "prompt": development.prompt,
                    "negative_prompt": "",
                    "vae_latent_sha256": development.latent_sha256,
                    "preprocessed_tensor_sha256": preprocessed_sha256,
                }
            ),
        )
        return value.validate()

    def validate(self) -> Self:
        if (
            self.scene_id != "development-00"
            or self.prompt != _DEVELOPMENT_PROMPT
            or self.negative_prompt != ""
            or len(self.vae_latent_sha256) != 9
            or len(self.preprocessed_tensor_sha256) != 9
        ):
            raise ValueError("nonconfirmatory development guide scene changed")
        for digest in (*self.vae_latent_sha256, *self.preprocessed_tensor_sha256):
            _sha256(digest, "development guide-scene identity")
        expected = canonical_sha256(
            {
                "format": "duet-x-ltx-development-guide-scene-v1",
                "scene_id": self.scene_id,
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "vae_latent_sha256": self.vae_latent_sha256,
                "preprocessed_tensor_sha256": self.preprocessed_tensor_sha256,
            }
        )
        if self._fingerprint != expected:
            raise ValueError("nonconfirmatory development guide-scene fingerprint changed")
        return self

    def fingerprint(self) -> str:
        return self.validate()._fingerprint


def _slot_ids(method: Method) -> tuple[str, ...]:
    if method is Method.FULL_HISTORY:
        return tuple(str(slot) for slot in range(9))
    if method is Method.RECENT_ANCHOR:
        return ("7", "2", "5", "8")
    if method in {Method.GATED_CORE, Method.DUET_CORE}:
        return ("method_core[0,1,3,4,6,7]", "2", "5", "8")
    raise ValueError("development method is outside the frozen roster")


@dataclass(frozen=True, slots=True)
class DevelopmentForwardBinding(FrozenRecord):
    """Immutable runtime-owned binding consumed by the operational lifecycle protocol."""

    method: Method
    seed: int
    prompt: str
    negative_prompt: str
    ordered_slot_ids: tuple[str, ...]
    initial_noise_sha256: str
    method_core_sha256: str | None
    ordered_guide_sha256: tuple[str, ...]

    def validate(self) -> Self:
        if (
            not isinstance(self.method, Method)
            or self.seed != 101
            or self.prompt != _DEVELOPMENT_PROMPT
            or self.negative_prompt != ""
            or self.ordered_slot_ids != _slot_ids(self.method)
        ):
            raise ValueError("development forward binding changed")
        _sha256(self.initial_noise_sha256, "development initial-noise SHA-256")
        expected_guides = 9 if self.method is Method.FULL_HISTORY else 4
        if len(self.ordered_guide_sha256) != expected_guides:
            raise ValueError("development forward ordered guide roster changed")
        for digest in self.ordered_guide_sha256:
            _sha256(digest, "development ordered guide SHA-256")
        if self.method in {Method.GATED_CORE, Method.DUET_CORE}:
            _sha256(self.method_core_sha256, "development method-core SHA-256")
            if self.method_core_sha256 != self.ordered_guide_sha256[0]:
                raise ValueError("development method-core binding changed")
        elif self.method_core_sha256 is not None:
            raise ValueError("raw development method claimed a method core")
        return self


@dataclass(frozen=True, slots=True)
class DevelopmentForwardPlan(FrozenRecord):
    method: Method
    seed: int
    prompt: str
    negative_prompt: str
    protected_slots: tuple[int, int]
    current_slot: int
    core_slots: tuple[int, ...]
    ordered_slot_ids: tuple[str, ...]
    ordered_guide_sha256: tuple[str, ...]
    method_core_sha256: str | None
    initial_noise_sha256: str
    engine_identity_sha256: str
    foundation_sha256: str
    runtime_identity_sha256: str

    def validate(self) -> Self:
        if (
            not isinstance(self.method, Method)
            or self.seed != 101
            or self.prompt != _DEVELOPMENT_PROMPT
            or self.negative_prompt != ""
            or self.protected_slots != _PROTECTED_SLOTS
            or self.current_slot != _CURRENT_SLOT
            or self.core_slots != _CORE_SLOTS
            or self.ordered_slot_ids != _slot_ids(self.method)
        ):
            raise ValueError("development forward plan changed")
        expected_count = 9 if self.method is Method.FULL_HISTORY else 4
        if len(self.ordered_guide_sha256) != expected_count:
            raise ValueError("development guide roster changed")
        for digest in (
            *self.ordered_guide_sha256,
            self.initial_noise_sha256,
            self.engine_identity_sha256,
            self.foundation_sha256,
            self.runtime_identity_sha256,
        ):
            _sha256(digest, "development forward identity")
        if self.method in {Method.GATED_CORE, Method.DUET_CORE}:
            _sha256(self.method_core_sha256, "development method-core identity")
            if self.method_core_sha256 != self.ordered_guide_sha256[0]:
                raise ValueError("development method-core guide identity changed")
        elif self.method_core_sha256 is not None:
            raise ValueError("raw development method claimed a method core")
        return self


class _OneStageEngine(Protocol):
    def generate(
        self,
        *,
        guides: tuple[torch.Tensor, ...],
        prompt: str,
        negative_prompt: str,
        seed: int,
        output_path: Path,
    ) -> OneStageMediaGenerationResult: ...


def _no_source_revalidation() -> str | None:
    return None


@dataclass(slots=True)
class DevelopmentForwardAdapter:
    """Own the engine while rebuilding and authenticating guides before every forward."""

    development: DevelopmentInput | None
    scene: _DevelopmentGuideScene
    modules: FrozenMethodModules | None
    engine: _OneStageEngine | None
    runtime: RuntimeCoordinate
    engine_identity_sha256: str
    foundation_sha256: str
    runtime_identity_sha256: str
    decoded_frames_digest: Callable[[Path], str] = decoded_frames_sha256
    media_probe: Callable[[Path], DecodedMediaFacts] = probe_decoded_video
    synchronize: Callable[[], None] = torch.cuda.synchronize
    empty_cache: Callable[[], None] = torch.cuda.empty_cache
    source_revalidator: Callable[[], str | None] = _no_source_revalidation
    source_identity_sha256: str | None = None
    _cache: dict[Method, DevelopmentForwardPlan] | None = None
    _latents: tuple[torch.Tensor, ...] = ()
    _scene_manifest_sha256: str = ""
    _closed: bool = False

    def __post_init__(self) -> None:
        development = self.development
        modules = self.modules
        if development is None or modules is None:
            raise ValueError("development adapter requires loaded development modules")
        development.validate()
        self.scene.validate()
        modules.validate()
        self.runtime.validate()
        if self.engine is None:
            raise ValueError("development adapter requires the official one-stage engine")
        for digest in (
            self.engine_identity_sha256,
            self.foundation_sha256,
            self.runtime_identity_sha256,
        ):
            _sha256(digest, "development execution identity")
        if self.scene.scene_id != development.scene_id:
            raise ValueError("development guide scene does not bind the loaded input")
        if self.source_identity_sha256 is not None:
            _sha256(self.source_identity_sha256, "development source identity")
        self._latents = tuple(
            latent.detach().clone().contiguous() for latent in development.latents
        )
        self._scene_manifest_sha256 = development.scene_manifest_sha256
        self.development = None
        self._cache = {}

    def _guard_open(self) -> None:
        if self._closed or self.engine is None:
            raise RuntimeError("development forward adapter is closed")

    def _built(self, method: Method, seed: int) -> tuple[BuiltMethodGuides, DevelopmentForwardPlan]:
        self._guard_open()
        if not isinstance(method, Method) or seed != 101:
            raise ValueError("development forward method or seed changed")
        cache = self._cache
        modules = self.modules
        if cache is None or modules is None or not self._latents:
            raise RuntimeError("development forward adapter is closed")
        observed_source = self.source_revalidator()
        if observed_source != self.source_identity_sha256:
            raise ValueError("pinned LTX source identity changed")
        self.scene.validate()
        modules.validate()
        self.runtime.validate()
        key = GenerationCellKey(self.scene.scene_id, seed, method).validate()
        built = build_method_guides(
            key,
            cast(Any, self.scene),
            tuple(latent.detach().clone().contiguous() for latent in self._latents),
            self._scene_manifest_sha256,
            modules,
        ).validate()
        hashes = tuple(tensor_sha256(guide) for guide in built.guides)
        method_core = hashes[0] if method in {Method.GATED_CORE, Method.DUET_CORE} else None
        initial_noise = canonical_sha256(
            {
                "format": "duet-x-initial-noise-identity-v1",
                "scene_id": self.scene.scene_id,
                "seed": seed,
                "runtime_coordinate_sha256": self.runtime.fingerprint(),
            }
        )
        plan = DevelopmentForwardPlan(
            method,
            seed,
            self.scene.prompt,
            self.scene.negative_prompt,
            _PROTECTED_SLOTS,
            _CURRENT_SLOT,
            _CORE_SLOTS,
            _slot_ids(method),
            hashes,
            method_core,
            initial_noise,
            self.engine_identity_sha256,
            self.foundation_sha256,
            self.runtime_identity_sha256,
        ).validate()
        if tuple(source.sha256 for source in built.receipt.sources) != hashes:
            raise ValueError("reviewed guide receipt and live ordered hashes disagree")
        cached = cache.get(method)
        if cached is not None and cached != plan:
            raise ValueError("live ordered guides no longer match their immutable binding")
        cache[method] = plan
        return built, plan

    def plan(self, method: Method, seed: int) -> DevelopmentForwardPlan:
        return self._built(method, seed)[1]

    def binding(self, method: Method, seed: int) -> DevelopmentForwardBinding:
        plan = self.plan(method, seed)
        return DevelopmentForwardBinding(
            plan.method,
            plan.seed,
            plan.prompt,
            plan.negative_prompt,
            plan.ordered_slot_ids,
            plan.initial_noise_sha256,
            plan.method_core_sha256,
            plan.ordered_guide_sha256,
        ).validate()

    def forward(self, method: Method, seed: int, output_path: Path) -> DevelopmentForwardResult:
        try:
            built, plan = self._built(method, seed)
            engine = self.engine
            modules = self.modules
            if engine is None or modules is None:
                raise RuntimeError("development forward adapter is closed")
            modules.validate()
            live_guides = tuple(guide.detach().clone().contiguous() for guide in built.guides)
            if tuple(tensor_sha256(guide) for guide in live_guides) != plan.ordered_guide_sha256:
                raise ValueError("live ordered guides changed before model forward")
            result = engine.generate(
                guides=live_guides,
                prompt=plan.prompt,
                negative_prompt=plan.negative_prompt,
                seed=seed,
                output_path=output_path,
            ).validate()
            if (
                result.checkpoint_file_sha256 != LTX_DEVELOPMENT_CHECKPOINT_SHA256
                or result.foundation_before_sha256 != plan.foundation_sha256
                or result.foundation_after_sha256 != plan.foundation_sha256
                or result.offload_mode != PINNED_LTX_OFFLOAD_MODE
                or result.foundation_guard_sha256 != STREAMING_FOUNDATION_GUARD_SHA256
            ):
                raise ValueError("official development engine identity changed")
            decoded = self.decoded_frames_digest(result.path)
            _sha256(decoded, "development decoded-frame SHA-256")
            facts = self.media_probe(result.path).validate()
            return DevelopmentForwardResult(
                result.path,
                result.final_latent,
                plan.initial_noise_sha256,
                plan.method_core_sha256,
                plan.ordered_guide_sha256,
                decoded,
                facts,
                plan.engine_identity_sha256,
                result.foundation_before_sha256,
                result.foundation_after_sha256,
                plan.runtime_identity_sha256,
            ).validate()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.synchronize()
        finally:
            self.engine = None
            self.modules = None
            self.development = None
            self._cache = None
            self._latents = ()
            self._scene_manifest_sha256 = ""
            self._closed = True
            try:
                gc.collect()
            finally:
                self.empty_cache()

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("development forward adapter is closed")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class PreparedDevelopmentRuntime:
    api: OfficialLTXMaterializationAPI
    adapter: DevelopmentForwardAdapter
    disjointness: DevelopmentDisjointnessReceipt


@dataclass(frozen=True, slots=True)
class PinnedQualityDevelopmentFactory:
    """Authenticate every frozen parent before invoking either model-construction callback."""

    identity: FrozenQualityRuntimeIdentity
    development_identity: FrozenDevelopmentIdentity
    official_factory: PinnedOfficialLTXFactory
    expected_gemma_inventory: GemmaInventoryReceipt | None = None
    file_identity: Callable[[Path], FileIdentity] = _file_identity
    gemma_inventory_reader: Callable[[Path], GemmaInventoryReceipt] = capture_gemma_inventory
    directory_check: Callable[[Path], bool] = _directory_check
    runtime_facts: Callable[[], ObservedRuntimeFacts] = _runtime_facts
    development_loader: Callable[[Path, Path, Path], DevelopmentInput] = load_development_input
    disjointness_validator: Callable[
        [DevelopmentInput, Sequence[Any], Sequence[object]], DevelopmentDisjointnessReceipt
    ] = validate_development_disjointness
    checkpoint_loader: Callable[..., FrozenMethodModules] = load_frozen_trainable_checkpoint
    engine_factory: Callable[..., _OneStageEngine] = _engine_factory
    executable_identity: Callable[[Path], FileIdentity] = _executable_file_identity

    def _verify_file(self, path: Path, expected: FileIdentity, name: str) -> None:
        observed = self.file_identity(path).validate()
        if observed != expected.validate():
            raise ValueError(f"frozen {name} file identity changed")

    def prepare(
        self,
        *,
        paths: DevelopmentRuntimePaths,
        observed_source_commit: str,
        observed_source_archive_sha256: str,
        clusters: Sequence[Any],
        confirmatory_ancestry: Sequence[object],
    ) -> PreparedDevelopmentRuntime:
        identity = self.identity.validate()
        development_identity = self.development_identity.validate()
        if (
            observed_source_commit != identity.source_commit
            or observed_source_archive_sha256 != identity.source_archive_sha256
        ):
            raise ValueError("frozen source commit or archive identity changed")
        observed_runtime = self.runtime_facts().validate()
        expected_runtime = ObservedRuntimeFacts(
            identity.python_executable,
            identity.python_version,
            identity.torch_version,
            identity.device,
        )
        if observed_runtime != expected_runtime:
            raise ValueError("frozen Python/torch/device runtime identity changed")
        verify_frozen_media_tools(identity, executable_identity=self.executable_identity)
        if (
            self.official_factory.checkout_root != Path(identity.ltx_checkout_root)
            or self.official_factory.pipelines_source_root
            != Path(identity.ltx_pipelines_source_root)
            or self.official_factory.core_source_root != Path(identity.ltx_core_source_root)
            or self.official_factory.dependency_source_root != Path(identity.dependency_source_root)
        ):
            raise ValueError("frozen LTX checkout or source roots changed")
        if not all(
            self.directory_check(path)
            for path in (
                Path(identity.ltx_checkout_root),
                Path(identity.ltx_pipelines_source_root),
                Path(identity.ltx_core_source_root),
                Path(identity.dependency_source_root),
            )
        ):
            raise ValueError("frozen LTX or dependency source root is unavailable")
        if (
            paths.teacher_checkpoint != Path(identity.teacher_checkpoint_path)
            or paths.gemma_root != Path(identity.gemma_root)
            or not self.directory_check(paths.gemma_root)
        ):
            raise ValueError("frozen teacher or Gemma path identity changed")
        if self.expected_gemma_inventory is None:
            raise ValueError("frozen Gemma inventory receipt is required")
        expected_gemma = self.expected_gemma_inventory.validate()
        if expected_gemma.root != str(paths.gemma_root):
            raise ValueError("frozen Gemma inventory root changed")
        if self.gemma_inventory_reader(paths.gemma_root).validate() != expected_gemma:
            raise ValueError("frozen Gemma inventory changed")
        self._verify_file(
            paths.teacher_checkpoint,
            FileIdentity(
                identity.teacher_checkpoint_size_bytes,
                identity.teacher_checkpoint_sha256,
            ),
            "teacher checkpoint",
        )
        expected_development_files = (
            (
                paths.scene_manifest,
                "development-00.json",
                FileIdentity(
                    development_identity.scene_manifest_size_bytes,
                    development_identity.scene_manifest_sha256,
                ),
            ),
            (
                paths.latent_bundle,
                "development-00.pt",
                FileIdentity(
                    development_identity.latent_bundle_size_bytes,
                    development_identity.latent_bundle_sha256,
                ),
            ),
            (
                paths.k2_receipt,
                "development-00-k2.json",
                FileIdentity(
                    development_identity.k2_receipt_size_bytes,
                    development_identity.k2_receipt_sha256,
                ),
            ),
        )
        for path, filename, expected in expected_development_files:
            if path.name != filename:
                raise ValueError("frozen development input filename changed")
            self._verify_file(path, expected, filename)
        if paths.trainable_checkpoint.name != "trainable-checkpoint.pt":
            raise ValueError("frozen trainable checkpoint filename changed")
        self._verify_file(
            paths.trainable_checkpoint,
            FileIdentity(
                identity.trainable_checkpoint_size_bytes,
                identity.trainable_checkpoint_sha256,
            ),
            "trainable checkpoint",
        )
        development = self.development_loader(
            paths.scene_manifest, paths.latent_bundle, paths.k2_receipt
        ).validate()
        observed_development = (
            development.scene_id,
            development.split,
            development.prompt,
            "",
            development.source_relative_path,
            development.source_video_sha256,
            development.protected_slots,
            development.current_slot,
            development.latent_bundle_sha256,
            development.latent_sha256,
            development.scene_manifest_sha256,
            development.k2_receipt_sha256,
        )
        expected_development = (
            development_identity.scene_id,
            development_identity.split,
            development_identity.prompt,
            development_identity.negative_prompt,
            development_identity.source_relative_path,
            development_identity.source_video_sha256,
            development_identity.protected_slots,
            development_identity.current_slot,
            development_identity.latent_bundle_sha256,
            development_identity.ordered_latent_sha256,
            development_identity.scene_manifest_sha256,
            development_identity.k2_receipt_sha256,
        )
        if observed_development != expected_development:
            raise ValueError("frozen development input identity changed")
        disjointness = self.disjointness_validator(
            development, clusters, confirmatory_ancestry
        ).validate()
        api = self.official_factory.load()
        source_identity = self.official_factory.reauthenticate_engine_api()
        modules = self.checkpoint_loader(
            paths.trainable_checkpoint, device=torch.device(identity.device)
        ).validate()
        media_probe, decoded_frames_digest = _frozen_media_callables(identity)
        engine = self.engine_factory(
            teacher_checkpoint_path=paths.teacher_checkpoint,
            teacher_checkpoint_file_sha256=identity.teacher_checkpoint_sha256,
            gemma_root=paths.gemma_root,
            expected_teacher_foundation_sha256=identity.expected_foundation_sha256,
            device=torch.device(identity.device),
            importer=self.official_factory.checked_importer,
            media_probe=media_probe,
        )
        scene = _DevelopmentGuideScene.build(
            development=development,
            preprocessed_sha256=development_identity.ordered_preprocessed_sha256,
        )
        engine_identity = canonical_sha256(
            {
                "format": "duet-x-ltx-development-engine-v1",
                "pipeline": RuntimeCoordinate.default().pipeline,
                "teacher_checkpoint_sha256": identity.teacher_checkpoint_sha256,
                "expected_foundation_sha256": identity.expected_foundation_sha256,
                "offload_mode": identity.offload_mode,
                "foundation_guard_sha256": identity.foundation_guard_sha256,
                "ffprobe_sha256": identity.ffprobe_sha256,
                "ffmpeg_sha256": identity.ffmpeg_sha256,
                "runtime_coordinate_sha256": identity.runtime_coordinate_sha256,
                "ltx_source_commit": identity.ltx_source_commit,
            }
        )
        adapter = DevelopmentForwardAdapter(
            development,
            scene,
            modules,
            engine,
            RuntimeCoordinate.default(),
            engine_identity,
            identity.expected_foundation_sha256,
            identity.fingerprint(),
            decoded_frames_digest=decoded_frames_digest,
            media_probe=media_probe,
            source_revalidator=self.official_factory.reauthenticate_engine_api,
            source_identity_sha256=source_identity,
        )
        return PreparedDevelopmentRuntime(api, adapter, disjointness)


__all__ = (
    "LTX_SOURCE_COMMIT",
    "CheckoutState",
    "DevelopmentForwardAdapter",
    "DevelopmentForwardBinding",
    "DevelopmentForwardPlan",
    "DevelopmentRuntimePaths",
    "FileIdentity",
    "FrozenDevelopmentIdentity",
    "FrozenQualityRuntimeIdentity",
    "GemmaInventoryEntry",
    "GemmaInventoryReceipt",
    "ObservedRuntimeFacts",
    "OfficialLTXMaterializationAPI",
    "PinnedOfficialLTXFactory",
    "PinnedQualityDevelopmentFactory",
    "PreparedDevelopmentRuntime",
    "capture_gemma_inventory",
    "load_gemma_inventory_receipt",
    "verify_frozen_media_tools",
    "verify_gemma_inventory",
)
