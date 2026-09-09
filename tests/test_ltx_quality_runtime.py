from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import inspect
import platform
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from comfy_story import ltx_media as media
from comfy_story import ltx_quality_runtime as quality_runtime
from comfy_story.contracts import tensor_sha256
from comfy_story.ltx_quality_generation import FrozenMethodModules
from comfy_story.ltx_quality_operations import (
    DevelopmentDisjointnessReceipt,
    DevelopmentForwardResult,
    PinnedLTXRGBMaterializer,
)
from comfy_story.ltx_quality_protocol import (
    FROZEN_TRAINABLE_CHECKPOINT_SHA256,
    Method,
    RuntimeCoordinate,
    SceneVariant,
    canonical_sha256,
)
from comfy_story.ltx_quality_runtime import (
    LTX_SOURCE_COMMIT,
    CheckoutState,
    DevelopmentForwardAdapter,
    DevelopmentRuntimePaths,
    FileIdentity,
    FrozenDevelopmentIdentity,
    FrozenQualityRuntimeIdentity,
    GemmaInventoryEntry,
    GemmaInventoryReceipt,
    ObservedRuntimeFacts,
    OfficialLTXMaterializationAPI,
    PinnedOfficialLTXFactory,
    PinnedQualityDevelopmentFactory,
    _DevelopmentGuideScene,
    capture_gemma_inventory,
    load_gemma_inventory_receipt,
    verify_frozen_media_tools,
    verify_gemma_inventory,
)
from comfy_story.media_probe import DecodedMediaFacts


class _Paths:
    checkpoint_path: str

    @classmethod
    def from_monolith(
        cls,
        checkpoint_path: str,
        gemma_root: str | None = None,
        *,
        video_vae_path: str | None = None,
    ) -> _Paths:
        del cls, gemma_root, video_vae_path
        value = object.__new__(_Paths)
        value.checkpoint_path = checkpoint_path
        return value

    def video_vae(self) -> str:
        return self.checkpoint_path


class _Conditioner:
    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: object | None = None,
        alloc_trim_strategy: object = "trim",
    ) -> None:
        self.arguments = (checkpoint_path, dtype, device, registry, alloc_trim_strategy)

    def __call__(self, fn: object) -> object:
        return fn


def _preprocess(
    frames: object,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    del frames, height, width, dtype, device
    return torch.empty(0)


def _encode_video(
    video: object,
    fps: int,
    audio: object,
    output_path: str,
    video_chunks_number: int,
    frame_converter: object = None,
    crf: int = 19,
    preset: str = "veryfast",
    thread_count: int = 0,
    *,
    color_space: object = None,
) -> None:
    del (
        video,
        fps,
        audio,
        output_path,
        video_chunks_number,
        frame_converter,
        crf,
        preset,
        thread_count,
        color_space,
    )


def _official_modules(root: Path) -> tuple[dict[str, object], dict[int, Path]]:
    _Conditioner.__module__ = "ltx_pipelines.utils.blocks"
    _preprocess.__module__ = "ltx_pipelines.utils.media_io.decode"
    _encode_video.__module__ = "ltx_pipelines.utils.media_io.encode"
    _Paths.__module__ = "ltx_pipelines.utils.model_paths"
    pipeline = root / "packages/ltx-pipelines/src"
    paths = {
        "ltx_pipelines.utils.blocks": pipeline / "ltx_pipelines/utils/blocks.py",
        "ltx_pipelines.utils.media_io": pipeline / "ltx_pipelines/utils/media_io/__init__.py",
        "ltx_pipelines.utils.model_paths": pipeline / "ltx_pipelines/utils/model_paths.py",
    }
    modules: dict[str, object] = {
        "ltx_pipelines.utils.blocks": argparse.Namespace(
            __name__="ltx_pipelines.utils.blocks",
            __file__=str(paths["ltx_pipelines.utils.blocks"]),
            ImageConditioner=_Conditioner,
        ),
        "ltx_pipelines.utils.media_io": argparse.Namespace(
            __name__="ltx_pipelines.utils.media_io",
            __file__=str(paths["ltx_pipelines.utils.media_io"]),
            video_preprocess=_preprocess,
            encode_video=_encode_video,
        ),
        "ltx_pipelines.utils.model_paths": argparse.Namespace(
            __name__="ltx_pipelines.utils.model_paths",
            __file__=str(paths["ltx_pipelines.utils.model_paths"]),
            ModelPaths=_Paths,
        ),
    }
    symbol_origins = {
        id(_Conditioner): paths["ltx_pipelines.utils.blocks"],
        id(_preprocess): pipeline / "ltx_pipelines/utils/media_io/decode.py",
        id(_encode_video): pipeline / "ltx_pipelines/utils/media_io/encode.py",
        id(_Paths): paths["ltx_pipelines.utils.model_paths"],
    }
    return modules, symbol_origins


def _official_factory(
    root: Path,
    modules: dict[str, object],
    origins: dict[int, Path],
    *,
    state: CheckoutState | None = None,
    imported: list[str] | None = None,
) -> PinnedOfficialLTXFactory:
    def importer(name: str) -> object:
        if imported is not None:
            imported.append(name)
        return modules[name]

    return PinnedOfficialLTXFactory(
        checkout_root=root,
        pipelines_source_root=root / "packages/ltx-pipelines/src",
        core_source_root=root / "packages/ltx-core/src",
        dependency_source_root=root / "deps",
        importer=importer,
        symbol_origin=lambda symbol: origins[id(symbol)],
        checkout_state_reader=lambda _root: state or CheckoutState(LTX_SOURCE_COMMIT, True),
        directory_check=lambda _path: True,
    )


def _recording_importer(imported: list[str], modules: dict[str, object], name: str) -> object:
    imported.append(name)
    return modules[name]


def _observed_gemma(receipt: GemmaInventoryReceipt, _path: Path) -> GemmaInventoryReceipt:
    return receipt


class _Pipeline:
    def __init__(
        self,
        model_paths: object,
        loras: object,
        device: object = None,
        quantization: object = None,
        registry: object = None,
        compilation_config: object = None,
        offload_mode: object = None,
        alloc_trim_strategy: object = None,
        prompt_enhancer_gemma_root: object = None,
        diffvae_optimization: object = None,
    ) -> None:
        del (
            model_paths,
            loras,
            device,
            quantization,
            registry,
            compilation_config,
            offload_mode,
            alloc_trim_strategy,
            prompt_enhancer_gemma_root,
            diffvae_optimization,
        )


class _OffloadMode:
    CPU = argparse.Namespace(value="cpu")


class _BlockStreamingWrapper:
    pass


class _X0Model:
    pass


class _WeightsProvider:
    pass


class _PinnedWeightSource:
    pass


class _DiskWeightSource:
    pass


def _carve_buffer(buffer: torch.Tensor, layout: object) -> dict[str, torch.Tensor]:
    del buffer, layout
    return {}


def _layout_nbytes(layout: object) -> int:
    del layout
    return 0


class _Reference:
    def __init__(
        self,
        latent: object,
        downscale_factor: int = 1,
        temporal_scale_factor: int = 1,
        strength: float = 1.0,
    ) -> None:
        del latent, downscale_factor, temporal_scale_factor, strength


class _Attention:
    def __init__(self, conditioning: object, attention_mask: object) -> None:
        del conditioning, attention_mask


class _Guider:
    def __init__(
        self,
        cfg_scale: float = 1.0,
        stg_scale: float = 0.0,
        stg_blocks: object = None,
        rescale_scale: float = 0.0,
        modality_scale: float = 1.0,
        skip_step: int = 0,
    ) -> None:
        del cfg_scale, stg_scale, stg_blocks, rescale_scale, modality_scale, skip_step


def _detect_params(checkpoint_path: str) -> object:
    return checkpoint_path


def _get_video_chunks_number(num_frames: int, tiling_config: object = None) -> int:
    del tiling_config
    return num_frames


def _engine_modules(root: Path) -> tuple[dict[str, object], dict[int, Path]]:
    modules, origins = _official_modules(root)
    pipeline = root / "packages/ltx-pipelines/src/ltx_pipelines"
    core = root / "packages/ltx-core/src/ltx_core"
    assignments = (
        (_Pipeline, "ltx_pipelines.ti2vid_one_stage"),
        (_Reference, "ltx_core.conditioning.types.reference_video_cond"),
        (_Attention, "ltx_core.conditioning.types.attention_strength_wrapper"),
        (_Guider, "ltx_core.components.guiders"),
        (_detect_params, "ltx_pipelines.utils.constants"),
        (_get_video_chunks_number, "ltx_core.model.video_vae.video_vae"),
        (_OffloadMode, "ltx_pipelines.utils.types"),
        (_BlockStreamingWrapper, "ltx_core.block_streaming.wrapper"),
        (_X0Model, "ltx_core.model.transformer.model"),
        (_WeightsProvider, "ltx_core.block_streaming.provider"),
        (_PinnedWeightSource, "ltx_core.block_streaming.source"),
        (_DiskWeightSource, "ltx_core.block_streaming.source"),
        (_carve_buffer, "ltx_core.block_streaming.utils"),
        (_layout_nbytes, "ltx_core.block_streaming.utils"),
    )
    for value, module in assignments:
        value.__module__ = module
    modules.update(
        {
            "ltx_pipelines": argparse.Namespace(
                __name__="ltx_pipelines",
                __file__=str(pipeline / "__init__.py"),
                TI2VidOneStagePipeline=_Pipeline,
            ),
            "ltx_pipelines.utils.constants": argparse.Namespace(
                __name__="ltx_pipelines.utils.constants",
                __file__=str(pipeline / "utils/constants.py"),
                detect_params=_detect_params,
            ),
            "ltx_core.conditioning": argparse.Namespace(
                __name__="ltx_core.conditioning",
                __file__=str(core / "conditioning/__init__.py"),
                VideoConditionByReferenceLatent=_Reference,
                ConditioningItemAttentionStrengthWrapper=_Attention,
            ),
            "ltx_core.components.guiders": argparse.Namespace(
                __name__="ltx_core.components.guiders",
                __file__=str(core / "components/guiders.py"),
                MultiModalGuiderParams=_Guider,
            ),
            "ltx_core.model.video_vae": argparse.Namespace(
                __name__="ltx_core.model.video_vae",
                __file__=str(core / "model/video_vae/__init__.py"),
                get_video_chunks_number=_get_video_chunks_number,
            ),
            "ltx_pipelines.utils.types": argparse.Namespace(
                __name__="ltx_pipelines.utils.types",
                __file__=str(pipeline / "utils/types.py"),
                OffloadMode=_OffloadMode,
            ),
            "ltx_core.block_streaming": argparse.Namespace(
                __name__="ltx_core.block_streaming",
                __file__=str(core / "block_streaming/__init__.py"),
                BlockStreamingWrapper=_BlockStreamingWrapper,
            ),
            "ltx_core.model.transformer": argparse.Namespace(
                __name__="ltx_core.model.transformer",
                __file__=str(core / "model/transformer/__init__.py"),
                X0Model=_X0Model,
            ),
            "ltx_core.block_streaming.provider": argparse.Namespace(
                __name__="ltx_core.block_streaming.provider",
                __file__=str(core / "block_streaming/provider.py"),
                WeightsProvider=_WeightsProvider,
            ),
            "ltx_core.block_streaming.source": argparse.Namespace(
                __name__="ltx_core.block_streaming.source",
                __file__=str(core / "block_streaming/source.py"),
                PinnedWeightSource=_PinnedWeightSource,
                DiskWeightSource=_DiskWeightSource,
            ),
            "ltx_core.block_streaming.utils": argparse.Namespace(
                __name__="ltx_core.block_streaming.utils",
                __file__=str(core / "block_streaming/utils.py"),
                carve_buffer=_carve_buffer,
                layout_nbytes=_layout_nbytes,
            ),
        }
    )
    origins.update(
        {
            id(_Pipeline): pipeline / "ti2vid_one_stage.py",
            id(_Reference): core / "conditioning/types/reference_video_cond.py",
            id(_Attention): core / "conditioning/types/attention_strength_wrapper.py",
            id(_Guider): core / "components/guiders.py",
            id(_detect_params): pipeline / "utils/constants.py",
            id(_get_video_chunks_number): core / "model/video_vae/video_vae.py",
            id(_OffloadMode): pipeline / "utils/types.py",
            id(_BlockStreamingWrapper): core / "block_streaming/wrapper.py",
            id(_X0Model): core / "model/transformer/model.py",
            id(_WeightsProvider): core / "block_streaming/provider.py",
            id(_PinnedWeightSource): core / "block_streaming/source.py",
            id(_DiskWeightSource): core / "block_streaming/source.py",
            id(_carve_buffer): core / "block_streaming/utils.py",
            id(_layout_nbytes): core / "block_streaming/utils.py",
        }
    )
    return modules, origins


def test_official_factory_is_import_only_and_verifies_origins_commit_and_signatures(
    tmp_path: Path,
) -> None:
    modules, origins = _official_modules(tmp_path)
    imported: list[str] = []

    factory = _official_factory(tmp_path, modules, origins, imported=imported)
    assert imported == []

    api = factory.load()

    assert imported == [
        "ltx_pipelines.utils.blocks",
        "ltx_pipelines.utils.media_io",
        "ltx_pipelines.utils.model_paths",
    ]
    assert isinstance(api, OfficialLTXMaterializationAPI)
    assert api.video_preprocess is _preprocess
    assert api.image_conditioner_type is _Conditioner
    assert api.model_paths_type is _Paths


@pytest.mark.parametrize(
    "drift",
    ["commit", "module_origin", "symbol_origin", "video_signature", "conditioner_signature"],
)
def test_official_factory_rejects_every_import_contract_drift(tmp_path: Path, drift: str) -> None:
    modules, origins = _official_modules(tmp_path)
    commit = LTX_SOURCE_COMMIT
    if drift == "commit":
        commit = "0" * 40
    if drift == "module_origin":
        cast_module = modules["ltx_pipelines.utils.blocks"]
        cast(argparse.Namespace, cast_module).__file__ = str(tmp_path / "site-packages/blocks.py")
    if drift == "symbol_origin":
        origins[id(_preprocess)] = tmp_path / "site-packages/decode.py"
    if drift == "video_signature":

        def drifted_preprocess(frame: object) -> object:
            return frame

        modules["ltx_pipelines.utils.media_io"].video_preprocess = drifted_preprocess  # type: ignore[attr-defined]
        origins[id(drifted_preprocess)] = origins[id(_preprocess)]
    if drift == "conditioner_signature":

        class DriftedConditioner:
            def __init__(self, path: str) -> None:
                del path

            def __call__(self, fn: object) -> object:
                return fn

        DriftedConditioner.__module__ = "ltx_pipelines.utils.blocks"
        modules["ltx_pipelines.utils.blocks"].ImageConditioner = DriftedConditioner  # type: ignore[attr-defined]
        origins[id(DriftedConditioner)] = origins[id(_Conditioner)]

    factory = _official_factory(
        tmp_path,
        modules,
        origins,
        state=CheckoutState(commit, True),
    )

    with pytest.raises((RuntimeError, ValueError), match="pinned LTX"):
        factory.load()


def test_checked_engine_importer_reauthenticates_all_modules_and_clean_state(
    tmp_path: Path,
) -> None:
    modules, origins = _engine_modules(tmp_path)
    imported: list[str] = []
    state = [CheckoutState(LTX_SOURCE_COMMIT, True)]
    factory = PinnedOfficialLTXFactory(
        checkout_root=tmp_path,
        pipelines_source_root=tmp_path / "packages/ltx-pipelines/src",
        core_source_root=tmp_path / "packages/ltx-core/src",
        dependency_source_root=tmp_path / "deps",
        importer=functools.partial(_recording_importer, imported, modules),
        symbol_origin=lambda symbol: origins[id(symbol)],
        checkout_state_reader=lambda _root: state[0],
        directory_check=lambda _path: True,
    )

    identity = factory.reauthenticate_engine_api()

    assert len(identity) == 64
    assert imported == [
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
    ]
    state[0] = CheckoutState(LTX_SOURCE_COMMIT, False)
    with pytest.raises(ValueError, match="clean"):
        factory.checked_importer("ltx_pipelines")


def test_checked_engine_importer_rejects_origin_and_signature_drift(tmp_path: Path) -> None:
    modules, origins = _engine_modules(tmp_path)
    factory = _official_factory(tmp_path, modules, origins)
    cast(argparse.Namespace, modules["ltx_core.conditioning"]).__file__ = str(
        tmp_path / "site-packages/conditioning.py"
    )
    with pytest.raises(ValueError, match="origin"):
        factory.reauthenticate_engine_api()

    modules, origins = _engine_modules(tmp_path)

    def drifted_detect() -> None:
        return None

    drifted_detect.__module__ = "ltx_pipelines.utils.constants"
    cast(
        argparse.Namespace, modules["ltx_pipelines.utils.constants"]
    ).detect_params = drifted_detect
    origins[id(drifted_detect)] = tmp_path / (
        "packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py"
    )
    with pytest.raises(RuntimeError, match="drifted"):
        _official_factory(tmp_path, modules, origins).reauthenticate_engine_api()


def test_checked_engine_importer_rejects_cpu_offload_member_drift(tmp_path: Path) -> None:
    modules, origins = _engine_modules(tmp_path)
    cast(argparse.Namespace, modules["ltx_pipelines.utils.types"]).OffloadMode = type(
        "DriftedOffloadMode",
        (),
        {"CPU": argparse.Namespace(value="disk")},
    )
    drifted = cast(argparse.Namespace, modules["ltx_pipelines.utils.types"]).OffloadMode
    drifted.__module__ = "ltx_pipelines.utils.types"
    origins[id(drifted)] = tmp_path / "packages/ltx-pipelines/src/ltx_pipelines/utils/types.py"

    with pytest.raises(ValueError, match="CPU offload"):
        _official_factory(tmp_path, modules, origins).reauthenticate_engine_api()


def test_frozen_runtime_identity_is_exact() -> None:
    identity = FrozenQualityRuntimeIdentity.default(
        source_commit="1" * 40,
        source_archive_sha256="2" * 64,
    )
    assert identity.validate() is identity
    assert identity.ltx_source_commit == LTX_SOURCE_COMMIT
    assert identity.python_version == "3.12.11"
    assert identity.torch_version == "2.8.0+cu128"
    assert identity.offload_mode == "cpu"
    assert identity.ffprobe_path == "/usr/bin/ffprobe"
    assert identity.ffprobe_size_bytes == 178_832
    assert (
        identity.ffprobe_sha256
        == "d4f3ef9c12be756793cad83dd2004d89f49c1c4094053bfbbe7e28925c8fa4fd"
    )
    assert identity.ffmpeg_path == "/usr/bin/ffmpeg"
    assert identity.ffmpeg_size_bytes == 301_544
    assert (
        identity.ffmpeg_sha256 == "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
    )
    assert identity.foundation_guard_sha256 == media.STREAMING_FOUNDATION_GUARD_SHA256
    assert (
        identity.expected_foundation_sha256
        == "c6cf979445daeae6672c6d565cec48fbf6a7212fb8e0a4ed58cd5f4ba42df65f"
    )

    drifted = dataclasses.replace(
        identity,
        teacher_checkpoint_size_bytes=identity.teacher_checkpoint_size_bytes + 1,
    )
    with pytest.raises(ValueError, match="frozen quality runtime identity"):
        drifted.validate()
    with pytest.raises(ValueError, match="frozen quality runtime identity"):
        dataclasses.replace(identity, offload_mode="none").validate()
    with pytest.raises(ValueError, match="frozen quality runtime identity"):
        dataclasses.replace(identity, foundation_guard_sha256="0" * 64).validate()
    with pytest.raises(ValueError, match="frozen quality runtime identity"):
        dataclasses.replace(identity, ffprobe_path="/usr/local/bin/ffprobe").validate()
    with pytest.raises(ValueError, match="frozen quality runtime identity"):
        dataclasses.replace(identity, ffmpeg_sha256="0" * 64).validate()


def test_frozen_media_tools_verify_exact_absolute_executable_identities() -> None:
    identity = FrozenQualityRuntimeIdentity.default(
        source_commit="1" * 40,
        source_archive_sha256="2" * 64,
    )
    expected = {
        Path(identity.ffprobe_path): FileIdentity(
            identity.ffprobe_size_bytes, identity.ffprobe_sha256
        ),
        Path(identity.ffmpeg_path): FileIdentity(
            identity.ffmpeg_size_bytes, identity.ffmpeg_sha256
        ),
    }
    observed: list[Path] = []

    def executable_identity(path: Path) -> FileIdentity:
        observed.append(path)
        return expected[path]

    assert verify_frozen_media_tools(identity, executable_identity=executable_identity) == expected
    assert observed == [Path("/usr/bin/ffprobe"), Path("/usr/bin/ffmpeg")]

    with pytest.raises(ValueError, match="ffmpeg executable"):
        verify_frozen_media_tools(
            identity,
            executable_identity=lambda path: (
                FileIdentity(identity.ffmpeg_size_bytes, "0" * 64)
                if path == Path(identity.ffmpeg_path)
                else expected[path]
            ),
        )


def test_frozen_media_callables_forward_only_exact_tool_paths_and_digests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = FrozenQualityRuntimeIdentity.default(
        source_commit="1" * 40,
        source_archive_sha256="2" * 64,
    )
    media_path = (tmp_path / "media.mp4").resolve()
    media_path.write_bytes(b"media")
    calls: list[tuple[str, Path, dict[str, object]]] = []
    facts = DecodedMediaFacts(384, 384, 25, 24.0, 25.0 / 24.0)

    def probe(path: Path, **kwargs: object) -> DecodedMediaFacts:
        calls.append(("probe", path, kwargs))
        return facts

    def decoder(path: Path, **kwargs: object) -> object:
        calls.append(("decode", path, kwargs))
        return argparse.Namespace(validate=lambda: None)

    def digest(path: Path, **kwargs: object) -> str:
        calls.append(("digest", path, kwargs))
        probe_callback = kwargs["probe"]
        decoder_callback = kwargs["decoder"]
        assert callable(probe_callback)
        assert callable(decoder_callback)
        probe_callback(path)
        decoder_callback(path)
        return _digest("frames")

    monkeypatch.setattr(quality_runtime, "probe_decoded_video", probe)
    monkeypatch.setattr(quality_runtime, "decode_rgb24_frames", decoder)
    monkeypatch.setattr(quality_runtime, "decoded_frames_sha256", digest)

    media_probe, decoded_digest = quality_runtime._frozen_media_callables(identity)

    assert media_probe(media_path) == facts
    assert decoded_digest(media_path) == _digest("frames")
    probe_calls = [row for row in calls if row[0] == "probe"]
    decode_calls = [row for row in calls if row[0] == "decode"]
    assert all(
        row[2]
        == {
            "ffprobe_path": Path(identity.ffprobe_path),
            "ffprobe_sha256": identity.ffprobe_sha256,
            "descriptor_stable": True,
        }
        for row in probe_calls
    )
    assert len(decode_calls) == 1
    assert decode_calls[0][2]["ffmpeg_path"] == Path(identity.ffmpeg_path)
    assert decode_calls[0][2]["ffmpeg_sha256"] == identity.ffmpeg_sha256
    assert decode_calls[0][2]["probe"] is media_probe
    assert decode_calls[0][2]["descriptor_stable"] is True


def test_development_factory_installs_exact_descriptor_stable_media_callables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = FrozenQualityRuntimeIdentity.default(
        source_commit="1" * 40,
        source_archive_sha256="2" * 64,
    )
    development_identity = FrozenDevelopmentIdentity.default()
    paths = DevelopmentRuntimePaths(
        teacher_checkpoint=Path(identity.teacher_checkpoint_path),
        gemma_root=Path(identity.gemma_root),
        scene_manifest=tmp_path / "development-00.json",
        latent_bundle=tmp_path / "development-00.pt",
        k2_receipt=tmp_path / "development-00-k2.json",
        trainable_checkpoint=tmp_path / "trainable-checkpoint.pt",
    )
    facts = {
        Path(identity.ffprobe_path): FileIdentity(
            identity.ffprobe_size_bytes, identity.ffprobe_sha256
        ),
        Path(identity.ffmpeg_path): FileIdentity(
            identity.ffmpeg_size_bytes, identity.ffmpeg_sha256
        ),
        paths.teacher_checkpoint: FileIdentity(
            identity.teacher_checkpoint_size_bytes, identity.teacher_checkpoint_sha256
        ),
        paths.scene_manifest: FileIdentity(
            development_identity.scene_manifest_size_bytes,
            development_identity.scene_manifest_sha256,
        ),
        paths.latent_bundle: FileIdentity(
            development_identity.latent_bundle_size_bytes,
            development_identity.latent_bundle_sha256,
        ),
        paths.k2_receipt: FileIdentity(
            development_identity.k2_receipt_size_bytes,
            development_identity.k2_receipt_sha256,
        ),
        paths.trainable_checkpoint: FileIdentity(
            identity.trainable_checkpoint_size_bytes,
            identity.trainable_checkpoint_sha256,
        ),
    }
    guides = tuple(torch.zeros((1, 128, 1, 12, 12)) for _ in range(9))
    development = _Development(guides)
    development.scene_manifest_sha256 = development_identity.scene_manifest_sha256
    development.latent_bundle_sha256 = development_identity.latent_bundle_sha256
    development.k2_receipt_sha256 = development_identity.k2_receipt_sha256
    development.source_video_sha256 = development_identity.source_video_sha256
    development.latent_sha256 = development_identity.ordered_latent_sha256
    inventory = GemmaInventoryReceipt.capture(
        paths.gemma_root,
        (GemmaInventoryEntry("config.json", 1, _digest("gemma")),),
    )
    disjointness = DevelopmentDisjointnessReceipt(
        "duet-x-ltx-development-disjointness-v1",
        development.scene_id,
        development.split,
        development.source_relative_path,
        development.source_video_sha256,
        development.latent_sha256,
        _digest("confirmatory-scenes"),
        True,
    ).validate()
    source_identity = _digest("official-source")
    api = cast(OfficialLTXMaterializationAPI, argparse.Namespace())

    class OfficialFactory:
        checkout_root = Path(identity.ltx_checkout_root)
        pipelines_source_root = Path(identity.ltx_pipelines_source_root)
        core_source_root = Path(identity.ltx_core_source_root)
        dependency_source_root = Path(identity.dependency_source_root)

        def load(self) -> OfficialLTXMaterializationAPI:
            return api

        def reauthenticate_engine_api(self) -> str:
            return source_identity

        def checked_importer(self, _name: str) -> object:
            return object()

    calls: list[tuple[str, Path, dict[str, object]]] = []
    media_facts = DecodedMediaFacts(384, 384, 25, 24.0, 25.0 / 24.0)

    def probe(path: Path, **kwargs: object) -> DecodedMediaFacts:
        calls.append(("probe", path, kwargs))
        return media_facts

    def decoder(path: Path, **kwargs: object) -> object:
        calls.append(("decode", path, kwargs))
        return object()

    def digest(path: Path, **kwargs: object) -> str:
        calls.append(("digest", path, kwargs))
        probe_callback = kwargs["probe"]
        decoder_callback = kwargs["decoder"]
        assert callable(probe_callback)
        assert callable(decoder_callback)
        probe_callback(path)
        decoder_callback(path)
        return _digest("decoded-frames")

    monkeypatch.setattr(quality_runtime, "probe_decoded_video", probe)
    monkeypatch.setattr(quality_runtime, "decode_rgb24_frames", decoder)
    monkeypatch.setattr(quality_runtime, "decoded_frames_sha256", digest)
    engine_arguments: dict[str, object] = {}
    engine = _Engine(identity.expected_foundation_sha256)

    def build_engine(**kwargs: object) -> _Engine:
        engine_arguments.update(kwargs)
        return engine

    prepared = PinnedQualityDevelopmentFactory(
        identity=identity,
        development_identity=development_identity,
        official_factory=cast(PinnedOfficialLTXFactory, OfficialFactory()),
        expected_gemma_inventory=inventory,
        file_identity=lambda path: facts[path],
        executable_identity=lambda path: facts[path],
        gemma_inventory_reader=lambda _path: inventory,
        directory_check=lambda _path: True,
        runtime_facts=lambda: ObservedRuntimeFacts(
            identity.python_executable,
            identity.python_version,
            identity.torch_version,
            identity.device,
        ),
        development_loader=lambda *_paths: cast(Any, development),
        disjointness_validator=lambda *_parents: disjointness,
        checkpoint_loader=lambda *_args, **_kwargs: _modules(),
        engine_factory=cast(Any, build_engine),
    ).prepare(
        paths=paths,
        observed_source_commit=identity.source_commit,
        observed_source_archive_sha256=identity.source_archive_sha256,
        clusters=(),
        confirmatory_ancestry=(),
    )

    media_path = (tmp_path / "generated.mp4").resolve()
    media_path.write_bytes(b"media")
    assert engine_arguments["media_probe"] is prepared.adapter.media_probe
    assert prepared.adapter.media_probe(media_path) == media_facts
    assert prepared.adapter.decoded_frames_digest(media_path) == _digest("decoded-frames")
    probe_calls = [row for row in calls if row[0] == "probe"]
    decode_calls = [row for row in calls if row[0] == "decode"]
    assert all(
        row[2]
        == {
            "ffprobe_path": Path(identity.ffprobe_path),
            "ffprobe_sha256": identity.ffprobe_sha256,
            "descriptor_stable": True,
        }
        for row in probe_calls
    )
    assert len(decode_calls) == 1
    assert decode_calls[0][2]["ffmpeg_path"] == Path(identity.ffmpeg_path)
    assert decode_calls[0][2]["ffmpeg_sha256"] == identity.ffmpeg_sha256
    assert decode_calls[0][2]["descriptor_stable"] is True


def test_frozen_cpu_stream_identity_reaches_the_media_guard_input() -> None:
    identity = FrozenQualityRuntimeIdentity.default(
        source_commit="1" * 40,
        source_archive_sha256="2" * 64,
    )

    engine = quality_runtime._engine_factory(
        teacher_checkpoint_path=Path(identity.teacher_checkpoint_path),
        teacher_checkpoint_file_sha256=identity.teacher_checkpoint_sha256,
        gemma_root=Path(identity.gemma_root),
        expected_teacher_foundation_sha256=identity.expected_foundation_sha256,
        device=torch.device(identity.device),
        importer=lambda _name: None,
    )

    assert isinstance(engine, media.PinnedLTXOneStageMediaEngine)
    assert (
        engine._expected_teacher_foundation_sha256
        == "c6cf979445daeae6672c6d565cec48fbf6a7212fb8e0a4ed58cd5f4ba42df65f"
    )


def test_public_gemma_inventory_loader_and_disk_verifier_are_exact(tmp_path: Path) -> None:
    gemma_root = tmp_path / "gemma"
    gemma_root.mkdir()
    (gemma_root / "config.json").write_bytes(b"config")
    (gemma_root / "model.safetensors").write_bytes(b"weights")
    receipt = capture_gemma_inventory(gemma_root)
    receipt_path = tmp_path / "gemma-inventory.json"
    receipt_path.write_bytes(receipt.to_json())
    receipt_path.chmod(0o600)

    loaded = load_gemma_inventory_receipt(receipt_path)

    assert loaded == receipt
    assert verify_gemma_inventory(gemma_root, loaded) == receipt


@pytest.mark.parametrize("unsafe", ["mode", "symlink", "hardlink", "noncanonical"])
def test_gemma_inventory_loader_rejects_unsafe_receipts(tmp_path: Path, unsafe: str) -> None:
    root = (tmp_path / "gemma").resolve()
    root.mkdir()
    (root / "config.json").write_bytes(b"config")
    receipt = capture_gemma_inventory(root)
    receipt_path = tmp_path / "gemma-inventory.json"
    receipt_path.write_bytes(receipt.to_json())
    receipt_path.chmod(0o600)
    candidate = receipt_path
    if unsafe == "mode":
        receipt_path.chmod(0o644)
    elif unsafe == "symlink":
        candidate = tmp_path / "receipt-link.json"
        candidate.symlink_to(receipt_path)
    elif unsafe == "hardlink":
        candidate = tmp_path / "receipt-hardlink.json"
        candidate.hardlink_to(receipt_path)
    else:
        receipt_path.write_bytes(receipt.to_json() + b"\n")

    with pytest.raises(ValueError, match="Gemma inventory"):
        load_gemma_inventory_receipt(candidate)


@pytest.mark.parametrize("drift", ["missing", "extra", "content", "hardlink"])
def test_gemma_inventory_verifier_rejects_every_disk_drift(tmp_path: Path, drift: str) -> None:
    root = (tmp_path / "gemma").resolve()
    root.mkdir()
    config = root / "config.json"
    config.write_bytes(b"config")
    receipt = capture_gemma_inventory(root)
    if drift == "missing":
        config.unlink()
    elif drift == "extra":
        (root / "extra.json").write_bytes(b"extra")
    elif drift == "content":
        config.write_bytes(b"drift")
    else:
        (tmp_path / "hardlink.json").hardlink_to(config)

    with pytest.raises(ValueError, match="Gemma inventory"):
        verify_gemma_inventory(root, receipt)


def test_test_fakes_match_the_pinned_signature_shapes() -> None:
    assert tuple(inspect.signature(_preprocess).parameters) == (
        "frames",
        "height",
        "width",
        "dtype",
        "device",
    )


def test_official_api_constructs_only_the_strict_materializer(tmp_path: Path) -> None:
    modules, origins = _official_modules(tmp_path)
    identity = FrozenQualityRuntimeIdentity.default(
        source_commit="1" * 40, source_archive_sha256="2" * 64
    )
    api = _official_factory(tmp_path, modules, origins).load()

    materializer = api.build_materializer(
        identity,
        device=torch.device("cuda"),
        file_identity=lambda _path: FileIdentity(
            identity.teacher_checkpoint_size_bytes, identity.teacher_checkpoint_sha256
        ),
        directory_check=lambda _path: True,
    )

    assert isinstance(materializer, PinnedLTXRGBMaterializer)
    conditioner = cast(_Conditioner, materializer.image_conditioner)
    assert conditioner.arguments[:3] == (
        identity.teacher_checkpoint_path,
        torch.bfloat16,
        torch.device("cuda"),
    )
    with pytest.raises(ValueError, match="cuda"):
        api.build_materializer(
            identity,
            device=torch.device("cpu"),
            file_identity=lambda _path: FileIdentity(
                identity.teacher_checkpoint_size_bytes, identity.teacher_checkpoint_sha256
            ),
            directory_check=lambda _path: True,
        )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _read_identity(values: dict[Path, FileIdentity], path: Path) -> FileIdentity:
    return values[path]


def _observed_runtime(value: ObservedRuntimeFacts) -> ObservedRuntimeFacts:
    return value


class _TorchVersionLike(str):
    pass


def test_runtime_facts_normalizes_torch_version_to_builtin_str(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "__version__", _TorchVersionLike("2.8.0+cu128"))

    observed = quality_runtime._runtime_facts()

    assert type(observed.torch_version) is str
    assert observed.torch_version == "2.8.0+cu128"


class _Gated(torch.nn.Module):
    def compress(self, history: torch.Tensor) -> torch.Tensor:
        return history.mean(dim=1)


class _Bridge(torch.nn.Module):
    def forward_exclusion(self, history: torch.Tensor, *, candidates: object) -> object:
        del candidates

        class Result:
            core_latent = history[:, (0, 1, 3, 4, 6, 7)].sum(dim=1)
            excluded_leaf_keys = tuple(type("Key", (), {"slot": slot})() for slot in (2, 5))

        return Result()


def _modules() -> FrozenMethodModules:
    gated = _Gated().eval().requires_grad_(False)
    bridge = _Bridge().eval().requires_grad_(False)
    return FrozenMethodModules.capture(
        FROZEN_TRAINABLE_CHECKPOINT_SHA256,
        gated,
        bridge,
        torch.device("cpu"),
        expected_gated_type=_Gated,
        expected_bridge_type=_Bridge,
    )


class _Development:
    def __init__(self, guides: tuple[torch.Tensor, ...]) -> None:
        self.scene_id = "development-00"
        self.split = "development"
        self.prompt = "A video of a marching band."
        self.source_relative_path = "train/BandMarching/v_BandMarching_g06_c01.avi"
        self.source_video_sha256 = _digest("source")
        self.protected_slots = (2, 5)
        self.current_slot = 8
        self.latent_bundle_sha256 = _digest("bundle")
        self.latent_sha256 = tuple(tensor_sha256(guide) for guide in guides)
        self.latents = guides
        self.scene_manifest_sha256 = _digest("scene")
        self.k2_receipt_sha256 = _digest("k2")

    def validate(self) -> _Development:
        return self


class _Engine:
    def __init__(
        self,
        foundation: str,
        *,
        offload_mode: str = "cpu",
        foundation_guard_sha256: str = media.STREAMING_FOUNDATION_GUARD_SHA256,
    ) -> None:
        self.foundation = foundation
        self.offload_mode = offload_mode
        self.foundation_guard_sha256 = foundation_guard_sha256
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        path = kwargs["output_path"]
        assert isinstance(path, Path)
        path.write_bytes(b"opaque")
        final = torch.zeros((1, 128, 4, 12, 12), dtype=torch.float32)
        return argparse.Namespace(
            validate=lambda: argparse.Namespace(
                path=path,
                final_latent=final,
                final_latent_sha256=tensor_sha256(final),
                checkpoint_file_sha256=(
                    "7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f"
                ),
                foundation_before_sha256=self.foundation,
                foundation_after_sha256=self.foundation,
                offload_mode=self.offload_mode,
                foundation_guard_sha256=self.foundation_guard_sha256,
                media_sha256=hashlib.sha256(b"opaque").hexdigest(),
            )
        )


def _synthetic_adapter(
    engine: _Engine,
    *,
    source_revalidator: object = None,
    source_identity_sha256: str | None = None,
) -> DevelopmentForwardAdapter:
    guides = tuple(
        torch.full((1, 128, 1, 12, 12), float(slot + 1), dtype=torch.float32) for slot in range(9)
    )
    development = _Development(guides)
    scene = _DevelopmentGuideScene.build(
        development=development,
        preprocessed_sha256=tuple(_digest(f"pre-{slot}") for slot in range(9)),
    )
    kwargs: dict[str, object] = {}
    if source_revalidator is not None:
        kwargs["source_revalidator"] = source_revalidator
        kwargs["source_identity_sha256"] = source_identity_sha256
    return DevelopmentForwardAdapter(
        development=development,  # type: ignore[arg-type]
        scene=scene,
        modules=_modules(),
        engine=engine,  # type: ignore[arg-type]
        runtime=RuntimeCoordinate.default(),
        engine_identity_sha256=_digest("engine"),
        foundation_sha256=engine.foundation,
        runtime_identity_sha256=_digest("runtime"),
        decoded_frames_digest=lambda _path: _digest("decoded"),
        media_probe=lambda _path: DecodedMediaFacts(384, 384, 25, 24.0, 25.0 / 24.0),
        synchronize=lambda: None,
        empty_cache=lambda: None,
        **kwargs,  # type: ignore[arg-type]
    )


def test_development_adapter_reuses_exact_four_method_rosters_and_hashes() -> None:
    engine = _Engine(_digest("foundation"))
    adapter = _synthetic_adapter(engine)
    expected_slots = {
        Method.FULL_HISTORY: tuple(str(slot) for slot in range(9)),
        Method.RECENT_ANCHOR: ("7", "2", "5", "8"),
        Method.GATED_CORE: ("method_core[0,1,3,4,6,7]", "2", "5", "8"),
        Method.DUET_CORE: ("method_core[0,1,3,4,6,7]", "2", "5", "8"),
    }

    for method in Method:
        plan = adapter.plan(method, 101)
        assert plan.prompt == "A video of a marching band."
        assert plan.negative_prompt == ""
        assert plan.protected_slots == (2, 5)
        assert plan.current_slot == 8
        assert plan.core_slots == (0, 1, 3, 4, 6, 7)
        assert plan.ordered_slot_ids == expected_slots[method]
        assert len(plan.ordered_guide_sha256) == (9 if method is Method.FULL_HISTORY else 4)
        if method in {Method.GATED_CORE, Method.DUET_CORE}:
            assert plan.method_core_sha256 == plan.ordered_guide_sha256[0]
        else:
            assert plan.method_core_sha256 is None
        assert plan.initial_noise_sha256 == canonical_sha256(
            {
                "format": "duet-x-initial-noise-identity-v1",
                "scene_id": "development-00",
                "seed": 101,
                "runtime_coordinate_sha256": RuntimeCoordinate.default().fingerprint(),
            }
        )

    scene = adapter.scene
    assert not isinstance(scene, SceneVariant)
    assert not hasattr(scene, "to_json")
    with pytest.raises((TypeError, ValueError)):
        SceneVariant.from_dict(scene)


def test_development_adapter_executes_official_engine_and_disposes_safely(
    tmp_path: Path,
) -> None:
    foundation = _digest("foundation")
    engine = _Engine(foundation)
    events: list[str] = []
    adapter = _synthetic_adapter(engine)
    adapter.synchronize = lambda: events.append("synchronize")
    adapter.empty_cache = lambda: events.append("empty_cache")
    output = (tmp_path / "development.mp4").resolve()

    result = adapter.forward(Method.DUET_CORE, 101, output)

    assert isinstance(result, DevelopmentForwardResult)
    assert len(engine.calls) == 1
    assert engine.calls[0]["prompt"] == "A video of a marching band."
    assert engine.calls[0]["negative_prompt"] == ""
    assert engine.calls[0]["seed"] == 101
    assert engine.calls[0]["output_path"] == output
    assert result.decoded_frames_sha256 == _digest("decoded")
    adapter.close()
    assert events == ["synchronize", "empty_cache"]
    with pytest.raises(RuntimeError, match="closed"):
        adapter.forward(Method.DUET_CORE, 101, output)


def test_runtime_owns_immutable_forward_binding() -> None:
    adapter = _synthetic_adapter(_Engine(_digest("foundation")))

    binding = adapter.binding(Method.DUET_CORE, 101)

    assert type(binding).__module__ == "comfy_story.ltx_quality_runtime"
    with pytest.raises(dataclasses.FrozenInstanceError):
        binding.prompt = "changed"  # type: ignore[misc]


def test_materializer_uses_the_model_paths_video_vae_without_monolith_equality() -> None:
    class DistinctPaths(_Paths):
        @classmethod
        def from_monolith(
            cls,
            checkpoint_path: str,
            gemma_root: str | None = None,
            *,
            video_vae_path: str | None = None,
        ) -> DistinctPaths:
            del checkpoint_path, gemma_root, video_vae_path
            return cls()

        def video_vae(self) -> str:
            return "/frozen/video_vae.json"

    api = OfficialLTXMaterializationAPI(_preprocess, _Conditioner, DistinctPaths)
    identity = FrozenQualityRuntimeIdentity.default(
        source_commit="1" * 40, source_archive_sha256="2" * 64
    )

    materializer = api.build_materializer(
        identity,
        device=torch.device("cuda"),
        file_identity=lambda _path: FileIdentity(
            identity.teacher_checkpoint_size_bytes, identity.teacher_checkpoint_sha256
        ),
        directory_check=lambda _path: True,
    )

    conditioner = cast(_Conditioner, materializer.image_conditioner)
    assert conditioner.arguments[0] == "/frozen/video_vae.json"


def test_forward_rebuilds_guides_and_rejects_stale_live_tensors(tmp_path: Path) -> None:
    engine = _Engine(_digest("foundation"))
    adapter = _synthetic_adapter(engine)
    plan = adapter.plan(Method.DUET_CORE, 101)
    assert not hasattr(adapter, "guides")

    adapter.forward(Method.DUET_CORE, 101, (tmp_path / "development.mp4").resolve())
    adapter.forward(Method.DUET_CORE, 101, (tmp_path / "development-2.mp4").resolve())

    first = cast(tuple[torch.Tensor, ...], engine.calls[0]["guides"])
    second = cast(tuple[torch.Tensor, ...], engine.calls[1]["guides"])
    assert tuple(tensor_sha256(value) for value in first) == plan.ordered_guide_sha256
    assert tuple(tensor_sha256(value) for value in second) == plan.ordered_guide_sha256
    assert all(left is not right for left, right in zip(first, second, strict=True))


def test_close_drops_every_model_and_tensor_reference() -> None:
    adapter = _synthetic_adapter(_Engine(_digest("foundation")))

    adapter.close()

    assert adapter.modules is None
    assert adapter.engine is None
    assert adapter._cache is None
    assert adapter._latents == ()


def test_forward_reauthenticates_source_and_failure_closes_every_path(tmp_path: Path) -> None:
    engine = _Engine(_digest("foundation"))
    source = [_digest("source-auth")]
    adapter = _synthetic_adapter(
        engine,
        source_revalidator=lambda: source[0],
        source_identity_sha256=source[0],
    )
    adapter.binding(Method.DUET_CORE, 101)
    source[0] = _digest("drift")

    with pytest.raises(ValueError, match="source identity"):
        adapter.forward(Method.DUET_CORE, 101, (tmp_path / "development.mp4").resolve())

    assert engine.calls == []
    assert adapter.engine is None
    assert adapter.modules is None
    with pytest.raises(RuntimeError, match="closed"):
        adapter.binding(Method.DUET_CORE, 101)


@pytest.mark.parametrize(
    ("offload_mode", "guard_sha256"),
    [("none", media.STREAMING_FOUNDATION_GUARD_SHA256), ("cpu", "0" * 64)],
)
def test_forward_rejects_streaming_execution_identity_drift(
    tmp_path: Path, offload_mode: str, guard_sha256: str
) -> None:
    engine = _Engine(
        _digest("foundation"),
        offload_mode=offload_mode,
        foundation_guard_sha256=guard_sha256,
    )
    adapter = _synthetic_adapter(engine)

    with pytest.raises(ValueError, match="engine identity"):
        adapter.forward(Method.DUET_CORE, 101, (tmp_path / "drift.mp4").resolve())


def test_official_factory_never_accepts_a_stale_cached_commit(tmp_path: Path) -> None:
    modules, origins = _official_modules(tmp_path)
    observed = [LTX_SOURCE_COMMIT]
    factory = PinnedOfficialLTXFactory(
        checkout_root=tmp_path,
        pipelines_source_root=tmp_path / "packages/ltx-pipelines/src",
        core_source_root=tmp_path / "packages/ltx-core/src",
        dependency_source_root=tmp_path / "deps",
        importer=lambda name: modules[name],
        symbol_origin=lambda symbol: origins[id(symbol)],
        checkout_state_reader=lambda _root: CheckoutState(observed[0], True),
        directory_check=lambda _path: True,
    )
    factory.load()
    observed[0] = "0" * 40

    with pytest.raises(ValueError, match="commit"):
        factory.load()


def test_all_identity_and_disjointness_failures_precede_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = FrozenQualityRuntimeIdentity.default(
        source_commit="1" * 40, source_archive_sha256="2" * 64
    )
    monkeypatch.setattr(sys, "executable", identity.python_executable)
    monkeypatch.setattr(
        platform,
        "python_version",
        lambda: identity.python_version,
    )
    monkeypatch.setattr(torch, "__version__", _TorchVersionLike(identity.torch_version))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    development_identity = FrozenDevelopmentIdentity.default()
    paths = DevelopmentRuntimePaths(
        teacher_checkpoint=Path(identity.teacher_checkpoint_path),
        gemma_root=Path(identity.gemma_root),
        scene_manifest=tmp_path / "development-00.json",
        latent_bundle=tmp_path / "development-00.pt",
        k2_receipt=tmp_path / "development-00-k2.json",
        trainable_checkpoint=tmp_path / "trainable-checkpoint.pt",
    )
    facts = {
        Path(identity.ffprobe_path): FileIdentity(
            identity.ffprobe_size_bytes, identity.ffprobe_sha256
        ),
        Path(identity.ffmpeg_path): FileIdentity(
            identity.ffmpeg_size_bytes, identity.ffmpeg_sha256
        ),
        paths.teacher_checkpoint: FileIdentity(
            identity.teacher_checkpoint_size_bytes, identity.teacher_checkpoint_sha256
        ),
        paths.scene_manifest: FileIdentity(
            development_identity.scene_manifest_size_bytes,
            development_identity.scene_manifest_sha256,
        ),
        paths.latent_bundle: FileIdentity(
            development_identity.latent_bundle_size_bytes,
            development_identity.latent_bundle_sha256,
        ),
        paths.k2_receipt: FileIdentity(
            development_identity.k2_receipt_size_bytes,
            development_identity.k2_receipt_sha256,
        ),
        paths.trainable_checkpoint: FileIdentity(
            identity.trainable_checkpoint_size_bytes,
            identity.trainable_checkpoint_sha256,
        ),
    }
    guides = tuple(torch.zeros((1, 128, 1, 12, 12)) for _ in range(9))
    development = _Development(guides)
    development.scene_manifest_sha256 = development_identity.scene_manifest_sha256
    development.latent_bundle_sha256 = development_identity.latent_bundle_sha256
    development.k2_receipt_sha256 = development_identity.k2_receipt_sha256
    development.source_video_sha256 = development_identity.source_video_sha256
    development.latent_sha256 = development_identity.ordered_latent_sha256
    loaded: list[str] = []
    gemma_inventory = GemmaInventoryReceipt.capture(
        paths.gemma_root,
        (GemmaInventoryEntry("config.json", 1, _digest("gemma")),),
    )
    factory = PinnedQualityDevelopmentFactory(
        identity=identity,
        development_identity=development_identity,
        official_factory=argparse.Namespace(  # type: ignore[arg-type]
            checkout_root=Path(identity.ltx_checkout_root),
            pipelines_source_root=Path(identity.ltx_pipelines_source_root),
            core_source_root=Path(identity.ltx_core_source_root),
            dependency_source_root=Path(identity.dependency_source_root),
            load=lambda: loaded.append("official"),
        ),
        expected_gemma_inventory=gemma_inventory,
        gemma_inventory_reader=lambda _path: gemma_inventory,
        file_identity=lambda path: facts[path],
        directory_check=lambda _path: True,
        development_loader=lambda *_args: development,  # type: ignore[arg-type]
        disjointness_validator=lambda *_args: (_ for _ in ()).throw(
            ValueError("development disjointness failed")
        ),
        checkpoint_loader=lambda *_args, **_kwargs: loaded.append("checkpoint"),  # type: ignore[arg-type]
        engine_factory=lambda **_kwargs: loaded.append("engine"),  # type: ignore[arg-type]
        executable_identity=lambda path: facts[path],
    )

    with pytest.raises(ValueError, match="disjointness"):
        factory.prepare(
            paths=paths,
            observed_source_commit=identity.source_commit,
            observed_source_archive_sha256=identity.source_archive_sha256,
            clusters=(),
            confirmatory_ancestry=(),
        )
    assert loaded == []

    for mutation in (
        "source_commit",
        "source_archive",
        "runtime",
        "teacher",
        "ffprobe",
        "ffmpeg",
        "gemma_inventory",
        "core_root",
        "development_file",
        "development_prompt",
    ):
        local_facts = dict(facts)
        observed_commit = identity.source_commit
        observed_archive = identity.source_archive_sha256
        runtime = ObservedRuntimeFacts(
            identity.python_executable,
            identity.python_version,
            identity.torch_version,
            identity.device,
        )
        development.prompt = development_identity.prompt
        core_source_root = Path(identity.ltx_core_source_root)
        observed_gemma = gemma_inventory
        if mutation == "source_commit":
            observed_commit = "3" * 40
        elif mutation == "source_archive":
            observed_archive = "4" * 64
        elif mutation == "runtime":
            runtime = dataclasses.replace(runtime, torch_version="drift")
        elif mutation == "teacher":
            local_facts[paths.teacher_checkpoint] = FileIdentity(
                identity.teacher_checkpoint_size_bytes, "5" * 64
            )
        elif mutation == "ffprobe":
            local_facts[Path(identity.ffprobe_path)] = FileIdentity(
                identity.ffprobe_size_bytes, "5" * 64
            )
        elif mutation == "ffmpeg":
            local_facts[Path(identity.ffmpeg_path)] = FileIdentity(
                identity.ffmpeg_size_bytes, "5" * 64
            )
        elif mutation == "gemma_inventory":
            observed_gemma = GemmaInventoryReceipt.capture(
                paths.gemma_root,
                (GemmaInventoryEntry("config.json", 2, _digest("gemma-drift")),),
            )
        elif mutation == "core_root":
            core_source_root = tmp_path / "site-packages"
        elif mutation == "development_file":
            local_facts[paths.scene_manifest] = FileIdentity(
                development_identity.scene_manifest_size_bytes, "6" * 64
            )
        else:
            development.prompt = "drift"
        loaded.clear()

        def valid_disjointness(
            _development: object,
            _clusters: object,
            _ancestry: object,
        ) -> DevelopmentDisjointnessReceipt:
            return cast(
                DevelopmentDisjointnessReceipt,
                argparse.Namespace(validate=lambda: None),
            )

        local = dataclasses.replace(
            factory,
            official_factory=argparse.Namespace(  # type: ignore[arg-type]
                checkout_root=Path(identity.ltx_checkout_root),
                pipelines_source_root=Path(identity.ltx_pipelines_source_root),
                core_source_root=core_source_root,
                dependency_source_root=Path(identity.dependency_source_root),
                load=lambda: loaded.append("official"),
            ),
            file_identity=functools.partial(_read_identity, local_facts),
            executable_identity=functools.partial(_read_identity, local_facts),
            runtime_facts=functools.partial(_observed_runtime, runtime),
            gemma_inventory_reader=functools.partial(_observed_gemma, observed_gemma),
            disjointness_validator=valid_disjointness,
        )
        with pytest.raises(ValueError, match="changed"):
            local.prepare(
                paths=paths,
                observed_source_commit=observed_commit,
                observed_source_archive_sha256=observed_archive,
                clusters=(),
                confirmatory_ancestry=(),
            )
        assert loaded == []
