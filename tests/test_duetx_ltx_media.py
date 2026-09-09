from __future__ import annotations

import dataclasses
import hashlib
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
import torch
from torch import nn

from duet.duetx import ltx_media as media
from duet.duetx.contracts import tensor_sha256
from duet.duetx.ltx_media import (
    PinnedLTXOneStageMediaEngine,
    PinnedLTXTwoStageMediaEngine,
)
from duet.duetx.media_probe import DecodedMediaFacts
from duet.duetx.training import _foundation_items_digest, foundation_digest


class _ModelPaths:
    @classmethod
    def from_monolith(cls, checkpoint: str, gemma: str) -> tuple[str, str]:
        return checkpoint, gemma


class _Lora:
    def __init__(self, path: str, strength: float, mapping: object) -> None:
        self.path = path
        self.strength = strength
        self.mapping = mapping


class _Reference:
    def __init__(self, **kwargs: object) -> None:
        self.values = kwargs


class _Attention:
    def __init__(self, reference: _Reference, *, attention_mask: float) -> None:
        self.reference = reference
        self.attention_mask = attention_mask


class _Guider:
    pass


class _OffloadMode(Enum):
    NONE = "none"
    CPU = "cpu"
    DISK = "disk"


class _ImageConditioner:
    def resolve_crf(self, images: object) -> object:
        return images

    def __call__(self, _encoder: object) -> list[object]:
        return []


class _Stage:
    def __init__(self) -> None:
        self._wrapper: Any = None

    def with_model_wrapper(self, wrapper: object) -> _Stage:
        result = _Stage()
        result._wrapper = wrapper
        return result

    def __call__(self) -> None:
        assert callable(self._wrapper)
        model = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(model.weight)
        self._wrapper(model, object())


class _Pipeline:
    references: ClassVar[list[object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.image_conditioner = _ImageConditioner()
        self.stage_1 = _Stage()
        self.stage_2 = _Stage()

    def __call__(self, **kwargs: object) -> object:
        assert kwargs["width"] == 384
        assert kwargs["height"] == 384
        assert kwargs["num_frames"] == 25
        assert kwargs["frame_rate"] == 24.0
        assert kwargs["num_inference_steps"] == 30
        self.image_conditioner.resolve_crf([])
        _Pipeline.references = self.image_conditioner(object())
        assert self.image_conditioner(object()) == []
        self.stage_1()
        self.stage_2()
        return SimpleNamespace(
            video=iter((torch.ones(1, 1, 1, 3),)),
            audio=None,
            num_frames=25,
            tiling_config=object(),
            video_latent=torch.ones(1, 128, 4, 12, 12),
        )


def test_pinned_two_stage_media_uses_official_path_and_stage_one_guides(tmp_path: Path) -> None:
    files = []
    for name in ("teacher.safetensors", "lora.safetensors", "upscaler.safetensors"):
        path = (tmp_path / name).resolve()
        path.write_bytes(name.encode())
        files.append(path)
    gemma = (tmp_path / "gemma").resolve()
    gemma.mkdir()
    expected_model = nn.Linear(2, 2, bias=False)
    nn.init.zeros_(expected_model.weight)
    expected = foundation_digest(expected_model)
    encoded: list[dict[str, object]] = []
    params = SimpleNamespace(video_guider_params=_Guider(), audio_guider_params=_Guider())
    modules = {
        "ltx_pipelines": SimpleNamespace(TI2VidTwoStagesPipeline=_Pipeline),
        "ltx_pipelines.utils.model_paths": SimpleNamespace(ModelPaths=_ModelPaths),
        "ltx_core.loader": SimpleNamespace(
            LoraPathStrengthAndSDOps=_Lora,
            LTXV_LORA_COMFY_RENAMING_MAP=object(),
        ),
        "ltx_core.conditioning": SimpleNamespace(
            VideoConditionByReferenceLatent=_Reference,
            ConditioningItemAttentionStrengthWrapper=_Attention,
        ),
        "ltx_core.components.guiders": SimpleNamespace(MultiModalGuiderParams=_Guider),
        "ltx_pipelines.utils.constants": SimpleNamespace(detect_params=lambda _path: params),
        "ltx_core.model.video_vae": SimpleNamespace(
            get_video_chunks_number=lambda frames, tiling: 1
        ),
        "ltx_pipelines.utils.media_io": SimpleNamespace(),
    }

    def encode_video(**kwargs: object) -> None:
        encoded.append(kwargs)
        Path(str(kwargs["output_path"])).write_bytes(b"mp4")

    modules["ltx_pipelines.utils.media_io"].encode_video = encode_video
    output = (tmp_path / "cell.mp4").resolve()
    engine = PinnedLTXTwoStageMediaEngine(
        teacher_checkpoint_path=files[0],
        teacher_checkpoint_file_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(),
        distilled_lora_path=files[1],
        distilled_lora_file_sha256=hashlib.sha256(files[1].read_bytes()).hexdigest(),
        spatial_upscaler_path=files[2],
        spatial_upscaler_file_sha256=hashlib.sha256(files[2].read_bytes()).hexdigest(),
        gemma_root=gemma,
        expected_teacher_foundation_sha256=expected,
        device=torch.device("cpu"),
        importer=lambda name: modules[name],
        media_probe=lambda _path: DecodedMediaFacts(384, 384, 25, 24.0, 25 / 24),
    )

    result = engine.generate(
        guides=(torch.ones(1, 128, 1, 6, 6),) * 4,
        prompt="prompt",
        negative_prompt="negative",
        seed=101,
        output_path=output,
    )

    assert output.read_bytes() == b"mp4"
    assert len(_Pipeline.references) == 4
    assert all(isinstance(item, _Attention) for item in _Pipeline.references)
    assert len(encoded) == 1
    assert result.teacher_foundation_before_sha256 == expected
    assert result.duration_seconds == 25 / 24
    assert hashlib.sha256(output.read_bytes()).hexdigest() == hashlib.sha256(b"mp4").hexdigest()


def test_pinned_two_stage_media_rejects_non_stage_one_guide_shape(tmp_path: Path) -> None:
    engine = PinnedLTXTwoStageMediaEngine(
        teacher_checkpoint_path=tmp_path / "missing",
        teacher_checkpoint_file_sha256="0" * 64,
        distilled_lora_path=tmp_path / "missing",
        distilled_lora_file_sha256="0" * 64,
        spatial_upscaler_path=tmp_path / "missing",
        spatial_upscaler_file_sha256="0" * 64,
        gemma_root=tmp_path,
        expected_teacher_foundation_sha256="0" * 64,
        device=torch.device("cpu"),
        importer=lambda _name: object(),
    )
    try:
        engine.generate(
            guides=(torch.ones(1, 128, 1, 5, 6),),
            prompt="prompt",
            negative_prompt="",
            seed=101,
            output_path=(tmp_path / "cell.mp4").resolve(),
        )
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("invalid guide shape was accepted")


class _OneStageFoundation(nn.Module):
    mutate: ClassVar[bool] = False
    exact_alias_observed: ClassVar[bool] = False
    inference_resident_exact_alias: ClassVar[bool] = False
    inference_resident: ClassVar[bool] = False
    inference_resident_views: ClassVar[bool] = False
    mutate_resident_transiently: ClassVar[bool] = False
    mutate_source_before_teardown: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        if type(self).inference_resident:
            with torch.inference_mode():
                self.resident = nn.Parameter(torch.zeros(1))
        else:
            self.resident = nn.Parameter(torch.zeros(1))
        if type(self).inference_resident_exact_alias:
            self.register_parameter("resident_alias", self.resident)
        if type(self).inference_resident_views:
            with torch.inference_mode():
                shared = torch.zeros(4)
                self.register_buffer("resident_left", shared[:3])
                self.register_buffer("resident_right", shared[1:])
        self.transformer_blocks = nn.ModuleList((_OneStageBlock(), _OneStageBlock()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if type(self).inference_resident_exact_alias:
            type(self).exact_alias_observed = self.resident is self.resident_alias
        if type(self).mutate_resident_transiently:
            self.resident.add_(1)
            value = value + self.resident
            self.resident.sub_(1)
        for block in self.transformer_blocks:
            value = block(value)
        return value


class _OneStageBlock(nn.Module):
    mutate_transiently: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if type(self).mutate_transiently:
            self.weight.add_(1)
            result = value + self.weight
            self.weight.sub_(1)
            return result
        if _OneStageFoundation.mutate:
            with torch.no_grad():
                self.weight.add_(1)
        return value + self.weight


class _ExpectedX0Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velocity_model = _OneStageFoundation()


class _PinnedBlock:
    def __init__(
        self, buffer: torch.Tensor, layout: dict[str, tuple[torch.Size, torch.dtype]]
    ) -> None:
        self.buffer = buffer
        self.layout = layout


class _PinnedWeightSource:
    cleanup_calls: ClassVar[int] = 0

    def __init__(self, block_count: int) -> None:
        layout = {"weight": (torch.Size((1,)), torch.float32)}
        self._blocks = {
            index: _PinnedBlock(torch.zeros(1, dtype=torch.float32).view(torch.uint8), layout)
            for index in range(block_count)
        }

    def block_layout(self, index: int) -> dict[str, tuple[torch.Size, torch.dtype]]:
        return self._blocks[index].layout

    @property
    def slot_nbytes(self) -> int:
        return max(block.buffer.numel() for block in self._blocks.values())

    def get(self, index: int) -> torch.Tensor:
        return self._blocks[index].buffer

    def cleanup(self) -> None:
        type(self).cleanup_calls += 1
        self._blocks.clear()


class _DiskWeightSource(_PinnedWeightSource):
    pass


class _WeightsProvider:
    def __init__(self, source: _PinnedWeightSource) -> None:
        self._source = source
        self._lora_sources: list[object] = []
        self._target_device = torch.device("cpu")
        self._pool = object()
        self._sync = object()

    def cleanup(self) -> None:
        self._source.cleanup()


def _layout_nbytes(layout: object) -> int:
    assert isinstance(layout, dict)
    return sum(shape.numel() * dtype.itemsize for shape, dtype in layout.values())


def _carve_buffer(buffer: torch.Tensor, layout: object) -> dict[str, torch.Tensor]:
    assert isinstance(layout, dict)
    assert list(layout) == ["weight"]
    shape, dtype = layout["weight"]
    return {"weight": buffer.view(dtype).reshape(shape)}


class _BlockStreamingWrapper(nn.Module):
    teardown_calls: ClassVar[int] = 0
    dispose_calls: ClassVar[int] = 0
    use_disk_source: ClassVar[bool] = False
    use_lora: ClassVar[bool] = False
    swap_provider: ClassVar[bool] = False
    swap_source: ClassVar[bool] = False
    missing_layout: ClassVar[bool] = False
    wrong_raw_buffer: ClassVar[bool] = False
    missing_nonblock: ClassVar[bool] = False
    extra_nonblock: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        self._model = _OneStageFoundation()
        if type(self).missing_nonblock:
            del self._model._parameters["resident"]
        if type(self).extra_nonblock:
            self._model.register_buffer("extra", torch.zeros(1))
        source_type = _DiskWeightSource if type(self).use_disk_source else _PinnedWeightSource
        source = source_type(len(self._model.transformer_blocks))
        if type(self).missing_layout:
            source._blocks[0].layout = {}
        if type(self).wrong_raw_buffer:
            source._blocks[0].buffer = torch.zeros(1, dtype=torch.float32)
        self._provider = _WeightsProvider(source)
        if type(self).use_lora:
            self._provider._lora_sources.append(object())
        self._target_device = torch.device("cpu")
        self._provider_hooks = [
            block.register_forward_pre_hook(
                lambda module, _inputs, index=index: setattr(
                    module,
                    "weight",
                    nn.Parameter(
                        _carve_buffer(
                            self._provider._source.get(index),
                            self._provider._source.block_layout(index),
                        )["weight"],
                        requires_grad=False,
                    ),
                )
            )
            for index, block in enumerate(self._model.transformer_blocks)
        ]

    @property
    def num_blocks(self) -> int:
        return len(self._model.transformer_blocks)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self._model(value))

    def teardown(self) -> None:
        type(self).teardown_calls += 1
        for handle in self._provider_hooks:
            handle.remove()
        self._provider_hooks.clear()
        self._provider.cleanup()

    def dispose(self) -> None:
        type(self).dispose_calls += 1
        for block in self._model.transformer_blocks:
            block.weight = nn.Parameter(torch.empty(1, device="meta"), requires_grad=False)


class _X0Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velocity_model = _BlockStreamingWrapper()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.velocity_model(value))


class _OneStageStage:
    no_forward: ClassVar[bool] = False
    double_build: ClassVar[bool] = False

    def __init__(self, wrapper: object | None = None) -> None:
        self._wrapper: Any = wrapper

    def with_model_wrapper(self, wrapper: object) -> _OneStageStage:
        return _OneStageStage(wrapper)

    def __call__(self) -> None:
        assert callable(self._wrapper)
        candidate = _X0Model()
        streaming = candidate.velocity_model
        try:
            model = self._wrapper(candidate, object())
            assert isinstance(model, nn.Module)
            if type(self).double_build:
                self._wrapper(_X0Model(), object())
            if _BlockStreamingWrapper.swap_provider:
                streaming._provider = _WeightsProvider(_PinnedWeightSource(streaming.num_blocks))
            if _BlockStreamingWrapper.swap_source:
                streaming._provider._source = _PinnedWeightSource(streaming.num_blocks)
            if not type(self).no_forward:
                model(torch.zeros(1))
                model(torch.zeros(1))
            if _OneStageFoundation.mutate_source_before_teardown:
                streaming._provider._source.get(0).data[0] = 1
        finally:
            streaming.teardown()
            streaming.dispose()


class _OneStagePipeline:
    calls: ClassVar[list[dict[str, object]]] = []
    references: ClassVar[list[object]] = []
    fail: ClassVar[bool] = False
    grad_enabled_observed: ClassVar[bool] = True
    mutate_references: ClassVar[bool] = False

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        if kwargs.get("offload_mode") is not _OffloadMode.CPU:
            raise RuntimeError("one-stage pipeline did not pin CPU offload")
        self.dtype = torch.bfloat16
        self.image_conditioner = _ImageConditioner()
        self.stage = _OneStageStage()

    def __call__(self, **kwargs: object) -> object:
        type(self).grad_enabled_observed = torch.is_grad_enabled()
        type(self).calls.append(kwargs)
        self.image_conditioner.resolve_crf([])
        type(self).references = self.image_conditioner(object())
        if type(self).mutate_references:
            reference = cast(_Attention, type(self).references[0]).reference
            latent = cast(torch.Tensor, reference.values["latent"])
            latent.add_(1)
        self.stage()
        if type(self).fail:
            raise RuntimeError("injected pipeline failure")
        seed = kwargs["seed"]
        assert isinstance(seed, int)
        ordered_hashes = [
            tensor_sha256(cast(torch.Tensor, cast(_Attention, item).reference.values["latent"]))
            for item in type(self).references
        ]
        seed_material = (
            str(seed).encode("ascii")
            + b"\0"
            + b"\0".join(item.encode("ascii") for item in ordered_hashes)
        )
        derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        generator = torch.Generator(device="cpu").manual_seed(derived_seed)
        latent = torch.rand((1, 128, 4, 12, 12), generator=generator, dtype=torch.float32)
        return SimpleNamespace(
            video=iter((torch.ones(1, 1, 1, 3),)),
            audio=None,
            num_frames=25,
            tiling_config=object(),
            video_latent=latent,
        )


@pytest.fixture(autouse=True)
def _reset_one_stage_fakes() -> None:
    _OneStagePipeline.calls.clear()
    _OneStagePipeline.fail = False
    _OneStagePipeline.grad_enabled_observed = True
    _OneStagePipeline.mutate_references = False
    _OneStageFoundation.mutate = False
    _OneStageFoundation.exact_alias_observed = False
    _OneStageFoundation.inference_resident_exact_alias = False
    _OneStageFoundation.inference_resident = False
    _OneStageFoundation.inference_resident_views = False
    _OneStageFoundation.mutate_resident_transiently = False
    _OneStageFoundation.mutate_source_before_teardown = False
    _OneStageBlock.mutate_transiently = False
    _OneStageStage.no_forward = False
    _OneStageStage.double_build = False
    _BlockStreamingWrapper.teardown_calls = 0
    _BlockStreamingWrapper.dispose_calls = 0
    _BlockStreamingWrapper.use_disk_source = False
    _BlockStreamingWrapper.use_lora = False
    _BlockStreamingWrapper.swap_provider = False
    _BlockStreamingWrapper.swap_source = False
    _BlockStreamingWrapper.missing_layout = False
    _BlockStreamingWrapper.wrong_raw_buffer = False
    _BlockStreamingWrapper.missing_nonblock = False
    _BlockStreamingWrapper.extra_nonblock = False
    _PinnedWeightSource.cleanup_calls = 0
    _DiskWeightSource.cleanup_calls = 0


def test_pinned_one_stage_media_runs_the_complete_pipeline_without_autograd(
    tmp_path: Path,
) -> None:
    engine, _ = _one_stage_engine(tmp_path)

    engine.generate(
        guides=(torch.ones(1, 128, 1, 12, 12, requires_grad=True),),
        prompt="prompt",
        negative_prompt="",
        seed=101,
        output_path=(tmp_path / "no-grad.mp4").resolve(),
    )

    assert not _OneStagePipeline.grad_enabled_observed


def _one_stage_engine(tmp_path: Path) -> tuple[PinnedLTXOneStageMediaEngine, Path]:
    checkpoint = (tmp_path / "teacher.safetensors").resolve()
    checkpoint.write_bytes(b"teacher")
    gemma = (tmp_path / "gemma").resolve()
    gemma.mkdir()
    expected = foundation_digest(_ExpectedX0Model())
    params = SimpleNamespace(video_guider_params=_Guider(), audio_guider_params=_Guider())
    modules = {
        "ltx_pipelines": SimpleNamespace(TI2VidOneStagePipeline=_OneStagePipeline),
        "ltx_pipelines.utils.model_paths": SimpleNamespace(ModelPaths=_ModelPaths),
        "ltx_core.conditioning": SimpleNamespace(
            VideoConditionByReferenceLatent=_Reference,
            ConditioningItemAttentionStrengthWrapper=_Attention,
        ),
        "ltx_core.components.guiders": SimpleNamespace(MultiModalGuiderParams=_Guider),
        "ltx_pipelines.utils.constants": SimpleNamespace(detect_params=lambda _path: params),
        "ltx_core.model.video_vae": SimpleNamespace(
            get_video_chunks_number=lambda frames, tiling: 1
        ),
        "ltx_pipelines.utils.media_io": SimpleNamespace(),
        "ltx_pipelines.utils.types": SimpleNamespace(OffloadMode=_OffloadMode),
        "ltx_core.block_streaming": SimpleNamespace(BlockStreamingWrapper=_BlockStreamingWrapper),
        "ltx_core.model.transformer": SimpleNamespace(X0Model=_X0Model),
        "ltx_core.block_streaming.provider": SimpleNamespace(WeightsProvider=_WeightsProvider),
        "ltx_core.block_streaming.source": SimpleNamespace(
            PinnedWeightSource=_PinnedWeightSource,
            DiskWeightSource=_DiskWeightSource,
        ),
        "ltx_core.block_streaming.utils": SimpleNamespace(
            carve_buffer=_carve_buffer,
            layout_nbytes=_layout_nbytes,
        ),
    }

    def encode_video(**kwargs: object) -> None:
        Path(str(kwargs["output_path"])).write_bytes(b"one-stage-mp4")

    modules["ltx_pipelines.utils.media_io"].encode_video = encode_video
    return (
        PinnedLTXOneStageMediaEngine(
            teacher_checkpoint_path=checkpoint,
            teacher_checkpoint_file_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            gemma_root=gemma,
            expected_teacher_foundation_sha256=expected,
            device=torch.device("cpu"),
            importer=lambda name: modules[name],
            media_probe=lambda _path: DecodedMediaFacts(384, 384, 25, 24.0, 25 / 24),
        ),
        checkpoint,
    )


def test_pinned_one_stage_media_locks_exact_runtime_and_receipts(tmp_path: Path) -> None:
    _OneStagePipeline.calls.clear()
    _OneStagePipeline.fail = False
    _OneStagePipeline.mutate_references = False
    _OneStageFoundation.mutate = False
    _OneStageFoundation.mutate_source_before_teardown = False
    _BlockStreamingWrapper.teardown_calls = 0
    engine, checkpoint = _one_stage_engine(tmp_path)
    guides = tuple(torch.full((1, 128, 1, 12, 12), float(index)) for index in range(4))
    output = (tmp_path / "one-stage.mp4").resolve()

    result = engine.generate(
        guides=guides,
        prompt="reveal the same hidden marking",
        negative_prompt="text, logos",
        seed=202,
        output_path=output,
    )

    assert len(_OneStagePipeline.calls) == 1
    call = _OneStagePipeline.calls[0]
    assert call == {
        "prompt": "reveal the same hidden marking",
        "negative_prompt": "text, logos",
        "seed": 202,
        "height": 384,
        "width": 384,
        "frame_rate": 24.0,
        "num_inference_steps": 30,
        "video_guider_params": call["video_guider_params"],
        "audio_guider_params": call["audio_guider_params"],
        "images": [],
        "num_frames": 25,
        "enhance_prompt": False,
        "enhance_static_cache": False,
        "vae_dtype": torch.bfloat16,
        "max_batch_size": 1,
        "generated_keyframes": 0,
    }
    assert len(_OneStagePipeline.references) == 4
    assert all(isinstance(item, _Attention) for item in _OneStagePipeline.references)
    reference_latents = [
        cast(torch.Tensor, cast(_Attention, item).reference.values["latent"])
        for item in _OneStagePipeline.references
    ]
    assert all(
        latent.dtype is torch.bfloat16 and tuple(latent.shape) == (1, 128, 1, 12, 12)
        for latent in reference_latents
    )
    assert result.final_latent.device.type == "cpu"
    assert result.final_latent.dtype is torch.float32
    assert not result.final_latent.requires_grad
    assert result.final_latent_sha256 == tensor_sha256(result.final_latent)
    assert result.checkpoint_file_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result.media_sha256 == hashlib.sha256(b"one-stage-mp4").hexdigest()
    assert result.foundation_before_sha256 == result.foundation_after_sha256
    assert result.foundation_before_sha256 == foundation_digest(_ExpectedX0Model())
    assert (result.width, result.height, result.frames, result.fps, result.steps) == (
        384,
        384,
        25,
        24,
        30,
    )
    assert result.dtype == "bfloat16"
    assert result.batch_size == 1
    assert result.offload_mode == media.PINNED_LTX_OFFLOAD_MODE == "cpu"
    assert result.foundation_guard_sha256 == media.STREAMING_FOUNDATION_GUARD_SHA256
    assert _OneStagePipeline.calls[0]["seed"] == 202
    assert cast(_OneStagePipeline, engine._pipeline).kwargs["offload_mode"] is _OffloadMode.CPU
    assert _BlockStreamingWrapper.teardown_calls == 1


def test_pinned_one_stage_media_normalizes_official_inference_resident(tmp_path: Path) -> None:
    _OneStageFoundation.inference_resident = True
    engine, _ = _one_stage_engine(tmp_path)

    result = engine.generate(
        guides=(torch.ones(1, 128, 1, 12, 12),),
        prompt="prompt",
        negative_prompt="",
        seed=101,
        output_path=(tmp_path / "inference-resident.mp4").resolve(),
    )

    assert result.validate() is result
    assert result.foundation_before_sha256 == result.foundation_after_sha256
    assert _BlockStreamingWrapper.teardown_calls == 1


def test_pinned_one_stage_media_detects_transient_inference_resident_mutation(
    tmp_path: Path,
) -> None:
    _OneStageFoundation.inference_resident = True
    _OneStageFoundation.mutate_resident_transiently = True
    engine, _ = _one_stage_engine(tmp_path)

    with pytest.raises(RuntimeError, match="foundation tensors mutated"):
        engine.generate(
            guides=(torch.ones(1, 128, 1, 12, 12),),
            prompt="prompt",
            negative_prompt="",
            seed=101,
            output_path=(tmp_path / "transient-inference-resident.mp4").resolve(),
        )

    assert _BlockStreamingWrapper.teardown_calls == 1
    assert _BlockStreamingWrapper.dispose_calls == 1


def test_pinned_one_stage_media_rejects_versionless_streaming_tensors(tmp_path: Path) -> None:
    engine, _ = _one_stage_engine(tmp_path)

    with (
        torch.inference_mode(),
        pytest.raises(RuntimeError, match="streaming foundation tensor lacks mutation versioning"),
    ):
        engine.generate(
            guides=(torch.ones(1, 128, 1, 12, 12),),
            prompt="prompt",
            negative_prompt="",
            seed=101,
            output_path=(tmp_path / "versionless-streaming.mp4").resolve(),
        )

    assert _BlockStreamingWrapper.teardown_calls == 1
    assert _BlockStreamingWrapper.dispose_calls == 1


def test_pinned_one_stage_media_rejects_distinct_resident_storage_aliases(
    tmp_path: Path,
) -> None:
    _OneStageFoundation.inference_resident_views = True
    engine, _ = _one_stage_engine(tmp_path)

    with pytest.raises(RuntimeError, match="resident tensor storage alias changed"):
        engine.generate(
            guides=(torch.ones(1, 128, 1, 12, 12),),
            prompt="prompt",
            negative_prompt="",
            seed=101,
            output_path=(tmp_path / "resident-storage-alias.mp4").resolve(),
        )

    assert _BlockStreamingWrapper.teardown_calls == 1
    assert _BlockStreamingWrapper.dispose_calls == 1


def test_pinned_one_stage_media_preserves_exact_resident_aliases(tmp_path: Path) -> None:
    _OneStageFoundation.inference_resident = True
    _OneStageFoundation.inference_resident_exact_alias = True
    engine, _ = _one_stage_engine(tmp_path)

    result = engine.generate(
        guides=(torch.ones(1, 128, 1, 12, 12),),
        prompt="prompt",
        negative_prompt="",
        seed=101,
        output_path=(tmp_path / "resident-exact-alias.mp4").resolve(),
    )

    assert result.validate() is result
    assert _OneStageFoundation.exact_alias_observed
    assert _BlockStreamingWrapper.teardown_calls == 1


def test_pinned_one_stage_media_detects_transient_streamed_block_mutation(
    tmp_path: Path,
) -> None:
    _OneStageBlock.mutate_transiently = True
    engine, _ = _one_stage_engine(tmp_path)

    with pytest.raises(RuntimeError, match="foundation tensors mutated"):
        engine.generate(
            guides=(torch.ones(1, 128, 1, 12, 12),),
            prompt="prompt",
            negative_prompt="",
            seed=101,
            output_path=(tmp_path / "transient-streamed-block.mp4").resolve(),
        )

    assert _BlockStreamingWrapper.teardown_calls == 1
    assert _BlockStreamingWrapper.dispose_calls == 1


def test_pinned_one_stage_media_rehashes_checkpoint_before_cached_call(tmp_path: Path) -> None:
    _OneStagePipeline.calls.clear()
    _OneStagePipeline.fail = False
    _OneStagePipeline.mutate_references = False
    _OneStageFoundation.mutate = False
    engine, checkpoint = _one_stage_engine(tmp_path)
    guide = (torch.ones(1, 128, 1, 12, 12),)

    first = engine.generate(
        guides=guide,
        prompt="prompt",
        negative_prompt="",
        seed=101,
        output_path=(tmp_path / "first.mp4").resolve(),
    )
    checkpoint.write_bytes(b"drifted-teacher")

    with pytest.raises(ValueError, match="checkpoint content SHA-256 mismatch"):
        engine.generate(
            guides=guide,
            prompt="prompt",
            negative_prompt="",
            seed=101,
            output_path=(tmp_path / "second.mp4").resolve(),
        )

    assert first.checkpoint_file_sha256 == hashlib.sha256(b"teacher").hexdigest()
    assert len(_OneStagePipeline.calls) == 1


@pytest.mark.parametrize(
    "guide",
    [
        torch.ones(1, 128, 1, 6, 6),
        torch.ones(1, 128, 2, 12, 12),
        torch.ones(2, 128, 1, 12, 12),
        torch.full((1, 128, 1, 12, 12), torch.nan),
        torch.ones(1, 128, 1, 12, 12, dtype=torch.int64),
    ],
)
def test_pinned_one_stage_media_rejects_non_exact_guides(
    tmp_path: Path, guide: torch.Tensor
) -> None:
    engine, _ = _one_stage_engine(tmp_path)

    with pytest.raises(ValueError, match=r"\[1,128,1,12,12\]"):
        engine.generate(
            guides=(guide,),
            prompt="prompt",
            negative_prompt="",
            seed=101,
            output_path=(tmp_path / "invalid.mp4").resolve(),
        )


@pytest.mark.parametrize("seed", [0, 100, 102, 202.0, True])
def test_pinned_one_stage_media_rejects_non_frozen_seed(tmp_path: Path, seed: object) -> None:
    engine, _ = _one_stage_engine(tmp_path)

    with pytest.raises(ValueError, match="three frozen decoded-quality seeds"):
        engine.generate(
            guides=(torch.ones(1, 128, 1, 12, 12),),
            prompt="prompt",
            negative_prompt="",
            seed=seed,  # type: ignore[arg-type]
            output_path=(tmp_path / "invalid.mp4").resolve(),
        )


def test_pinned_one_stage_media_restores_pipeline_after_failure(tmp_path: Path) -> None:
    _OneStagePipeline.fail = True
    _OneStageFoundation.mutate = False
    engine, _ = _one_stage_engine(tmp_path)
    engine._load_pipeline(engine._load_api())
    pipeline = engine._pipeline
    assert isinstance(pipeline, _OneStagePipeline)
    original_conditioner = pipeline.image_conditioner
    original_stage = pipeline.stage

    with pytest.raises(RuntimeError, match="injected pipeline failure"):
        engine.generate(
            guides=(torch.ones(1, 128, 1, 12, 12),),
            prompt="prompt",
            negative_prompt="",
            seed=303,
            output_path=(tmp_path / "failure.mp4").resolve(),
        )

    assert pipeline.image_conditioner is original_conditioner
    assert pipeline.stage is original_stage
    assert engine._foundation_guard is None
    _OneStagePipeline.fail = False


def test_pinned_one_stage_media_rejects_foundation_mutation_and_restores(
    tmp_path: Path,
) -> None:
    _OneStagePipeline.fail = False
    _OneStageFoundation.mutate = True
    engine, _ = _one_stage_engine(tmp_path)
    engine._load_pipeline(engine._load_api())
    pipeline = engine._pipeline
    assert isinstance(pipeline, _OneStagePipeline)
    original_conditioner = pipeline.image_conditioner
    original_stage = pipeline.stage

    with pytest.raises(RuntimeError, match="foundation tensors mutated"):
        engine.generate(
            guides=(torch.ones(1, 128, 1, 12, 12),),
            prompt="prompt",
            negative_prompt="",
            seed=101,
            output_path=(tmp_path / "mutated.mp4").resolve(),
        )

    assert pipeline.image_conditioner is original_conditioner
    assert pipeline.stage is original_stage
    assert engine._foundation_guard is None
    assert _BlockStreamingWrapper.teardown_calls == 1
    assert _BlockStreamingWrapper.dispose_calls == 1
    assert _PinnedWeightSource.cleanup_calls == 1
    _OneStageFoundation.mutate = False


def test_pinned_one_stage_media_terminal_hash_catches_unversioned_source_mutation_and_cleans_up(
    tmp_path: Path,
) -> None:
    _OneStagePipeline.fail = False
    _OneStageFoundation.mutate = False
    _OneStageFoundation.mutate_source_before_teardown = True
    _BlockStreamingWrapper.teardown_calls = 0
    engine, _ = _one_stage_engine(tmp_path)

    with pytest.raises(RuntimeError, match="foundation changed") as error:
        engine.generate(
            guides=(torch.ones(1, 128, 1, 12, 12),),
            prompt="prompt",
            negative_prompt="",
            seed=101,
            output_path=(tmp_path / "source-mutated.mp4").resolve(),
        )

    assert error.value.__cause__ is not error.value
    assert _BlockStreamingWrapper.teardown_calls == 1
    assert _BlockStreamingWrapper.dispose_calls == 1
    assert _PinnedWeightSource.cleanup_calls == 1
    assert engine._foundation_guard is None
    _OneStageFoundation.mutate_source_before_teardown = False


def test_pinned_one_stage_media_hashes_complete_streamed_foundation_only_at_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _ = _one_stage_engine(tmp_path)
    calls = 0
    delegate = _foundation_items_digest

    def counted(items: object) -> str:
        nonlocal calls
        calls += 1
        return delegate(cast(Any, items))

    monkeypatch.setattr(media, "_foundation_items_digest", counted)

    engine.generate(
        guides=(torch.ones(1, 128, 1, 12, 12),),
        prompt="prompt",
        negative_prompt="",
        seed=101,
        output_path=(tmp_path / "boundary-hashes.mp4").resolve(),
    )

    assert calls == 2


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("use_disk_source", "provider changed"),
        ("use_lora", "provider changed"),
        ("missing_layout", "pinned source changed"),
        ("wrong_raw_buffer", "pinned source changed"),
        ("missing_nonblock", "loaded-state lock"),
        ("extra_nonblock", "loaded-state lock"),
    ],
)
def test_pinned_one_stage_media_rejects_inauthentic_streaming_foundation_and_cleans_up(
    tmp_path: Path, flag: str, message: str
) -> None:
    setattr(_BlockStreamingWrapper, flag, True)
    engine, _ = _one_stage_engine(tmp_path)

    with pytest.raises((RuntimeError, ValueError), match=message):
        engine.generate(
            guides=(torch.ones(1, 128, 1, 12, 12),),
            prompt="prompt",
            negative_prompt="",
            seed=101,
            output_path=(tmp_path / f"{flag}.mp4").resolve(),
        )

    assert _BlockStreamingWrapper.teardown_calls == 1
    assert _BlockStreamingWrapper.dispose_calls == 1
    assert _PinnedWeightSource.cleanup_calls + _DiskWeightSource.cleanup_calls == 1


@pytest.mark.parametrize("flag", ["swap_provider", "swap_source"])
def test_pinned_one_stage_media_rejects_streaming_identity_swap_and_cleans_up(
    tmp_path: Path, flag: str
) -> None:
    setattr(_BlockStreamingWrapper, flag, True)
    engine, _ = _one_stage_engine(tmp_path)

    with pytest.raises(RuntimeError, match="foundation tensors mutated"):
        engine.generate(
            guides=(torch.ones(1, 128, 1, 12, 12),),
            prompt="prompt",
            negative_prompt="",
            seed=101,
            output_path=(tmp_path / f"{flag}.mp4").resolve(),
        )

    assert _BlockStreamingWrapper.teardown_calls == 1
    assert _BlockStreamingWrapper.dispose_calls == 1


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("no_forward", "not measured across a model forward"),
        ("double_build", "built exactly once"),
    ],
)
def test_pinned_one_stage_media_rejects_streaming_lifecycle_drift_and_cleans_up(
    tmp_path: Path, flag: str, message: str
) -> None:
    setattr(_OneStageStage, flag, True)
    engine, _ = _one_stage_engine(tmp_path)

    with pytest.raises(RuntimeError, match=message):
        engine.generate(
            guides=(torch.ones(1, 128, 1, 12, 12),),
            prompt="prompt",
            negative_prompt="",
            seed=101,
            output_path=(tmp_path / f"{flag}.mp4").resolve(),
        )

    assert _BlockStreamingWrapper.teardown_calls == 1
    assert _BlockStreamingWrapper.dispose_calls == 1


def test_one_stage_result_rejects_offload_or_streaming_guard_identity_drift(
    tmp_path: Path,
) -> None:
    _OneStageFoundation.mutate = False
    _OneStageFoundation.mutate_source_before_teardown = False
    engine, _ = _one_stage_engine(tmp_path)
    result = engine.generate(
        guides=(torch.ones(1, 128, 1, 12, 12),),
        prompt="prompt",
        negative_prompt="",
        seed=101,
        output_path=(tmp_path / "identity.mp4").resolve(),
    )

    with pytest.raises(ValueError, match="guard identity"):
        dataclasses.replace(result, offload_mode="none").validate()
    with pytest.raises(ValueError, match="guard identity"):
        dataclasses.replace(result, foundation_guard_sha256="0" * 64).validate()


def test_pinned_one_stage_media_isolates_caller_guides_from_pipeline_mutation(
    tmp_path: Path,
) -> None:
    _OneStagePipeline.fail = False
    _OneStagePipeline.mutate_references = True
    _OneStageFoundation.mutate = False
    engine, _ = _one_stage_engine(tmp_path)
    guide = torch.ones(1, 128, 1, 12, 12, requires_grad=True)
    before = guide.detach().clone()

    engine.generate(
        guides=(guide,),
        prompt="prompt",
        negative_prompt="",
        seed=202,
        output_path=(tmp_path / "mutated-reference.mp4").resolve(),
    )

    torch.testing.assert_close(guide, before)
    assert guide.requires_grad
    _OneStagePipeline.mutate_references = False


def test_pinned_one_stage_media_seed_is_invariant_to_method_order(tmp_path: Path) -> None:
    _OneStagePipeline.fail = False
    _OneStagePipeline.mutate_references = False
    _OneStageFoundation.mutate = False
    engine, _ = _one_stage_engine(tmp_path)
    methods = {
        "full": tuple(torch.full((1, 128, 1, 12, 12), float(index)) for index in range(9)),
        "recent": tuple(torch.full((1, 128, 1, 12, 12), float(index)) for index in range(4)),
        "gated": tuple(torch.full((1, 128, 1, 12, 12), float(index + 4)) for index in range(4)),
        "duet": tuple(torch.full((1, 128, 1, 12, 12), float(index + 8)) for index in range(4)),
    }

    def run(order: tuple[str, ...], prefix: str) -> dict[str, str]:
        return {
            method: engine.generate(
                guides=methods[method],
                prompt="prompt",
                negative_prompt="",
                seed=303,
                output_path=(tmp_path / f"{prefix}-{method}.mp4").resolve(),
            ).final_latent_sha256
            for method in order
        }

    forward = run(("full", "recent", "gated", "duet"), "forward")
    torch.manual_seed(999_001)
    torch.rand(257)
    _OneStagePipeline.references = [
        _Attention(
            _Reference(latent=torch.full((1, 128, 1, 12, 12), 991.0)),
            attention_mask=1.0,
        )
    ]
    reverse = run(("duet", "gated", "recent", "full"), "reverse")

    assert forward == reverse
    assert len(set(forward.values())) == 4
    reversed_full = engine.generate(
        guides=tuple(reversed(methods["full"])),
        prompt="prompt",
        negative_prompt="",
        seed=303,
        output_path=(tmp_path / "reversed-full.mp4").resolve(),
    )
    assert reversed_full.final_latent_sha256 != forward["full"]
