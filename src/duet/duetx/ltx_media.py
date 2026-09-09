"""Pinned LTX media generation with explicit latent history guides."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, cast

import torch

from .contracts import tensor_sha256
from .media_probe import DecodedMediaFacts, probe_decoded_video
from .training import _foundation_items_digest, foundation_digest

_PINNED_MEDIA_ERROR = "pinned LTX two-stage media API contract is unavailable or has drifted"
_PINNED_ONE_STAGE_MEDIA_ERROR = (
    "pinned LTX one-stage media API contract is unavailable or has drifted"
)
_WIDTH = 384
_HEIGHT = 384
_FRAMES = 25
_FPS = 24
_STEPS = 30
_ONE_STAGE_GUIDE_SHAPE = (1, 128, 1, 12, 12)
_ONE_STAGE_FINAL_LATENT_SHAPE = (1, 128, 4, 12, 12)
_FROZEN_QUALITY_SEEDS = frozenset((101, 202, 303))
_SHA256_HEX = frozenset("0123456789abcdef")
_MISSING = object()
PINNED_LTX_OFFLOAD_MODE = "cpu"
_STREAMING_FOUNDATION_GUARD_SPEC = {
    "format": "duet-x-ltx-streaming-foundation-guard-v1",
    "offload_mode": PINNED_LTX_OFFLOAD_MODE,
    "authenticity": "checkpoint-and-official-loader",
    "logical_content": "complete-digest-at-install-and-before-official-teardown",
    "resident_versioning": "content-identical-normal-clones-before-lock",
    "resident_aliasing": "exact-object-only-shared-storage-rejected",
    "per_forward": "exact-identity-version-roster-layout-device-and-block-order",
}
STREAMING_FOUNDATION_GUARD_SHA256 = hashlib.sha256(
    json.dumps(
        _STREAMING_FOUNDATION_GUARD_SPEC,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("model artifact must be a regular non-symlink file")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("model artifact must be a regular file")
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MediaGenerationResult:
    """One decoded, content-addressable media result plus loaded-model integrity."""

    path: Path
    final_latent: torch.Tensor
    teacher_foundation_before_sha256: str
    teacher_foundation_after_sha256: str
    stage_2_loaded_before_sha256: str
    stage_2_loaded_after_sha256: str
    width: int = _WIDTH
    height: int = _HEIGHT
    frames: int = _FRAMES
    fps: int = _FPS
    steps: int = _STEPS
    duration_seconds: float = _FRAMES / _FPS

    def validate(self) -> Self:
        if self.path.is_symlink() or not self.path.is_file() or self.path.stat().st_size <= 0:
            raise ValueError("generated media must be a nonempty regular non-symlink file")
        if (
            not isinstance(self.final_latent, torch.Tensor)
            or self.final_latent.layout != torch.strided
            or self.final_latent.numel() == 0
            or not torch.is_floating_point(self.final_latent)
            or not bool(torch.isfinite(self.final_latent).all().item())
        ):
            raise ValueError("generated final latent must be a finite floating tensor")
        for before, after, name in (
            (
                self.teacher_foundation_before_sha256,
                self.teacher_foundation_after_sha256,
                "teacher foundation",
            ),
            (
                self.stage_2_loaded_before_sha256,
                self.stage_2_loaded_after_sha256,
                "stage-2 loaded model",
            ),
        ):
            if len(before) != 64 or any(
                character not in "0123456789abcdef" for character in before
            ):
                raise ValueError(f"{name} digest must be lowercase SHA-256")
            if before != after:
                raise ValueError(f"{name} changed during media generation")
        if (self.width, self.height, self.frames, self.fps, self.steps) != (
            _WIDTH,
            _HEIGHT,
            _FRAMES,
            _FPS,
            _STEPS,
        ):
            raise ValueError("media settings do not match the Decision 1 canary lock")
        if not math.isclose(
            self.duration_seconds,
            _FRAMES / _FPS,
            rel_tol=0.0,
            abs_tol=1.0 / _FPS,
        ):
            raise ValueError("decoded media duration does not match the Decision 1 canary lock")
        return self


@dataclass(frozen=True, slots=True)
class OneStageMediaGenerationResult:
    """Decoded one-stage media plus independently auditable runtime receipts."""

    path: Path
    final_latent: torch.Tensor
    final_latent_sha256: str
    checkpoint_file_sha256: str
    foundation_before_sha256: str
    foundation_after_sha256: str
    media_sha256: str
    offload_mode: str
    foundation_guard_sha256: str
    width: int = _WIDTH
    height: int = _HEIGHT
    frames: int = _FRAMES
    fps: int = _FPS
    steps: int = _STEPS
    duration_seconds: float = _FRAMES / _FPS
    dtype: str = "bfloat16"
    batch_size: int = 1

    def validate(self) -> Self:
        if self.path.is_symlink() or not self.path.is_file() or self.path.stat().st_size <= 0:
            raise ValueError("generated media must be a nonempty regular non-symlink file")
        if (
            not isinstance(self.final_latent, torch.Tensor)
            or self.final_latent.layout != torch.strided
            or tuple(self.final_latent.shape) != _ONE_STAGE_FINAL_LATENT_SHAPE
            or self.final_latent.device.type != "cpu"
            or self.final_latent.dtype is not torch.float32
            or self.final_latent.requires_grad
            or not bool(torch.isfinite(self.final_latent).all().item())
        ):
            raise ValueError("one-stage final latent must be finite CPU float32 [1,128,4,12,12]")
        for digest, name in (
            (self.final_latent_sha256, "final latent"),
            (self.checkpoint_file_sha256, "checkpoint file"),
            (self.foundation_before_sha256, "foundation before"),
            (self.foundation_after_sha256, "foundation after"),
            (self.media_sha256, "media"),
        ):
            if len(digest) != 64 or any(character not in _SHA256_HEX for character in digest):
                raise ValueError(f"{name} digest must be lowercase SHA-256")
        if tensor_sha256(self.final_latent) != self.final_latent_sha256:
            raise ValueError("one-stage final latent bytes changed after hashing")
        if _file_sha256(self.path) != self.media_sha256:
            raise ValueError("generated media bytes changed after hashing")
        if self.foundation_before_sha256 != self.foundation_after_sha256:
            raise ValueError("one-stage foundation changed during media generation")
        if (self.width, self.height, self.frames, self.fps, self.steps) != (
            _WIDTH,
            _HEIGHT,
            _FRAMES,
            _FPS,
            _STEPS,
        ):
            raise ValueError("media settings do not match the frozen decoded-quality lock")
        if not math.isclose(
            self.duration_seconds,
            _FRAMES / _FPS,
            rel_tol=0.0,
            abs_tol=1.0 / _FPS,
        ):
            raise ValueError("decoded media duration does not match the decoded-quality lock")
        if self.dtype != "bfloat16" or self.batch_size != 1:
            raise ValueError("one-stage generation must use BF16 and batch size one")
        if (
            self.offload_mode != PINNED_LTX_OFFLOAD_MODE
            or self.foundation_guard_sha256 != STREAMING_FOUNDATION_GUARD_SHA256
        ):
            raise ValueError("one-stage streaming foundation guard identity changed")
        return self


class Decision1MediaEngine(Protocol):
    """Minimal fakeable boundary used by the sealed evaluation operation."""

    def generate(
        self,
        *,
        guides: tuple[torch.Tensor, ...],
        prompt: str,
        negative_prompt: str,
        seed: int,
        output_path: Path,
    ) -> MediaGenerationResult: ...


@dataclass(frozen=True, slots=True)
class _PinnedMediaAPI:
    pipeline_type: type[Any]
    model_paths_type: type[Any]
    lora_type: type[Any]
    lora_mapping: object
    reference_type: type[Any]
    attention_wrapper_type: type[Any]
    guider_params_type: type[Any]
    detect_params: Callable[[str], object]
    get_video_chunks_number: Callable[[int, object], int]
    encode_video: Callable[..., object]


@dataclass(frozen=True, slots=True)
class _PinnedOneStageMediaAPI:
    pipeline_type: type[Any]
    model_paths_type: type[Any]
    reference_type: type[Any]
    attention_wrapper_type: type[Any]
    guider_params_type: type[Any]
    detect_params: Callable[[str], object]
    get_video_chunks_number: Callable[[int, object], int]
    encode_video: Callable[..., object]
    offload_cpu: object
    x0_model_type: type[Any]
    streaming_wrapper_type: type[Any]
    weights_provider_type: type[Any]
    pinned_source_type: type[Any]
    disk_source_type: type[Any]
    carve_buffer: Callable[[torch.Tensor, object], dict[str, torch.Tensor]]
    layout_nbytes: Callable[[object], int]


class _ConditionerProxy:
    """Delegate the official image conditioner and append guides to its first call only."""

    def __init__(
        self, delegate: object, references: list[object], *, expected_calls: int = 2
    ) -> None:
        self._delegate = delegate
        self._references = references
        self._expected_calls = expected_calls
        self._calls = 0

    def resolve_crf(self, images: object) -> object:
        resolver = getattr(self._delegate, "resolve_crf", None)
        if not callable(resolver):
            raise RuntimeError(_PINNED_MEDIA_ERROR)
        return resolver(images)

    def __call__(self, encoder: object) -> list[object]:
        if not callable(self._delegate):
            raise RuntimeError(_PINNED_MEDIA_ERROR)
        result = self._delegate(encoder)
        if not isinstance(result, list):
            raise RuntimeError(_PINNED_MEDIA_ERROR)
        self._calls += 1
        if self._calls == 1:
            result.extend(self._references)
        elif self._calls > self._expected_calls:
            raise RuntimeError("pipeline invoked image conditioning an unexpected number of times")
        return cast(list[object], result)

    def validate(self) -> None:
        if self._calls != self._expected_calls:
            raise RuntimeError("pipeline invoked image conditioning an unexpected number of times")


@dataclass(frozen=True, slots=True)
class _FoundationTensorState:
    name: str
    identity: int
    version: int
    data_ptr: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    layout: str
    requires_grad: bool
    grad_is_none: bool


def _tensor_state(name: str, tensor: torch.Tensor) -> _FoundationTensorState:
    if tensor.is_inference():
        raise RuntimeError("streaming foundation tensor lacks mutation versioning")
    return _FoundationTensorState(
        name,
        id(tensor),
        int(tensor._version),
        tensor.data_ptr(),
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        str(tensor.layout),
        tensor.requires_grad,
        tensor.grad is None,
    )


def _state_items(items: list[tuple[str, torch.Tensor]]) -> tuple[_FoundationTensorState, ...]:
    return tuple(_tensor_state(name, tensor) for name, tensor in sorted(items))


def _registered_tensor(
    root: torch.nn.Module, name: str
) -> tuple[torch.nn.Module, str, torch.Tensor, bool]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        child = getattr(parent, part, None)
        if not isinstance(child, torch.nn.Module):
            raise RuntimeError("streaming foundation resident tensor path changed")
        parent = child
    leaf = parts[-1]
    tensor: torch.Tensor | None
    if leaf in parent._parameters:
        tensor = parent._parameters[leaf]
        is_parameter = True
    elif leaf in parent._buffers:
        tensor = parent._buffers[leaf]
        is_parameter = False
    else:
        raise RuntimeError("streaming foundation resident tensor path changed")
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError("streaming foundation resident tensor path changed")
    return parent, leaf, tensor, is_parameter


def _normalize_inference_resident_tensors(raw_model: torch.nn.Module) -> None:
    state = raw_model.state_dict(keep_vars=True)
    aliases: dict[int, list[str]] = {}
    storage_owners: dict[tuple[str, int, int], int] = {}
    for name, tensor in state.items():
        if not name.startswith("transformer_blocks."):
            aliases.setdefault(id(tensor), []).append(name)
            storage = tensor.untyped_storage()
            storage_nbytes = storage.nbytes()
            if storage_nbytes > 0:
                storage_key = (str(tensor.device), storage.data_ptr(), storage_nbytes)
                owner = storage_owners.setdefault(storage_key, id(tensor))
                if owner != id(tensor):
                    raise RuntimeError("streaming foundation resident tensor storage alias changed")
    del state
    for names in aliases.values():
        name = names[0]
        parent, leaf, tensor, is_parameter = _registered_tensor(raw_model, name)
        if not tensor.is_inference():
            continue
        with torch.inference_mode(False), torch.no_grad():
            clone = tensor.detach().clone(memory_format=torch.preserve_format)
            replacement = torch.nn.Parameter(clone, requires_grad=False) if is_parameter else clone
        if replacement.is_inference() or not torch.equal(replacement, tensor):
            raise RuntimeError("streaming foundation resident normalization changed content")
        for alias in names:
            parent, leaf, observed, observed_is_parameter = _registered_tensor(raw_model, alias)
            if observed is not tensor or observed_is_parameter is not is_parameter:
                raise RuntimeError("streaming foundation resident tensor alias changed")
            if is_parameter:
                parent._parameters[leaf] = cast(torch.nn.Parameter, replacement)
            else:
                parent._buffers[leaf] = replacement


def _matches_target_device(actual: torch.device, expected: torch.device) -> bool:
    if expected.type != "cuda":
        return actual == expected
    if actual.type != "cuda":
        return False
    expected_index = cast(object, expected.index)
    actual_index = cast(object, actual.index)
    if expected_index is not None:
        return actual_index == expected_index
    return actual_index in (None, 0)


@dataclass(frozen=True, slots=True)
class _StreamingStructureState:
    wrapper_identity: int
    raw_model_identity: int
    provider_identity: int
    source_identity: int
    pool_identity: int
    sync_identity: int
    block_identities: tuple[int, ...]
    layouts: tuple[tuple[tuple[str, tuple[int, ...], str], ...], ...]
    source_buffers: tuple[_FoundationTensorState, ...]
    resident_tensors: tuple[_FoundationTensorState, ...]


class _FoundationMutationGuard:
    """Authenticate the logical foundation while allowing official block-buffer rebinding."""

    def __init__(
        self,
        expected_sha256: str,
        api: _PinnedOneStageMediaAPI,
        device: torch.device,
    ) -> None:
        self._expected_sha256 = expected_sha256
        self._api = api
        self._device = device
        self._structure: _StreamingStructureState | None = None
        self._block_modules: tuple[torch.nn.Module, ...] = ()
        self._block_layouts: tuple[tuple[tuple[str, tuple[int, ...], str], ...], ...] = ()
        self._active_block_states: dict[int, tuple[_FoundationTensorState, ...]] = {}
        self._active_sequence: list[int] | None = None
        self._handles: list[object] = []
        self._model: torch.nn.Module | None = None
        self._streaming: Any | None = None
        self._original_teardown: Callable[[], object] | None = None
        self._instance_teardown: object = _MISSING
        self._guard_error: BaseException | None = None
        self._terminal_verified = False
        self._builds = 0
        self._forwards = 0
        self.before_sha256: str | None = None
        self.after_sha256: str | None = None

    def install(self, model: object, _video_tools: object) -> object:
        if type(model) is not self._api.x0_model_type or not isinstance(model, torch.nn.Module):
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR)
        self._builds += 1
        if self._builds != 1:
            raise RuntimeError("one-stage foundation must be built exactly once")
        self._model = model
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        streaming = getattr(model, "velocity_model", None)
        if type(streaming) is not self._api.streaming_wrapper_type or not isinstance(
            streaming, torch.nn.Module
        ):
            self._release_model()
            raise RuntimeError("one-stage foundation did not use pinned block streaming")
        self._streaming = streaming
        raw_model = getattr(streaming, "_model", None)
        if not isinstance(raw_model, torch.nn.Module):
            self._release_model()
            raise RuntimeError("one-stage foundation did not use pinned block streaming")
        _normalize_inference_resident_tensors(raw_model)
        logical_items, structure, blocks, layouts = self._logical_foundation(streaming)
        self._structure = structure
        self._block_modules = blocks
        self._block_layouts = layouts
        self.before_sha256 = _foundation_items_digest(logical_items)
        if self.before_sha256 != self._expected_sha256:
            self._release_model()
            raise ValueError("media teacher foundation does not match the loaded-state lock")

        teardown = getattr(streaming, "teardown", None)
        if not callable(teardown):
            self._release_model()
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR)
        self._original_teardown = teardown
        self._instance_teardown = vars(streaming).get("teardown", _MISSING)

        def guarded_teardown() -> object:
            try:
                self._finish_before_teardown()
            except BaseException as error:
                if self._guard_error is None:
                    self._guard_error = error
            finally:
                self._restore_teardown()
            try:
                return teardown()
            finally:
                self._release_model()

        cast(Any, streaming).teardown = guarded_teardown

        self._handles.extend(
            (
                model.register_forward_pre_hook(
                    lambda module, inputs: self._guard_hook(self._before_model, module, inputs)
                ),
                model.register_forward_hook(
                    lambda module, inputs, output: self._guard_hook(
                        self._after_model, module, inputs, output
                    )
                ),
            )
        )
        for index, block in enumerate(self._block_modules):
            self._handles.extend(
                (
                    block.register_forward_pre_hook(
                        lambda module, _inputs, index=index: self._guard_hook(
                            self._before_block, index, module
                        )
                    ),
                    block.register_forward_hook(
                        lambda module, _inputs, _output, index=index: self._guard_hook(
                            self._after_block, index, module
                        )
                    ),
                )
            )
        return model

    def _guard_hook(self, callback: Callable[..., None], *arguments: object) -> None:
        try:
            callback(*arguments)
        except BaseException as error:
            if self._guard_error is None:
                self._guard_error = error
            raise

    def _logical_foundation(
        self, streaming: torch.nn.Module
    ) -> tuple[
        list[tuple[str, torch.Tensor]],
        _StreamingStructureState,
        tuple[torch.nn.Module, ...],
        tuple[tuple[tuple[str, tuple[int, ...], str], ...], ...],
    ]:
        api = self._api
        provider = getattr(streaming, "_provider", None)
        raw_model = getattr(streaming, "_model", None)
        source = getattr(provider, "_source", None)
        provider_device = getattr(provider, "_target_device", None)
        streaming_device = getattr(streaming, "_target_device", None)
        if (
            type(provider) is not api.weights_provider_type
            or type(source) is not api.pinned_source_type
            or type(source) is api.disk_source_type
            or not isinstance(raw_model, torch.nn.Module)
            or getattr(provider, "_lora_sources", None) != []
            or not isinstance(provider_device, torch.device)
            or not isinstance(streaming_device, torch.device)
            or not _matches_target_device(provider_device, self._device)
            or not _matches_target_device(streaming_device, self._device)
        ):
            raise RuntimeError("pinned CPU streaming foundation provider changed")
        blocks = getattr(raw_model, "transformer_blocks", None)
        raw_blocks = getattr(source, "_blocks", None)
        if (
            not isinstance(blocks, torch.nn.ModuleList)
            or type(raw_blocks) is not dict
            or set(raw_blocks) != set(range(len(blocks)))
            or getattr(streaming, "num_blocks", None) != len(blocks)
            or len(blocks) <= 0
        ):
            raise RuntimeError("streaming foundation block roster changed")

        model_state = raw_model.state_dict(keep_vars=True)
        logical: list[tuple[str, torch.Tensor]] = []
        resident: list[tuple[str, torch.Tensor]] = []
        expected_keys = {f"velocity_model.{name}" for name in model_state}
        for name, tensor in model_state.items():
            if not name.startswith("transformer_blocks."):
                logical.append((f"velocity_model.{name}", tensor))
                resident.append((name, tensor))

        layouts: list[tuple[tuple[str, tuple[int, ...], str], ...]] = []
        source_buffers: list[tuple[str, torch.Tensor]] = []
        for index in range(len(blocks)):
            pinned = raw_blocks[index]
            raw = getattr(pinned, "buffer", None)
            layout = getattr(pinned, "layout", None)
            if (
                not isinstance(raw, torch.Tensor)
                or raw.device != torch.device("cpu")
                or raw.dtype is not torch.uint8
                or raw.ndim != 1
                or not raw.is_contiguous()
                or raw.requires_grad
                or type(layout) is not dict
                or raw.numel() != api.layout_nbytes(layout)
            ):
                raise RuntimeError("streaming foundation pinned source changed")
            views = api.carve_buffer(raw, layout)
            if set(views) != set(layout):
                raise RuntimeError("streaming foundation block layout changed")
            layout_rows: list[tuple[str, tuple[int, ...], str]] = []
            for name, view in views.items():
                specification = layout.get(name)
                if (
                    type(name) is not str
                    or not name
                    or not isinstance(specification, tuple)
                    or len(specification) != 2
                    or not isinstance(view, torch.Tensor)
                    or tuple(view.shape) != tuple(specification[0])
                    or view.dtype is not specification[1]
                ):
                    raise RuntimeError("streaming foundation block layout changed")
                logical.append((f"velocity_model.transformer_blocks.{index}.{name}", view))
                layout_rows.append((name, tuple(view.shape), str(view.dtype)))
            layouts.append(tuple(sorted(layout_rows)))
            source_buffers.append((str(index), raw))
        if {name for name, _tensor in logical} != expected_keys or len(logical) != len(
            expected_keys
        ):
            raise RuntimeError("streaming foundation logical tensor roster changed")
        slot_nbytes = getattr(source, "slot_nbytes", None)
        if type(slot_nbytes) is not int or slot_nbytes != max(
            api.layout_nbytes(raw_blocks[index].layout) for index in range(len(blocks))
        ):
            raise RuntimeError("streaming foundation pinned slot size changed")
        structure = _StreamingStructureState(
            id(streaming),
            id(raw_model),
            id(provider),
            id(source),
            id(getattr(provider, "_pool", None)),
            id(getattr(provider, "_sync", None)),
            tuple(id(block) for block in blocks),
            tuple(layouts),
            _state_items(source_buffers),
            _state_items(resident),
        )
        return logical, structure, cast(tuple[torch.nn.Module, ...], tuple(blocks)), tuple(layouts)

    def _assert_structure_unchanged(self) -> None:
        streaming = self._streaming
        if not isinstance(streaming, torch.nn.Module) or self._structure is None:
            raise RuntimeError("streaming foundation was unavailable")
        _logical, observed, blocks, layouts = self._logical_foundation(streaming)
        if (
            observed != self._structure
            or blocks != self._block_modules
            or layouts != self._block_layouts
        ):
            raise RuntimeError("frozen LTX foundation tensors mutated during media generation")

    def _before_model(self, _model: torch.nn.Module, _inputs: object) -> None:
        if self._active_sequence is not None:
            raise RuntimeError("streaming foundation forward lifecycle changed")
        self._assert_structure_unchanged()
        self._active_sequence = []

    def _before_block(self, index: int, block: torch.nn.Module) -> None:
        sequence = self._active_sequence
        if sequence is None or index != len(sequence) or index in self._active_block_states:
            raise RuntimeError("streaming foundation block order changed")
        state = _state_items(list(block.state_dict(keep_vars=True).items()))
        if not state:
            raise RuntimeError("streaming foundation block tensor roster changed")
        observed_layout = tuple(sorted((item.name, item.shape, item.dtype) for item in state))
        if observed_layout != self._block_layouts[index] or any(
            item.device == "meta" for item in state
        ):
            raise RuntimeError("streaming foundation block tensor roster changed")
        self._active_block_states[index] = state
        sequence.append(index)

    def _after_block(self, index: int, block: torch.nn.Module) -> None:
        expected = self._active_block_states.pop(index, None)
        if (
            expected is None
            or _state_items(list(block.state_dict(keep_vars=True).items())) != expected
        ):
            raise RuntimeError("frozen LTX foundation tensors mutated during media generation")

    def _after_model(self, model: torch.nn.Module, _inputs: object, _output: object) -> None:
        if (
            self._active_sequence != list(range(len(self._block_modules)))
            or self._active_block_states
        ):
            raise RuntimeError("streaming foundation block roster changed")
        self._assert_structure_unchanged()
        self._active_sequence = None
        self._forwards += 1

    def _finish_before_teardown(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        if self._active_sequence is not None or self._active_block_states:
            raise RuntimeError("streaming foundation forward lifecycle changed")
        self._assert_structure_unchanged()
        streaming = self._streaming
        if not isinstance(streaming, torch.nn.Module):
            raise RuntimeError("streaming foundation was unavailable")
        logical, _state, _blocks, _layouts = self._logical_foundation(streaming)
        self.after_sha256 = _foundation_items_digest(logical)
        if self.after_sha256 != self.before_sha256 or self.after_sha256 != self._expected_sha256:
            raise RuntimeError("frozen LTX foundation changed during media generation")
        self._terminal_verified = True

    def raise_if_failed(self) -> None:
        if self._guard_error is not None:
            raise self._guard_error

    def finalize(self) -> tuple[str, str]:
        self.raise_if_failed()
        if (
            self._builds != 1
            or self._forwards <= 0
            or not self._terminal_verified
            or self.before_sha256 is None
            or self.after_sha256 is None
        ):
            raise RuntimeError("one-stage foundation was not measured across a model forward")
        return self.before_sha256, self.after_sha256

    def _restore_teardown(self) -> None:
        streaming = self._streaming
        if streaming is None or self._original_teardown is None:
            return
        if self._instance_teardown is _MISSING:
            with suppress(AttributeError):
                delattr(streaming, "teardown")
        else:
            cast(Any, streaming).teardown = self._instance_teardown
        self._original_teardown = None
        self._instance_teardown = _MISSING

    def _release_model(self) -> None:
        for handle in self._handles:
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
        self._handles.clear()
        self._model = None
        self._streaming = None
        self._block_modules = ()
        self._active_block_states.clear()
        self._active_sequence = None

    def close(self) -> None:
        self._restore_teardown()
        self._release_model()


class PinnedLTXOneStageMediaEngine:
    """Run the pinned ``TI2VidOneStagePipeline`` at the frozen quality coordinate."""

    def __init__(
        self,
        *,
        teacher_checkpoint_path: Path,
        teacher_checkpoint_file_sha256: str,
        gemma_root: Path,
        expected_teacher_foundation_sha256: str,
        device: torch.device,
        importer: Callable[[str], object] = importlib.import_module,
        media_probe: Callable[[Path], DecodedMediaFacts] = probe_decoded_video,
    ) -> None:
        self._teacher_checkpoint_path = teacher_checkpoint_path
        self._teacher_checkpoint_file_sha256 = teacher_checkpoint_file_sha256
        self._gemma_root = gemma_root
        self._expected_teacher_foundation_sha256 = expected_teacher_foundation_sha256
        self._device = device
        self._importer = importer
        self._media_probe = media_probe
        self._api: _PinnedOneStageMediaAPI | None = None
        self._pipeline: object | None = None
        self._image_conditioner: object | None = None
        self._stage: object | None = None
        self._foundation_guard: _FoundationMutationGuard | None = None

    def _load_api(self) -> _PinnedOneStageMediaAPI:
        if self._api is not None:
            return self._api
        try:
            pipelines = self._importer("ltx_pipelines")
            model_paths = self._importer("ltx_pipelines.utils.model_paths")
            conditioning = self._importer("ltx_core.conditioning")
            guiders = self._importer("ltx_core.components.guiders")
            constants = self._importer("ltx_pipelines.utils.constants")
            video_vae = self._importer("ltx_core.model.video_vae")
            media_io = self._importer("ltx_pipelines.utils.media_io")
            runtime_types = self._importer("ltx_pipelines.utils.types")
            transformer = self._importer("ltx_core.model.transformer")
            streaming = self._importer("ltx_core.block_streaming")
            provider = self._importer("ltx_core.block_streaming.provider")
            source = self._importer("ltx_core.block_streaming.source")
            streaming_utils = self._importer("ltx_core.block_streaming.utils")
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR) from error
        values = (
            getattr(pipelines, "TI2VidOneStagePipeline", None),
            getattr(model_paths, "ModelPaths", None),
            getattr(conditioning, "VideoConditionByReferenceLatent", None),
            getattr(conditioning, "ConditioningItemAttentionStrengthWrapper", None),
            getattr(guiders, "MultiModalGuiderParams", None),
            getattr(transformer, "X0Model", None),
            getattr(streaming, "BlockStreamingWrapper", None),
            getattr(provider, "WeightsProvider", None),
            getattr(source, "PinnedWeightSource", None),
            getattr(source, "DiskWeightSource", None),
        )
        callables = (
            getattr(constants, "detect_params", None),
            getattr(video_vae, "get_video_chunks_number", None),
            getattr(media_io, "encode_video", None),
            getattr(streaming_utils, "carve_buffer", None),
            getattr(streaming_utils, "layout_nbytes", None),
        )
        if any(not isinstance(value, type) for value in values) or any(
            not callable(value) for value in callables
        ):
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR)
        offload_type = getattr(runtime_types, "OffloadMode", None)
        offload_cpu = getattr(offload_type, "CPU", None)
        if (
            not isinstance(offload_type, type)
            or getattr(offload_cpu, "value", None) != PINNED_LTX_OFFLOAD_MODE
        ):
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR)
        typed = cast(
            tuple[
                type[Any],
                type[Any],
                type[Any],
                type[Any],
                type[Any],
                type[Any],
                type[Any],
                type[Any],
                type[Any],
                type[Any],
            ],
            values,
        )
        functions = cast(
            tuple[
                Callable[[str], object],
                Callable[[int, object], int],
                Callable[..., object],
                Callable[[torch.Tensor, object], dict[str, torch.Tensor]],
                Callable[[object], int],
            ],
            callables,
        )
        self._api = _PinnedOneStageMediaAPI(
            pipeline_type=typed[0],
            model_paths_type=typed[1],
            reference_type=typed[2],
            attention_wrapper_type=typed[3],
            guider_params_type=typed[4],
            detect_params=functions[0],
            get_video_chunks_number=functions[1],
            encode_video=functions[2],
            offload_cpu=offload_cpu,
            x0_model_type=typed[5],
            streaming_wrapper_type=typed[6],
            weights_provider_type=typed[7],
            pinned_source_type=typed[8],
            disk_source_type=typed[9],
            carve_buffer=functions[3],
            layout_nbytes=functions[4],
        )
        return self._api

    def _load_pipeline(self, api: _PinnedOneStageMediaAPI) -> object:
        if self._pipeline is not None:
            return self._pipeline
        path = self._teacher_checkpoint_path
        if (
            path.is_symlink()
            or not path.is_file()
            or _file_sha256(path) != self._teacher_checkpoint_file_sha256
        ):
            raise ValueError("teacher checkpoint content SHA-256 mismatch")
        if self._gemma_root.is_symlink() or not self._gemma_root.is_dir():
            raise ValueError("Gemma root must be an existing non-symlink directory")
        model_paths_factory = getattr(api.model_paths_type, "from_monolith", None)
        if not callable(model_paths_factory):
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR)
        model_paths = model_paths_factory(str(path), str(self._gemma_root))
        self._pipeline = api.pipeline_type(
            model_paths=model_paths,
            loras=[],
            device=self._device,
            offload_mode=api.offload_cpu,
            prompt_enhancer_gemma_root=str(self._gemma_root),
        )
        if (
            not callable(self._pipeline)
            or getattr(self._pipeline, "dtype", None) is not torch.bfloat16
        ):
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR)
        self._image_conditioner = getattr(self._pipeline, "image_conditioner", None)
        self._stage = getattr(self._pipeline, "stage", None)
        if self._image_conditioner is None or self._stage is None:
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR)
        return self._pipeline

    def _references(
        self, api: _PinnedOneStageMediaAPI, guides: tuple[torch.Tensor, ...]
    ) -> list[object]:
        if not guides:
            raise ValueError("media generation requires at least one latent guide")
        references: list[object] = []
        for guide in guides:
            if (
                not isinstance(guide, torch.Tensor)
                or guide.layout != torch.strided
                or tuple(guide.shape) != _ONE_STAGE_GUIDE_SHAPE
                or not torch.is_floating_point(guide)
                or not bool(torch.isfinite(guide).all().item())
            ):
                raise ValueError("one-stage media guides must be finite [1,128,1,12,12] latents")
            frozen = (
                guide.detach().to(device=self._device, dtype=torch.bfloat16, copy=True).contiguous()
            )
            reference = api.reference_type(
                latent=frozen,
                downscale_factor=1,
                temporal_scale_factor=1,
                strength=1.0,
            )
            references.append(api.attention_wrapper_type(reference, attention_mask=1.0))
        return references

    def _wrapped_stage(self, stage: object, guard: _FoundationMutationGuard) -> object:
        wrapper = getattr(stage, "with_model_wrapper", None)
        if not callable(wrapper):
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR)
        result = wrapper(guard.install)
        if not callable(result):
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR)
        return result

    @torch.no_grad()
    def generate(
        self,
        *,
        guides: tuple[torch.Tensor, ...],
        prompt: str,
        negative_prompt: str,
        seed: int,
        output_path: Path,
    ) -> OneStageMediaGenerationResult:
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(negative_prompt, str)
        ):
            raise ValueError("media prompts must be explicit strings")
        if type(seed) is not int or seed not in _FROZEN_QUALITY_SEEDS:
            raise ValueError("media seed must be one of the three frozen decoded-quality seeds")
        if (
            output_path.is_symlink()
            or not output_path.is_absolute()
            or not output_path.parent.is_dir()
        ):
            raise ValueError(
                "media output must be an absolute non-symlink path below an existing directory"
            )
        observed_checkpoint_sha256 = _file_sha256(self._teacher_checkpoint_path)
        if observed_checkpoint_sha256 != self._teacher_checkpoint_file_sha256:
            raise ValueError("teacher checkpoint content SHA-256 mismatch")
        api = self._load_api()
        pipeline = self._load_pipeline(api)
        image_conditioner = self._image_conditioner
        stage = self._stage
        if image_conditioner is None or stage is None:
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR)
        params = api.detect_params(str(self._teacher_checkpoint_path))
        video_guider = getattr(params, "video_guider_params", None)
        audio_guider = getattr(params, "audio_guider_params", None)
        if not isinstance(video_guider, api.guider_params_type) or not isinstance(
            audio_guider, api.guider_params_type
        ):
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR)
        conditioner_proxy = _ConditionerProxy(
            image_conditioner,
            self._references(api, guides),
            expected_calls=1,
        )
        guard = _FoundationMutationGuard(
            self._expected_teacher_foundation_sha256, api, self._device
        )
        wrapped_stage = self._wrapped_stage(stage, guard)
        mutable_pipeline = cast(Any, pipeline)
        mutable_pipeline.image_conditioner = conditioner_proxy
        mutable_pipeline.stage = wrapped_stage
        self._foundation_guard = guard
        try:
            result = cast(
                Any,
                mutable_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=_HEIGHT,
                    width=_WIDTH,
                    frame_rate=float(_FPS),
                    num_inference_steps=_STEPS,
                    video_guider_params=video_guider,
                    audio_guider_params=audio_guider,
                    images=[],
                    num_frames=_FRAMES,
                    enhance_prompt=False,
                    enhance_static_cache=False,
                    vae_dtype=torch.bfloat16,
                    max_batch_size=1,
                    generated_keyframes=0,
                ),
            )
            conditioner_proxy.validate()
            foundation_before, foundation_after = guard.finalize()
        except BaseException as pipeline_error:
            try:
                guard.raise_if_failed()
            except BaseException as guard_error:
                if guard_error is pipeline_error:
                    raise
                raise guard_error from pipeline_error
            raise
        finally:
            mutable_pipeline.image_conditioner = image_conditioner
            mutable_pipeline.stage = stage
            guard.close()
            self._foundation_guard = None
        if _file_sha256(self._teacher_checkpoint_path) != observed_checkpoint_sha256:
            raise ValueError("teacher checkpoint content SHA-256 mismatch")
        video = getattr(result, "video", None)
        audio = getattr(result, "audio", None)
        num_frames = getattr(result, "num_frames", None)
        tiling_config = getattr(result, "tiling_config", None)
        device_final_latent = getattr(result, "video_latent", None)
        if (
            video is None
            or num_frames != _FRAMES
            or not isinstance(device_final_latent, torch.Tensor)
            or device_final_latent.layout != torch.strided
            or tuple(device_final_latent.shape) != _ONE_STAGE_FINAL_LATENT_SHAPE
            or not torch.is_floating_point(device_final_latent)
            or not bool(torch.isfinite(device_final_latent).all().item())
        ):
            raise RuntimeError(_PINNED_ONE_STAGE_MEDIA_ERROR)
        final_latent = (
            device_final_latent.detach()
            .to(device="cpu", dtype=torch.float32, copy=True)
            .contiguous()
        )
        final_latent_sha256 = tensor_sha256(final_latent)
        del device_final_latent
        del result
        chunks = api.get_video_chunks_number(num_frames, tiling_config)
        api.encode_video(
            video=cast(torch.Tensor | Iterator[torch.Tensor], video),
            fps=_FPS,
            audio=audio,
            output_path=str(output_path),
            video_chunks_number=chunks,
        )
        decoded = self._media_probe(output_path).validate()
        if (decoded.width, decoded.height, decoded.frames) != (
            _WIDTH,
            _HEIGHT,
            _FRAMES,
        ) or not math.isclose(decoded.fps, float(_FPS), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "decoded media frames, dimensions, or FPS do not match the decoded-quality lock"
            )
        return OneStageMediaGenerationResult(
            path=output_path,
            final_latent=final_latent,
            final_latent_sha256=final_latent_sha256,
            checkpoint_file_sha256=observed_checkpoint_sha256,
            foundation_before_sha256=foundation_before,
            foundation_after_sha256=foundation_after,
            media_sha256=_file_sha256(output_path),
            offload_mode=PINNED_LTX_OFFLOAD_MODE,
            foundation_guard_sha256=STREAMING_FOUNDATION_GUARD_SHA256,
            width=decoded.width,
            height=decoded.height,
            frames=decoded.frames,
            fps=int(decoded.fps),
            duration_seconds=decoded.duration_seconds,
        ).validate()


class PinnedLTXTwoStageMediaEngine:
    """Run the exact pinned ``TI2VidTwoStagesPipeline`` with stage-one latent guides."""

    def __init__(
        self,
        *,
        teacher_checkpoint_path: Path,
        teacher_checkpoint_file_sha256: str,
        distilled_lora_path: Path,
        distilled_lora_file_sha256: str,
        spatial_upscaler_path: Path,
        spatial_upscaler_file_sha256: str,
        gemma_root: Path,
        expected_teacher_foundation_sha256: str,
        device: torch.device,
        importer: Callable[[str], object] = importlib.import_module,
        media_probe: Callable[[Path], DecodedMediaFacts] = probe_decoded_video,
    ) -> None:
        self._teacher_checkpoint_path = teacher_checkpoint_path
        self._teacher_checkpoint_file_sha256 = teacher_checkpoint_file_sha256
        self._distilled_lora_path = distilled_lora_path
        self._distilled_lora_file_sha256 = distilled_lora_file_sha256
        self._spatial_upscaler_path = spatial_upscaler_path
        self._spatial_upscaler_file_sha256 = spatial_upscaler_file_sha256
        self._gemma_root = gemma_root
        self._expected_teacher_foundation_sha256 = expected_teacher_foundation_sha256
        self._device = device
        self._importer = importer
        self._media_probe = media_probe
        self._api: _PinnedMediaAPI | None = None
        self._pipeline: object | None = None
        self._image_conditioner: object | None = None
        self._stage_1: object | None = None
        self._stage_2: object | None = None

    def _load_api(self) -> _PinnedMediaAPI:
        if self._api is not None:
            return self._api
        try:
            pipelines = self._importer("ltx_pipelines")
            model_paths = self._importer("ltx_pipelines.utils.model_paths")
            loader = self._importer("ltx_core.loader")
            conditioning = self._importer("ltx_core.conditioning")
            guiders = self._importer("ltx_core.components.guiders")
            constants = self._importer("ltx_pipelines.utils.constants")
            video_vae = self._importer("ltx_core.model.video_vae")
            media_io = self._importer("ltx_pipelines.utils.media_io")
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(_PINNED_MEDIA_ERROR) from error
        values = (
            getattr(pipelines, "TI2VidTwoStagesPipeline", None),
            getattr(model_paths, "ModelPaths", None),
            getattr(loader, "LoraPathStrengthAndSDOps", None),
            getattr(conditioning, "VideoConditionByReferenceLatent", None),
            getattr(conditioning, "ConditioningItemAttentionStrengthWrapper", None),
            getattr(guiders, "MultiModalGuiderParams", None),
        )
        callables = (
            getattr(constants, "detect_params", None),
            getattr(video_vae, "get_video_chunks_number", None),
            getattr(media_io, "encode_video", None),
        )
        lora_mapping = getattr(loader, "LTXV_LORA_COMFY_RENAMING_MAP", None)
        if (
            any(not isinstance(value, type) for value in values)
            or any(not callable(value) for value in callables)
            or lora_mapping is None
        ):
            raise RuntimeError(_PINNED_MEDIA_ERROR)
        typed = cast(
            tuple[type[Any], type[Any], type[Any], type[Any], type[Any], type[Any]],
            values,
        )
        functions = cast(
            tuple[
                Callable[[str], object],
                Callable[[int, object], int],
                Callable[..., object],
            ],
            callables,
        )
        self._api = _PinnedMediaAPI(
            pipeline_type=typed[0],
            model_paths_type=typed[1],
            lora_type=typed[2],
            lora_mapping=lora_mapping,
            reference_type=typed[3],
            attention_wrapper_type=typed[4],
            guider_params_type=typed[5],
            detect_params=functions[0],
            get_video_chunks_number=functions[1],
            encode_video=functions[2],
        )
        return self._api

    def _load_pipeline(self, api: _PinnedMediaAPI) -> object:
        if self._pipeline is not None:
            return self._pipeline
        for path, expected, name in (
            (
                self._teacher_checkpoint_path,
                self._teacher_checkpoint_file_sha256,
                "teacher checkpoint",
            ),
            (self._distilled_lora_path, self._distilled_lora_file_sha256, "distilled LoRA"),
            (
                self._spatial_upscaler_path,
                self._spatial_upscaler_file_sha256,
                "spatial upscaler",
            ),
        ):
            if path.is_symlink() or not path.is_file() or _file_sha256(path) != expected:
                raise ValueError(f"{name} content SHA-256 mismatch")
        model_paths_factory = getattr(api.model_paths_type, "from_monolith", None)
        if not callable(model_paths_factory):
            raise RuntimeError(_PINNED_MEDIA_ERROR)
        model_paths = model_paths_factory(str(self._teacher_checkpoint_path), str(self._gemma_root))
        distilled_lora = api.lora_type(str(self._distilled_lora_path), 1.0, api.lora_mapping)
        self._pipeline = api.pipeline_type(
            model_paths=model_paths,
            distilled_lora=[distilled_lora],
            spatial_upsampler_path=str(self._spatial_upscaler_path),
            loras=[],
            device=self._device,
            prompt_enhancer_gemma_root=str(self._gemma_root),
        )
        if not callable(self._pipeline):
            raise RuntimeError(_PINNED_MEDIA_ERROR)
        self._image_conditioner = getattr(self._pipeline, "image_conditioner", None)
        self._stage_1 = getattr(self._pipeline, "stage_1", None)
        self._stage_2 = getattr(self._pipeline, "stage_2", None)
        if self._image_conditioner is None or self._stage_1 is None or self._stage_2 is None:
            raise RuntimeError(_PINNED_MEDIA_ERROR)
        return self._pipeline

    def _references(self, api: _PinnedMediaAPI, guides: tuple[torch.Tensor, ...]) -> list[object]:
        if not guides:
            raise ValueError("media generation requires at least one latent guide")
        references = []
        for guide in guides:
            if (
                not isinstance(guide, torch.Tensor)
                or guide.layout != torch.strided
                or tuple(guide.shape) != (1, 128, 1, 6, 6)
                or not torch.is_floating_point(guide)
                or not bool(torch.isfinite(guide).all().item())
            ):
                raise ValueError("media guides must be finite [1,128,1,6,6] stage-one latents")
            reference = api.reference_type(
                latent=guide.to(device=self._device, dtype=torch.bfloat16),
                downscale_factor=1,
                temporal_scale_factor=1,
                strength=1.0,
            )
            references.append(api.attention_wrapper_type(reference, attention_mask=1.0))
        return references

    @staticmethod
    def _wrapped_stage(stage: object, captured: list[tuple[str, torch.nn.Module]]) -> object:
        wrapper = getattr(stage, "with_model_wrapper", None)
        if not callable(wrapper):
            raise RuntimeError(_PINNED_MEDIA_ERROR)

        def measure(model: object, _video_tools: object) -> object:
            if not isinstance(model, torch.nn.Module):
                raise RuntimeError(_PINNED_MEDIA_ERROR)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            captured.append((foundation_digest(model), model))
            return model

        result = wrapper(measure)
        if not callable(result):
            raise RuntimeError(_PINNED_MEDIA_ERROR)
        return result

    def generate(
        self,
        *,
        guides: tuple[torch.Tensor, ...],
        prompt: str,
        negative_prompt: str,
        seed: int,
        output_path: Path,
    ) -> MediaGenerationResult:
        if not prompt.strip() or not isinstance(negative_prompt, str):
            raise ValueError("media prompts must be explicit strings")
        if type(seed) is not int or seed not in {101, 202, 303}:
            raise ValueError("media seed must be one of the three Decision 1 seeds")
        if (
            output_path.is_symlink()
            or not output_path.is_absolute()
            or not output_path.parent.is_dir()
        ):
            raise ValueError(
                "media output must be an absolute non-symlink path below an existing directory"
            )
        api = self._load_api()
        pipeline = self._load_pipeline(api)
        image_conditioner = self._image_conditioner
        stage_1 = self._stage_1
        stage_2 = self._stage_2
        if image_conditioner is None or stage_1 is None or stage_2 is None:
            raise RuntimeError(_PINNED_MEDIA_ERROR)
        stage_1_models: list[tuple[str, torch.nn.Module]] = []
        stage_2_models: list[tuple[str, torch.nn.Module]] = []
        mutable_pipeline = cast(Any, pipeline)
        mutable_pipeline.image_conditioner = _ConditionerProxy(
            image_conditioner, self._references(api, guides)
        )
        mutable_pipeline.stage_1 = self._wrapped_stage(stage_1, stage_1_models)
        mutable_pipeline.stage_2 = self._wrapped_stage(stage_2, stage_2_models)
        params = api.detect_params(str(self._teacher_checkpoint_path))
        video_guider = getattr(params, "video_guider_params", None)
        audio_guider = getattr(params, "audio_guider_params", None)
        if not isinstance(video_guider, api.guider_params_type) or not isinstance(
            audio_guider, api.guider_params_type
        ):
            raise RuntimeError(_PINNED_MEDIA_ERROR)
        try:
            result = cast(
                Any,
                mutable_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=_HEIGHT,
                    width=_WIDTH,
                    frame_rate=float(_FPS),
                    num_inference_steps=_STEPS,
                    video_guider_params=video_guider,
                    audio_guider_params=audio_guider,
                    images=[],
                    num_frames=_FRAMES,
                    max_batch_size=1,
                ),
            )
        finally:
            mutable_pipeline.image_conditioner = image_conditioner
            mutable_pipeline.stage_1 = stage_1
            mutable_pipeline.stage_2 = stage_2
        if len(stage_1_models) != 1 or len(stage_2_models) != 1:
            raise RuntimeError("each two-stage foundation must be built exactly once")
        teacher_before, teacher_model = stage_1_models[0]
        stage_2_before, stage_2_model = stage_2_models[0]
        teacher_after = foundation_digest(teacher_model)
        stage_2_after = foundation_digest(stage_2_model)
        if teacher_before != self._expected_teacher_foundation_sha256:
            raise ValueError("media teacher foundation does not match the loaded-state lock")
        video = getattr(result, "video", None)
        audio = getattr(result, "audio", None)
        num_frames = getattr(result, "num_frames", None)
        tiling_config = getattr(result, "tiling_config", None)
        final_latent = getattr(result, "video_latent", None)
        if video is None or num_frames != _FRAMES or not isinstance(final_latent, torch.Tensor):
            raise RuntimeError(_PINNED_MEDIA_ERROR)
        chunks = api.get_video_chunks_number(num_frames, tiling_config)
        api.encode_video(
            video=cast(torch.Tensor | Iterator[torch.Tensor], video),
            fps=_FPS,
            audio=audio,
            output_path=str(output_path),
            video_chunks_number=chunks,
        )
        decoded = self._media_probe(output_path).validate()
        dimensions_match = (decoded.width, decoded.height, decoded.frames) == (
            _WIDTH,
            _HEIGHT,
            _FRAMES,
        )
        if not dimensions_match or not math.isclose(
            decoded.fps, float(_FPS), rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                "decoded media frames, dimensions, or FPS do not match the canary lock"
            )
        return MediaGenerationResult(
            output_path,
            final_latent.detach(),
            teacher_before,
            teacher_after,
            stage_2_before,
            stage_2_after,
            width=decoded.width,
            height=decoded.height,
            frames=decoded.frames,
            fps=int(decoded.fps),
            duration_seconds=decoded.duration_seconds,
        ).validate()


__all__ = (
    "Decision1MediaEngine",
    "MediaGenerationResult",
    "OneStageMediaGenerationResult",
    "PinnedLTXOneStageMediaEngine",
    "PinnedLTXTwoStageMediaEngine",
)
