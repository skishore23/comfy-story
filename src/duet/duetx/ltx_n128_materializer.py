"""One-pass N=128 Duet-X materialization for authenticated Comfy PNG inputs.

The materializer keeps the two early hard-pinned observations out of every bounded core.  It
publishes one calibrated Duet core, recent-anchor and learned gated matched-budget controls, the
two byte-exact exception PNGs, and a byte-exact current PNG.  Raw history latents and source PNG
bytes are consumed one at a time rather than retained as a full-history batch.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, Self, cast

import numpy as np
import torch
from PIL import Image
from torch import nn

from duet.duetx.contracts import (
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    RawEvidencePointer,
    SpatialLocation,
    TimeFrameRange,
    tensor_sha256,
)
from duet.duetx.ltx_bridge import LTXLatentHistoryBridge
from duet.duetx.ltx_n128_guide_memory import N128LTXGuideMemory
from duet.duetx.ltx_n128_memory_bundle import (
    N128LTXMemoryBundleReceipt,
    publish_n128_ltx_memory_bundle,
)
from duet.duetx.ltx_quality_generation import FrozenMethodModules
from duet.duetx.ltx_quality_protocol import (
    FROZEN_TRAINABLE_CHECKPOINT_SHA256,
    FrozenRecord,
)
from duet.duetx.ltx_quality_scenes import RGBFrame, rgb_frame_sha256
from duet.duetx.streaming_long_context import (
    StreamingEquivalenceAudit,
    StreamingHistoryLeaf,
    StreamingLongContextContractV1,
    materialize_streaming_long_context,
)

_FORMAT = "duet-x-n128-comfy-materialization-v1"
_IDENTITY_FORMAT = "duet-x-n128-comfy-identity-v1"
_HISTORY_ITEMS = 128
_PIN_COUNT = 2
_EARLY_STOP = _HISTORY_ITEMS // 2
_SOURCE_SIZE = (1024, 1024)
_MATERIALIZED_SIZE = (384, 384)
_LATENT_SHAPE = (1, 128, 1, 12, 12)
_TOKEN_SHAPE = (1, 144, 128)
_LATENT_BYTES = 1 * 128 * 1 * 12 * 12 * 4
_SHA256_HEX = frozenset("0123456789abcdef")


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_order_sha256(rows: tuple[N128HistoryReceipt, ...]) -> str:
    payload = b"".join(
        row.slot.to_bytes(2, "big") + bytes.fromhex(row.source_png_sha256) for row in rows
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class AuthenticatedObservationPNG:
    """One local source path bound to its caller-authenticated encoded PNG digest."""

    path: Path
    sha256: str

    def validate(self) -> Self:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("authenticated observation PNG path must be absolute")
        _sha256(self.sha256, "authenticated observation PNG SHA-256")
        return self


@dataclass(frozen=True, slots=True)
class N128ComfyIdentity(FrozenRecord):
    """Source, preprocessing, foundation, workflow, and rights identities."""

    format: str
    source_contract_sha256: str
    source_registry_sha256: str
    preprocessing_sha256: str
    vae_sha256: str
    workflow_sha256: str
    rights_sha256: str

    def validate(self) -> Self:
        if self.format != _IDENTITY_FORMAT:
            raise ValueError("N=128 Comfy identity format changed")
        for name in (
            "source_contract_sha256",
            "source_registry_sha256",
            "preprocessing_sha256",
            "vae_sha256",
            "workflow_sha256",
            "rights_sha256",
        ):
            _sha256(getattr(self, name), name)
        return self


@dataclass(frozen=True, slots=True)
class N128ComfyRequest:
    """The exact chronological history, early exception slots, current, and destination."""

    history: tuple[AuthenticatedObservationPNG, ...]
    pinned_slots: tuple[int, int]
    current: AuthenticatedObservationPNG
    output_dir: Path
    identity: N128ComfyIdentity

    def validate(self) -> Self:
        if not isinstance(self.history, tuple) or len(self.history) != _HISTORY_ITEMS:
            raise ValueError("N=128 Comfy history requires exactly 128 authenticated PNGs")
        for source in self.history:
            if not isinstance(source, AuthenticatedObservationPNG):
                raise ValueError("N=128 history entries must be authenticated observation PNGs")
            source.validate()
        if (
            not isinstance(self.pinned_slots, tuple)
            or len(self.pinned_slots) != _PIN_COUNT
            or any(type(slot) is not int for slot in self.pinned_slots)
        ):
            raise ValueError("N=128 materialization requires exactly two pinned slots")
        if self.pinned_slots != tuple(sorted(set(self.pinned_slots))):
            raise ValueError("N=128 pinned slots must be ordered unique integers")
        if any(not 0 <= slot < _EARLY_STOP for slot in self.pinned_slots):
            raise ValueError("N=128 pinned exceptions must be in the early half of history")
        self.current.validate()
        if not isinstance(self.output_dir, Path) or not self.output_dir.is_absolute():
            raise ValueError("N=128 output directory must be absolute")
        if self.output_dir.exists() or self.output_dir.is_symlink():
            raise ValueError("N=128 output directory must not exist")
        self.identity.validate()
        return self


@dataclass(frozen=True, slots=True)
class N128HistoryReceipt(FrozenRecord):
    """Digest-only record for one consumed source observation."""

    slot: int
    source_png_sha256: str
    normalized_rgb_sha256: str
    normalized_latent_sha256: str
    normalized_token_sha256: str
    frame_receipt_sha256: str
    pinned: bool

    def validate(self) -> Self:
        if type(self.slot) is not int or not 0 <= self.slot < _HISTORY_ITEMS:
            raise ValueError("history receipt slot is outside N=128")
        for name in (
            "source_png_sha256",
            "normalized_rgb_sha256",
            "normalized_latent_sha256",
            "normalized_token_sha256",
            "frame_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        if type(self.pinned) is not bool:
            raise ValueError("history receipt pinned flag must be boolean")
        return self


@dataclass(frozen=True, slots=True)
class N128StreamingAuditReceipt(FrozenRecord):
    """Topology-equivalence evidence copied from the one-pass materializer."""

    history_items: int
    ordinary_leaf_count: int
    excluded_leaf_count: int
    centralized_operator_sha256: str
    balanced_operator_sha256: str
    max_abs_error: float
    equivalent: bool

    @classmethod
    def capture(cls, audit: StreamingEquivalenceAudit) -> Self:
        audit.validate()
        return cls(
            audit.history_items,
            audit.ordinary_leaf_count,
            audit.excluded_leaf_count,
            audit.centralized_operator_sha256,
            audit.balanced_operator_sha256,
            audit.max_abs_error,
            audit.equivalent,
        ).validate()

    def validate(self) -> Self:
        if (
            self.history_items != _HISTORY_ITEMS
            or self.ordinary_leaf_count != _HISTORY_ITEMS - _PIN_COUNT
            or self.excluded_leaf_count != _PIN_COUNT
        ):
            raise ValueError("N=128 topology audit counts changed")
        _sha256(self.centralized_operator_sha256, "centralized_operator_sha256")
        _sha256(self.balanced_operator_sha256, "balanced_operator_sha256")
        if type(self.max_abs_error) is not float or self.max_abs_error < 0.0:
            raise ValueError("N=128 topology audit error must be nonnegative")
        if self.equivalent is not True:
            raise ValueError("N=128 centralized and balanced products differ")
        return self


@dataclass(frozen=True, slots=True)
class N128RetentionReceipt(FrozenRecord):
    """Exact bounded-state accounting for the streaming latent boundary."""

    input_passes: int
    source_latent_bytes: int
    full_history_latent_bytes: int
    retained_exception_token_bytes: int
    retained_exception_png_bytes: int
    recent_anchor_bytes: int
    gated_accumulator_bytes: int
    duet_operator_bytes: int
    peak_balanced_partial_operators: int
    peak_source_png_bytes: int
    bounded_state_bytes: int
    full_history_latents_retained: bool
    full_history_png_bytes_retained: bool

    def validate(self) -> Self:
        expected_partials = max(value.bit_count() for value in range(1, 127))
        if (
            self.input_passes != 1
            or self.source_latent_bytes != _LATENT_BYTES
            or self.full_history_latent_bytes != _HISTORY_ITEMS * _LATENT_BYTES
            or self.retained_exception_token_bytes != _PIN_COUNT * _LATENT_BYTES
            or self.recent_anchor_bytes != _LATENT_BYTES
            or self.gated_accumulator_bytes != _LATENT_BYTES + 4
            or self.peak_balanced_partial_operators != expected_partials
        ):
            raise ValueError("N=128 bounded-state accounting changed")
        for name in (
            "retained_exception_png_bytes",
            "duet_operator_bytes",
            "peak_source_png_bytes",
            "bounded_state_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        expected_total = (
            self.retained_exception_token_bytes
            + self.retained_exception_png_bytes
            + self.recent_anchor_bytes
            + self.gated_accumulator_bytes
            + self.duet_operator_bytes * (1 + self.peak_balanced_partial_operators)
            + self.peak_source_png_bytes
        )
        if self.bounded_state_bytes != expected_total:
            raise ValueError("N=128 bounded-state byte total changed")
        if self.full_history_latents_retained or self.full_history_png_bytes_retained:
            raise ValueError("N=128 receipt must attest no full-history retention")
        return self


@dataclass(frozen=True, slots=True)
class N128OutputReceipt(FrozenRecord):
    role: str
    filename: str
    png_sha256: str
    source_slot: int

    def validate(self) -> Self:
        allowed = {
            "duet-core": "duet-core.png",
            "recent-anchor": "recent-anchor.png",
            "gated-core": "gated-core.png",
            "exception-00": "exception-00.png",
            "exception-01": "exception-01.png",
            "current": "current.png",
        }
        if allowed.get(self.role) != self.filename:
            raise ValueError("N=128 output role or filename changed")
        _sha256(self.png_sha256, "output PNG SHA-256")
        if self.role.startswith("exception-"):
            if type(self.source_slot) is not int or not 0 <= self.source_slot < _EARLY_STOP:
                raise ValueError("exception output must bind one early source slot")
        elif self.source_slot != -1:
            raise ValueError("non-exception output must use the no-source sentinel")
        return self


@dataclass(frozen=True, slots=True)
class N128ComfyReceipt(FrozenRecord):
    """Complete content and state binding for one N=128 publication."""

    format: str
    identity_sha256: str
    history_items: int
    pinned_slots: tuple[int, int]
    anchor_slot: int
    checkpoint_sha256: str
    core_modules_sha256: str
    streaming_contract_sha256: str
    source_order_sha256: str
    history: tuple[N128HistoryReceipt, ...]
    streaming_audit: N128StreamingAuditReceipt
    streaming_audit_sha256: str
    retention: N128RetentionReceipt
    retention_sha256: str
    duet_core_operator_sha256: str
    duet_core_tokens_sha256: str
    duet_calibrated_latent_sha256: str
    recent_anchor_latent_sha256: str
    gated_core_latent_sha256: str
    exception_token_sha256: tuple[str, str]
    current_source_sha256: str
    outputs: tuple[N128OutputReceipt, ...]

    def validate(self) -> Self:
        if self.format != _FORMAT or self.history_items != _HISTORY_ITEMS:
            raise ValueError("N=128 receipt format or history length changed")
        _sha256(self.identity_sha256, "identity_sha256")
        if (
            not isinstance(self.pinned_slots, tuple)
            or len(self.pinned_slots) != _PIN_COUNT
            or self.pinned_slots != tuple(sorted(set(self.pinned_slots)))
            or any(not 0 <= slot < _EARLY_STOP for slot in self.pinned_slots)
        ):
            raise ValueError("N=128 receipt requires two ordered early pinned slots")
        expected_anchor = max(set(range(_HISTORY_ITEMS)) - set(self.pinned_slots))
        if self.anchor_slot != expected_anchor:
            raise ValueError("N=128 recent anchor must be the latest ordinary slot")
        if self.checkpoint_sha256 != FROZEN_TRAINABLE_CHECKPOINT_SHA256:
            raise ValueError("N=128 frozen checkpoint changed")
        for name in (
            "core_modules_sha256",
            "streaming_contract_sha256",
            "source_order_sha256",
            "streaming_audit_sha256",
            "retention_sha256",
            "duet_core_operator_sha256",
            "duet_core_tokens_sha256",
            "duet_calibrated_latent_sha256",
            "recent_anchor_latent_sha256",
            "gated_core_latent_sha256",
            "current_source_sha256",
        ):
            _sha256(getattr(self, name), name)
        if not isinstance(self.history, tuple) or len(self.history) != _HISTORY_ITEMS:
            raise ValueError("N=128 receipt requires 128 ordered history rows")
        for slot, row in enumerate(self.history):
            row.validate()
            if row.slot != slot or row.pinned != (slot in self.pinned_slots):
                raise ValueError("N=128 receipt source order or pin binding changed")
        if _source_order_sha256(self.history) != self.source_order_sha256:
            raise ValueError("N=128 source order digest changed")
        self.streaming_audit.validate()
        if self.streaming_audit.fingerprint() != self.streaming_audit_sha256:
            raise ValueError("N=128 topology audit parent changed")
        if self.streaming_audit.centralized_operator_sha256 != self.duet_core_operator_sha256:
            raise ValueError("N=128 topology audit does not bind the Duet core")
        self.retention.validate()
        if self.retention.fingerprint() != self.retention_sha256:
            raise ValueError("N=128 retention receipt parent changed")
        if (
            not isinstance(self.exception_token_sha256, tuple)
            or len(self.exception_token_sha256) != _PIN_COUNT
        ):
            raise ValueError("N=128 receipt requires exactly two exception token hashes")
        if self.exception_token_sha256 != tuple(
            self.history[slot].normalized_token_sha256 for slot in self.pinned_slots
        ):
            raise ValueError("N=128 exception hashes do not bind the pinned observations")
        for digest in self.exception_token_sha256:
            _sha256(digest, "exception token SHA-256")
        expected_roles = (
            "duet-core",
            "recent-anchor",
            "gated-core",
            "exception-00",
            "exception-01",
            "current",
        )
        if (
            not isinstance(self.outputs, tuple)
            or tuple(output.role for output in self.outputs) != expected_roles
        ):
            raise ValueError("N=128 output roster or order changed")
        for output in self.outputs:
            output.validate()
        if tuple(output.source_slot for output in self.outputs[3:5]) != self.pinned_slots:
            raise ValueError("N=128 exception outputs do not bind the pinned slots")
        if tuple(output.png_sha256 for output in self.outputs[3:5]) != tuple(
            self.history[slot].source_png_sha256 for slot in self.pinned_slots
        ):
            raise ValueError("N=128 exception outputs are not byte-exact source PNGs")
        if self.outputs[5].png_sha256 != self.current_source_sha256:
            raise ValueError("N=128 current output is not a byte-exact source PNG")
        return self


class N128FrameMaterializer(Protocol):
    def materialize_frame(self, frame: RGBFrame) -> Any: ...


N128LatentDecoder = Callable[[torch.Tensor], RGBFrame]


@dataclass(frozen=True, slots=True)
class _LoadedPNG:
    source: AuthenticatedObservationPNG
    encoded: bytes
    normalized: RGBFrame


def _load_png(source: AuthenticatedObservationPNG) -> _LoadedPNG:
    source.validate()
    path = source.path
    if path.is_symlink() or not path.is_file():
        raise ValueError("authenticated observation PNG must be a regular non-symlink file")
    encoded = path.read_bytes()
    if not encoded or len(encoded) > 64 * 1024 * 1024:
        raise ValueError("authenticated observation PNG encoded size is unsafe")
    if _file_sha256(encoded) != source.sha256:
        raise ValueError("authenticated observation PNG content SHA-256 changed")
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size != _SOURCE_SIZE
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError("authenticated observation PNG must be single-frame RGB 1024x1024")
            normalized_image = image.resize(_MATERIALIZED_SIZE, Image.Resampling.LANCZOS)
            normalized = np.asarray(normalized_image, dtype=np.uint8).copy(order="C")
    except (OSError, SyntaxError) as error:
        raise ValueError("authenticated observation PNG could not be decoded") from error
    if normalized.shape != (384, 384, 3) or not normalized.flags.c_contiguous:
        raise ValueError("authenticated observation PNG normalization changed")
    return _LoadedPNG(source, encoded, normalized)


def _validate_latent(value: object) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or tuple(value.shape) != _LATENT_SHAPE
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or value.requires_grad
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError("N=128 frame latent must be detached CPU float32 [1,128,1,12,12]")
    return value


def _tokens(latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = latent.to(device=device).permute(0, 2, 3, 4, 1).flatten(1, 3).contiguous()
    if tuple(value.shape) != _TOKEN_SHAPE:
        raise RuntimeError("N=128 latent token conversion changed")
    return value


def _calibrated_duet_latent(
    bridge: LTXLatentHistoryBridge,
    core_tokens: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    anchor_tokens = anchor.permute(0, 2, 3, 4, 1).flatten(1, 3)
    calibrated_tokens = cast(
        torch.Tensor,
        anchor_tokens + bridge.residual(core_tokens).to(dtype=anchor_tokens.dtype),
    )
    calibrated = calibrated_tokens.reshape(1, 1, 12, 12, 128).permute(0, 4, 1, 2, 3)
    calibrated = calibrated.contiguous()
    if not bool(torch.isfinite(calibrated).all().item()):
        raise ValueError("N=128 calibrated Duet latent must be finite")
    return calibrated


class _OnlineGatedCore:
    """Exact online form of the frozen matched-budget gated weighted mean."""

    def __init__(self, gated: nn.Module, device: torch.device) -> None:
        gate = getattr(gated, "gate", None)
        if not isinstance(gate, nn.Module):
            raise ValueError("frozen gated control lacks its learned gate module")
        parameters = tuple(gate.parameters())
        if not parameters:
            raise ValueError("frozen gated control has no learned parameters")
        self._gate = gate
        self._device = device
        self._dtype = parameters[0].dtype
        self._numerator: torch.Tensor | None = None
        self._denominator: torch.Tensor | None = None

    def append(self, latent: torch.Tensor) -> None:
        value = latent.to(device=self._device, dtype=self._dtype)
        descriptor = value.mean(dim=(2, 3, 4)).unsqueeze(1)
        gate = self._gate(descriptor).sigmoid()
        if tuple(gate.shape) != (1, 1, 1) or not bool(torch.isfinite(gate).all().item()):
            raise ValueError("frozen gated control produced an invalid per-observation gate")
        weighted = value * gate[:, 0, :, None, None]
        self._numerator = weighted if self._numerator is None else self._numerator + weighted
        denominator = gate[:, 0, :]
        self._denominator = (
            denominator if self._denominator is None else self._denominator + denominator
        )

    def finish(self) -> torch.Tensor:
        if self._numerator is None or self._denominator is None:
            raise ValueError("gated control requires at least one ordinary observation")
        denominator = self._denominator.clamp_min(torch.finfo(self._denominator.dtype).eps)
        core = self._numerator / denominator[:, :, None, None]
        if tuple(core.shape) != _LATENT_SHAPE or not bool(torch.isfinite(core).all().item()):
            raise ValueError("online gated control latent boundary changed")
        return core.to(dtype=torch.float32).contiguous()


def _exception(
    *,
    slot: int,
    loaded: _LoadedPNG,
    tokens: torch.Tensor,
    identity: N128ComfyIdentity,
    memory_contract_sha256: str,
) -> ExceptionItem:
    embedding = tokens.detach().float().mean(dim=(0, 1)).contiguous()
    return ExceptionItem(
        f"n128-exception-{slot:03d}",
        EvidenceProvenance(
            LeafKey(f"n128-source-{slot:03d}", slot, slot),
            TimeFrameRange(slot, slot + 1, slot, slot + 1),
            SpatialLocation("source-png-q16-v1", "bbox", (0, 0, 65_536, 65_536)),
            "hard-pinned-early-whole-frame-v1",
            RawEvidencePointer(
                f"duet-evidence://n128/sha256/{loaded.source.sha256}",
                loaded.source.sha256,
                0,
                len(loaded.encoded),
            ),
            identity.source_registry_sha256,
            identity.preprocessing_sha256,
            identity.vae_sha256,
            identity.workflow_sha256,
            identity.rights_sha256,
            memory_contract_sha256,
        ),
        embedding,
        slot,
        pinned=True,
    )


def _png_bytes(frame: object) -> bytes:
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or tuple(frame.shape) != (384, 384, 3)
        or not frame.flags.c_contiguous
    ):
        raise ValueError("N=128 latent decoder must return C-contiguous RGB uint8 384x384")
    target = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(target, format="PNG", optimize=False)
    return target.getvalue()


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _publish(output_dir: Path, files: tuple[tuple[str, bytes], ...]) -> None:
    parent = output_dir.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("N=128 output parent must be a regular directory")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    temporary.chmod(0o700)
    try:
        for filename, data in files:
            _write_new(temporary / filename, data)
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError("N=128 output directory must not exist")
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@torch.inference_mode()
def _materialize_n128_comfy(
    request: N128ComfyRequest,
    modules: FrozenMethodModules,
    frame_materializer: N128FrameMaterializer,
    latent_decoder: N128LatentDecoder,
    *,
    memory_publisher: Callable[[N128LTXGuideMemory], None] | None,
) -> N128ComfyReceipt:
    """Publish one authenticated, bounded, exactly-once N=128 Comfy guide bundle."""
    request.validate()
    if not isinstance(modules, FrozenMethodModules):
        raise ValueError("N=128 materialization requires frozen method modules")
    modules.validate()
    if not callable(frame_materializer.materialize_frame) or not callable(latent_decoder):
        raise ValueError("N=128 materialization dependencies must be callable")
    bridge = modules.bridge
    if type(bridge) is not LTXLatentHistoryBridge:
        raise ValueError("N=128 materialization requires the frozen LTX latent bridge")
    bridge_device = bridge.fusion.input_norm.weight.device
    contract = replace(
        StreamingLongContextContractV1.default(
            source_contract_sha256=request.identity.source_contract_sha256,
            history_items=_HISTORY_ITEMS,
            exception_capacity=_PIN_COUNT,
            latent_channels=128,
        ),
        equivalence_atol=1e-5,
        equivalence_rtol=1e-5,
    ).validate()
    memory_contract_sha256 = contract.sparse.fingerprint()
    gated = _OnlineGatedCore(modules.gated, modules.device)
    rows: list[N128HistoryReceipt] = []
    loaded_pins: dict[int, _LoadedPNG] = {}
    recent_anchor: torch.Tensor | None = None
    peak_source_png_bytes = 0

    def leaves() -> Any:
        nonlocal peak_source_png_bytes, recent_anchor
        for slot, source in enumerate(request.history):
            loaded = _load_png(source)
            peak_source_png_bytes = max(peak_source_png_bytes, len(loaded.encoded))
            result = frame_materializer.materialize_frame(loaded.normalized)
            latent = _validate_latent(result.latent)
            receipt_fingerprint = getattr(result.receipt, "fingerprint", None)
            if not callable(receipt_fingerprint):
                raise ValueError("N=128 frame receipt is not fingerprintable")
            frame_receipt_sha256 = receipt_fingerprint()
            _sha256(frame_receipt_sha256, "frame receipt SHA-256")
            tokens = _tokens(latent, bridge_device)
            pinned = slot in request.pinned_slots
            rows.append(
                N128HistoryReceipt(
                    slot,
                    source.sha256,
                    rgb_frame_sha256(loaded.normalized),
                    tensor_sha256(latent),
                    tensor_sha256(tokens),
                    frame_receipt_sha256,
                    pinned,
                ).validate()
            )
            if pinned:
                loaded_pins[slot] = loaded
            else:
                recent_anchor = latent.to(device=bridge_device, copy=True)
                gated.append(latent)
            yield StreamingHistoryLeaf(
                slot,
                tokens,
                _exception(
                    slot=slot,
                    loaded=loaded,
                    tokens=tokens,
                    identity=request.identity,
                    memory_contract_sha256=memory_contract_sha256,
                )
                if pinned
                else None,
            )

    streaming = materialize_streaming_long_context(contract, leaves(), bridge.fusion)
    if recent_anchor is None or len(loaded_pins) != _PIN_COUNT:
        raise RuntimeError("N=128 stream did not retain its bounded anchor and exceptions")
    duet_latent = _calibrated_duet_latent(bridge, streaming.core_tokens, recent_anchor)
    if memory_publisher is not None:
        memory_publisher(
            N128LTXGuideMemory(
                streaming,
                duet_latent,
                tuple(f"n128-ordinary-{row.slot:03d}" for row in rows if not row.pinned),
                modules.checkpoint_sha256,
                modules.core_modules_sha256,
            )
        )
    gated_latent = gated.finish()
    recent_latent = recent_anchor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    duet_png = _png_bytes(latent_decoder(duet_latent))
    recent_png = _png_bytes(latent_decoder(recent_latent))
    gated_png = _png_bytes(latent_decoder(gated_latent))
    current = _load_png(request.current)
    pin_sources = tuple(loaded_pins[slot] for slot in request.pinned_slots)
    output_bytes = (
        duet_png,
        recent_png,
        gated_png,
        pin_sources[0].encoded,
        pin_sources[1].encoded,
        current.encoded,
    )
    roles = (
        "duet-core",
        "recent-anchor",
        "gated-core",
        "exception-00",
        "exception-01",
        "current",
    )
    filenames = (
        "duet-core.png",
        "recent-anchor.png",
        "gated-core.png",
        "exception-00.png",
        "exception-01.png",
        "current.png",
    )
    outputs = tuple(
        N128OutputReceipt(role, filename, _file_sha256(data), source_slot).validate()
        for role, filename, data, source_slot in zip(
            roles,
            filenames,
            output_bytes,
            (-1, -1, -1, request.pinned_slots[0], request.pinned_slots[1], -1),
            strict=True,
        )
    )
    audit = N128StreamingAuditReceipt.capture(streaming.audit)
    operator_bytes = streaming.core_operator.numel() * streaming.core_operator.element_size()
    exception_png_bytes = sum(len(source.encoded) for source in pin_sources)
    peak_partials = max(value.bit_count() for value in range(1, 127))
    bounded_state_bytes = (
        _PIN_COUNT * _LATENT_BYTES
        + exception_png_bytes
        + _LATENT_BYTES
        + _LATENT_BYTES
        + 4
        + operator_bytes * (1 + peak_partials)
        + peak_source_png_bytes
    )
    retention = N128RetentionReceipt(
        1,
        _LATENT_BYTES,
        _HISTORY_ITEMS * _LATENT_BYTES,
        _PIN_COUNT * _LATENT_BYTES,
        exception_png_bytes,
        _LATENT_BYTES,
        _LATENT_BYTES + 4,
        operator_bytes,
        peak_partials,
        peak_source_png_bytes,
        bounded_state_bytes,
        False,
        False,
    ).validate()
    history = tuple(rows)
    receipt = N128ComfyReceipt(
        _FORMAT,
        request.identity.fingerprint(),
        _HISTORY_ITEMS,
        request.pinned_slots,
        max(set(range(_HISTORY_ITEMS)) - set(request.pinned_slots)),
        modules.checkpoint_sha256,
        modules.core_modules_sha256,
        contract.fingerprint(),
        _source_order_sha256(history),
        history,
        audit,
        audit.fingerprint(),
        retention,
        retention.fingerprint(),
        tensor_sha256(streaming.core_operator),
        tensor_sha256(streaming.core_tokens),
        tensor_sha256(duet_latent),
        tensor_sha256(recent_latent),
        tensor_sha256(gated_latent),
        cast(
            tuple[str, str],
            tuple(exception.raw_leaf_sha256 for exception in streaming.exceptions),
        ),
        current.source.sha256,
        outputs,
    ).validate()
    modules.validate()
    _publish(
        request.output_dir,
        (*tuple(zip(filenames, output_bytes, strict=True)), ("manifest.json", receipt.to_json())),
    )
    return receipt


def materialize_n128_comfy(
    request: N128ComfyRequest,
    modules: FrozenMethodModules,
    frame_materializer: N128FrameMaterializer,
    latent_decoder: N128LatentDecoder,
) -> N128ComfyReceipt:
    """Publish the existing PNG bundle without changing its public API or receipt."""
    return _materialize_n128_comfy(
        request,
        modules,
        frame_materializer,
        latent_decoder,
        memory_publisher=None,
    )


def materialize_n128_comfy_with_memory_bundle(
    request: N128ComfyRequest,
    modules: FrozenMethodModules,
    frame_materializer: N128FrameMaterializer,
    latent_decoder: N128LatentDecoder,
    bundle_root: Path,
) -> tuple[N128ComfyReceipt, N128LTXMemoryBundleReceipt]:
    """Publish PNG evidence and the live tensor memory through separate atomic boundaries."""
    bundle_receipt: N128LTXMemoryBundleReceipt | None = None

    def publish(memory: N128LTXGuideMemory) -> None:
        nonlocal bundle_receipt
        bundle_receipt = publish_n128_ltx_memory_bundle(bundle_root, memory)

    comfy_receipt = _materialize_n128_comfy(
        request,
        modules,
        frame_materializer,
        latent_decoder,
        memory_publisher=publish,
    )
    if bundle_receipt is None:
        raise RuntimeError("N=128 live tensor bundle was not published")
    return comfy_receipt, bundle_receipt


__all__ = (
    "AuthenticatedObservationPNG",
    "N128ComfyIdentity",
    "N128ComfyReceipt",
    "N128ComfyRequest",
    "N128FrameMaterializer",
    "N128HistoryReceipt",
    "N128LatentDecoder",
    "N128OutputReceipt",
    "N128RetentionReceipt",
    "N128StreamingAuditReceipt",
    "materialize_n128_comfy",
    "materialize_n128_comfy_with_memory_bundle",
)
