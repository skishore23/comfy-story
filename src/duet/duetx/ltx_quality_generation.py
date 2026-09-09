"""Sealed, exactly-once generation for the frozen LTX decoded-quality gate."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import time
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, Self, Union, cast, get_args, get_origin, get_type_hints

import torch
from torch import nn

from duet.duetx.checkpoint import TrainingModules
from duet.duetx.contracts import (
    DuetXContract,
    EvidenceProvenance,
    ExceptionItem,
    LeafKey,
    RawEvidencePointer,
    SpatialLocation,
    TimeFrameRange,
    tensor_sha256,
)
from duet.duetx.ltx_media import (
    PINNED_LTX_OFFLOAD_MODE,
    STREAMING_FOUNDATION_GUARD_SHA256,
    OneStageMediaGenerationResult,
)
from duet.duetx.ltx_quality_file_io import read_single_link_regular_file
from duet.duetx.ltx_quality_protocol import (
    FROZEN_TRAINABLE_CHECKPOINT_SHA256,
    GENERATION_CELL_COUNT,
    MATCHED_SEEDS,
    CellState,
    CommittedGenerationCell,
    CommittedGenerationMatrix,
    CounterfactualCluster,
    FrozenRecord,
    GenerationCellKey,
    GenerationCellLifecycle,
    GuideSource,
    GuideSourceKind,
    Method,
    MethodGuideReceipt,
    QualityProtocol,
    ReviewPhase,
    ReviewPhaseHistory,
    RuntimeCoordinate,
    SceneVariant,
    canonical_json,
    canonical_sha256,
    committed_generation_manifest_sha256,
    scene_manifest_sha256,
    validate_confirmatory_clusters,
    validate_generation_cell_keys,
)
from duet.duetx.training import build_training_modules

_FORMAT = "duet-x-ltx-quality-generation-v1"
_HOST_IDS = ("forge1", "forge2")
_GUIDE_SHAPE = (1, 128, 1, 12, 12)
_CORE_INPUT_SLOTS = (0, 1, 3, 4, 6, 7)
_PROTECTED_CURRENT_SLOTS = (2, 5, 8)
_SHA256_HEX = frozenset("0123456789abcdef")
_CANONICAL_GATED_TYPE = "duet.duetx.methods._GatedCoreMethod"
_CANONICAL_BRIDGE_TYPE = "duet.duetx.ltx_bridge.LTXLatentHistoryBridge"


def _matches_frozen_target_device(actual: torch.device, expected: torch.device) -> bool:
    if expected.type != "cuda":
        return actual == expected
    if actual.type != "cuda":
        return False
    expected_index = cast(object, expected.index)
    actual_index = cast(object, actual.index)
    if expected_index is not None:
        return actual_index == expected_index
    return actual_index in (None, 0)


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_regular_bytes(
    path: Path,
    name: str = "required file",
    *,
    required_mode: int | None = None,
    max_bytes: int | None = None,
) -> bytes:
    """Read one stable, single-link regular file through its checked descriptor."""
    return read_single_link_regular_file(
        path,
        name=name,
        required_mode=required_mode,
        max_bytes=max_bytes,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(_read_regular_bytes(path)).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in generation evidence: {key}")
        result[key] = value
    return result


def _decode_wire(annotation: object, raw: object, name: str) -> object:
    origin = get_origin(annotation)
    if origin is tuple:
        if not isinstance(raw, list):
            raise ValueError(f"{name} must be an array")
        arguments = get_args(annotation)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_decode_wire(arguments[0], item, f"{name} item") for item in raw)
        if len(raw) != len(arguments):
            raise ValueError(f"{name} has the wrong tuple length")
        return tuple(
            _decode_wire(expected, item, f"{name}[{index}]")
            for index, (expected, item) in enumerate(zip(arguments, raw, strict=True))
        )
    if origin in {types.UnionType, Union}:
        arguments = get_args(annotation)
        if raw is None and type(None) in arguments:
            return None
        candidates = tuple(item for item in arguments if item is not type(None))
        errors: list[Exception] = []
        for candidate in candidates:
            try:
                return _decode_wire(candidate, raw, name)
            except (TypeError, ValueError) as error:
                errors.append(error)
        raise ValueError(f"{name} does not match its union schema") from errors[-1]
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        if type(raw) is not str:
            raise ValueError(f"{name} must be a string enum")
        return annotation(raw)
    if isinstance(annotation, type) and issubclass(annotation, FrozenRecord):
        if not isinstance(raw, Mapping) or any(type(key) is not str for key in raw):
            raise ValueError(f"{name} must be an object")
        record_fields = fields(cast(Any, annotation))
        expected = {field.name for field in record_fields}
        if set(raw) != expected:
            raise ValueError(f"{name} has missing or unknown fields")
        annotations = get_type_hints(annotation)
        instance = annotation(
            **{
                field.name: _decode_wire(
                    annotations[field.name], raw[field.name], f"{name}.{field.name}"
                )
                for field in record_fields
            }
        )
        return instance.validate()
    if annotation in {str, int, float, bool}:
        if type(raw) is not annotation:
            raise ValueError(f"{name} has the wrong primitive type")
        return raw
    raise TypeError(f"unsupported generation evidence annotation: {annotation!r}")


def _from_json(data: bytes, cls: type[Any]) -> Any:
    try:
        raw = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError("generation evidence is not valid JSON") from error
    return _decode_wire(cls, raw, cls.__name__)


def _module_state_digest(modules: TrainingModules) -> str:
    rows: list[list[str]] = []
    for module_name, module in sorted(modules.named().items()):
        for tensor_name, tensor in sorted(module.state_dict().items()):
            rows.append([module_name, tensor_name, tensor_sha256(tensor)])
    payload = b"duet-x-ltx-probe-modules-v1\0" + canonical_json(rows) + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _core_module_digest(gated: nn.Module, bridge: nn.Module) -> str:
    rows: list[object] = []
    for root_name, root in (("gated", gated), ("bridge", bridge)):
        rows.append(
            [
                root_name,
                f"{type(root).__module__}.{type(root).__qualname__}",
                [
                    [name, f"{type(module).__module__}.{type(module).__qualname__}"]
                    for name, module in root.named_modules()
                ],
                [
                    [name, tensor_sha256(tensor)]
                    for name, tensor in sorted(root.state_dict().items())
                ],
            ]
        )
    return canonical_sha256({"format": "duet-x-frozen-core-modules-v1", "modules": rows})


def _tensor_guards(gated: nn.Module, bridge: nn.Module) -> tuple[tuple[str, int, int, int], ...]:
    guards: list[tuple[str, int, int, int]] = []
    for root_name, root in (("gated", gated), ("bridge", bridge)):
        for name, tensor in sorted(root.state_dict(keep_vars=True).items()):
            guards.append(
                (
                    f"{root_name}.{name}",
                    id(tensor),
                    tensor._version,
                    tensor.data_ptr(),
                )
            )
    return tuple(guards)


@dataclass(frozen=True, slots=True)
class HostRuntimeIdentity(FrozenRecord):
    """One explicitly named host/runtime/foundation identity."""

    host_id: str
    runtime_identity_sha256: str
    model_foundation_sha256: str
    runtime_equivalence_receipt_sha256: str

    def validate(self) -> Self:
        if self.host_id not in _HOST_IDS:
            raise ValueError("host identity must be forge1 or forge2")
        _sha256(self.runtime_identity_sha256, "runtime_identity_sha256")
        _sha256(self.model_foundation_sha256, "model_foundation_sha256")
        _sha256(
            self.runtime_equivalence_receipt_sha256,
            "runtime_equivalence_receipt_sha256",
        )
        return self


class GenerationStatusCode(StrEnum):
    """Finite public states that cannot encode scientific results."""

    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    TERMINAL_EVIDENCE_FAILURE = "terminal_evidence_failure"


class ApparatusErrorCode(StrEnum):
    """Finite non-disclosing apparatus failures permitted before reveal."""

    INVALID_SEAL = "invalid_seal"
    PARTITION_MISMATCH = "partition_mismatch"
    STARTED_WITHOUT_ARTIFACT = "started_without_artifact"
    ARTIFACT_INTEGRITY_FAILED = "artifact_integrity_failed"
    RUNTIME_INTEGRITY_FAILED = "runtime_integrity_failed"
    NOT_CONFIGURED = "not_configured"


@dataclass(frozen=True, slots=True)
class PartitionTerminalReceipt(FrozenRecord):
    """Immutable host-partition failure that forbids a later model retry."""

    format: str
    host_id: str
    preregistration_sha256: str
    partition_sha256: str
    lifecycle_authorization_sha256: str
    completed_cells: int
    apparatus_error: ApparatusErrorCode

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-partition-terminal-v1":
            raise ValueError("partition terminal receipt format changed")
        if self.host_id not in {"forge1", "forge2"}:
            raise ValueError("partition terminal host changed")
        _sha256(self.preregistration_sha256, "terminal preregistration SHA-256")
        _sha256(self.partition_sha256, "terminal partition SHA-256")
        _sha256(
            self.lifecycle_authorization_sha256,
            "terminal lifecycle authorization SHA-256",
        )
        if type(self.completed_cells) is not int or not 0 <= self.completed_cells <= 216:
            raise ValueError("partition terminal completed count is invalid")
        if not isinstance(self.apparatus_error, ApparatusErrorCode):
            raise ValueError("partition terminal apparatus error is not frozen")
        return self


def load_partition_terminal_receipt(path: Path) -> PartitionTerminalReceipt:
    encoded = _read_regular_bytes(
        path,
        "partition terminal receipt",
        required_mode=0o600,
        max_bytes=1024 * 1024,
    )
    receipt = cast(
        PartitionTerminalReceipt,
        _from_json(encoded, PartitionTerminalReceipt),
    ).validate()
    if receipt.to_json() != encoded:
        raise ValueError("partition terminal receipt must be canonical")
    return receipt


@dataclass(frozen=True, slots=True)
class PhaseInputSeal(FrozenRecord):
    """Canonical prior-phase parent accepted by pre-preregistration CLI commands."""

    format: str
    next_command: str
    phases: ReviewPhaseHistory
    parent_sha256: str

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-phase-input-v1":
            raise ValueError("phase input seal format changed")
        expected = {
            "materialize": (ReviewPhase.SOURCE_AUTHORED,),
            "pilot": (ReviewPhase.SOURCE_AUTHORED,),
            "preregister": (
                ReviewPhase.SOURCE_AUTHORED,
                ReviewPhase.DEVELOPMENT_REHEARSED,
            ),
        }
        if self.next_command not in expected:
            raise ValueError("phase input seal command changed")
        self.phases.validate()
        if self.phases.phases != expected[self.next_command]:
            raise ValueError("phase input seal history is invalid for the requested command")
        _sha256(self.parent_sha256, "phase input parent SHA-256")
        return self


def load_phase_input_seal(path: Path) -> PhaseInputSeal:
    encoded = _read_regular_bytes(
        path, "phase input seal", required_mode=0o600, max_bytes=1024 * 1024
    )
    seal = cast(PhaseInputSeal, _from_json(encoded, PhaseInputSeal)).validate()
    if seal.to_json() != encoded:
        raise ValueError("phase input seal is not canonical")
    return seal


@dataclass(frozen=True, slots=True)
class GenerationPrerequisites(FrozenRecord):
    """The complete preregistered parent chain required by confirmatory commands."""

    format: str
    phases: ReviewPhaseHistory
    protocol_sha256: str
    scene_manifest_sha256: str
    clusters: tuple[CounterfactualCluster, ...]
    materialized_population_sha256: str
    trainable_checkpoint_sha256: str
    runtime: RuntimeCoordinate
    pilot_evidence_sha256: str
    rehearsal_evidence_sha256: str
    source_commit: str
    source_archive_sha256: str
    operational_configuration_sha256: str
    runtime_equivalence_seal_sha256: str
    hosts: tuple[HostRuntimeIdentity, HostRuntimeIdentity]

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-preregistered-v1":
            raise ValueError("preregistration format changed")
        self.phases.validate()
        expected_phases = (
            ReviewPhase.SOURCE_AUTHORED,
            ReviewPhase.DEVELOPMENT_REHEARSED,
            ReviewPhase.PREREGISTERED,
        )
        if self.phases.phases != expected_phases:
            raise ValueError("confirmatory generation requires the sealed preregistration phase")
        if self.protocol_sha256 != QualityProtocol.default().fingerprint():
            raise ValueError("preregistration protocol binding changed")
        for name in (
            "scene_manifest_sha256",
            "materialized_population_sha256",
            "pilot_evidence_sha256",
            "rehearsal_evidence_sha256",
            "source_archive_sha256",
            "operational_configuration_sha256",
            "runtime_equivalence_seal_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            type(self.source_commit) is not str
            or len(self.source_commit) != 40
            or any(character not in _SHA256_HEX for character in self.source_commit)
        ):
            raise ValueError("confirmatory source commit must be a lowercase 40-hex commit")
        clusters = validate_confirmatory_clusters(self.clusters)
        if self.scene_manifest_sha256 != scene_manifest_sha256(clusters):
            raise ValueError("preregistration scene manifest binding changed")
        if self.trainable_checkpoint_sha256 != FROZEN_TRAINABLE_CHECKPOINT_SHA256:
            raise ValueError("preregistration trainable checkpoint binding changed")
        self.runtime.validate()
        if type(self.hosts) is not tuple or len(self.hosts) != 2:
            raise ValueError("preregistration requires exactly two host identities")
        for host in self.hosts:
            host.validate()
        if tuple(host.host_id for host in self.hosts) != _HOST_IDS:
            raise ValueError("host identities must be ordered forge1 then forge2")
        if len({host.runtime_identity_sha256 for host in self.hosts}) != 2:
            raise ValueError("the two hosts require distinct runtime identities")
        if len({host.model_foundation_sha256 for host in self.hosts}) != 1:
            raise ValueError("both hosts must bind the same frozen model foundation")
        return self


def load_generation_prerequisites(path: Path) -> GenerationPrerequisites:
    """Load an exact canonical preregistration seal from one checked descriptor."""
    encoded = _read_regular_bytes(
        path,
        "generation prerequisite seal",
        required_mode=0o600,
        max_bytes=16 * 1024 * 1024,
    )
    prerequisite = cast(
        GenerationPrerequisites, _from_json(encoded, GenerationPrerequisites)
    ).validate()
    if prerequisite.to_json() != encoded:
        raise ValueError("generation prerequisite seal is not canonical")
    return prerequisite


@dataclass(frozen=True, slots=True)
class FrozenMethodModules:
    """Only the two trained core modules needed by the four-method gate."""

    checkpoint_sha256: str
    gated: nn.Module
    bridge: nn.Module
    device: torch.device
    expected_gated_type: type[nn.Module]
    expected_bridge_type: type[nn.Module]
    core_modules_sha256: str
    tensor_guards: tuple[tuple[str, int, int, int], ...]

    def validate(self) -> Self:
        if self.checkpoint_sha256 != FROZEN_TRAINABLE_CHECKPOINT_SHA256:
            raise ValueError("frozen trainable checkpoint SHA-256 changed")
        if not isinstance(self.gated, nn.Module) or not isinstance(self.bridge, nn.Module):
            raise ValueError("frozen core modules must be torch modules")
        if (
            type(self.gated) is not self.expected_gated_type
            or type(self.bridge) is not self.expected_bridge_type
        ):
            raise ValueError("frozen core module architecture changed")
        if not isinstance(self.device, torch.device):
            raise ValueError("module device must be a torch.device")
        _sha256(self.core_modules_sha256, "core_modules_sha256")
        for module in (self.gated, self.bridge):
            for tensor in module.state_dict().values():
                if (
                    tensor.layout != torch.strided
                    or not bool(torch.isfinite(tensor).all().item())
                    or not _matches_frozen_target_device(tensor.device, self.device)
                ):
                    raise ValueError("core module state must be finite on the declared device")
            if module.training:
                raise ValueError("frozen core modules must be in evaluation mode")
            if any(parameter.requires_grad for parameter in module.parameters()):
                raise ValueError("frozen core module parameters must not require gradients")
        if _core_module_digest(self.gated, self.bridge) != self.core_modules_sha256:
            raise ValueError("frozen core module tensor state changed")
        if _tensor_guards(self.gated, self.bridge) != self.tensor_guards:
            raise ValueError("frozen core module tensor identity or version changed")
        return self

    @classmethod
    def capture(
        cls,
        checkpoint_sha256: str,
        gated: nn.Module,
        bridge: nn.Module,
        device: torch.device,
        *,
        expected_gated_type: type[nn.Module],
        expected_bridge_type: type[nn.Module],
    ) -> Self:
        return cls(
            checkpoint_sha256,
            gated,
            bridge,
            device,
            expected_gated_type,
            expected_bridge_type,
            _core_module_digest(gated, bridge),
            _tensor_guards(gated, bridge),
        ).validate()


def load_frozen_trainable_checkpoint(
    path: Path,
    *,
    device: torch.device,
    module_factory: Callable[[], TrainingModules] | None = None,
    expected_module_types: tuple[type[nn.Module], type[nn.Module]] | None = None,
) -> FrozenMethodModules:
    """Strictly load the exact successful real-probe trainable checkpoint."""
    checkpoint_bytes = _read_regular_bytes(path, "frozen trainable checkpoint")
    before = hashlib.sha256(checkpoint_bytes).hexdigest()
    if before != FROZEN_TRAINABLE_CHECKPOINT_SHA256:
        raise ValueError("frozen trainable checkpoint content SHA-256 mismatch")
    try:
        raw = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("frozen trainable checkpoint could not be loaded safely") from error
    expected = {
        "format",
        "foundation_sha256",
        "module_state_sha256",
        "modules",
        "optimizer_steps",
        "protocol_sha256",
        "training_trace",
        "training_trace_sha256",
    }
    if type(raw) is not dict or set(raw) != expected:
        raise ValueError("frozen trainable checkpoint envelope is malformed")
    payload = cast(dict[str, Any], raw)
    if payload["format"] != "duet-x-ltx-probe-trainable-v1":
        raise ValueError("frozen trainable checkpoint format changed")
    for name in (
        "foundation_sha256",
        "module_state_sha256",
        "protocol_sha256",
        "training_trace_sha256",
    ):
        _sha256(payload[name], name)
    if type(payload["optimizer_steps"]) is not int or payload["optimizer_steps"] <= 0:
        raise ValueError("frozen trainable checkpoint optimizer step count is malformed")
    states = payload["modules"]
    if type(states) is not dict or set(states) != {"bridge", "gated", "resampler", "scorer"}:
        raise ValueError("frozen trainable checkpoint module roster changed")
    if module_factory is None:
        if expected_module_types is not None:
            raise ValueError("production module loader does not accept substituted module types")
        modules = build_training_modules().validate()
        type_names = (
            f"{type(modules.gated).__module__}.{type(modules.gated).__qualname__}",
            f"{type(modules.bridge).__module__}.{type(modules.bridge).__qualname__}",
        )
        if type_names != (_CANONICAL_GATED_TYPE, _CANONICAL_BRIDGE_TYPE):
            raise ValueError("production frozen core module architecture changed")
        expected_types = (type(modules.gated), type(modules.bridge))
    else:
        if expected_module_types is None:
            raise ValueError("an injected test module factory requires explicit expected types")
        modules = module_factory().validate()
        expected_types = expected_module_types
    for name, module in modules.named().items():
        state = cast(dict[str, object], states).get(name)
        if type(state) is not dict:
            raise ValueError(f"frozen trainable checkpoint module {name} is malformed")
        try:
            module.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                f"frozen trainable checkpoint module {name} is incompatible"
            ) from error
    if _module_state_digest(modules) != payload["module_state_sha256"]:
        raise ValueError("frozen trainable checkpoint tensor digest changed")
    for module in modules.named().values():
        module.to(device)
        module.eval()
        module.requires_grad_(False)
    return FrozenMethodModules.capture(
        before,
        modules.gated,
        modules.bridge,
        device,
        expected_gated_type=expected_types[0],
        expected_bridge_type=expected_types[1],
    )


@dataclass(frozen=True, slots=True)
class BuiltMethodGuides:
    """Live guide tensors paired with their immutable protocol receipt."""

    guides: tuple[torch.Tensor, ...]
    receipt: MethodGuideReceipt
    core_modules_sha256: str

    def validate(self) -> Self:
        self.receipt.validate()
        _sha256(self.core_modules_sha256, "core_modules_sha256")
        expected = 9 if self.receipt.key.method is Method.FULL_HISTORY else 4
        if type(self.guides) is not tuple or len(self.guides) != expected:
            raise ValueError("method guide tensor count changed")
        for guide in self.guides:
            _quality_guide(guide, "method guide")
        return self


def _quality_guide(value: object, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or tuple(value.shape) != _GUIDE_SHAPE
        or value.device.type != "cpu"
        or value.dtype is not torch.float32
        or value.requires_grad
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(f"{name} must be finite contiguous CPU float32 with shape {_GUIDE_SHAPE}")
    return value


def _evidence(scene: SceneVariant, slot: int) -> EvidenceProvenance:
    digest = scene.vae_latent_sha256[slot]
    sparse = DuetXContract.default(decision1_fingerprint="0" * 64).sparse.fingerprint()
    return EvidenceProvenance(
        LeafKey(f"{scene.scene_id}-slot-{slot}", slot, slot),
        TimeFrameRange(slot * 2 + 1, slot * 2 + 2, slot, slot + 1),
        SpatialLocation("normalized", "bbox", (0, 0, 65_536, 65_536)),
        "quality-guide",
        RawEvidencePointer(f"duet-evidence://quality/sha256/{digest}", digest, 0, 1),
        scene.fingerprint(),
        scene.preprocessed_tensor_sha256[slot],
        digest,
        canonical_sha256("official-ltx-reference-boundary"),
        canonical_sha256("frozen-quality-selector"),
        sparse,
    ).validate()


def _core_for_method(
    method: Method,
    scene: SceneVariant,
    guides: tuple[torch.Tensor, ...],
    modules: FrozenMethodModules,
) -> torch.Tensor:
    modules.validate()
    history = torch.stack(guides[:8], dim=1).to(modules.device)
    core: object
    if method is Method.GATED_CORE:
        compress = getattr(modules.gated, "compress", None)
        if not callable(compress):
            raise ValueError("frozen gated module lacks the common compress boundary")
        core_history = history[:, _CORE_INPUT_SLOTS]
        with torch.inference_mode():
            core = compress(core_history)
    elif method is Method.DUET_CORE:
        forward_exclusion = getattr(modules.bridge, "forward_exclusion", None)
        if not callable(forward_exclusion):
            raise ValueError("frozen Duet module lacks the common exclusion boundary")
        candidates = tuple(
            ExceptionItem(
                f"{scene.scene_id}-slot-{slot}",
                _evidence(scene, slot),
                history[0, slot].detach().float().mean(dim=(1, 2, 3)),
                1_000_000 if slot in (2, 5) else 0,
            )
            for slot in range(8)
        )
        with torch.inference_mode():
            output = forward_exclusion(history, candidates=candidates)
        excluded = getattr(output, "excluded_leaf_keys", ())
        if frozenset(getattr(key, "slot", None) for key in excluded) != {2, 5}:
            raise ValueError("Duet core excluded different leaves than the frozen protected slots")
        core = getattr(output, "core_latent", None)
    else:
        raise ValueError("only frozen core methods produce a learned core")
    if not isinstance(core, torch.Tensor):
        raise ValueError("frozen core method returned no tensor")
    modules.validate()
    normalized = core.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
    return _quality_guide(normalized, "method core")


def build_method_guides(
    key: GenerationCellKey,
    scene: SceneVariant,
    scene_guides: Sequence[torch.Tensor],
    scene_manifest_digest: str,
    modules: FrozenMethodModules,
) -> BuiltMethodGuides:
    """Reconstruct all four methods at one checked 12-by-12 guide boundary."""
    key.validate()
    scene.validate()
    modules.validate()
    _sha256(scene_manifest_digest, "scene manifest SHA-256")
    if key.scene_id != scene.scene_id:
        raise ValueError("method guide key does not match its scene")
    guides = tuple(scene_guides)
    if len(guides) != 9:
        raise ValueError("one scene must supply exactly nine materialized guides")
    for slot, (guide, expected) in enumerate(zip(guides, scene.vae_latent_sha256, strict=True)):
        checked = _quality_guide(guide, f"scene guide {slot}")
        if tensor_sha256(checked) != expected:
            raise ValueError(f"scene guide {slot} hash does not match its materialization manifest")
    core_input_sha256: str | None = None
    if key.method is Method.FULL_HISTORY:
        selected = guides
        sources = tuple(
            GuideSource(GuideSourceKind.SCENE_VAE_LATENT, slot, scene.vae_latent_sha256[slot])
            for slot in range(9)
        )
    elif key.method is Method.RECENT_ANCHOR:
        slots = (7, 2, 5, 8)
        selected = tuple(guides[slot] for slot in slots)
        sources = tuple(
            GuideSource(GuideSourceKind.SCENE_VAE_LATENT, slot, scene.vae_latent_sha256[slot])
            for slot in slots
        )
    else:
        core = _core_for_method(key.method, scene, guides, modules)
        selected = (core, *(guides[slot] for slot in _PROTECTED_CURRENT_SLOTS))
        core_input_sha256 = canonical_sha256(
            {
                "format": "duet-x-method-core-input-v1",
                "method": key.method.value,
                "trainable_checkpoint_sha256": FROZEN_TRAINABLE_CHECKPOINT_SHA256,
                "scene_manifest_sha256": scene_manifest_digest,
                "scene_variant_sha256": scene.fingerprint(),
                "history_vae_latent_sha256": tuple(
                    scene.vae_latent_sha256[index] for index in _CORE_INPUT_SLOTS
                ),
            }
        )
        sources = (
            GuideSource(GuideSourceKind.METHOD_CORE, None, tensor_sha256(core)),
            *(
                GuideSource(
                    GuideSourceKind.SCENE_VAE_LATENT,
                    slot,
                    scene.vae_latent_sha256[slot],
                )
                for slot in _PROTECTED_CURRENT_SLOTS
            ),
        )
    receipt = MethodGuideReceipt(
        key,
        scene_manifest_digest,
        scene.fingerprint(),
        sources,
        core_input_sha256,
    ).validate()
    return BuiltMethodGuides(selected, receipt, modules.core_modules_sha256).validate()


def _initial_noise_sha256(key: GenerationCellKey, runtime: RuntimeCoordinate) -> str:
    key.validate()
    runtime.validate()
    return canonical_sha256(
        {
            "format": "duet-x-initial-noise-identity-v1",
            "scene_id": key.scene_id,
            "seed": key.seed,
            "runtime_coordinate_sha256": runtime.fingerprint(),
        }
    )


def build_cell_keys(clusters: Sequence[CounterfactualCluster]) -> tuple[GenerationCellKey, ...]:
    """Build the exact matrix in a deterministic, rotating execution order."""
    values = validate_confirmatory_clusters(clusters)
    scene_ids = tuple(sorted(scene.scene_id for cluster in values for scene in cluster.variants))
    methods = tuple(Method)
    keys: list[GenerationCellKey] = []
    for scene_index, scene_id in enumerate(scene_ids):
        for seed_index, seed in enumerate(MATCHED_SEEDS):
            rotation = (scene_index * len(MATCHED_SEEDS) + seed_index) % len(methods)
            order = (*methods[rotation:], *methods[:rotation])
            keys.extend(GenerationCellKey(scene_id, seed, method) for method in order)
    validate_generation_cell_keys(keys, scene_ids)
    return tuple(keys)


@dataclass(frozen=True, slots=True)
class GenerationPartition(FrozenRecord):
    """One explicit host assignment over complete generation-cell keys."""

    format: str
    host_id: str
    host_runtime_identity_sha256: str
    preregistration_sha256: str
    keys: tuple[GenerationCellKey, ...]

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-partition-v1":
            raise ValueError("generation partition format changed")
        if self.host_id not in _HOST_IDS:
            raise ValueError("generation partition host identity changed")
        _sha256(self.host_runtime_identity_sha256, "host runtime identity")
        _sha256(self.preregistration_sha256, "preregistration SHA-256")
        if type(self.keys) is not tuple or not self.keys:
            raise ValueError("generation partition must contain cell keys")
        for key in self.keys:
            key.validate()
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("generation partition contains duplicate cells")
        return self


def build_partitions(
    keys: Sequence[GenerationCellKey], prerequisites: GenerationPrerequisites
) -> tuple[GenerationPartition, GenerationPartition]:
    """Partition the complete matrix between Forge1 and Forge2 without inference from paths."""
    prerequisites.validate()
    values = tuple(keys)
    scene_ids = tuple(sorted({key.scene_id for key in values}))
    validate_generation_cell_keys(values, scene_ids)
    result = tuple(
        GenerationPartition(
            "duet-x-ltx-quality-partition-v1",
            host.host_id,
            host.runtime_identity_sha256,
            prerequisites.fingerprint(),
            values[index::2],
        ).validate()
        for index, host in enumerate(prerequisites.hosts)
    )
    combined = {*result[0].keys, *result[1].keys}
    if set(result[0].keys) & set(result[1].keys) or combined != set(values):
        raise RuntimeError("two-host partition is not disjoint and complete")
    return cast(tuple[GenerationPartition, GenerationPartition], result)


@dataclass(frozen=True, slots=True)
class CellAbsentReceipt(FrozenRecord):
    format: str
    lifecycle: GenerationCellLifecycle
    preregistration_sha256: str
    host_identity_sha256: str
    lifecycle_authorization_sha256: str

    def validate(self) -> Self:
        if self.format != _FORMAT:
            raise ValueError("cell receipt format changed")
        self.lifecycle.validate()
        if self.lifecycle.current is not CellState.ABSENT:
            raise ValueError("absent receipt requires absent lifecycle")
        _sha256(self.preregistration_sha256, "preregistration_sha256")
        _sha256(self.host_identity_sha256, "host_identity_sha256")
        _sha256(self.lifecycle_authorization_sha256, "lifecycle_authorization_sha256")
        return self


@dataclass(frozen=True, slots=True)
class CellStartedReceipt(FrozenRecord):
    format: str
    lifecycle: GenerationCellLifecycle
    preregistration_sha256: str
    host_identity_sha256: str
    lifecycle_authorization_sha256: str
    core_modules_sha256: str

    def validate(self) -> Self:
        if self.format != _FORMAT:
            raise ValueError("cell receipt format changed")
        self.lifecycle.validate()
        if self.lifecycle.current is not CellState.STARTED:
            raise ValueError("started receipt requires started lifecycle")
        _sha256(self.preregistration_sha256, "preregistration_sha256")
        _sha256(self.host_identity_sha256, "host_identity_sha256")
        _sha256(self.lifecycle_authorization_sha256, "lifecycle_authorization_sha256")
        _sha256(self.core_modules_sha256, "core_modules_sha256")
        return self


@dataclass(frozen=True, slots=True)
class CellGuideBindingReceipt(FrozenRecord):
    """Append-only guide binding added only after a successful core construction."""

    format: str
    lifecycle: GenerationCellLifecycle
    preregistration_sha256: str
    host_identity_sha256: str
    lifecycle_authorization_sha256: str
    core_modules_sha256: str
    guide_receipt: MethodGuideReceipt

    def validate(self) -> Self:
        if self.format != _FORMAT:
            raise ValueError("cell receipt format changed")
        self.lifecycle.validate()
        if self.lifecycle.current is not CellState.STARTED:
            raise ValueError("guide binding requires started lifecycle")
        for name in (
            "preregistration_sha256",
            "host_identity_sha256",
            "lifecycle_authorization_sha256",
            "core_modules_sha256",
        ):
            _sha256(getattr(self, name), name)
        self.guide_receipt.validate()
        if self.guide_receipt.key != self.lifecycle.key:
            raise ValueError("guide binding key changed")
        return self


@dataclass(frozen=True, slots=True)
class ArtifactInstallReceipt(FrozenRecord):
    """All independently verifiable facts atomically installed with one media artifact."""

    format: str
    lifecycle: GenerationCellLifecycle
    preregistration_sha256: str
    host_identity_sha256: str
    lifecycle_authorization_sha256: str
    guide_receipt: MethodGuideReceipt
    core_modules_sha256: str
    initial_noise_sha256: str
    final_latent_sha256: str
    decoded_frames_sha256: str
    media_sha256: str
    duration_seconds: float
    peak_memory_bytes: int
    wall_time_seconds: float
    model_foundations_before_sha256: str
    model_foundations_after_sha256: str
    runtime_coordinate_sha256: str
    runtime_identity_sha256: str
    host_id: str
    protocol_sha256: str

    def validate(self) -> Self:
        if self.format != _FORMAT:
            raise ValueError("artifact receipt format changed")
        self.lifecycle.validate()
        if self.lifecycle.current is not CellState.ARTIFACT_INSTALLED:
            raise ValueError("artifact receipt requires artifact-installed lifecycle")
        self.guide_receipt.validate()
        if self.guide_receipt.key != self.lifecycle.key:
            raise ValueError("artifact guide receipt key changed")
        for name in (
            "preregistration_sha256",
            "host_identity_sha256",
            "lifecycle_authorization_sha256",
            "core_modules_sha256",
            "initial_noise_sha256",
            "final_latent_sha256",
            "decoded_frames_sha256",
            "media_sha256",
            "model_foundations_before_sha256",
            "model_foundations_after_sha256",
            "runtime_coordinate_sha256",
            "runtime_identity_sha256",
            "protocol_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.model_foundations_before_sha256 != self.model_foundations_after_sha256:
            raise ValueError("model foundation mutation invalidates installed evidence")
        for name in ("duration_seconds", "wall_time_seconds"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive float")
        if type(self.peak_memory_bytes) is not int or self.peak_memory_bytes < 0:
            raise ValueError("peak memory must be a nonnegative integer")
        if self.host_id not in _HOST_IDS:
            raise ValueError("artifact host identity changed")
        return self

    def committed(self) -> CommittedGenerationCell:
        key = self.lifecycle.key
        return CommittedGenerationCell(
            key,
            self.lifecycle.transition(CellState.COMMITTED),
            self.guide_receipt,
            self.initial_noise_sha256,
            self.final_latent_sha256,
            self.decoded_frames_sha256,
            self.media_sha256,
            self.duration_seconds,
            self.peak_memory_bytes,
            self.wall_time_seconds,
            self.model_foundations_before_sha256,
            self.model_foundations_after_sha256,
            self.runtime_coordinate_sha256,
            self.runtime_identity_sha256,
            self.host_id,
            self.lifecycle_authorization_sha256,
            self.protocol_sha256,
        ).validate()


def _commit_bytes(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class MediaEngine(Protocol):
    def generate(
        self,
        *,
        guides: tuple[torch.Tensor, ...],
        prompt: str,
        negative_prompt: str,
        seed: int,
        output_path: Path,
    ) -> OneStageMediaGenerationResult: ...


@dataclass(slots=True)
class CellStore:
    """One host's append-only, content-addressed confirmatory cell store."""

    root: Path
    prerequisites: GenerationPrerequisites
    host: HostRuntimeIdentity
    lifecycle_authorization_sha256: str

    def validate(self) -> Self:
        if (
            self.root.is_symlink()
            or not self.root.is_dir()
            or not self.root.is_absolute()
            or self.root.resolve(strict=True) != self.root
            or self.root.stat(follow_symlinks=False).st_mode & 0o777 != 0o700
        ):
            raise ValueError(
                "cell store root must be an existing absolute mode 0700 non-symlink directory"
            )
        self.prerequisites.validate()
        self.host.validate()
        _sha256(
            self.lifecycle_authorization_sha256,
            "cell store lifecycle authorization SHA-256",
        )
        expected = {item.host_id: item for item in self.prerequisites.hosts}.get(self.host.host_id)
        if expected != self.host:
            raise ValueError("cell store host does not match preregistration")
        return self

    def cell_root(self, key: GenerationCellKey) -> Path:
        key.validate()
        return self.root / key.fingerprint()

    def assigned_partition(self) -> GenerationPartition:
        partitions = build_partitions(
            build_cell_keys(self.prerequisites.clusters), self.prerequisites
        )
        return next(partition for partition in partitions if partition.host_id == self.host.host_id)

    def authorize_key(self, key: GenerationCellKey) -> None:
        key.validate()
        if key not in frozenset(self.assigned_partition().keys):
            raise ValueError("generation key is not assigned to this host")

    def _paths(self, key: GenerationCellKey) -> tuple[Path, Path, Path, Path, Path]:
        root = self.cell_root(key)
        return (
            root / "00-absent.json",
            root / "01-started.json",
            root / "artifact",
            root / "artifact" / "02-artifact-installed.json",
            root / "03-committed.json",
        )

    def _guide_binding_path(self, key: GenerationCellKey) -> Path:
        return self.cell_root(key) / "01-guide-bound.json"

    def _load_started(self, key: GenerationCellKey) -> tuple[CellAbsentReceipt, CellStartedReceipt]:
        absent_path, started_path, *_ = self._paths(key)
        absent = cast(
            CellAbsentReceipt,
            _from_json(
                _read_regular_bytes(
                    absent_path, "absent receipt", required_mode=0o600, max_bytes=1024 * 1024
                ),
                CellAbsentReceipt,
            ),
        )
        started = cast(
            CellStartedReceipt,
            _from_json(
                _read_regular_bytes(
                    started_path,
                    "started receipt",
                    required_mode=0o600,
                    max_bytes=1024 * 1024,
                ),
                CellStartedReceipt,
            ),
        )
        if (
            absent.lifecycle.key != key
            or started.lifecycle != absent.lifecycle.transition(CellState.STARTED)
            or absent.preregistration_sha256 != self.prerequisites.fingerprint()
            or started.preregistration_sha256 != absent.preregistration_sha256
            or absent.host_identity_sha256 != self.host.fingerprint()
            or started.host_identity_sha256 != absent.host_identity_sha256
            or absent.lifecycle_authorization_sha256 != self.lifecycle_authorization_sha256
            or started.lifecycle_authorization_sha256 != absent.lifecycle_authorization_sha256
        ):
            raise ValueError("started cell lifecycle, key, or parent chain is damaged")
        return absent, started

    def _load_guide_binding(
        self, key: GenerationCellKey, started: CellStartedReceipt
    ) -> CellGuideBindingReceipt:
        binding = cast(
            CellGuideBindingReceipt,
            _from_json(
                _read_regular_bytes(
                    self._guide_binding_path(key),
                    "guide binding receipt",
                    required_mode=0o600,
                    max_bytes=1024 * 1024,
                ),
                CellGuideBindingReceipt,
            ),
        )
        if (
            binding.lifecycle != started.lifecycle
            or binding.preregistration_sha256 != started.preregistration_sha256
            or binding.host_identity_sha256 != started.host_identity_sha256
            or binding.lifecycle_authorization_sha256 != started.lifecycle_authorization_sha256
            or binding.core_modules_sha256 != started.core_modules_sha256
        ):
            raise ValueError("persisted guide binding conflicts with started receipt")
        return binding

    def begin(self, key: GenerationCellKey, core_modules_sha256: str) -> CellStartedReceipt:
        self.authorize_key(key)
        _sha256(core_modules_sha256, "core_modules_sha256")
        root = self.cell_root(key)
        root.mkdir(mode=0o700, exist_ok=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or root.stat(follow_symlinks=False).st_mode & 0o777 != 0o700
        ):
            raise ValueError("cell directory must be a real directory")
        absent_path, started_path, _, _, _ = self._paths(key)
        absent = CellAbsentReceipt(
            _FORMAT,
            GenerationCellLifecycle(key, (CellState.ABSENT,)),
            self.prerequisites.fingerprint(),
            self.host.fingerprint(),
            self.lifecycle_authorization_sha256,
        ).validate()
        if absent_path.exists() or absent_path.is_symlink():
            existing_absent = cast(
                CellAbsentReceipt,
                _from_json(
                    _read_regular_bytes(
                        absent_path,
                        "absent receipt",
                        required_mode=0o600,
                        max_bytes=1024 * 1024,
                    ),
                    CellAbsentReceipt,
                ),
            )
            if existing_absent != absent:
                raise ValueError("existing absent receipt conflicts with this cell")
        else:
            _commit_bytes(absent_path, absent.to_json())
        started = CellStartedReceipt(
            _FORMAT,
            absent.lifecycle.transition(CellState.STARTED),
            absent.preregistration_sha256,
            absent.host_identity_sha256,
            absent.lifecycle_authorization_sha256,
            core_modules_sha256,
        ).validate()
        _commit_bytes(started_path, started.to_json())
        return started

    def bind_guides(
        self, started: CellStartedReceipt, guides: BuiltMethodGuides
    ) -> CellGuideBindingReceipt:
        """Append the exact successful guide receipt without weakening pre-core started evidence."""
        self.validate()
        started.validate()
        guides.validate()
        key = started.lifecycle.key
        self.authorize_key(key)
        if self.status(key) is not CellState.STARTED:
            raise ValueError("guide binding requires a cell that is exactly started")
        _, persisted = self._load_started(key)
        if started != persisted:
            raise ValueError("supplied started receipt differs from persisted started receipt")
        if guides.receipt.key != key or guides.core_modules_sha256 != persisted.core_modules_sha256:
            raise ValueError("guide binding key or core module parent changed")
        binding = CellGuideBindingReceipt(
            _FORMAT,
            persisted.lifecycle,
            persisted.preregistration_sha256,
            persisted.host_identity_sha256,
            persisted.lifecycle_authorization_sha256,
            persisted.core_modules_sha256,
            guides.receipt,
        ).validate()
        path = self._guide_binding_path(key)
        if path.exists() or path.is_symlink():
            existing = self._load_guide_binding(key, persisted)
            if existing != binding:
                raise ValueError("persisted guide binding conflicts with supplied guides")
            return existing
        _commit_bytes(path, binding.to_json())
        return binding

    def _load_installed(self, key: GenerationCellKey) -> ArtifactInstallReceipt:
        _, _, artifact, receipt_path, _ = self._paths(key)
        if artifact.is_symlink() or not artifact.is_dir():
            raise ValueError("installed artifact directory is missing or unsafe")
        if {item.name for item in artifact.iterdir()} != {
            "02-artifact-installed.json",
            "media.mp4",
        }:
            raise ValueError("installed artifact contains missing or extra evidence")
        absent, started = self._load_started(key)
        binding = self._load_guide_binding(key, started)
        receipt = cast(
            ArtifactInstallReceipt,
            _from_json(
                _read_regular_bytes(
                    receipt_path,
                    "installed receipt",
                    required_mode=0o600,
                    max_bytes=1024 * 1024,
                ),
                ArtifactInstallReceipt,
            ),
        )
        if receipt.lifecycle.key != key:
            raise ValueError("installed artifact key changed")
        if (
            started.lifecycle != absent.lifecycle.transition(CellState.STARTED)
            or receipt.lifecycle != started.lifecycle.transition(CellState.ARTIFACT_INSTALLED)
            or binding.guide_receipt != receipt.guide_receipt
            or started.core_modules_sha256 != receipt.core_modules_sha256
            or absent.preregistration_sha256 != started.preregistration_sha256
            or started.preregistration_sha256 != receipt.preregistration_sha256
            or absent.host_identity_sha256 != started.host_identity_sha256
            or started.host_identity_sha256 != receipt.host_identity_sha256
            or absent.lifecycle_authorization_sha256 != started.lifecycle_authorization_sha256
            or started.lifecycle_authorization_sha256 != receipt.lifecycle_authorization_sha256
        ):
            raise ValueError("installed artifact lifecycle or parent chain is damaged")
        media = artifact / "media.mp4"
        if _file_sha256(media) != receipt.media_sha256:
            raise ValueError("installed media SHA-256 mismatch")
        if (
            receipt.preregistration_sha256 != self.prerequisites.fingerprint()
            or receipt.host_identity_sha256 != self.host.fingerprint()
            or receipt.lifecycle_authorization_sha256 != self.lifecycle_authorization_sha256
            or receipt.host_id != self.host.host_id
            or receipt.runtime_identity_sha256 != self.host.runtime_identity_sha256
            or receipt.runtime_coordinate_sha256 != self.prerequisites.runtime.fingerprint()
            or receipt.model_foundations_before_sha256 != self.host.model_foundation_sha256
            or receipt.model_foundations_after_sha256 != self.host.model_foundation_sha256
            or receipt.protocol_sha256 != self.prerequisites.protocol_sha256
            or receipt.initial_noise_sha256
            != _initial_noise_sha256(key, self.prerequisites.runtime)
        ):
            raise ValueError("installed artifact parent or host binding changed")
        return receipt

    def _load_committed(self, key: GenerationCellKey) -> CommittedGenerationCell:
        if {item.name for item in self.cell_root(key).iterdir()} != {
            "00-absent.json",
            "01-started.json",
            "01-guide-bound.json",
            "artifact",
            "03-committed.json",
        }:
            raise ValueError("committed cell contains missing or extra evidence")
        *_, committed_path = self._paths(key)
        committed = cast(
            CommittedGenerationCell,
            _from_json(
                _read_regular_bytes(
                    committed_path,
                    "committed receipt",
                    required_mode=0o600,
                    max_bytes=1024 * 1024,
                ),
                CommittedGenerationCell,
            ),
        )
        installed = self._load_installed(key)
        if committed != installed.committed():
            raise ValueError("committed receipt conflicts with installed artifact")
        return committed

    def status(self, key: GenerationCellKey) -> CellState | None:
        absent, started, artifact, receipt, committed = self._paths(key)
        cell_root = self.cell_root(key)
        if cell_root.exists():
            if cell_root.is_symlink() or not cell_root.is_dir():
                raise ValueError("cell path is not a safe directory")
            names = {item.name for item in cell_root.iterdir()}
        else:
            names = set()
        if committed.exists():
            self._load_committed(key)
            return CellState.COMMITTED
        if artifact.exists() or receipt.exists():
            if names != {
                "00-absent.json",
                "01-started.json",
                "01-guide-bound.json",
                "artifact",
            }:
                raise ValueError("installed cell contains missing or extra lifecycle evidence")
            self._load_installed(key)
            return CellState.ARTIFACT_INSTALLED
        if started.exists():
            allowed = {"00-absent.json", "01-started.json"}
            if names not in (allowed, {*allowed, "01-guide-bound.json"}):
                raise ValueError("started cell contains missing or extra lifecycle evidence")
            _, persisted = self._load_started(key)
            if "01-guide-bound.json" in names:
                self._load_guide_binding(key, persisted)
            return CellState.STARTED
        if absent.exists():
            if names != {"00-absent.json"}:
                raise ValueError("absent cell contains extra lifecycle evidence")
            _from_json(
                _read_regular_bytes(
                    absent, "absent receipt", required_mode=0o600, max_bytes=1024 * 1024
                ),
                CellAbsentReceipt,
            )
            return CellState.ABSENT
        if names:
            raise ValueError("cell directory contains unrecognized lifecycle evidence")
        return None

    def promote(self, key: GenerationCellKey) -> CommittedGenerationCell:
        installed = self._load_installed(key)
        committed = installed.committed()
        committed_path = self._paths(key)[-1]
        if committed_path.exists():
            return self._load_committed(key)
        _commit_bytes(committed_path, committed.to_json())
        return committed

    def install(
        self,
        key: GenerationCellKey,
        started: CellStartedReceipt,
        engine: MediaEngine,
        guides: BuiltMethodGuides,
        scene: SceneVariant,
        scene_guides: Sequence[torch.Tensor],
        modules: FrozenMethodModules,
        decoded_frames_sha256: Callable[[Path], str],
        peak_memory_bytes: Callable[[], int],
    ) -> ArtifactInstallReceipt:
        self.validate()
        self.authorize_key(key)
        if self.status(key) is not CellState.STARTED:
            raise ValueError("artifact installation requires a cell that is exactly started")
        _, persisted_started = self._load_started(key)
        if started.validate() != persisted_started or started.lifecycle.key != key:
            raise ValueError("supplied started receipt or key differs from persisted evidence")
        binding = self._load_guide_binding(key, persisted_started)
        guides.validate()
        if (
            started.core_modules_sha256 != guides.core_modules_sha256
            or binding.guide_receipt != guides.receipt
        ):
            raise ValueError("persisted guide or core-module binding changed")
        for index, (guide, source) in enumerate(
            zip(guides.guides, binding.guide_receipt.sources, strict=True)
        ):
            if tensor_sha256(_quality_guide(guide, f"live guide tensor {index}")) != source.sha256:
                raise ValueError(f"live guide tensor {index} changed after its receipt")
        authoritative_scene = next(
            (
                candidate
                for cluster in self.prerequisites.clusters
                for candidate in cluster.variants
                if candidate.scene_id == key.scene_id
            ),
            None,
        )
        if (
            authoritative_scene is None
            or scene != authoritative_scene
            or scene.scene_id != key.scene_id
        ):
            raise ValueError("generation scene is not the authoritative preregistered scene")
        if (
            binding.guide_receipt.scene_manifest_sha256 != self.prerequisites.scene_manifest_sha256
            or binding.guide_receipt.scene_variant_sha256 != authoritative_scene.fingerprint()
        ):
            raise ValueError("persisted guide receipt scene parent changed")
        authoritative_guides = build_method_guides(
            key,
            authoritative_scene,
            scene_guides,
            self.prerequisites.scene_manifest_sha256,
            modules,
        )
        if (
            authoritative_guides.receipt != binding.guide_receipt
            or authoritative_guides.core_modules_sha256 != started.core_modules_sha256
            or len(authoritative_guides.guides) != len(guides.guides)
            or any(
                not torch.equal(authoritative, supplied)
                for authoritative, supplied in zip(
                    authoritative_guides.guides, guides.guides, strict=True
                )
            )
        ):
            raise ValueError("supplied guides differ from authoritative method guides")
        root = self.cell_root(key)
        temporary = Path(tempfile.mkdtemp(prefix=".artifact.tmp-", dir=root))
        try:
            media = temporary / "media.mp4"
            start = time.perf_counter()
            result = engine.generate(
                guides=guides.guides,
                prompt=scene.prompt,
                negative_prompt=scene.negative_prompt,
                seed=key.seed,
                output_path=media,
            ).validate()
            media.chmod(0o600, follow_symlinks=False)
            media_metadata = media.stat(follow_symlinks=False)
            if media_metadata.st_nlink != 1 or media_metadata.st_mode & 0o777 != 0o600:
                raise ValueError("generated media must be a single-link mode 0600 file")
            elapsed = max(time.perf_counter() - start, float.fromhex("0x1p-52"))
            if result.checkpoint_file_sha256 != self.prerequisites.runtime.model_checkpoint_sha256:
                raise ValueError("generated media checkpoint binding changed")
            if (
                result.offload_mode != PINNED_LTX_OFFLOAD_MODE
                or result.foundation_guard_sha256 != STREAMING_FOUNDATION_GUARD_SHA256
            ):
                raise ValueError("generated media generation engine identity changed")
            if (
                result.foundation_before_sha256 != self.host.model_foundation_sha256
                or result.foundation_after_sha256 != self.host.model_foundation_sha256
            ):
                raise ValueError("generated media model foundation binding changed")
            frames_digest = decoded_frames_sha256(media)
            _sha256(frames_digest, "decoded frame SHA-256")
            receipt = ArtifactInstallReceipt(
                _FORMAT,
                started.lifecycle.transition(CellState.ARTIFACT_INSTALLED),
                started.preregistration_sha256,
                started.host_identity_sha256,
                started.lifecycle_authorization_sha256,
                guides.receipt,
                started.core_modules_sha256,
                _initial_noise_sha256(key, self.prerequisites.runtime),
                result.final_latent_sha256,
                frames_digest,
                result.media_sha256,
                result.duration_seconds,
                peak_memory_bytes(),
                elapsed,
                result.foundation_before_sha256,
                result.foundation_after_sha256,
                self.prerequisites.runtime.fingerprint(),
                self.host.runtime_identity_sha256,
                self.host.host_id,
                self.prerequisites.protocol_sha256,
            ).validate()
            _commit_bytes(temporary / "02-artifact-installed.json", receipt.to_json())
            if _file_sha256(media) != receipt.media_sha256:
                raise ValueError("media changed before atomic installation")
            directory = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            artifact = self._paths(key)[2]
            if artifact.exists():
                raise FileExistsError("artifact was installed concurrently")
            temporary.rename(artifact)
            parent = os.open(root, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
            return receipt
        finally:
            if temporary.exists():
                for child in temporary.iterdir():
                    child.unlink()
                temporary.rmdir()

    def audit_assigned_committed(
        self, partition: GenerationPartition
    ) -> tuple[CommittedGenerationCell, ...]:
        """Require the store to contain exactly its 216 complete assigned directories."""
        self.validate()
        if partition != self.assigned_partition():
            raise ValueError("generation store partition disagrees with preregistration")
        expected = {key.fingerprint(): key for key in partition.keys}
        entries = tuple(self.root.iterdir())
        if len(entries) != len(expected) or {entry.name for entry in entries} != set(expected):
            raise ValueError("generation store has missing or extra assigned cells")
        cells: list[CommittedGenerationCell] = []
        for entry in entries:
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError("generation store contains an extra unsafe entry")
            key = expected[entry.name]
            cells.append(self._load_committed(key))
        return tuple(cells)

    def committed_keys(self) -> tuple[GenerationCellKey, ...]:
        return tuple(cell.key for cell in self.audit_assigned_committed(self.assigned_partition()))


def generate_cell_once(
    *,
    store: CellStore,
    key: GenerationCellKey,
    scene: SceneVariant,
    scene_guides: Sequence[torch.Tensor],
    modules: FrozenMethodModules,
    engine: MediaEngine,
    decoded_frames_sha256: Callable[[Path], str],
    peak_memory_bytes: Callable[[], int],
) -> CommittedGenerationCell:
    """Run one absent cell, or perform only the permitted disk recovery."""
    store.validate()
    key.validate()
    store.authorize_key(key)
    scene_by_id = {
        variant.scene_id: variant
        for cluster in store.prerequisites.clusters
        for variant in cluster.variants
    }
    if scene_by_id.get(key.scene_id) != scene:
        raise ValueError("generation scene does not match the preregistered scene manifest")
    try:
        state = store.status(key)
    except (OSError, TypeError, ValueError) as error:
        if store.cell_root(key).exists():
            raise RuntimeError(
                "terminal evidence failure: persisted cell evidence is invalid"
            ) from error
        raise
    if state is CellState.COMMITTED:
        return store._load_committed(key)
    if state is CellState.ARTIFACT_INSTALLED:
        try:
            return store.promote(key)
        except ValueError as error:
            raise RuntimeError(
                "terminal evidence failure: installed artifact is invalid"
            ) from error
    if state is CellState.STARTED:
        raise RuntimeError("terminal evidence failure: started without valid installed artifact")
    modules.validate()
    started = store.begin(key, modules.core_modules_sha256)
    built = build_method_guides(
        key,
        scene,
        scene_guides,
        store.prerequisites.scene_manifest_sha256,
        modules,
    )
    store.bind_guides(started, built)
    store.install(
        key,
        started,
        engine,
        built,
        scene,
        scene_guides,
        modules,
        decoded_frames_sha256,
        peak_memory_bytes,
    )
    return store.promote(key)


@dataclass(frozen=True, slots=True)
class GenerationAudit(FrozenRecord):
    """Complete, canonical disk audit prior to blind-package construction."""

    format: str
    preregistration_sha256: str
    partition_sha256: tuple[str, str]
    confirmatory_forge_receipt_sha256: tuple[str, str]
    cells: tuple[CommittedGenerationCell, ...]

    def validate(self) -> Self:
        if self.format != "duet-x-ltx-quality-generation-audit-v1":
            raise ValueError("generation audit format changed")
        _sha256(self.preregistration_sha256, "preregistration_sha256")
        if len(self.partition_sha256) != 2:
            raise ValueError("generation audit requires two partition hashes")
        for digest in self.partition_sha256:
            _sha256(digest, "partition SHA-256")
        if (
            len(self.confirmatory_forge_receipt_sha256) != 2
            or len(set(self.confirmatory_forge_receipt_sha256)) != 2
        ):
            raise ValueError("generation audit requires two distinct Forge lifecycle receipts")
        for digest in self.confirmatory_forge_receipt_sha256:
            _sha256(digest, "confirmatory Forge receipt SHA-256")
        if len(self.cells) != GENERATION_CELL_COUNT:
            raise ValueError("generation audit requires exactly 432 committed cells")
        for cell in self.cells:
            cell.validate()
        if len({cell.key for cell in self.cells}) != GENERATION_CELL_COUNT:
            raise ValueError("generation audit contains duplicate or conflicting cells")
        return self


def load_generation_audit(path: Path) -> GenerationAudit:
    """Strict-load one canonical complete-generation audit from disk."""
    encoded = _read_regular_bytes(
        path,
        "generation audit",
        required_mode=0o600,
        max_bytes=256 * 1024 * 1024,
    )
    audit = cast(GenerationAudit, _from_json(encoded, GenerationAudit)).validate()
    if canonical_json(audit) != encoded:
        raise ValueError("generation audit must be canonical")
    return audit


def aggregate_generation(
    stores: Sequence[CellStore],
    partitions: tuple[GenerationPartition, GenerationPartition],
    clusters: Sequence[CounterfactualCluster],
    confirmatory_forge_receipt_sha256: tuple[str, str],
) -> GenerationAudit:
    """Audit both stores and reject any incomplete, damaged, foreign, or conflicting cell."""
    cluster_values = validate_confirmatory_clusters(clusters)
    expected_keys = build_cell_keys(cluster_values)
    if len(stores) != 2:
        raise ValueError("generation audit requires exactly two host stores")
    store_values = tuple(stores)
    for store in store_values:
        store.validate()
    preregistration = store_values[0].prerequisites
    if any(store.prerequisites != preregistration for store in store_values[1:]):
        raise ValueError("generation stores disagree on preregistration parent")
    if preregistration.scene_manifest_sha256 != scene_manifest_sha256(cluster_values):
        raise ValueError("generation preregistration scene parent drift")
    expected_partitions = build_partitions(expected_keys, preregistration)
    if partitions != expected_partitions:
        raise ValueError("generation partition host assignment or parent binding changed")
    if tuple(store.host.host_id for store in store_values) != _HOST_IDS:
        raise ValueError("generation stores are in the wrong host order")
    cells: list[CommittedGenerationCell] = []
    for store, partition in zip(store_values, partitions, strict=True):
        partition_cells = store.audit_assigned_committed(partition)
        for cell in partition_cells:
            cell.validate()
            if cell.host_id != partition.host_id:
                raise ValueError("committed cell has a cross-host identity")
            cells.append(cell)
    method_order = {method: index for index, method in enumerate(Method)}
    ordered = tuple(
        sorted(
            cells,
            key=lambda cell: (
                cell.key.scene_id,
                MATCHED_SEEDS.index(cell.key.seed),
                method_order[cell.key.method],
            ),
        )
    )
    validate_generation_cell_keys(
        tuple(cell.key for cell in ordered), tuple(sorted({cell.key.scene_id for cell in ordered}))
    )
    manifest_digest = committed_generation_manifest_sha256(
        ordered, cluster_values, preregistration.runtime
    )
    CommittedGenerationMatrix(
        tuple(sorted({cell.key.scene_id for cell in ordered})),
        cluster_values,
        preregistration.scene_manifest_sha256,
        preregistration.runtime,
        confirmatory_forge_receipt_sha256,
        ordered,
        manifest_digest,
        manifest_digest,
    ).validate()
    return GenerationAudit(
        "duet-x-ltx-quality-generation-audit-v1",
        preregistration.fingerprint(),
        (partitions[0].fingerprint(), partitions[1].fingerprint()),
        confirmatory_forge_receipt_sha256,
        ordered,
    ).validate()


@dataclass(frozen=True, slots=True)
class PublicGenerationStatus(FrozenRecord):
    """The only pre-reveal CLI output: counts, integrity, and apparatus errors."""

    command: str
    phase: ReviewPhase
    status: GenerationStatusCode
    completed_cells: int
    total_cells: int
    apparatus_errors: tuple[ApparatusErrorCode, ...]

    def validate(self) -> Self:
        if self.command not in {
            "author-sources",
            "author-preregistration-input",
            "materialize",
            "pilot",
            "preregister",
            "preflight-partition",
            "generate-partition",
            "audit-generation",
            "finalize",
        }:
            raise ValueError("public status command changed")
        if not isinstance(self.phase, ReviewPhase):
            raise ValueError("public status phase must be frozen")
        if not isinstance(self.status, GenerationStatusCode):
            raise ValueError("public status must be a frozen code")
        if (
            type(self.completed_cells) is not int
            or type(self.total_cells) is not int
            or not 0 <= self.completed_cells <= self.total_cells
            or self.total_cells != GENERATION_CELL_COUNT
        ):
            raise ValueError("public status cell counts are invalid")
        if type(self.apparatus_errors) is not tuple:
            raise ValueError("public status apparatus errors must be a tuple")
        if any(not isinstance(error, ApparatusErrorCode) for error in self.apparatus_errors):
            raise ValueError("public apparatus errors must use frozen codes")
        return self


def _atomic_idempotent(path: Path, data: bytes) -> bytes:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError("finalization output conflicts with existing bytes")
        return data
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise ValueError("finalization output conflicts with concurrent bytes") from None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return data


def finalize_generation(
    stores: Sequence[CellStore],
    partitions: tuple[GenerationPartition, GenerationPartition],
    clusters: Sequence[CounterfactualCluster],
    prerequisites: GenerationPrerequisites,
    confirmatory_forge_receipt_sha256: tuple[str, str],
    output_path: Path,
) -> bytes:
    """Finalize from disk only; a second call must reproduce exact bytes."""
    prerequisites.validate()
    if output_path.is_symlink() or not output_path.is_absolute() or not output_path.parent.is_dir():
        raise ValueError("finalization output must be below an existing absolute directory")
    audit = aggregate_generation(stores, partitions, clusters, confirmatory_forge_receipt_sha256)
    if audit.preregistration_sha256 != prerequisites.fingerprint():
        raise ValueError("generation audit does not bind the supplied preregistration")
    digest = committed_generation_manifest_sha256(
        audit.cells, tuple(clusters), prerequisites.runtime
    )
    matrix = CommittedGenerationMatrix(
        tuple(sorted({cell.key.scene_id for cell in audit.cells})),
        tuple(clusters),
        prerequisites.scene_manifest_sha256,
        prerequisites.runtime,
        confirmatory_forge_receipt_sha256,
        audit.cells,
        digest,
        digest,
    ).validate()
    encoded = matrix.to_json()
    return _atomic_idempotent(output_path, encoded)


__all__ = (
    "ApparatusErrorCode",
    "ArtifactInstallReceipt",
    "BuiltMethodGuides",
    "CellGuideBindingReceipt",
    "CellStore",
    "FrozenMethodModules",
    "GenerationAudit",
    "GenerationPartition",
    "GenerationPrerequisites",
    "GenerationStatusCode",
    "HostRuntimeIdentity",
    "PartitionTerminalReceipt",
    "PhaseInputSeal",
    "PublicGenerationStatus",
    "aggregate_generation",
    "build_cell_keys",
    "build_method_guides",
    "build_partitions",
    "finalize_generation",
    "generate_cell_once",
    "load_frozen_trainable_checkpoint",
    "load_generation_audit",
    "load_generation_prerequisites",
    "load_partition_terminal_receipt",
    "load_phase_input_seal",
)
