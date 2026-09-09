"""Model-neutral ordered latent-history compression for Duet-X."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from duet.duetx.contracts import DuetXContract, ExceptionItem, LeafKey
from duet.duetx.exclusion_state import (
    ExclusionDuetXProductTree,
    ExclusionDuetXState,
    exclusion_leaf_state,
)
from duet.fusion import FusionSpec, MatrixSemigroupFusion


@dataclass(frozen=True, slots=True)
class LatentBridgeOutput:
    """One calibrated latent core and its auditable compression diagnostics."""

    core_latent: torch.Tensor
    core_operator: torch.Tensor
    per_reference_rmse: torch.Tensor
    uncertainty: torch.Tensor


@dataclass(frozen=True, slots=True)
class ExclusionLatentBridgeOutput:
    """A bridge result produced from the exact whole-leaf exclusion root."""

    core_latent: torch.Tensor
    core_operator: torch.Tensor
    per_reference_rmse: torch.Tensor
    uncertainty: torch.Tensor
    anchor_latent: torch.Tensor
    anchor_slot: int
    excluded_leaf_keys: tuple[LeafKey, ...]
    exception_items: tuple[ExceptionItem, ...]
    state: ExclusionDuetXState


def _flatten_history(history: torch.Tensor) -> torch.Tensor:
    """Present aligned latent positions as ``[B,S,FHW,C]`` Duet tokens."""
    return history.permute(0, 1, 3, 4, 5, 2).flatten(2, 4)


def _flatten_anchor(anchor: torch.Tensor) -> torch.Tensor:
    """Present one latent as ``[B,FHW,C]`` tokens."""
    return anchor.permute(0, 2, 3, 4, 1).flatten(1, 3)


def _restore_latent(tokens: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    batch, channels, frames, height, width = anchor.shape
    return (
        tokens.reshape(batch, frames, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()
    )


def _validate_inputs(
    history: torch.Tensor,
    anchor: torch.Tensor,
    availability: torch.Tensor | None,
    *,
    channels: int,
    device: torch.device,
) -> torch.Tensor:
    if (
        not isinstance(history, torch.Tensor)
        or history.layout != torch.strided
        or history.ndim != 6
    ):
        raise ValueError("history must be a strided [B,S,C,F,H,W] tensor")
    if not torch.is_floating_point(history):
        raise ValueError("history must use a floating dtype")
    batch, streams, history_channels, frames, height, width = history.shape
    if min(batch, streams, frames, height, width) <= 0:
        raise ValueError("history dimensions must be positive")
    if history_channels != channels:
        raise ValueError(f"history channels must equal {channels}")
    expected_anchor = (batch, channels, frames, height, width)
    if (
        not isinstance(anchor, torch.Tensor)
        or anchor.layout != torch.strided
        or anchor.ndim != 5
        or tuple(anchor.shape) != expected_anchor
    ):
        raise ValueError(f"anchor must have shape {expected_anchor}")
    if not torch.is_floating_point(anchor):
        raise ValueError("anchor must use a floating dtype")
    if anchor.dtype != history.dtype or anchor.device != history.device:
        raise ValueError("anchor must match the history dtype and device")
    if history.device != device:
        raise ValueError("history and anchor must be on the bridge device")
    if not bool(torch.isfinite(history).all().item()):
        raise ValueError("history must be finite")
    if not bool(torch.isfinite(anchor).all().item()):
        raise ValueError("anchor must be finite")
    if availability is None:
        return torch.ones((batch, streams), dtype=torch.bool, device=history.device)
    if (
        not isinstance(availability, torch.Tensor)
        or availability.dtype is not torch.bool
        or tuple(availability.shape) != (batch, streams)
    ):
        raise ValueError(f"availability must be bool[{batch},{streams}]")
    if availability.device != history.device:
        raise ValueError("availability must be on the history device")
    if not bool(availability.any(dim=1).all().item()):
        raise ValueError("availability must include at least one reference per batch row")
    return availability


def _build_output(
    core_tokens: torch.Tensor,
    core_operator: torch.Tensor,
    history: torch.Tensor,
    anchor: torch.Tensor,
    availability: torch.Tensor,
) -> LatentBridgeOutput:
    core_latent = _restore_latent(core_tokens, anchor)
    difference = history.float() - core_latent[:, None].float()
    per_reference_rmse = difference.square().mean(dim=(2, 3, 4, 5)).sqrt()
    uncertainty = per_reference_rmse.masked_fill(~availability, -torch.inf).amax(dim=1)
    return LatentBridgeOutput(
        core_latent=core_latent,
        core_operator=core_operator,
        per_reference_rmse=per_reference_rmse,
        uncertainty=uncertainty,
    )


class LatentHistoryBridge(nn.Module):
    """Compress ordered latent history into an anchor-preserving guide.

    The module owns only Duet fusion and residual-calibration parameters. Model
    foundations and codecs remain outside this checkpoint boundary.
    """

    def __init__(self, channels: int, *, operator_size: int = 16) -> None:
        super().__init__()
        self.fusion = MatrixSemigroupFusion(
            channels,
            FusionSpec("balanced"),
            operator_size=operator_size,
        )
        self.residual = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.residual.weight)

    def forward(
        self,
        history: torch.Tensor,
        *,
        anchor: torch.Tensor,
        availability: torch.Tensor | None = None,
    ) -> LatentBridgeOutput:
        """Compress ``history`` while preserving ``anchor`` at initialization."""
        fusion_dtype = self.fusion.input_norm.weight.dtype
        fusion_device = self.fusion.input_norm.weight.device
        available = _validate_inputs(
            history,
            anchor,
            availability,
            channels=self.fusion.channels,
            device=fusion_device,
        )
        with torch.autocast(device_type=history.device.type, enabled=False):
            tokens = _flatten_history(history).to(dtype=fusion_dtype)
            encoded = self.fusion.encode_stream(tokens)
            operators = tuple(
                self.fusion.mask_operator(operator, available[:, index])
                for index, operator in enumerate(encoded.unbind(dim=1))
            )
            combined = self.fusion.combine_ordered_operators(operators)
        core_latent = self.materialize_operator(combined, anchor=anchor)
        core_tokens = _flatten_anchor(core_latent)
        return _build_output(core_tokens, combined, history, anchor, available)

    def encode_operators(
        self,
        history: torch.Tensor,
        *,
        availability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode each aligned stream without exposing fusion internals to adapters."""
        if not isinstance(history, torch.Tensor) or history.ndim != 6:
            raise ValueError("history must be a strided [B,S,C,F,H,W] tensor")
        anchor = history[:, 0]
        fusion_dtype = self.fusion.input_norm.weight.dtype
        fusion_device = self.fusion.input_norm.weight.device
        available = _validate_inputs(
            history,
            anchor,
            availability,
            channels=self.fusion.channels,
            device=fusion_device,
        )
        with torch.autocast(device_type=history.device.type, enabled=False):
            encoded = self.fusion.encode_stream(_flatten_history(history).to(dtype=fusion_dtype))
            masked = tuple(
                self.fusion.mask_operator(operator, available[:, index])
                for index, operator in enumerate(encoded.unbind(dim=1))
            )
        return torch.stack(masked, dim=1)

    def materialize_operator(self, operator: torch.Tensor, *, anchor: torch.Tensor) -> torch.Tensor:
        """Decode one ordered operator through the calibrated latent boundary."""
        if (
            not isinstance(anchor, torch.Tensor)
            or anchor.layout != torch.strided
            or anchor.ndim != 5
            or anchor.shape[1] != self.fusion.channels
            or not torch.is_floating_point(anchor)
            or not bool(torch.isfinite(anchor).all().item())
        ):
            raise ValueError("anchor must be a finite floating [B,C,F,H,W] tensor")
        expected_steps = anchor.shape[2] * anchor.shape[3] * anchor.shape[4]
        if (
            not isinstance(operator, torch.Tensor)
            or operator.layout != torch.strided
            or tuple(operator.shape)
            != (
                anchor.shape[0],
                expected_steps,
                self.fusion.operator_size,
                self.fusion.operator_size,
            )
            or operator.dtype != torch.float32
            or operator.device != anchor.device
            or not bool(torch.isfinite(operator).all().item())
        ):
            raise ValueError("operator must be finite float32 and match the anchor token boundary")
        fusion_device = self.fusion.input_norm.weight.device
        if anchor.device != fusion_device:
            raise ValueError("operator and anchor must be on the bridge device")
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            decoded = self.fusion.decode_operator(operator)
        anchor_tokens = _flatten_anchor(anchor)
        residual = self.residual(decoded).to(dtype=anchor_tokens.dtype)
        return _restore_latent(cast(torch.Tensor, anchor_tokens + residual), anchor)

    def forward_exclusion(
        self,
        history: torch.Tensor,
        *,
        candidates: tuple[ExceptionItem, ...],
        contract: DuetXContract | None = None,
    ) -> ExclusionLatentBridgeOutput:
        """Decode the exact mergeable root variant excluding its whole-leaf Top-K."""
        if not isinstance(candidates, tuple) or len(candidates) != 8:
            raise ValueError("exclusion bridge requires one whole-leaf candidate per history slot")
        if history.ndim != 6 or history.shape[:3] != (1, 8, self.fusion.channels):
            raise ValueError("exclusion bridge history must have shape [1,8,C,F,H,W]")
        placeholder_anchor = history[:, 0]
        available = _validate_inputs(
            history,
            placeholder_anchor,
            None,
            channels=self.fusion.channels,
            device=self.fusion.input_norm.weight.device,
        )
        resolved_contract = (
            DuetXContract.default(decision1_fingerprint="0" * 64)
            if contract is None
            else contract.validate()
        )
        if resolved_contract.latent_channels != self.fusion.channels:
            raise ValueError("exclusion contract channels do not match the bridge")
        expected_sparse = resolved_contract.sparse.fingerprint()
        for slot, candidate in enumerate(candidates):
            candidate.validate(resolved_contract.sparse)
            if candidate.provenance.leaf.slot != slot:
                raise ValueError("exclusion bridge candidates must align with slots 0 through 7")
            if candidate.provenance.memory_contract_sha256 != expected_sparse:
                raise ValueError("exclusion bridge candidate memory contract does not match K=2")
        fusion_dtype = self.fusion.input_norm.weight.dtype
        with torch.autocast(device_type=history.device.type, enabled=False):
            tokens = _flatten_history(history).to(dtype=fusion_dtype)
            encoded = self.fusion.encode_stream(tokens)
            leaves = tuple(
                exclusion_leaf_state(
                    resolved_contract,
                    slot,
                    encoded[:, slot],
                    (candidates[slot],),
                )
                for slot in range(8)
            )
            state = ExclusionDuetXProductTree.from_states(leaves).root
        excluded_slots = frozenset(key.slot for key in state.excluded_leaf_keys)
        anchor_slot = max(slot for slot in range(8) if slot not in excluded_slots)
        anchor = history[:, anchor_slot]
        core_latent = self.materialize_operator(state.excluded_operator, anchor=anchor)
        core_tokens = _flatten_anchor(core_latent)
        base = _build_output(
            core_tokens,
            state.excluded_operator,
            history,
            anchor,
            available,
        )
        return ExclusionLatentBridgeOutput(
            core_latent=base.core_latent,
            core_operator=base.core_operator,
            per_reference_rmse=base.per_reference_rmse,
            uncertainty=base.uncertainty,
            anchor_latent=anchor,
            anchor_slot=anchor_slot,
            excluded_leaf_keys=state.excluded_leaf_keys,
            exception_items=state.exceptions,
            state=state,
        )


__all__ = (
    "ExclusionLatentBridgeOutput",
    "LatentBridgeOutput",
    "LatentHistoryBridge",
)
