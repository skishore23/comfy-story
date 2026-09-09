"""Backward-compatible LTX facade for the shared latent-history bridge."""

from __future__ import annotations

from comfy_story.latent_bridge import (
    ExclusionLatentBridgeOutput,
    LatentBridgeOutput,
    LatentHistoryBridge,
)

LTXLatentBridgeOutput = LatentBridgeOutput
LTXExclusionBridgeOutput = ExclusionLatentBridgeOutput


class LTXLatentHistoryBridge(LatentHistoryBridge):
    """The historical 128-channel LTX bridge and checkpoint boundary."""

    def __init__(self, channels: int = 128, *, operator_size: int = 16) -> None:
        super().__init__(channels, operator_size=operator_size)


__all__ = (
    "LTXExclusionBridgeOutput",
    "LTXLatentBridgeOutput",
    "LTXLatentHistoryBridge",
)
