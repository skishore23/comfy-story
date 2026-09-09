"""Native-latent Duet-X bridge for MiniMax H3 video memory."""

from __future__ import annotations

from duet.duetx.latent_bridge import LatentHistoryBridge


class MiniMaxH3LatentHistoryBridge(LatentHistoryBridge):
    """A strict 24-channel MiniMax H3 latent-history bridge."""

    def __init__(self, *, operator_size: int = 16) -> None:
        super().__init__(24, operator_size=operator_size)


__all__ = ("MiniMaxH3LatentHistoryBridge",)
