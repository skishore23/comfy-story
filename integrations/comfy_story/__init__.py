"""ComfyUI V3 registration for the Comfy Story product node."""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from .h3_quality_nodes import ComfyH3RefinementSigmas, ComfyH3ReplaceVideoLatent, ComfyH3VideoLatent
from .nodes import (
    ComfyH3BalancedCache,
    ComfyH3UltraFast,
    ComfyStory,
    ComfyStoryCommit,
    ComfyStoryKeyframeTiming,
    ComfyStoryTrim,
)
from .reference_nodes import ComfyStoryReferenceVAE

WEB_DIRECTORY = "./web"


class ComfyStoryExtension(ComfyExtension):
    """Register one public story node and its internal transaction node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ComfyStory,
            ComfyStoryCommit,
            ComfyStoryTrim,
            ComfyStoryKeyframeTiming,
            ComfyH3BalancedCache,
            ComfyH3UltraFast,
            ComfyH3VideoLatent,
            ComfyH3ReplaceVideoLatent,
            ComfyH3RefinementSigmas,
            ComfyStoryReferenceVAE,
        ]


async def comfy_entrypoint() -> ComfyStoryExtension:
    from server import PromptServer

    from .routes import register_routes

    register_routes(PromptServer.instance)
    return ComfyStoryExtension()


__all__ = (
    "WEB_DIRECTORY",
    "ComfyStory",
    "ComfyStoryCommit",
    "ComfyStoryExtension",
    "ComfyStoryTrim",
    "comfy_entrypoint",
)
