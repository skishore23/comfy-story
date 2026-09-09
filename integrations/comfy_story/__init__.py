"""ComfyUI V3 registration for the Comfy Story product node."""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from .h3_quality_nodes import DuetH3RefinementSigmas, DuetH3ReplaceVideoLatent, DuetH3VideoLatent
from .h3_reference_node import (
    DuetH3BenchmarkReceipt,
    DuetH3CompileReferences,
    DuetH3TelemetryEnd,
    DuetH3TelemetryStart,
)
from .nodes import (
    DuetH3BalancedCache,
    DuetH3UltraFast,
    DuetStory,
    DuetStoryCanon,
    DuetStoryCommit,
    DuetStoryKeyframeTiming,
    DuetStoryTrim,
)

WEB_DIRECTORY = "./web"


class DuetStoryExtension(ComfyExtension):
    """Register one public story node and its internal transaction node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DuetStory,
            DuetStoryCanon,
            DuetStoryCommit,
            DuetStoryTrim,
            DuetStoryKeyframeTiming,
            DuetH3BalancedCache,
            DuetH3UltraFast,
            DuetH3CompileReferences,
            DuetH3TelemetryStart,
            DuetH3TelemetryEnd,
            DuetH3BenchmarkReceipt,
            DuetH3VideoLatent,
            DuetH3ReplaceVideoLatent,
            DuetH3RefinementSigmas,
        ]


async def comfy_entrypoint() -> DuetStoryExtension:
    from server import PromptServer

    from .routes import register_routes

    register_routes(PromptServer.instance)
    return DuetStoryExtension()


__all__ = (
    "WEB_DIRECTORY",
    "DuetH3BenchmarkReceipt",
    "DuetH3CompileReferences",
    "DuetH3TelemetryEnd",
    "DuetH3TelemetryStart",
    "DuetStory",
    "DuetStoryCanon",
    "DuetStoryCommit",
    "DuetStoryExtension",
    "DuetStoryTrim",
    "comfy_entrypoint",
)
