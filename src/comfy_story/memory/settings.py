"""Required, portable checkpoint pins for associative Story memory."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch

import comfy_story.memory.catalog as catalog
from comfy_story.memory.checkpoint import MemoryProtocol
from comfy_story.memory.runtime import MiniMaxH3Codec, MiniMaxH3StoryRuntime
from comfy_story.story_contracts import canonical_story_json


@dataclass(frozen=True, slots=True)
class MemoryConfiguration:
    checkpoint: str
    checkpoint_sha256: str
    foundation_sha256: str
    model_configuration_sha256: str
    vae_sha256: str

    def validate(self) -> Self:
        path = Path(self.checkpoint)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("associative memory checkpoint must be an absolute regular file")
        for name, value in self.binding().items():
            if name == "format":
                continue
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        return self

    def binding(self) -> dict[str, str]:
        # Bind serialized state and reuse to the actual runtime shipped in the wheel.
        implementation = hashlib.sha256()
        for source in sorted(Path(__file__).parent.glob("*.py")):
            implementation.update(source.name.encode() + b"\0" + source.read_bytes() + b"\0")
        return {
            "implementation_sha256": implementation.hexdigest(),
            "format": "comfy-story-associative-memory-v1",
            "checkpoint_sha256": self.checkpoint_sha256,
            "foundation_sha256": self.foundation_sha256,
            "model_configuration_sha256": self.model_configuration_sha256,
            "vae_sha256": self.vae_sha256,
            "protocol_sha256": MemoryProtocol.default().fingerprint(),
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_story_json(self.binding())).hexdigest()

    def load(self, codec: MiniMaxH3Codec) -> MiniMaxH3StoryRuntime:
        self.validate()
        return MiniMaxH3StoryRuntime.load_pinned(
            Path(self.checkpoint),
            expected_checkpoint_sha256=self.checkpoint_sha256,
            expected_foundation_sha256=self.foundation_sha256,
            expected_protocol_sha256=MemoryProtocol.default().fingerprint(),
            model_configuration_sha256=self.model_configuration_sha256,
            vae_sha256=self.vae_sha256,
            codec=codec,
            device=torch.device("cpu"),
        )


class InspectionCodec:
    """Checkpoint preflight never allocates a foundation VAE."""

    def encode_frame(self, frame: np.ndarray[Any, Any]) -> torch.Tensor:
        raise RuntimeError("checkpoint inspection cannot encode images")

    def decode_frame(self, latent: torch.Tensor) -> np.ndarray[Any, Any]:
        raise RuntimeError("checkpoint inspection cannot decode images")

    def close(self) -> None:
        pass


def configured_memory() -> MemoryConfiguration:
    mode = os.environ.get("COMFY_STORY_MEMORY", "associative")
    if mode != "associative":
        raise ValueError(
            "Comfy Story requires associative memory; remove COMFY_STORY_MEMORY "
            "or set it to associative, then configure the checkpoint and identity hashes"
        )
    fields = {
        "checkpoint": "CHECKPOINT",
        "checkpoint_sha256": "CHECKPOINT_SHA256",
        "foundation_sha256": "FOUNDATION_SHA256",
        "model_configuration_sha256": "MODEL_CONFIGURATION_SHA256",
        "vae_sha256": "VAE_SHA256",
    }
    # Advanced overrides are all-or-nothing; never mix a custom checkpoint with shipped pins.
    if not any("COMFY_STORY_MEMORY_" + suffix in os.environ for suffix in fields.values()):
        checkpoint = Path(__file__).parents[1] / "models" / catalog.CHECKPOINT_FILENAME
        if not checkpoint.is_file():
            raise ValueError(
                "Bundled associative memory checkpoint is missing; install the complete "
                "Comfy Story release or run install.py from the source checkout"
            )
        return MemoryConfiguration(
            str(checkpoint),
            catalog.CHECKPOINT_SHA256,
            catalog.FOUNDATION_SHA256,
            catalog.MODEL_CONFIGURATION_SHA256,
            catalog.VAE_SHA256,
        ).validate()
    values = {}
    for field, suffix in fields.items():
        value = os.environ.get("COMFY_STORY_MEMORY_" + suffix)
        if not value:
            raise ValueError("COMFY_STORY_MEMORY_" + suffix + " is required for associative memory")
        values[field] = value
    return MemoryConfiguration(**values).validate()
