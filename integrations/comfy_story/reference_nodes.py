"""Cache only reference VAE calls while preserving native H3 preprocessing and text encoding."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from comfy_api.latest import io

from comfy_story.reference_cache import ReferenceCache

from .comfy_adapter import _generation_asset_digest, _story_root


class ReferenceVAE:
    def __init__(self, vae: Any, cache: ReferenceCache):
        self.vae, self.cache = vae, cache

    def __getattr__(self, name: str) -> Any:
        return getattr(self.vae, name)

    def encode(self, pixels: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if args or kwargs:
            return self.vae.encode(pixels, *args, **kwargs)
        encoded = self.cache.encode(pixels, self.vae.encode)
        return encoded.to(self.vae.output_device)


class ComfyStoryReferenceVAE(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="ComfyStoryReferenceVAE",
            display_name="Comfy Story Reference Cache",
            category="_Comfy/Internal",
            inputs=[io.Vae.Input("vae"), io.String.Input("vae_name")],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(cls, vae: Any, vae_name: str) -> Any:
        # Bind both weights and encoder implementation. Resized pixels carry crop, scale,
        # frame selection and changing associative-history content into the cache key.
        files = {inspect.getfile(type(vae)), inspect.getfile(type(vae.first_stage_model))}
        identity = json.dumps(
            {
                "weights": _generation_asset_digest("vae", vae_name),
                "code": {
                    name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    for name in sorted(files)
                },
                "dtype": str(vae.vae_dtype),
                "device": str(vae.device),
                "torch": str(torch.__version__),
            },
            sort_keys=True,
        )
        return io.NodeOutput(
            ReferenceVAE(vae, ReferenceCache(_story_root() / "reference_cache", identity))
        )
