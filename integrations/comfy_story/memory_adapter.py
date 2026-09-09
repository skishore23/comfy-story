"""Comfy-owned video VAE access for associative Story memory."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class ComfyMiniMaxH3Codec:
    """Narrow adapter around the MiniMax video VAE already owned by Comfy."""

    def __init__(self, vae: object) -> None:
        if not callable(getattr(vae, "encode", None)) or not callable(getattr(vae, "decode", None)):
            raise ValueError("memory_vae must be a loaded Comfy VAE")
        self._vae = vae

    def encode_frame(self, frame: np.ndarray[Any, Any]) -> torch.Tensor:
        if frame.dtype != np.uint8 or frame.shape != (384, 384, 3):
            raise ValueError("MiniMax H3 codec requires RGB uint8 384x384")
        images = (
            torch.from_numpy(np.array(frame, copy=True)).to(torch.float32).div(255).unsqueeze(0)
        )
        latent = self._vae.encode(images)  # type: ignore[attr-defined]
        if (
            not isinstance(latent, torch.Tensor)
            or latent.ndim != 5
            or latent.shape[0] != 1
            or latent.shape[1] != 24
            or any(dimension <= 0 for dimension in latent.shape[2:])
            or not torch.is_floating_point(latent)
            or not bool(torch.isfinite(latent).all().item())
        ):
            raise ValueError("Comfy MiniMax H3 VAE returned an invalid 24-channel latent")
        return latent.detach()

    def decode_frame(self, latent: torch.Tensor) -> np.ndarray[Any, Any]:
        decoded = self._vae.decode(latent)  # type: ignore[attr-defined]
        if not isinstance(decoded, torch.Tensor) or not torch.is_floating_point(decoded):
            raise ValueError("Comfy MiniMax H3 VAE returned an invalid image tensor")
        if decoded.ndim == 5 and decoded.shape[0] == 1:
            image = decoded[0, -1]
        elif decoded.ndim == 4:
            image = decoded[-1]
        else:
            raise ValueError("Comfy MiniMax H3 VAE returned an unexpected image boundary")
        if image.shape != (384, 384, 3) or not bool(torch.isfinite(image).all().item()):
            raise ValueError("Comfy MiniMax H3 VAE returned an unexpected RGB frame")
        value = image.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1)
        return np.ascontiguousarray(value.mul(255).round().to(torch.uint8).numpy())

    def close(self) -> None:
        """The VAE lifecycle remains owned by Comfy's graph executor."""
