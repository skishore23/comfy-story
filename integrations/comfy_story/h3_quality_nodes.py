"""Internal AV-safe latent upscaling and a second native H3 sampling pass."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from comfy_api.latest import io

from comfy_story.h3_quality import REFINEMENT_SIGMAS, UPSCALER_MODEL, UPSCALER_SOURCE_SHA256


def require_upscaler(node_classes: Mapping[str, object]) -> None:
    """Bind refinement to the reviewed external implementation before GPU work."""
    node = node_classes.get("MinimaxH3LatentUpscaler3D")
    if node is None:
        raise ValueError("Full HD 2-pass requires the MiniMax H3 3D latent upscaler")
    try:
        source = Path(inspect.getfile(node)).read_bytes()
    except (TypeError, OSError) as error:
        raise ValueError("Cannot verify the installed H3 latent upscaler") from error
    if hashlib.sha256(source).hexdigest() != UPSCALER_SOURCE_SHA256:
        raise ValueError("Full HD 2-pass requires the documented, pinned H3 latent upscaler")


def build_refinement(
    graph: Any,
    prepared: Any,
    configuration: dict[str, Any],
    inputs: dict[str, Any],
    clip: Any,
    vae: Any,
    audio_vae: Any,
    model: Any,
    latent: Any,
) -> Any:
    """Rebuild spatial conditioning; reference images never inherit low-res guide latents."""
    width, height = configuration["output_width"], configuration["output_height"]
    video = graph.node("DuetH3VideoLatent", latent=latent)
    upscale = graph.node(
        "MinimaxH3LatentUpscaler3D",
        latent=video.out(0),
        model_name=UPSCALER_MODEL,
        mode="target dimensions",
        **{"mode.width": width, "mode.height": height},
        align=32,
        enable_temporal_chunking=True,
        force_unload=True,
        device="cuda",
        precision="fp16",
    )
    joined = graph.node("DuetH3ReplaceVideoLatent", original=latent, video=upscale.out(0))
    ending = getattr(prepared, "ending_frame", None)
    profile = getattr(prepared, "render_profile", "Reference shot")
    common = {
        "clip": clip,
        "vae": vae,
        "prompt": prepared.resolved_prompt,
        "width": width,
        "height": height,
        "length": prepared.frame_count,
    }
    if profile == "Animate frame":
        fitted = graph.node(
            "ImageScale",
            image=prepared.visual_guides[0],
            upscale_method="bilinear",
            width=width,
            height=height,
            crop="center",
        )
        conditioning = graph.node(
            "MiniMaxH3ImageToVideo",
            **common,
            first_frame=fitted.out(0),
            **({"last_frame": ending} if ending is not None else {}),
        )
    else:
        references = {
            f"ref_images.ref_image_{i}": image for i, image in enumerate(prepared.visual_guides)
        }
        if ending is not None:
            references[f"ref_images.ref_image_{len(prepared.visual_guides)}"] = ending
        motion = getattr(prepared, "motion_reference", None)
        if motion is not None:
            references["ref_videos.ref_video_0"] = motion
        conditioning = graph.node(
            "MiniMaxH3ReferenceToVideo",
            **common,
            audio_vae=audio_vae,
            ref_image_size="match",
            **references,
        )
    positive = conditioning.out(0)
    if (
        profile != "Animate frame"
        and inputs.get("Composition", "Continue frame") == "Continue frame"
    ):
        positive = graph.node(
            "MiniMaxH3AddGuide",
            positive=positive,
            latent=joined.out(0),
            vae=vae,
            image=prepared.visual_guides[0],
            frame_idx=0,
        ).out(0)
    if ending is not None:
        duration = getattr(prepared, "output_duration_ms", 0)
        last_index = duration * 24 // 1000 - 1 if duration else prepared.frame_count - 1
        if profile == "Animate frame":
            positive = graph.node(
                "DuetStoryKeyframeTiming", positive=positive, last_index=last_index
            ).out(0)
        else:
            positive = graph.node(
                "MiniMaxH3AddGuide",
                positive=positive,
                latent=joined.out(0),
                vae=vae,
                image=ending,
                frame_idx=last_index,
            ).out(0)
    guider = graph.node("BasicGuider", model=model, conditioning=positive)
    noise = graph.node("RandomNoise", noise_seed=(prepared.variation + 10000) % (2**64))
    sigmas = graph.node("DuetH3RefinementSigmas")
    sampler = graph.node("KSamplerSelect", sampler_name="euler")
    return graph.node(
        "SamplerCustomAdvanced",
        noise=noise.out(0),
        guider=guider.out(0),
        sampler=sampler.out(0),
        sigmas=sigmas.out(0),
        latent_image=joined.out(0),
    ).out(0)


def _streams(latent: Any) -> tuple[Any, Any]:
    samples = latent["samples"]
    if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
        raise ValueError("H3 upscaling requires native video/audio latents")
    video, audio = samples.tensors
    if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4 or audio.shape[1] != 32:
        raise ValueError("H3 upscaling requires 24-channel video and 32-channel audio")
    return video, audio


class DuetH3VideoLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="DuetH3VideoLatent",
            display_name="H3 video latent",
            category="_Duet/Internal",
            inputs=[io.Latent.Input("latent")],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, latent: Any) -> Any:
        video, _ = _streams(latent)
        return io.NodeOutput({"samples": video})


class DuetH3ReplaceVideoLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="DuetH3ReplaceVideoLatent",
            display_name="H3 replace video latent",
            category="_Duet/Internal",
            inputs=[io.Latent.Input("original"), io.Latent.Input("video")],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, original: Any, video: Any) -> Any:
        from comfy.nested_tensor import NestedTensor

        old, audio = _streams(original)
        new = video["samples"]
        if new.ndim != 5 or new.shape[:3] != old.shape[:3] or new.shape[-2:] != (68, 120):
            raise ValueError("H3 Full HD upscaler changed the temporal contract or target canvas")
        return io.NodeOutput({**original, "samples": NestedTensor((new, audio))})


class DuetH3RefinementSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="DuetH3RefinementSigmas",
            display_name="H3 refinement sigmas",
            category="_Duet/Internal",
            inputs=[],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls) -> Any:
        return io.NodeOutput(torch.tensor(REFINEMENT_SIGMAS, dtype=torch.float32))
