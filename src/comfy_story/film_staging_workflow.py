"""Explicit, optional Qwen opening-image recipe for the normal film controller."""

from __future__ import annotations

from comfy_story.film_plan import FilmPlan, planned_shot_conditions
from comfy_story.film_state_audit import state_definitions
from comfy_story.film_workflow import _input_file
from comfy_story.story_video_canvas import (
    H3_FIRST_FRAME_CROP,
    H3_FIRST_FRAME_METHOD,
    H3_VIDEO_HEIGHT,
    H3_VIDEO_WIDTH,
)

STAGING_PROTOCOL = "comfy-film-opening-stage-v3"
STAGING_MODELS = {
    "diffusion_models": "qwen_image_edit_2511_int8_convrot.safetensors",
    "text_encoders": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae": "qwen_image_vae.safetensors",
}


def opening_stage_prompt(
    plan: FilmPlan,
    index: int,
    description: str,
    roles: tuple[str, ...],
    mode: str = "Compose",
) -> str:
    """Compile only the opening, never treat a future effect as an accomplished fact."""
    if mode not in ("Compose", "Refine"):
        raise ValueError("opening mode must be Compose or Refine")
    shot = plan.shots[index]
    conditions = planned_shot_conditions(plan, index)
    definitions = state_definitions(plan)
    lines = [
        (
            "Refine the supplied opening image into one natural film frame. Image 1 fixes the "
            "layout: preserve its subjects, positions, relative sizes and facing directions. "
            "Harmonize pasted edges, ground contact and lighting while preserving identities. "
            "Do not add, remove or relocate subjects."
            if mode == "Refine"
            else "Compose one opening frame for a film shot, preserving referenced identities."
        ),
        description.strip(),
        *[f"Image {i}: {role}." for i, role in enumerate(roles, 1)],
        "Required visible starting state:",
    ]
    for key, value in sorted(conditions.items()):
        lines.append(f"{key} = {value}. " + definitions.get(key, {}).get(value, ""))
    if shot.visible_throughout:
        lines.append("Visible subjects: " + ", ".join(shot.visible_throughout) + ".")
    if shot.fully_visible_throughout:
        lines.append(
            "Keep entirely inside the picture with margins and no covering objects: "
            + ", ".join(shot.fully_visible_throughout)
            + "."
        )
    if shot.absent:
        lines.append("Exclude these entities: " + ", ".join(shot.absent) + ".")
    return "\n".join(lines)


def compile_opening_stage(
    prompt: str, references: tuple[str, ...], seed: int, mode: str = "Compose"
) -> dict[str, object]:
    """Preserve the raw 40-step edit and assess the fitted H3 first-frame canvas.

    Qwen uses the first reference's scaled aspect and latent size. Fit its result before
    assessment, so H3 cannot subsequently crop away content the opening check relied on.
    No optional Lightning adapter or sampler switch is hidden in this configuration.
    The host must provide these standard Comfy nodes and the explicitly named weights.
    """
    if mode not in ("Compose", "Refine"):
        raise ValueError("opening mode must be Compose or Refine")
    if not 1 <= len(references) <= 3:
        raise ValueError("opening staging supports one to three reference images")
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise ValueError("opening staging seed must fit an unsigned 64-bit integer")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000 or "\x00" in prompt:
        raise ValueError("opening staging prompt must be nonempty bounded text")
    graph: dict[str, object] = {}

    def node(key: str, kind: str, **inputs: object) -> None:
        graph[key] = {"class_type": kind, "inputs": inputs}

    images: dict[str, object] = {}
    for i, filename in enumerate(references, 1):
        _input_file(filename)
        key = str(100 + i)
        node(key, "LoadImage", image=filename)
        images[f"image{i}"] = [key, 0]
    node("23", "FluxKontextImageScale", image=["101", 0])
    images["image1"] = ["23", 0]
    node("13", "UNETLoader", unet_name=STAGING_MODELS["diffusion_models"], weight_dtype="default")
    node(
        "14",
        "CLIPLoader",
        clip_name=STAGING_MODELS["text_encoders"],
        type="qwen_image",
        device="default",
    )
    node("3", "VAELoader", vae_name=STAGING_MODELS["vae"])
    node("2", "ModelSamplingAuraFlow", shift=3.1, model=["13", 0])
    node("8", "CFGNorm", strength=1.0, pre_cfg=False, model=["2", 0])
    for key, text in (("6", ""), ("7", prompt)):
        node(
            key, "TextEncodeQwenImageEditPlus", prompt=text, clip=["14", 0], vae=["3", 0], **images
        )
    for key, source in (("4", "6"), ("5", "7")):
        node(
            key,
            "FluxKontextMultiReferenceLatentMethod",
            reference_latents_method="index_timestep_zero",
            conditioning=[source, 0],
        )
    node("12", "VAEEncode", pixels=["23", 0], vae=["3", 0])
    node(
        "21",
        "KSampler",
        seed=seed,
        sampler_name="euler",
        scheduler="simple",
        denoise=0.35 if mode == "Refine" else 1.0,
        model=["8", 0],
        steps=40,
        cfg=4.0,
        positive=["5", 0],
        negative=["4", 0],
        latent_image=["12", 0],
    )
    node("22", "VAEDecode", samples=["21", 0], vae=["3", 0])
    node("25", "SaveImage", filename_prefix="comfy_story_staging/raw", images=["22", 0])
    node(
        "26",
        "ImageScale",
        image=["22", 0],
        upscale_method=H3_FIRST_FRAME_METHOD,
        width=H3_VIDEO_WIDTH,
        height=H3_VIDEO_HEIGHT,
        crop=H3_FIRST_FRAME_CROP,
    )
    node("24", "SaveImage", filename_prefix="comfy_story_staging/opening", images=["26", 0])
    return graph
