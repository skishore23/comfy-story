"""Public Comfy Story graph expansion and hidden post-decode commit node."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import folder_paths
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from comfy_story.film_timing import trim_shot_media
from comfy_story.h3_prompt import H3_PROMPT_FORMAT
from comfy_story.samplers import StorySampler
from comfy_story.story_contracts import ShotIntent
from comfy_story.story_video_canvas import H3_FIRST_FRAME_CROP, H3_FIRST_FRAME_METHOD

from .comfy_adapter import (
    commit_node_generation,
    prepare_node_generation,
    recover_node_generation,
    render_configuration,
    require_speed_sampler,
    story_inspector_summary,
)
from .h3_quality_nodes import build_refinement

StoryType = io.Custom("COMFY_STORY")
PreparedType = io.Custom("COMFY_STORY_PREPARED")


def _world_upload() -> Any:
    root = Path(folder_paths.get_input_directory()).resolve()
    options = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(root)
        and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    return io.Combo.Input(
        "World / starting frame",
        options=["None", *options],
        default="None",
        upload=io.UploadType.image,
        optional=True,
    )


class ComfyStory(io.ComfyNode):
    """Generate one finished shot while carrying shared story state forward."""

    @classmethod
    def define_schema(cls) -> Any:
        inputs = [
            StoryType.Input("Previous Story", optional=True),
            io.Image.Input("Previous Frame", optional=True),
            io.Combo.Input(
                "Create",
                options=[intent.value for intent in ShotIntent],
                default=ShotIntent.START_STORY.value,
            ),
            io.String.Input("Story Library", default="[]", multiline=True),
            _world_upload(),
            io.Image.Input("Motion reference", optional=True),
            io.String.Input(
                "What happens next?",
                default="@Maya enters the scene.",
                multiline=True,
                dynamic_prompts=True,
            ),
            io.Combo.Input(
                "Shot length",
                options=["5 seconds", "10 seconds", "15 seconds"],
                default="5 seconds",
            ),
            io.Int.Input(
                "Variation",
                default=101,
                min=0,
                max=0xFFFFFFFFFFFFFFFF,
                advanced=True,
            ),
            io.String.Input("Story revision", default="", advanced=True),
            io.Combo.Input(
                "Reference policy",
                options=["Automatic", "Prompt mentions only"],
                default="Automatic",
                advanced=True,
            ),
            io.Combo.Input(
                "Sampler",
                options=[
                    "Native res_multistep",
                    "SPEED Euler 2-stage",
                    "Turbo 4-step",
                    "Turbo 8-step",
                    "Full HD 2-pass",
                    "NVFP4 Exact",
                    "NVFP4 Balanced",
                    "NVFP4 Ultra Fast",
                    "NVFP4 Turbo 4-step",
                ],
                default="Native res_multistep",
                advanced=True,
            ),
            io.String.Input("Memory actions", default="[]", advanced=True),
            io.Combo.Input(
                "Composition",
                options=["Continue frame", "New composition"],
                default="Continue frame",
                optional=True,
                advanced=True,
            ),
            io.Audio.Input("Authored audio", optional=True),
            io.Image.Input("Ending frame", optional=True),
            io.String.Input("Scene entities", default="", optional=True, advanced=True),
            io.Int.Input(
                "Output duration (ms)", default=0, min=0, max=15000, optional=True, advanced=True
            ),
            io.Combo.Input(
                "Render profile",
                options=["Reference shot", "Animate frame"],
                default="Reference shot",
                optional=True,
            ),
            io.String.Input("Shot state evidence", default="{}", optional=True, advanced=True),
            io.Image.Input("Starting image", optional=True),
            io.Combo.Input(
                "Prompt format",
                options=["Current", "Structured reference (experimental)", H3_PROMPT_FORMAT],
                default="Current",
                optional=True,
                advanced=True,
            ),
        ]
        return io.Schema(
            node_id="ComfyStory",
            display_name="Comfy Story",
            category="Comfy/Story",
            description=(
                "Create the next shot with retrieved references, authenticated history, and "
                "creator-confirmed canon behind one Story State wire."
            ),
            search_aliases=["consistent characters", "long story video", "story memory"],
            enable_expand=True,
            is_output_node=True,
            inputs=inputs,
            hidden=[io.Hidden.unique_id],
            outputs=[
                io.Video.Output("video", display_name="Finished Video"),
                io.Image.Output("last_frame", display_name="Last Frame"),
                StoryType.Output("story_state", display_name="Story State"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **inputs: Any) -> str:
        """Re-enter durable recovery on each queue; cached expansions can lose UI receipts.

        Recovery verifies the stored generation and returns its video without a sampling graph.
        This also refreshes the inspector's owner ID when a completed node is copied.
        """
        # Comfy writes this scheduler nonce into history; keep strict JSON valid.
        return uuid.uuid4().hex

    @classmethod
    def execute(cls, **inputs: Any) -> Any:
        hidden = getattr(cls, "hidden", None)
        owner_node_id = str(
            inputs.pop(
                "__owner_node_id",
                inputs.pop("unique_id", getattr(hidden, "unique_id", "") or ""),
            )
        )
        prepared = prepare_node_generation(inputs)
        recovered = recover_node_generation(prepared)
        if recovered is not None:
            from comfy_api.latest import InputImpl

            loaded, video_path, last_frame = recovered
            ui = (
                None
                if not owner_node_id
                else {"comfy_story": [story_inspector_summary(loaded, owner_node_id=owner_node_id)]}
            )
            return io.NodeOutput(
                InputImpl.VideoFromFile(str(video_path)), last_frame, loaded.state, ui=ui
            )
        if prepared.sampler is StorySampler.SPEED_EULER_2STAGE:
            try:
                import nodes as comfy_nodes
            except ImportError as error:
                raise ValueError("MiniMaxH3SPEEDSampler is not installed") from error
            require_speed_sampler(getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}))
        nvfp4 = prepared.sampler in {
            StorySampler.NVFP4_EXACT,
            StorySampler.NVFP4_BALANCED,
            StorySampler.NVFP4_ULTRA_FAST,
            StorySampler.NVFP4_TURBO,
        }
        if prepared.sampler is StorySampler.FULL_HD_2PASS:
            import nodes as comfy_nodes

            from .h3_quality_nodes import require_upscaler

            require_upscaler(getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}))
        graph = GraphBuilder()
        profile = getattr(prepared, "render_profile", "Reference shot")
        configuration = render_configuration(profile, prepared.sampler)
        model = graph.node(
            "UNETLoader",
            unet_name=configuration["model"],
            weight_dtype="default",
        )
        turbo = "lora" in configuration
        if turbo:
            model = graph.node(
                "LoraLoaderModelOnly",
                model=model.out(0),
                lora_name=configuration["lora"],
                strength_model=configuration["lora_strength"],
            )
        if "shift_video" in configuration:
            model = graph.node(
                "MiniMaxH3SigmaShift",
                model=model.out(0),
                shift_video=configuration["shift_video"],
                shift_audio=configuration["shift_audio"],
            )
        if nvfp4:
            if prepared.sampler is StorySampler.NVFP4_ULTRA_FAST:
                model = graph.node("ComfyH3UltraFast", model=model.out(0))
            elif prepared.sampler is StorySampler.NVFP4_BALANCED:
                model = graph.node("ComfyH3BalancedCache", model=model.out(0))
        clip = graph.node(
            "CLIPLoader",
            clip_name=configuration["clip"],
            type="minimax",
            device="default",
        )
        video_vae = graph.node("VAELoader", vae_name=configuration["video_vae"])
        audio_vae = graph.node("VAELoader", vae_name=configuration["audio_vae"])
        noise = graph.node("RandomNoise", noise_seed=prepared.variation)
        sampler = (
            graph.node("KSamplerSelect", sampler_name=configuration["sampler"])
            if prepared.sampler is not StorySampler.SPEED_EULER_2STAGE
            else None
        )
        sigmas = graph.node(
            "BasicScheduler",
            model=model.out(0),
            scheduler=configuration["scheduler"],
            steps=configuration["steps"],
            denoise=1.0,
        )
        automatic_prompt = inputs.get("Prompt format", "Current") == H3_PROMPT_FORMAT
        ending_frame = getattr(prepared, "ending_frame", None)
        conditioning_inputs: dict[str, Any] = {
            "clip": clip.out(0),
            "vae": video_vae.out(0),
            "audio_vae": audio_vae.out(0),
            "prompt": prepared.resolved_prompt,
            "width": configuration["width"],
            "height": configuration["height"],
            "length": prepared.frame_count,
            "ref_image_size": "match" if turbo else "max",
            **{
                f"ref_images.ref_image_{index}": guide
                for index, guide in enumerate(prepared.visual_guides)
            },
        }
        if automatic_prompt and ending_frame is not None and profile != "Animate frame":
            conditioning_inputs[f"ref_images.ref_image_{len(prepared.visual_guides)}"] = (
                ending_frame
            )
        motion_reference = getattr(prepared, "motion_reference", None)
        if motion_reference is not None:
            conditioning_inputs["ref_videos.ref_video_0"] = motion_reference
        if profile == "Animate frame":
            fitted = graph.node(
                "ImageScale",
                image=prepared.visual_guides[0],
                upscale_method=H3_FIRST_FRAME_METHOD,
                width=configuration["width"],
                height=configuration["height"],
                crop=H3_FIRST_FRAME_CROP,
            )
            conditioning = graph.node(
                "MiniMaxH3ImageToVideo",
                clip=clip.out(0),
                vae=video_vae.out(0),
                prompt=prepared.resolved_prompt,
                width=configuration["width"],
                height=configuration["height"],
                length=prepared.frame_count,
                first_frame=fitted.out(0),
                **(
                    {"last_frame": ending_frame}
                    if automatic_prompt and ending_frame is not None
                    else {}
                ),
            )
        else:
            reference_vae = graph.node(
                "ComfyStoryReferenceVAE",
                vae=video_vae.out(0),
                vae_name=configuration["video_vae"],
            )
            conditioning_inputs["vae"] = reference_vae.out(0)
            conditioning = graph.node("MiniMaxH3ReferenceToVideo", **conditioning_inputs)
        positive = conditioning.out(0)
        if (
            profile != "Animate frame"
            and inputs.get("Composition", "Continue frame") == "Continue frame"
        ):
            guided = graph.node(
                "MiniMaxH3AddGuide",
                positive=positive,
                latent=conditioning.out(1),
                vae=video_vae.out(0),
                image=prepared.visual_guides[0],
                frame_idx=0,
            )
            positive = guided.out(0)
        ending_frame = getattr(prepared, "ending_frame", None)
        if ending_frame is not None:
            duration_ms = getattr(prepared, "output_duration_ms", 0)
            last_index = duration_ms * 24 // 1000 - 1 if duration_ms else prepared.frame_count - 1
            if automatic_prompt and profile == "Animate frame":
                guided = graph.node(
                    "ComfyStoryKeyframeTiming", positive=positive, last_index=last_index
                )
            else:
                guided = graph.node(
                    "MiniMaxH3AddGuide",
                    positive=positive,
                    latent=conditioning.out(1),
                    vae=video_vae.out(0),
                    image=ending_frame,
                    frame_idx=last_index,
                )
            positive = guided.out(0)
        guider = graph.node("BasicGuider", model=model.out(0), conditioning=positive)
        if prepared.sampler is not StorySampler.SPEED_EULER_2STAGE:
            if sampler is None:
                raise RuntimeError("core sampler graph was not initialized")
            sampled = graph.node(
                "SamplerCustomAdvanced",
                noise=noise.out(0),
                guider=guider.out(0),
                sampler=sampler.out(0),
                sigmas=sigmas.out(0),
                latent_image=conditioning.out(1),
            )
        else:
            sampled = graph.node(
                "MiniMaxH3SPEEDSampler",
                noise=noise.out(0),
                guider=guider.out(0),
                sigmas=sigmas.out(0),
                latent_image=conditioning.out(1),
                stages=2,
                noise_policy="direct_coarse",
                **{
                    "Tolerance (Delta)": 0.01,
                    "noise_amplitude": 7.394,
                    "noise_decay_exponent": 0.62,
                    "seed_offset": 10_000,
                },
            )
        sampled_latent = sampled.out(0)
        if prepared.sampler is StorySampler.FULL_HD_2PASS:
            sampled_latent = build_refinement(
                graph,
                prepared,
                configuration,
                inputs,
                clip.out(0),
                video_vae.out(0),
                audio_vae.out(0),
                model.out(0),
                sampled_latent,
            )
        images = (
            graph.node(
                "VAEDecodeTiled",
                samples=sampled_latent,
                vae=video_vae.out(0),
                tile_size=256,
                overlap=32,
                temporal_size=64,
                temporal_overlap=8,
            )
            if prepared.sampler is StorySampler.FULL_HD_2PASS
            else graph.node("VAEDecode", samples=sampled_latent, vae=video_vae.out(0))
        )
        authored_audio = getattr(prepared, "authored_audio", None)
        audio_input = (
            authored_audio.as_comfy()
            if authored_audio is not None
            else graph.node("VAEDecodeAudio", samples=sampled_latent, vae=audio_vae.out(0)).out(0)
        )
        image_output = images.out(0)
        output_duration = getattr(prepared, "output_duration_ms", 0)
        if output_duration:
            trimmed = graph.node(
                "ComfyStoryTrim",
                images=image_output,
                audio=audio_input,
                duration_ms=output_duration,
            )
            image_output, audio_input = trimmed.out(0), trimmed.out(1)
        video = graph.node(
            "CreateVideo", images=image_output, audio=audio_input, fps=24.0, bit_depth=8
        )
        saved = graph.node(
            "SaveVideo",
            video=video.out(0),
            filename_prefix="comfy_story/shot",
            format="auto",
            codec="auto",
        )
        commit_inputs: dict[str, Any] = {
            "decoded_images": image_output,
            "saved_video": saved.out(0),
            "filename_prefix": "comfy_story/shot",
            "prepared": prepared.prepared,
        }
        if (
            getattr(getattr(prepared.prepared, "request", None), "associative_memory", None)
            is not None
        ):
            commit_inputs["memory_vae"] = video_vae.out(0)
        commit_inputs["owner_node_id"] = owner_node_id
        committed = graph.node(
            "ComfyStoryCommit",
            **commit_inputs,
        )
        story_state = committed.out(0)
        return io.NodeOutput(saved.out(0), committed.out(1), story_state, expand=graph.finalize())


class ComfyStoryCommit(io.ComfyNode):
    """Internal transaction boundary that publishes state only after decoding succeeds."""

    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="ComfyStoryCommit",
            display_name="_Comfy Story Commit",
            category="_Comfy/Internal",
            inputs=[
                io.Image.Input("decoded_images"),
                io.Video.Input("saved_video"),
                io.String.Input("filename_prefix", default="comfy_story/shot"),
                PreparedType.Input("prepared"),
                io.Vae.Input("memory_vae", optional=True),
                io.String.Input("owner_node_id", default="", optional=True),
            ],
            outputs=[
                StoryType.Output("story_state"),
                io.Image.Output("last_frame"),
            ],
        )

    @classmethod
    def execute(cls, **inputs: Any) -> Any:
        result = commit_node_generation(
            prepared=inputs["prepared"],
            decoded_images=inputs["decoded_images"],
            saved_video=inputs["saved_video"],
            filename_prefix=inputs["filename_prefix"],
            memory_vae=inputs.get("memory_vae"),
        )
        owner_node_id = str(inputs.get("owner_node_id", ""))
        ui = (
            None
            if not owner_node_id
            else {
                "comfy_story": [
                    story_inspector_summary(
                        result.loaded_revision,
                        owner_node_id=owner_node_id,
                    )
                ]
            }
        )
        return io.NodeOutput(result.state, result.last_frame, ui=ui)


class ComfyStoryTrim(io.ComfyNode):
    """Keep the exact edited interval before saving and publishing continuation state."""

    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="ComfyStoryTrim",
            display_name="_Comfy Story Trim",
            category="_Comfy/Internal",
            inputs=[
                io.Image.Input("images"),
                io.Audio.Input("audio"),
                io.Int.Input("duration_ms", min=1, max=15000),
            ],
            outputs=[io.Image.Output("images"), io.Audio.Output("audio")],
        )

    @classmethod
    def execute(cls, **inputs: Any) -> Any:
        images, audio = trim_shot_media(inputs["images"], inputs["audio"], inputs["duration_ms"])
        return io.NodeOutput(images, audio)


__all__ = ("ComfyStory", "ComfyStoryCommit", "ComfyStoryTrim")


class ComfyH3BalancedCache(io.ComfyNode):
    """Internal native-H3 model patch for the public Balanced sampler."""

    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="ComfyH3BalancedCache",
            display_name="Comfy H3 Balanced Cache",
            category="_Comfy/Internal",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model: Any) -> Any:
        from comfy.cli_args import args

        from comfy_story.h3_acceleration import patch_h3_balanced

        if not getattr(args, "disable_comfy_compiler", True):
            raise ValueError("NVFP4 Balanced requires ComfyUI --disable-comfy-compiler")

        return io.NodeOutput(patch_h3_balanced(model))


class ComfyH3UltraFast(io.ComfyNode):
    """Native Ref2VA Balanced cache plus the Space's Sol-Attn policy."""

    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="ComfyH3UltraFast",
            display_name="Comfy H3 Ultra Fast",
            category="_Comfy/Internal",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model: Any) -> Any:
        from comfy.cli_args import args

        from comfy_story.h3_acceleration import patch_h3_balanced

        if not getattr(args, "disable_comfy_compiler", True):
            raise ValueError("NVFP4 Ultra Fast requires ComfyUI --disable-comfy-compiler")
        return io.NodeOutput(patch_h3_balanced(model, sol_attention=True))


class ComfyStoryKeyframeTiming(io.ComfyNode):
    """Move the native FL2VA endpoint onto the final exported frame."""

    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="ComfyStoryKeyframeTiming",
            display_name="_Comfy Story Keyframe Timing",
            category="_Comfy/Internal",
            inputs=[io.Conditioning.Input("positive"), io.Int.Input("last_index", min=1)],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, positive: Any, last_index: int) -> Any:
        if type(last_index) is not int or last_index < 1 or not positive:
            raise ValueError("H3 ending timing needs a positive final frame index and conditioning")
        result = []
        for embedding, metadata in positive:
            keyframes = metadata.get("minimax_keyframes", [])
            if (
                len(keyframes) != 2
                or keyframes[0].get("resolved_frame_index") != 0
                or type(keyframes[1].get("resolved_frame_index")) is not int
                or keyframes[1]["resolved_frame_index"] < last_index
                or any("latent" not in frame for frame in keyframes)
            ):
                raise ValueError("H3 ending timing requires native first/last image conditioning")
            frames = [dict(frame) for frame in keyframes]
            frames[1]["resolved_frame_index"] = last_index
            result.append([embedding, {**metadata, "minimax_keyframes": frames}])
        return io.NodeOutput(result)
