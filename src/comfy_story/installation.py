"""Preflight an installed ComfyUI host using the exact runtime about to be installed."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from comfy_story.memory.health import history_effect_report
from comfy_story.memory.settings import InspectionCodec, configured_memory

REQUIRED_MODELS = {
    "diffusion_models": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "text_encoders": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_installation(root: Path, extra_model_paths: list[Path]) -> dict[str, Any]:
    """Authenticate memory and required H3 model files without loading GPU weights."""
    root = root.resolve()
    if not (root / "comfy_extras/nodes_minimax_h3.py").is_file():
        raise ValueError("Update ComfyUI: this installation is missing native MiniMax H3 nodes")
    sys.path.insert(0, str(root))
    folders = importlib.import_module("folder_paths")
    default_paths = root / "extra_model_paths.yaml"
    configs = ([default_paths] if default_paths.is_file() else []) + extra_model_paths
    if configs:
        loader = importlib.import_module("utils.extra_config")
        for config in configs:
            loader.load_extra_path_config(str(config.resolve()))
    missing = []
    models = {}
    for kind, name in REQUIRED_MODELS.items():
        folder = "vae" if kind.endswith("vae") else kind
        resolved = folders.get_full_path(folder, name)
        if resolved is None or not Path(resolved).is_file():
            missing.append(f"{folder}/{name}")
        else:
            models[kind] = Path(resolved)
    if missing:
        raise ValueError(
            "Required H3 models are missing: "
            + ", ".join(missing)
            + ". Install the authorized H3 model pack, or supply --extra-model-paths-config "
            "for an existing shared model directory."
        )
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise ValueError("Install ffmpeg and ffprobe on PATH for film export and media validation")
    memory = configured_memory()
    for kind, expected in (
        ("diffusion_models", memory.foundation_sha256),
        ("video_vae", memory.vae_sha256),
    ):
        if _digest(models[kind]) != expected:
            raise ValueError(f"H3 {kind} does not match the supported memory checkpoint")
    runtime = memory.load(InspectionCodec())
    try:
        report = history_effect_report(runtime.bridge, require_effect=True)
    finally:
        runtime.close()
    return {
        "checkpoint_authenticated": True,
        "history_effect": report,
        "memory_configuration": memory.binding(),
        "model_files": {kind: str(path) for kind, path in models.items()},
        "cuda_available": torch.cuda.is_available(),
        "rendered_quality_validated": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Comfy Story memory and H3 prerequisites")
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--extra-model-paths-config", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        report = check_installation(args.comfy_root, args.extra_model_paths_config)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    if not report["cuda_available"]:
        print("CUDA is unavailable: project editing is supported; GPU generation is not ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
