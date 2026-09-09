"""Authoring estimates for native H3 image references, without loading a model."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from comfy_story.film_bundle import _asset_path


def image_reference_size(
    width: int, height: int, canvas: tuple[int, int], mode: str
) -> tuple[int, int]:
    if min(width, height, *canvas) <= 0 or mode not in {"match", "max"}:
        raise ValueError("Invalid reference dimensions or sizing mode")
    scale = (
        min(1.0, math.sqrt(canvas[0] * canvas[1] / (width * height)))
        if mode == "match"
        else min(1.0, 2048 / min(width, height))
    )
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)


def review_image_references(
    root: Path, references: object, canvas: tuple[int, int], mode: str
) -> dict[str, object]:
    if not isinstance(references, list) or len(references) > 9:
        raise ValueError("Review up to nine named image references")
    rows = []
    for ref in references:
        if (
            not isinstance(ref, dict)
            or not isinstance(ref.get("name"), str)
            or not isinstance(ref.get("file"), str)
        ):
            raise ValueError("Each reference needs a name and uploaded image filename")
        if len(ref["name"]) > 96:
            raise ValueError("Reference name is too long")
        path = _asset_path(root, ref["file"], "image")
        with Image.open(path) as image:
            width, height = image.size
        tw, th = image_reference_size(width, height, canvas, mode)
        rows.append(
            {
                "name": ref["name"],
                "file": ref["file"],
                "width": width,
                "height": height,
                "encoded_width": tw,
                "encoded_height": th,
                "tokens": (tw // 32) * (th // 32),
            }
        )
    return {
        "references": rows,
        "named_image_tokens": sum(int(row["tokens"]) for row in rows),
        "sizing": mode,
        "scope": (
            "Named original images only. Approved evidence, historical context, ending frames "
            "and motion references are checked at render time. "
            "Token counts are estimates, not VRAM predictions."
        ),
    }
