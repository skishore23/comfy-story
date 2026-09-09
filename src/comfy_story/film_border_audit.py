"""Conservative pixel observations for an explicit opening-border contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from comfy_story.film_audit import ShotVisualAudit, VisualFinding


def black_borders(path: Path) -> tuple[float, float, float, float] | None:
    """Return top/bottom/left/right fractions; dark frames are unobservable.

    Only almost entirely black rows/columns count. This does not identify every
    matte, establish camera stability, or distinguish an intentional bar from a
    generated one. The caller must explicitly request preservation.
    """
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
    height, width = pixels.shape[:2]
    dark = pixels.max(axis=2) <= 16
    rows = dark.mean(axis=1) >= 0.995
    columns = dark.mean(axis=0) >= 0.995
    lengths = [
        next((index for index, value in enumerate(edge) if not value), len(edge))
        for edge in (rows, rows[::-1], columns, columns[::-1])
    ]
    top, bottom, left, right = lengths
    if top + bottom >= height * 0.8 or left + right >= width * 0.8:
        return None
    return top / height, bottom / height, left / width, right / width


def apply_border_policy(
    audit: ShotVisualAudit, frames: list[Path], indices: tuple[int, ...]
) -> ShotVisualAudit:
    if len(frames) != len(indices) or len(frames) < 2:
        raise ValueError("border assessment requires aligned opening and later frames")
    observed = [black_borders(path) for path in frames]
    opening = observed[0]
    findings = []
    for index, borders in zip(indices[1:], observed[1:], strict=True):
        if opening is None or borders is None:
            status = "uncertain"
            evidence = "Opening-border preservation cannot be measured in a nearly black frame."
        else:
            # One percent tolerates edge compression and subpixel resizing noise.
            changed = any(abs(a - b) > 0.01 for a, b in zip(opening, borders, strict=True))
            status = "fail" if changed else "pass"
            evidence = (
                "Preserve opening borders: black top/bottom/left/right fractions "
                f"{tuple(round(x, 4) for x in opening)} -> "
                f"{tuple(round(x, 4) for x in borders)}. "
                "Measured pixels only; unsampled intervals and camera motion remain unverified."
            )
        findings.append(VisualFinding("temporal_stability", status, evidence, (indices[0], index)))
    failed = audit.status == "fail" or any(row.status == "fail" for row in findings)
    uncertain = audit.status == "needs_review" or any(row.status == "uncertain" for row in findings)
    return replace(
        audit,
        findings=(*audit.findings, *findings),
        status="fail" if failed else "needs_review" if uncertain else "machine_pass",
    )
