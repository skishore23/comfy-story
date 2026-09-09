"""Inspect an uploaded soundtrack without accepting arbitrary host filesystem paths."""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

from duet.duetx.film_bundle import _asset_path
from duet.duetx.film_export import file_digest


def _soundtrack_levels(path: Path) -> dict[str, object]:
    """Measure decoded sample levels; this is not a listening or clipping verdict."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-nostats",
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                "wav,flac,mp3,ogg,mov,matroska,webm,aac",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-af",
                "astats=metadata=0:reset=0:measure_perchannel=none:"
                "measure_overall=Peak_level+RMS_level+Number_of_samples",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        values = dict(
            re.findall(
                r"^\[Parsed_astats_0 @ [^\]]+\] "
                r"(Peak level dB|RMS level dB|Number of samples): ([^\r\n]+)$",
                result.stderr,
                re.MULTILINE,
            )
        )
        peak = float(values["Peak level dB"])
        rms = float(values["RMS level dB"])
        samples = float(values["Number of samples"])
        if (
            not math.isfinite(samples)
            or samples <= 0
            or any(not math.isfinite(value) and value != -math.inf for value in (peak, rms))
        ):
            raise ValueError("audio levels are not finite")
        return {
            "status": "measured",
            "peak_dbfs": None if peak == -math.inf else peak,
            "rms_dbfs": None if rms == -math.inf else rms,
            "silent": peak == -math.inf,
            "reaches_full_scale": peak >= -0.001,
        }
    except (OSError, subprocess.SubprocessError, KeyError, ValueError):
        # A missing/slow analyzer must not masquerade as a clean signal or change existing
        # metadata-only upload compatibility. Rendering still validates the audio asset.
        return {"status": "unavailable"}


def inspect_soundtrack(comfy_input: Path, name: str) -> dict[str, object]:
    """Bind an existing Comfy audio input to exact bytes and a usable duration."""
    path = _asset_path(comfy_input.resolve(), name, "audio")
    before = path.stat()
    if not path.is_file() or not 0 < before.st_size <= 256 * 1024 * 1024:
        raise ValueError("soundtrack must be a regular audio file of at most 256 MiB")
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                "wav,flac,mp3,ogg,mov,matroska,webm,aac",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        if not info.get("streams") or not math.isfinite(duration) or duration <= 0:
            raise ValueError("soundtrack must contain a decodable audio stream")
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("could not inspect the uploaded soundtrack") from error
    levels = _soundtrack_levels(path)
    digest = file_digest(path)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("soundtrack changed while reading; choose the file again")
    duration_ms = math.floor(duration * 1000)
    if duration_ms <= 0:
        raise ValueError("soundtrack is too short")
    return {"path": name, "sha256": digest, "duration_ms": duration_ms, "levels": levels}
