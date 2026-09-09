"""Fail-closed decoded-video probing for production media evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Self, cast

from .secure_filesystem import SecureFilesystemRoot


class _DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate ffprobe JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid ffprobe JSON constant: {value}")


@dataclass(frozen=True, slots=True)
class DecodedMediaFacts:
    """Facts measured by fully decoding/counting the first video stream."""

    width: int
    height: int
    frames: int
    fps: float
    duration_seconds: float

    def validate(self) -> Self:
        for integer_value, name in (
            (self.width, "decoded media width"),
            (self.height, "decoded media height"),
            (self.frames, "decoded media frame count"),
        ):
            if type(integer_value) is not int or integer_value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for float_value, name in (
            (self.fps, "decoded media FPS"),
            (self.duration_seconds, "decoded media duration"),
        ):
            if type(float_value) is not float or not math.isfinite(float_value) or float_value <= 0:
                raise ValueError(f"{name} must be a positive finite float")
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "frames": self.frames,
            "height": self.height,
            "width": self.width,
        }


def _strict_positive_int(value: object, name: str) -> int:
    if type(value) is int and value > 0:
        return value
    if type(value) is str and value.isascii() and value.isdigit() and int(value) > 0:
        return int(value)
    raise ValueError(f"ffprobe {name} must be a positive integer")


def _strict_positive_float(value: object, name: str) -> float:
    if type(value) not in {str, int, float}:
        raise ValueError(f"ffprobe {name} must be numeric")
    try:
        result = float(cast(str | int | float, value))
    except ValueError as error:
        raise ValueError(f"ffprobe {name} must be numeric") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"ffprobe {name} must be positive and finite")
    return result


def _regular_sha256(path: Path, name: str) -> str:
    try:
        parent = path.parent.resolve(strict=True)
        if parent != path.parent:
            raise ValueError(f"{name} parent must already be resolved")
        with SecureFilesystemRoot.open_native(parent) as custody:
            return custody.describe_regular(path.name).sha256
    except ValueError as error:
        raise ValueError(f"{name} must be a custody-checked regular file") from error


def _descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(descriptor, 1024 * 1024, offset):
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _same_open_file(path: Path, metadata: os.stat_result) -> bool:
    try:
        observed = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISREG(observed.st_mode)
        and observed.st_dev == metadata.st_dev
        and observed.st_ino == metadata.st_ino
        and observed.st_mode == metadata.st_mode
        and observed.st_nlink == metadata.st_nlink
        and observed.st_size == metadata.st_size
    )


def _linux_descriptor_execution_available() -> bool:
    return sys.platform == "linux"


def _run_linux_descriptor_stable_media_tool(
    *,
    executable_path: Path,
    executable_sha256: str,
    media_path: Path,
    arguments: Callable[[str], tuple[str, ...]],
) -> subprocess.CompletedProcess[bytes]:
    """Execute exact Linux tool/media inodes through inherited procfs descriptors."""
    if not _linux_descriptor_execution_available():
        raise ValueError("descriptor-stable media-tool execution requires Linux")
    if (
        not isinstance(executable_path, Path)
        or not executable_path.is_absolute()
        or not isinstance(media_path, Path)
        or not media_path.is_absolute()
    ):
        raise ValueError("descriptor-stable media-tool paths must be absolute")
    if (
        type(executable_sha256) is not str
        or len(executable_sha256) != 64
        or any(character not in "0123456789abcdef" for character in executable_sha256)
    ):
        raise ValueError("descriptor-stable executable SHA-256 is malformed")
    if not callable(arguments):
        raise ValueError("descriptor-stable media-tool arguments must be callable")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        executable_descriptor = os.open(executable_path, flags)
    except OSError as error:
        raise ValueError("descriptor-stable executable is missing or unsafe") from error
    try:
        executable_metadata = os.fstat(executable_descriptor)
        if (
            not stat.S_ISREG(executable_metadata.st_mode)
            or executable_metadata.st_nlink != 1
            or stat.S_IMODE(executable_metadata.st_mode) & 0o111 == 0
        ):
            raise ValueError("descriptor-stable executable must be executable and single-link")
        if _descriptor_sha256(executable_descriptor) != executable_sha256:
            raise ValueError("descriptor-stable executable differs from the frozen bytes")
        try:
            media_descriptor = os.open(media_path, flags)
        except OSError as error:
            raise ValueError("descriptor-stable media is missing or unsafe") from error
        try:
            media_metadata = os.fstat(media_descriptor)
            if not stat.S_ISREG(media_metadata.st_mode) or media_metadata.st_nlink != 1:
                raise ValueError("descriptor-stable media must be a single-link regular file")
            media_sha256 = _descriptor_sha256(media_descriptor)
            executable_fd_path = f"/proc/self/fd/{executable_descriptor}"
            media_fd_path = f"/proc/self/fd/{media_descriptor}"
            suffix = arguments(media_fd_path)
            if type(suffix) is not tuple or not all(type(value) is str for value in suffix):
                raise ValueError("descriptor-stable media-tool arguments are malformed")
            completed = subprocess.run(
                (executable_fd_path, *suffix),
                check=True,
                capture_output=True,
                pass_fds=(executable_descriptor, media_descriptor),
            )
            if (
                _descriptor_sha256(executable_descriptor) != executable_sha256
                or _descriptor_sha256(media_descriptor) != media_sha256
            ):
                raise ValueError(
                    "executable or media bytes changed during descriptor-stable execution"
                )
            if not _same_open_file(executable_path, executable_metadata) or not _same_open_file(
                media_path, media_metadata
            ):
                raise ValueError(
                    "executable or media path changed during descriptor-stable execution"
                )
            return completed
        finally:
            os.close(media_descriptor)
    finally:
        os.close(executable_descriptor)


def probe_decoded_video(
    path: Path,
    *,
    ffprobe_path: Path | None = None,
    ffprobe_sha256: str | None = None,
    descriptor_stable: bool = False,
) -> DecodedMediaFacts:
    """Decode/count a non-symlink media file through ffprobe using an already-open fd."""
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise ValueError("decoded media path must be an absolute non-symlink regular file")
    if type(descriptor_stable) is not bool:
        raise ValueError("descriptor-stable ffprobe mode must be boolean")
    if descriptor_stable:
        if ffprobe_path is None or ffprobe_sha256 is None:
            raise ValueError("descriptor-stable ffprobe requires an exact path and digest")
        try:
            completed = _run_linux_descriptor_stable_media_tool(
                executable_path=ffprobe_path,
                executable_sha256=ffprobe_sha256,
                media_path=path,
                arguments=lambda media_fd_path: (
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,nb_read_frames,nb_frames,duration:format=duration",
                    "-of",
                    "json",
                    media_fd_path,
                ),
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("media bytes could not be decoded by ffprobe") from error
    else:
        completed = _probe_decoded_video_by_path(path, ffprobe_path, ffprobe_sha256)
    return _decoded_media_facts(completed)


def _probe_decoded_video_by_path(
    path: Path, ffprobe_path: Path | None, ffprobe_sha256: str | None
) -> subprocess.CompletedProcess[bytes]:
    if ffprobe_path is not None:
        try:
            if (
                not ffprobe_path.is_absolute()
                or ffprobe_path.is_symlink()
                or not ffprobe_path.is_file()
                or ffprobe_sha256 is None
            ):
                raise ValueError("ffprobe path and digest must name the preflighted executable")
            executable_digest = _regular_sha256(ffprobe_path, "ffprobe executable")
            if executable_digest != ffprobe_sha256:
                raise ValueError("ffprobe executable differs from the preflight bytes")
            media_digest = _regular_sha256(path, "decoded media")
            command = (
                str(ffprobe_path),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames,nb_frames,duration:format=duration",
                "-of",
                "json",
                str(path),
            )
            completed = subprocess.run(command, check=True, capture_output=True)
            if (
                _regular_sha256(ffprobe_path, "ffprobe executable") != ffprobe_sha256
                or _regular_sha256(path, "decoded media") != media_digest
            ):
                raise ValueError("ffprobe executable or decoded media changed during probing")
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("media bytes could not be decoded by ffprobe") from error
    else:
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
            )
        except OSError as error:
            raise ValueError(
                "decoded media path must be an absolute non-symlink regular file"
            ) from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("decoded media path must be an absolute non-symlink regular file")
            command = (
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames,nb_frames,duration:format=duration",
                "-of",
                "json",
                f"/dev/fd/{descriptor}",
            )
            completed = subprocess.run(
                command, check=True, capture_output=True, pass_fds=(descriptor,)
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("media bytes could not be decoded by ffprobe") from error
        finally:
            os.close(descriptor)

    return completed


def _decoded_media_facts(completed: subprocess.CompletedProcess[bytes]) -> DecodedMediaFacts:
    try:
        payload = json.loads(
            completed.stdout,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (_DuplicateKeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ffprobe returned malformed strict JSON") from error
    if type(payload) is not dict or set(payload) not in (
        {"format", "streams"},
        {"format", "programs", "streams"},
    ):
        raise ValueError("ffprobe output fields do not match the decoded-video schema")
    if "programs" in payload and (type(payload["programs"]) is not list or payload["programs"]):
        raise ValueError("ffprobe programs must be an exactly empty array")
    streams = cast(dict[str, object], payload)["streams"]
    format_row = cast(dict[str, object], payload)["format"]
    if type(streams) is not list or len(streams) != 1 or type(streams[0]) is not dict:
        raise ValueError("media must decode to exactly one selected video stream")
    if type(format_row) is not dict:
        raise ValueError("ffprobe format facts are missing")
    stream = cast(dict[str, object], streams[0])
    if set(stream) - {
        "avg_frame_rate",
        "duration",
        "height",
        "nb_frames",
        "nb_read_frames",
        "width",
    }:
        raise ValueError("ffprobe video stream contains unknown fields")
    rate = stream.get("avg_frame_rate")
    if type(rate) is not str:
        raise ValueError("ffprobe average frame rate is missing")
    try:
        fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("ffprobe average frame rate is invalid") from error
    frame_value = stream.get("nb_read_frames", stream.get("nb_frames"))
    duration_value = stream.get("duration", cast(dict[str, object], format_row).get("duration"))
    return DecodedMediaFacts(
        _strict_positive_int(stream.get("width"), "width"),
        _strict_positive_int(stream.get("height"), "height"),
        _strict_positive_int(frame_value, "decoded frame count"),
        _strict_positive_float(fps, "average frame rate"),
        _strict_positive_float(duration_value, "duration"),
    ).validate()


__all__ = ("DecodedMediaFacts", "probe_decoded_video")
