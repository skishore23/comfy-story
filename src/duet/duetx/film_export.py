"""Conform accepted Duet takes and authored sound/text to an exact film timeline."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from duet.duetx.film_plan import (
    AcceptedTake,
    FilmPlan,
    RenderedTake,
    accepted_state,
    subtitle_srt,
    validate_rendered_takes,
)
from duet.duetx.story_contracts import canonical_story_json


@dataclass(frozen=True, slots=True)
class FilmAudio:
    """An authored audio asset placed on the film timeline; speech carries its cue ID."""

    path: Path
    sha256: str
    start_ms: int
    duration_ms: int
    gain: float = 1.0
    cue_id: str = ""


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _run(arguments: list[str], directory: Path) -> None:
    subprocess.run(arguments, cwd=directory, check=True, capture_output=True)


def _duration(path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return round(float(payload["format"]["duration"]) * 1000)


def _speech_duck_expression(
    audio: tuple[FilmAudio, ...], windows: tuple[tuple[int, int], ...] = ()
) -> str:
    """Lower background gain over actual speech assets with short, click-free ramps."""
    windows += tuple(
        (track.start_ms, track.start_ms + track.duration_ms) for track in audio if track.cue_id
    )
    envelopes = [
        f"min(max((t+0.15-{start / 1000})/0.15,0),min(max(({end / 1000}+0.35-t)/0.35,0),1))"
        for start, end in windows
    ]
    return "1-0.75*min(1," + "+".join(envelopes) + ")"


def _audio_channel_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams or type(streams[0].get("channels")) is not int or streams[0]["channels"] < 1:
        raise ValueError("speech asset must have a known audio channel count")
    return int(streams[0]["channels"])


def export_film(
    plan: FilmPlan,
    takes: tuple[AcceptedTake, ...],
    assets: Mapping[str, Path],
    directory: Path,
    *,
    audio: tuple[FilmAudio, ...] = (),
    fps: int = 24,
    burn_subtitles: bool = True,
    prioritize_speech: bool = True,
    generated_audio: bool = False,
    output_size: tuple[int, int] = (1344, 768),
) -> Path:
    """Export only a fully accepted film. No generated soundtrack is silently carried through.

    Source bytes, source trim ranges, exact frame durations, speech coverage and final duration
    are checked. Creative and linguistic review must precede acceptance; these checks cannot
    prove that a renderer depicted a planned event or pronounced a line correctly.
    """
    accepted_state(plan, takes)
    if len(takes) != len(plan.shots):
        raise ValueError("film export requires an accepted take for every shot")
    return _export_timeline(
        plan,
        takes,
        assets,
        directory,
        audio=audio,
        fps=fps,
        burn_subtitles=burn_subtitles,
        prioritize_speech=prioritize_speech,
        generated_audio=generated_audio,
        output_size=output_size,
        reviewed=True,
    )


def export_film_draft(
    plan: FilmPlan,
    takes: tuple[RenderedTake, ...],
    assets: Mapping[str, Path],
    directory: Path,
    *,
    audio: tuple[FilmAudio, ...] = (),
    fps: int = 24,
    burn_subtitles: bool = True,
    prioritize_speech: bool = True,
    generated_audio: bool = False,
    output_size: tuple[int, int] = (1344, 768),
) -> Path:
    """Automatically conform a complete first pass; never manufacture creative approval.

    The same source, timing, speech and caption checks apply as for a reviewed export.
    An explicit unreviewed receipt and rendered-takes sidecar preserve the distinction.
    """
    validate_rendered_takes(plan, takes)
    return _export_timeline(
        plan,
        takes,
        assets,
        directory,
        audio=audio,
        fps=fps,
        burn_subtitles=burn_subtitles,
        prioritize_speech=prioritize_speech,
        generated_audio=generated_audio,
        output_size=output_size,
        reviewed=False,
    )


def _export_timeline(
    plan: FilmPlan,
    takes: tuple[AcceptedTake | RenderedTake, ...],
    assets: Mapping[str, Path],
    directory: Path,
    *,
    audio: tuple[FilmAudio, ...],
    fps: int,
    burn_subtitles: bool,
    prioritize_speech: bool,
    reviewed: bool,
    generated_audio: bool,
    output_size: tuple[int, int],
) -> Path:
    if type(generated_audio) is not bool:
        raise ValueError("generated_audio must be a boolean")
    if (
        not isinstance(output_size, tuple)
        or len(output_size) != 2
        or any(type(n) is not int or n < 64 or n > 4096 or n % 2 for n in output_size)
    ):
        raise ValueError("film output dimensions must be even integers from 64 to 4096")
    width, height = output_size
    if any(shot.text for shot in plan.shots):
        raise ValueError("visible text overlays require an explicit typography compositor")
    if type(fps) is not int or not 1 <= fps <= 120:
        raise ValueError("film fps must be an integer from 1 to 120")
    if type(prioritize_speech) is not bool:
        raise ValueError("prioritize_speech must be a boolean")
    if any(shot.duration_ms * fps % 1000 for shot in plan.shots):
        raise ValueError("shot duration must be an exact number of output frames")
    sources: list[Path] = []
    for shot, take in zip(plan.shots, takes, strict=True):
        source = assets.get(take.video_sha256)
        if source is None or file_digest(source) != take.video_sha256:
            raise ValueError("accepted film source is missing or its SHA-256 changed")
        if _duration(source) < take.source_in_ms + shot.duration_ms:
            raise ValueError("actual source duration is shorter than the accepted trim")
        if generated_audio:
            _audio_channel_count(source)
        sources.append(source.resolve())
    expected_cues: dict[str, tuple[int, int]] = {}
    offset = 0
    for shot in plan.shots:
        for cue in shot.dialogue:
            expected_cues[cue.cue_id] = (offset + cue.start_ms, offset + cue.end_ms)
        offset += shot.duration_ms
    spoken: set[str] = set()
    for track in audio:
        if file_digest(track.path) != track.sha256:
            raise ValueError("authored film audio SHA-256 changed")
        if (
            type(track.start_ms) is not int
            or type(track.duration_ms) is not int
            or track.start_ms < 0
            or track.duration_ms <= 0
            or track.start_ms + track.duration_ms > plan.target_duration_ms
            or not 0 < track.gain <= 10
        ):
            raise ValueError("audio must fit the film timeline with a bounded gain")
        if _duration(track.path) < track.duration_ms:
            raise ValueError("audio asset is shorter than its declared use")
        if track.cue_id:
            if track.cue_id not in expected_cues or track.cue_id in spoken:
                raise ValueError("speech requires a unique planned cue")
            start, end = expected_cues[track.cue_id]
            if track.start_ms != start or track.start_ms + track.duration_ms > end:
                raise ValueError("speech must fit its exact approved cue window")
            spoken.add(track.cue_id)
    if generated_audio and spoken:
        raise ValueError("Generated dialogue cannot also have authored speech tracks")
    if not generated_audio and spoken != set(expected_cues):
        raise ValueError("every dialogue cue needs an authored audio asset")
    if burn_subtitles and expected_cues:
        capabilities = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            check=True,
            capture_output=True,
            text=True,
        )
        if not any("subtitles" in line.split() for line in capabilities.stdout.splitlines()):
            raise ValueError(
                "burned captions require an FFmpeg build with the libass subtitles filter"
            )
    if directory.exists():
        raise ValueError("use a new export directory to preserve existing work")
    directory.mkdir(parents=True)
    directory = directory.resolve()
    (directory / "film-plan.json").write_bytes(canonical_story_json(plan))
    takes_filename = "accepted-takes.json" if reviewed else "rendered-takes.json"
    (directory / takes_filename).write_bytes(canonical_story_json(takes))
    (directory / "narration.srt").write_text(subtitle_srt(plan), encoding="utf-8")
    for index, (shot, take, source) in enumerate(zip(plan.shots, takes, sources, strict=True)):
        _run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-ss",
                str(take.source_in_ms / 1000),
                "-i",
                str(source),
                "-an",
                "-vf",
                f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
                "-frames:v",
                str(shot.duration_ms * fps // 1000),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                f"shot-{index:04}.mp4",
            ],
            directory,
        )
    (directory / "concat.txt").write_text(
        "".join(f"file 'shot-{index:04}.mp4'\n" for index in range(len(takes)))
    )
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            "concat.txt",
            "-c",
            "copy",
            "picture.mp4",
        ],
        directory,
    )
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", "picture.mp4"]
    filters: list[str] = []
    native_windows = tuple(expected_cues.values()) if generated_audio else ()
    speech_mix = prioritize_speech and bool(spoken or native_windows)
    for index, track in enumerate(audio, 1):
        audio_path = track.path.resolve()
        if speech_mix and track.cue_id:
            # Normalize each finite source separately before the timeline mixing graph.
            audio_path = directory / f"speech-{index:04}.wav"
            _run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(track.path.resolve()),
                    "-af",
                    f"atrim=duration={track.duration_ms / 1000},"
                    "asetpts=PTS-STARTPTS,loudnorm=I=-16:LRA=7:TP=-2",
                    "-ar",
                    "48000",
                    "-ac",
                    str(_audio_channel_count(track.path)),
                    "-c:a",
                    "pcm_f32le",
                    str(audio_path),
                ],
                directory,
            )
        command.extend(["-i", str(audio_path)])
        # Duck after adelay so the envelope uses film time, including a bed's own offset.
        duck = (
            f",volume='{_speech_duck_expression(audio, native_windows)}':eval=frame"
            if speech_mix and not track.cue_id
            else ""
        )
        filters.append(
            f"[{index}:a]atrim=duration={track.duration_ms / 1000},asetpts=PTS-STARTPTS,"
            f"volume={track.gain},adelay={track.start_ms}:all=1{duck}[a{index}]"
        )
    audio_count = len(audio)
    if generated_audio:
        offset_ms = 0
        for shot, take, source in zip(plan.shots, takes, sources, strict=True):
            audio_count += 1
            command.extend(["-i", str(source)])
            # Decode native audio directly onto the timeline. Concatenating separately
            # AAC-encoded clips accumulates encoder padding and shifts picture cuts.
            filters.append(
                f"[{audio_count}:a]atrim=start={take.source_in_ms / 1000}:"
                f"duration={shot.duration_ms / 1000},asetpts=PTS-STARTPTS,"
                f"apad,atrim=duration={shot.duration_ms / 1000},"
                f"adelay={offset_ms}:all=1[a{audio_count}]"
            )
            offset_ms += shot.duration_ms
    if audio_count:
        inputs = "".join(f"[a{index}]" for index in range(1, audio_count + 1))
        filters.append(
            inputs + f"amix=inputs={audio_count}:normalize=0,apad,"
            f"atrim=duration={plan.target_duration_ms / 1000},"
            "loudnorm=I=-14:LRA=9:TP=-1.5[mix]"
        )
        command.extend(["-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[mix]"])
    else:
        command.extend(["-map", "0:v", "-an"])
    if burn_subtitles and expected_cues:
        command.extend(
            [
                "-vf",
                "subtitles=narration.srt:force_style='FontSize=19,Outline=1,Shadow=0,MarginV=28'",
            ]
        )
    command.extend(
        [
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-t",
            str(plan.target_duration_ms / 1000),
            "film.mp4",
        ]
    )
    _run(command, directory)
    output = directory / "film.mp4"
    measured = _duration(output)
    if abs(measured - plan.target_duration_ms) > 1:
        raise ValueError(f"export duration mismatch: measured {measured} ms")
    (directory / "export-receipt.json").write_bytes(
        canonical_story_json(
            {
                "film_sha256": file_digest(output),
                "duration_ms": measured,
                "fps": fps,
                "plan_sha256": hashlib.sha256(canonical_story_json(plan)).hexdigest(),
                "review_status": "reviewed" if reviewed else "unreviewed",
                "accepted_take_sha256s": [take.digest for take in takes] if reviewed else [],
                "rendered_take_sha256s": [] if reviewed else [take.digest for take in takes],
                "audio_sha256s": [track.sha256 for track in audio],
                "speech_mix": {
                    "enabled": speech_mix,
                    "speech_lufs": -16 if speech_mix else None,
                    "background_gain_during_speech": 0.25 if speech_mix else 1.0,
                    "attack_ms": 150 if speech_mix else 0,
                    "release_ms": 350 if speech_mix else 0,
                },
                "subtitles_burned": burn_subtitles and bool(expected_cues),
                **(
                    {"generated_audio": True, "generated_dialogue_verified": False}
                    if generated_audio
                    else {}
                ),
                **({"output_size": output_size} if output_size != (1344, 768) else {}),
            }
        )
    )
    shutil.copyfile(directory / "narration.srt", directory / "film.srt")
    return output
