from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from duet.duetx.film_export import export_film, file_digest
from duet.duetx.film_plan import AcceptedTake, FilmPlan, FilmShot
from duet.duetx.story_contracts import canonical_story_json


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required for actual film conform")
def test_actual_export_has_exact_runtime_and_preserves_accepted_asset_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=24:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    plan = FilmPlan(
        "export-test",
        "A short film",
        1000,
        (),
        (
            FilmShot("one", 500, "Establish", "Hold", ()),
            FilmShot("two", 500, "Resolve", "Hold", (), depends_on=("one",)),
        ),
    ).validate()
    digest = file_digest(source)
    parent = hashlib.sha256(canonical_story_json(plan.initial_facts)).hexdigest()
    first = AcceptedTake(
        "one", plan.shots[0].digest, "a" * 64, digest, parent, (), "test", "Blue frame", 1000
    )
    second = AcceptedTake(
        "two",
        plan.shots[1].digest,
        "b" * 64,
        digest,
        first.digest,
        (),
        "test",
        "Blue frame",
        1000,
        500,
    )
    output = export_film(plan, (first, second), {digest: source}, tmp_path / "export")
    receipt = json.loads((output.parent / "export-receipt.json").read_text())
    assert receipt["duration_ms"] == 1000
    assert receipt["film_sha256"] == file_digest(output)
    assert receipt["accepted_take_sha256s"] == [first.digest, second.digest]
    source.write_bytes(b"changed asset")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        export_film(plan, (first, second), {digest: source}, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_export_refuses_incomplete_review_before_writing_files(tmp_path: Path) -> None:
    plan = FilmPlan(
        "incomplete", "Unreviewed", 1000, (), (FilmShot("one", 1000, "Setup", "Wait", ()),)
    )
    with pytest.raises(ValueError, match="every shot"):
        export_film(plan, (), {}, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required for actual film conform")
def test_spoken_export_checks_cue_coverage_and_uses_authored_caption(tmp_path: Path) -> None:
    from dataclasses import replace

    from duet.duetx.film_export import FilmAudio
    from duet.duetx.film_plan import FilmCue

    source = tmp_path / "source.mp4"
    voice = tmp_path / "voice.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=24:d=1",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5",
            str(voice),
        ],
        check=True,
    )
    plan = FilmPlan(
        "spoken",
        "Exact language",
        1000,
        (),
        (
            FilmShot(
                "one",
                1000,
                "Resolve",
                "Hold",
                (),
                dialogue=(FilmCue("line", 100, 900, "The light came back.", "en", "Narrator"),),
            ),
        ),
    ).validate()
    digest = file_digest(source)
    take = AcceptedTake(
        "one",
        plan.shots[0].digest,
        "a" * 64,
        digest,
        hashlib.sha256(canonical_story_json(plan.initial_facts)).hexdigest(),
        (),
        "test",
        "Fixture only; not a linguistic evaluation",
        1000,
    )
    audio = FilmAudio(voice, file_digest(voice), 100, 500, cue_id="line")
    with pytest.raises(ValueError, match="every dialogue cue"):
        export_film(plan, (take,), {digest: source}, tmp_path / "missing")
    with pytest.raises(ValueError, match="exact approved cue window"):
        export_film(
            plan,
            (take,),
            {digest: source},
            tmp_path / "mistimed",
            audio=(replace(audio, start_ms=400),),
        )
    with pytest.raises(ValueError, match="unique planned cue"):
        export_film(
            plan,
            (take,),
            {digest: source},
            tmp_path / "duplicate",
            audio=(audio, audio),
        )
    filters = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"], check=True, capture_output=True, text=True
    ).stdout
    if not any("subtitles" in line.split() for line in filters.splitlines()):
        with pytest.raises(ValueError, match="libass subtitles filter"):
            export_film(plan, (take,), {digest: source}, tmp_path / "no-libass", audio=(audio,))
        assert not (tmp_path / "no-libass").exists()
        pytest.skip("This FFmpeg build lacks libass; caption rendering is exercised on Forge1")
    output = export_film(
        plan,
        (take,),
        {digest: source},
        tmp_path / "complete",
        audio=(audio,),
    )
    assert "The light came back." in (output.parent / "film.srt").read_text()
    receipt = json.loads((output.parent / "export-receipt.json").read_text())
    assert receipt["subtitles_burned"] is True
    assert receipt["audio_sha256s"] == [audio.sha256]
    assert receipt["duration_ms"] == 1000


def test_export_does_not_silently_drop_planned_visible_text(tmp_path: Path) -> None:
    from duet.duetx.film_plan import FilmCue

    plan = FilmPlan(
        "visible-text",
        "Exact writing",
        1000,
        (),
        (
            FilmShot(
                "one",
                1000,
                "Read",
                "Hold",
                (),
                text=(FilmCue("sign", 0, 1000, "Harbor closed", "en"),),
            ),
        ),
    ).validate()
    take = AcceptedTake(
        "one",
        plan.shots[0].digest,
        "a" * 64,
        "b" * 64,
        hashlib.sha256(canonical_story_json(plan.initial_facts)).hexdigest(),
        (),
        "test",
        "Visible text must have an explicit compositor",
        1000,
    )
    with pytest.raises(ValueError, match="typography compositor"):
        export_film(plan, (take,), {}, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required for actual audio mixing")
@pytest.mark.parametrize("prioritize_speech", [False, True])
def test_background_is_ducked_at_film_time_only_when_speech_priority_is_enabled(
    tmp_path: Path, prioritize_speech: bool
) -> None:
    from duet.duetx.film_export import FilmAudio
    from duet.duetx.film_plan import FilmCue

    source, voice, bed = (tmp_path / name for name in ("picture.mp4", "voice.wav", "bed.wav"))
    for signal, destination in [
        ("color=c=blue:s=64x64:r=24:d=3", source),
        ("sine=frequency=440:duration=0.5:sample_rate=24000", voice),
        ("sine=frequency=1000:duration=2.7:sample_rate=48000", bed),
    ]:
        codec = ["-c:a", "pcm_s24le"] if destination == voice else []
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", signal, *codec, str(destination)],
            check=True,
        )
    plan = FilmPlan(
        "speech-mix",
        "Speech must remain audible",
        3000,
        (),
        (
            FilmShot(
                "one",
                3000,
                "Speak",
                "Hold",
                (),
                dialogue=(FilmCue("line", 1200, 1700, "Test tone", "en", "Narrator"),),
            ),
        ),
    ).validate()
    digest = file_digest(source)
    take = AcceptedTake(
        "one",
        plan.shots[0].digest,
        "a" * 64,
        digest,
        hashlib.sha256(canonical_story_json(plan.initial_facts)).hexdigest(),
        (),
        "test",
        "Audio measurement fixture, not language evidence",
        3000,
    )
    output = export_film(
        plan,
        (take,),
        {digest: source},
        tmp_path / "mixed",
        audio=(
            FilmAudio(voice, file_digest(voice), 1200, 500, cue_id="line"),
            FilmAudio(bed, file_digest(bed), 300, 2700),
        ),
        burn_subtitles=False,
        prioritize_speech=prioritize_speech,
    )
    pcm = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(output),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "f32le",
            "-",
        ]
    )
    samples = np.frombuffer(pcm, dtype="<f4")

    def bed_amplitude(start: float) -> float:
        window = samples[round(start * 48000) : round((start + 0.15) * 48000)]
        phase = np.exp(-2j * np.pi * 1000 * np.arange(len(window)) / 48000)
        return float(2 * abs(np.mean(window * phase)))

    before, during, after = (bed_amplitude(at) for at in (0.6, 1.35, 2.4))
    assert before > 0.001
    assert 0.7 < after / before < 1.3
    if prioritize_speech:
        assert during / before < 0.45
    else:
        assert 0.7 < during / before < 1.3
    receipt = json.loads((output.parent / "export-receipt.json").read_text())
    assert receipt["speech_mix"]["enabled"] is prioritize_speech
    if prioritize_speech:
        normalized = output.parent / "speech-0001.wav"
        stream = json.loads(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(normalized)]
            )
        )["streams"][0]
        assert stream["channels"] == 1
        assert stream["sample_rate"] == "48000"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")
@pytest.mark.parametrize("generated", [False, True])
def test_native_audio_and_full_hd_survive_film_conform(tmp_path: Path, generated: bool) -> None:
    source = tmp_path / "native.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=24:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )
    plan = FilmPlan(
        "native",
        "Native sound",
        1000,
        (),
        (
            FilmShot("one", 500, "Setup", "Hold", ()),
            FilmShot("two", 500, "Resolve", "Hold", (), depends_on=("one",)),
        ),
    ).validate()
    digest = file_digest(source)
    parent = hashlib.sha256(canonical_story_json(plan.initial_facts)).hexdigest()
    first = AcceptedTake(
        "one", plan.shots[0].digest, "a" * 64, digest, parent, (), "test", "Fixture", 1000
    )
    second = AcceptedTake(
        "two",
        plan.shots[1].digest,
        "b" * 64,
        digest,
        first.digest,
        (),
        "test",
        "Fixture",
        1000,
        500,
    )
    output = export_film(
        plan,
        (first, second),
        {digest: source},
        tmp_path / "export",
        generated_audio=generated,
        output_size=(1920, 1088),
    )
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(output),
            ]
        )
    )
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"], int(video["nb_frames"])) == (1920, 1088, 24)
    assert any(s["codec_type"] == "audio" for s in probe["streams"]) is generated
    assert abs(float(probe["format"]["duration"]) - 1) < 0.05
    receipt = json.loads((output.parent / "export-receipt.json").read_text())
    if generated:
        assert receipt["generated_dialogue_verified"] is False
        decoded = subprocess.check_output(
            ["ffmpeg", "-v", "error", "-i", str(output), "-vn", "-f", "f32le", "-"]
        )
        assert np.abs(np.frombuffer(decoded, dtype=np.float32)).mean() > 0.001


def test_generated_cue_windows_duck_only_the_soundtrack() -> None:
    from duet.duetx.film_export import _speech_duck_expression

    expression = _speech_duck_expression((), ((500, 1500), (5500, 6500)))
    assert "t+0.15-0.5" in expression
    assert "1.5+0.35-t" in expression
    assert "t+0.15-5.5" in expression
