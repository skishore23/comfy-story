from __future__ import annotations

from typing import cast

import pytest
import torch

from duet.duetx.contracts import tensor_sha256
from duet.duetx.h3_reference_conditioning import (
    extract_visual_references,
    native_visual_rows,
    replace_visual_references,
)
from duet.duetx.h3_reference_contracts import (
    H3CompiledBlock,
    H3CompiledContext,
    H3CompileReceipt,
    H3ReferenceBudget,
    H3ReferenceKind,
    H3ReferenceMethod,
    H3SelectedSource,
)


def _conditioning() -> list[list[object]]:
    return [
        [
            torch.zeros(1, 4, 8),
            {
                "minimax_token_tags": torch.ones(1, dtype=torch.long),
                "minimax_refs": [
                    {
                        "kind": "image",
                        "latent_h": 48,
                        "latent_w": 84,
                        "latent": torch.zeros(1, 24, 1, 48, 84),
                    },
                    {
                        "kind": "video",
                        "latent_t": 37,
                        "latent_h": 48,
                        "latent_w": 84,
                        "ref_audio_t": 0,
                        "latent": torch.zeros(1, 24, 37, 48, 84),
                        "audio_latent": None,
                    },
                ],
            },
        ]
    ]


def _sources() -> tuple[H3SelectedSource, ...]:
    return (
        H3SelectedSource("portrait", H3ReferenceKind.IMAGE, 0, True),
        H3SelectedSource("drive", H3ReferenceKind.VIDEO, 1, False),
    )


def _compiled_context() -> H3CompiledContext:
    latent = torch.ones(1, 24, 1, 48, 84)
    block = H3CompiledBlock(H3ReferenceKind.IMAGE, latent, ("portrait",), True)
    receipt = H3CompileReceipt(
        "duet-x-h3-reference-compile-receipt-v1",
        H3ReferenceMethod.EXCEPTIONS_ONLY,
        2,
        2,
        38_304,
        1008,
        ("a" * 64, "b" * 64),
        (tensor_sha256(latent),),
        None,
        "bypass",
        0,
        0,
    )
    return H3CompiledContext((block,), receipt, H3ReferenceBudget(1008, 1)).validate()


def test_extracts_image_and_silent_video_in_native_order() -> None:
    references = extract_visual_references(_conditioning(), _sources())

    assert tuple(reference.source_id for reference in references) == ("portrait", "drive")
    assert sum(reference.visual_rows for reference in references) == 1008 + 37_296
    assert native_visual_rows(_conditioning()) == 38_304


def test_replacement_does_not_mutate_input_or_text_conditioning() -> None:
    original = _conditioning()
    context = _compiled_context()
    replaced = replace_visual_references(original, context)
    replaced_metadata = cast(dict[str, object], replaced[0][1])
    original_metadata = cast(dict[str, object], original[0][1])

    assert replaced is not original
    assert replaced[0][0] is original[0][0]
    assert replaced_metadata["minimax_token_tags"] is original_metadata["minimax_token_tags"]
    assert original[0][1]["minimax_refs"][0]["kind"] == "image"  # type: ignore[index]
    assert replaced[0][1]["minimax_refs"][0]["latent"] is context.blocks[0].latent  # type: ignore[index]


def test_extraction_rejects_audio_and_audio_bearing_video() -> None:
    audio = _conditioning()
    audio[0][1]["minimax_refs"] = [{"kind": "audio"}]  # type: ignore[index]
    with pytest.raises(ValueError, match="audio references"):
        extract_visual_references(
            audio,
            (H3SelectedSource("voice", H3ReferenceKind.IMAGE, 0, False),),
        )

    video = _conditioning()
    block = video[0][1]["minimax_refs"][1]  # type: ignore[index]
    block["kind"] = "video_audio"
    block["audio_latent"] = torch.zeros(1, 32, 2, 10)
    with pytest.raises(ValueError, match="audio-bearing"):
        extract_visual_references(video, _sources())


def test_extraction_rejects_unknown_fields_and_declared_shape_drift() -> None:
    unknown = _conditioning()
    unknown[0][1]["minimax_refs"][0]["surprise"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="missing or unknown"):
        extract_visual_references(unknown, _sources())

    drifted = _conditioning()
    drifted[0][1]["minimax_refs"][0]["latent_h"] = 40  # type: ignore[index]
    with pytest.raises(ValueError, match="latent_h"):
        extract_visual_references(drifted, _sources())


def test_extraction_rejects_manifest_mismatch_and_inconsistent_branches() -> None:
    with pytest.raises(ValueError, match="manifest does not match"):
        extract_visual_references(_conditioning(), _sources()[:1])

    branches = _conditioning()
    branches.append([branches[0][0], dict(cast(dict[str, object], branches[0][1]))])
    second_refs = [dict(block) for block in branches[1][1]["minimax_refs"]]  # type: ignore[index]
    second_refs[0]["latent"] = torch.ones(1, 24, 1, 48, 84)
    branches[1][1]["minimax_refs"] = second_refs  # type: ignore[index]
    with pytest.raises(ValueError, match="inconsistent"):
        extract_visual_references(branches, _sources())
