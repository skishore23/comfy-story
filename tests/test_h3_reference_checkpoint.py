from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from comfy_story.h3_reference_checkpoint import (
    COMPILER_CHECKPOINT_FORMAT,
    export_h3_reference_compiler_checkpoint,
    load_h3_reference_compiler_checkpoint,
)
from comfy_story.h3_reference_contracts import H3ReferenceMethod
from comfy_story.minimax_h3_training import (
    build_minimax_h3_training_modules,
    save_minimax_h3_runtime_checkpoint,
)

_FOUNDATION_SHA256 = "1" * 64
_MODEL_CONFIGURATION_SHA256 = "2" * 64


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parent(path: Path) -> Path:
    save_minimax_h3_runtime_checkpoint(
        path,
        build_minimax_h3_training_modules(),
        foundation_sha256=_FOUNDATION_SHA256,
        model_configuration_sha256=_MODEL_CONFIGURATION_SHA256,
        optimizer_steps=2000,
        training_trace=({"optimizer_step": 2000},),
    )
    return path


def test_export_has_distinct_format_and_retains_parent_identity(tmp_path: Path) -> None:
    source = _write_parent(tmp_path / "story.pt")
    output = tmp_path / "compiler.pt"

    identity = export_h3_reference_compiler_checkpoint(source, output)

    assert identity.format == COMPILER_CHECKPOINT_FORMAT
    assert identity.parent_checkpoint_sha256 == _file_sha256(source)
    assert identity.model_configuration_sha256 == _MODEL_CONFIGURATION_SHA256
    assert identity.latent_channels == 24
    assert identity.operator_size == 16
    assert identity.warm_start_scope == "story-memory-visual-latents"
    assert identity.checkpoint_sha256 == _file_sha256(output)


def test_load_authenticates_and_restores_all_three_compiler_families(tmp_path: Path) -> None:
    source = _write_parent(tmp_path / "story.pt")
    output = tmp_path / "compiler.pt"
    identity = export_h3_reference_compiler_checkpoint(source, output)

    loaded = load_h3_reference_compiler_checkpoint(
        output,
        expected_sha256=identity.checkpoint_sha256,
        expected_model_configuration_sha256=_MODEL_CONFIGURATION_SHA256,
        device=torch.device("cpu"),
    )

    assert loaded.identity == identity
    assert loaded.compressor_for(H3ReferenceMethod.GATED) is loaded.gated
    assert loaded.compressor_for(H3ReferenceMethod.RESAMPLER) is loaded.resampler
    assert loaded.compressor_for(H3ReferenceMethod.DUET) is loaded.duet
    duet_x = loaded.compressor_for(H3ReferenceMethod.DUET_X)
    assert duet_x.method is H3ReferenceMethod.DUET_X
    assert all(not parameter.requires_grad for parameter in duet_x.parameters())


def test_load_rejects_module_hash_drift_even_with_new_file_hash(tmp_path: Path) -> None:
    source = _write_parent(tmp_path / "story.pt")
    output = tmp_path / "compiler.pt"
    export_h3_reference_compiler_checkpoint(source, output)
    payload = torch.load(output, map_location="cpu", weights_only=True)
    payload["modules"]["bridge"]["residual.weight"][0, 0] = 1
    torch.save(payload, output)

    with pytest.raises(ValueError, match="module state SHA-256"):
        load_h3_reference_compiler_checkpoint(
            output,
            expected_sha256=_file_sha256(output),
            expected_model_configuration_sha256=_MODEL_CONFIGURATION_SHA256,
            device=torch.device("cpu"),
        )


def test_load_rejects_file_and_model_identity_mismatch(tmp_path: Path) -> None:
    source = _write_parent(tmp_path / "story.pt")
    output = tmp_path / "compiler.pt"
    identity = export_h3_reference_compiler_checkpoint(source, output)

    with pytest.raises(ValueError, match="content SHA-256"):
        load_h3_reference_compiler_checkpoint(
            output,
            expected_sha256="f" * 64,
            expected_model_configuration_sha256=_MODEL_CONFIGURATION_SHA256,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="model configuration"):
        load_h3_reference_compiler_checkpoint(
            output,
            expected_sha256=identity.checkpoint_sha256,
            expected_model_configuration_sha256="e" * 64,
            device=torch.device("cpu"),
        )


def test_checkpoint_paths_must_be_absolute_regular_files(tmp_path: Path) -> None:
    source = _write_parent(tmp_path / "story.pt")
    with pytest.raises(ValueError, match="absolute"):
        export_h3_reference_compiler_checkpoint(source, Path("compiler.pt"))
    missing = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="absolute regular file"):
        load_h3_reference_compiler_checkpoint(
            missing,
            expected_sha256="a" * 64,
            expected_model_configuration_sha256=_MODEL_CONFIGURATION_SHA256,
            device=torch.device("cpu"),
        )
