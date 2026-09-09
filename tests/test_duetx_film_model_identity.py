from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from duet.duetx import film_audit_cli as audit
from duet.duetx.film_export import file_digest
from duet.duetx.story_contracts import canonical_story_json


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights-one")
    (model / "config.json").write_bytes(b"{}")
    return model


def test_repeated_identity_preserves_historical_digest_and_reads_actual_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    expected = hashlib.sha256(
        canonical_story_json(
            {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model.iterdir()}
        )
    ).hexdigest()
    reads: list[Path] = []
    original = file_digest

    def observed(path: Path) -> str:
        reads.append(path)
        return original(path)

    monkeypatch.setattr(audit, "file_digest", observed)
    assert audit._model_identity(model) == expected
    assert audit._model_identity(model) == expected
    assert sorted(p.name for p in reads) == [
        "config.json",
        "config.json",
        "model.safetensors",
        "model.safetensors",
    ]


def test_same_size_edit_with_restored_mtime_changes_identity(tmp_path: Path) -> None:
    model = _model(tmp_path)
    before = audit._model_identity(model)
    weights = model / "model.safetensors"
    info = weights.stat()
    weights.write_bytes(b"weights-two")
    os.utime(weights, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert audit._model_identity(model) != before


def test_replacement_and_model_file_inventory_changes_invalidate(tmp_path: Path) -> None:
    model = _model(tmp_path)
    first = audit._model_identity(model)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"weights-two")
    replacement.replace(model / "model.safetensors")
    second = audit._model_identity(model)
    assert second != first
    (model / "template.jinja").write_text("template")
    third = audit._model_identity(model)
    assert third != second
    (model / "template.jinja").unlink()
    assert audit._model_identity(model) == second
    (model / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="safetensors"):
        audit._model_identity(model)


def test_symlink_retarget_changes_identity(tmp_path: Path) -> None:
    model = _model(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    link = model / "extra.safetensors"
    link.symlink_to(a)
    before = audit._model_identity(model)
    link.unlink()
    link.symlink_to(b)
    assert audit._model_identity(model) != before


@pytest.mark.parametrize("change", ["hashed_file", "previous_file", "new_file"])
def test_mutation_during_hashing_is_rejected_and_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    model = _model(tmp_path)
    original = file_digest
    changed = False

    def mutate(path: Path) -> str:
        nonlocal changed
        result = original(path)
        if path.suffix == ".safetensors" and not changed:
            changed = True
            target = {
                "hashed_file": path,
                "previous_file": model / "config.json",
                "new_file": model / "new.json",
            }[change]
            target.write_bytes(b"changed")
        return result

    monkeypatch.setattr(audit, "file_digest", mutate)
    with pytest.raises(ValueError, match="changed during identity verification"):
        audit._model_identity(model)
    expected = hashlib.sha256(
        canonical_story_json(
            {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model.iterdir()}
        )
    ).hexdigest()
    assert audit._model_identity(model) == expected


def test_no_weights_or_missing_directory_still_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing local directory"):
        audit._model_identity(tmp_path / "missing")
    with pytest.raises(ValueError, match="safetensors"):
        audit._model_identity(tmp_path)


def test_rehashes_even_when_metadata_does_not_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    monkeypatch.setattr(audit, "_model_file_signature", lambda _: (1, 2, 3, 4, 5))
    reads: list[Path] = []

    def observed(path: Path) -> str:
        reads.append(path)
        return file_digest(path)

    monkeypatch.setattr(audit, "file_digest", observed)
    first = audit._model_identity(model)
    assert audit._model_identity(model) == first
    assert len(reads) == 4
    (model / "model.safetensors").write_bytes(b"weights-two")
    assert audit._model_identity(model) != first
    assert len(reads) == 6
