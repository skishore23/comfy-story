from pathlib import Path

import pytest

from duet.duetx.config import Method, Selector, load_config


def test_decision1_freezes_axes() -> None:
    config = load_config(Path("configs/duetx/decision1.yaml"))
    assert config.experiment.history_items == 8
    assert config.experiment.protected_exceptions == 2
    assert tuple(config.experiment.methods) == tuple(Method)
    assert tuple(config.experiment.selectors) == tuple(Selector)


def test_unknown_config_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("format: duet-x-ltx-decision1-v1\nunknown: true\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)


def test_fingerprint_is_stable() -> None:
    config = load_config(Path("configs/duetx/decision1.yaml"))
    assert config.fingerprint() == load_config(Path("configs/duetx/decision1.yaml")).fingerprint()
