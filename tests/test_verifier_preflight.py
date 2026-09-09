from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from comfy_story import film_audit_cli


@pytest.mark.parametrize("kernel_fails", [False, True])
def test_reviewer_exercises_lazy_vision_kernel_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kernel_fails: bool
) -> None:
    limits: list[int] = []
    messages: list[Any] = []

    class Inputs(dict[str, Any]):
        def to(self, device: str) -> Inputs:
            assert device == "cpu"
            return self

    class Model:
        device = "cpu"

        def eval(self) -> Model:
            return self

        def generate(self, **kwargs: Any) -> torch.Tensor:
            limits.append(kwargs["max_new_tokens"])
            if kernel_fails:
                raise RuntimeError("lazy kernel unavailable")
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3]])

    class Processor:
        def apply_chat_template(self, value: Any, **kwargs: Any) -> Inputs:
            messages.append(value)
            return Inputs(input_ids=torch.tensor([[1, 2]]))

        def decode(self, value: Any, **kwargs: Any) -> str:
            return "black"

    def load_model(*args: Any, **kwargs: Any) -> Model:
        assert kwargs["trust_remote_code"] is False
        assert kwargs["local_files_only"] is True
        return Model()

    fake = SimpleNamespace(
        Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=load_model),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: Processor()),
    )
    original = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake if name == "transformers" else original(name),
    )
    if kernel_fails:
        with pytest.raises(RuntimeError, match="lazy kernel unavailable"):
            film_audit_cli._local_reviewer(tmp_path, "cpu")
    else:
        review = film_audit_cli._local_reviewer(tmp_path, "cpu")
        assert limits == [1]
        assert review([{"type": "text", "text": "Review a customer frame"}]) == "black"
        assert limits == [1, 2400]
    preflight = messages[0][0]["content"]
    assert isinstance(preflight[0]["image"], Image.Image)
    assert preflight[0]["image"].size == (128, 128)
    assert preflight[0]["image"].tobytes() == bytes(128 * 128 * 3)
    assert not list(tmp_path.iterdir())
