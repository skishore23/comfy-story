"""H3 cache lifecycle and packed-stream approximation contracts."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from comfy_story.h3_acceleration import H3BlockCache, patch_h3_balanced


class MiniMaxH3Model:
    blocks = [None] * 50


class Patcher:
    def __init__(self) -> None:
        self.model_options: dict[str, Any] = {}
        self.patches: dict[int, Any] = {}
        self.wrapper: Any = None

    def get_model_object(self, name: str) -> MiniMaxH3Model:
        return MiniMaxH3Model()

    def clone(self) -> Patcher:
        return copy.deepcopy(self)

    def set_model_patch_replace(self, patch: Any, kind: str, block: str, index: int) -> None:
        self.patches[index] = patch

    def add_wrapper_with_key(self, kind: str, key: str, wrapper: Any) -> None:
        self.wrapper = wrapper


def test_bounded_cache_forecasts_entire_packed_tail_and_keeps_dense_ends() -> None:
    cache = H3BlockCache(10)
    skipped = []
    # All rows (text, references, audio and video) must be updated, including rows not probed.
    for step in range(10):
        head = torch.full((33, 4), float(step))
        before = head[::8].clone()
        head.add_(1)
        cache.first(before, head, ("packed",))
        skipped.append(cache.skip)
        expected_tail = torch.arange(132).reshape(33, 4) * (1 + step * 0.1)
        result = cache.finish(head if cache.skip else head + expected_tail)
        torch.testing.assert_close(result, head + expected_tail)
    assert skipped == [False, False, False, True, True, False, True, True, False, False]
    assert cache.reused == 4


def test_layout_change_and_extra_model_calls_fail_closed() -> None:
    cache = H3BlockCache(1)
    value = torch.ones(8, 2)
    cache.first(value, value, ("first",))
    with pytest.raises(ValueError, match="layout changed"):
        cache.first(value, value, ("second",))
    cache.finish(value)
    with pytest.raises(ValueError, match="one model evaluation"):
        cache.first(value, value, ("first",))


def test_large_or_nonfinite_probe_difference_never_reuses() -> None:
    cache = H3BlockCache(8)
    for step in range(4):
        before = torch.zeros(2, 2)
        after = torch.full((16, 2), 1.0 if step < 3 else 100.0)
        cache.first(before, after, "same")
        assert not cache.skip
        cache.finish(after + 2)


def test_patcher_clones_and_clears_state_on_success_and_failure() -> None:
    original = Patcher()
    patched = patch_h3_balanced(original)
    assert not original.patches
    calls: list[int] = []
    layout = SimpleNamespace(segments=[(0, 8, "text"), (8, 16, "video")])

    def trajectory(*args: Any, fail: bool = False, **kwargs: Any) -> list[torch.Tensor]:
        outputs = []
        for step in range(10):
            value = torch.zeros(16, 2)
            for index in range(50):

                def block(values: dict[str, Any], i: int = index) -> dict[str, torch.Tensor]:
                    calls.append(i)
                    # Native H3 mutates residual buffers in place.
                    return {"img": values["img"].add_(1)}

                value = patched.patches[index](
                    {"img": value, "layout": layout}, {"original_block": block}
                )["img"]
            outputs.append(value)
            if fail and step == 3:
                raise RuntimeError("interrupted")
        return outputs

    sigmas = torch.linspace(1, 0, 11)
    for _ in range(2):
        calls.clear()
        outputs = patched.wrapper(trajectory, sigmas=sigmas)
        assert calls.count(0) == 10
        assert calls.count(49) == 6
        assert all(torch.equal(value, torch.full((16, 2), 50.0)) for value in outputs)
    with pytest.raises(RuntimeError, match="interrupted"):
        patched.wrapper(trajectory, sigmas=sigmas, fail=True)
    with pytest.raises(ValueError, match="outside its sampling context"):
        patched.patches[0]({}, {})
    calls.clear()
    patched.wrapper(trajectory, sigmas=sigmas)
    assert calls.count(49) == 6


def test_patcher_rejects_conflicting_patches_and_bad_schedule() -> None:
    model = Patcher()
    model.model_options = {"transformer_options": {"patches_replace": {"dit": {"existing": 1}}}}
    with pytest.raises(ValueError, match="overwrite"):
        patch_h3_balanced(model)
    patched = patch_h3_balanced(Patcher())
    with pytest.raises(ValueError, match="strictly decreasing"):
        patched.wrapper(lambda **kw: None, sigmas=torch.ones(8))
    with pytest.raises(ValueError, match="exactly one H3 call"):
        patched.wrapper(lambda **kw: None, sigmas=torch.linspace(1, 0, 8))


def test_sol_hook_is_model_scoped_and_context_is_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    from comfy_story import h3_acceleration

    def kernel(*a: Any, **kw: Any) -> torch.Tensor:
        return torch.empty(0)

    monkeypatch.setattr(h3_acceleration, "load_sol_kernel", lambda: kernel)
    model = Patcher()
    patched = patch_h3_balanced(model, sol_attention=True)
    assert model.model_options == {}
    attention = patched.model_options["transformer_options"]["optimized_attention_override"]
    with pytest.raises(ValueError, match="outside its sampling context"):
        attention(None)

    def interrupted(**kwargs: Any) -> None:
        x = torch.ones(1, 1, 4, 2)
        assert attention(lambda *a, **kw: x, x, x, x, 1) is x
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        patched.wrapper(interrupted, sigmas=torch.linspace(1, 0, 11))
    with pytest.raises(ValueError, match="outside its sampling context"):
        attention(None)
    conflict = Patcher()
    conflict.model_options = {"transformer_options": {"optimized_attention_override": object()}}
    with pytest.raises(ValueError, match="existing attention override"):
        patch_h3_balanced(conflict, sol_attention=True)
