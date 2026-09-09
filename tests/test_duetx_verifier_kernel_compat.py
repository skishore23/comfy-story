from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from duet.duetx.verifier_kernel_compat import normalize_kernel_user_agent


@pytest.mark.parametrize("suffix", ["", None, "caller/1.0", {"caller": "1.0"}])
def test_kernel_client_only_normalizes_empty_suffix(
    monkeypatch: pytest.MonkeyPatch, suffix: Any
) -> None:
    calls: list[Any] = []
    api = SimpleNamespace(user_agent=suffix, token=False, endpoint="https://example.invalid")

    def factory(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return api

    module = SimpleNamespace(_get_hf_api=factory)
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    normalize_kernel_user_agent()
    adapter = module._get_hf_api
    normalize_kernel_user_agent()
    assert module._get_hf_api is adapter
    assert adapter("caller", extra=False) is api
    assert calls == [(("caller",), {"extra": False})]
    assert api.user_agent == (None if suffix == "" else suffix)
    assert api.token is False
    assert api.endpoint == "https://example.invalid"


@pytest.mark.parametrize("missing", ["kernels", "kernels_data"])
def test_optional_kernel_package_does_not_hide_broken_install(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    def load(name: str) -> Any:
        raise ModuleNotFoundError(name=missing)

    monkeypatch.setattr(importlib, "import_module", load)
    if missing == "kernels":
        normalize_kernel_user_agent()
    else:
        with pytest.raises(ModuleNotFoundError):
            normalize_kernel_user_agent()


def test_kernel_client_errors_are_not_bypassed(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory() -> Any:
        raise RuntimeError("publisher client unavailable")

    module = SimpleNamespace(_get_hf_api=factory)
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    normalize_kernel_user_agent()
    with pytest.raises(RuntimeError, match="publisher client unavailable"):
        module._get_hf_api()
