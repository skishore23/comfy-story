"""Narrow compatibility for kernels' telemetry-disabled Hub requests."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any


def normalize_kernel_user_agent() -> None:
    """Keep an empty optional suffix from producing an invalid HTTP header.

    kernels 0.15.2 supplies user_agent="" when telemetry is disabled (ComfyUI's
    default). huggingface-hub 1.25.1 appends that as a trailing "; ", which HTTPX
    rejects before sending the publisher check. None omits only that empty suffix.
    Scope the adapter to kernels' client factory; retain its authentication,
    endpoint, publisher checks and all nonempty user-agent values unchanged.
    """
    try:
        module = importlib.import_module("kernels.utils")
    except ModuleNotFoundError as error:
        if error.name == "kernels":
            return
        raise
    original = module._get_hf_api
    if getattr(original, "_duet_normalizes_empty_user_agent", False):
        return

    @wraps(original)
    def client(*args: Any, **kwargs: Any) -> Any:
        api = original(*args, **kwargs)
        if api.user_agent == "":
            api.user_agent = None
        return api

    client.__dict__["_duet_normalizes_empty_user_agent"] = True
    module.__dict__["_get_hf_api"] = client
