"""Stable content hashes for runtime tensors."""

import hashlib
import json

import torch


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous CPU bytes without device identity."""
    if tensor.layout != torch.strided:
        raise ValueError("tensor_sha256 requires a strided tensor")
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    header = json.dumps(
        {"dtype": _dtype_name(tensor.dtype), "shape": list(tensor.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\n" + raw).hexdigest()
