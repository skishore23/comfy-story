"""Disposable, bounded full-quality reference encodings; never authoritative story state."""

from __future__ import annotations

import fcntl
import hashlib
import io
import logging
import os
import pickle
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch

from comfy_story.tensors import tensor_sha256

_LOG = logging.getLogger(__name__)
_MAX_ENTRY = 256 * 1024 * 1024
DEFAULT_CACHE_BYTES = 2 * 1024 * 1024 * 1024


class ReferenceCache:
    """Cache exact encoder outputs under an encoder identity and preprocessed pixel digest."""

    def __init__(self, root: Path, identity: str, maximum_bytes: int = DEFAULT_CACHE_BYTES):
        self.root = root
        self.identity = identity
        self.maximum_bytes = maximum_bytes

    def encode(
        self, pixels: torch.Tensor, encoder: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if self.maximum_bytes <= 0:
            return encoder(pixels)
        key = hashlib.sha256(
            ("comfy-reference-cache-v1:" + self.identity + ":" + tensor_sha256(pixels)).encode()
        ).hexdigest()
        # Cache availability must not prevent generation. Encoder errors still propagate.
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            lock = (self.root / ".lock").open("a+b")
        except OSError:
            _LOG.warning("Reference cache unavailable; encoding normally")
            return encoder(pixels)
        with lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / (key + ".pt")
            cached = self._read(path)
            if cached is not None:
                _LOG.info("Comfy Story reference cache hit %s (%s)", key[:12], tuple(cached.shape))
                return cached
            result = encoder(pixels)
            # Keep the original result for the cold run; persistence never changes its precision.
            try:
                self._write(path, result)
            except OSError:
                _LOG.warning("Could not persist reference encoding; generation continues")
            _LOG.info("Comfy Story reference cache miss %s (%s)", key[:12], tuple(result.shape))
            return result

    def _read(self, path: Path) -> torch.Tensor | None:
        try:
            if path.is_symlink() or not 32 < path.stat().st_size <= _MAX_ENTRY:
                return None
            encoded = path.read_bytes()
            if hashlib.sha256(encoded[32:]).digest() != encoded[:32]:
                return None
            value = torch.load(io.BytesIO(encoded[32:]), map_location="cpu", weights_only=True)
            if not isinstance(value, torch.Tensor) or value.ndim != 5 or value.shape[1] != 24:
                return None
            if not value.is_floating_point() or not torch.isfinite(value).all():
                return None
            os.utime(path, None)
            return value
        except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError):
            return None

    def _write(self, path: Path, tensor: torch.Tensor) -> None:
        if tensor.ndim != 5 or tensor.shape[1] != 24 or not torch.isfinite(tensor).all():
            return
        buffer = io.BytesIO()
        torch.save(tensor.detach().cpu().contiguous(), buffer)
        payload = buffer.getvalue()
        encoded = hashlib.sha256(payload).digest() + payload
        if len(encoded) > min(_MAX_ENTRY, self.maximum_bytes):
            return
        entries = sorted(self.root.glob("*.pt"), key=lambda p: p.stat().st_mtime_ns)
        total = sum(p.stat().st_size for p in entries)
        for entry in entries:
            if total + len(encoded) <= self.maximum_bytes:
                break
            total -= entry.stat().st_size
            entry.unlink()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
