"""Durable completed-attempt index; immutable revisions remain authoritative."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from duet.duetx.story_contracts import DuetStoryStateRef, canonical_story_json


class StoryAttemptIndex:
    """Reuse completed identical requests across Comfy cache eviction and restart.

    This indexes completed work only. It never treats a running or failed request
    as completed, and callers authenticate the referenced revision before reuse.
    A changed Variation is a different request and retains the previous take.
    """

    def __init__(self, root: Path) -> None:
        self.root = root / "completed-attempts"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, request_sha256: str) -> Path:
        if len(request_sha256) != 64 or any(c not in "0123456789abcdef" for c in request_sha256):
            raise ValueError("attempt request must be a SHA-256 digest")
        return self.root / f"{request_sha256}.json"

    def load(self, request_sha256: str) -> DuetStoryStateRef | None:
        path = self._path(request_sha256)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
            raise ValueError("completed attempt must be a bounded regular file")
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict) or set(value) != {"request_sha256", "state", "state_sha256"}:
            raise ValueError("completed attempt is malformed")
        state = canonical_story_json(value["state"])
        if (
            value["request_sha256"] != request_sha256
            or value["state_sha256"] != hashlib.sha256(state).hexdigest()
        ):
            raise ValueError("completed attempt binding changed")
        return DuetStoryStateRef.from_json(state)

    def publish(self, request_sha256: str, state: DuetStoryStateRef) -> None:
        target = self._path(request_sha256)
        encoded_state = state.to_json()
        encoded = canonical_story_json(
            {
                "request_sha256": request_sha256,
                "state": json.loads(encoded_state),
                "state_sha256": hashlib.sha256(encoded_state).hexdigest(),
            }
        )
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if self.load(request_sha256) != state:
                    raise ValueError("a different take already completed this request") from None
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
