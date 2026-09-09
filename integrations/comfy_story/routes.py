"""Same-origin, digest-addressed evidence previews for the Story inspector."""

from __future__ import annotations

import asyncio
import io
import re
from pathlib import Path
from typing import Any

from aiohttp import web
from PIL import Image

from comfy_story.story_store import StoryProjectStore

from .comfy_adapter import _story_root


async def evidence_preview(request: web.Request) -> web.Response:
    digest = request.match_info["digest"]
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise web.HTTPNotFound()
    store = StoryProjectStore(_story_root())
    path = store.root / "assets" / "sha256" / digest
    try:
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("preview is too large")
        content = store.load_asset(digest)
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
            media_type = Image.MIME.get(image.format or "", "application/octet-stream")
    except (OSError, ValueError):
        raise web.HTTPNotFound() from None
    return web.Response(
        body=content,
        content_type=media_type,
        headers={
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _verified_video_path(store: StoryProjectStore, digest: str) -> Path:
    path = store.verified_asset_path(digest)
    with path.open("rb") as handle:
        if handle.read(8)[4:8] != b"ftyp":
            raise ValueError("asset is not an MP4 clip")
    return path


async def accepted_video(request: web.Request) -> web.FileResponse:
    """Serve authenticated accepted clips without buffering entire videos on the event loop."""
    digest = request.match_info["digest"]
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise web.HTTPNotFound()
    store = StoryProjectStore(_story_root())
    try:
        path = await asyncio.to_thread(_verified_video_path, store, digest)
    except (OSError, ValueError):
        raise web.HTTPNotFound() from None
    return web.FileResponse(
        path,
        headers={
            "Content-Type": "video/mp4",
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )


def register_routes(server: Any) -> None:
    if getattr(server, "_duet_story_routes_registered", False):
        return
    server.routes.get("/duet/story/evidence/{digest}")(evidence_preview)
    server.routes.get("/duet/story/video/{digest}")(accepted_video)
    from .film_routes import register_film_routes

    register_film_routes(server)
    server._duet_story_routes_registered = True
