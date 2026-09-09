# Contributing to Comfy Story

Keep changes focused on the Comfy Story product. Preserve saved workflows, node IDs, typed state,
checkpoint identities, and existing project formats. Add regression tests for behavior changes.

Install the locked development environment with `uv sync --locked --extra dev`. Run the formatter,
Ruff, format verification, mypy, the full Python suite, and `node --test tests/js/*.mjs` before a PR.
See [README.md](README.md) for exact commands.

Keep credentials, models, reference media, generated clips, caches, and experiment outputs outside
Git under `artifacts/`. Do not weaken integrity checks or quality rules to make a test pass.
Contributions use AGPL-3.0-only; retain copyright and third-party notices.

This repository remains private until the owner approves publication based on the evidence in
[VALIDATION.md](docs/VALIDATION.md). Passing unit tests does not prove film quality.
