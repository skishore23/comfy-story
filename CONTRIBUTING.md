# Contributing

Keep changes focused on Comfy Story. Add regression tests for behavior changes and preserve project
integrity, immutable revisions, model authentication, and completed-shot recovery.

Install the development environment with `uv sync --locked --extra dev`. Before committing, run
the formatter, Ruff, format verification, strict mypy, the complete Python suite, and the frontend
tests shown in [README.md](README.md).

Keep credentials, models, media, generated output, and caches outside Git under `artifacts/`.
Do not weaken validation or quality rules to make checks pass. Retain copyright and third-party
notices. Contributions use AGPL-3.0-only.
