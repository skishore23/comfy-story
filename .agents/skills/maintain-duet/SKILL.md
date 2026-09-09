---
name: maintain-duet
description: Keep the Comfy Story Python repository clean and release-ready. Use when changing Python code, dependencies, tests, lint or type-check configuration, generated-output handling, repository structure, or when preparing Duet changes for commit or publication.
---

# Maintain Duet

## Inspect scope

Read `AGENTS.md`, `pyproject.toml`, and the files relevant to the requested change. Run
`git status --short` when Git is available. Treat local datasets, audio, checkpoints, experiment
outputs, credentials, and caches as private generated state; never stage them.

## Make changes

Keep importable code in `src/duet/`, tests in `tests/`, and tool configuration in
`pyproject.toml`. Preserve the audio, condition, checkpoint, and fusion-tree contracts unless the
task explicitly changes them. Add focused tests for behavior changes.

Use a narrow suppression with an explanation only when a third-party package has incomplete type
metadata. Do not relax repository-wide rules to accommodate one call site.

## Verify

Run the formatter, then the complete quality gate from the repository root:

```bash
python -m ruff format .
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest
```

If dependencies are not installed, run `python -m pip install -e '.[dev]'` once and repeat
the failed command. On supported Linux and Apple Silicon environments, `uv sync --extra dev` uses the checked-in lockfile. PyTorch 2.8 does not publish Intel-macOS wheels, so
do not replace a working Intel-macOS environment with a fresh uv environment. Report any check that
could not run and the concrete reason.

## Prepare a commit

Review ignored and staged files before committing. Ensure `.env`, `artifacts/`, `assets/`, model
weights, generated audio, and tool caches are absent. Keep the commit limited to the intended task.
