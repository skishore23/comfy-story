# Comfy Story agent guide

## Scope

Keep the repository source-first. Do not commit datasets, generated audio, checkpoints, experiment
outputs, credentials, caches, or editor state. Store local research outputs under an ignored path
such as `artifacts/`.

## Required workflow

1. Read `README.md`, `pyproject.toml`, and the files directly related to the task.
2. Make the smallest coherent change and preserve the typed-stream, fusion, and historical
   checkpoint contracts.
3. Add or update tests for behavior changes.
4. Run `python -m ruff format .` before the quality gate.
5. Run `python -m ruff check .`, `python -m ruff format --check .`, `python -m mypy`, and
   `python -m pytest` before handing off or committing.

Use the project skill at `.agents/skills/maintain-duet/SKILL.md` for repository cleanup, dependency
updates, quality configuration, or pre-publish verification.

## Repository hygiene

- Put importable code under `src/duet/` and tests under `tests/`.
- Keep configuration in `pyproject.toml`; do not add overlapping formatter or linter configs.
- Use narrow, explained suppressions only when a third-party library lacks usable type information.
- Never weaken a quality rule merely to make a check pass.
- Keep commits focused and do not stage unrelated local files.
