# Comfy Story development

Keep importable code in `src/comfy_story/`, the ComfyUI extension in `integrations/comfy_story/`,
and tests in `tests/`. Keep tool configuration in `pyproject.toml`.

Read the README, project configuration, and affected files before editing. Add regression coverage
for behavior changes. Preserve project integrity, immutable revisions, and output recovery.
Do not weaken validation, type checking, or quality rules to make checks pass.

Before committing or handing off, run:

```bash
python -m ruff format .
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest
node --test tests/js/*.mjs
```

Keep models, media, credentials, generated output, and caches out of Git under `artifacts/`.
Stage only changes belonging to the current task. Retain copyright and third-party notices.
