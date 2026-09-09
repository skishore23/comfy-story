# Comfy Story

Create MiniMax H3 shots in ComfyUI with a reusable cast, named references, approved visual evidence,
immutable story revisions, and completed-shot recovery. Plan, generate, edit, and export a film from
the Story editor.

**Private alpha.** This repository stays private until installation, recovery, and fresh-film
acceptance have been demonstrated. Unit tests do not establish GPU compatibility or creative quality.
See [validation](docs/VALIDATION.md) for the current evidence and remaining gates.

## Install

Use an existing authorized MiniMax H3 ComfyUI installation with Python 3.11 or newer. Linux and
macOS support story storage; real GPU generation has been exercised on Linux. Windows storage is
not supported. Models, reference media, credentials, and generated films are not included.

From a clean committed checkout, build the verified installer bundle:

```bash
python scripts/build_duet_story_internal.py --output-root artifacts/releases
```

Extract the resulting ZIP, then run its installer with your ComfyUI Python:

```bash
/path/to/ComfyUI/.venv/bin/python install.py --comfy-root /path/to/ComfyUI --check-only
/path/to/ComfyUI/.venv/bin/python install.py --comfy-root /path/to/ComfyUI
```

Resolve any missing dependencies in the staged ComfyUI environment before installing. The installer
uses `--no-deps` to preserve the host's Torch/CUDA build. This is an explicit bundle installation;
cloning this repository into `custom_nodes` alone does not install it.

The distribution is `comfy-story`; its Python namespace remains `duet` for saved-project and
checkpoint compatibility. Use a separate environment from the research `duet` distribution, since
both own that namespace. For an upgrade, preserve the old environment and story store for rollback.

Restart ComfyUI and add **Comfy Story** from **Comfy / Story**. Start new projects with the native
reference runtime. Follow the [quickstart](docs/DUET_STORY_QUICKSTART.md) to create two shots,
reload the workflow, and verify completed-shot recovery.

## Product guides

- [Film planning, generation, and export](docs/DUET_STORY_FILM_WORKFLOW.md)
- [Prompting H3](docs/DUET_STORY_H3_PROMPTING.md)
- [Current quality limitations](docs/DUET_STORY_FAILURE_ANALYSIS.md)
- [Customer acceptance criteria](docs/DUET_STORY_CUSTOMER_ACCEPTANCE.md)
- [Legacy names and compatibility](docs/COMFY_STORY_REBRAND.md)
- [Extraction scope and provenance](docs/REPOSITORY_SCOPE.md)

Native references are the default. Trained memory, compiled references, and Full HD refinement
remain optional paths with separate prerequisites and unproven quality benefits. Live/FastH3 work
is not included in this source snapshot.

## Development

```bash
uv sync --locked --extra dev
uv run python -m ruff format .
uv run python -m ruff check .
uv run python -m ruff format --check .
uv run python -m mypy
uv run python -m pytest
node --test tests/js/*.mjs
```

Runtime code is under `src/duet/`; the ComfyUI extension and frontend are under
`integrations/comfyui_duetx_continuity/`. Historical identifiers and console commands remain stable;
`comfy-story-film` and `comfy-story-audit` also name the product commands.

The source uses AGPL-3.0-only. Preserve [LICENSE](LICENSE), [NOTICE](NOTICE), and the historical
license notices. Model access is governed separately; see [LICENSING.md](LICENSING.md).
