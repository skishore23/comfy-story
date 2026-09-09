# Comfy Story

Build a film in ComfyUI with a reusable cast, named visual references, shot planning, and saved
Story State. Generate a first cut, review individual takes, change a shot, and export the result.
An optional checkpoint-backed associative runtime adds ordered dense history and persistent fusion
trees; see [memory configuration](docs/getting-started.md#associative-memory) for setup and limits.

## Get started

Comfy Story requires Python 3.11 or newer and an existing ComfyUI installation with working,
authorized MiniMax H3 models. Linux and macOS support project storage. GPU generation has been
exercised on Linux; Windows storage is not supported.

1. Follow the [installation guide](docs/getting-started.md).
2. Add **Comfy Story** from **Comfy / Story**.
3. Add named references, write the first shot, and generate.
4. Use **Add Next Shot** for a sequence or **Open Story** to plan a film.

[Film workflow](docs/films.md) · [Writing prompts](docs/prompting.md) · [Security](SECURITY.md)

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

Keep models, credentials, source media, generated output, and local caches outside Git.
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

Comfy Story is licensed under [AGPL-3.0-only](LICENSE). Model weights and other dependencies have
separate terms. See [LICENSING.md](LICENSING.md) and [NOTICE](NOTICE).
