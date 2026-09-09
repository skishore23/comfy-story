# Installation and first sequence

Use Python 3.11 or newer in an existing ComfyUI environment with working MiniMax H3 nodes and
authorized model access. Install the foundation, text/vision encoders, and video/audio VAEs using
your supported ComfyUI model setup. Model files and credentials are not included.

## Install

Build the installer from a clean committed checkout:

```bash
python scripts/build_release.py --output-root artifacts/releases
```

Extract the ZIP and use the ComfyUI environment's Python:

```bash
/path/to/ComfyUI/.venv/bin/python install.py --comfy-root /path/to/ComfyUI --check-only
/path/to/ComfyUI/.venv/bin/python install.py --comfy-root /path/to/ComfyUI
```

The preflight checks the bundle, installed dependencies, and destination. Resolve any reported
missing dependencies in the ComfyUI environment. Installation preserves its Torch/CUDA build.
Restart ComfyUI after installation. Cloning the source into `custom_nodes` alone does not install it.

For an upgrade, validate a staged installation first. Back up the existing node, environment,
workflow, and complete story directory before switching while the render queue is empty.

## Create two shots

1. Add **Comfy Story** from **Comfy / Story** and select **Start Story**.
2. Name the project and provide a starting image. A connected **Starting image** takes precedence
   over an uploaded starting frame for **Start Story** and **New Scene**.
3. Add character, prop, and location references with short names. Use `@Name` in the prompt to select
   them.
4. Generate a short shot. Review the output and choose **Keep this detail** for evidence you approve.
5. Select **Add Next Shot** to connect Story State and Last Frame. Write the next action and generate.
6. Save the workflow, restart ComfyUI, and queue unchanged completed inputs to verify recovery.

Story data is stored under `comfy_story` in the ComfyUI user directory. Set `COMFY_STORY_ROOT`
to an absolute directory before starting ComfyUI to use another location. A saved workflow is not
a project backup: preserve the complete story directory as well.
Completed-shot recovery reopens stored results; it does not resume interrupted sampling.

## Operation

Story storage supports Linux and macOS. Windows storage is unsupported. The exercised generation
host used Linux, Python 3.13.12, Torch 2.12.1+cu130, ComfyUI 0.34.0, and a 96 GB RTX PRO 6000.
These describe a tested host, not minimum hardware requirements.

Story uses native H3 visual references and saved shot evidence.
Review generated dialogue, identity, prop state, action, and motion before approving a take.

Unknown reference names, missing assets, or changed model identities fail validation. Restore the
matching inputs instead of bypassing checks. Projects currently support up to 128 shots.
