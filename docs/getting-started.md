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
Configure and preflight the required [associative memory](#associative-memory) checkpoint before
restarting ComfyUI. Cloning the source into `custom_nodes` alone does not install it.

For an upgrade, validate a staged installation first. Back up the existing node, environment,
workflow, and complete story directory before switching while the render queue is empty.

## Create two shots

1. Add **Comfy Story** from **Comfy / Story** and select **Start Story**.
2. Name the project and provide a starting image. A connected **Starting image** takes precedence
   over an uploaded starting frame for **Start Story** and **New Scene**.
3. Add character, prop, and location references with short names. Use `@Name` in the prompt to select
   them.
4. Generate a short shot. Review the output and choose **Keep this moment** or **Keep @Name look** for evidence you approve.
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

## Associative memory

Associative memory is required for generation, alongside approved reference images and shot
evidence. It encodes opening/change/closing observations into ordered operators,
maintains sixteen eight-shot fusion trees, and saves authenticated dense-memory snapshots.
With **Reference shot** and **Continue frame**, it supplies a historical context image alongside
exact references. That image uses one of H3's nine source slots. New composition and Animate frame
omit the inherited context image while continuing to store memory. Forgetting an observation
removes its whole source shot from dense history in subsequent revisions.

The trained checkpoint must respond to changes in history and order;
a successful preflight does not establish rendered continuity quality. The foundation model and
video VAE must match the configured hashes. Checkpoints are installed separately from source and
are not included in the installer. No public checkpoint download is currently provided by this
repository; obtain compatible trained weights and their identity manifest from the checkpoint
provider. Without them, you can install and edit projects, but generation cannot run.

Set these variables before starting ComfyUI, using independently recorded checkpoint/model hashes:

```bash
export COMFY_STORY_MEMORY_CHECKPOINT=/absolute/path/to/h3-memory.pt
export COMFY_STORY_MEMORY_CHECKPOINT_SHA256='CHECKPOINT_SHA256'
export COMFY_STORY_MEMORY_FOUNDATION_SHA256='FOUNDATION_MODEL_SHA256'
export COMFY_STORY_MEMORY_MODEL_CONFIGURATION_SHA256='CHECKPOINT_TRAINING_CONFIGURATION_SHA256'
export COMFY_STORY_MEMORY_VAE_SHA256='VIDEO_VAE_SHA256'
```

Replace the uppercase placeholders with the actual lowercase SHA-256 hashes. The configuration
hash identifies the checkpoint training configuration, not the workflow. Associative memory is
selected automatically; remove old `COMFY_STORY_MEMORY=native` settings. If set explicitly, only
`COMFY_STORY_MEMORY=associative` is supported. Missing pins or unsupported modes fail before a node
creates its story store or prepares generation.

Start a new story when changing checkpoint, model or memory runtime implementation; an existing
branch requires its original configuration. Keep the previous installation for historical branches.
Legacy reference-only history is not automatically converted to learned memory. Preserve all
project assets when backing up or moving a story, including its memory snapshots.

Before allocating GPU models, inspect the checkpoint using ComfyUI's Python environment:

```bash
python -m comfy_story.memory.inspect \
  --checkpoint "$COMFY_STORY_MEMORY_CHECKPOINT" \
  --sha256 "$COMFY_STORY_MEMORY_CHECKPOINT_SHA256" \
  --foundation-sha256 "$COMFY_STORY_MEMORY_FOUNDATION_SHA256" \
  --model-configuration-sha256 "$COMFY_STORY_MEMORY_MODEL_CONFIGURATION_SHA256" \
  --vae-sha256 "$COMFY_STORY_MEMORY_VAE_SHA256"
```

The check verifies the training envelope, tensor hashes, architecture and causal history response.
ComfyUI also checks the actual generation model and VAE files. It rejects inactive or mismatched
checkpoints rather than silently substituting reference-only storage. Completed-shot recovery and
the creator's exact-evidence approvals remain part of the workflow. A successful report includes
`checkpoint_authenticated: true` and nonzero history/order deltas. It explicitly leaves
`rendered_quality_validated: false`: checkpoint integrity does not establish visual quality.
