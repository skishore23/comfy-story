# Installation and first sequence

Use Python 3.11 or newer in an existing ComfyUI environment with working MiniMax H3 nodes and
authorized model access. Install the foundation, text/vision encoders, and video/audio VAEs using
your supported ComfyUI model setup. Model files and credentials are not included.

## Install

The complete installer ZIP from [Releases](https://github.com/skishore23/comfy-story/releases/latest)
contains the runtime wheel, ComfyUI extension and authenticated memory checkpoint. Extract it and
run with ComfyUI's Python:

```bash
/path/to/ComfyUI/.venv/bin/python install.py --comfy-root /path/to/ComfyUI
```

The same command works from a clean cloned source checkout. It fetches the pinned memory asset
and builds the installable bundle automatically. Private repository downloads require an
authenticated `gh` CLI; the complete ZIP needs no checkpoint download or GitHub credentials.

The installer verifies the bundle, supplies associative memory automatically, installs missing
Python dependencies and checks required H3 files, checkpoint behavior, and ffmpeg/ffprobe. It pins
the host's existing Torch version during dependency installation. A missing or incompatible Torch
build must be resolved in the ComfyUI environment first. Restart ComfyUI after successful setup.

For inspection without installing packages or nodes:

```bash
/path/to/ComfyUI/.venv/bin/python install.py --comfy-root /path/to/ComfyUI --check-only
```

A source preflight may download/cache the small checkpoint before checking the host. To install
offline, transfer the complete ZIP and ensure dependencies and H3 models are already installed.
To build offline from source, add `--memory-checkpoint /path/to/h3-associative-memory.pt`; the supplied
file must match the shipped checksum. To build a redistributable ZIP, run
`python scripts/build_release.py --output-root artifacts/releases` from a clean committed checkout.

Cloning into `custom_nodes` or installing the Python project alone is insufficient. The installer
places the actual extension in `ComfyUI/custom_nodes/comfy_story` and installs the runtime and
checkpoint together. Existing node directories are preserved: for upgrades, stage a separate
installation and back up the workflow, story store and matching old runtime before switching.

### H3 models

The default **Reference shot / Native res_multistep** workflow needs these files in ComfyUI:

| Folder | File |
| --- | --- |
| `diffusion_models` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| `text_encoders` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| `vae` | `minimax_h3_video_vae_fp16.safetensors` |
| `vae` | `minimax_h3_audio_vae_fp32.safetensors` |

The installer reads ComfyUI's default `extra_model_paths.yaml`. For a launch-specific shared model
configuration, pass `--extra-model-paths-config /path/to/models.yaml` as well. It checks actual
foundation and video-VAE hashes, not just filenames. The shipped memory checkpoint targets the
reference model above; a different foundation model requires a compatible checkpoint.

Turbo sampling requires its corresponding LoRA. Full HD 2-pass also requires the pinned H3 latent
upscaler. Those presets are additional configurations; the baseline installation preflight checks
the default Reference shot workflow. In particular, Animate frame uses another foundation model
and cannot use this checkpoint's identity pins unchanged.

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

The installer includes the supported trained checkpoint and selects it automatically. It checks
that history and order affect the readout and that the actual H3 model and video VAE match.
The weights are stored inside the installed `comfy_story/models/` package directory, separately
from the source repository. A missing or corrupt checkpoint stops generation with an actionable
error. Checkpoint integrity does not guarantee rendered continuity quality.

### Advanced: custom checkpoints

Normal installations need none of the following variables. To use a different compatible trained
checkpoint, supply **all five** identity settings; partial overrides are rejected so custom weights
cannot accidentally inherit the bundled checkpoint's pins. Remove all five to return to the bundled
configuration. An old `COMFY_STORY_MEMORY=native` setting is an error; remove it.

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
creates its story store or prepares generation when using an explicit override.

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
