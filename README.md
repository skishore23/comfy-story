# Comfy Story

Comfy Story is a film-making custom node for ComfyUI and MiniMax H3. Build a reusable cast,
name your characters and props, plan a sequence of shots, and carry Story State forward as you
render. Review takes, revise a scene, add a soundtrack, and export an assembled film from the
same project.

**Checkpoint-backed associative memory is a required part of Comfy Story.** Every new shot records
learned history alongside its visual evidence. Generation requires a compatible trained memory
checkpoint and its identity hashes; there is no reference-only fallback or memory-off mode.

## What you can do

- **Keep a cast and visual library:** give characters, props and locations stable names, attach
  reference images, and mention them as `@Name` in shot prompts.
- **Build a sequence:** connect Story State and Last Frame with **Add Next Shot**, or open the film
  editor to plan shot actions, durations, subjects and intended state changes.
- **Carry history across shots:** retain ordered associative memory, fusion trees and dense state
  in saved revisions, together with the exact images you approve as evidence.
- **Iterate on a film:** save a plan, generate a first cut, review individual takes and change the
  affected shots. Unchanged completed work can be recovered when its inputs and runtime match.
- **Finish and transfer:** add soundtrack cues, export the assembled video, save the ComfyUI
  workflow and transfer a project's input bundle to another compatible installation.

## How Story memory works

A reference library describes what your characters, objects and locations should look like.
Associative memory adds a learned representation of what has happened in the sequence. Comfy Story
keeps both; they serve different purposes.

| Component | What it preserves | How it is used |
| --- | --- | --- |
| Named references and approved evidence | Exact images of your cast, props, locations and selected moments | Supplies explicit visual references; creator approval remains separate from generation |
| Ordered associative history | Encoded opening, change and closing observations from rendered shots | Composes learned operators in sequence so history and order can affect the memory readout |
| Fusion trees | Sixteen persistent blocks with eight shot slots each | Maintains up to 128 shots of structured history and supports removing a source shot from later history |
| Dense memory state | The composed historical operators and authenticated snapshots | Restores the saved memory state without reconstructing the story from prompts alone |
| Trained checkpoint | The learned memory modules and their model contract | Produces the historical readout; checked against the foundation model, training configuration and video VAE |

For **Reference shot → Continue frame**, the previous revision's memory readout supplies a historical
context image alongside the selected references. It occupies one of H3's nine visual-source slots.
The first shot has no previous history. **New composition** and **Animate frame** still record memory,
but omit that inherited context image so a previous scene is not imposed on the new composition.
Forgetting an observation removes its entire source shot from dense history in subsequent revisions.

Memory is guidance, not a guarantee of visual correctness. Review faces, clothing, weapon ownership,
prop state and action before keeping a take. Rendering a planned event does not automatically make
it confirmed canon. **Keep this moment** and **Keep @Name look** record evidence you approve.

## Requirements

- Python 3.11 or newer and an existing ComfyUI installation with working, authorized MiniMax H3
  generation models, text/vision encoders and video/audio VAEs.
- A compatible **trained associative-memory checkpoint**, plus trusted SHA-256 identities for that
  checkpoint, its foundation model, its training configuration and its video VAE.
- A supported story-storage host: Linux or macOS. GPU generation has been exercised on Linux;
  Windows storage is not supported. See the installation guide for the tested host configuration.

Model weights, the memory checkpoint and credentials are **not included in this source repository
or the installer**. Obtain the compatible checkpoint and its identity manifest from its provider
before attempting generation. This repository does not currently provide a public checkpoint
download. The configuration hash is the checkpoint's training-configuration identity; it is not a
hash of your workflow JSON. An arbitrary checkpoint or a newly initialized memory module is not a
substitute for compatible trained weights.

## Get started

1. Follow the [installation guide](docs/getting-started.md) to build and install the ComfyUI extension
   into your existing environment. Cloning the source into `custom_nodes` alone is insufficient.
2. Configure the required memory checkpoint in the environment that launches ComfyUI:

   ```bash
   export COMFY_STORY_MEMORY_CHECKPOINT=/absolute/path/to/h3-memory.pt
   export COMFY_STORY_MEMORY_CHECKPOINT_SHA256='CHECKPOINT_SHA256'
   export COMFY_STORY_MEMORY_FOUNDATION_SHA256='FOUNDATION_MODEL_SHA256'
   export COMFY_STORY_MEMORY_MODEL_CONFIGURATION_SHA256='CHECKPOINT_TRAINING_CONFIGURATION_SHA256'
   export COMFY_STORY_MEMORY_VAE_SHA256='VIDEO_VAE_SHA256'
   ```

   Replace the placeholders with the independently recorded lowercase, 64-character hashes.
   Associative memory is selected automatically. Remove an old `COMFY_STORY_MEMORY=native` setting;
   if `COMFY_STORY_MEMORY` is present, its only supported value is `associative`.
3. Run the [checkpoint preflight](docs/getting-started.md#associative-memory) using ComfyUI's Python.
   It authenticates the checkpoint and checks that changing history and its order changes the
   readout, without loading the foundation model or allocating its GPU VAE. ComfyUI also verifies
   the actual generation-model and VAE files before rendering. Missing configuration, inactive
   weights or mismatched identities stop generation with an error.
4. Start or restart ComfyUI with those environment settings. Add **Comfy Story** from **Comfy / Story**,
   choose **Start Story**, add your references and describe a short, visible action.
5. Generate and review the first shot. Use **Add Next Shot** to connect Story State and Last Frame,
   or **Open Story** to plan a film. Save your workflow and project as you work.

[Film workflow](docs/films.md) · [Writing prompts](docs/prompting.md) · [Security](SECURITY.md)

## Saving, recovery and upgrades

Story data lives under `comfy_story` in ComfyUI's user directory. Set `COMFY_STORY_ROOT` to an
absolute directory before launching ComfyUI to choose another location. Preserve the **complete
story directory**, including memory snapshots and media, as well as the workflow JSON. A film
inputs bundle transfers the plan and referenced images/audio; it does not contain model weights,
completed takes, approvals or the full memory archive.

Completed-shot recovery reopens verified saved output; it does not resume interrupted sampling.
Checkpoint, model and runtime identities are bound to saved revisions. Keep the matching software
and models when reopening a branch. Start a new story when changing that configuration; legacy
reference-only stories cannot be continued by silently inventing missing associative history.
When upgrading the memory runtime, retain a matching installation for historical branches and
start new stories on the upgraded version.

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
