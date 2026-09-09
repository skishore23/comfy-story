# Comfy Story

Build a film one shot at a time. Comfy Story brings a reusable cast, story memory, a visual
shot timeline, and soundtrack editing to ComfyUI and MiniMax H3.

Give your characters and props names, describe what happens next, and develop the story across
scenes. Preview your takes, try a different direction, and bring everything together in one project.

[Get started](#get-started) · [Download](https://github.com/skishore23/comfy-story/releases/latest) · [Film guide](docs/films.md) · [Prompt guide](docs/prompting.md)

## Make the story yours

- **Build your cast.** Add reference images for characters, props, and locations. Mention them as
  `@Name` in your prompts to tell Comfy Story who and what belongs in each shot.
- **Plan the action.** Arrange shots on a visual timeline, set their duration, and choose starting
  and ending frames. Describe the action and how you want the camera to follow it.
- **Carry the story forward.** Built-in associative memory preserves shot history alongside your
  references, giving later shots context from what came before.
- **Watch and refine.** Preview a saved take or the assembled film. Edit a shot, try a new variation,
  and see which parts of the sequence need rendering again.
- **Add the sound.** Upload music or dialogue, choose when it starts, and adjust its volume.
  Export the film with your soundtrack and save the workflow for later.

## Get started

You'll need:

- **ComfyUI with a working MiniMax H3 setup**, including the authorized generation models,
  encoders, and video/audio VAEs, plus Python 3.11 or newer and suitable hardware.
- **Linux or macOS** for story storage. GPU generation has been tested on Linux; Windows story
  storage is not currently supported.
- **FFmpeg and ffprobe** available on your system for film export.

Download the **complete installer ZIP** from [Releases](https://github.com/skishore23/comfy-story/releases/latest)
and extract it. From that folder, run the installer using ComfyUI's Python environment:

```bash
/path/to/ComfyUI/.venv/bin/python install.py --comfy-root /path/to/ComfyUI
```

The installer sets up Comfy Story and its story memory automatically, checks your H3 models,
and installs missing Python dependencies while preserving your existing Torch/CUDA build.
The H3 foundation models and encoders are installed separately through your ComfyUI model setup.

<details>
<summary>Install from source</summary>

Clone the repository outside ComfyUI's `custom_nodes` folder, then run the same installer:

```bash
git clone https://github.com/skishore23/comfy-story.git
cd comfy-story
/path/to/ComfyUI/.venv/bin/python install.py --comfy-root /path/to/ComfyUI
```

The installer downloads the small memory model automatically. No GitHub credentials or manual
memory configuration are needed. Use the installer to set up the complete custom node;
cloning the repository or running `pip install .` alone does not complete installation.

</details>

See the [installation guide](docs/getting-started.md) for model filenames, shared model paths,
offline setup, and the `--check-only` option.

## Create your first film

1. **Open Comfy Story.** Restart ComfyUI and add **Comfy Story** from **Comfy / Story**. Select
   **Open Story** to open the film editor.
2. **Meet your cast.** In **Cast & world**, name your characters, props, and locations and upload
   their reference images.
3. **Write the opening.** In **Shots**, add a starting image and describe a short, visible action.
   Select the references that should appear and choose the shot's duration.
4. **Generate a first cut.** Save the plan, then choose **Generate saved plan**. Select a shot on
   the timeline to preview its take, or choose **Watch assembled film** to watch the sequence.
5. **Make it your own.** Add more shots, refine the action, and use **Soundtrack** to add music.
   Save your changes and generate again to create the updated film.

Start with one clear action. For example, after adding references named `Aiko` and `Moonblade`:

> @Aiko draws @Moonblade as rain scatters across the rooftop. She turns toward the distant gate.
> The camera moves slowly alongside her, keeping her face and the sword in view.

You can also build directly on the ComfyUI canvas: choose **Start Story** for the opening, then
use **Add Next Shot** to connect the previous shot's Story State and Last Frame.

## Keep your characters and story connected

Your reference library establishes the look of your cast and world. Story memory keeps a history
of rendered shots so later shots can draw on earlier events. It is set up for you during installation.

When you like a result, the node's **Keep this moment** and **Keep @Name look** controls let you
save visual evidence for future shots. Review faces, clothing, props, and action as you go;
memory helps guide continuity, but a generated take still needs your eye.

Use **Continue with references** to continue from the previous frame with historical context.
Choose **Compose from references** to create a new view from your selected references. The
[film guide](docs/films.md) explains other render approaches and their model requirements.

## Edit without starting over

Saving an edit shows which earlier shots are eligible for reuse and which shots need generation
review. Changing a shot can affect everything after it because later shots inherit its story state.
Unchanged earlier shots can reuse saved output when their inputs and runtime still match.

Soundtrack-only edits can reuse the visuals. Uploaded audio is mixed into the finished film;
it does not drive the generated motion. To keep H3's own speech and effects, choose **Generated
speech and effects** and follow the [audio instructions](docs/films.md#audio-and-timing).

Open **Generation history & review details** to revisit previous runs and their assessments.
Generation continues on the ComfyUI host when you close the editor. The editor asks you to save
or discard unsaved edits before closing.

## Save and return later

Save your ComfyUI workflow and back up the complete story directory under `comfy_story` in
ComfyUI's user directory. It contains the media, history, and memory needed to return to your work.
To choose another location, set `COMFY_STORY_ROOT` to an absolute directory before launching ComfyUI.

Use **Download inputs bundle** to transfer a plan and its reference images/audio to another
compatible installation. Keep your story-directory backup too: the inputs bundle does not include
completed takes, approvals, model weights, or the full memory archive.

Saved work is tied to the software and models that created it. Keep a matching installation for
older runs, and start a new story when changing the memory runtime or model configuration.
Recovery reopens completed output; it does not resume sampling that was interrupted.

## Development

```bash
uv sync --locked --extra dev --python 3.11
uv run python -m ruff format .
uv run python -m ruff check .
uv run python -m ruff format --check .
uv run python -m mypy
uv run python -m pytest
node --test tests/js/*.mjs
```

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute and [SECURITY.md](SECURITY.md) to report a
security issue. Keep models, credentials, source media, and generated output outside Git.

## License

Comfy Story is licensed under [AGPL-3.0-only](LICENSE). Model weights and other dependencies have
separate terms. See [LICENSING.md](LICENSING.md) and [NOTICE](NOTICE).
