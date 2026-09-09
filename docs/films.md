# Plan, generate, and edit a film

Select **Open Story** on the Comfy Story node. In **Cast & world**, name your characters, props and
locations and upload reference images. In **Shots**, select a shot on the thumbnail timeline,
describe its visible action, choose its subjects and supply a starting image. Start/end frame and
camera controls are beside the action; seeds, state checks and model settings are under Advanced.

New films use **Continue with references**, compatible with the bundled memory checkpoint. Use
**Compose from references** for a new composition. Existing animation profiles are preserved, but
require a compatible checkpoint for their different foundation model; changing the UI selection
does not install that model or checkpoint.

## First cut

Save the plan, keep seeds fixed, and select **Generate saved plan**. **First cut** renders the saved
sequence as one ComfyUI job. Closing the editor does not stop it. The resulting film is a draft for
your review; generating every shot does not establish story coherence or visual quality.

Review the film and edit the relevant shot cards. Save a new revision and generate again. Unchanged
earlier shots can recover their saved outputs; dependent later shots regenerate. Use **New take
seed** only when you want a new variation.

Select a timeline card to preview its saved take, or **Watch assembled film** to review the film
with sound. Earlier-plan footage is labeled when it differs from your current draft. Saving shows
how many shots are eligible for reuse and marks affected shots for generation review. Reuse also
requires matching runtime identities and intact saved outputs. Open **Generation history & review
details** for prior runs, machine assessments and recovery information; previews do not approve takes.

## Audio and timing

Open **Soundtrack** to upload reviewed audio directly (up to 256 MiB), or expand **Use an already
uploaded file** to choose an existing Comfy input. Set the start, duration and volume; the placement
bar shows its interval within the film. Tracks mix together. This is soundtrack mixing, not native
audio conditioning of motion. Soundtrack edits can reuse the existing visuals.
Generated speech and effects requires the H3 automatic prompt format. Supply the intended language,
exact dialogue, speaker, and timing window for each line, then listen to the output.

The export uses the saved shot durations and audio offsets. Keep source trims and dialogue within
their assigned windows. Prompt constraints do not guarantee exact speech or the absence of extra
words, so review the final mix before sharing it.

## Review and recovery

**Check each shot** is an optional mode for a host configured with a compatible local visual verifier.
It can stop on failed or uncertain assessments and supports an attempt budget. The verifier does
not replace creator approval. Staged openings and selected-state recall require this mode. Install the `film-audit` optional
dependencies listed in `pyproject.toml`, set `COMFY_STORY_VERIFY_MODEL` to the local verifier
model directory, and restart ComfyUI. `COMFY_STORY_VERIFY_DEVICE` selects the device (default: `cpu`).

**Resume selected run** retains the saved recipe, generation mode, and attempt budget. Changed
inputs, models, or runtime source require a new run. Conflicting saves fail instead of overwriting a
newer revision. An uncertain queue submission is never automatically repeated.

## Transfer a project

Use **Download inputs bundle** to transfer the saved recipe and referenced images/audio. Import it
through **Open Story → Import inputs bundle** on another compatible installation. Assets are checked
before restoration. Models, completed takes, approvals, and the full story store are not included.

Keep a separate backup of the complete story directory and workflow to preserve completed shots.
