# Plan, generate, and edit a film

Select **Open Story** on the Comfy Story node. Create a project, add references, and supply a starting
image. Describe each shot's visible action, required subjects, and starting and ending state.

## First cut

Save the plan, keep seeds fixed, and select **Generate saved plan**. **First cut** renders the saved
sequence as one ComfyUI job. Closing the editor does not stop it. The resulting film is a draft for
your review; generating every shot does not establish story coherence or visual quality.

Review the film and edit the relevant shot cards. Save a new revision and generate again. Unchanged
earlier shots can recover their saved outputs; dependent later shots regenerate. Use **New take
seed** only when you want a new variation.

## Audio and timing

Use the soundtrack controls to add reviewed audio. Soundtrack edits can reuse the existing visuals.
Generated speech and effects requires the H3 automatic prompt format. Supply the intended language,
exact dialogue, speaker, and timing window for each line, then listen to the output.

The export uses the saved shot durations and audio offsets. Keep source trims and dialogue within
their assigned windows. Prompt constraints do not guarantee exact speech or the absence of extra
words, so review the final mix before sharing it.

## Review and recovery

**Check each shot** is an optional mode for a host configured with a compatible local visual verifier.
It can stop on failed or uncertain assessments and supports an attempt budget. The verifier does
not replace creator approval. Staged openings and selected-state recall require this mode.

**Resume selected run** retains the saved recipe, generation mode, and attempt budget. Changed
inputs, models, or runtime source require a new run. Conflicting saves fail instead of overwriting a
newer revision. An uncertain queue submission is never automatically repeated.

## Transfer a project

Use **Download inputs bundle** to transfer the saved recipe and referenced images/audio. Import it
through **Open Story → Import inputs bundle** on another compatible installation. Assets are checked
before restoration. Models, completed takes, approvals, and the full story store are not included.

Keep a separate backup of the complete story directory and workflow to preserve completed shots.
