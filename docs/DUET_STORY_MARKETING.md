# Comfy Story

## Your cast. Your approved details. Your next shot.

**Comfy Story brings creator-controlled visual continuity to MiniMax H3 workflows in ComfyUI.**
Build a reusable character and prop library, call references into a shot with `@Name`, and decide
which generated looks and details should carry forward. Keep the video, its references, and its
story revision together as you develop a sequence.

### Make the next shot with the right context

A character returns. A prop changes hands. A look needs to stay recognizable across a new scene.
Comfy Story gives creators a place to manage those decisions inside the workflow where they generate.

- **A reusable cast and prop library.** Give references stable names and use them in shot prompts.
- **Creator-approved continuity.** Inspect generated evidence and choose which looks or details to
  use in later shots. Approval records your choice; you remain the judge of the rendered result.
- **A connected sequence.** Add the next shot with story state and the preceding frame connected.
  Create a different future from an earlier revision without overwriting that revision.
- **Recover completed work.** Matching completed requests can reopen their stored video and last
  frame without another generation. Reopening verifies the stored assets and model identity.
- **Native ComfyUI workflow.** Work through one Comfy Story node with a shot panel, evidence inspector,
  and native MiniMax H3 generation.
- **A saved film workspace.** Keep shot plans, references and seeds together. Start a bounded film
  run, close the panel, and return to its progress and machine-selected candidates. Review remains
  a creator decision.
- **Deliberate shot timing.** Set precise cut lengths and carry the actual retained closing frame
  into the next shot. H3 still renders in supported duration blocks before trimming.
- **Portable creator inputs.** Export a saved recipe with its reference images and authored audio.
  Import it as a fresh project with the same inputs and seeds; model weights and take approvals
  stay separate.
- **Bring approved narration.** Connect each shot's recorded or TTS audio to keep its chosen words
  and voice in the finished output. This replaces generated sound; automatic lip sync is not included.

### Who it is for

ComfyUI creators developing recurring characters, storyboard sequences, short narrative scenes,
and concept films who want deliberate control over their visual references and accepted details.
The current alpha is for collaborators who already have an authorized, compatible MiniMax H3 setup.

### The experience

1. Add a character and an important prop to the reference library.
2. Describe a shot using their names and generate it with H3.
3. Review the result and approve the look or detail worth carrying forward.
4. Add the next shot, change the action or scene, and continue from that story revision.
5. Save the workflow and reopen completed takes when you return.

### Current availability

Comfy Story is an **internal alpha for existing MiniMax H3 users**. The distribution includes code,
an installer, and example workflows. Model weights and H3 access are supplied separately through
the user's authorized setup. New projects use an exact reference archive without a Duet training
checkpoint; existing trained-memory projects retain their compatibility runtime. Exact references
and approved state are reused, but learned H3 memory improvement is not established. The integration has been
exercised on a Linux ComfyUI H3 installation; other installations need their own compatibility check.

The alpha supports up to 128 shots in a history and two exact state packets per shot. Outputs still
need creative review: reference selection does not guarantee perfect character or prop consistency.
Completed-shot recovery is available; resuming an interrupted, unfinished generation is not.

The saved film workspace mixes the chosen authored soundtrack. Shot audio is silent by default;
the experimental **Generated speech and effects** option retains native H3 audio and directs
speech using the screenplay's language and dialogue cues. Exact wording, stable voices, and lip
synchronization are not guaranteed or automatically verified. The film editor does not add
captions; the separate command-line finishing path supports authored dialogue and optional
captions. An experimental **Full HD 2-pass** sampler renders at 1920 × 1088 with a separately
installed latent upscaler; higher resolution alone does not establish better storytelling or
motion. See the [film workflow](DUET_STORY_FILM_WORKFLOW.md) for requirements and limitations.
Unattended trials have failed story continuity or stopped within their attempt budgets. A
coherent three-minute customer film remains an unmet acceptance target, and whole-film quality
requires review.

### Shareable short description

> Comfy Story helps you build connected MiniMax H3 scenes in ComfyUI. Reuse a named cast and prop
> library, approve the visual details that matter, carry those decisions into the next shot, and
> reopen completed takes without generating them again. An internal alpha for creators with
> existing compatible H3 access.

### Suggested demo

Show a character carrying a distinctive prop. Approve a visible detail, add another shot, and show
which evidence was selected. Return to the saved workflow and reopen the completed take. Show the
actual outputs alongside the inspector so viewers can assess both the workflow and the visual result.
Use only references and generated clips cleared for sharing.

**Try it with a short sequence you can judge:** one recurring character, one important prop, and a
clear continuity decision. See the bundled quickstart for setup and first-shot instructions.

---

### Claims for reviewers and launch editors

The demonstrated value is reference management, explicit creator decisions, durable revisions, and
completed-shot recovery in ComfyUI. We have not established superiority over community workflows,
a general character-consistency guarantee, faster H3 sampling, or improved learned long-term recall.
The internal product uses a native reference archive rather than a learned associative bridge.
Optional selected-state recall carries checked visual evidence into later reference shots; its
long-film quality benefit remains under evaluation. The earlier research bridge had a zero
history-residual readout. Learned memory remains research behind an evaluation gate and is not the
basis of this alpha's marketing promise.

## Licensing

Comfy Story's code is open source under AGPL-3.0-only. Comfy has a commercial license agreement
with MiniMax for the applicable model offering. The source license and model agreement are
separate; this source release includes no model weights and grants no additional model rights.
See [licensing details](../LICENSING.md).
