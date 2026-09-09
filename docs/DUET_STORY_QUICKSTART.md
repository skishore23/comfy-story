# Comfy Story alpha: setup and first sequence

Comfy Story is the current MiniMax H3 node. The package is intended for an existing authorized H3
ComfyUI installation. It does not provision H3 access or contain models, reference images, generated
clips, datasets, or credentials.

## Prerequisites

- Python 3.11 or newer, with ComfyUI and its native MiniMax H3 nodes already working.
- Authorized H3 foundation weights, text/vision encoders, and video/audio VAEs. The public Native
  reference path stores exact RGB evidence and does not require a Duet training checkpoint.
- Linux or macOS for the durable store. Real H3 acceptance used Linux, Python 3.13.12,
  Torch 2.12.1+cu130, ComfyUI 0.34.0, frontend 1.51.9, and a 96 GB RTX PRO 6000 Blackwell GPU.
  These are the tested specifications, not a measured minimum hardware requirement.
- A persistent directory writable by the ComfyUI service account for story assets and revisions.
  Preserve this directory with the saved workflow; the workflow alone is not a project backup.
- Dependencies listed in the wheel metadata, plus `packaging` to run the installer. Use the
  ComfyUI interpreter for every installation command. Preserve its selected Torch/CUDA build.

The Comfy Story package permits the Comfy-owned
`torch>=2.8,<2.13` and `cryptography>=43,<50` profiles. Compatibility ranges do not mean every version
has passed H3 acceptance. Windows and a standalone one-click model installation are not supported.

## Install into a fresh or staged ComfyUI installation

Extract the source-only bundle and use the existing ComfyUI Python:

```bash
/path/to/ComfyUI/.venv/bin/python install.py --comfy-root /path/to/ComfyUI --check-only
/path/to/ComfyUI/.venv/bin/python install.py --comfy-root /path/to/ComfyUI
```

The check verifies bundled file hashes, installed dependency versions, and the destination. It
performs no installation. If a dependency is missing or incompatible, resolve the reported package
in the staged environment and rerun the check. The installer uses `pip --no-deps` and refuses to
replace an existing `custom_nodes/duet_story`. Its checks do not load models or prove H3 generation.

For upgrades, use a separate staged installation and store. Preserve the old node, installed Comfy Story
wheel, environment configuration, and story directory for rollback. Validate there before switching
your main installation while its queue is empty. Do not point a candidate at your only story copy.

## Configure the service

Set these environment variables for the ComfyUI process, using your own paths and verified hashes:

| Variable | Value |
| --- | --- |
| `DUET_STORY_ROOT` | Optional absolute persistent story directory; defaults to `duet_story` inside Comfy’s configured user directory |
| `DUET_STORY_MEMORY_RUNTIME` | `native-reference` for new public projects; `trained` for existing trained-memory projects |

A clean installation defaults to `native-reference`. If `DUET_STORY_MINIMAX_CHECKPOINT` is already
configured, the default remains `trained` to preserve existing installations. Set the mode
explicitly in the service configuration. Native and trained revisions are separate versioned
formats; changing this setting does not migrate existing projects. Preserve the original runtime
and store to reopen them. Native archives keep reference images, approved canon, chronological RGB
observations and exact output receipts; they do not compute learned associative operators.

Only the trained compatibility runtime requires these additional variables:

| Variable | Value |
| --- | --- |
| `DUET_STORY_MINIMAX_CHECKPOINT` | Absolute compatible Duet MiniMax memory checkpoint path |
| `DUET_STORY_MINIMAX_CHECKPOINT_SHA256` | SHA-256 of that checkpoint |
| `DUET_STORY_MINIMAX_FOUNDATION_SHA256` | SHA-256 of the matching installed H3 foundation file |
| `DUET_STORY_MINIMAX_VAE_SHA256` | SHA-256 of its matching video VAE file |

For trained memory, the checkpoint and configured model identities must agree. Request validation rejects mismatched
assets. Keep model placement and native H3 loader configuration consistent with your working host.
The service account needs read access to the model/checkpoint files and write access to story storage.

Use **Native** reference context with the native archive. **Compiled preview** is supported only by
the trained compatibility runtime and additionally requires compatible
`DUET_H3_COMPILER_CHECKPOINT`, its `DUET_H3_COMPILER_CHECKPOINT_SHA256`, and a writable absolute
`DUET_H3_COMPILER_CACHE` path. It is experimental and is not required for the first sequence.

Restart ComfyUI. Load
`custom_nodes/duet_story/example_workflows/duet-story-living-canon-two-shot.json`, or add
**Comfy Story** under **Comfy / Story**. Use your own reference images; examples do not bundle media.

## Make two shots

1. Name the project, choose **Start Story**, and supply a starting image.
   You can connect an upstream image node to the optional **Starting image** socket. For
   **Start Story** and **New Scene**, that single image takes precedence over the uploaded
   **World / starting frame** filename. This lets a Comfy image composition/editing workflow
   feed Duet directly. Existing **Next Shot** behavior still uses its connected **Previous Frame**;
   choose **New Scene** to begin from a newly staged image while retaining Story State.
   The actual starting pixels participate in execution recovery. A connected image is not an
   approved keyframe or proof that its composition, identity or state is correct.
2. Add one character and one prop to the library with stable names. Mention them as `@Name` in the
   prompt. Choose **Native** and a short duration. Run the workflow.
3. Inspect the video and evidence thumbnails. On the completed shot, stage **Keep @Name look** or
   **Keep this detail** for evidence you approve.
4. Use **Add Next Shot**. It connects Story State and Last Frame and transfers the staged decisions.
   Edit the child's prompt to use names from your library. Generate and inspect the next shot.
5. Save the ComfyUI workflow. Reopen it and run unchanged completed inputs to recover the same
   revision and stored outputs. Use **New take** when you want a new variation.

The inspector describes stored decisions and selected evidence. It does not automatically verify
that the generated frames obey those decisions. Judge identity, prop state, action, and motion.

## Acceptance and troubleshooting

Before sharing an installation, verify a real first shot, a wired continuation, saved-workflow
reload, and completed-shot recovery after a server restart. Confirm the recovered revision matches
and does not construct a new H3 generation graph. Cold model verification can take around two
minutes on the tested host; recovery saves generation work but is not instant.

- **Unknown `@Name`:** match the library name exactly and edit the child prompt before running.
- **Missing approved evidence:** restore its original support or replace the approved reference.
  Removing evidence does not erase pixels already visible in a previous-frame or scene anchor.
- **Too many exact requests:** reduce to two exact state packets for this shot. The node fails
  explicitly rather than silently dropping approved state.
- **Full history:** the current preflight limit is 128 shots; begin a new project/sequence.
- **Checkpoint or model mismatch:** use a matching authenticated checkpoint/model set. Do not
  disable digest checks to force compatibility.
- **An unfinished job was interrupted:** completed-shot recovery does not resume partial sampling.
- **A character must be absent:** stage off-screen state and use a clean scene anchor. Explicitly
  mentioning the character later requests it again.

Back up both the persistent story directory and the workflow to preserve completed takes and their
approvals. The candidate editor below can transfer a saved recipe and its source inputs; it does
not transfer completed-take approvals.

## Film project editor (release candidate)

The candidate adds **Open Story** to the existing node. It opens a host-persisted project editor
for prepared shot plans, named references, fixed seeds and complete first-cut generation. This interface is
under acceptance testing; its presence does not establish reliable multi-minute film quality.

The default **First cut** mode renders the complete saved plan without a visual reviewer.
Watch the film, edit the relevant shot cards, save, and generate again. Unchanged earlier shots
remain eligible for saved-output recovery; later dependent shots must be regenerated. Soundtrack
edits can reuse the visuals. Seeds remain locked until you explicitly change them.

For optional **Check each shot** mode under **Project tools and run settings**, configure a
locally installed compatible Qwen3-VL verifier on the Comfy host:

```bash
DUET_STORY_VERIFY_MODEL=/absolute/path/to/local-qwen3-vl-model
DUET_STORY_VERIFY_DEVICE=cuda
```

The verifier weights are separate from the H3 text encoder and are not included in the package.
Without this configuration, the panel can save/edit plans but cannot start verified production.
The existing single-shot workflow remains available. Model loading uses local files only.

1. Choose **Open Story**. Start an empty project or reuse the node's existing references.
   Name the film, add references, and supply the first shot's starting image.
2. Give each shot a visible action, relevant starting/ending facts, and its required subjects.
   Use **Must remain visible throughout** only when exits, occlusion and cutaways are forbidden.
3. Keep each seed locked for a controlled comparison. **New take seed** changes it explicitly.
   Imported seeds outside JavaScript's exact integer range are rejected without saving rounded
   values; retain those recipes in the Python CLI.
4. Save the plan and inspect the affected-shot list. Editing a shot invalidates later Story State
   descendants. Unchanged earlier recipes remain eligible for exact saved-output recovery.
5. Choose **Generate saved plan**. The default **First cut** renders the whole film as one Comfy
   job. Closing the panel does not stop it. Open the assembled film when it finishes, then edit
   the shot cards and save a new revision. It is an unreviewed draft, not a claim of story quality.
6. **Resume selected run** keeps the original revision, generation mode and attempt budget.
   The editor checks uploaded input bytes, model hashes and runtime source. Changed dependencies
   require an explicit new run. Execution failures remain visible; an unknown queue submission
   is never automatically repeated.

**Check each shot** is optional and can stop on uncertain or failed visual assessments. Its
attempt budget and **Pause after current shot** apply only to that mode. Staged openings and
selected-state recall currently require it; normal first cuts use uploaded scenes or references.
First cuts retain generation ancestry without adding unchecked footage to approved memory.
Neither mode creates creator approval.

Project recipes are immutable under `DUET_STORY_ROOT/film_projects`. Conflicting saves from two
editors fail instead of silently overwriting the newer draft. Downloaded recipe JSON contains no
media/model assets; keep the full story-store backup. Use **Download inputs bundle** when
you also need the uploaded reference images, starting frames and authored audio. The soundtrack controls accept uploaded audio; changing the soundtrack creates a new export
without changing visual shot recipes.

New portable input documents may use `shots_by_id` instead of the historical positional `shots`
list. Exactly one format is allowed. Every planned shot ID must have settings; inserting a shot
requires a new entry and does not renumber existing seeds. Existing positional inputs remain
readable. Bind them to their current plan before reordering, using `bind_film_settings` from
`duet.duetx.film_io` or the project editor's save operation.

## Move a film's inputs to another installation

Save the plan, then choose **Download inputs bundle**. The ZIP contains the saved recipe and its
referenced input images/audio with SHA-256 hashes. Repeated assets are stored once. It does not
contain H3 weights, credentials, completed clips, memory approvals or the authenticated story store.

On another compatible Comfy Story installation, choose **Open Story → Import inputs bundle**.
Import validates the entire archive before installing assets, creates a new project, and retains the
shot IDs, full stored seeds and media bytes. Input filenames become content-addressed paths inside
Comfy's input directory. Conflicting existing files are rejected rather than replaced. Generation
still requires the same compatible models and verifier; restored inputs alone do not guarantee a
bit-identical fresh render on a different runtime.

The input bundle limit is 256 MiB per asset, 1 GiB total and 512 distinct assets. The host's Comfy or
proxy upload limit may be lower. Only referenced images and audio are included. Keep a separate
backup of the original story store and workflow to recover existing completed takes and approvals.
Recipe-only JSON export remains available for small source documents.

## Experimental reference prompt format

**Prompt format → Structured reference (experimental)** separates reusable subjects, frame anchors,
retained state and the creator's action using MiniMax's reference-prompt section layout. It retains
image order, seeds, authored dialogue and quoted visible text. It does not invent story beats or
claim stronger action/identity quality. **Current** remains the default and preserves prior execution
identities. Film input rows can opt in with `"prompt_format": "Structured reference (experimental)"`.
Saved workflows and Add Next Shot retain the selected format.

This is a deterministic formatter, not MiniMax's full prompt-rewrite service: it does not expand a
short direction into a longer screenplay. Its quality effect requires matched rendering tests before
promotion. Format changes create different execution identities and do not reuse the old render as
if it were a new result. See MiniMax's
[reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
