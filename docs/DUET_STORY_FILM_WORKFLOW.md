# Building a coherent film with Comfy Story

A finished film needs correct identities, visible causes and consequences, exact language, and an
edit. The Story node supplies reusable visual references and immutable generated revisions. The
film layer adds a screenplay, reviewed take selection, and an exact timeline. A saved generation
is not automatically an accepted story event.

The film layer provides an Open Story editor inside ComfyUI, a command for unattended first
passes, and Python APIs for reviewed finishing. The editor saves recipes and runs the host-side
film controller; the planning and export APIs live in `duet.duetx.film_plan` and
`duet.duetx.film_export`.

The editor starts with the creative inputs: story intent, cast images, shot actions and timing,
and optional soundtrack. Each shot keeps its starting image and a short setup explanation visible.
**Advanced shot settings** contains seeds, samplers, render approaches, conditioning and explicit
state checks. **Advanced film settings** contains state memory, initial facts and state meanings.
Import, export and attempt budgets are under **Project tools and run settings**. Existing settings
remain effective when these sections are closed; opening them does not change a recipe. Saving
reveals an invalid field even when its section is closed. Resume appears for unfinished runs;
shot-boundary pause is available only for checked production.
Automatic brief-to-screenplay creation is still missing; this editor uses a prepared shot plan.

## Run a complete first pass

In **Open Story**, save the plan and choose **Generate saved plan**. The default **First cut**
uses the existing `run_film` renderer to queue the complete public-node workflow, recover saved
outputs, and assemble a film. No visual reviewer is required. Watch the first cut, change shot
cards or soundtrack, save a new revision, and generate again. Locked seeds and unchanged earlier
shot recipes support recovery; downstream shots remain dependent on the changed scene.

First cuts finish as `draft_ready`, with rendered takes separate from machine-selected evidence.
They do not create approval or claim that authored events occurred. Structural input validation,
media hashes, generation ancestry, exact timeline checks and ambiguous-submission protection
remain enforced. The whole queue runs without shot-boundary pause. Advanced staged openings
and selected-state recall require **Check each shot**; they are never silently omitted.

The editor sends an explicit generation mode. Legacy API clients that omit it retain checked
production. Resuming a run retains its mode; changing modes starts a separate run.

When the first shot has an uploaded opening image and declared initial conditions, the verified
runner checks that image before queuing video. The checker receives identity references, possible
state labels and their observable definitions, without the intended action or preferred state.
A mismatch or uncertain observation stops with a setup explanation and preserves the assessment.
Reopening the same run reuses that assessment; fix the image or initial conditions and save a new
recipe to try again. Missing opening images defer observation to the generated shot and are not
reported as a preflight pass. This catches setup mismatches; it does not establish the checker's
accuracy or guarantee that H3 will preserve a correct starting state.

The customer target is one prepared story, one run, and a complete film with consistent cast,
props, causal events, and intelligible authored language. The three-minute acceptance film tests
that target without per-shot operator intervention. Improvements must live in the reusable
product; manual repairs to a demonstration do not establish unattended product quality.

The editor's **Seconds** field supports precise cut lengths from 0.125 to 15 seconds in
0.125-second increments, including values such as 2.5 or 3 seconds. This matches the existing
24 fps, integer-millisecond timeline contract. H3 generates the smallest supported 5, 10 or
15-second block that contains the cut; Duet trims its tail. The saved video and the next shot's
starting frame use that same trimmed interval. An optional ending guide targets the last retained
frame. Seeds stay unchanged when editing timing, but the changed request must generate or recover
its own matching result. Shorter cuts can support deliberate pacing; they do not guarantee lower
rendering cost, better motion, or a coherent film. Review the resulting edit normally.
Opening a new film from a configured Story node preserves its **Output duration (ms)** cut when
set; a zero or omitted value uses the node's full **Shot length**. An invalid cut is rejected
instead of silently changing the timeline.

The current command supplies execution, recovery, and finishing. It still requires a prepared
screenplay, references, and optional authored sound. Automatic screenplay creation remains product work. The Open Story editor supports authored
shot plans, uploads, saved projects, bounded generation, and review. The experimental verified mode below adds bounded automatic
retries; its effect on film quality must still be demonstrated.

Install the current Duet package and restart the matching Comfy node. Upload the cast, prop,
world and optional shot-audio files to Comfy as usual. Save a `FilmPlan` as JSON and an input
document containing a `library`, an ordered `shots` list, and optional authored `audio` tracks.
Then run:

```bash
duet-story-film --plan film-plan.json --inputs film-inputs.json \
  --server http://127.0.0.1:8188 --output artifacts/my-film/first-pass
```

The equivalent source command is `python -m duet.duetx.film_runner`. Each shot input contains
`world` (an uploaded filename), `variation` (a seed), and optional `audio_file` (an uploaded file
for the node's Authored audio socket). Set `intent` to `Next Shot` or `Continue This Shot` to
carry the actual preceding Last Frame; `world` may then be `null`. The default `New Scene`
requires an uploaded world. The first shot always starts a new story. The library uses the
same names, roles, notes and uploaded
filenames as the ordinary Story panel. The input document's `audio` entries contain `path`
(relative to that document), `sha256`, `start_ms`, `duration_ms`, optional `gain`, and `cue_id`
for speech. Every planned dialogue cue requires exactly one matching authored track. These
finishing tracks are independent of the optional per-shot audio socket.
Set the input document's `"burn_subtitles": false` to keep dialogue captions out of the picture.
The default remains `true` for existing saved runs; no captions appear when the plan has no
dialogue. The switch must be a JSON boolean. Changing it changes the run identity, so use a new
output directory. Captionless runs do not require FFmpeg's libass filter. The SRT sidecar remains
available separately when dialogue is present.

The command compiles one ordinary Comfy graph of public Comfy Story nodes, connected by their
Story State outputs. It queues the graph once, waits, retrieves its exact generated videos,
checks actual generation ancestry, and streams them through the existing film compositor.
The compiler carries persistent facts for visible entities and explicitly referenced fact
namespaces into each shot. Use entity names as fact prefixes (for example `Leon.place`);
an explicitly required prop fact such as `cup.owner` also brings in other `cup.*` facts.
Facts being changed become rendering goals, with an opening condition only when `requires`
explicitly specifies one. Unattended candidates label preceding planned effects as unverified;
reviewed compilation uses the accepted prefix. Neither prompt compilation nor a successful
render establishes that the requested state appeared in the video.
The public Story node enters its durable recovery path on each queue. This refreshes its output
receipts even when Comfy would otherwise reuse an expanded graph without returning the internal
inspector output. Matching completed requests return their stored video without another sampler
graph; Comfy's transient cache is not the authority for completed Story output.
It does not hold all film frames in one large image batch. Each shot uses the smallest supported
5/10/15-second generation window that covers its planned duration; the first planned interval
is selected automatically with no handpicked trim. The result has the exact planned runtime
and the authored soundtrack. Speech and captions are optional; a speech-priority mix applies
when dialogue is supplied. Arbitrary scene text remains unsupported.

The output includes `workflow.json` (also importable as a Comfy API workflow), the queue ID,
source videos, plan, input content hashes, generated-take records and `export/film.mp4`.
An operating-system lock prevents simultaneous launches from submitting the same run twice;
it is released when the command exits, including after a crash.
Re-running the same command resumes the known job or returns its verified completed film.
Changed inputs or replaced uploaded bytes require a new run directory. A failed finishing pass
can retry from the saved renders in a new export-attempt directory without another GPU job.
An uncertain initial submission is not automatically repeated; recover its prompt ID from
Comfy before retrying so a network failure cannot silently double the generation cost.

This path creates **unreviewed output**. It does not fabricate `AcceptedTake` records, advance
reviewed facts, approve pending canon, repair shots or change prompts between shots. A coherent
first pass must be demonstrated in the actual film. The command automates execution and export;
it does not automatically repair or approve shots. Use the separate verified controller below
for bounded candidate retries, or Open Story to run that controller from ComfyUI.

An optional local diagnostic audit is available through `duet-story-audit`. It requires a local
Qwen3-VL model in safetensors format, Pillow, and compatible Transformers version 4.57 or newer.
Use the installed command in an already compatible Comfy environment. For source-based setup,
install the optional `film-audit` extra into a separate diagnostic environment:

```bash
python3.11 -m venv artifacts/film-audit-env
artifacts/film-audit-env/bin/python -m pip install -e '.[film-audit]'
```

Run that environment's `duet-story-audit` command against the completed film files. This keeps
the source checkout's research Torch 2.8 requirement separate from Comfy's validated Torch/CUDA
installation. The internal runtime wheel uses the Comfy dependency profile.
It samples nine frames per selected shot, compares them with the source references and intended
action, and produces a separate blind retelling from uncaptioned source midpoints in film order.
The blind visual review excludes authored narration and captions so the script cannot supply
missing visual causes or actions. Check the final soundtrack and caption rendering separately.
Model findings are bound to source, plan, model and sampled-frame hashes. Missing evidence or
malformed output remains `needs_review`; the audit never creates creator approvals or updates
canon. Its machine judgments require validation before use as an automatic acceptance gate.

```bash
duet-story-audit --run artifacts/my-film/first-pass --inputs film-inputs.json \
  --comfy-input /path/to/ComfyUI/input --model /path/to/local/Qwen3-VL-8B-Instruct \
  --device cuda --output artifacts/my-film/visual-audit
```

Run the audit after generation with sufficient free model memory. It operates on local files
and loads the supplied model with remote code and network model loading disabled. It does not
check audio, exhaustively inspect every frame, or establish a film-quality guarantee.

## Experimental verified production

Use this mode to prevent a known failed event from becoming the next shot's parent. It operates
through the same public Story nodes and durable recovery as the first-pass command:

```bash
duet-story-film --plan film-plan.json --inputs film-inputs.json \
  --server http://127.0.0.1:8188 --output artifacts/my-film/verified \
  --verify-model /path/to/local/Qwen3-VL-8B-Instruct \
  --comfy-input /path/to/ComfyUI/input --verify-device cuda --max-attempts 2
```

The attempt budget is 1–4 per shot, including the original candidate. The command validates the
whole plan and soundtrack and loads the checker before rendering. For each candidate it queues
only the selected prefix and the candidate, reviews the candidate's sampled frames, then either
advances, retries, or stops. A retry changes the seed and reinforces bounded action and ending
targets from the saved plan. Rejected observations remain in the assessment and stop reason; they
are not copied into generation instructions. It keeps the original shot contract and the exact
selected parent. Retry targets cannot activate additional reference mentions. Rejected attempts
remain available for inspection. This separation does not guarantee that a retry achieves the action.
Matching preceding nodes recover stored outputs; they should not sample again.

`production.json` records progress, assessment hashes, selected revisions, and any stop reason.
Each shot/attempt directory preserves its plan, inputs, run, Comfy prompt ID, source videos, and
raw visual assessment. Re-run the same command and directory after an interruption. Unknown
submission outcomes still require prompt-ID recovery; the controller will not guess and resubmit.
A changed plan, input document, checker identity, or retry budget requires a new production directory.

An uncertain or malformed check stops without another render. An explicit failure can consume the
remaining attempt budget. Exhaustion stops before downstream shots are queued. After every shot
passes, the command repeats the shot checks and performs a blind whole-film retelling. A machine
pass produces `ready_for_review`, never creator approval. The report binds observations to source,
reference, model, frame, and response hashes; changed evidence cannot silently reuse a saved pass.

This is an experimental scheduling safeguard, not a customer-quality certification. The checker
samples frames, can miss motion or misread events, and has produced false positives in real films.
A broad "coherent" retelling can omit the intended motivation; the optional intended-story check
below compares that retelling with the creator’s four story fields. Audio and fine visual detail still
need separate assessment. Prefix exports also repeat finishing work; avoiding that overhead is a
future optimization. Measure automatic repair success and false passes against independent human
reviews before claiming customer-quality reliability. Open Story exposes this controller through **Check each shot** under **Project tools and run
settings**, with a fixed attempt budget and persisted status. It is optional; First cut is the
default delivery path.

## Write the film before rendering

Choose a small cast, one clear objective, an obstacle, an attempted solution, and a visible
resolution. Give every shot one purpose. Establish a mechanism before using it to resolve the
story. Show a character arriving before cutting to them working inside a new location.

For a three-minute film, assign shot durations that total exactly 180,000 milliseconds. Generate
longer source clips when necessary and select continuous intervals in the edit. Avoid extending
weak clips merely to reach the requested runtime.

A `FilmPlan` contains:

- Stable project and shot identifiers, the title, languages, and exact target duration.
- Initial facts such as `key.owner=Sol`, `light=off`, and `Mira.location=workshop`.
- Each shot's purpose, action, present and absent entities, required facts, and intended effects.
- Earlier shots that the current action depends on.
- Exact narration/dialogue and visible-text cues with language and local millisecond timings.

`FilmPlan.validate()` checks planned causality and timing. This establishes that the screenplay
is internally consistent; it cannot establish what the generated video actually depicts.

## Separate identity from old actions

A reference showing a key handoff can induce another handoff even when the next prompt requests a
different action. Review each reference for incidental people, hands, props, writing, and poses.
Use an approved face/costume reference and a clean prop reference when the original scene contains
an unwanted event. Preserve the original evidence and record any derived crop.

Use `Composition = New composition` for a cut to a new camera setup. This retains native H3
reference conditioning but omits the first-frame latent guide and the inherited inclusive scene
image. For Next Shot or Continue This Shot cuts, the previous frame is also omitted from
reference conditioning; select at least one visual reference or the cut fails before generation.
Start Story and New Scene retain the explicitly supplied world image. Saved associative state,
canon, and explicitly selected historical evidence remain available.
This avoids forcing an old scene into a new setup; it does not establish learned-memory quality.
Internal experiments can still explicitly compare inclusion of the scene image. The default `Continue frame`
behavior remains available for shots that should begin from the supplied frame. New composition
is currently an H3 option. It is not a guarantee that H3 will follow a requested camera angle.

Native reference conditioning sends original identity and approved current-state images separately.
It does not rasterize those two sources into a contact sheet. The saved evidence packet and its
provenance remain unchanged. The current-state image takes precedence for mutable state; the
original supplies recognizable identity. Two recalled packets can therefore consume up to four
image slots within H3's nine-source limit. Compiled preview and historical research defaults retain
the previous packet layout for reproducibility; those paths remain experimental.

Reuse reviewed location and machinery frames when returning to the same setting. A verbal
instruction such as “the same four-spoke handwheel” alone is insufficient evidence of its design.
Keep characters absent from both the shot roster and active reference mentions when they should
remain elsewhere. The film prompt compiler does not turn absence names into `@` references.

Set `FilmShot.reference_names` when visible entities should not all supply separate images. For
example, the current doorway frame can establish the lighthouse location while only Mira and
the medallion activate identity/prop references. The selected names must be a unique subset of
`present`. With an explicit scope, mentions inside prose and facts do not activate additional
images. Omitting the scope retains the previous behavior and historical shot digest.

## Review before advancing the story

`compile_shot_prompt(plan, index, accepted_takes)` checks the actual accepted prefix before
preparing the next shot. Out-of-order production can use `compile_candidate_prompt(plan, index)`;
its output is an unreviewed candidate based on intended screenplay facts. It does not advance the
accepted story. Keep the candidate's actual generation parent and inputs in its render receipt.

An `AcceptedTake` binds a review to the exact shot contract, Duet revision, media digest, selected
source interval, and preceding acceptance. The reviewer records observed state changes and a
specific note. `accepted_state` rejects an out-of-order, stale, or undersized take and rejects
observed changes that differ from the intended effects.

Review the selected interval for identity, wardrobe, prop shape and ownership, location, action,
background writing, and causality. A technically successful render can still be rejected. A
failed key handoff must not establish a successful ownership change. If the screenplay is revised,
record the reason and review any affected downstream takes again.

The acceptance chain protects review records from accidental reuse after edits. It is not a
computer-vision verifier, and a caller could supply an incorrect human judgment. Preserve review
images or timestamps so another person can assess the decision.

The editorial acceptance chain does not rewrite the generation ancestry of its source takes.
If an edit combines candidates from different branches, the last clip's `Story State` does not
contain the whole edited film. Continue through the reviewed film facts and explicitly transfer
appropriate approved canon/evidence to the generation branch. In particular, a screenplay fact
such as `light=bright` does not automatically replace an earlier dark-lamp canon note or image.
An integrated editor needs an explicit, provenance-preserving operation for rebuilding continuity
from accepted selections; the current exporter does not perform that operation.

## Author language and sound explicitly

Choose sound from the creator's brief. Narration, dialogue, and subtitles are separate creative
choices; do not add them to explain an event that the footage failed to depict. A film without
speech uses empty `dialogue` and `text` arrays in every shot, and only instrumental music or
nonverbal ambience in the input document's `audio` tracks. With no authored audio, the export is
silent. Generated H3 audio is always excluded from the film export, so an unrequested generated
voice cannot leak into its soundtrack. Review music and ambience too: an instrumental prompt
does not guarantee that a sound model produced no vocals.

When speech is requested, author it separately from picture. Keep on-camera characters silent
unless a dedicated dialogue/lip-sync workflow has been validated. Check the transcript against
the authored line and review pronunciation, pacing, and voice continuity. Automatic transcription
is supporting evidence, not a substitute for listening. None of these controls fixes missing
visual causes, premature state changes, or incorrect character actions.

Check the assembled soundtrack as well as the source voice files: music can obscure words that
were clear in isolation. The exporter defaults to `prioritize_speech=True`: it normalizes each
spoken asset to -16 LUFS and lowers non-dialogue tracks to one quarter of their configured gain
during the actual speech asset, with a 150 ms lead-in and 350 ms release. This uses film time even
when a background track starts later. Set `prioritize_speech=False` for an already mastered mix.
Source assets remain unchanged, and the export receipt records the speech-mixing policy. This
improves mix control; it does not certify intelligibility or repair a mispronounced source line.

The node's optional **Authored audio** connection accepts a reviewed mono or stereo Comfy AUDIO
track, such as a recording or the output of a chosen TTS workflow. It replaces generated H3 audio,
pads a shorter track with silence, and rejects a track that would be truncated. Its actual samples
and sample rate participate in completed-shot recovery, so changing the dialogue cannot recover
an old soundtrack. The node does not provide automatic lip sync or certify pronunciation. Use
the same approved voice configuration across the film and provide each shot's own line.

`FilmAudio` binds each spoken asset to its cue ID and exact start. The exporter verifies that
all dialogue cues have matching assets and that each asset fits its allotted window. Source H3
audio is excluded. Music and ambience are explicit timeline assets with controlled gains.

`subtitle_srt` preserves the authored dialogue and timing. The exporter can burn those captions
into the picture. Arbitrary `FilmShot.text` overlays are not currently composited by the exporter;
use an explicit typography pass for title cards or other writing and retain its source and receipt.
Do not rely on generated signs or labels to carry essential plot information.

## Export and evaluate the actual film

`export_film` requires a complete accepted sequence, verifies source and audio hashes, checks
source trim lengths, and conforms clips to exact frame durations. It produces an MP4, SRT,
canonical plan, accepted-take record, and receipt identifying the final bytes. FFmpeg and ffprobe
are required; burned captions also require the libass subtitles filter. Output is currently 1344×768, 24 fps by default, with optional burned captions.

Review the assembled film as well as individual shots. Check screen direction, geography,
repeated actions, sudden prop or costume changes, dialogue intelligibility, caption readability,
music balance, and whether the opening problem is visibly resolved. Exact runtime and a valid
receipt do not prove any of those creative qualities.

Keep source media, models, audio, exports, and review artifacts in ignored project storage. Commit
the reusable source and tests. Share only a reviewed export through the intended access boundary.

## Product work after the production proof

Bring these same contracts into the existing Story side panel: screenplay and timing, evidence
preview, candidate review, accepted cut list, stale downstream indicators, and export. Keep one
public product rather than introducing separate research-facing nodes for each responsibility.

Measure finished-film success, accepted seconds per render, correction time, repeat-event rate,
prop/identity failures, and language errors against a matched native H3 workflow. Existing
associative-memory benchmarks do not establish a narrative-quality advantage. That claim needs
film-level evidence with the same script, model, references, sampling budget, and review criteria.


## Faster iteration with MiniMax H3 Turbo

Set the public node's advanced **Sampler** to **Turbo 4-step**, or add
`"sampler": "Turbo 4-step"` to each desired shot in the film runner's inputs JSON.
Omitting this field preserves the standard 20-step sampler. The same Story State chain,
reference selection, authored audio and completed-shot recovery run in both modes.

Turbo requires `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` in Comfy's
`models/loras` directory and the core `MiniMaxH3SigmaShift` node. It uses LoRA strength 1,
Euler, the simple scheduler, four steps, video/audio shifts 12/3, and reference size `match`,
following the [author's Ref2VA recipe](https://github.com/ModelTC/Minimax-H3-Turbo).
Duet keeps its current 1344×768 output resolution. The adapter was trained at 544p;
quality at this larger resolution needs evaluation. Turbo reduces sampling work and reference
tokens; model loading, text encoding, decoding and memory commits still take time.


**Turbo 8-step** also supports Reference shot, using
`minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` in `models/loras`.
This 768p adapter uses eight Euler/simple steps, LoRA strength 1, video/audio shifts 12/3,
and reference size `match`, following the [author's release](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/51).
The selected render profile determines the adapter. Existing Animate frame Turbo 8-step runs
keep their original first-frame adapter and sampling configuration. Missing weights cause an
explicit error; Duet does not substitute another adapter. Seeds and saved sampler choices are
preserved, and the adapter bytes contribute to the execution fingerprint. This option does not
guarantee coherent action, identity, language, or a finished film; evaluate it before delivery.

Use Turbo for quick story and composition experiments. Four steps do not establish story,
identity, dialogue, or text correctness. Review Turbo outputs separately and run the final
acceptance film through the chosen delivery sampler. A sampler change or a change to the
installed LoRA bytes creates a distinct execution fingerprint, while preserving compatible
Story memory. Turbo uses built-in Comfy nodes; the separate optional SPEED plugin is unrelated.


Measured on Forge1, a fixed ten-second shot took 221.81 seconds with Turbo, versus 429.87
seconds for the recorded standard run (about 1.94× faster end to end). Turbo denoising took
74 seconds. A second Turbo run with a new seed and warm models took 106.15 seconds
(about 1 minute 46 seconds), with a new generated revision rather than recovered output.
This comparison includes different process/cache conditions, and the Turbo preset
also changes reference resizing; it is not a controlled model-only benchmark. Both outputs are
1344×768, 243 frames, with authored audio. The opening is clearly outside the café; later
framing produced disagreement between visual inspection and the local automated audit, which
reported an outside location and machine pass. Treat spatial continuity as needing review.
Faster iteration is demonstrated; equivalent continuity quality is not.


## Exact continuation boundaries (release candidate)

The film compiler passes each planned duration to the public Story node. The node trims decoded
video and audio before saving the clip and committing its last frame. A five-second shot therefore
uses 120 frames at 24 fps even when the model decodes 124. The next shot receives the last frame
retained in the edit; discarded padding no longer becomes its starting pose. No frame interpolation
or speed change is applied. Too-short decoded media fails before publication.

The advanced **Output duration (ms)** input is included in completed-shot recovery identity. Its
default of zero preserves existing single-shot workflows and their native frame counts. Film graphs
set an exact interval automatically, so existing completed shots cannot silently satisfy the changed
output recipe. This fixes an edit-boundary mismatch; it does not establish action or camera fidelity.

### Opening-state assessment

Protocol v13 checks every scoped starting condition using a separate observation of frame zero,
including facts a return shot must preserve. That call receives identity references, the opening
frame, and the authored state alternatives and meanings. It receives no preferred answer, action,
or later frames. The action review still sees the sampled interval and checks the facts
that must remain true. Both raw responses and frame hashes are preserved and revalidated by the
production controller. A matching opening cannot clear another failed check; missing or malformed
opening observations require review. This adds one assessment call for shots with starting facts,
with the same single bounded retry for malformed JSON. It does not add video-generation attempts.

This separation addresses temporal confusion in the checker; it does not establish that the
vision model is always correct or that H3 will obey a camera or action request. Opening facts
must describe visible states. Hidden contents, intentions and other unobservable facts can stop
a run as uncertain and should not be used as visual acceptance requirements.

### Explicit camera policy

Set a shot's **Camera policy** to **Locked frame** when framing must stay fixed. The compiler adds
that requirement to the generation prompt, and a separate assessment compares every sampled frame
to the opening view using background scale, position and cropping. It receives no character
reference images or story target to distract from framing. Changed framing rejects the candidate;
missing or uncertain observations require review. The raw response is hashed and checked again
before any candidate is selected. Movement between sampled frames can still be missed.

The default **Follow prompt** preserves existing shot identities and allows intentional camera
movement. **Continue frame** alone does not mean the camera should be locked. The separate camera
assessment adds one model call, with one retry only for malformed JSON. Neither setting guarantees
H3 obeys the requested camera direction, and neither is a replacement for independent film review.

### Optional ending guide

A shot can include an **Ending image (optional)** uploaded before generation. The public Story
node also exposes an **Ending frame** image input. The film compiler uses ordinary LoadImage and
native MiniMaxH3AddGuide nodes; the guide targets the last frame that will actually be saved, not
extra decoded frames outside the planned edit. A legacy native-length shot targets its native last
frame. Only a single image is accepted as an ending guide.

Ending images are included in portable input bundles and generation recovery identity. Changing
an ending image invalidates reuse of that request. Existing recipes without one keep their old
workflow behavior. An older installed node rejects the unsupported guide before a GPU submission.
This is an optional conditioning tool, not a guarantee of the exact ending pose or pixels. It must
be evaluated for motion, identity and transition quality before making an improvement claim.

### Declared cast and conditioning images

The film's `present` names describe the declared cast. Its `reference_names` setting selects
conditioning images; an empty list omits those images without removing the declared cast from
pending memory observations. The compiler passes the cast through the advanced **Scene entities**
node input, independently of image selection. Changing that declaration changes generation
recovery identity. An older node without this input is rejected before film submission.

These are creator-declared labels on pending evidence, not proof that the model rendered those
entities and not creator-approved canon. Visual review still checks presence and identity. Ordinary
saved node workflows with a blank Scene entities input retain their existing reference-based
observation behavior. A JSON empty list explicitly declares no scene entities.


## Render profiles

The Story node and each film shot offer two explicit profiles. Saved workflows retain
`Reference shot` unless the creator selects another profile.

- **Reference shot** uses the H3 reference checkpoint and selected cast/evidence imagery.
  Use it when the shot needs an identity reference, recalled appearance, or new composition.
- **Animate frame** uses the H3 first-frame checkpoint to animate the visible starting scene.
  It requires `Continue frame`; it does not use cast reference pictures or the reconstructed
  memory image as visual conditioning. Names and declared scene entities still receive pending
  observations, and approved text state still informs the prompt. It does not reconstruct an
  absent character from learned memory. Image-recall and restore-original commands require
  Reference shot instead of being silently ignored.

Animate frame fits the starting image to the 1344×768 canvas with a center crop, preserving
proportions. Check that the starting composition includes the intended cast. The optional ending
image is anchored to the final saved frame; durations and continuation use the same exact trim.
Native sampling uses 20 res_multistep/beta steps. Turbo uses the separate 768p first-frame LoRA,
four Euler/simple steps, and video/audio shifts of 6/3. Reference Turbo retains its own LoRA and
12/3 shifts. These profiles are not interchangeable model files.

Animate frame additionally requires `minimax_h3_fl2va_pruned_int8_convrot.safetensors`; its Turbo
option requires `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors`. Reference-only
projects do not require these optional weights. The shared text encoder and video/audio VAEs
remain the same. Missing weights fail before GPU submission in the film editor. Model bytes,
profile, seed, inputs and implementation are part of run/recovery identity. Changing profiles
requires a new run; it does not silently reinterpret an old take.

CLI shot settings select the same feature with `"render_profile": "Animate frame"`. The screenplay
must explicitly set `"composition": "Continue frame"` for those shots. An older installed Story
node fails preflight before submitting a profile it does not support.

This is a creator control, not a guarantee of character counts, action completion, or coherent
multi-minute films. The native first-frame path showed narrower positive behavior in a small
paired action diagnostic; broader customer acceptance is still required. Publisher specifications
for the distinct Turbo variants are at
[MiniMax H3 Turbo model specifications](https://github.com/ModelTC/Minimax-H3-Turbo#1-model-specs).


### Runtime package changes and recovery

Film identity records the installed versions of the generation, attention, decoding and verifier
packages, including Comfy Kitchen, AIMDO, Triton, kernels, tokenizers, safetensors, Accelerate,
Pillow and PyAV alongside Torch, Transformers and NumPy. A missing optional package is recorded as
unavailable, so adding it also changes the runtime identity. A package-only update therefore
requires an explicit new film run even when model files and source code are unchanged.

Keep the Comfy source and its required package versions together in a reproducible installation.
Do not update a running film's environment in place. These checks run at start/resume; they do not
lock installed files throughout generation or promise identical regeneration across hardware.
Completed stored videos remain the authority for exact take recovery.

### Optional final-frame counts

Use **Ending visible counts** for a creator-declared category that must have an exact visible
count at the end of a shot, for example `red mugs=2` or `people=0`, one category per line. Leave it
blank to disable this additional check. Categories are scene inputs, not built-in story rules.
The field supports up to eight categories and integer counts from zero to 64. Changing it changes
the shot and recipe identity; historical recipes with no counts retain their existing identities.

The renderer receives the requested count. A separate verifier call sees only the category and
final edited frame, without the expected number, character-reference portraits or intended story.
It inventories distinct physical instances, including distinguishable partial instances. Code
compares the observed number with the saved requirement. A mismatch rejects the take; an unknown
count or malformed response after one format retry requires review. Counts inconsistent with the
listed instances cannot pass. A matching count cannot clear an action, identity or camera failure.

This check protects a specific continuation boundary. It does not establish identity, detect every
interior-frame duplication, or prove visual counting reliable across arbitrary categories. A
five-image diagnostic detected known duplicate subjects while preserving three clean controls on
the larger verifier. The smaller verifier produced an inconsistent inventory on one overlapping
frame; the parser rejected that response, but its allowed format retry then missed the extra
partial subject. Model choice therefore still matters. Broader category and occlusion validation is
still needed.
No machine result grants creator approval.

### Inspecting a candidate without approving it

New runs record a preview after each completed render/export, before visual checking. **Preview
latest candidate film** plays that rendered prefix with the chosen film soundtrack or declared
silence. It may include rejected or unchecked footage and may be shorter than the requested film.
The preview remains inspectable when checking stops; opening it never changes selection, approval,
retry limits or run status. The normal completed-film endpoint still requires ready-for-review.

The host verifies preview bytes and confines the file to its run directory before serving it.
Older runs without a recorded preview retain their existing selected-take controls. Individual
visual take previews are muted by default because they contain source audio, which can differ from
the assembled soundtrack. Use the candidate film preview to judge the soundtrack.

### Run progress

The film editor distinguishes checking saved inputs, preparing the video reviewer, rendering a
shot, reviewing a shot and reviewing the complete film. The host persists the current stage so it
remains available after closing and reopening the editor. A stopped or paused run shows its last
stage alongside its status and reason. Older runs without a recorded stage keep their existing
status display; no progress is inferred. Finishing machine checks still means ready for review,
not creator approval.

### Blind narrative review contract

The final controller requires the blind retelling to contain bounded, nonempty descriptions of
the summary, protagonist's goal, decision and outcome, plus a valid coherence label and a list
of contradiction descriptions. A bare `coherent` label cannot make a film ready for review.
Malformed or incomplete retellings stop with completed candidates preserved; contradictions or
uncertainty also stop. Protocol v7 prevents reusing older assessments under this contract.

This validates assessment completeness, not the truth of the model's observations. The retelling
still uses sampled source midpoints and is not an exhaustive motion, audio or story-intent check.
Independent viewers must still establish whether the intended causes and outcomes are visible.

### Verifier inference preflight

Duet normalizes an empty optional kernel HTTP user-agent suffix to `None`. This works around
the `kernels 0.15.2` / `huggingface-hub 1.25.1` combination producing a trailing space when
ComfyUI disables telemetry. The HTTP client otherwise rejects the publisher lookup before it
reaches the server. Telemetry remains disabled and publisher verification remains required;
the adapter does not change credentials, endpoints, or nonempty user-agent values.

Before the verified controller submits a video, its local checker now processes a synthetic
single-color image and generates one token. Loading model weights alone does not initialize lazy
quantization or attention kernels. This preflight exercises those paths without using project
images, producing an assessment or consuming a video attempt. A failure stops before rendering.
It does not establish checker accuracy or guarantee that every later prompt fits available memory.

Some quantized verifier runtimes load separately distributed kernels and verify their publisher
online, even when model weights are local. Pin and test those dependencies in the same environment
as Comfy. Do not treat `local_files_only` on the model loader as proof that every kernel dependency
works offline, and do not disable publisher-trust checks to make a test pass. Public kernel metadata
can be accessed without implicit account credentials; private model downloads require their own
explicit access configuration.


### Explicit frame-border preservation

The optional per-shot `border_policy: "Preserve opening borders"` is available as **Frame borders**
in the film editor. It adds the canvas requirement to generation and compares near-black edge
rows and columns in the nine sampled output frames with the opening frame. A change greater than
one percent of the corresponding dimension fails the temporal-stability check even if the visual
model reports a pass. A nearly black frame is unobservable and requires review. Findings cite the
opening and later source-frame indices and measured border fractions. Ordinary bounded retries
consume this finding; the checker never crops, pads, stabilizes, or approves the video.

Keep **Follow prompt**, the backward-compatible default, for projects that intentionally change
borders. Unchanged existing letterboxing is permitted by preservation. This conservative check
does not detect every matte, enforce a wide shot, measure camera shake, or inspect unsampled
intervals. It is a specific artifact check, not a general quality guarantee.

The version-8 blind retelling receives uncaptioned opening, midpoint and ending frames for each
shot in plans of at most 32 shots. Longer plans retain midpoint sampling to bound the additional
image load. The prompt distinguishes incompatible visible states from ordinary independent
behavior or delayed responses. Missing evidence remains uncertainty. Neither these samples nor
a valid retelling replace independent full-film and soundtrack review.


## Start another film

Choose **New film** in the film editor to start a blank draft. Save any current edits first.
The new project has its own shot IDs and seeds, an empty reference library, and no soundtrack
or captions. Add the story and starting images, then save. Previously saved projects remain in
**Open film project**; creating a draft does not pause or replace an existing host run.

The installed `example_workflows/` folder contains Comfy graph templates only. Prepared CLI
screenplay and input examples live in `custom_nodes/duet_story/example_films/` as
`film-first-pass-plan.json` and `film-first-pass-inputs.json`. These are film data, not graphs to
open in Comfy's template browser. Supply your own authorized reference assets before generating.

## Independent ending-state review

When a shot declares or carries visible facts, the verified controller also inspects its final
frame separately from the intended action. The observer receives identity references and the
possible state values used in the screenplay, with no preferred ending value. The controller then
compares that observation with the required ending. Changed facts use their authored effects;
other scoped facts must still hold at the end.

A contradictory ending uses the same bounded retry for the current shot, before later shots are
queued. Uncertain or malformed observations require attention. The final frame, response and state
vocabulary are bound to the assessment; resuming cannot silently substitute different evidence.
The complete screenplay supplies the vocabulary even while only an early prefix is being rendered.
Older runs retain their original protocol and cannot be resumed under a changed verification policy.

This is an additional machine check, not proof of rendered correctness or creator approval. It
can miss visual distinctions and does not inspect every intervening frame. Facts that cannot be
visually established may stop a run; use explicit, visible conditions in prepared screenplays.

## Change music in the film editor

Upload authorized audio to Comfy through its ordinary Load Audio control, then enter the uploaded
filename under **Soundtrack → Uploaded soundtrack filename** and choose **Add soundtrack**.
The host checks that the file contains audio and supplies its duration and SHA-256. No manual hash
or recipe JSON is needed. Files must remain inside Comfy's input directory and be at most 256 MiB.

Set the track's start time, duration and volume, or remove it. A new track starts at zero and uses
the shorter of its source duration and the film duration. Multiple tracks mix together; adding a
replacement does not silently remove another track. There is currently no source-offset or
waveform editing control. Imported speech cue bindings are retained.

Save the revision and inspect the edit impact. Soundtrack-only changes retain visual recipes;
the previous recipe and run remain available. Generate the saved revision for its new export.
Saved visual requests can be recovered, but this does not bypass current verification or promise
that an independently rechecked run will receive the same machine decisions. Use no tracks for a
silent film. Adding music does not enable generated shot audio, narration or captions.

Canvas preservation is phrased as matching the opening picture area and margins. The renderer
does not enumerate unwanted border artifacts in this instruction, because such words can
condition their appearance even when negated. The separate pixel-based border check still applies.
Prompt wording is guidance, not a guarantee of stable framing or correct action.

### Inspect your cast and shot guides before generating

The film editor previews uploaded cast, starting and ending images beside their filename controls.
Open a full image to inspect the composition. Changing a filename updates its preview; clearing an
optional guide hides it. A missing file shows an unavailable message. Viewing an image does not
save a revision, generate a shot, or approve its contents.

Animate frame animates the starting scene without separate cast-image conditioning. Stage the
characters and props needed for the action in that scene. Reference shot can use cast and recalled
images. A preview helps inspect the supplied material; it is not an automatic staging or identity
check, and an ending guide is not proof that the renderer will execute the transition correctly.

### Give state labels an observable meaning

Open **State meanings (optional)** in the film editor to add a state key, a possible value and
what must be visible to establish that value. For a parcel, `Parcel.position=in_box` could mean
that the box supports and encloses the parcel; being beside the box is a different state. Define
all planned values of a key once you add meanings for it. You may also define unwanted alternatives
that never appear in the intended screenplay. Keep definitions independent of shot numbers and
expected outcomes. Remove a definition with **Remove state meaning**.

Write meanings as descriptions of visible conditions, not instructions to the checker. For
example, describe which surface supports the parcel. Do not put JSON formatting, scoring rules or
"return null" instructions in a meaning: relevant meanings also reach the video model. The
checker already has its own uncertainty and output-format rules.

These are ordinary creator data. Runtime code contains no special entity names, locations or
story-specific state tests. Generation receives the meanings relevant to its current starting and
ending targets. The visual checks receive alternative meanings for the keys being assessed. The
independent ending check still receives no desired ending value or intended action. An unclear
observation remains uncertain; a definition does not turn a planned fact into an observed fact.

Portable plans store optional `state_definitions` records with `key`, `value` and `definition`.
They are limited to 256 unique key/value pairs, with nonempty values and definitions of at most
2,000 characters. Definition text does not activate `@` references. The empty default preserves
historical plan serialization and acceptance roots. Reports record the declared meanings and bind
them to the plan under audit protocol v11.

Editing a meaning changes the saved plan. The editor invalidates the first affected generation
prompt and its descendants; unchanged visual recipes may still recover saved bytes. Existing
creator acceptances cannot silently establish newly defined states: definitions also bind the
acceptance chain's initial root. Code authoring acceptance records can obtain that root with
`initial_acceptance_digest(plan)`. Original runs and their reports remain intact.

Definitions help make requirements assessable. Their semantic clarity, the model's observation
accuracy and the renderer's execution still need separate evaluation. This feature is not a claim
of general story-quality improvement or automatic screenplay creation.

Generation prompts describe requested quantities as things to depict. The separate counting
check inventories actual instances, including distinguishable partial objects; its counting
instructions do not belong in the rendering direction. When several unconfirmed reference notes
are present, one shared precedence rule keeps those notes subordinate to the shot direction.
Reducing repeated instructions does not establish better action-following or film coherence.

### Model verification startup

Verifier identity checks read the actual local model files, so cold startup can take time.
File timestamps alone are not treated as proof that model bytes are unchanged: rapid edits can
retain the same metadata on some filesystems. The checker compares the file inventory before
and after verification and rejects detected changes during a file read. Keep model files fixed
throughout a production run. Historical model digest formatting is preserved. This change adds
mutation checks; it does not establish faster startup or broader generation determinism.


### Locked camera verification

Protocol v11 compares each later sampled frame with the opening frame in a separate model call.
Nine sampled frames require eight comparisons. Each comparison receives only those two images
and the framing instructions, without the intended action or cast references. This addresses a
known development failure where a nine-image assessment accepted visible camera tracking.
The same model detected the movement with paired inputs; this is not a calibrated accuracy claim.

Every raw pair response and its frame indices are included in the hash-bound camera assessment.
Recovery replays those responses. Malformed or missing observations remain uncertain and cannot
cancel another pair's observed failure. Existing bounded format retries still apply separately
per pair. This increases verifier calls and latency; it does not add video generation requests.
Unsampled motion and model observation errors remain possible. Earlier audit protocols cannot
be reused as v11 approvals, and machine passes still require creator review.


### Choose a render approach

Each shot has a collapsible summary showing its purpose and cut duration. The first shot starts
open; expand the others as needed. Expanded sections follow their shots when reordered and remain
open through edits and saves in the current editor session. Newly added shots open for editing.
Opening or closing a section does not change the saved recipe, seed, or generation state.
Advanced sections start closed, remember their disclosure state during the editor session, and
remain available for an explicit change. No renderer or seed is silently replaced by this layout.

**Camera policy → Single continuous shot** requests one take while allowing camera movement.
New editor shots use `direction_version: 2` (Advanced → Prompt instructions → Positive instructions).
This describes the desired continuous view instead of naming forbidden edit effects in positive
conditioning, and omits the absent-entity list from generation instructions. Absent entities remain
part of staging and output review. Existing plans without this field retain version 1 wording and
request identities; opening or saving them does not migrate them. Portable plans can explicitly
set version 2 per shot. Changing the version changes shot and compiled request identities while
preserving the chosen seed. This prompting change remains under evaluation; it does not guarantee
action, framing, exclusion or uninterrupted motion.
Its additional reviewer inspects adjacent pairs of sampled frames for visible dissolves,
superimposed scene layers, and wipe boundaries. A detected artifact rejects the candidate;
uncertain or malformed observations cannot automatically pass it. The existing video-attempt
budget still applies. Follow prompt and Locked frame retain their existing behavior.

This addresses an observed false pass: a selected development take visibly dissolved from a wide
view into a closer one, while the general temporal reviewer reported no cuts. The separate paired
check detected both affected development pairs and left four clear-frame controls unflagged.
Those six pairs are not independent held-out calibration. Nine sampled frames require eight
additional calls, plus a bounded format retry for malformed output. Unobserved intervals and
hard cuts without a visible blend can still be missed; a clear pair is not proof of continuous
motion or a coherent film.

The exact pair indices and raw responses are retained and hash-bound to the assessment. Recovery
replays them, so a general pass label cannot cancel a detected blend. Visual-audit protocol v16
separates the new assessment contract from earlier cached checks. Selecting this policy changes
the shot contract and invalidates affected descendants; it does not change the seed. Existing
recipes are not silently switched to this policy.

The film editor groups composition and render profile under **Render approach**:

- **Animate starting frame** uses Animate frame with Continue frame. Supply a staged starting
  image for a new scene; following shots can use the previous frame. Separate cast images do not
  condition this profile. New blank films start with this explicit combination.
- **Compose from references** uses Reference shot with New composition. It requests a new view
  using selected references and does not promise to preserve the starting composition.
- **Custom settings** preserves existing combinations. Expand **Advanced shot settings** to edit
  transition, composition, sampler and render profile independently.

An explicit approach change retains the seed, images, story requirements and camera policy.
Switching Turbo 8 animation to reference composition retains Turbo 8 and selects its
profile-specific reference adapter. Added shots inherit the preceding composition and
render profile. Existing saved recipes and configured node drafts are not migrated. Unsupported
Animate/SPEED combinations are rejected rather than silently changing the sampler.

A development reference-profile comparison produced a cropped close-up followed by an internal
composition change despite a first-frame guide. Neither approach guarantees a cut-free result,
correct action, or identity. Camera and story checks remain necessary; this UI change is clearer
setup, not evidence that film quality is solved.

### Selected state evidence across cuts

The film editor's **State memory → Selected state evidence** option uses the native Story archive
to carry observed appearance changes into later **Compose from references** shots. For example,
a prop changes from intact to cracked, leaves the frame, then returns in a new composition.
The controller retrieves a selected ending observation that covers the prop's current declared
facts. For an explicitly selected native state, the selected appearance supplies the entity's
render guide. Its original identity remains in the library and authenticated evidence packet for
independent verification, rather than presenting an obsolete appearance alongside the current
one to the generator. Other entities retain their original references. Creator-approved evidence
without this shot-local selection and historical trained-memory behavior retain their existing
guide policy. The exact guide roles and pixels bind execution recovery.

This conditioning policy addresses a measured failure where a return shot replayed an earlier
state change despite receiving the correct closing evidence. It is a hypothesis about generation,
not a guarantee: opening-state checks must still reject contradictions, and fresh unattended
customer trials must establish whether it helps without losing identity or composition control.

New native observations retain the selected decoded frame's resolution and aspect ratio. Compact
384-square images are used to select the maximum-change frame, not as the stored rendering guide.
The capture policy is versioned and bound to execution recovery. Historical compact observations
remain readable as their original bytes; a changed capture policy does not reuse a completed
request under the old policy. Full-resolution evidence uses more storage and does not guarantee
that H3 will follow a state or action instruction.

This is an opt-in first-pass tool, not creator approval or learned associative compression.
The controller first authenticates a passing assessment and the selected archive's video and
revision lineage. It chooses at most two entity snapshots, rejects forgotten or missing evidence,
and records the source frames, state facts, video/revision digests and assessment digest in
`selected-state.json`. A resumed run checks that record and preserves earlier shot bindings.
It does not change approved canon or select a rejected shot as the next parent.

Use entity-scoped facts such as `Vessel.surface=wet` and define their observable meanings in the
plan. States without an earlier generated change continue to use the original references.
A snapshot must cover all relevant changed facts for an entity; the system does not combine
contradictory appearances from separate moments. Split a shot when its state needs exceed the
reference budget. **Animate starting frame** uses only its starting image and does not claim to
recall separate evidence. The automatic path currently requires native reference archives on
the Comfy host; configured trained archives retain their existing explicit creator workflow.

The advanced Story input **Shot state evidence** also accepts a bounded canonical JSON mapping
of declared entity IDs to exact earlier evidence IDs for that shot. It is an explicit selection,
not a claim that those images passed a model review. The verified film controller supplies it
only after its own authenticated checks. Its contents and actual guides bind execution recovery.
Model assessment remains fallible, and visual comparison across cuts is still required before
claiming this improves customer film quality.

## Optional opening staging (experimental)

In the film editor, **Stage then animate** composes an opening still and feeds it to the
existing H3 **Animate frame** path. Use **Opening composition** for what is visible before the
action; keep **Visible action** for what changes during the shot. For a state transition, declare
both its required starting fact and its ending fact. No story names or example plots are built
into the staging code.

The editor reports when a draft has no explicit starting or ending facts. Opening prose,
state meanings without facts, and the intended-story description do not enable state checks
by themselves. General action and visibility review can still run. For a spatially dependent
action, declare the visible relationship needed at its start and define distinguishable
alternatives: for example, an object across a subject's facing direction versus beside it.
These are project data, not built-in scenes. A successful identity or visibility check does
not establish that the geometry permits the action. Even declared state checks remain
fallible model observations; they do not verify every detail of the opening description.

The normal **Generate saved plan** worker runs this step with a separate image-attempt budget
(default one, maximum four). A passing opening is reused across video attempts. Each attempt,
source hash, seed, exact model identity, original response and assessment is retained. A failed
or uncertain opening stops before the video request. A known completed stage is reused on resume;
an unknown submission outcome stops instead of silently spending another image request.

This optional path requires native Story archives and these additional installed model files:

- `diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors`
- `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`
- `vae/qwen_image_vae.safetensors`

The explicit staging recipe uses 40 steps, Euler/simple, CFG 4, shift 3.1 and CFGNorm 1, with no
Lightning adapter. H3 can still use Turbo 8-step for the subsequent animation. These are different
models and different compute budgets. An installation using another weight filename needs an
explicit supported profile; the worker does not silently substitute models.

There are at most three source images: the optional uploaded scene plus present subjects.
Eligible machine-selected closing evidence replaces the subject's old appearance **for this
staging request**. Its original identity remains in the library and verifier. This does not create
creator approval or demonstrate learned associative recall. The first source controls the raw
Qwen image aspect; use a suitable landscape scene or reference when framing matters. Staging
protocol v2 retains that raw image and fits a second copy to H3’s 1344 × 768 canvas using the
same bilinear center crop as animation. Starting-state and visibility checks inspect the fitted
copy, which becomes the animation input. Both files are bound to the saved rendering receipt.
This is one generated image, not two image attempts. The protocol change requires a new
production run; old staged assessments cannot silently count as checks of the fitted image.

The opening checks cover declared starting state, excluded entities, and required visibility,
including full extent when requested. They do not independently prove freeform geography, exact fine identity, or
composition quality. Ending counts apply to the ending, not the opening. Subjects declared as
present may enter later, so presence alone does not require opening visibility. Model observations
can be wrong. The subsequent video and whole-film checks still run; successful checks mean ready
for creator review, never automatic approval or a public-release quality guarantee.

Staging protocol v3 also checks `absent` before animation. The same independent extent observer
sees the requested subjects and any available original identity references, without being told
which should appear. A recognizable excluded subject, including a cropped body part, fails the
opening. Missing or uncertain observations stop for review. Generic categories without reference
images use their literal names; ambiguous descriptions can therefore require review. These checks
share the existing image-attempt budget and retain raw observations. The workspace shows expected
absence beside observed extent. This prevents an observed exclusion failure from reaching H3;
it does not make the observer accurate or guarantee that staging removes unwanted subjects.
The changed staging policy requires a new run; old v1/v2 records remain readable as historical
evidence and are not upgraded to v3 assessments.

A direct one-graph draft command cannot execute an unresolved staged recipe. Use the verified
film editor/controller. Portable recipe export includes authored inputs and stage settings;
it does not bundle model weights or claim identical pixels across different hardware.

For a staged change of view in the same location, choose **Opening scene guide → Previous
selected frame** and leave the uploaded starting image blank. This uses the exact preceding
selected revision as scene context, alongside current-state or original subject references.
It is unavailable for the first shot. Choose **Uploaded scene or references** for an independently
authored scene. Shared scene/subject pixels are bound once with their combined roles. Carrying
a scene guide does not guarantee identical geography after generation; review the actual output.

The blind assembled-film reviewer receives explicit editorial shot boundaries, without the
screenplay. An ordinary cutaway or an unchanged subject returning after a cut is not itself a
physical contradiction. Missing visible goals, choices and consequential actions still require
review, and incompatible visible states cannot be excused merely by calling them a cut.

### Optional intended-story check

The film editor's **Intended story (optional)** section records four descriptions: the intended
**goal**, **obstacle**, **decision**, and **outcome**. Fill all four, or leave all four blank.
These are creator-authored project data; there are no built-in characters, species, or plots.
They travel with the saved recipe and exported plan as the optional `narrative` object.

The unattended controller first records its blind visual retelling without these descriptions.
For a coherent retelling without reported contradictions, it then compares that preserved
retelling with the intended story. Each field must be supported, with an exact quotation from
the retelling. A contradicted or unestablished field stops the run with its completed candidates
preserved. A valid negative comparison is not rerolled on resume. The request, raw response,
findings, retelling, intent, and reviewer model are bound by hashes and checked again on resume.

The comparison does not inject the ending or future decisions into shot prompts. Changing only
this intent leaves individual shot conditioning and seeds unchanged, but changes the film recipe
and its whole-film assessment. Legacy plans without `narrative` retain their serialized plan and
shot behavior. Removing the optional check does not establish that a previously rejected film
has become good.

This is a check against a **fallible retelling**, not independent visual truth. An exact quotation
can still be misinterpreted, and the blind reviewer can miss or invent an event. The check does
not write a screenplay, repair motion, establish character identity, or guarantee narrative
quality. The three-minute customer acceptance criterion still requires watching the complete
unattended result and listening to its soundtrack. Audit protocol v15 invalidates older cached
assessments; immutable generated media can remain available for reuse.

When production stops during opening staging, **Generate and review** shows the preserved
opening attempts, the assessed image, and the expected versus observed facts and visibility
observations. Protocol v2 also links the uncropped source; legacy runs identify the original
image that their checker actually inspected. These are fallible machine observations, not new
creator approvals. Viewing the evidence does not retry, edit or select a take. If the recorded
images, request or history are missing or changed, the preview remains unavailable. The live
run is not inspected through this stopped-run view.

### Uploaded opening images and the H3 crop

When an uploaded first frame has declared starting conditions, production fits it to the native
1344 × 768 H3 canvas before checking those conditions. The checker and subsequent renderer use
that same saved crop. This avoids checking content that H3 would later crop out of a portrait or
panoramic upload. An image already at the native dimensions is used directly.

Canvas preparation uses only native LoadImage, ImageScale and SaveImage nodes. It does not spend
an opening-generation or video-generation attempt. The original upload and recipe remain intact;
the prepared run retains its source hash, job request, Comfy history, raw image and fitted image.
An uncertain submission is not automatically resubmitted. Changing the source or recorded output
invalidates recovery. Earlier starting-state assessments cannot be reused under the new policy.

This repair aligns the evidence with rendering geometry. It does not establish that the image's
story setup is correct or that the model's state classification is reliable.

### Refine an existing opening composition (alpha)

Under **Stage then animate**, **Opening image mode** offers two explicit choices:

- **Compose** remains the default. Use references to create a new arrangement.
- **Refine** starts with a scene in which the required subjects are already placed. It tries to
  blend boundaries, ground contact and lighting while retaining subject count, placement, size
  and facing direction. Supply an uploaded starting image or a previous selected frame. An empty
  background is insufficient when subjects must be added; use Compose for that task.

You can prepare a layout with ordinary Comfy segmentation, crop and masked-composite nodes, or
bring your own prepared image. This option does not automatically segment subjects, choose their
positions, or author the screenplay. Describe the desired opening separately from the shot action.
Use Compose when you need a substantially different arrangement or visible state.

Refine uses the same Qwen Edit 2511 models, 40-step sampler and reference bindings, with denoise
0.35 instead of Compose's 1.0. This is a conservative alpha profile, not an optimality claim. Both
modes keep the same opening-attempt budget, raw/fitted receipts, starting-state checks and declared
visibility checks. A failed refinement cannot advance the film. The saved mode is part of the
recipe and render request; changing it creates different work rather than reusing an incompatible
opening. Once staged, every video attempt uses the exact selected fitted image.

In JSON shot inputs, set `"opening_mode": "Refine"` alongside `opening_prompt` and the existing
scene source. Omitting it preserves Compose behavior. Removing staging in the editor also removes
this setting. Layout preservation, identity and action still require review; this feature is not
evidence of reliable unattended long-film production.

### Start from references without a scene image

With the native Story runtime, **Compose from references** can leave the starting image blank
on the first shot or a **New Scene**. Select at least one cast, prop or location reference.
The renderer receives those selected references without an additional scene image; write the
new setting and visible action in the shot direction. An explicitly supplied starting image
still participates as scene context. Existing saved inputs and seeds are preserved.

**Animate starting frame** and **Continue frame** still need a starting image for a new story
or scene. Later frame continuation still uses the previous output. Trained memory runtimes
retain their historical world-image requirements. Reference-only composition is an explicit
conditioning choice, not a guarantee of action, multi-character presence or scene consistency.
## Soundtrack levels before rendering

Adding an uploaded soundtrack measures the first audio stream's decoded sample peak and RMS level
when FFmpeg can complete the analysis. **Check audio levels** repeats this check for an existing
track and rejects a changed file hash. The measurements do not alter the audio, saved recipe,
volume, seeds or visual work. A missing or timed-out analyzer is reported as unavailable.

Full-scale peaks are a reason to listen for distortion; they are not proof that a soundtrack is
clipped. This check does not measure true peaks between samples, the final mix, musical quality,
speech, or language. Multiple tracks or increased gain can overload a mix even when each source
has headroom. Silence is reported separately.

When generating music with other Comfy nodes, leave headroom in the decoded audio before an
integer audio export. Comfy's standard **Adjust Audio Volume** node can sit between the audio VAE
decoder and **Save Audio (Advanced)**. Lowering an already clipped file cannot recover lost peaks.
Use the resulting uploaded asset through the normal soundtrack controls; no additional Duet model
or custom audio node is required.

### Understand a stopped opening check

A stopped run shows its explanation directly in Generate and review. Recovery file paths remain
in optional details. When an uploaded opening fails its starting-state check, the editor shows
the exact assessed image with expected facts, observed facts and the original explanation.
Uncertain observations stay uncertain. This also applies when the checker used a prepared H3 crop.

The preview authenticates the saved report, original request and response, and image bytes. Missing
or changed evidence is reported as unavailable. Opening this view does not rerun the checker,
change the screenplay, generate a replacement or approve an image. These fallible observations
help explain a stop; they do not establish story quality or automatic recovery.

## Experimental Full HD and generated shot audio

**Full HD 2-pass** is an explicit advanced sampler preset for **H3 automatic v1** prompts.
It renders 960×544 in six steps using the existing 8-step H3 LoRA, upscales the video latent
with the optional MiniMax H3 3D upscaler, then refines at 1920×1088 in three Euler steps.
The audio latent is retained when replacing the video latent. Conditioning is rebuilt at the
larger canvas; endpoint guides still target the last retained frame. Refinement uses the saved
shot seed plus 10,000, modulo 2⁶⁴. Tiled VAE decoding uses tile 256 and overlap 32.
This is an experimental quality preset, not a measured guarantee of improved detail, a 177-second
runtime, or the W4A8 implementation shown in community demonstrations.

Install [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
at commit `d7c01b9011f2e8439493f6c02c29995a27df276f` and restart ComfyUI. Install
`minimax_h3_latent_upscaler_3d_fp16.safetensors` from
[the model repository](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/tree/13ccf95d85d120bdbc92c05b1247a6e147bf54bf)
under `models/latent_upscale_models`. Duet checks the node implementation and records the model
hash. The preset does not silently substitute another upscaler. Film export preserves the
1920×1088 canvas when any shot selects this preset; lower-resolution shots in a mixed film are
conformed to that canvas, not retroactively regenerated at Full HD.

Under **Soundtrack**, **Shot audio → Generated speech and effects** enables H3-native audio.
All shots must use H3 automatic prompts. Typed dialogue cues supply language, exact text, timing
and stable speaker labels to the prompt. H3 still determines the actual performance: labels do
not guarantee identical voices or accurate words. The export receipt explicitly leaves generated
dialogue unverified. Listen to the result before publishing. Authored speech and generated speech
cannot be combined in this mode; an instrumental soundtrack can still be placed on the timeline.
Generated audio is decoded from each source and placed at its shot offset, avoiding accumulated
AAC padding between cuts. Silent shots remain the default, and existing recipes are unchanged.
