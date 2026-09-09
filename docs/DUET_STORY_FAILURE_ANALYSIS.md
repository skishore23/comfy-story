# Why Comfy Story drifts, and what must improve

The product goal is a coherent, editable film with recognizable characters, stable props, visible
causes and consequences, and the creator's exact language. A successful H3 render, an immutable
revision, and an associative-memory benchmark are three different kinds of evidence. None alone
establishes that a film meets that goal.

The three-minute production is an acceptance test for reusable code. Editing around a bad take
can help finish a film, but must not be counted as a fix to the generator or its conditioning.

The first unattended three-minute test, **The Second Cup**, completed but failed story continuity.
See the [acceptance results and next product gates](DUET_STORY_UNATTENDED_ACCEPTANCE.md).
This failure is evidence against advertising unattended coherent filmmaking today.
The subsequent fixed-input H3 regression still failed location after the state and reference
prompt corrections. The updated checker detected the failure. This separates a verified software
correction from an unproven rendering-quality improvement. A subsequent `Continue frame` control
kept Leon outside in sampled output. That mode changes the opening anchor and inherited scene
context together; the narrow result supports further composition-policy tests, not a film-level pass.

## Findings from code and actual renders

| Failure | Evidence and mechanism | Reusable correction | What remains to prove |
|---|---|---|---|
| A library image is presented as confirmed current state | Baseline-only exact packets used the same prompt wording as approved generated evidence: “exact current evidence.” A character reference can contain an incidental location or pose. | Keep the historical guide roles and bindings, but describe baseline packets and supporting references by their library role. Only selected state evidence receives current-evidence wording. | Tests establish source-role separation. Render comparisons must measure whether this reduces incidental-content leakage. |
| Persistent location is omitted from later shot prompts | The film compiler validated the state prefix, then discarded it and emitted only each shot's repeated `requires`. The Second Cup establishes Leon outside; a later shot omitted that fact from conditioning and rendered him inside before the invitation. The action did still say exterior, so omission is a code defect, not a complete causal explanation of H3's failure. | Compile scoped persistent state and explicit end-state goals, distinguish planned from reviewed facts, and avoid locking a changing fact to its old value. | Regression tests establish that state reaches conditioning. A fresh unassisted render must establish any visual improvement. |
| A matching setting masks incorrect story state | The first visual audit marked the premature interior shot as a machine pass. Its prompt omitted even the shot's explicit preconditions and had no separate continuity check. | Supply planned continuity context and explicit starting conditions; require a separate continuity finding. Keep all machine findings unapproved. | The critic also confused references with sequence frames. It needs spatial and temporal controls; adding instructions is not validation. |
| The critic describes a contradiction but labels it passing | In the paired spatial control, the critic explicitly described Leon inside while marking an outside requirement as pass. | Require observed condition values alongside check labels. Compare those values with the same scoped conditions sent to rendering; mismatches fail and omissions need review. | This catches reported contradictions, not confidently incorrect observations. Held-out calibration remains required. |
| Requeueing a completed film loses output receipts | Actual Comfy replay completed with an empty history output map when expanded nodes were cached. The film runner could not recover its shot receipts. | Re-enter the public node's durable recovery gateway on every queue using a JSON-safe scheduler fingerprint. Completed requests return stored outputs without a new sampler graph. | Cold and warm full-workflow replay recovered all twenty original takes and the identical final-film bytes. This does not improve film content. |
| Earlier actions reappear in new scenes | The adapter previously supplied an inherited inclusive scene image even for New composition. Several journey/final-shot candidates replayed workshop content. A same-seed final-shot control retained the other two guide hashes and omitted this image; sampled frames then stayed with the harbor. | Public New composition omits inherited scene imagery and the frame-zero latent guide. Saved associative state and selected evidence remain. | One matched pair supports this policy; multi-script comparisons must establish its overall effect on identity, action, and acceptance rate. |
| A collage appears inside the video | `render_story_evidence_packet` packs baseline and current state side by side in a gray 384×384 canvas. A door render reproduced a gray reference grid at 7.833 seconds. The service described the combined image as exact current evidence, despite mixing different times and poses. | The public Native path supplies identity and state separately and states which controls mutable appearance. Packet storage and historical experiment behavior remain compatible. | A same-input H3 comparison is required in addition to regression tests. Separate images can still contain unwanted incidental scene content. |
| A restored light appears before the repair | The original lighthouse reference is lit; the approved later evidence is dark. The previous combined packet showed both states together. One arrival candidate replayed the blackout. | Separate identity from state; retain explicit state notes and review the active guide roster. Reference the actual approved location frame. | Precedence instructions are conditioning, not an enforcement mechanism inside H3. Review must detect premature state changes. |
| An action reference induces the wrong action | Original cast/prop images contain a handoff, other people, and harbor scenery. Some candidates repeat the handoff or introduce incidental people. | Preserve original provenance, derive clean references, and keep absent entities out of the active roster. A reference inspector should expose the exact images sent to H3. | The product still needs a convenient reference-isolation/crop workflow and evaluation of incidental-content leakage. A name or an absence instruction cannot erase pixels already supplied. |
| The panel rejects a character, prop, and setting together | The public panel reused the two-packet recall limit as a two-name input limit, although the service separately budgets explicit references and expanded state images. | Separate mention policy from the image ceiling. Permit the native named-reference envelope and let server preflight validate the final identity/state image set. | Eight named references is an upper bound, not guaranteed capacity alongside recalled state images. More supplied imagery can also reduce creative quality; use the smallest useful roster. |
| Older projects reopen with shifted controls | An actual saved production workflow placed its image-upload value last and appended the DOM editor marker. The migration did not recognize this nine-value layout: the story prompt appeared in Reference context, and the duration appeared in the prompt field. | Recognize the historical duration/variation positions, restore the world and prompt by their original roles, discard only the editor marker, and serialize named fields for future reloads. Cover the actual panel lifecycle and both historical Canon image positions. | Regression coverage does not establish that every experimental historical layout is supported. The observed workflow must also reopen correctly in the installed browser. |
| Composition changes are lost | The actual panel serialized eleven widgets after Composition became a twelfth; Add Next Shot also omitted it. Helper-only migration tests missed those hooks. | Actual panel serialization, restoration, controls, and next-shot copying now preserve Composition. Lifecycle tests execute the real panel code. | Browser and installed-runtime checks remain necessary alongside JavaScript tests. |
| The plot becomes a sequence of unrelated events | Generation is shot-scoped. Reference retrieval and saved appearance do not enforce a protagonist's objective, prerequisites, attempted solution, or resolution. | `FilmPlan` validates planned dependencies and exact timing. Acceptance records observed effects against an ordered reviewed prefix. Export rejects incomplete or stale selections. | These production APIs are not yet a complete screenplay/review editor in the ordinary node panel. A caller's review is still human judgment, not an automatic vision guarantee. |
| Speech changes or fails to follow the script | The original output graph always decoded H3 audio from the sampled latent. There was no independent approved-speech input or transcript contract in that node. This is a confirmed architectural gap; attributing every earlier reported language failure requires review of those clips. | Authored audio can replace generated sound. Validation prevents silent truncation, and audio samples participate in cache identity. The film API binds each narration asset to an exact cue and produces authored captions. | TTS pronunciation, language, speaker identity, pacing, and lip sync require their own review. An audio socket does not create or certify those properties. |
| Approved speech becomes harder to recover in the final mix | Final-mix ASR misheard three short lines that source-only ASR recovered, alongside a Sol/Saul name ambiguity. The exporter previously summed speech, music and effects without per-cue normalization or background ducking. This is evidence of mix/transcription sensitivity, not a human listening verdict. | The film exporter can normalize each finite speech asset before mixing and lower background gain during its actual timeline interval. A signal-based regression checks attenuation and recovery, including an offset background track. The receipt records the policy; an already mastered mix can opt out. | Recheck the assembled soundtrack. Exact captions and source transcripts do not establish intelligibility after mixing; pronunciation still needs listening review. |
| Visible writing is unreliable | A prose request to generate writing is not a typography pipeline. The film exporter previously had no reason to assume an arbitrary requested text overlay was actually composited. | Exact narration captions and explicit title finishing are authored separately. Unsupported arbitrary text overlays fail rather than silently disappear from a film export. | The shot panel still needs integrated typography controls. Generated in-scene signs must be reviewed or replaced; caption support does not certify them. |

## What associative memory contributes

Duet provides ordered state composition, cache reuse, authenticated provenance, and a framework for
retrieving exceptions. These are useful systems properties. Preserving the same face is different
from preserving who has the key, whether a door has opened, or why a character chooses an action.

The active renderer consumes images and text. A compressed operator helps visual continuity only
if its trained readout makes relevant history available to that renderer and that conditioning
improves outputs. A zero-initialized residual readout can leave a guide equal to its anchor even
while operators change. The actual deployed checkpoint must be inspected and tested with history
perturbed while anchor and exact references stay fixed; initialization alone is not that audit.

The Forge1 checkpoint audit on 2026-09-05 verified the configured checkpoint SHA-256
`23865617f5bad0c667dac12edb5f36ee993f0cfe6b54fbb54b3a2172a3d0062b` before weights-only
loading. Its `bridge.residual.weight` is 24×24 with zero nonzero elements and an absolute maximum
of 0. This confirms that the deployed residual path contributes zero to the anchor-preserving
guide. Exact evidence retrieval and state accumulation remain active; learned visual-history
contribution through this readout must not be claimed for that checkpoint.
In a direct sensitivity check, two finite operators differed by a maximum of 4.659965 while their
materialized outputs differed by 0. Both outputs exactly equaled the fixed anchor. This tests the
readout mechanism, not the quality of a generated film.

The same-input packet comparison removed the gray contact-sheet presentation, but an unwanted
cut remained at 7.875 seconds. That result does **not** establish coherent-shot success. A further
same-seed test used `FilmShot.reference_names` to retain the doorway setting without activating
the full-tower image; its sampled frames stayed at the doorway. These are narrow production
comparisons. Their raw outputs, inputs, and rejected takes remain part of the evaluation record.

Do not replace the existing research benchmarks. Reuse their algebraic and cache tests, and add
film-specific evidence. Category-theoretic composition can support valid ordered state changes;
it does not make a lossy visual decoder obey a story. A state lattice is useful for distinguishing
unknown, proposed, observed, and approved facts. It should prevent an unreviewed proposal from
becoming an accepted event, rather than assigning scientific terminology to an unverified output.

## Required product workflow

1. Approve the script, cast, props, locations, language, voice, and exact runtime.
2. Break the script into shots with a purpose, required state, and intended visible change.
3. Review clean identity references and current-state evidence as separate inputs.
4. Generate a candidate with explicit present/absent entities and one primary action.
5. Review the actual clip. Record what happened, including failures, instead of copying prompt
   intentions into memory as if they were observations.
6. Advance the accepted story only after review. Retakes must invalidate dependent approvals.
7. Use approved speech and deterministic typography; review the final assembled film for both
   local consistency and overall narrative meaning.

Steps 1–7 need to become one discoverable workflow in Comfy Story. The current Python planning and
export APIs are reusable foundations, but should not be marketed as a finished one-click film
director. The public node remains one product; internal implementation stages need no separate
research-facing products.

## Evaluation before stronger claims

Use at least three scripts, multiple seeds, identical H3/model versions and source references, and
the same sampling budget. Include a handoff, an object-state change, an absent character, a return
to an earlier location, a failed action followed by a different solution, and exact language cues.

Compare native H3 without Duet memory, the previous packet path, separate identity/state images,
and any learned-history candidate. Record actual guide hashes, generated media, rejected takes,
and all editing corrections. A corrected export and its original render are separate results.

Report:

- Accepted seconds per generated second and per unit of compute.
- Identity/wardrobe errors, prop shape/ownership errors, and absent-character intrusions.
- Repeated actions, premature state changes, and unexplained transitions.
- Script-to-speech agreement, voice continuity, intelligibility, caption correctness, and unwanted
  speech. Automatic transcription is supporting evidence, with human review for pronunciation.
- Film-level comprehension and pacing in a blind review, plus correction time and recovery behavior.

Passing unit tests establishes the code contracts. Passing these comparisons would establish
product quality. Community superiority remains a claim to earn with a matched comparison.

## Unconfirmed library notes were labeled as approved state

The opening of the new two-minute Turbo test, *The Red Patch*, displayed its red repair patch
before the repair was meant to occur. Inspection of the actual resolved prompt found a library
note describing the later repair appended as `Approved state of @Kite`. The baseline canon
entry had no confirmed revision. This incorrectly elevated free-form reference notes into
current approved state, and placed them after the shot's direction.

Prompt construction now labels notes without a confirmed revision as baseline reference notes,
explicitly distinguishes them from approved current state, and places the shot direction after
reference/state context. Creator-confirmed notes retain their approval label. Tests cover both
paths. This corrects provenance and prompt precedence; it does not establish that H3 will obey
all temporal constraints. The original film run remains on its frozen release for evaluation.

Input authoring also matters: keep reference notes focused on baseline appearance, and attach
mutable prop facts to the visible entity's namespace (for example `Kite.patch = absent`). A
separate `Patch.attached` fact is not automatically included in a shot scoped only to `Kite`
unless the shot explicitly requires that fact. Automatic relationship-aware scope expansion and
validated visual state enforcement remain unresolved.

## An obstacle must exist in the image, not only in the screenplay

The prepared three-minute elephant trial *A Safer Path* selected its first shot, then stopped
at shot two after both permitted attempts continued moving along a branch instead of clearly
stopping to inspect it. The opening did not clearly put the branch across the animals' facing
direction. This is distinct from recognizing the two animals and preserving a piece of wood.
The plan declared the setup in prose but supplied no typed starting or ending facts. Its
general action review could fail the shot; there was no explicit spatial-state check before
video. The failed customer run remains unchanged.

Three bounded diagnostics tested possible corrections with all outputs retained:

| Comparison | Observed result | Decision |
|---|---|---|
| Native 20-step versus Turbo 8 at two fixed seeds and one exact predecessor | Both profiles continued along the branch; neither established the joint stop. Three clips were new and one exact customer request was recovered. | More native sampling is not an established action fix. |
| Two newly staged frontal openings, each automatically animated with the original action | One setup held the animals nearer the obstacle; the other retained sideways movement. Neither opening showed the requested right-end bypass. The opening already placed them near rest, so holding that pose did not prove a walking-to-stopping transition. | Staging affects motion but did not reliably establish causal geography or the requested transition. |
| Automatic SAM3.1 calf masks with native H3 Fun ControlNet strength zero versus one at both staged opening/seed pairs | The patch ran and reduced unwanted movement in sampled frames, but the calf's appearance changed and the desired head-lowering action remained unreliable. | Do not expose this combination as a validated quality fix or make it a default. |

These are development diagnostics with sampled frame review, not independent full-motion film
acceptance. No masks, openings or takes were manually selected during their execution. Different
model profiles change more than step count; the sampler result is a profile comparison. The
masked-control test used native Comfy nodes, not a shipped Duet motion-control feature.

For the next prepared film, describe observable setup relationships and their alternatives in
project facts and state meanings. Keep starting composition separate from action and ending
state. Verify the opening before committing the video budget, then evaluate the actual transition
over time. A correct ending alone cannot prove how an object arrived there. Do not confuse
additional validation with a demonstrated improvement in generation, and do not add plot-specific
names or geometry to runtime code. A fresh unattended three-minute acceptance film is still required.


## Opening checks must inspect the image H3 receives

The subsequent unattended elephant trial *Where We Rest* stopped after its two opening-image
attempts, before any video. The local reviewer reported canopy shade on elephants that appeared
mostly sunlit in visual inspection. The generated tree placement also failed to clearly establish
the intended route toward shade. This is an unresolved combination of generation and observation
errors; the failed trial has not been waived or resumed.

Inspection exposed a separate pipeline defect: staged opening checks saw Qwen’s original image,
while animation later center-cropped it to the H3 canvas. A check could therefore rely on an
object outside the eventual video frame. Staging now saves both the original and fitted image,
and checks the fitted 1344 × 768 view that animation receives. Shared geometry constants keep the
two graphs aligned. Recovery checks bind both files and the staging protocol; changed protocols
require a new production run. Tests cover a subject disappearing in the crop, malformed output
dimensions, retained source bytes, and recovery without another generation.

This corrects staged-opening geometry. It does not establish the cause of the lighting errors,
fix reviewer accuracy, or prove a successful film. Authored starting images outside staging remain
a separate preflight path and are not covered by this change.
