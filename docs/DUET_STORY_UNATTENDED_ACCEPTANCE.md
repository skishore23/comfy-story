# Duet-X: usable filmmaking is the product goal

Duet-X should let a creator prepare a story, cast, props, locations and language once, then run a
connected film workflow in ComfyUI. The output should preserve both appearance and events: who is
where, who holds what, what changed, and why the next shot follows. A creator should be able to
change a shot and regenerate the affected continuation without losing the rest of the project.
Comfy Story is the current node and product interface for that work.

The acceptance test is a new three-minute film produced through the supported workflow, without
an operator replacing individual takes, rewriting prompts during generation, or choosing favorable
trim intervals. A beautiful but incoherent film fails. A completed queue is an execution result,
not a quality verdict.

## First unattended test: The Second Cup

The frozen test used twenty original H3 takes, selecting the first nine seconds of each by policy.
It used a prepared screenplay, named character/prop references, one English narrator, exact
authored captions, and a prepared music mix. Reference and voice preparation preceded submission.
No individual film take was regenerated or selected after viewing its output.

| Area | Actual result | Interpretation |
|---|---|---|
| Execution and export | All twenty takes completed; 180.000 seconds, 4,320 frames, 24 fps, 1344×768 | Technical completion passed. |
| Completed-job recovery | Initial repeat failed because Comfy omitted cached expansion receipts. After the reusable node fix, both cold and warm replay recovered all twenty original revisions and media hashes. | Recovery defect reproduced and fixed; no replacement takes. |
| Spoken language | All ten authored cue windows matched normalized English transcription in the assembled mix. Whole-track transcription omitted/merged lines. | Cue evidence supports script delivery. Whole-track ASR is inconsistent; no claim of a completed human listening review. |
| Story geography | At approximately 0:58, Leon appears inside the café before the invitation that should bring him inside. | Story continuity fails. Character resemblance does not compensate for this error. |
| Other visual checks | A local model flagged missing prop inserts, incomplete actions, unwanted generated lettering, extra people/objects and an internal cut. | Diagnostic findings require visual confirmation; they are not all established defects merely because the model reported them. |
| Critic reliability | The same critic marked the premature interior shot as passing and sometimes treated reference imagery as sequence evidence. | Machine passes cannot be used as autonomous approval. |

Original queue: `3c604f9d-6b22-4bd9-859b-b06a0c4b56f9`.
Original, cold-recovered and warm-recovered film SHA-256:
`0fcaf72b58fc4f367a7d7e47928dfb574e12bae48e27e6a90e3397721c6ce51c`.
Private source media and detailed receipts remain under `/mnt/data/duet-x/second-cup` on Forge1;
they are excluded from the source distribution.

**Verdict: execution works; unattended film quality has not passed.**

## Follow-up result from the reusable fixes

Commit `fc8c54b` carries scoped state and end goals into rendering, corrects baseline-reference
wording, and compares observed continuity values independently of model pass labels. Its complete
supported-runtime gate passed **2,983 tests, with 3 skips**, plus formatting, lint, type checks
and all 28 frontend tests. It is installed on the internal Forge1 alpha.

The paired spatial check now rejects an indoor frame when outside is required and accepts the
same frame when inside is required. This is a narrow control on one inspected frame, not a
general quality guarantee.

The actual H3 regression reused the original shot-seven parent, seed, references, duration and
audio, with only the reusable conditioning changes. It **still rendered Leon inside**. The new
checker rejected that result, and browser inspection confirmed it. Therefore the code fixes
improve state delivery and failure detection but do not yet establish spatial control over H3.
The original film was preserved; the regression is not a replacement take or a quality pass.

Regression queue: `b539c254-aa3e-47fb-be54-d5bf566632ca`.
Regression video SHA-256:
`f0835b17cf24e021c8264fef1b0a827d4a49fd2e01888878bdada4803cd70a88`.

The subsequent control changed only the public Composition input to `Continue frame`. Its
sampled output kept Leon outside, including the independently inspected midpoint. Queue:
`be22b771-da36-4c72-9e00-5420afff4ed8`; video SHA-256:
`f893abf297ab83a2a6b351fdb763513ff5c7aaede12d5714bc24f4151cc9888b`.
This mode enables the frame-zero anchor and can restore inherited scene context, so the result
does not isolate the anchor's contribution. The critic still downplayed possible window signage;
its overall machine pass is not a complete creative approval.

The product needs a way to prepare and verify a correct shot opening before animation, with
explicit choice between preserving that opening and recomposing the scene. Do not silently
override a creator's explicit composition selection or treat an identity portrait as a verified
storyboard frame. Test this policy across location changes, prop inserts and multi-character
actions before another full-film acceptance claim. One improved control is not a coherent film.

## Technical work that changes the product

1. **Carry state into conditioning.** The compiler previously validated the plan's persistent
   facts but discarded that state when writing a shot prompt, and omitted intended effects.
   Scoped continuity and end-state goals now reach the prompt. Planned effects stay explicitly
   unverified; accepted-state compilation still requires the reviewed prefix. This fixes missing
   information, not H3's obedience to it.
2. **Observe state independently.** A visual checker must distinguish inside/outside, ownership,
   open/closed, present/absent and completed/incomplete actions. It must identify evidence from
   output frames, independently of reference images and intended events. The diagnostic prompt
   now requests starting conditions and a separate continuity finding. Its reliability still needs
   held-out positive and negative controls before it may drive autonomous repairs.
   A paired spatial control exposed a further failure: the model described the character as inside
   while labeling an outside requirement as passing. The checker now requires observed condition
   values and compares them in code; contradictions fail even when every model label says pass,
   and missing observations remain unreviewed.
3. **Resolve conditioning conflicts.** Test clean identity imagery separately from scene/layout
   evidence. Compare frame continuation and new composition on held-fixed scripts and seeds.
   A supplied character image can carry incidental background; additional prose cannot be assumed
   to erase it. Record the actual images and prompt reaching H3 in every comparison.
   The node now distinguishes baseline library imagery from approved current evidence in its H3
   instructions; previously it described some baseline packets as confirmed current state.
4. **Make failure handling part of the normal workflow.** Keep a bounded generation budget,
   durable output receipts and explicit failed/uncertain outcomes. Once a checker is validated,
   retry the affected shot and invalidate dependent proposed state automatically. Do not silently
   advance the story using an event that was requested but did not occur.
5. **Finish language deterministically.** Preserve authored voice and captions, check the final
   mix, and detect unwanted writing in source frames. Clean captions do not repair invented signs
   inside the scene. Automatic lip sync remains separate, unproven work.
6. **Expose the film workflow to ordinary users.** Bring screenplay, duration, language, cast,
   progress, failure reasons and export into one discoverable Comfy experience. The existing node,
   typed plan, runner, compositor and store should be reused. A CLI assembled by an engineer is
   not yet the complete customer experience.

## Evidence required for the next release claim

First rerun recorded failure cases through the changed compiler and checker. Preserve baseline
outputs, fixed seeds, references, prompt versions and all failures. Then run a fresh three-minute
film with the product's frozen policy and no per-shot intervention. Review the actual complete
film, including audio, against the authored causal sequence. Record correction time and compute,
not just whether an MP4 exists.

Before claiming general superiority, repeat on multiple scripts and seeds against a native H3
workflow with the same references and generation budget. Measure identity/wardrobe, prop ownership,
spatial continuity, event order, language, accepted seconds per generated second and creator effort.
This test has not established an advantage over community workflows.

Reuse the existing associative memory, authenticated revisions, exact-evidence retrieval and
cache machinery. Their systems value is real. The deployed H3 history-residual readout is zero,
so the current film is not evidence of learned associative-memory quality improvement. Train and
evaluate a useful readout only against a fixed native-reference baseline. Use state composition
and lattices to preserve the distinction between unknown, proposed, observed and approved facts;
they cannot substitute for checking what the video depicts.
