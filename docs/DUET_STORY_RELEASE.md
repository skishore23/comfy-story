# Comfy Story: implementation and release record

Date: 2026-09-04. Base: `5d1870d83d987f473d4c1f22b0b0452816045849` plus the working-tree changes described here.
This is an internal H3 product candidate. A successful render is not a comparative quality result.

## The product we are building

**Keep the right approved character and prop state available for the next shot, and make an accepted
sequence safe to revise and reopen.** Comfy Story is the single public ComfyUI node. Duet-X is the
internal memory and conditioning implementation; original Duet remains the research foundation.

The customer workflow is: establish a reference library, describe a shot with `@Name`, generate,
inspect the result, approve a visible look or important detail, add the next shot, and return to the
same accepted work after a restart. A creator must still review visual quality. Stored canon records
what was approved; it does not certify what H3 depicted.

Read the [product thesis](https://github.com/skishore23/duet/blob/65dfb7db079d2e00810b1402abe43c0dcf51f91d/docs/DUET_PRODUCT_THESIS.md) for positioning, the
[build guide](https://github.com/skishore23/duet/blob/65dfb7db079d2e00810b1402abe43c0dcf51f91d/docs/DUET_STORY_BUILD_GUIDE.md) for the longer roadmap, and the
[reuse audit](https://github.com/skishore23/duet/blob/65dfb7db079d2e00810b1402abe43c0dcf51f91d/docs/DUET_REUSE_AND_DIFFERENTIATION.md) for the relationship to existing benchmarks.

## What this implementation changes

| Area | Implemented behavior | Practical limit |
| --- | --- | --- |
| Completed-shot recovery | A content-bound attempt index finds an authenticated revision and returns its video, lossless last-frame tensor, and Story State before constructing a GPU graph | An interrupted, uncommitted attempt is not resumable; no distributed running-job reservation yet |
| Request identity | Binds the resolved prompt, actual model-file hashes, variation, parent, ordered guides, memory checkpoint, commands, and compiler configuration | Cold model hashing has a measurable startup cost; the execution format must be versioned when semantics change |
| Canon and exact recall | Approved evidence comes from the durable evidence registry, including when it is absent from the bounded candidate reservoir | Two exact state packets per shot; excess demand fails rather than silently substituting baseline |
| Forget | Tombstones persist through later appends; affected dense leaves become identity; stale parent context is omitted on that transition | Conservatively removes the whole source leaf from dense history. It does not erase pixels already present in a scene/previous-frame reference or purge old revisions |
| Missing approved evidence | Missing or forgotten sole support produces an actionable error; the creator can replace evidence or restore the original | No automatic semantic reconstruction of deleted support |
| Inspector | Correct V3 hidden owner ID and list-valued ComfyUI output; previews address verified image and video assets by digest | Inspector summaries are projections of authenticated state, not a vision judge |
| Next-shot actions | Draft approvals are serialized separately and move to a newly wired child; the completed parent's generating inputs remain unchanged | Changes staged on a completed node apply through Add Next Shot |
| New composition | Omits inherited scene imagery and the frame-zero latent guide while retaining saved state, canon, and selected exact evidence; persists through save/load and Add Next Shot | The previous-frame reference leak on inherited cuts is fixed and covered by roster tests; fresh film quality remains unverified. Explicit Start Story/New Scene world images and Continue frame retain their behavior |
| Panel | Prompt, shot duration, starting image, library, and inspector in one scrollable panel; named field persistence; explicit New take variation | Run workflow executes the ComfyUI workflow, including dependent outputs |
| Timing and capacity | Uses actual decoded frame count at 24 fps; rejects full 128-shot history before generation | The 5-second preset currently decodes 124 frames, approximately 5.167 seconds |
| Internal packaging | Wheel metadata includes dependencies and license; explicit Comfy-owned Torch profile; preflight checks before installation; refuses accidental node overwrite | Existing installations require a staged upgrade with rollback, not a blind installer overwrite |

The public `DuetStoryCanon` alias is retained for historical workflows. Research code, checkpoints,
and benchmark protocols have not been indiscriminately deleted or promoted. The unmerged attention
research branch remains separate because its scientific and compatibility issues require their own
validated changes.

## What the Forge1 inspection established

The acceptance runtime uses the existing special MiniMax H3 installation on Forge1, through root
SSH, in a separate ComfyUI process. Its initial configuration is Python 3.13.12, PyTorch
2.12.1+cu130, ComfyUI 0.34.0, frontend 1.51.9, and an RTX PRO 6000 Blackwell Server Edition.
No H3 weights are included in the package.

The inspected deployed memory checkpoint has SHA-256
`23865617f5bad0c667dac12edb5f36ee993f0cfe6b54fbb54b3a2172a3d0062b`.
Its bridge `residual.weight` is a 24-by-24 tensor with **zero nonzero elements**. The bridge computes
an anchor plus a learned residual from the associative operator. Thus this checkpoint does not
currently contribute learned historical information through that residual. Ordered associative
state is accumulated, but a claim that this deployed checkpoint improves long-term H3 identity
would be incorrect.

The useful working path is native H3 with exact reference/evidence conditioning and durable creator
state. The original associative benchmarks remain valuable evidence about order, reassociation,
and sparse updates; their accuracy and speed results are not H3 continuity results.

Initial real acceptance observations:

- A first Mira/Satchel workshop shot completed through the public node in about 297 seconds.
- After restarting the isolated server, the same request returned the same revision without an H3
  generation graph. Cold verification took about 109 seconds. Treat this as saved GPU work, not
  an instant-start claim.
- A wired child shot completed in about 156 seconds; its authenticated receipt selected the
  approved closing evidence from shot one and retained its canon support.
- A deliberately incomplete recovery-only child lacked its current/world frame and failed before
  sampling. The ordinary Add Next Shot path supplies both Story State and Last Frame.
- An absence/return pair completed: the absence receipt excluded Mira's exact evidence, and the
  return selected her approved shot-one evidence. This proves state/conditioning behavior in this
  example, not visually scored generalization or a five-shot gap.
- A child created, edited, and queued through the browser completed in about 149 seconds, reusing
  both ancestors. Saved prompts, duration, pending actions, and nine evidence previews survived a
  browser reload.
- Actual ComfyUI execution exposed an inspector aggregation bug missed by mock-only tests; the
  candidate now emits the required one-element list. Browser testing also exposed field-position
  drift, addressed through named serialization and restoration.

Machine-readable prompt histories, package/source manifests, final check logs, and further
acceptance results belong under the ignored `artifacts/duet-story-release/2026-09-04/` locally and
`/mnt/data/duet-x/product-hardening/2026-09-04/` on Forge1. Media stays on Forge1.
The source archive digest identifies a working-tree candidate more precisely than its base commit.

## Operating the node

1. Add **Comfy Story**. Name the project, add a character/prop image, and give each a stable name.
2. Select **Start Story**, choose a starting image, and describe the action with `@Name`. Start with
   one character and one important prop. Use **Native** reference context.
3. Run the workflow. Inspect both the video and the evidence thumbnails. Generated observations
   remain pending until the creator decides to use them.
4. Use **Keep @Name look**, **Keep this detail**, **Off screen next shot**, or **Restore original**
   on a completed node. These controls stage changes for its next child.
5. Select **Add Next Shot**. It copies the shared settings, connects Story State and Last Frame,
   and transfers the staged changes. Edit the new node's prompt. **New Scene** requires a new
   starting image. **Continue This Shot** requires the connected frame.
6. Use **New take** to explicitly change variation. Re-running identical completed inputs recovers
   the existing take. Save the ComfyUI workflow to retain the graph and draft panel state.
7. To try an alternate future, create another child from the chosen completed parent. Immutable
   revision ancestry separates sibling futures. A dedicated branch browser is still future work.

A prior off-screen entity can be explicitly requested again with `@Name`. Off-screen state suppresses
its automatic recall; it is not an absolute prohibition against a later explicit mention. Current
frame and world references can themselves contain a supposedly absent entity, so a clean scene
anchor matters. Absence remains a rendered-output criterion as well as a state criterion.

## Release and upgrade discipline

Build with `python scripts/build_duet_story_internal.py --help` and the documented output argument.
The archive contains source, node code, an installer, dependency metadata, and a payload manifest.
It must not contain model weights, generated media, credentials, or datasets.

The primary research dependency remains Torch 2.8. The internal Story wheel explicitly supports a
Comfy-owned Torch range of `>=2.8,<2.13`; this is not evidence that every version in that range has
passed H3 acceptance. Its cryptography profile is `>=43,<50`, including Forge's tested 49.0.0;
the research profile remains unchanged. The recorded Forge runtime is the tested target. Install/upgrade checks
must use ComfyUI's interpreter and preserve its CUDA/model stack.

Stage a new release and separate story storage first. Verify the schema, saved workflow, real H3
continuation, restart recovery, and dependency compatibility. Keep the old source target and
runtime settings for rollback. Switch the main installation only when its queue is empty. Model
access and redistribution terms remain those of the user's existing H3 installation.

The merge cleanup fixes the baseline strict-typing failures with explicit buffer, constructor,
factory, and fixture contracts. It preserves checkpoint serialization and arithmetic. The lockfile
now includes the declared cryptography dependency and the test suite's aiohttp dependency; CI uses
`uv sync --locked` so dependency drift fails explicitly. Credential-custody tests require a private `TMPDIR` outside the source checkout because they
correctly reject world-writable ancestry. CI uses `runner.temp`; Forge validation uses a private
root-owned temporary directory outside its shared `/mnt/data` mount. The production security
checks and test assertions remain unchanged. New installer tests cover check-only mode,
missing dependencies, and refusal to overwrite an installed node. Generated brainstorming state
and the nested local worktree are ignored and excluded from distribution.

The package now includes the [quickstart](DUET_STORY_QUICKSTART.md),
[shareable marketing overview](DUET_STORY_MARKETING.md), root license, file manifest, and checksum.
Use `--check-only` to validate a destination without installing. This is a code distribution for an
existing authorized H3 setup, not a bundle of model weights or a general availability release.

The local Intel-macOS Torch 2.2 environment is outside the research dependency range. Preserve that
working environment; Torch 2.8 has no Intel-macOS wheels. Merge validation also reproduces the
checked-in Python 3.11/Torch 2.8 environment on Forge1. Source-identity tests require a frozen tree:
edits during an all-tests run invalidate their manifests by design.

## What to add next, in order

1. **Earn the continuity claim.** Run the build guide's five-shot-absence, changed-prop, lookalike,
   ownership, restored-baseline, and sibling-branch cases. Blindly judge identity, state, unwanted
   presence, motion, and prompt adherence. Compare against native H3 with well-managed manual
   references and a strong exact-retrieval workflow on matched assets/seeds/budgets. Report
   generation failures and operator time, not just attractive clips.
2. **Finish creator operations.** Add an explicit accepted-take list, ordered clip export, project
   export/import, pre-generation evidence preview, and readable recovery selection. Inherited libraries are read-only in the child panel; add an
   explicit versioned library-edit operation when creators need to introduce new entities mid-story.
3. **Make recovery economical and robust.** Add an authenticated persistent model-digest cache,
   a running-attempt reservation and failure journal, crash-before/after-publication tests, and
   model/backend migration rules. Benchmark cold and warm start separately.
4. **Train history only behind an ablation gate.** Compare exact-only, anchor-only, shuffled history,
   ordered associative history, and resampler history at matched training and inference budgets.
   Require source-disjoint continuity improvement before enabling a learned H3 readout by default.
5. **Use algebra for concrete correctness.** Keep ordered operator composition for chronology.
   Use an append-only union of evidence and approval events with explicit conflicts for merges;
   never average incompatible canon. Add laws for parent immutability, replay idempotence,
   tombstone monotonicity, and branch-local approvals. A lattice of evidence sets does not imply
   that narrative time is commutative. Category diagrams should enforce real adapter contracts.
6. **Revisit H3 attention compression after its pilot is corrected.** Include target, text, audio,
   and reference rows in the full joint-attention comparison. Measure full-shot time and board
   memory, not isolated K/V savings. Keep source-disjoint transfer and visual quality gates.

Do not market “better than the community,” learned long-term character memory, constant total
storage, unlimited history, voice identity, or faster H3 until the matching evidence exists.
The differentiated product hypothesis is creator control plus reliable continuity editing and
recovery inside ComfyUI. The purpose of the next comparison is to determine whether that combination
actually helps creators finish sequences with less correction and wasted generation.
