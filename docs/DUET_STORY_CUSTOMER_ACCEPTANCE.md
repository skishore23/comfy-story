# Comfy Story: what must work for customers

## Product outcome

A creator supplies a story brief, a reusable cast and world, and optional sound preferences. Duet
turns those into an editable short film whose events make sense, whose characters and important
props retain their identities, and whose language matches the requested script. Customers should
be able to understand the story by watching it. Narration must not conceal missing visual events.
No speech or subtitles should be introduced when they were not requested.

The release acceptance target is a three-minute film from one prepared screenplay and one run,
without a developer changing prompts or replacing clips during generation. Shorter stories are
useful regression tests but do not satisfy that target. Duration alone is not success. Prepared plans and reference assets are still
required today; brief-to-screenplay generation is a separate missing customer capability.

## General-purpose behavior is a release requirement

Story-specific content belongs in project data: the brief, entities, reference assets, world,
relationships, events, dialogue language, audio policy, timing and budget. A user or a planner may
create that data. The runtime must consume the same schema for unfamiliar projects without source
edits. It must not select behavior by a demo title, character name, prop, theme or known screenplay.

Prompt compilation and repair must derive their instructions from the current project and actual
observations. A failed take cannot change the intended story into an easier one. Examples and
benchmarks may contain concrete stories; customer execution must not depend on those fixtures.
Test new names, nonhuman subjects, unfamiliar fact namespaces and languages, and validate held-out
stories that were not used to design the controller.

Fixed short fixtures can diagnose a specific code or model-profile issue. They cannot establish
customer readiness. After a proposed product improvement, test a fresh story and newly generated
video through the normal user workflow with a frozen plan, inputs, seeds and budget. Preserve the
previous failed film. Do not keep repairing that film and present the repaired result as evidence
that unfamiliar customers will succeed. Require a measurable reusable benefit before adding a
control; new users should not have to choose among diagnostic settings to obtain a useful result.

Installation configuration must resolve against the user's Comfy installation or explicit local
settings. Model/checkpoint identity and capability constraints remain validated backend contracts;
paths, credentials, project choices and deployment provenance must not inherit a developer's host.
Supported model profiles and configurable defaults must be documented rather than presented as
unlimited generator or checker compatibility. No mandatory narration or story-specific speech
should be introduced by a generic repair strategy.

## What the current evidence says

The existing native H3 reference path helps preserve recognizable identities. Immutable revisions,
reference provenance, film contracts, and completed-request recovery are reusable product assets.
The associative memory research establishes systems properties and results on its measured tasks;
it does not establish better H3 films. Learned H3 recall and superiority to other Comfy workflows
remain unproven.

The recent two-minute fox film preserved its broad subject, but important events failed: the fox
left shelter before the scripted departure and the visual story did not clearly establish rain
as its motivation. A blind model retelling called the sequence coherent while missing that cause.
That is a failed acceptance film, not proof of customer readiness.

A subsequent three-minute elephant trial with automatic opening staging stopped on its first shot
after two permitted video attempts. Direct frame inspection found premature water-reaching action,
an unwanted camera push-in, and unclear alternate-route geometry. It also contradicted several
checker explanations: the calf moved, and its feet remained dry despite its trunk reaching water.
A higher-resolution observer corrected some judgments but falsely passed a known cropped-object
control. Neither additional checking nor higher resolution is established as a general quality fix.
Preserve both genuine rendering failures and reviewer errors when measuring customer success.

## Engineering work that directly addresses the failure

1. Preserve the authored event contract: opening state, visible action, ending state, and dependencies.
2. Render a candidate from the selected parent, then inspect what actually happened before advancing.
3. Retry an explicit failure within a finite budget from the same parent. Preserve rejected attempts.
4. Stop on uncertainty or exhausted attempts and expose a recoverable reason. Do not invent success.
5. Assess the assembled film independently of its screenplay, then compare what viewers understood
   with the intended cause, decision, and outcome. Validate identity, props, audio, and unwanted text.
6. Expose planning, progress, failure recovery, and selected takes in the Comfy panel once the
   controller's behavior and quality benefit are demonstrated.

The experimental verified film command implements bounded scheduling around existing rendering,
audit, and export code. It does not train a new memory model, automatically approve canon, solve
screenwriting, or prove the vision checker reliable. See [the workflow](DUET_STORY_FILM_WORKFLOW.md).

## Release experiment

Freeze a small suite before evaluation: a nonhuman shelter story, an object delivery with an
ownership change, a repair story where a visible intervention changes an outcome, and a return to
an earlier character and prop after several unrelated shots. Include one dialogue case separately;
use exact authored audio with an explicit language and separately inspect the finished soundtrack.

For each story, run the same references, plans, renderer, seeds and budget through the first-pass
command and the verified controller. Preserve all attempts, queue IDs, runtime, and costs. Report
how often the original candidate failed, how often an automatic retry repaired it, how often the
checker falsely passed it, and how often the run stopped. Do not discard stopped runs from success
rates. Have reviewers watch without the script first, then check each required event against it.

A release candidate needs all of the following:

- Viewers identify the intended goal, obstacle, decision and outcome from the film itself.
- Every required cause precedes its consequence; no contradicted state is passed downstream.
- Main identities and story-critical prop details remain correct across cuts and absence.
- No unwanted speech, people, subtitles or generated text; requested language is intelligible.
- One ordinary run completes without operator repairs, with recoverable interruptions and a
  disclosed attempt budget. A stop is honest failure handling, not a completed-film success.
- Another internal user can install, run, inspect and resume the same workflow without an engineer.

Record pass/fail per film and per requirement, with sample counts and disagreement between reviewers.
Choose and publish a quantitative launch threshold before expanding evaluation. Do not claim
community superiority until a comparable baseline and independent judgments support it.

## Technical experiments after the controller

Prioritize failure-specific tests over adding mathematical machinery without a measured outcome.
Compare fixed-duration versus shorter single-event shots; native versus Turbo sampling; continuation
frames versus new compositions; original identity references versus incidental old-action images;
and endpoint-only versus denser motion inspection. Hold other inputs fixed for each comparison.
Measure whether each change improves film success per unit cost, not just isolated image scores.

Use typed events and explicit state transitions to catch impossible plans. Use lattice semantics
for evidence status—unknown, supported, contradicted—only with explicit conflict handling; never
promote planned text into observed truth. Use associative summaries for efficient eligible-history
retrieval and invalidation while retaining exact evidence for consequential facts. Benchmark recall
quality on actual H3 story tasks before adding learned memory to the customer claim.

Keep the end goal visible: customers finish stories they can share and edit. Better algebra,
faster kernels, more checks, or longer demos matter only when they improve that outcome.

### Verifier output recovery

The local verifier requests concise observations and permits one automatic retry when an
assessment violates the JSON output contract. The retry uses the same source frames and does
not generate another video. Both raw responses are saved as `model-response-attempt-*.txt`.
A valid failure or uncertain assessment is not retried to obtain a pass; two malformed responses
leave production needing attention.

Assessments are bound to a verification protocol version as well as source and model hashes.
After a verification protocol change, use a new production output directory: old assessments
must not silently satisfy the new policy. Machine assessments remain fallible and are never
creator approval, even when the output is valid JSON.

## Explicit continuous visibility

Prepared shots may declare `visible_throughout`, a list of entities from `present` that must stay
on screen, including the final frame. Leave it empty for shots that permit exits, cutaways or
occlusion. A cast member appearing somewhere in a shot does not imply continuous visibility.
The constraint participates in the shot digest and renderer prompt; existing shots without it
retain their historical shot identity.

The versioned audit requests independent visibility observations for each declared entity at each
sampled frame. A reported absence fails even when every model check label says pass. Missing or
uncertain observations require review. These findings feed the ordinary bounded controller and
its recorded correction path; no film-specific names or manual take replacements are required.

This contract prevents a reported contradiction from being accepted. It does not make the model's
observations correct, inspect every video frame, or establish improved film quality. Test those
properties separately on frozen customer runs. Do not change a running film's plan or verifier to
retrofit this requirement; preserve that run and use a new, explicitly versioned acceptance run.

## Full visibility is different from presence

Shots may additionally declare `fully_visible_throughout`, a unique subset of `present`. The film
editor exposes this as **Must remain fully visible**. It requires each named subject to remain
entirely inside the picture and unobscured by other subjects or objects at every inspected frame.
Ordinary self-occlusion from a natural viewpoint is allowed. Leave this empty when the story permits
overlap, occlusion or close framing; it is not inferred from the cast roster.

A separate observer sees identity references and one current frame, without the screenplay or
desired visibility. It records `entire`, `partial`, `absent` or `uncertain`; code compares those
observations with the authored requirement. It checks the ending first, then the other sampled
frames, and stops this extra inspection on a contradiction or uncertainty. A pass requires all
sampled frames. This can add up to nine observations per constrained shot; it runs only when
full visibility is explicitly requested. The separate assembled-film audit may inspect the
selected shots again. Partial visibility or absence fails this requirement
even if the entity remains countable and all general check labels say pass. Missing observations
or uncertainty require review. No new canon approval is created. The existing presence requirement
retains its meaning, and empty full-visibility declarations preserve historical shot and plan
identities. Editing this requirement changes the shot digest and invalidates dependent work.

This adds an explicit checkable contract; it does not make the visual observer infallible or cover
unsampled frames. Reported machine success still requires creative review.
