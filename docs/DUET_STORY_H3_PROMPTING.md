# H3 prompting in Comfy Story

New nodes and films created in the editor use **H3 automatic v1**. Write the visible action
in the existing story editor; the render approach selects the prompt structure. No additional
setup or LLM service is required. Existing saved recipes retain their original prompt format.
To try the new compiler on an existing film, select it under advanced shot settings and save a
new revision. This changes generation identity and requires new renders of affected shots.

The compiler follows MiniMax's official
[frame prompting guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
and [reference prompting guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).

- **Animate frame:** frame alignment followed by the three multimodal description sections.
  With an ending image, both images enter the text encoder, and the ending guide is moved
  onto the last exported frame, including shortened cuts.
- **Reference shot:** six sections distinguish reusable subjects, image sources, appearance
  retention, action and sound. Source numbers follow the actual encoder roster. An ending
  image is included in that roster as well as the final-frame guide.
- **Film direction:** authored state definitions become opening, persistent or ending visual
  instructions. Review labels and symbolic fact syntax are excluded. The compiler preserves
  the action, camera policy and exact quoted text; it does not invent screenplay events.
- **Changing appearance:** identity retention allows explicitly directed transformations.
  Selected shot evidence is described as selected appearance, never as confirmed observation.
- **Sound:** film shots request silence because the film exporter uses separately authored
  tracks. Direct shots retain authored language tags and dialogue strings. The compiler does
  not translate, synthesize speech, or guarantee that H3 renders lettering or language correctly.

The advanced legacy formats remain available. API workflows that omit `Prompt format` keep
`Current`; new API clients should explicitly send `H3 automatic v1`. Compiled-reference
experiments are not supported by this formatter because they do not provide the native roster.

The execution receipt stores the exact resolved prompt, compiler protocol, seed, model assets,
source images and ending image identity. Prompt changes invalidate completed-shot recovery;
old recipe bindings remain unchanged. A seed alone does not guarantee bitwise reproduction
across changed models, software or hardware.

## What this fixes and what remains to measure

These changes correct the conditioning contract and prompt construction. They do not establish
that long-film causality, generated dialogue, fine text, identity or loop seams are solved.
No generated film has been manually repaired as part of this fix.

Evaluate the new compiler using fresh human, nonhuman and abstract briefs through the saved
workflow and normal Generate action. Compare with legacy prompts using the same model, seed,
references and durations. Record completion, identity drift, premature endpoints, camera drift,
causal progression and cost. Assess the outputs before claiming a quality improvement.

A future optional screenplay expansion pass can turn sparse action into richer physical staging,
but must preserve explicit facts, language, source identities and seed recipes; it must expose its
saved compiled result. More words alone are not evidence of better H3 prompting.
