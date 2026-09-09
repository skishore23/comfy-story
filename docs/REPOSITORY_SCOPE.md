# Standalone repository scope

Comfy Story starts from the merged `main` snapshot of `skishore23/duet`, after PRs #5, #6, #7, and #8.
The new repository has fresh history so unrelated research history and local working-copy changes
are not transferred. The exact upstream commit, SHA-256 of every examined source file, retention
choices, and dependency roots are recorded in [SOURCE_INVENTORY.json](SOURCE_INVENTORY.json).

## Retained

The Comfy Story extension, editor, example workflows and film recipes; film planning, rendering,
review and export; native reference archives and secure story storage; shared fusion, typed state
and checkpoint contracts; optional trained H3 memory, reference compilation, and historical LTX
readers required by existing Story backends. Their transitive dependencies remain in the source.
Tests, installer, quality configuration, customer guides, attribution, and licensing accompany them.
Two small Decision 1 configurations support retained memory/checkpoint regression tests.

## Omitted

The WHISPERS manuscript; Salinas, EuroSAT, Assembly101 and HOTC benchmarks; synthetic/Foley audio
training, acquisition and evaluation; audio demo and model code; standalone LTX research and
production orchestration tools; unused research CLI commands; remote RunPod launchers; research
plans and unrelated configurations. They remain available in the original repository.
Uncommitted Live/FastH3 work is excluded because it is outside the merged PR candidate.

## Standalone adjustments

The distribution is named `comfy-story`, with focused runtime/development dependencies and a new
lockfile. The installer and wheel agree on that name. Both new product CLI aliases and existing
`duet-story-*` commands remain available. Python modules, node IDs, routes, workflow fields, project
formats, and checkpoint class paths retain their historical names. Root lazy exports for omitted
research APIs are removed; retained memory exports and their structural canary remain tested.

Customer documents keep their historical filenames to preserve links and installer contracts.
References to research-only documents point to the exact upstream snapshot. The original Duet
checkout and its uncommitted work remain untouched. Publication is a separate decision after the
[private alpha gates](VALIDATION.md).
