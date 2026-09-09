# Validation and private alpha gate

Repository visibility must remain private. Publication requires an explicit owner decision after
reviewing the evidence below; no script or workflow changes visibility automatically.

## Extraction checks

The extraction retains the existing Story, film, memory, checkpoint, frontend, and installer tests.
A dependency-closure check rejects imports of omitted research modules, and lazy public exports
are exercised. The source inventory records every file examined in the merged Duet main snapshot.

The initial standalone source passed:

- Ruff formatting, lint, format verification, and strict mypy (168 checked source files).
- 1,381 Python tests passed; 4 tests skipped for external-runtime or version prerequisites.
- All 80 JavaScript frontend tests passed.
- Offline lockfile consistency, wheel/source builds, exact wheel module inventory, and an installed
  wheel smoke test of the retained lazy exports and both product CLI commands.
- Source inventory validation against merged Duet main
  `b2e93d805b71bedcc2099c8d60b0887d42a149d7`, and no broken local documentation links.

Local tests used the existing Intel macOS environment (Python 3.12.8, Torch 2.2.2). This checks CPU
contracts without replacing that environment; it does not satisfy the supported Torch >=2.8 runtime
installation gate. GitHub Quality installs the locked Linux/Python 3.11 environment independently.
HTTP tests need permission to bind a local loopback test server; the final complete suite ran with
that permission. No assertions or integrity checks were disabled.

The upstream candidate also passed formatting, lint, mypy, 3,622 Python tests (7 skipped), and 80
frontend tests. Its only sandbox failure was the HTTP server bind; that test passed with loopback
access. The final merged main has exactly the same Git tree as that tested candidate.

## Required runtime evidence before considering publication

- Install the built bundle into a fresh supported ComfyUI environment with the documented H3
  model and runtime versions. Verify dependency preflight, node registration, and first generation.
- Generate a two-shot sequence, save/reload it, restart ComfyUI, and recover completed outputs with
  matching revisions and hashes. Verify branching and a changed shot.
- Generate a fresh film through the Story editor. Verify exact duration, audio offsets, export,
  interrupted-run recovery, and shot editing. Review the actual video for cast and prop consistency,
  causal story progression, language compliance, unplanned speech, and visual/motion defects.
- Test staged upgrade and rollback while preserving the original story store and workflows.
- Review the standalone source and provenance before public visibility. Keep models, credentials,
  private media, and generated outputs outside Git.

The upstream PR stack documented successful Linux film completion but also unresolved story,
refinement, and unplanned-speech concerns. Those results are inherited context, not a fresh GPU
acceptance run of this extracted repository. See [customer criteria](DUET_STORY_CUSTOMER_ACCEPTANCE.md).
