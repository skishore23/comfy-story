# Security policy

Comfy Story currently targets a trusted, single ComfyUI installation. Its project and media routes
use the host's access boundary; a content hash authenticates bytes, not the requesting person.
It does not provide account authentication, per-user project authorization, tenant isolation,
or storage quotas. Do not expose a shared installation to untrusted users without an authenticated
access layer and separate storage/process boundaries. Same-origin checks do not replace these controls.

Workflow prompts, filenames and imported projects are untrusted input. Keep path containment,
immutable receipt validation, bounded uploads and checkpoint authentication intact when contributing.
The custom node must not install dependencies during startup or generation. Use the explicit
installer in ComfyUI's interpreter, preserving its Torch/CUDA packages.

The bundle manifest detects corruption and unexpected files. It is not a publisher signature:
obtain releases and their checksums from a trusted maintainer-controlled channel. A modified installer
can bypass its own checks. Private release signing and reporting arrangements must be configured
before a public release is advertised as verified.

Do not attach credentials, private media, complete environment dumps or exploit details to a public
issue. Use the repository's private vulnerability reporting channel if enabled. Otherwise ask a
maintainer for a private contact without posting sensitive details. Private reporting availability
is a release checklist item, not an established service-level commitment.

The current supported security target is the latest reviewed source candidate. No long-term support
or response-time guarantee is advertised yet. Models and separately installed custom nodes have
their own licenses and security/update policies.
