# Comfy Story naming and compatibility

Comfy Story is the customer-facing ComfyUI product, previously called Duet Story.
Duet remains the research architecture and Python package name.

The public node appears as **Comfy Story** in **Comfy / Story**. The film editor,
installer help and customer documentation use the same product name.

Existing node class IDs (`DuetStory`, `DuetStoryCanon`), socket types, API routes,
extension identifiers, asset paths, storage directories and immutable recipe formats
are unchanged. Old display names remain recognized by the editor. Do not replace
node type IDs or rename the installed custom-node directory to migrate workflows.
Previously saved canvases can retain their custom titles; their nodes still load.
Document and script filenames are retained so existing links and commands work.

The rename changes presentation, not generation quality or model support. MiniMax H3
remains the supported rendering engine. Character consistency and film coherence
remain subject to the documented acceptance criteria; the rename adds no quality claim.
