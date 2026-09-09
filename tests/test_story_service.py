from pathlib import Path

import pytest
import torch

from comfy_story.story_contracts import ReferenceRole, ShotIntent, StoryLibrary, StoryReference
from comfy_story.story_service import StoryGenerationRequest, prepare_story_generation
from comfy_story.story_store import StoryProjectStore


@pytest.mark.parametrize(
    ("count", "starting_frame", "accepted"), [(8, True, True), (9, True, False), (9, False, True)]
)
def test_native_reference_budget_includes_starting_frame(
    tmp_path: Path,
    count: int,
    starting_frame: bool,
    accepted: bool,
) -> None:
    store = StoryProjectStore(tmp_path)
    digest = store.put_asset(b"reference fixture")
    image = torch.full((1, 8, 8, 3), 0.5)
    references = tuple(
        StoryReference(
            f"Ref{index}",
            ReferenceRole.CHARACTER,
            "",
            digest,
            f"comfy-story://assets/sha256/{digest}",
            (digest,),
            "a" * 64,
        )
        for index in range(count)
    )
    request = StoryGenerationRequest(
        intent=ShotIntent.START_STORY,
        project_id="budget",
        branch_id="main",
        previous_story=None,
        previous_frame=None,
        world_frame=image if starting_frame else None,
        library=StoryLibrary("Budget", references),
        reference_images={reference.name: image for reference in references},
        prompt=" ".join("@" + reference.name for reference in references),
        shot_length_seconds=5,
        variation=1,
        model_configuration_sha256="b" * 64,
        composition="Continue frame" if starting_frame else "New composition",
    )
    if not accepted:
        with pytest.raises(ValueError, match="at most nine semantic guides"):
            prepare_story_generation(request, store=store)
        return
    prepared = prepare_story_generation(request, store=store)
    assert len(prepared.visual_guides) == 9
    assert set(prepared.active_reference_names) == {reference.name for reference in references}
    assert prepared.recall_decision is not None
    assert len(prepared.recall_decision.guide_bindings) == 9
