import comfy_story


def test_public_runtime_api_resolves() -> None:
    assert set(comfy_story.__all__) == {
        "StoryGenerationRequest",
        "StorySampler",
        "NativeReferenceArchive",
        "prepare_story_generation",
        "StoryLibrary",
        "ShotIntent",
        "commit_native_story_generation",
        "StoryReference",
        "PreparedStoryGeneration",
        "ComfyStoryStateRef",
    }
    for name in comfy_story.__all__:
        assert getattr(comfy_story, name) is not None
