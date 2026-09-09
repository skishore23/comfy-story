from __future__ import annotations

import asyncio
import hashlib
import importlib
import io
import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from comfy_story.samplers import StorySampler

_MODULE = "integrations.comfy_story"


@dataclass(frozen=True)
class _Field:
    kind: str
    direction: str
    id: str | None
    options: dict[str, Any]


@dataclass(frozen=True)
class _Schema:
    node_id: str
    display_name: str
    category: str
    inputs: list[_Field]
    outputs: list[_Field]
    hidden: list[object] | None = None
    description: str = ""
    search_aliases: list[str] | None = None
    enable_expand: bool = False
    is_output_node: bool = False
    is_deprecated: bool = False


class _Type:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.Input = self._factory("input")
        self.Output = self._factory("output")

    def _factory(self, direction: str) -> Any:
        kind = self.kind

        def build(field_id: str | None = None, **options: Any) -> _Field:
            return _Field(kind, direction, field_id, options)

        return build


class _NodeOutput:
    def __init__(
        self,
        *args: Any,
        expand: dict[str, Any] | None = None,
        ui: dict[str, object] | None = None,
    ) -> None:
        self.args = args
        self.expand = expand
        self.ui = ui


class _GraphNode:
    def __init__(self, node_id: str, class_type: str, inputs: dict[str, Any]) -> None:
        self.node_id = node_id
        self.class_type = class_type
        self.inputs = inputs

    def out(self, index: int) -> list[object]:
        return [self.node_id, index]


class _GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, _GraphNode] = {}

    def node(self, class_type: str, **inputs: Any) -> _GraphNode:
        node_id = str(len(self.nodes) + 1)
        node = _GraphNode(node_id, class_type, inputs)
        self.nodes[node_id] = node
        return node

    def finalize(self) -> dict[str, Any]:
        return {
            node_id: {"class_type": node.class_type, "inputs": node.inputs}
            for node_id, node in self.nodes.items()
        }


class _ComfyNode:
    pass


class _ComfyExtension:
    pass


class _UploadType(Enum):
    image = "image"


class _Hidden(Enum):
    unique_id = "UNIQUE_ID"


@pytest.fixture
def continuity_integration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    for name in tuple(sys.modules):
        if name == _MODULE or name.startswith(_MODULE + "."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    latest = ModuleType("comfy_api.latest")
    latest.ComfyExtension = _ComfyExtension  # type: ignore[attr-defined]
    latest.io = SimpleNamespace(  # type: ignore[attr-defined]
        Audio=_Type("AUDIO"),
        Combo=_Type("COMBO"),
        ComfyNode=_ComfyNode,
        Conditioning=_Type("CONDITIONING"),
        Sigmas=_Type("SIGMAS"),
        Custom=lambda kind: _Type(kind),
        Image=_Type("IMAGE"),
        Hidden=_Hidden,
        Int=_Type("INT"),
        Latent=_Type("LATENT"),
        Model=_Type("MODEL"),
        NodeOutput=_NodeOutput,
        Schema=_Schema,
        String=_Type("STRING"),
        UploadType=_UploadType,
        Vae=_Type("VAE"),
        Video=_Type("VIDEO"),
    )
    comfy_api = ModuleType("comfy_api")
    comfy_api.latest = latest  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)
    graph_utils = ModuleType("comfy_execution.graph_utils")
    graph_utils.GraphBuilder = _GraphBuilder  # type: ignore[attr-defined]
    comfy_execution = ModuleType("comfy_execution")
    comfy_execution.graph_utils = graph_utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy_execution", comfy_execution)
    monkeypatch.setitem(sys.modules, "comfy_execution.graph_utils", graph_utils)
    folder_paths = ModuleType("folder_paths")
    folder_paths.get_user_directory = lambda: str(tmp_path / "user")  # type: ignore[attr-defined]
    folder_paths.get_input_directory = lambda: str(tmp_path)  # type: ignore[attr-defined]
    folder_paths.get_output_directory = lambda: str(tmp_path / "output")  # type: ignore[attr-defined]
    model_fixture = tmp_path / "model-fixture.bin"
    model_fixture.write_bytes(b"model fixture")
    folder_paths.get_full_path = lambda folder, name: str(model_fixture)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    server = ModuleType("server")
    server.PromptServer = SimpleNamespace(  # type: ignore[attr-defined]
        instance=SimpleNamespace(
            routes=SimpleNamespace(
                get=lambda path: lambda handler: handler,
                post=lambda path: lambda handler: handler,
                put=lambda path: lambda handler: handler,
            )
        )
    )
    monkeypatch.setitem(sys.modules, "server", server)
    return importlib.import_module(_MODULE)


def _image(value: float) -> torch.Tensor:
    return torch.full((1, 8, 12, 3), value, dtype=torch.float32)


def test_public_schema_is_one_chainable_value_node(continuity_integration: ModuleType) -> None:
    schema = continuity_integration.ComfyStory.define_schema()
    assert schema.node_id == "ComfyStory"
    assert schema.display_name == "Comfy Story"
    assert [field.id for field in schema.inputs] == [
        "Previous Story",
        "Previous Frame",
        "Create",
        "Story Library",
        "World / starting frame",
        "Motion reference",
        "What happens next?",
        "Shot length",
        "Variation",
        "Story revision",
        "Reference policy",
        "Sampler",
        "Memory actions",
        "Composition",
        "Authored audio",
        "Ending frame",
        "Scene entities",
        "Output duration (ms)",
        "Render profile",
        "Shot state evidence",
        "Starting image",
        "Prompt format",
    ]
    assert [(field.id, field.kind) for field in schema.outputs] == [
        ("video", "VIDEO"),
        ("last_frame", "IMAGE"),
        ("story_state", "COMFY_STORY"),
    ]
    world = next(field for field in schema.inputs if field.id == "World / starting frame")
    assert world.options["upload"].value == "image"
    assert schema.hidden == [_Hidden.unique_id]
    sampler = next(field for field in schema.inputs if field.id == "Sampler")
    assert sampler.options["options"] == [
        "Native res_multistep",
        "SPEED Euler 2-stage",
        "Turbo 4-step",
        "Turbo 8-step",
        "Full HD 2-pass",
        "NVFP4 Exact",
        "NVFP4 Balanced",
        "NVFP4 Ultra Fast",
        "NVFP4 Turbo 4-step",
    ]
    assert sampler.options["default"] == "Native res_multistep"
    assert all(field.id != "Reference context" for field in schema.inputs)


def test_extension_exposes_story_nodes_and_hidden_internal_nodes(
    continuity_integration: ModuleType,
) -> None:
    extension = asyncio.run(continuity_integration.comfy_entrypoint())
    registered = asyncio.run(extension.get_node_list())
    assert registered == [
        continuity_integration.ComfyStory,
        continuity_integration.ComfyStoryCommit,
        continuity_integration.ComfyStoryTrim,
        continuity_integration.ComfyStoryKeyframeTiming,
        continuity_integration.ComfyH3BalancedCache,
        continuity_integration.ComfyH3UltraFast,
        continuity_integration.ComfyH3VideoLatent,
        continuity_integration.ComfyH3ReplaceVideoLatent,
        continuity_integration.ComfyH3RefinementSigmas,
    ]
    assert registered[0].define_schema().is_deprecated is False
    assert all(not node.define_schema().is_deprecated for node in registered)
    assert all(node.define_schema().category == "Comfy/Story" for node in registered[:1])
    assert all(node.define_schema().category == "_Comfy/Internal" for node in registered[1:])


@pytest.mark.parametrize("node_name", ["ComfyStory"])
def test_living_canon_expansion_marks_request_and_routes_commit_to_visible_node(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, node_name: str
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1),),
        resolved_prompt="resolved prompt",
        frame_count=124,
        variation=123,
        sampler=StorySampler.NATIVE_RES_MULTISTEP,
    )
    observed: list[bool] = []

    def prepare(inputs: dict[str, object]) -> object:
        del inputs
        observed.append(True)
        return prepared

    monkeypatch.setattr(integration_nodes, "prepare_node_generation", prepare)
    monkeypatch.setattr(
        getattr(continuity_integration, node_name),
        "hidden",
        SimpleNamespace(unique_id="42"),
        raising=False,
    )

    output = getattr(continuity_integration, node_name).execute()

    assert observed == [True]
    assert output.expand["16"]["inputs"]["owner_node_id"] == "42"


def test_internal_commit_returns_inspector_metadata_for_the_visible_canon_node(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    loaded = object()
    monkeypatch.setattr(
        integration_nodes,
        "commit_node_generation",
        lambda **inputs: SimpleNamespace(state="state", last_frame="frame", loaded_revision=loaded),
    )
    monkeypatch.setattr(
        integration_nodes,
        "story_inspector_summary",
        lambda revision, *, owner_node_id: {
            "owner_node_id": owner_node_id,
            "revision": revision is loaded,
        },
    )

    output = continuity_integration.ComfyStoryCommit.execute(
        prepared="prepared",
        decoded_images="images",
        saved_video="video",
        filename_prefix="comfy_story/shot",
        memory_vae="vae",
        owner_node_id="42",
    )

    assert output.args == ("state", "frame")
    assert output.ui == {"comfy_story": [{"owner_node_id": "42", "revision": True}]}


def test_expand_generates_decodes_saves_then_commits(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1), _image(0.2), _image(0.3), _image(0.4)),
        resolved_prompt="resolved prompt",
        frame_count=124,
        variation=123,
        sampler=StorySampler.NATIVE_RES_MULTISTEP,
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )
    output = continuity_integration.ComfyStory.execute(
        **{
            "Previous Story": None,
            "Previous Frame": None,
            "Create": "Start Story",
            "Story Library": "{}",
            "World / starting frame": _image(0.1),
            "What happens next?": "@Maya opens @WoodenChest.",
            "Shot length": "5 seconds",
            "Variation": 123,
            "Story revision": "",
            "Reference policy": "Automatic",
            "Memory backend": "MiniMax H3",
        }
    )
    assert output.expand is not None
    assert [node["class_type"] for node in output.expand.values()] == [
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "VAELoader",
        "RandomNoise",
        "KSamplerSelect",
        "BasicScheduler",
        "MiniMaxH3ReferenceToVideo",
        "MiniMaxH3AddGuide",
        "BasicGuider",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
        "ComfyStoryCommit",
    ]
    commit = output.expand["16"]
    assert commit["inputs"]["decoded_images"] == ["12", 0]
    assert commit["inputs"]["saved_video"] == ["15", 0]
    assert "memory_vae" not in commit["inputs"]
    assert commit["inputs"]["filename_prefix"] == "comfy_story/shot"
    assert output.args == (["15", 0], ["16", 1], ["16", 0])
    generator = output.expand["8"]["inputs"]
    assert generator["width"] == 1344
    assert generator["height"] == 768
    assert generator["length"] == 124
    assert generator["prompt"] == "resolved prompt"


def test_native_story_graph_bypasses_compiler_exactly(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1), _image(0.2)),
        resolved_prompt="resolved prompt",
        frame_count=124,
        variation=123,
        sampler=StorySampler.NATIVE_RES_MULTISTEP,
        motion_reference=None,
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )

    output = continuity_integration.ComfyStory.execute()

    assert output.expand is not None
    types = [node["class_type"] for node in output.expand.values()]
    assert "ComfyH3CompileReferences" not in types
    native_id = next(
        node_id
        for node_id, node in output.expand.items()
        if node["class_type"] == "MiniMaxH3ReferenceToVideo"
    )
    guide = next(
        node for node in output.expand.values() if node["class_type"] == "MiniMaxH3AddGuide"
    )
    assert guide["inputs"]["positive"] == [native_id, 0]


@pytest.mark.parametrize(
    ("sampler", "steps", "adapter_name"),
    [
        (StorySampler.TURBO_4STEP, 4, "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"),
        (
            StorySampler.TURBO_8STEP,
            8,
            "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
        ),
    ],
)
def test_turbo_uses_ref2va_lora_and_coherent_schedule(
    continuity_integration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    sampler: StorySampler,
    steps: int,
    adapter_name: str,
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1), _image(0.2)),
        resolved_prompt="resolved prompt",
        frame_count=243,
        variation=123,
        sampler=sampler,
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )
    output = continuity_integration.ComfyStory.execute()
    assert output.expand is not None
    nodes = {n["class_type"]: (key, n["inputs"]) for key, n in output.expand.items()}
    lora_id, lora = nodes["LoraLoaderModelOnly"]
    shift_id, shift = nodes["MiniMaxH3SigmaShift"]
    assert lora["lora_name"] == adapter_name
    assert nodes["UNETLoader"][1]["unet_name"] == (
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    )
    assert lora["strength_model"] == 1.0
    assert lora["model"] == [nodes["UNETLoader"][0], 0]
    assert shift == {"model": [lora_id, 0], "shift_video": 12.0, "shift_audio": 3.0}
    assert nodes["BasicScheduler"][1] == {
        "model": [shift_id, 0],
        "scheduler": "simple",
        "steps": steps,
        "denoise": 1.0,
    }
    assert nodes["BasicGuider"][1]["model"] == [shift_id, 0]
    assert nodes["KSamplerSelect"][1]["sampler_name"] == "euler"
    conditioning = nodes["MiniMaxH3ReferenceToVideo"][1]
    assert (conditioning["width"], conditioning["height"], conditioning["length"]) == (
        1344,
        768,
        243,
    )
    assert conditioning["ref_image_size"] == "match"
    assert "SamplerCustomAdvanced" in nodes
    assert "MiniMaxH3SPEEDSampler" not in nodes
    assert "ComfyStoryCommit" in nodes


def test_speed_sampler_replaces_only_native_sampling_nodes(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")

    class _PublishedSpeedNode:
        @classmethod
        def INPUT_TYPES(cls) -> dict[str, object]:
            del cls
            return {
                "required": {
                    "noise": ("NOISE",),
                    "guider": ("GUIDER",),
                    "sigmas": ("SIGMAS",),
                    "latent_image": ("LATENT",),
                    "stages": ("INT",),
                    "noise_policy": (["direct_coarse", "coupled_full_grid"],),
                    "Tolerance (Delta)": ("FLOAT",),
                    "noise_amplitude": ("FLOAT",),
                    "noise_decay_exponent": ("FLOAT",),
                    "seed_offset": ("INT",),
                }
            }

    comfy_nodes = ModuleType("nodes")
    comfy_nodes.NODE_CLASS_MAPPINGS = {  # type: ignore[attr-defined]
        "MiniMaxH3SPEEDSampler": _PublishedSpeedNode
    }
    monkeypatch.setitem(sys.modules, "nodes", comfy_nodes)
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1), _image(0.2)),
        resolved_prompt="resolved prompt",
        frame_count=124,
        variation=123,
        sampler=StorySampler.SPEED_EULER_2STAGE,
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )

    output = continuity_integration.ComfyStory.execute()

    assert output.expand is not None
    class_types = [node["class_type"] for node in output.expand.values()]
    assert class_types.count("MiniMaxH3SPEEDSampler") == 1
    assert "KSamplerSelect" not in class_types
    assert "SamplerCustomAdvanced" not in class_types
    speed = next(
        node["inputs"]
        for node in output.expand.values()
        if node["class_type"] == "MiniMaxH3SPEEDSampler"
    )
    assert speed["stages"] == 2
    assert speed["noise_policy"] == "direct_coarse"
    assert speed["Tolerance (Delta)"] == 0.01
    assert speed["noise_amplitude"] == 7.394
    assert speed["noise_decay_exponent"] == 0.62
    assert speed["seed_offset"] == 10_000


def test_speed_sampler_fails_closed_when_external_node_is_missing(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    monkeypatch.delitem(sys.modules, "nodes", raising=False)
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1),),
        resolved_prompt="resolved prompt",
        frame_count=124,
        variation=123,
        sampler=StorySampler.SPEED_EULER_2STAGE,
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )

    with pytest.raises(ValueError, match="MiniMaxH3SPEEDSampler is not installed"):
        continuity_integration.ComfyStory.execute()


def test_speed_sampler_fails_closed_on_external_schema_drift(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")

    class _DriftedSpeedNode:
        @classmethod
        def INPUT_TYPES(cls) -> dict[str, object]:
            del cls
            return {"required": {"noise": ("NOISE",)}}

    comfy_nodes = ModuleType("nodes")
    comfy_nodes.NODE_CLASS_MAPPINGS = {  # type: ignore[attr-defined]
        "MiniMaxH3SPEEDSampler": _DriftedSpeedNode
    }
    monkeypatch.setitem(sys.modules, "nodes", comfy_nodes)
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1),),
        resolved_prompt="resolved prompt",
        frame_count=124,
        variation=123,
        sampler=StorySampler.SPEED_EULER_2STAGE,
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )

    with pytest.raises(ValueError, match="schema is incompatible"):
        continuity_integration.ComfyStory.execute()


def test_memory_actions_are_canonical_bounded_and_stale_safe_at_service_boundary(
    continuity_integration: ModuleType,
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    encoded = json.dumps(
        [
            {
                "action": "use_in_this_shot",
                "parent_revision_sha256": "a" * 64,
                "presence": None,
                "state_note": "",
                "supporting_evidence_ids": [],
                "target_id": "key-bent",
            }
        ],
        separators=(",", ":"),
        sort_keys=True,
    )

    commands = adapter.parse_memory_actions(encoded)

    assert commands[0].target_id == "key-bent"
    with pytest.raises(ValueError, match="canonical"):
        adapter.parse_memory_actions(encoded + "\n")


def test_invalid_input_fails_before_generation_graph(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    monkeypatch.setattr(
        integration_nodes,
        "prepare_node_generation",
        lambda inputs, **kwargs: (_ for _ in ()).throw(ValueError("too many exact references")),
    )
    with pytest.raises(ValueError, match="too many exact references"):
        continuity_integration.ComfyStory.execute()


def test_saved_video_path_recovers_native_savevideo_destination(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    output = tmp_path / "output" / "comfy_story"
    output.mkdir(parents=True)
    saved = output / "shot_00003_.mp4"
    saved.write_bytes(b"finished-video")
    monkeypatch.setattr(
        adapter.folder_paths,
        "get_save_image_path",
        lambda prefix, root, width, height: (
            str(output),
            "shot",
            4,
            "comfy_story",
            prefix,
        ),
        raising=False,
    )
    video = SimpleNamespace(get_dimensions=lambda: (1344, 768))

    assert adapter._saved_video_path(video, "comfy_story/shot") == saved


def test_legacy_archive_commit_recall_and_recovery_without_checkpoint(
    continuity_integration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from comfy_story.story_contracts import canonical_story_json
    from comfy_story.story_native_archive import NativeArchiveRevision
    from comfy_story.story_product_contracts import (
        CanonPresence,
        MemoryAction,
        ObservationKind,
        StoryMemoryCommand,
    )
    from comfy_story.story_store import StoryProjectStore

    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    # Exercise historical archive contracts, not the current production configuration gate.
    monkeypatch.setattr(adapter, "configured_memory", lambda: None)
    monkeypatch.delenv("COMFY_STORY_MEMORY_RUNTIME", raising=False)
    monkeypatch.delenv("COMFY_STORY_MINIMAX_CHECKPOINT", raising=False)
    monkeypatch.delenv("COMFY_H3_COMPILER_INTERNAL_TEST", raising=False)
    monkeypatch.setenv("COMFY_STORY_ROOT", str(tmp_path / "stories"))

    def forbidden_checkpoint() -> None:
        pytest.fail("Legacy archive fixture must not load a trained checkpoint")

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: forbidden_checkpoint())
    reference = tmp_path / "acorn.png"
    Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(reference)
    inputs = {
        "Create": "Start Story",
        "Story Library": json.dumps(
            {
                "project_name": "Acorn",
                "references": [
                    {"file": reference.name, "name": "Acorn", "role": "Prop", "note": "shell"}
                ],
            }
        ),
        "World / starting frame": _image(0.1),
        "What happens next?": "@Acorn rests on moss.",
        "Shot length": "5 seconds",
        "Variation": 321,
        "Reference policy": "Automatic",
    }
    prepared = adapter.prepare_node_generation(inputs)
    assert not hasattr(prepared.prepared.request, "checkpoint_sha256")
    connected = adapter.prepare_node_generation(
        {**inputs, "World / starting frame": "missing-unused.png", "Starting image": _image(0.1)},
    )
    # An upstream image takes precedence over an unused filename and binds the
    # same execution as the identical historical starting pixels.
    torch.testing.assert_close(connected.visual_guides[0], _image(0.1), rtol=0, atol=0)
    assert connected.prepared.request.execution_sha256 == prepared.prepared.request.execution_sha256
    changed_start = adapter.prepare_node_generation({**inputs, "Starting image": _image(0.2)})
    assert (
        changed_start.prepared.request.execution_sha256
        != prepared.prepared.request.execution_sha256
    )
    with pytest.raises(ValueError, match="Starting image must contain exactly one image"):
        adapter.prepare_node_generation({**inputs, "Starting image": torch.ones((2, 16, 16, 3))})
    store = StoryProjectStore(tmp_path / "stories")
    binding = json.loads(store.load_asset(prepared.prepared.request.execution_sha256))
    assert "memory_runtime" in binding
    assert "memory_checkpoint" not in binding
    assert binding["native_capture_policy"] == adapter.NATIVE_RGB_CAPTURE_POLICY_SHA256
    automatic = adapter.prepare_node_generation({**inputs, "Prompt format": "H3 automatic v1"})
    new_binding = json.loads(store.load_asset(automatic.prepared.request.execution_sha256))
    assert new_binding["prompt_compiler"] == "comfy-h3-mode-aware-v1"
    assert "prompt_compiler" not in binding
    assert new_binding["variation"] == binding["variation"]
    assert new_binding["guides"] == binding["guides"]
    assert "subject_definitions:" in new_binding["prompt"]
    assert automatic.prepared.request.execution_sha256 != prepared.prepared.request.execution_sha256
    legacy_again = adapter.prepare_node_generation(inputs)
    assert (
        legacy_again.prepared.request.execution_sha256 == prepared.prepared.request.execution_sha256
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"container fixture; media decoding is covered separately")
    frames = torch.full((3, 8, 12, 3), 0.314159)
    frames[..., 0] = torch.linspace(0, 1, 12)
    frames[..., 1] = torch.linspace(0, 1, 8).unsqueeze(-1)
    committed = adapter.commit_node_generation(
        prepared=prepared.prepared, decoded_images=frames, saved_video=video
    )
    assert isinstance(committed.loaded_revision, NativeArchiveRevision)
    restarted = adapter.prepare_node_generation(inputs)
    recovered = adapter.recover_node_generation(restarted)
    assert recovered is not None
    loaded, saved, last = recovered
    assert loaded.state == committed.state
    assert saved.read_bytes() == video.read_bytes()
    torch.testing.assert_close(last, frames[-1:], rtol=0, atol=0)
    assert adapter.story_inspector_summary(loaded, owner_node_id="1")["shot_count"] == 1
    for intent, expected_frame in (("New Scene", _image(0.6)), ("Next Shot", last)):
        staged = adapter.prepare_node_generation(
            {
                **inputs,
                "Create": intent,
                "Previous Story": loaded.state,
                "Previous Frame": last,
                "Starting image": _image(0.6),
            },
        )
        torch.testing.assert_close(staged.visual_guides[0], expected_frame, rtol=0, atol=0)
        assert staged.prepared.request.previous_story == loaded.state
    with monkeypatch.context() as policy:
        policy.setattr(adapter, "NATIVE_RGB_CAPTURE_POLICY_SHA256", "0" * 64)
        old_capture = adapter.prepare_node_generation(inputs)
        assert adapter.recover_node_generation(old_capture) is None
    changed = adapter.prepare_node_generation({**inputs, "Variation": 322})
    assert adapter.recover_node_generation(changed) is None
    product = loaded.require_product_state()
    closing = next(e for e in product.evidence_records if e.kind is ObservationKind.CLOSING)
    for observation in product.observation_packets[0].observations:
        with Image.open(io.BytesIO(store.load_asset(observation.asset_sha256))) as image:
            assert image.size == (12, 8)
            expected = frames[observation.frame_index].mul(255).round().to(torch.uint8).numpy()
            np.testing.assert_array_equal(np.asarray(image), expected)
    assert closing.asset_sha256 == loaded.state.last_frame_sha256
    command = StoryMemoryCommand(
        MemoryAction.UPDATE_CANON,
        loaded.state.revision_sha256,
        "acorn",
        "The shell is wet.",
        CanonPresence.PRESENT,
        (closing.evidence_id,),
    )
    continuation = {
        **inputs,
        "Create": "Next Shot",
        "Previous Story": loaded.state,
        "Previous Frame": last,
        "Composition": "New composition",
        "Memory actions": canonical_story_json([command]).decode(),
    }
    one_shot_inputs = {
        **continuation,
        "Memory actions": "[]",
        "Shot state evidence": canonical_story_json({"acorn": closing.evidence_id}).decode(),
    }
    one_shot = adapter.prepare_node_generation(one_shot_inputs)
    assert one_shot.prepared.product_state.canon == product.canon
    assert one_shot.prepared.visual_roles == ("evidence-acorn",)
    assert one_shot.prepared.visual_guides[0].shape == (1, 8, 12, 3)
    torch.testing.assert_close(
        one_shot.prepared.visual_guides[0], frames[-1:].mul(255).round().div(255), rtol=0, atol=0
    )
    # Render conditioning uses the current appearance without discarding the
    # original identity or rewriting the historical evidence packet/canon.
    assert one_shot.prepared.library == prepared.prepared.library
    assert one_shot.prepared.recall_decision is not None
    packet = one_shot.prepared.recall_decision.packet_specs[0]
    assert len(packet.source_sha256s) == 2
    assert packet.source_sha256s[-1] == closing.asset_sha256
    assert "<Picture 1> is selected earlier visual evidence" in one_shot.resolved_prompt
    assert "<Picture 2>" not in one_shot.resolved_prompt
    assert "selected earlier visual evidence" in one_shot.resolved_prompt
    assert "Approved state of @Acorn" not in one_shot.resolved_prompt
    one_binding = json.loads(store.load_asset(one_shot.prepared.request.execution_sha256))
    assert one_binding["shot_state_evidence"] == [["acorn", closing.evidence_id]]
    one_commit = adapter.commit_node_generation(
        prepared=one_shot.prepared, decoded_images=frames, saved_video=video
    )
    prior_canon = product.canon.entities[0]
    next_canon = one_commit.loaded_revision.product_state.canon.entities[0]
    assert next_canon.supporting_evidence_ids == prior_canon.supporting_evidence_ids == ()
    assert next_canon.confirmed_revision_sha256 is prior_canon.confirmed_revision_sha256 is None
    assert next_canon.state_note == prior_canon.state_note
    assert len(next_canon.pending) == len(prior_canon.pending) + 1
    one_recovery = adapter.recover_node_generation(adapter.prepare_node_generation(one_shot_inputs))
    assert one_recovery is not None
    assert one_recovery[0].state == one_commit.state
    no_evidence = adapter.prepare_node_generation({**one_shot_inputs, "Shot state evidence": "{}"})
    assert (
        no_evidence.prepared.request.execution_sha256 != one_shot.prepared.request.execution_sha256
    )
    assert adapter.recover_node_generation(no_evidence) is None
    next_shot = adapter.prepare_node_generation(continuation)
    assert "current" not in next_shot.prepared.visual_roles
    assert "inclusive-core" not in next_shot.prepared.visual_roles
    assert next_shot.prepared.recall_decision.selected_evidence_ids == (closing.evidence_id,)
    assert "Approved state of @Acorn: The shell is wet." in next_shot.resolved_prompt
    second = adapter.commit_node_generation(
        prepared=next_shot.prepared, decoded_images=frames, saved_video=video
    )
    assert second.state.shot_count == 2
    assert (
        second.loaded_revision.require_product_state().observation_packets[0]
        == product.observation_packets[0]
    )
    assert loaded.require_product_state() == product
    stored_video = store.root / "assets" / "sha256" / hashlib.sha256(video.read_bytes()).hexdigest()
    stored_video.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        adapter.recover_node_generation(restarted)


def test_story_media_routes_reject_wrong_assets_and_expose_range_video(
    continuity_integration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asyncio

    from aiohttp import web

    from comfy_story.story_store import StoryProjectStore

    routes = importlib.import_module(f"{_MODULE}.routes")
    monkeypatch.setenv("COMFY_STORY_ROOT", str(tmp_path / "story"))
    store = StoryProjectStore(tmp_path / "story")
    clip = store.put_asset(b"\x00\x00\x00\x18ftypmp42" + b"test clip")

    def reject_buffered_load(*args: object) -> bytes:
        raise AssertionError("video serving must not buffer the full asset")

    import threading

    caller_thread = threading.get_ident()
    verify = StoryProjectStore.verified_asset_path

    def verify_off_loop(self: StoryProjectStore, digest: str) -> Path:
        assert threading.get_ident() != caller_thread
        return verify(self, digest)

    monkeypatch.setattr(StoryProjectStore, "verified_asset_path", verify_off_loop)
    monkeypatch.setattr(StoryProjectStore, "load_asset", reject_buffered_load)
    response = asyncio.run(routes.accepted_video(SimpleNamespace(match_info={"digest": clip})))
    assert isinstance(response, web.FileResponse)
    assert response.headers["Content-Type"] == "video/mp4"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    wrong = store.put_asset(b"not a video")
    for digest in (wrong, "../escape", "0" * 64):
        with pytest.raises(web.HTTPNotFound):
            asyncio.run(routes.accepted_video(SimpleNamespace(match_info={"digest": digest})))


@pytest.mark.parametrize("authored", [False, True])
def test_editorial_cut_keeps_identity_references_without_frame_zero_constraint(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, authored: bool
) -> None:
    from comfy_story.story_language import prepare_authored_audio

    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1), _image(0.2)),
        resolved_prompt="new camera angle with the same person",
        frame_count=124,
        variation=123,
        sampler=StorySampler.NATIVE_RES_MULTISTEP,
        authored_audio=prepare_authored_audio(
            {"waveform": torch.full((1, 1, 8_000), 0.125), "sample_rate": 8_000}
            if authored
            else None,
            frame_count=124,
        ),
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )
    result = continuity_integration.ComfyStory.execute(Composition="New composition")
    graph = result.expand
    assert graph is not None
    assert "MiniMaxH3AddGuide" not in {node["class_type"] for node in graph.values()}
    generator = next(
        node for node in graph.values() if node["class_type"] == "MiniMaxH3ReferenceToVideo"
    )
    assert "ref_images.ref_image_0" in generator["inputs"]
    assert "ref_images.ref_image_1" in generator["inputs"]
    assert any(node["class_type"] == "ComfyStoryCommit" for node in graph.values())
    classes = {node["class_type"] for node in graph.values()}
    assert ("VAEDecodeAudio" in classes) is not authored
    if authored:
        video = next(node for node in graph.values() if node["class_type"] == "CreateVideo")
        output = video["inputs"]["audio"]["waveform"]
        assert output.shape[-1] == round(124 * 8_000 / 24)
        torch.testing.assert_close(output[..., :8_000], torch.full((1, 1, 8_000), 0.125))


def test_story_storage_defaults_to_this_comfy_installation(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    monkeypatch.delenv("COMFY_STORY_ROOT", raising=False)
    assert adapter._story_root() == tmp_path / "user" / "comfy_story"
    assert not (tmp_path / "user").exists(), "Resolving configuration must not create storage"
    monkeypatch.setenv("COMFY_STORY_ROOT", str(tmp_path / "custom"))
    assert adapter._story_root() == tmp_path / "custom"


@pytest.mark.parametrize("value", ["", " ", "relative/stories"])
def test_invalid_explicit_storage_does_not_fall_back_to_another_project(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    monkeypatch.setenv("COMFY_STORY_ROOT", value)
    with pytest.raises(ValueError, match="COMFY_STORY_ROOT"):
        adapter._story_root()


@pytest.mark.parametrize("duration", [2375, 5000])
def test_exact_interval_feeds_both_saved_video_and_committed_last_frame(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, duration: int
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1),),
        resolved_prompt="action",
        frame_count=124,
        variation=7,
        sampler=StorySampler.TURBO_4STEP,
        output_duration_ms=duration,
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )
    output = continuity_integration.ComfyStory.execute()
    graph = output.expand
    trim_id, trim = next(
        (key, node) for key, node in graph.items() if node["class_type"] == "ComfyStoryTrim"
    )
    assert trim["inputs"]["duration_ms"] == duration
    video = next(node for node in graph.values() if node["class_type"] == "CreateVideo")
    commit = next(node for node in graph.values() if node["class_type"] == "ComfyStoryCommit")
    assert video["inputs"]["images"] == commit["inputs"]["decoded_images"] == [trim_id, 0]
    assert video["inputs"]["audio"] == [trim_id, 1]


def test_starting_image_choices_include_portable_subfolders_and_exclude_symlinks(
    continuity_integration: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "input"
    nested = root / "comfy_story_inputs"
    nested.mkdir(parents=True)
    (nested / "scene.png").write_bytes(b"image")
    (root / "root.jpg").write_bytes(b"image")
    (nested / "notes.txt").write_text("not an image")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (root / "linked.png").symlink_to(outside)
    folder_paths = importlib.import_module("folder_paths")
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(root))
    schema = continuity_integration.ComfyStory.define_schema()
    world = next(field for field in schema.inputs if field.id == "World / starting frame")
    assert world.options["options"] == ["None", "comfy_story_inputs/scene.png", "root.jpg"]


@pytest.mark.parametrize(("duration", "expected"), [(0, 123), (2375, 56), (5000, 119)])
def test_ending_guide_anchors_the_final_saved_frame(
    continuity_integration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    duration: int,
    expected: int,
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1),),
        resolved_prompt="action",
        frame_count=124,
        variation=7,
        sampler=StorySampler.TURBO_4STEP,
        output_duration_ms=duration,
        ending_frame=_image(0.8),
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )
    graph = continuity_integration.ComfyStory.execute().expand
    guides = [
        (key, node) for key, node in graph.items() if node["class_type"] == "MiniMaxH3AddGuide"
    ]
    assert [node["inputs"]["frame_idx"] for _, node in guides] == [0, expected]
    assert guides[-1][1]["inputs"]["positive"] == [guides[0][0], 0]
    guider = next(node for node in graph.values() if node["class_type"] == "BasicGuider")
    assert guider["inputs"]["conditioning"] == [guides[-1][0], 0]


@pytest.mark.parametrize(
    "sampler",
    [StorySampler.NATIVE_RES_MULTISTEP, StorySampler.TURBO_4STEP, StorySampler.TURBO_8STEP],
)
def test_animate_frame_uses_its_own_checkpoint_and_exact_ending_guide(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, sampler: StorySampler
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1),),
        resolved_prompt="The leaf turns.",
        frame_count=124,
        variation=7,
        sampler=sampler,
        render_profile="Animate frame",
        output_duration_ms=5000,
        ending_frame=_image(0.8),
    )
    monkeypatch.setattr(integration_nodes, "prepare_node_generation", lambda inputs, **kw: prepared)
    graph = continuity_integration.ComfyStory.execute().expand
    nodes = {n["class_type"]: (key, n["inputs"]) for key, n in graph.items()}
    assert nodes["UNETLoader"][1]["unet_name"] == "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    assert "MiniMaxH3ReferenceToVideo" not in nodes
    assert nodes["ImageScale"][1]["image"] is prepared.visual_guides[0]
    assert nodes["ImageScale"][1]["crop"] == "center"
    from comfy_story.film_staging_workflow import compile_opening_stage

    staging_fit: Any = compile_opening_stage("Opening", ("scene.png",), 7)["26"]
    for field in ("width", "height", "crop", "upscale_method"):
        assert staging_fit["inputs"][field] == nodes["ImageScale"][1][field]
    assert nodes["MiniMaxH3ImageToVideo"][1]["first_frame"] == [nodes["ImageScale"][0], 0]
    guides = [n for n in graph.values() if n["class_type"] == "MiniMaxH3AddGuide"]
    assert len(guides) == 1
    assert guides[0]["inputs"]["frame_idx"] == 119
    assert guides[0]["inputs"]["positive"] == [nodes["MiniMaxH3ImageToVideo"][0], 0]
    assert nodes["RandomNoise"][1]["noise_seed"] == 7
    if sampler is StorySampler.TURBO_4STEP:
        assert nodes["LoraLoaderModelOnly"][1]["lora_name"] == (
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
        )
        assert nodes["MiniMaxH3SigmaShift"][1]["shift_video"] == 6.0
        assert nodes["MiniMaxH3SigmaShift"][1]["shift_audio"] == 3.0
        assert nodes["BasicScheduler"][1]["steps"] == 4
    elif sampler is StorySampler.TURBO_8STEP:
        lora_id, lora = nodes["LoraLoaderModelOnly"]
        assert lora["lora_name"] == "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
        assert lora["strength_model"] == 1.0
        assert "MiniMaxH3SigmaShift" not in nodes
        assert nodes["KSamplerSelect"][1]["sampler_name"] == "res_multistep"
        assert nodes["BasicScheduler"][1] == {
            "model": [lora_id, 0],
            "scheduler": "simple",
            "steps": 8,
            "denoise": 1.0,
        }
        assert nodes["BasicGuider"][1]["model"] == [lora_id, 0]
    else:
        assert "LoraLoaderModelOnly" not in nodes
        assert nodes["BasicScheduler"][1]["steps"] == 20


def test_legacy_reference_only_start_recovers_and_changes_scene_without_world(
    continuity_integration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    # Keep legacy archive behavior covered without offering a production opt-out.
    monkeypatch.setattr(adapter, "configured_memory", lambda: None)
    monkeypatch.setenv("COMFY_STORY_MEMORY_RUNTIME", "native-reference")
    monkeypatch.setenv("COMFY_STORY_ROOT", str(tmp_path / "stories"))
    reference = tmp_path / "acorn.png"
    Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(reference)
    inputs = {
        "Create": "Start Story",
        "Story Library": json.dumps(
            {
                "project_name": "Acorn",
                "references": [
                    {"file": reference.name, "name": "Acorn", "role": "Prop", "note": "shell"}
                ],
            }
        ),
        "World / starting frame": "None",
        "Composition": "New composition",
        "What happens next?": "@Acorn rests on moss.",
        "Shot length": "5 seconds",
        "Variation": 321,
        "Reference policy": "Prompt mentions only",
    }
    prepared = adapter.prepare_node_generation(inputs)
    assert prepared.prepared.visual_roles == ("evidence-acorn",)
    assert "<Picture 2>" not in prepared.resolved_prompt
    assert "prior visual context" not in prepared.resolved_prompt
    assert prepared.prepared.request.world_frame is None
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"container fixture; media decoding tested separately")
    frames = torch.full((3, 8, 12, 3), 0.314159)
    committed = adapter.commit_node_generation(
        prepared=prepared.prepared, decoded_images=frames, saved_video=video
    )
    recovered = adapter.recover_node_generation(adapter.prepare_node_generation(inputs))
    assert recovered is not None
    assert recovered[0].state == committed.loaded_revision.state
    new_scene = adapter.prepare_node_generation(
        {
            **inputs,
            "Create": "New Scene",
            "Previous Story": recovered[0].state,
            "What happens next?": "@Acorn rests on a stone.",
            "Variation": 322,
        },
    )
    assert "current" not in new_scene.prepared.visual_roles
    assert new_scene.prepared.request.execution_sha256 != prepared.prepared.request.execution_sha256
    assert new_scene.prepared.parent.state == recovered[0].state
    with pytest.raises(ValueError, match="requires World / starting frame"):
        adapter.prepare_node_generation({**inputs, "Composition": "Continue frame"})


@pytest.mark.parametrize(
    "sampler",
    [
        StorySampler.NVFP4_EXACT,
        StorySampler.NVFP4_BALANCED,
        StorySampler.NVFP4_ULTRA_FAST,
        StorySampler.NVFP4_TURBO,
    ],
)
def test_nvfp4_keeps_native_references_and_frame_guide(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, sampler: StorySampler
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1), _image(0.2)),
        resolved_prompt="resolved prompt",
        frame_count=124,
        variation=123,
        sampler=sampler,
    )
    monkeypatch.setattr(integration_nodes, "prepare_node_generation", lambda inputs, **kw: prepared)
    output = continuity_integration.ComfyStory.execute()
    assert output.expand is not None
    nodes = {n["class_type"]: (key, n["inputs"]) for key, n in output.expand.items()}
    assert nodes["UNETLoader"][1]["unet_name"] == "minimax_h3_ref2va_pruned_nvfp4.safetensors"
    assert nodes["BasicScheduler"][1]["steps"] == (4 if sampler is StorySampler.NVFP4_TURBO else 27)
    assert nodes["BasicScheduler"][1]["scheduler"] == "simple"
    assert nodes["KSamplerSelect"][1]["sampler_name"] == "euler"
    assert ("LoraLoaderModelOnly" in nodes) == (sampler is StorySampler.NVFP4_TURBO)
    assert ("ComfyH3UltraFast" in nodes) == (sampler is StorySampler.NVFP4_ULTRA_FAST)
    assert ("ComfyH3BalancedCache" in nodes) == (sampler is StorySampler.NVFP4_BALANCED)
    assert nodes["MiniMaxH3AddGuide"][1]["frame_idx"] == 0
    refs = nodes["MiniMaxH3ReferenceToVideo"][1]
    assert refs["ref_images.ref_image_0"] is prepared.visual_guides[0]
    assert refs["ref_images.ref_image_1"] is prepared.visual_guides[1]
    assert "ComfyStoryCommit" in nodes


def test_balanced_rejects_allocator_compiler_before_sampling(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = ModuleType("comfy.cli_args")
    cli.args = SimpleNamespace(disable_comfy_compiler=False)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy.cli_args", cli)
    with pytest.raises(ValueError, match="--disable-comfy-compiler"):
        continuity_integration.ComfyH3BalancedCache.execute(object())


@pytest.mark.parametrize(
    "sampler",
    [
        StorySampler.NVFP4_EXACT,
        StorySampler.NVFP4_BALANCED,
        StorySampler.NVFP4_ULTRA_FAST,
        StorySampler.NVFP4_TURBO,
    ],
)
def test_nvfp4_render_configuration_rejects_animate_profile(
    continuity_integration: ModuleType, sampler: StorySampler
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    with pytest.raises(ValueError, match="requires Reference shot"):
        adapter.render_configuration("Animate frame", sampler)


@pytest.mark.parametrize("profile", ["Animate frame", "Reference shot"])
def test_automatic_h3_encoder_receives_ending_image(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    ending = _image(0.8)
    prepared = SimpleNamespace(
        prepared={},
        visual_guides=(_image(0.1),),
        resolved_prompt="action",
        frame_count=124,
        variation=7,
        sampler=StorySampler.TURBO_4STEP,
        render_profile=profile,
        output_duration_ms=5000,
        ending_frame=ending,
    )
    monkeypatch.setattr(integration_nodes, "prepare_node_generation", lambda *a, **kw: prepared)
    graph = continuity_integration.ComfyStory.execute(**{"Prompt format": "H3 automatic v1"}).expand
    nodes = {n["class_type"]: n["inputs"] for n in graph.values()}
    if profile == "Animate frame":
        assert nodes["MiniMaxH3ImageToVideo"]["last_frame"] is ending
        assert nodes["ComfyStoryKeyframeTiming"]["last_index"] == 119
        assert "MiniMaxH3AddGuide" not in nodes
    else:
        assert nodes["MiniMaxH3ReferenceToVideo"]["ref_images.ref_image_1"] is ending
        assert nodes["MiniMaxH3AddGuide"]["frame_idx"] == 119
    assert nodes["RandomNoise"]["noise_seed"] == 7


def test_h3_keyframe_timing_preserves_cached_conditioning(
    continuity_integration: ModuleType,
) -> None:
    latent = _image(0.5)
    original: Any = [
        [
            "embedding",
            {
                "minimax_keyframes": [
                    {"resolved_frame_index": 0, "latent": latent},
                    {"resolved_frame_index": 123, "latent": latent},
                ],
                "other": 7,
            },
        ]
    ]
    result = continuity_integration.ComfyStoryKeyframeTiming.execute(original, 119).args[0]
    assert original[0][1]["minimax_keyframes"][1]["resolved_frame_index"] == 123
    assert result[0][1]["minimax_keyframes"][1]["resolved_frame_index"] == 119
    assert result[0][1]["minimax_keyframes"][1]["latent"] is latent
    assert result[0][1]["other"] == 7
    with pytest.raises(ValueError, match="first/last"):
        continuity_integration.ComfyStoryKeyframeTiming.execute(original, 124)


def test_full_hd_rebuilds_conditioning_and_preserves_audio_path(
    continuity_integration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    quality = importlib.import_module(f"{_MODULE}.h3_quality_nodes")
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))
    monkeypatch.setattr(quality, "require_upscaler", lambda nodes: None)
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1), _image(0.2)),
        resolved_prompt="A fresh scene",
        frame_count=123,
        output_duration_ms=5000,
        variation=123,
        sampler=StorySampler.FULL_HD_2PASS,
        ending_frame=_image(0.3),
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )
    output = continuity_integration.ComfyStory.execute()
    assert output.expand is not None
    rows = list(output.expand.items())
    samples = [
        (key, row["inputs"]) for key, row in rows if row["class_type"] == "SamplerCustomAdvanced"
    ]
    assert len(samples) == 2
    conditioning = [
        row["inputs"] for _, row in rows if row["class_type"] == "MiniMaxH3ReferenceToVideo"
    ]
    assert [(row["width"], row["height"]) for row in conditioning] == [(960, 544), (1920, 1088)]
    assert conditioning[1]["ref_images.ref_image_2"] is prepared.ending_frame
    by_type = {row["class_type"]: (key, row["inputs"]) for key, row in rows}
    assert by_type["ComfyH3VideoLatent"][1]["latent"] == [samples[0][0], 0]
    assert by_type["ComfyH3ReplaceVideoLatent"][1]["original"] == [samples[0][0], 0]
    assert samples[1][1]["latent_image"] == [by_type["ComfyH3ReplaceVideoLatent"][0], 0]
    assert by_type["RandomNoise"][1]["noise_seed"] == 10123
    assert by_type["VAEDecodeTiled"][1]["tile_size"] == 256
    assert by_type["VAEDecodeTiled"][1]["overlap"] == 32
    assert by_type["MiniMaxH3AddGuide"][1]["frame_idx"] == 119


def test_full_hd_latent_replacement_keeps_original_audio_and_time(
    continuity_integration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module(f"{_MODULE}.h3_quality_nodes")

    class Nested:
        is_nested = True

        def __init__(self, tensors: tuple[torch.Tensor, ...]) -> None:
            self.tensors = tensors

    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", SimpleNamespace(NestedTensor=Nested))
    video = torch.zeros(1, 24, 2, 34, 60)
    audio = torch.randn(1, 32, 12, 16)
    original = {"samples": Nested((video, audio)), "identity": "preserved"}
    assert quality.ComfyH3VideoLatent.execute(original).args[0]["samples"] is video
    enlarged = torch.zeros(1, 24, 2, 68, 120)
    result = quality.ComfyH3ReplaceVideoLatent.execute(original, {"samples": enlarged}).args[0]
    assert result["samples"].tensors[1] is audio
    assert result["identity"] == "preserved"
    with pytest.raises(ValueError, match="temporal contract"):
        quality.ComfyH3ReplaceVideoLatent.execute(original, {"samples": enlarged[:, :, :1]})
    with pytest.raises(ValueError, match="requires"):
        quality.require_upscaler({})
    with pytest.raises(ValueError, match="pinned"):
        quality.require_upscaler({"MinimaxH3LatentUpscaler3D": quality.ComfyH3VideoLatent})


def test_full_hd_identity_binds_upscaler_weights(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    requested: list[tuple[str, str]] = []

    def digest(folder: str, name: str) -> str:
        requested.append((folder, name))
        return "d" * 64

    monkeypatch.setattr(adapter, "_generation_asset_digest", digest)
    config = adapter.render_configuration("Reference shot", StorySampler.FULL_HD_2PASS)
    assert adapter._generation_assets(config)["upscaler"] == "d" * 64
    assert ("latent_upscale_models", "minimax_h3_latent_upscaler_3d_fp16.safetensors") in requested


def test_associative_graph_passes_the_generation_vae_to_commit(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(f"{_MODULE}.nodes")
    prepared = SimpleNamespace(
        prepared=SimpleNamespace(request=SimpleNamespace(associative_memory=object())),
        visual_guides=(_image(0.1),),
        resolved_prompt="A shot",
        frame_count=124,
        variation=7,
        sampler=StorySampler.NATIVE_RES_MULTISTEP,
    )
    monkeypatch.setattr(module, "prepare_node_generation", lambda inputs: prepared)
    output = continuity_integration.ComfyStory.execute()
    assert output.expand is not None
    commit = next(row for row in output.expand.values() if row["class_type"] == "ComfyStoryCommit")
    vae = next(key for key, row in output.expand.items() if row["class_type"] == "VAELoader")
    assert commit["inputs"]["memory_vae"] == [vae, 0]
    schema = continuity_integration.ComfyStoryCommit.define_schema()
    field = next(field for field in schema.inputs if field.id == "memory_vae")
    assert field.options["optional"] is True


@pytest.mark.parametrize("mode", [None, "native"])
def test_public_generation_requires_memory_before_project_or_gpu_work(
    continuity_integration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str | None,
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    if mode is None:
        monkeypatch.delenv("COMFY_STORY_MEMORY", raising=False)
    else:
        monkeypatch.setenv("COMFY_STORY_MEMORY", mode)
    monkeypatch.delenv("COMFY_STORY_MEMORY_CHECKPOINT", raising=False)
    root = tmp_path / "must-not-be-created"
    monkeypatch.setenv("COMFY_STORY_ROOT", str(root))
    expected = "CHECKPOINT is required" if mode is None else "requires associative memory"
    with pytest.raises(ValueError, match=expected):
        adapter.prepare_node_generation({})
    assert not root.exists()
