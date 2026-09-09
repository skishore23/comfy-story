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
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from duet.duetx.h3_reference_compressors import H3ReferenceCompiler
from duet.duetx.h3_reference_contracts import (
    H3CompileReceipt,
    H3ReferenceBudget,
    H3ReferenceKind,
    H3ReferenceMethod,
    H3SelectedSource,
)
from duet.duetx.minimax_h3_training import (
    build_minimax_h3_training_modules,
    save_minimax_h3_runtime_checkpoint,
)
from duet.duetx.story_memory_backend import StoryMemoryBackend, StorySampler

_MODULE = "integrations.comfyui_duetx_continuity"


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
    schema = continuity_integration.DuetStory.define_schema()
    assert schema.node_id == "DuetStory"
    assert schema.display_name == "Comfy Story"
    assert [field.id for field in schema.inputs] == [
        "Previous Story",
        "Previous Frame",
        "Create",
        "Story Library",
        "Reference context",
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
        ("story_state", "DUET_STORY"),
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
    context = next(field for field in schema.inputs if field.id == "Reference context")
    assert context.options["options"] == ["Native", "Compiled preview"]
    assert context.options["default"] == "Native"


def test_extension_exposes_story_nodes_and_hidden_internal_nodes(
    continuity_integration: ModuleType,
) -> None:
    extension = asyncio.run(continuity_integration.comfy_entrypoint())
    registered = asyncio.run(extension.get_node_list())
    assert registered == [
        continuity_integration.DuetStory,
        continuity_integration.DuetStoryCanon,
        continuity_integration.DuetStoryCommit,
        continuity_integration.DuetStoryTrim,
        continuity_integration.DuetStoryKeyframeTiming,
        continuity_integration.DuetH3BalancedCache,
        continuity_integration.DuetH3UltraFast,
        continuity_integration.DuetH3CompileReferences,
        continuity_integration.DuetH3TelemetryStart,
        continuity_integration.DuetH3TelemetryEnd,
        continuity_integration.DuetH3BenchmarkReceipt,
        continuity_integration.DuetH3VideoLatent,
        continuity_integration.DuetH3ReplaceVideoLatent,
        continuity_integration.DuetH3RefinementSigmas,
    ]
    assert registered[0].define_schema().is_deprecated is False
    assert registered[1].define_schema().is_deprecated is True
    assert all(node.define_schema().category == "Comfy/Story" for node in registered[:2])
    assert all(node.define_schema().category == "_Duet/Internal" for node in registered[2:])


def test_living_canon_schema_is_a_deprecated_compatibility_alias(
    continuity_integration: ModuleType,
) -> None:
    schema = continuity_integration.DuetStoryCanon.define_schema()
    node = continuity_integration.DuetStoryCanon
    assert node.fingerprint_inputs() != node.fingerprint_inputs()

    assert schema.node_id == "DuetStoryCanon"
    assert schema.display_name == "Comfy Story (legacy workflow alias)"
    assert schema.is_deprecated is True
    assert "Memory backend" not in [field.id for field in schema.inputs]
    context = next(field for field in schema.inputs if field.id == "Reference context")
    assert context.options["options"] == ["Native", "Compiled preview"]
    assert any(field.id == "Motion reference" for field in schema.inputs)
    actions = next(field for field in schema.inputs if field.id == "Memory actions")
    assert actions.kind == "STRING"
    assert actions.options["advanced"] is True
    assert schema.hidden == [_Hidden.unique_id]
    assert [field.id for field in schema.outputs] == ["video", "last_frame", "story_state"]


@pytest.mark.parametrize("node_name", ["DuetStory", "DuetStoryCanon"])
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

    def prepare(inputs: dict[str, object], *, living_canon: bool = False) -> object:
        del inputs
        observed.append(living_canon)
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

    output = continuity_integration.DuetStoryCommit.execute(
        prepared="prepared",
        decoded_images="images",
        saved_video="video",
        filename_prefix="duet_story/shot",
        memory_vae="vae",
        owner_node_id="42",
    )

    assert output.args == ("state", "frame")
    assert output.ui == {"duet_story": [{"owner_node_id": "42", "revision": True}]}


def test_scientific_method_widget_exists_only_in_internal_test_mode(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    public_fields = {field.id for field in continuity_integration.DuetStory.define_schema().inputs}
    assert "Compiler experiment" not in public_fields
    assert "Story context experiment" not in public_fields

    monkeypatch.setenv("DUET_H3_COMPILER_INTERNAL_TEST", "1")
    for node in (continuity_integration.DuetStory, continuity_integration.DuetStoryCanon):
        schema = node.define_schema()
        experiment = next(field for field in schema.inputs if field.id == "Compiler experiment")
        story_context = next(
            field for field in schema.inputs if field.id == "Story context experiment"
        )

        assert experiment.options["advanced"] is True
        assert experiment.options["default"] == "duet_x"
        assert experiment.options["options"] == [
            "native_full",
            "native_trimmed",
            "gated",
            "resampler",
            "duet",
            "exceptions_only",
            "duet_x",
        ]
        assert story_context.options["advanced"] is True
        assert story_context.options["default"] == "core_plus_recall"
        assert story_context.options["options"] == ["core_plus_recall", "retrieval_only"]


def test_public_story_context_ignores_hidden_ablation_input(
    continuity_integration: ModuleType,
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")

    assert (
        adapter._story_context_experiment(
            {"Story context experiment": "retrieval_only"}, internal_test=False
        )
        == "core_plus_recall"
    )


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
    output = continuity_integration.DuetStory.execute(
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
        "DuetStoryCommit",
    ]
    commit = output.expand["16"]
    assert commit["inputs"]["decoded_images"] == ["12", 0]
    assert commit["inputs"]["saved_video"] == ["15", 0]
    assert commit["inputs"]["memory_vae"] == ["3", 0]
    assert commit["inputs"]["filename_prefix"] == "duet_story/shot"
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
        reference_compile=None,
        telemetry_enabled=False,
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )

    output = continuity_integration.DuetStory.execute()

    assert output.expand is not None
    types = [node["class_type"] for node in output.expand.values()]
    assert "DuetH3CompileReferences" not in types
    native_id = next(
        node_id
        for node_id, node in output.expand.items()
        if node["class_type"] == "MiniMaxH3ReferenceToVideo"
    )
    guide = next(
        node for node in output.expand.values() if node["class_type"] == "MiniMaxH3AddGuide"
    )
    assert guide["inputs"]["positive"] == [native_id, 0]


def test_compiled_preview_inserts_hidden_compiler_and_motion_reference(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    request = object()
    motion = torch.zeros((12, 8, 12, 3), dtype=torch.float32)
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1), _image(0.2)),
        resolved_prompt="resolved prompt",
        frame_count=124,
        variation=123,
        sampler=StorySampler.NATIVE_RES_MULTISTEP,
        motion_reference=motion,
        reference_compile=request,
        telemetry_enabled=False,
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )

    output = continuity_integration.DuetStory.execute()

    assert output.expand is not None
    native_id, native = next(
        (node_id, node)
        for node_id, node in output.expand.items()
        if node["class_type"] == "MiniMaxH3ReferenceToVideo"
    )
    compiler_id, compiler = next(
        (node_id, node)
        for node_id, node in output.expand.items()
        if node["class_type"] == "DuetH3CompileReferences"
    )
    guide = next(
        node for node in output.expand.values() if node["class_type"] == "MiniMaxH3AddGuide"
    )
    assert native["inputs"]["ref_videos.ref_video_0"] is motion
    assert compiler["inputs"] == {"positive": [native_id, 0], "request": request}
    assert guide["inputs"]["positive"] == [compiler_id, 0]


def test_internal_telemetry_wraps_only_the_denoising_region(
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
        reference_compile=None,
        telemetry_enabled=True,
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )

    output = continuity_integration.DuetStory.execute()

    assert output.expand is not None
    start_id = next(
        node_id
        for node_id, node in output.expand.items()
        if node["class_type"] == "DuetH3TelemetryStart"
    )
    sampler_id = next(
        node_id
        for node_id, node in output.expand.items()
        if node["class_type"] == "SamplerCustomAdvanced"
    )
    end_id, end = next(
        (node_id, node)
        for node_id, node in output.expand.items()
        if node["class_type"] == "DuetH3TelemetryEnd"
    )
    decode = next(node for node in output.expand.values() if node["class_type"] == "VAEDecode")
    assert end["inputs"]["latent"] == [sampler_id, 0]
    assert end["inputs"]["telemetry"] == [start_id, 1]
    assert decode["inputs"]["samples"] == [end_id, 0]


def test_internal_benchmark_graph_publishes_compiler_and_denoising_receipt(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration_nodes = importlib.import_module(f"{_MODULE}.nodes")
    request = object()
    prepared = SimpleNamespace(
        prepared={"opaque": "prepared"},
        visual_guides=(_image(0.1), _image(0.2)),
        resolved_prompt="resolved prompt",
        frame_count=124,
        variation=101,
        sampler=StorySampler.NATIVE_RES_MULTISTEP,
        motion_reference=None,
        reference_compile=request,
        telemetry_enabled=True,
        benchmark_phase="screening",
        benchmark_workload="performance_transfer",
        benchmark_method="duet_x",
        benchmark_cell="performance_transfer__duet_x__101",
    )
    monkeypatch.setattr(
        integration_nodes, "prepare_node_generation", lambda inputs, **kwargs: prepared
    )

    output = continuity_integration.DuetStory.execute()

    assert output.expand is not None
    compiler_id, compiler = next(
        (node_id, node)
        for node_id, node in output.expand.items()
        if node["class_type"] == "DuetH3CompileReferences"
    )
    telemetry_id = next(
        node_id
        for node_id, node in output.expand.items()
        if node["class_type"] == "DuetH3TelemetryEnd"
    )
    receipt_id, receipt = next(
        (node_id, node)
        for node_id, node in output.expand.items()
        if node["class_type"] == "DuetH3BenchmarkReceipt"
    )
    assert compiler["inputs"]["request"] is request
    assert receipt["inputs"]["request"] is request
    assert receipt["inputs"]["compile_receipt"] == [compiler_id, 1]
    assert receipt["inputs"]["context_build_ns"] == [compiler_id, 2]
    assert receipt["inputs"]["denoising"] == [telemetry_id, 1]
    assert receipt["inputs"]["cell_id"] == "performance_transfer__duet_x__101"
    assert output.args[2] == [receipt_id, 0]


def test_hidden_benchmark_receipt_is_canonical_and_root_scoped(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hidden = importlib.import_module(f"{_MODULE}.h3_reference_node")
    monkeypatch.setenv("DUET_H3_BENCHMARK_ROOT", str(tmp_path))
    compile_receipt = H3CompileReceipt(
        "duet-x-h3-reference-compile-receipt-v1",
        H3ReferenceMethod.NATIVE_FULL,
        2,
        2,
        32,
        32,
        ("1" * 64, "2" * 64),
        ("3" * 64, "4" * 64),
        None,
        "bypass",
        0,
        0,
    ).validate()
    denoising = hidden.DenoisingTelemetry(123_000_000, 40, 50)
    state = object()
    sources = (
        H3SelectedSource("story:current", H3ReferenceKind.IMAGE, 0, False),
        H3SelectedSource("story:evidence-Aya", H3ReferenceKind.IMAGE, 1, True),
    )
    request = hidden.PreparedReferenceCompile(
        sources,
        H3ReferenceBudget(64, 1),
        H3ReferenceCompiler(H3ReferenceMethod.NATIVE_FULL, H3ReferenceBudget(64, 1)),
    )

    result = hidden.DuetH3BenchmarkReceipt.execute(
        state,
        request,
        compile_receipt,
        12_000_000,
        denoising,
        "screening",
        "performance_transfer",
        "native_full",
        101,
        "performance_transfer__native_full__101",
    )

    receipt_path = tmp_path / "screening" / "performance_transfer__native_full__101.json"
    payload = json.loads(receipt_path.read_text())
    assert result.args == (state,)
    assert result.ui == {"duet_h3_benchmark": [payload]}
    assert payload["format"] == "duet-x-h3-runtime-receipt-v1"
    assert payload["compiler"]["input_visual_rows"] == 32
    assert payload["selection"] == {
        "protected_source_ids": ["story:evidence-Aya"],
        "sources": [
            {
                "kind": "image",
                "ordinal": 0,
                "protected": False,
                "source_id": "story:current",
            },
            {
                "kind": "image",
                "ordinal": 1,
                "protected": True,
                "source_id": "story:evidence-Aya",
            },
        ],
    }
    assert payload["telemetry"] == {
        "context_build_ns": 12_000_000,
        "denoising_elapsed_ns": 123_000_000,
        "max_memory_allocated": 40,
        "max_memory_reserved": 50,
    }
    assert receipt_path.read_bytes().endswith(b"\n")

    monkeypatch.delenv("DUET_H3_BENCHMARK_ROOT")
    history_only = hidden.DuetH3BenchmarkReceipt.execute(
        state,
        request,
        compile_receipt,
        12_000_000,
        denoising,
        "screening",
        "performance_transfer",
        "native_full",
        202,
        "performance_transfer__native_full__202",
    )
    assert history_only.ui is not None
    assert history_only.ui["duet_h3_benchmark"][0]["cell"]["seed"] == 202
    assert not (tmp_path / "screening" / "performance_transfer__native_full__202.json").exists()

    with pytest.raises(ValueError, match="cell id"):
        hidden.DuetH3BenchmarkReceipt.execute(
            state,
            request,
            compile_receipt,
            1,
            denoising,
            "screening",
            "performance_transfer",
            "native_full",
            101,
            "../escape",
        )


def test_hidden_compiler_rewrites_only_visual_reference_blocks(
    continuity_integration: ModuleType,
) -> None:
    hidden = importlib.import_module(f"{_MODULE}.h3_reference_node")
    sources = (
        H3SelectedSource("world", H3ReferenceKind.IMAGE, 0, False),
        H3SelectedSource("prop", H3ReferenceKind.IMAGE, 1, False),
        H3SelectedSource("hero", H3ReferenceKind.IMAGE, 2, True),
    )
    positive = [
        [
            torch.zeros(1, 4, 8),
            {
                "minimax_token_tags": torch.ones(1, dtype=torch.long),
                "minimax_refs": [
                    {
                        "kind": "image",
                        "latent_h": 4,
                        "latent_w": 6,
                        "latent": torch.full((1, 24, 1, 4, 6), float(index)),
                    }
                    for index in range(3)
                ],
            },
        ]
    ]
    budget = H3ReferenceBudget(12, 1)
    compiler = H3ReferenceCompiler(
        H3ReferenceMethod.DUET_X,
        budget,
        checkpoint_sha256="a" * 64,
    )
    request = hidden.PreparedReferenceCompile(sources, budget, compiler)

    output = hidden.DuetH3CompileReferences.execute(positive, request)

    conditioned, receipt, context_build_ns = output.args
    original_metadata = cast(dict[str, object], positive[0][1])
    assert conditioned is not positive
    assert conditioned[0][0] is positive[0][0]
    assert len(conditioned[0][1]["minimax_refs"]) == 2
    assert len(cast(list[object], original_metadata["minimax_refs"])) == 3
    assert receipt.input_visual_rows == 18
    assert receipt.output_visual_rows == 12
    assert type(context_build_ns) is int
    assert context_build_ns > 0


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
    output = continuity_integration.DuetStory.execute()
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
    assert "DuetStoryCommit" in nodes


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

    output = continuity_integration.DuetStory.execute()

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
        continuity_integration.DuetStory.execute()


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
        continuity_integration.DuetStory.execute()


def test_comfy_minimax_codec_reuses_native_vae_tensor_contract(
    continuity_integration: ModuleType,
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")

    class _VAE:
        def encode(self, images: torch.Tensor) -> torch.Tensor:
            assert images.shape == (1, 384, 384, 3)
            assert images.dtype == torch.float32
            return torch.full((1, 24, 1, 24, 24), 0.25, dtype=torch.float16)

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            assert latent.shape == (1, 24, 1, 24, 24)
            return torch.full((1, 1, 384, 384, 3), 0.5, dtype=torch.float32)

    codec = adapter.ComfyMiniMaxH3Codec(_VAE())
    frame = np.zeros((384, 384, 3), dtype=np.uint8)

    latent = codec.encode_frame(frame)
    decoded = codec.decode_frame(latent)

    assert latent.shape == (1, 24, 1, 24, 24)
    assert latent.dtype == torch.float16
    assert decoded.shape == (384, 384, 3)
    assert decoded.dtype == np.uint8
    assert np.all(decoded == 128)


@pytest.mark.parametrize("living_canon", [False, True])
def test_prepare_defaults_to_authenticated_minimax_memory(
    continuity_integration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    living_canon: bool,
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    reference = tmp_path / "maya.png"
    Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(reference)
    checkpoint = tmp_path / "minimax-memory.pt"
    foundation_sha256 = "1" * 64
    vae_sha256 = "3" * 64
    save_minimax_h3_runtime_checkpoint(
        checkpoint,
        build_minimax_h3_training_modules(),
        foundation_sha256=foundation_sha256,
        model_configuration_sha256=adapter.MINIMAX_MODEL_CONFIGURATION_SHA256,
        optimizer_steps=2_000,
        training_trace=({"loss": 0.25, "optimizer_step": 2_000},),
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setenv("DUET_STORY_ROOT", str(tmp_path / "stories"))
    monkeypatch.setenv("DUET_STORY_MINIMAX_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("DUET_STORY_MINIMAX_CHECKPOINT_SHA256", checkpoint_sha256)
    monkeypatch.setenv("DUET_STORY_MINIMAX_FOUNDATION_SHA256", foundation_sha256)
    monkeypatch.setenv("DUET_STORY_MINIMAX_VAE_SHA256", vae_sha256)

    inputs = {
        "Previous Story": None,
        "Previous Frame": None,
        "Create": "Start Story",
        "Story Library": json.dumps(
            {
                "project_name": "MiniMax Story",
                "references": [
                    {
                        "file": reference.name,
                        "name": "Maya",
                        "note": "lead",
                        "role": "Character",
                    }
                ],
            }
        ),
        "World / starting frame": _image(0.1),
        "What happens next?": "@Maya enters.",
        "Shot length": "5 seconds",
        "Variation": 123,
        "Story revision": "",
        "Reference policy": "Automatic",
    }
    result = adapter.prepare_node_generation(inputs, living_canon=living_canon)

    request = result.prepared.request
    assert request.memory_backend is StoryMemoryBackend.MINIMAX_H3
    assert request.include_associative_core is True
    assert request.separate_evidence_images is True
    assert request.sampler is StorySampler.NATIVE_RES_MULTISTEP
    assert request.checkpoint_sha256 == checkpoint_sha256
    assert request.model_configuration_sha256 == adapter.MINIMAX_MODEL_CONFIGURATION_SHA256
    if living_canon:
        animated = adapter.prepare_node_generation(
            {**inputs, "Render profile": "Animate frame"}, living_canon=True
        )
        assert animated.render_profile == "Animate frame"
        assert animated.prepared.active_reference_names == ()
        assert animated.prepared.visual_roles == ("current",)
        assert animated.prepared.request.execution_sha256 != request.execution_sha256
        folder_paths = importlib.import_module("folder_paths")
        original_path = folder_paths.get_full_path
        with monkeypatch.context() as patch:
            patch.setattr(
                folder_paths,
                "get_full_path",
                lambda folder, name: None if "h3_fl2va" in name else original_path(folder, name),
            )
            with pytest.raises(ValueError, match="requires the installed MiniMax H3 model"):
                adapter.prepare_node_generation(
                    {**inputs, "Render profile": "Animate frame"}, living_canon=True
                )
        for unsupported in (
            {"Composition": "New composition"},
            {"Sampler": "SPEED Euler 2-stage"},
            {"Reference context": "Compiled preview"},
            {"Motion reference": _image(0.5)},
        ):
            with pytest.raises(ValueError, match="Animate frame"):
                adapter.prepare_node_generation(
                    {**inputs, "Render profile": "Animate frame", **unsupported}, living_canon=True
                )
        eight_inputs = {**inputs, "Render profile": "Animate frame", "Sampler": "Turbo 8-step"}
        eight = adapter.prepare_node_generation(eight_inputs, living_canon=True)
        assert eight.prepared.request.sampler is StorySampler.TURBO_8STEP
        reference_eight = adapter.prepare_node_generation(
            {**inputs, "Sampler": "Turbo 8-step"}, living_canon=True
        )
        assert reference_eight.render_profile == "Reference shot"
        assert reference_eight.prepared.request.sampler is StorySampler.TURBO_8STEP
        assert reference_eight.prepared.request.execution_sha256 not in {
            request.execution_sha256,
            eight.prepared.request.execution_sha256,
        }
        with monkeypatch.context() as eight_assets:
            original_eight_path = adapter.folder_paths.get_full_path
            eight_file = tmp_path / "eight-step-lora.bin"
            eight_file.write_bytes(b"changed eight-step weights")
            eight_assets.setattr(
                adapter.folder_paths,
                "get_full_path",
                lambda folder, name: (
                    str(eight_file) if folder == "loras" else original_eight_path(folder, name)
                ),
            )
            changed_eight = adapter.prepare_node_generation(eight_inputs, living_canon=True)
            assert (
                changed_eight.prepared.request.execution_sha256
                != eight.prepared.request.execution_sha256
            )
            eight_assets.setattr(
                adapter.folder_paths,
                "get_full_path",
                lambda folder, name: (
                    None if folder == "loras" else original_eight_path(folder, name)
                ),
            )
            with pytest.raises(ValueError, match="requires the installed MiniMax H3 model"):
                adapter.prepare_node_generation(eight_inputs, living_canon=True)
            with pytest.raises(ValueError, match="requires the installed MiniMax H3 model"):
                adapter.prepare_node_generation(
                    {**inputs, "Sampler": "Turbo 8-step"}, living_canon=True
                )
        declared = adapter.prepare_node_generation(
            {**inputs, "Scene entities": '["Maya"]'}, living_canon=True
        )
        assert declared.prepared.request.scene_entity_names == ("Maya",)
        assert (
            declared.prepared.request.execution_sha256 != result.prepared.request.execution_sha256
        )
        empty_cast = adapter.prepare_node_generation(
            {**inputs, "Scene entities": "[]"}, living_canon=True
        )
        assert (
            empty_cast.prepared.request.execution_sha256
            != declared.prepared.request.execution_sha256
        )
        for malformed in ("{}", "[1]", "bad json", '[""]'):
            with pytest.raises(ValueError, match="Scene entities"):
                adapter.prepare_node_generation(
                    {**inputs, "Scene entities": malformed}, living_canon=True
                )
        exact = adapter.prepare_node_generation(
            {**inputs, "Output duration (ms)": 5000}, living_canon=True
        )
        assert exact.output_duration_ms == 5000
        assert exact.prepared.request.execution_sha256 != result.prepared.request.execution_sha256
        native = adapter.prepare_node_generation(
            {**inputs, "Output duration (ms)": 0}, living_canon=True
        )
        assert native.prepared.request.execution_sha256 == result.prepared.request.execution_sha256
        ending = adapter.prepare_node_generation(
            {**inputs, "Ending frame": _image(0.5)}, living_canon=True
        )
        assert ending.prepared.request.execution_sha256 != result.prepared.request.execution_sha256
        changed_ending = adapter.prepare_node_generation(
            {**inputs, "Ending frame": _image(0.6)}, living_canon=True
        )
        assert (
            changed_ending.prepared.request.execution_sha256
            != ending.prepared.request.execution_sha256
        )
        with pytest.raises(ValueError, match="exactly one image"):
            adapter.prepare_node_generation(
                {**inputs, "Ending frame": _image(0.5).repeat(2, 1, 1, 1)}, living_canon=True
            )
        for invalid_duration in (True, -1, 5001, 6000):
            with pytest.raises(ValueError, match="Output duration"):
                adapter.prepare_node_generation(
                    {**inputs, "Output duration (ms)": invalid_duration}, living_canon=True
                )

        structured = adapter.prepare_node_generation(
            {**inputs, "Prompt format": "Structured reference (experimental)"}, living_canon=True
        )
        assert structured.resolved_prompt.startswith("subject_definitions:")
        assert "<Subject 1> enters." in structured.resolved_prompt
        assert structured.prepared.request.execution_sha256 != request.execution_sha256
        explicit_current = adapter.prepare_node_generation(
            {**inputs, "Prompt format": "Current"}, living_canon=True
        )
        assert explicit_current.prepared.request.execution_sha256 == request.execution_sha256
        assert len(structured.visual_guides) == len(result.visual_guides)
        assert all(
            torch.equal(a, b)
            for a, b in zip(structured.visual_guides, result.visual_guides, strict=True)
        )
        turbo_inputs = {**inputs, "Sampler": "Turbo 4-step"}
        turbo = adapter.prepare_node_generation(turbo_inputs, living_canon=living_canon)
        assert turbo.prepared.request.sampler is StorySampler.TURBO_4STEP
        assert (
            turbo.prepared.request.model_configuration_sha256 == request.model_configuration_sha256
        )
        assert turbo.prepared.request.execution_sha256 != request.execution_sha256
        for mode in ("NVFP4 Exact", "NVFP4 Balanced", "NVFP4 Ultra Fast", "NVFP4 Turbo 4-step"):
            accelerated = adapter.prepare_node_generation(
                {**inputs, "Sampler": mode}, living_canon=True
            )
            assert (
                accelerated.prepared.request.model_configuration_sha256
                == request.model_configuration_sha256
            )
            assert accelerated.prepared.request.checkpoint_sha256 == request.checkpoint_sha256
            assert accelerated.prepared.request.execution_sha256 != request.execution_sha256
            assert (
                accelerated.prepared.request.execution_sha256
                != turbo.prepared.request.execution_sha256
            )

        with monkeypatch.context() as assets:
            original_path = adapter.folder_paths.get_full_path
            lora_file = tmp_path / "turbo-lora.bin"
            lora_file.write_bytes(b"alternate turbo weights")
            assets.setattr(
                adapter.folder_paths,
                "get_full_path",
                lambda folder, name: (
                    str(lora_file) if folder == "loras" else original_path(folder, name)
                ),
            )
            changed_lora = adapter.prepare_node_generation(turbo_inputs, living_canon=living_canon)
            assert (
                changed_lora.prepared.request.execution_sha256
                != turbo.prepared.request.execution_sha256
            )
            unchanged_native = adapter.prepare_node_generation(inputs, living_canon=living_canon)
            assert unchanged_native.prepared.request.execution_sha256 == request.execution_sha256
            assets.setattr(
                adapter.folder_paths,
                "get_full_path",
                lambda folder, name: None if folder == "loras" else original_path(folder, name),
            )
            with pytest.raises(ValueError, match=r"requires the installed MiniMax H3 model.*turbo"):
                adapter.prepare_node_generation(turbo_inputs, living_canon=living_canon)
    assert result.prepared.visual_roles == (
        ("current", "evidence-maya")
        if living_canon
        else ("current-or-starting-frame", "exact-reference:Maya")
    )
    if living_canon:

        class MemoryVAE:
            def encode(self, images: torch.Tensor) -> torch.Tensor:
                return torch.full((1, 24, 1, 24, 24), float(images.mean()))

            def decode(self, latent: torch.Tensor) -> torch.Tensor:
                return torch.full((1, 1, 384, 384, 3), 0.5)

        video = tmp_path / "shot.mp4"
        video.write_bytes(b"fixture video; real container checked on Forge1")
        images = torch.full((2, 8, 12, 3), 0.314159)
        committed = adapter.commit_node_generation(
            prepared=result.prepared,
            decoded_images=images,
            saved_video=video,
            memory_vae=MemoryVAE(),
        )
        restarted = adapter.prepare_node_generation(inputs, living_canon=True)
        recovered = adapter.recover_node_generation(restarted)
        assert recovered is not None
        loaded, saved, last = recovered
        assert loaded.state == committed.state
        summary = adapter.story_inspector_summary(loaded, owner_node_id="10")
        assert summary["parent_revision_sha256"] == loaded.state.parent_revision_sha256
        assert summary["revision_sha256"] == loaded.state.revision_sha256
        assert saved.read_bytes() == video.read_bytes()
        torch.testing.assert_close(last, committed.last_frame, rtol=0, atol=0)
        assert (
            loaded.shot_metadata["execution_sha256"] == restarted.prepared.request.execution_sha256
        )
        changed = adapter.prepare_node_generation({**inputs, "Variation": 124}, living_canon=True)
        assert adapter.recover_node_generation(changed) is None
        cut = adapter.prepare_node_generation(
            {**inputs, "Composition": "New composition"}, living_canon=True
        )
        assert cut.prepared.request.execution_sha256 != result.prepared.request.execution_sha256
        assert adapter.recover_node_generation(cut) is None
        with_audio = adapter.prepare_node_generation(
            {
                **inputs,
                "Authored audio": {"waveform": torch.zeros(1, 1, 8_000), "sample_rate": 8_000},
            },
            living_canon=True,
        )
        assert with_audio.authored_audio is not None
        assert (
            with_audio.prepared.request.execution_sha256 != result.prepared.request.execution_sha256
        )
        assert adapter.recover_node_generation(with_audio) is None
        next_inputs = {
            **inputs,
            "Create": "New Scene",
            "Previous Story": committed.state,
            "Composition": "New composition",
        }
        new_scene = adapter.prepare_node_generation(next_inputs, living_canon=True)
        assert new_scene.prepared.request.include_associative_core is False
        assert new_scene.prepared.request.previous_story == committed.state
        assert "inclusive-core" not in new_scene.prepared.visual_roles
        assert "evidence-maya" in new_scene.prepared.visual_roles
        continued = adapter.prepare_node_generation(
            {**next_inputs, "Composition": "Continue frame"}, living_canon=True
        )
        assert "inclusive-core" in continued.prepared.visual_roles
        with monkeypatch.context() as experiment:
            experiment.setenv("DUET_H3_COMPILER_INTERNAL_TEST", "1")
            experimental = adapter.prepare_node_generation(next_inputs, living_canon=True)
            assert "inclusive-core" in experimental.prepared.visual_roles
            assert experimental.prepared.request.separate_evidence_images is False
        assert (
            new_scene.prepared.request.execution_sha256
            != continued.prepared.request.execution_sha256
        )
        monkeypatch.setattr(
            importlib.import_module(f"{_MODULE}.nodes"),
            "GraphBuilder",
            lambda: pytest.fail("replay must not construct a GPU graph"),
        )
        latest = sys.modules["comfy_api.latest"]
        monkeypatch.setattr(
            latest, "InputImpl", SimpleNamespace(VideoFromFile=lambda path: path), raising=False
        )
        node = continuity_integration.DuetStory
        # Comfy must enter the durable gateway again even for an identical queued graph.
        assert node.fingerprint_inputs(**inputs) != node.fingerprint_inputs(**inputs)
        json.dumps(node.fingerprint_inputs(**inputs), allow_nan=False)
        for owner in ("10", "20"):
            replay = node.execute(**inputs, __owner_node_id=owner)
            assert replay.expand is None
            assert replay.args[2] == committed.state
            torch.testing.assert_close(replay.args[1], committed.last_frame, rtol=0, atol=0)
            assert replay.ui["duet_story"][0]["owner_node_id"] == owner
            assert replay.ui["duet_story"][0]["revision_sha256"] == committed.state.revision_sha256


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
        continuity_integration.DuetStory.execute()


def test_saved_video_path_recovers_native_savevideo_destination(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    output = tmp_path / "output" / "duet_story"
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
            "duet_story",
            prefix,
        ),
        raising=False,
    )
    video = SimpleNamespace(get_dimensions=lambda: (1344, 768))

    assert adapter._saved_video_path(video, "duet_story/shot") == saved


def test_public_native_archive_commit_recall_and_recovery_without_checkpoint(
    continuity_integration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from duet.duetx.story_contracts import canonical_story_json
    from duet.duetx.story_native_archive import NativeArchiveRevision
    from duet.duetx.story_product_contracts import (
        CanonPresence,
        MemoryAction,
        ObservationKind,
        StoryMemoryCommand,
    )
    from duet.duetx.story_store import StoryProjectStore

    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    monkeypatch.delenv("DUET_STORY_MEMORY_RUNTIME", raising=False)
    monkeypatch.delenv("DUET_STORY_MINIMAX_CHECKPOINT", raising=False)
    monkeypatch.delenv("DUET_H3_COMPILER_INTERNAL_TEST", raising=False)
    monkeypatch.setenv("DUET_STORY_ROOT", str(tmp_path / "stories"))

    def forbidden_checkpoint() -> None:
        pytest.fail("Public native archive must not inspect or load a trained checkpoint")

    monkeypatch.setattr(adapter, "_minimax_runtime_settings", forbidden_checkpoint)
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
    prepared = adapter.prepare_node_generation(inputs, living_canon=True)
    assert prepared.prepared.request.checkpoint_sha256 is None
    assert prepared.prepared.request.native_reference_archive
    assert not prepared.prepared.request.include_associative_core
    connected = adapter.prepare_node_generation(
        {**inputs, "World / starting frame": "missing-unused.png", "Starting image": _image(0.1)},
        living_canon=True,
    )
    # An upstream image takes precedence over an unused filename and binds the
    # same execution as the identical historical starting pixels.
    torch.testing.assert_close(connected.visual_guides[0], _image(0.1), rtol=0, atol=0)
    assert connected.prepared.request.execution_sha256 == prepared.prepared.request.execution_sha256
    changed_start = adapter.prepare_node_generation(
        {**inputs, "Starting image": _image(0.2)}, living_canon=True
    )
    assert (
        changed_start.prepared.request.execution_sha256
        != prepared.prepared.request.execution_sha256
    )
    with pytest.raises(ValueError, match="Starting image must contain exactly one image"):
        adapter.prepare_node_generation(
            {**inputs, "Starting image": torch.ones((2, 16, 16, 3))}, living_canon=True
        )
    store = StoryProjectStore(tmp_path / "stories")
    binding = json.loads(store.load_asset(prepared.prepared.request.execution_sha256))
    assert "memory_runtime" in binding
    assert "memory_checkpoint" not in binding
    assert binding["native_capture_policy"] == adapter.NATIVE_RGB_CAPTURE_POLICY_SHA256
    automatic = adapter.prepare_node_generation(
        {**inputs, "Prompt format": "H3 automatic v1"}, living_canon=True
    )
    new_binding = json.loads(store.load_asset(automatic.prepared.request.execution_sha256))
    assert new_binding["prompt_compiler"] == "duet-h3-mode-aware-v1"
    assert "prompt_compiler" not in binding
    assert new_binding["variation"] == binding["variation"]
    assert new_binding["guides"] == binding["guides"]
    assert "subject_definitions:" in new_binding["prompt"]
    assert automatic.prepared.request.execution_sha256 != prepared.prepared.request.execution_sha256
    legacy_again = adapter.prepare_node_generation(inputs, living_canon=True)
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
    restarted = adapter.prepare_node_generation(inputs, living_canon=True)
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
            living_canon=True,
        )
        torch.testing.assert_close(staged.visual_guides[0], expected_frame, rtol=0, atol=0)
        assert staged.prepared.request.previous_story == loaded.state
    with monkeypatch.context() as policy:
        policy.setattr(adapter, "NATIVE_RGB_CAPTURE_POLICY_SHA256", "0" * 64)
        old_capture = adapter.prepare_node_generation(inputs, living_canon=True)
        assert adapter.recover_node_generation(old_capture) is None
    changed = adapter.prepare_node_generation({**inputs, "Variation": 322}, living_canon=True)
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
    one_shot = adapter.prepare_node_generation(one_shot_inputs, living_canon=True)
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
    one_recovery = adapter.recover_node_generation(
        adapter.prepare_node_generation(one_shot_inputs, living_canon=True)
    )
    assert one_recovery is not None
    assert one_recovery[0].state == one_commit.state
    no_evidence = adapter.prepare_node_generation(
        {**one_shot_inputs, "Shot state evidence": "{}"}, living_canon=True
    )
    assert (
        no_evidence.prepared.request.execution_sha256 != one_shot.prepared.request.execution_sha256
    )
    assert adapter.recover_node_generation(no_evidence) is None
    next_shot = adapter.prepare_node_generation(continuation, living_canon=True)
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


def test_native_runtime_selection_is_explicit_and_preserves_trained_installations(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    monkeypatch.delenv("DUET_STORY_MEMORY_RUNTIME", raising=False)
    monkeypatch.setenv("DUET_STORY_MINIMAX_CHECKPOINT", "/configured/checkpoint")
    assert not adapter._native_reference_runtime()
    monkeypatch.setenv("DUET_STORY_MEMORY_RUNTIME", "native-reference")
    assert adapter._native_reference_runtime()
    monkeypatch.setenv("DUET_STORY_MEMORY_RUNTIME", "unknown")
    with pytest.raises(ValueError, match="DUET_STORY_MEMORY_RUNTIME"):
        adapter._native_reference_runtime()


def test_continuity_readme_documents_backend_and_sampler_contracts() -> None:
    text = Path("integrations/comfyui_duetx_continuity/README.md").read_text()
    for required in (
        "DUET_STORY_MINIMAX_CHECKPOINT",
        "DUET_STORY_MINIMAX_CHECKPOINT_SHA256",
        "DUET_STORY_MINIMAX_FOUNDATION_SHA256",
        "DUET_STORY_MINIMAX_VAE_SHA256",
        "duet-x-minimax-h3-v1",
        "Native res_multistep",
        "SPEED Euler 2-stage",
        "Sigma Harvest",
        "PolyForm Noncommercial 1.0.0",
        "no training checkpoint",
        "```mermaid",
    ):
        assert required in text


def test_story_media_routes_reject_wrong_assets_and_expose_range_video(
    continuity_integration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asyncio

    from aiohttp import web

    from duet.duetx.story_store import StoryProjectStore

    routes = importlib.import_module(f"{_MODULE}.routes")
    monkeypatch.setenv("DUET_STORY_ROOT", str(tmp_path / "story"))
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
    from duet.duetx.story_language import prepare_authored_audio

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
    result = continuity_integration.DuetStory.execute(Composition="New composition")
    graph = result.expand
    assert graph is not None
    assert "MiniMaxH3AddGuide" not in {node["class_type"] for node in graph.values()}
    generator = next(
        node for node in graph.values() if node["class_type"] == "MiniMaxH3ReferenceToVideo"
    )
    assert "ref_images.ref_image_0" in generator["inputs"]
    assert "ref_images.ref_image_1" in generator["inputs"]
    assert any(node["class_type"] == "DuetStoryCommit" for node in graph.values())
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
    monkeypatch.delenv("DUET_STORY_ROOT", raising=False)
    assert adapter._story_root() == tmp_path / "user" / "duet_story"
    assert not (tmp_path / "user").exists(), "Resolving configuration must not create storage"
    monkeypatch.setenv("DUET_STORY_ROOT", str(tmp_path / "custom"))
    assert adapter._story_root() == tmp_path / "custom"


@pytest.mark.parametrize("value", ["", " ", "relative/stories"])
def test_invalid_explicit_storage_does_not_fall_back_to_another_project(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    monkeypatch.setenv("DUET_STORY_ROOT", value)
    with pytest.raises(ValueError, match="DUET_STORY_ROOT"):
        adapter._story_root()


def test_legacy_checkpoint_has_no_developer_machine_fallback(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    monkeypatch.delenv("DUET_STORY_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="DUET_STORY_CHECKPOINT"):
        adapter._checkpoint()
    checkpoint = tmp_path / "configured.pt"
    monkeypatch.setenv("DUET_STORY_CHECKPOINT", str(checkpoint))
    with pytest.raises(ValueError, match="existing local checkpoint"):
        adapter._checkpoint()
    checkpoint.write_bytes(b"fixture; content authentication occurs in the pinned loader")
    assert adapter._checkpoint() == checkpoint


@pytest.mark.parametrize("commit", [None, "", "0" * 40, "not-a-commit"])
def test_legacy_source_provenance_cannot_use_a_placeholder(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, commit: str | None
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    monkeypatch.delenv("DUET_STORY_SOURCE_COMMIT", raising=False)
    if commit is not None:
        monkeypatch.setenv("DUET_STORY_SOURCE_COMMIT", commit)
    with pytest.raises(ValueError, match="actual deployed source revision"):
        adapter._ltx_source_identity()


def test_legacy_source_archive_requires_an_explicit_nonzero_identity(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    monkeypatch.setenv("DUET_STORY_SOURCE_COMMIT", "a" * 40)
    monkeypatch.delenv("DUET_STORY_SOURCE_ARCHIVE_SHA256", raising=False)
    with pytest.raises(ValueError, match="DUET_STORY_SOURCE_ARCHIVE_SHA256"):
        adapter._ltx_source_identity()
    monkeypatch.setenv("DUET_STORY_SOURCE_ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="actual source archive"):
        adapter._ltx_source_identity()
    monkeypatch.setenv("DUET_STORY_SOURCE_ARCHIVE_SHA256", "b" * 64)
    assert adapter._ltx_source_identity() == ("a" * 40, "b" * 64)


def test_legacy_configuration_fails_before_any_reference_or_gpu_work(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    monkeypatch.setenv("DUET_STORY_ROOT", str(tmp_path / "stories"))
    monkeypatch.delenv("DUET_STORY_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="DUET_STORY_CHECKPOINT"):
        adapter.prepare_node_generation({"Memory backend": "LTX"})


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
    output = continuity_integration.DuetStory.execute()
    graph = output.expand
    trim_id, trim = next(
        (key, node) for key, node in graph.items() if node["class_type"] == "DuetStoryTrim"
    )
    assert trim["inputs"]["duration_ms"] == duration
    video = next(node for node in graph.values() if node["class_type"] == "CreateVideo")
    commit = next(node for node in graph.values() if node["class_type"] == "DuetStoryCommit")
    assert video["inputs"]["images"] == commit["inputs"]["decoded_images"] == [trim_id, 0]
    assert video["inputs"]["audio"] == [trim_id, 1]


def test_starting_image_choices_include_portable_subfolders_and_exclude_symlinks(
    continuity_integration: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "input"
    nested = root / "duet_story_inputs"
    nested.mkdir(parents=True)
    (nested / "scene.png").write_bytes(b"image")
    (root / "root.jpg").write_bytes(b"image")
    (nested / "notes.txt").write_text("not an image")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (root / "linked.png").symlink_to(outside)
    folder_paths = importlib.import_module("folder_paths")
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(root))
    schema = continuity_integration.DuetStory.define_schema()
    world = next(field for field in schema.inputs if field.id == "World / starting frame")
    assert world.options["options"] == ["None", "duet_story_inputs/scene.png", "root.jpg"]


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
    graph = continuity_integration.DuetStory.execute().expand
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
    graph = continuity_integration.DuetStory.execute().expand
    nodes = {n["class_type"]: (key, n["inputs"]) for key, n in graph.items()}
    assert nodes["UNETLoader"][1]["unet_name"] == "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    assert "MiniMaxH3ReferenceToVideo" not in nodes
    assert nodes["ImageScale"][1]["image"] is prepared.visual_guides[0]
    assert nodes["ImageScale"][1]["crop"] == "center"
    from duet.duetx.film_staging_workflow import compile_opening_stage

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


def test_reference_only_public_start_recovers_and_changes_scene_without_world(
    continuity_integration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = importlib.import_module(f"{_MODULE}.comfy_adapter")
    monkeypatch.setenv("DUET_STORY_MEMORY_RUNTIME", "native-reference")
    monkeypatch.setenv("DUET_STORY_ROOT", str(tmp_path / "stories"))
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
    prepared = adapter.prepare_node_generation(inputs, living_canon=True)
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
    recovered = adapter.recover_node_generation(
        adapter.prepare_node_generation(inputs, living_canon=True)
    )
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
        living_canon=True,
    )
    assert "current" not in new_scene.prepared.visual_roles
    assert new_scene.prepared.request.execution_sha256 != prepared.prepared.request.execution_sha256
    assert new_scene.prepared.parent.state == recovered[0].state
    with pytest.raises(ValueError, match="requires World / starting frame"):
        adapter.prepare_node_generation(
            {**inputs, "Composition": "Continue frame"}, living_canon=True
        )


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
    output = continuity_integration.DuetStory.execute()
    assert output.expand is not None
    nodes = {n["class_type"]: (key, n["inputs"]) for key, n in output.expand.items()}
    assert nodes["UNETLoader"][1]["unet_name"] == "minimax_h3_ref2va_pruned_nvfp4.safetensors"
    assert nodes["BasicScheduler"][1]["steps"] == (4 if sampler is StorySampler.NVFP4_TURBO else 27)
    assert nodes["BasicScheduler"][1]["scheduler"] == "simple"
    assert nodes["KSamplerSelect"][1]["sampler_name"] == "euler"
    assert ("LoraLoaderModelOnly" in nodes) == (sampler is StorySampler.NVFP4_TURBO)
    assert ("DuetH3UltraFast" in nodes) == (sampler is StorySampler.NVFP4_ULTRA_FAST)
    assert ("DuetH3BalancedCache" in nodes) == (sampler is StorySampler.NVFP4_BALANCED)
    assert nodes["MiniMaxH3AddGuide"][1]["frame_idx"] == 0
    refs = nodes["MiniMaxH3ReferenceToVideo"][1]
    assert refs["ref_images.ref_image_0"] is prepared.visual_guides[0]
    assert refs["ref_images.ref_image_1"] is prepared.visual_guides[1]
    assert "DuetStoryCommit" in nodes


def test_balanced_rejects_allocator_compiler_before_sampling(
    continuity_integration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = ModuleType("comfy.cli_args")
    cli.args = SimpleNamespace(disable_comfy_compiler=False)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy.cli_args", cli)
    with pytest.raises(ValueError, match="--disable-comfy-compiler"):
        continuity_integration.DuetH3BalancedCache.execute(object())


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
    graph = continuity_integration.DuetStory.execute(**{"Prompt format": "H3 automatic v1"}).expand
    nodes = {n["class_type"]: n["inputs"] for n in graph.values()}
    if profile == "Animate frame":
        assert nodes["MiniMaxH3ImageToVideo"]["last_frame"] is ending
        assert nodes["DuetStoryKeyframeTiming"]["last_index"] == 119
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
    result = continuity_integration.DuetStoryKeyframeTiming.execute(original, 119).args[0]
    assert original[0][1]["minimax_keyframes"][1]["resolved_frame_index"] == 123
    assert result[0][1]["minimax_keyframes"][1]["resolved_frame_index"] == 119
    assert result[0][1]["minimax_keyframes"][1]["latent"] is latent
    assert result[0][1]["other"] == 7
    with pytest.raises(ValueError, match="first/last"):
        continuity_integration.DuetStoryKeyframeTiming.execute(original, 124)


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
    output = continuity_integration.DuetStory.execute()
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
    assert by_type["DuetH3VideoLatent"][1]["latent"] == [samples[0][0], 0]
    assert by_type["DuetH3ReplaceVideoLatent"][1]["original"] == [samples[0][0], 0]
    assert samples[1][1]["latent_image"] == [by_type["DuetH3ReplaceVideoLatent"][0], 0]
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
    assert quality.DuetH3VideoLatent.execute(original).args[0]["samples"] is video
    enlarged = torch.zeros(1, 24, 2, 68, 120)
    result = quality.DuetH3ReplaceVideoLatent.execute(original, {"samples": enlarged}).args[0]
    assert result["samples"].tensors[1] is audio
    assert result["identity"] == "preserved"
    with pytest.raises(ValueError, match="temporal contract"):
        quality.DuetH3ReplaceVideoLatent.execute(original, {"samples": enlarged[:, :, :1]})
    with pytest.raises(ValueError, match="requires"):
        quality.require_upscaler({})
    with pytest.raises(ValueError, match="pinned"):
        quality.require_upscaler({"MinimaxH3LatentUpscaler3D": quality.DuetH3VideoLatent})


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
