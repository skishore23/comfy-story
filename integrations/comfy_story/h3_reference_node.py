"""Internal Nodes 2.0 boundary for H3 reference compilation and telemetry."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Self

import torch
from comfy_api.latest import io

from comfy_story.h3_reference_cache import CompiledReferenceCache
from comfy_story.h3_reference_compressors import H3ReferenceCompiler
from comfy_story.h3_reference_conditioning import (
    extract_visual_references,
    replace_visual_references,
)
from comfy_story.h3_reference_contracts import (
    CacheStatus,
    H3CompiledBlock,
    H3CompiledContext,
    H3CompileReceipt,
    H3ReferenceBudget,
    H3ReferenceMethod,
    H3SelectedSource,
)

PreparedReferenceType = io.Custom("DUET_H3_REFERENCE_COMPILE")
CompileReceiptType = io.Custom("DUET_H3_COMPILE_RECEIPT")
TelemetryType = io.Custom("DUET_H3_TELEMETRY")
TelemetryResultType = io.Custom("DUET_H3_TELEMETRY_RESULT")
StoryType = io.Custom("DUET_STORY")

_BENCHMARK_FORMAT = "duet-x-h3-runtime-receipt-v1"
_CELL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True, slots=True)
class PreparedReferenceCompile:
    """Authenticated compiler, selected sources, and optional persistent cache."""

    sources: tuple[H3SelectedSource, ...]
    budget: H3ReferenceBudget
    compiler: H3ReferenceCompiler
    cache: CompiledReferenceCache | None = None

    def validate(self) -> Self:
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("prepared H3 reference compile requires selected sources")
        for source in self.sources:
            source.validate()
        self.budget.validate()
        if not isinstance(self.compiler, H3ReferenceCompiler):
            raise ValueError("prepared H3 reference compile requires an H3 compiler")
        if self.compiler.budget != self.budget:
            raise ValueError("prepared H3 reference compiler budget changed")
        if self.cache is not None and not isinstance(self.cache, CompiledReferenceCache):
            raise ValueError("prepared H3 reference cache is invalid")
        return self


@dataclass(frozen=True, slots=True)
class TelemetryStart:
    start_ns: int
    cuda: bool


@dataclass(frozen=True, slots=True)
class DenoisingTelemetry:
    elapsed_ns: int
    max_memory_allocated: int
    max_memory_reserved: int


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical_json(value) + b"\n")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _move_context(
    context: H3CompiledContext,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cache_status: CacheStatus,
) -> H3CompiledContext:
    blocks = tuple(
        H3CompiledBlock(
            block.kind,
            block.latent.to(device=device, dtype=dtype),
            block.source_ids,
            block.exact,
        )
        for block in context.blocks
    )
    receipt = replace(context.receipt, cache_status=cache_status)
    return H3CompiledContext(blocks, receipt, context.budget).validate()


class DuetH3CompileReferences(io.ComfyNode):
    """Replace only H3's visual reference list with the bounded compiled blocks."""

    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="DuetH3CompileReferences",
            display_name="_Duet H3 Compile References",
            category="_Duet/Internal",
            inputs=[
                io.Conditioning.Input("positive"),
                PreparedReferenceType.Input("request"),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                CompileReceiptType.Output("receipt"),
                io.Int.Output("context_build_ns"),
            ],
        )

    @classmethod
    def execute(cls, positive: object, request: PreparedReferenceCompile) -> Any:
        del cls
        started_ns = time.perf_counter_ns()
        request.validate()
        references = extract_visual_references(positive, request.sources)
        device = references[0].latent.device
        dtype = references[0].latent.dtype
        if any(reference.latent.device != device for reference in references):
            raise ValueError("H3 visual references must share one device")
        context: H3CompiledContext | None = None
        key: str | None = None
        if request.cache is not None:
            key = request.cache.key_for(
                references,
                request.compiler.method,
                request.budget,
                checkpoint_sha256=request.compiler.checkpoint_sha256,
            )
            cached = request.cache.load(key)
            if cached is not None:
                context = _move_context(cached, device=device, dtype=dtype, cache_status="hit")
        if context is None:
            request.compiler.to(device)
            try:
                context = request.compiler.compile_references(references)
            finally:
                request.compiler.to("cpu")
            if request.cache is not None and key is not None:
                stored = _move_context(
                    context,
                    device=torch.device("cpu"),
                    dtype=dtype,
                    cache_status="updated",
                )
                request.cache.store(key, stored)
                context = replace(context, receipt=replace(context.receipt, cache_status="miss"))
                context.validate()
        conditioned = replace_visual_references(positive, context)
        elapsed_ns = time.perf_counter_ns() - started_ns
        return io.NodeOutput(conditioned, context.receipt, elapsed_ns)


class DuetH3TelemetryStart(io.ComfyNode):
    """Internal start marker for the denoising-only measurement region."""

    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="DuetH3TelemetryStart",
            display_name="_Duet H3 Telemetry Start",
            category="_Duet/Internal",
            inputs=[io.Conditioning.Input("positive")],
            outputs=[
                io.Conditioning.Output("positive"),
                TelemetryType.Output("telemetry"),
            ],
        )

    @classmethod
    def execute(cls, positive: object) -> Any:
        del cls
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return io.NodeOutput(positive, TelemetryStart(time.perf_counter_ns(), cuda))


class DuetH3TelemetryEnd(io.ComfyNode):
    """Internal end marker that leaves the sampled latent unchanged."""

    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="DuetH3TelemetryEnd",
            display_name="_Duet H3 Telemetry End",
            category="_Duet/Internal",
            inputs=[
                io.Latent.Input("latent"),
                TelemetryType.Input("telemetry"),
            ],
            outputs=[
                io.Latent.Output("latent"),
                TelemetryResultType.Output("result"),
            ],
        )

    @classmethod
    def execute(cls, latent: object, telemetry: TelemetryStart) -> Any:
        del cls
        if not isinstance(telemetry, TelemetryStart):
            raise ValueError("Duet H3 telemetry start marker is invalid")
        if telemetry.cuda:
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
        else:
            allocated = 0
            reserved = 0
        result = DenoisingTelemetry(
            time.perf_counter_ns() - telemetry.start_ns,
            allocated,
            reserved,
        )
        return io.NodeOutput(latent, result)


class DuetH3BenchmarkReceipt(io.ComfyNode):
    """Publish a canonical receipt to history and, when configured, a local mirror."""

    @classmethod
    def define_schema(cls) -> Any:
        return io.Schema(
            node_id="DuetH3BenchmarkReceipt",
            display_name="_Duet H3 Benchmark Receipt",
            category="_Duet/Internal",
            inputs=[
                StoryType.Input("story_state"),
                PreparedReferenceType.Input("request"),
                CompileReceiptType.Input("compile_receipt"),
                io.Int.Input("context_build_ns", min=0),
                TelemetryResultType.Input("denoising"),
                io.String.Input("phase"),
                io.String.Input("workload_id"),
                io.String.Input("method"),
                io.Int.Input("seed", min=0),
                io.String.Input("cell_id"),
            ],
            outputs=[StoryType.Output("story_state")],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        story_state: object,
        request: PreparedReferenceCompile,
        compile_receipt: object,
        context_build_ns: int,
        denoising: object,
        phase: str,
        workload_id: str,
        method: str,
        seed: int,
        cell_id: str,
    ) -> Any:
        del cls
        request.validate()
        if not isinstance(compile_receipt, H3CompileReceipt):
            raise ValueError("Duet H3 compile receipt is invalid")
        compile_receipt.validate()
        if request.compiler.method is not compile_receipt.method:
            raise ValueError("benchmark request method differs from compiler receipt")
        if len(request.sources) != compile_receipt.input_semantic_items:
            raise ValueError("benchmark source roster differs from compiler receipt")
        if not isinstance(denoising, DenoisingTelemetry):
            raise ValueError("Duet H3 denoising telemetry is invalid")
        if type(context_build_ns) is not int or context_build_ns < 0:
            raise ValueError("context build time must be a nonnegative integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("benchmark seed must be a nonnegative integer")
        if phase not in {"screening", "confirmation"}:
            raise ValueError("benchmark phase is unsupported")
        if not workload_id or _CELL_ID.fullmatch(workload_id) is None:
            raise ValueError("benchmark workload id is invalid")
        if _CELL_ID.fullmatch(cell_id) is None:
            raise ValueError("benchmark cell id is invalid")
        try:
            resolved_method = H3ReferenceMethod(method)
        except ValueError as error:
            raise ValueError("benchmark method is unsupported") from error
        if resolved_method is not compile_receipt.method:
            raise ValueError("benchmark method differs from compiler receipt")
        compiler = asdict(compile_receipt)
        compiler["method"] = compile_receipt.method.value
        payload = {
            "cell": {
                "cell_id": cell_id,
                "method": method,
                "phase": phase,
                "seed": seed,
                "workload_id": workload_id,
            },
            "compiler": compiler,
            "format": _BENCHMARK_FORMAT,
            "selection": {
                "protected_source_ids": [
                    source.source_id for source in request.sources if source.protected
                ],
                "sources": [
                    {
                        "kind": source.kind.value,
                        "ordinal": source.ordinal,
                        "protected": source.protected,
                        "source_id": source.source_id,
                    }
                    for source in request.sources
                ],
            },
            "telemetry": {
                "context_build_ns": context_build_ns,
                "denoising_elapsed_ns": denoising.elapsed_ns,
                "max_memory_allocated": denoising.max_memory_allocated,
                "max_memory_reserved": denoising.max_memory_reserved,
            },
        }
        # Normalize tuples and enums to the exact JSON-native structure Comfy
        # stores in history; the disk mirror and API receipt remain byte-equivalent.
        payload = json.loads(_canonical_json(payload))
        root_value = os.environ.get("DUET_H3_BENCHMARK_ROOT")
        if root_value:
            unresolved_root = Path(root_value)
            if unresolved_root.is_symlink():
                raise ValueError("benchmark receipt root must be a non-symlink directory")
            root = unresolved_root.resolve()
            if root.exists() and not root.is_dir():
                raise ValueError("benchmark receipt root must be a non-symlink directory")
            _atomic_json(root / phase / f"{cell_id}.json", payload)
        return io.NodeOutput(story_state, ui={"duet_h3_benchmark": [payload]})


__all__ = (
    "DenoisingTelemetry",
    "DuetH3BenchmarkReceipt",
    "DuetH3CompileReferences",
    "DuetH3TelemetryEnd",
    "DuetH3TelemetryStart",
    "PreparedReferenceCompile",
)
