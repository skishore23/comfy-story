"""Forge runtime adapter for the N=128 Duet-X Comfy materializer.

This module deliberately keeps orchestration thin.  It authenticates a twelve-source ordinary
history, places two variant-specific early exceptions at slots 2 and 19, places the current frame
at slot 127, and delegates the one-pass bounded computation to :mod:`ltx_n128_materializer`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from duet.duetx.contracts import DuetXContract
from duet.duetx.ltx_n128_materializer import (
    AuthenticatedObservationPNG,
    N128ComfyIdentity,
    N128ComfyReceipt,
    N128ComfyRequest,
    materialize_n128_comfy,
)
from duet.duetx.ltx_quality_generation import load_frozen_trainable_checkpoint
from duet.duetx.ltx_quality_protocol import canonical_json
from duet.duetx.ltx_quality_runtime import FrozenQualityRuntimeIdentity, PinnedOfficialLTXFactory

PINNED_SLOTS = (2, 19)
HISTORY_ITEMS = 128
ORDINARY_SOURCE_COUNT = 12
CURRENT_SLOT = 127


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash one value with the experiment's canonical JSON encoding."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def authenticate(path: Path) -> AuthenticatedObservationPNG:
    """Bind one absolute, regular, non-symlink PNG path to its bytes."""
    if path.is_symlink():
        raise ValueError(f"source observation is not an authenticated PNG: {path}")
    absolute = path.resolve()
    if not absolute.is_file() or absolute.suffix.lower() != ".png":
        raise ValueError(f"source observation is not an authenticated PNG: {absolute}")
    return AuthenticatedObservationPNG(absolute, file_sha256(absolute)).validate()


def build_history(
    ordinary: tuple[AuthenticatedObservationPNG, ...],
    exception_identity: AuthenticatedObservationPNG,
    exception_object: AuthenticatedObservationPNG,
    current: AuthenticatedObservationPNG,
) -> tuple[AuthenticatedObservationPNG, ...]:
    """Build the frozen chronological roster without materializing image bytes in memory."""
    if len(ordinary) != ORDINARY_SOURCE_COUNT:
        raise ValueError("N=128 runtime requires exactly twelve ordinary sources")
    rows: list[AuthenticatedObservationPNG] = []
    ordinary_index = 0
    for slot in range(HISTORY_ITEMS):
        if slot == PINNED_SLOTS[0]:
            rows.append(exception_identity)
        elif slot == PINNED_SLOTS[1]:
            rows.append(exception_object)
        elif slot == CURRENT_SLOT:
            rows.append(current)
        else:
            rows.append(ordinary[ordinary_index % len(ordinary)])
            ordinary_index += 1
    return tuple(rows)


class OfficialLatentDecoder:
    """Decode one LTX latent through the pinned official VAE decoder."""

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        loader_module = importlib.import_module("ltx_trainer.model_loader")
        load_video_vae_decoder = cast(Any, loader_module.load_video_vae_decoder)
        self.device = device
        self.decoder = load_video_vae_decoder(
            checkpoint,
            device=device,
            dtype=torch.bfloat16,
        )
        self.decoder.eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, latent: torch.Tensor) -> np.ndarray[Any, Any]:
        value = self.decoder(latent.to(device=self.device, dtype=torch.bfloat16))
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, 3, 1, 384, 384):
            raise ValueError("official LTX decoder returned an unexpected frame boundary")
        return np.ascontiguousarray(
            ((value[0, :, 0].float() + 1.0) / 2.0)
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    def close(self) -> None:
        self.decoder.to("cpu")


def _source_registry(
    ordinary: tuple[AuthenticatedObservationPNG, ...],
    a_identity: AuthenticatedObservationPNG,
    a_object: AuthenticatedObservationPNG,
    b_identity: AuthenticatedObservationPNG,
    b_object: AuthenticatedObservationPNG,
    current: AuthenticatedObservationPNG,
) -> dict[str, object]:
    return {
        "ordinary": [{"path": str(source.path), "sha256": source.sha256} for source in ordinary],
        "variant_a": {
            "identity": {"path": str(a_identity.path), "sha256": a_identity.sha256},
            "object": {"path": str(a_object.path), "sha256": a_object.sha256},
        },
        "variant_b": {
            "identity": {"path": str(b_identity.path), "sha256": b_identity.sha256},
            "object": {"path": str(b_object.path), "sha256": b_object.sha256},
        },
        "current": {"path": str(current.path), "sha256": current.sha256},
    }


def _assert_counterfactual_pair(a: N128ComfyReceipt, b: N128ComfyReceipt) -> None:
    """Require A/B to differ only in the two excluded exception observations."""
    if a.pinned_slots != b.pinned_slots or a.pinned_slots != PINNED_SLOTS:
        raise ValueError("A/B pinned slots changed")
    for name in (
        "duet_core_operator_sha256",
        "duet_core_tokens_sha256",
        "duet_calibrated_latent_sha256",
        "recent_anchor_latent_sha256",
        "gated_core_latent_sha256",
        "current_source_sha256",
    ):
        if getattr(a, name) != getattr(b, name):
            raise ValueError(f"A/B non-exception state differs: {name}")
    differing_slots = tuple(
        left.slot
        for left, right in zip(a.history, b.history, strict=True)
        if left.source_png_sha256 != right.source_png_sha256
    )
    if differing_slots != PINNED_SLOTS:
        raise ValueError("A/B source histories must differ only at the two pinned slots")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ordinary", type=Path, nargs=ORDINARY_SOURCE_COUNT, required=True)
    parser.add_argument("--a-identity", type=Path, required=True)
    parser.add_argument("--a-object", type=Path, required=True)
    parser.add_argument("--b-identity", type=Path, required=True)
    parser.add_argument("--b-object", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workflow-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Materialize the matched A/B N=128 guide bundles exactly once."""
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("N=128 output root must be fresh")
    output_root.mkdir(mode=0o700, parents=True)

    ordinary = tuple(authenticate(path) for path in args.ordinary)
    a_identity = authenticate(args.a_identity)
    a_object = authenticate(args.a_object)
    b_identity = authenticate(args.b_identity)
    b_object = authenticate(args.b_object)
    current = authenticate(args.current)
    registry = _source_registry(ordinary, a_identity, a_object, b_identity, b_object, current)
    runtime = FrozenQualityRuntimeIdentity.default(
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
    ).validate()
    device = torch.device(runtime.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("frozen CUDA runtime is unavailable")
    identity = N128ComfyIdentity(
        "duet-x-n128-comfy-identity-v1",
        DuetXContract.default(decision1_fingerprint="0" * 64).fingerprint(),
        canonical_sha256(registry),
        canonical_sha256(
            {
                "history_items": HISTORY_ITEMS,
                "ordinary_sources": ORDINARY_SOURCE_COUNT,
                "resize": "PIL RGB LANCZOS 1024-to-384",
                "roster": "cycle ordinary; exceptions at 2,19; current at 127",
            }
        ),
        runtime.teacher_checkpoint_sha256,
        args.workflow_sha256,
        canonical_sha256({"commercial": False, "sources": "self-authored Comfy source plates"}),
    ).validate()
    modules = load_frozen_trainable_checkpoint(args.checkpoint.resolve(), device=device).validate()
    factory = PinnedOfficialLTXFactory(
        checkout_root=Path(runtime.ltx_checkout_root),
        pipelines_source_root=Path(runtime.ltx_pipelines_source_root),
        core_source_root=Path(runtime.ltx_core_source_root),
        dependency_source_root=Path(runtime.dependency_source_root),
    )
    api = factory.load()
    frame_materializer = api.build_materializer(runtime, device=device)
    decoder = OfficialLatentDecoder(Path(runtime.teacher_checkpoint_path), device)
    receipts: dict[str, N128ComfyReceipt] = {}
    try:
        for variant, exception_identity, exception_object in (
            ("variant-a", a_identity, a_object),
            ("variant-b", b_identity, b_object),
        ):
            request = N128ComfyRequest(
                build_history(ordinary, exception_identity, exception_object, current),
                PINNED_SLOTS,
                current,
                output_root / variant,
                identity,
            ).validate()
            receipts[variant] = materialize_n128_comfy(
                request, modules, frame_materializer, decoder
            )
    finally:
        decoder.close()
        frame_materializer.close()
        del modules
        gc.collect()
        torch.cuda.empty_cache()

    _assert_counterfactual_pair(receipts["variant-a"], receipts["variant-b"])
    report = {
        "format": "duet-x-n128-comfy-materialization-run-v1",
        "history_items": HISTORY_ITEMS,
        "ordinary_source_count": ORDINARY_SOURCE_COUNT,
        "pinned_slots": list(PINNED_SLOTS),
        "current_slot": CURRENT_SLOT,
        "source_registry": registry,
        "source_registry_sha256": canonical_sha256(registry),
        "checkpoint_sha256": receipts["variant-a"].checkpoint_sha256,
        "variants": {name: json.loads(receipt.to_json()) for name, receipt in receipts.items()},
    }
    (output_root / "run.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CURRENT_SLOT",
    "HISTORY_ITEMS",
    "ORDINARY_SOURCE_COUNT",
    "PINNED_SLOTS",
    "OfficialLatentDecoder",
    "authenticate",
    "build_history",
    "canonical_sha256",
    "file_sha256",
    "main",
)
