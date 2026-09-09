from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from comfy_story import ltx_n128_materializer as n128_materializer
from comfy_story.config import Method
from comfy_story.contracts import tensor_sha256
from comfy_story.ltx_bridge import LTXLatentHistoryBridge
from comfy_story.ltx_n128_materializer import (
    AuthenticatedObservationPNG,
    N128ComfyIdentity,
    N128ComfyRequest,
    materialize_n128_comfy,
)
from comfy_story.ltx_n128_memory_bundle import load_n128_ltx_memory_bundle
from comfy_story.ltx_quality_generation import FrozenMethodModules
from comfy_story.ltx_quality_protocol import FROZEN_TRAINABLE_CHECKPOINT_SHA256
from comfy_story.ltx_quality_scenes import RGBFrame
from comfy_story.methods import build_method


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def _png(path: Path, value: int) -> AuthenticatedObservationPNG:
    pixels = np.full((1024, 1024, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")
    return AuthenticatedObservationPNG(path, hashlib.sha256(path.read_bytes()).hexdigest())


def _identity() -> N128ComfyIdentity:
    return N128ComfyIdentity(
        "duet-x-n128-comfy-identity-v1",
        _digest("source-contract"),
        _digest("source-registry"),
        _digest("preprocessing"),
        _digest("vae"),
        _digest("workflow"),
        _digest("rights"),
    ).validate()


def _modules() -> FrozenMethodModules:
    with torch.random.fork_rng():
        torch.manual_seed(20260901)
        bridge = LTXLatentHistoryBridge()
        gated = build_method(Method.GATED_CORE_TOPK)
    with torch.no_grad():
        bridge.residual.weight.copy_(torch.eye(128))
    bridge.eval().requires_grad_(False)
    gated.eval().requires_grad_(False)
    return FrozenMethodModules.capture(
        FROZEN_TRAINABLE_CHECKPOINT_SHA256,
        gated,
        bridge,
        torch.device("cpu"),
        expected_gated_type=type(gated),
        expected_bridge_type=LTXLatentHistoryBridge,
    )


def _latent(value: int) -> torch.Tensor:
    channels = torch.sin(torch.arange(128, dtype=torch.float32) * (value + 1) / 255.0)
    return channels.reshape(1, 128, 1, 1, 1).expand(1, 128, 1, 12, 12).contiguous()


@dataclass
class _FakeMaterializer:
    calls: list[int]

    def materialize_frame(self, frame: RGBFrame) -> Any:
        value = int(frame[0, 0, 0])
        self.calls.append(value)
        latent = _latent(value)
        return SimpleNamespace(
            latent=latent,
            receipt=SimpleNamespace(fingerprint=lambda: tensor_sha256(latent)),
        )


@dataclass
class _FakeDecoder:
    received: list[torch.Tensor]

    def __call__(self, latent: torch.Tensor) -> np.ndarray[Any, Any]:
        self.received.append(latent.detach().clone())
        value = int(float(latent.mean().clamp(0, 1).item()) * 255)
        return np.full((384, 384, 3), value, dtype=np.uint8)


def _request(tmp_path: Path, *, pinned_slots: tuple[int, int] = (2, 5)) -> N128ComfyRequest:
    tmp_path.mkdir(parents=True, exist_ok=True)
    neutral = _png(tmp_path / "neutral.png", 48)
    identity = _png(tmp_path / "identity.png", 96)
    fact = _png(tmp_path / "fact.png", 160)
    recent = _png(tmp_path / "recent.png", 208)
    history = tuple(
        identity
        if slot == pinned_slots[0]
        else fact
        if slot == pinned_slots[1]
        else recent
        if slot == 127
        else neutral
        for slot in range(128)
    )
    current = _png(tmp_path / "current.png", 224)
    return N128ComfyRequest(history, pinned_slots, current, tmp_path / "output", _identity())


def test_materializes_one_pass_n128_bundle_with_duet_and_matched_controls(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    modules = _modules()
    frame_materializer = _FakeMaterializer([])
    decoder = _FakeDecoder([])

    receipt = materialize_n128_comfy(request, modules, frame_materializer, decoder)

    assert len(frame_materializer.calls) == 128
    assert receipt.history_items == 128
    assert receipt.pinned_slots == (2, 5)
    assert receipt.anchor_slot == 127
    assert receipt.checkpoint_sha256 == FROZEN_TRAINABLE_CHECKPOINT_SHA256
    assert receipt.core_modules_sha256 == modules.core_modules_sha256
    assert (
        receipt.source_order_sha256
        == hashlib.sha256(
            b"".join(
                slot.to_bytes(2, "big") + bytes.fromhex(source.sha256)
                for slot, source in enumerate(request.history)
            )
        ).hexdigest()
    )
    assert receipt.streaming_audit.equivalent is True
    assert receipt.streaming_audit.ordinary_leaf_count == 126
    assert receipt.streaming_audit.excluded_leaf_count == 2
    assert receipt.retention.input_passes == 1
    assert receipt.retention.full_history_latents_retained is False
    assert receipt.retention.full_history_png_bytes_retained is False
    assert receipt.retention.full_history_latent_bytes == 128 * 128 * 12 * 12 * 4
    assert receipt.retention.peak_balanced_partial_operators <= 7

    assert len(decoder.received) == 3
    assert torch.equal(decoder.received[1], _latent(208))
    ordinary_values = [48] * 128
    ordinary_values[127] = 208
    ordinary_values = [value for slot, value in enumerate(ordinary_values) if slot not in (2, 5)]
    ordinary_history = torch.stack(tuple(_latent(value) for value in ordinary_values), dim=1)
    from comfy_story.methods import HistoryMethod

    assert isinstance(modules.gated, HistoryMethod)
    expected_gated = modules.gated.compress(ordinary_history)
    torch.testing.assert_close(decoder.received[2], expected_gated)

    assert sorted(path.name for path in request.output_dir.iterdir()) == [
        "current.png",
        "duet-core.png",
        "exception-00.png",
        "exception-01.png",
        "gated-core.png",
        "manifest.json",
        "recent-anchor.png",
    ]
    assert (request.output_dir / "exception-00.png").read_bytes() == request.history[
        2
    ].path.read_bytes()
    assert (request.output_dir / "exception-01.png").read_bytes() == request.history[
        5
    ].path.read_bytes()
    assert (request.output_dir / "current.png").read_bytes() == request.current.path.read_bytes()
    assert (request.output_dir / "manifest.json").read_bytes() == receipt.to_json()
    assert tuple(output.role for output in receipt.outputs) == (
        "duet-core",
        "recent-anchor",
        "gated-core",
        "exception-00",
        "exception-01",
        "current",
    )
    for output in receipt.outputs:
        assert (
            hashlib.sha256((request.output_dir / output.filename).read_bytes()).hexdigest()
            == output.png_sha256
        )


def test_sibling_materializer_publishes_tensor_bundle_before_live_state_is_discarded(
    tmp_path: Path,
) -> None:
    sibling = getattr(n128_materializer, "materialize_n128_comfy_with_memory_bundle", None)
    assert sibling is not None, "the bundle-emitting sibling materializer is missing"
    request = _request(tmp_path / "source")
    bundle_root = tmp_path / "bundles"
    bundle_root.mkdir()

    comfy_receipt, bundle_receipt = sibling(
        request,
        _modules(),
        _FakeMaterializer([]),
        _FakeDecoder([]),
        bundle_root,
    )
    restored, loaded_receipt = load_n128_ltx_memory_bundle(
        bundle_root,
        bundle_receipt.bundle_id,
        device=torch.device("cpu"),
    )

    assert loaded_receipt == bundle_receipt
    assert tensor_sha256(restored.materialization.core_tokens) == (
        comfy_receipt.duet_core_tokens_sha256
    )
    assert tensor_sha256(restored.materialization.core_operator) == (
        comfy_receipt.duet_core_operator_sha256
    )
    assert tensor_sha256(restored.core_latent) == comfy_receipt.duet_calibrated_latent_sha256
    assert restored.checkpoint_sha256 == comfy_receipt.checkpoint_sha256


def test_pinned_source_changes_do_not_change_the_excluding_k_cores(tmp_path: Path) -> None:
    left_request = _request(tmp_path / "left")
    right_request = _request(tmp_path / "right")
    changed = list(right_request.history)
    changed[2] = _png(tmp_path / "right" / "changed-identity.png", 12)
    changed[5] = _png(tmp_path / "right" / "changed-fact.png", 244)
    right_request = N128ComfyRequest(
        tuple(changed),
        right_request.pinned_slots,
        right_request.current,
        right_request.output_dir,
        right_request.identity,
    )

    left_decoder = _FakeDecoder([])
    right_decoder = _FakeDecoder([])
    materialize_n128_comfy(left_request, _modules(), _FakeMaterializer([]), left_decoder)
    materialize_n128_comfy(right_request, _modules(), _FakeMaterializer([]), right_decoder)

    for index in range(3):
        torch.testing.assert_close(left_decoder.received[index], right_decoder.received[index])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("count", "exactly 128"),
        ("late-pin", "early half"),
        ("hash", "content SHA-256"),
        ("size", "1024x1024"),
        ("existing-output", "must not exist"),
    ],
)
def test_rejects_invalid_n128_inputs_without_partial_publication(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    request = _request(tmp_path)
    if mutation == "count":
        request = N128ComfyRequest(
            request.history[:-1],
            request.pinned_slots,
            request.current,
            request.output_dir,
            request.identity,
        )
    elif mutation == "late-pin":
        request = N128ComfyRequest(
            request.history,
            (2, 96),
            request.current,
            request.output_dir,
            request.identity,
        )
    elif mutation == "hash":
        changed = list(request.history)
        changed[10] = AuthenticatedObservationPNG(changed[10].path, "0" * 64)
        request = N128ComfyRequest(
            tuple(changed),
            request.pinned_slots,
            request.current,
            request.output_dir,
            request.identity,
        )
    elif mutation == "size":
        path = tmp_path / "bad.png"
        Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8), mode="RGB").save(path)
        changed = list(request.history)
        changed[10] = AuthenticatedObservationPNG(
            path, hashlib.sha256(path.read_bytes()).hexdigest()
        )
        request = N128ComfyRequest(
            tuple(changed),
            request.pinned_slots,
            request.current,
            request.output_dir,
            request.identity,
        )
    elif mutation == "existing-output":
        request.output_dir.mkdir()
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        materialize_n128_comfy(request, _modules(), _FakeMaterializer([]), _FakeDecoder([]))

    assert not request.output_dir.exists() or not any(request.output_dir.iterdir())
