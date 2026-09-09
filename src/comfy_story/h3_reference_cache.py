"""Content-addressed disk and logarithmic Duet caches for H3 references."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch

from comfy_story.cache import AuthoritativeLeaf, CacheUpdate, DuetXProductTree
from comfy_story.contracts import DuetXContract, LeafKey
from comfy_story.h3_reference_alignment import AlignedReferencePack
from comfy_story.h3_reference_contracts import (
    H3CompiledBlock,
    H3CompiledContext,
    H3CompileReceipt,
    H3ReferenceBudget,
    H3ReferenceKind,
    H3ReferenceMethod,
    H3VisualReference,
)
from comfy_story.latent_bridge import LatentHistoryBridge
from comfy_story.minimax_h3_training import module_sha256

_CACHE_FORMAT = "duet-x-h3-compiled-reference-cache-v1"
_MAX_CACHE_BYTES = 1024 * 1024 * 1024


def _digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _plain_mapping(value: object, field: str, keys: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be a plain mapping")
    result = cast(dict[str, object], value)
    if any(type(key) is not str for key in result) or set(result) != keys:
        raise ValueError(f"{field} field roster changed")
    return result


class CompiledReferenceCache:
    """Store validated compiled contexts under deterministic input identities."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path) or not root.is_absolute() or root.is_symlink():
            raise ValueError("compiled reference cache root must be an absolute non-symlink path")
        self.root = root

    def key_for(
        self,
        references: tuple[H3VisualReference, ...],
        method: H3ReferenceMethod,
        budget: H3ReferenceBudget,
        *,
        checkpoint_sha256: str | None,
    ) -> str:
        if not isinstance(references, tuple) or not references:
            raise ValueError("compiled reference cache requires a nonempty source tuple")
        for reference in references:
            reference.validate()
        if not isinstance(method, H3ReferenceMethod):
            raise ValueError("compiled reference cache method is unsupported")
        budget.validate()
        if checkpoint_sha256 is not None:
            _digest(checkpoint_sha256, "checkpoint_sha256")
        payload = {
            "budget": asdict(budget),
            "checkpoint_sha256": checkpoint_sha256,
            "format": _CACHE_FORMAT,
            "method": method.value,
            "sources": [
                {
                    "kind": reference.kind.value,
                    "latent_dtype": str(reference.latent.dtype),
                    "latent_shape": list(reference.latent.shape),
                    "ordinal": reference.ordinal,
                    "protected": reference.protected,
                    "source_id": reference.source_id,
                    "tensor_sha256": reference.tensor_sha256,
                }
                for reference in references
            ],
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()

    def _path(self, key: str) -> Path:
        _digest(key, "compiled reference cache key")
        return self.root / f"{key}.pt"

    def store(self, key: str, context: H3CompiledContext) -> Path:
        context.validate()
        path = self._path(key)
        payload = {
            "blocks": [
                {
                    "exact": block.exact,
                    "kind": block.kind.value,
                    "latent": block.latent.detach().contiguous().cpu(),
                    "source_ids": block.source_ids,
                }
                for block in context.blocks
            ],
            "budget": asdict(context.budget),
            "format": _CACHE_FORMAT,
            "key": key,
            "receipt": {
                **asdict(context.receipt),
                "method": context.receipt.method.value,
            },
        }
        self.root.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as handle:
                temporary = Path(handle.name)
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def load(self, key: str) -> H3CompiledContext | None:
        path = self._path(key)
        if not path.exists():
            return None
        if (
            path.is_symlink()
            or not path.is_file()
            or not 0 < path.stat().st_size <= _MAX_CACHE_BYTES
        ):
            raise ValueError("compiled reference cache entry must be a bounded regular file")
        try:
            raw = torch.load(io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError("compiled reference cache entry could not be loaded safely") from error
        payload = _plain_mapping(
            raw,
            "compiled reference cache entry",
            frozenset({"blocks", "budget", "format", "key", "receipt"}),
        )
        if payload["format"] != _CACHE_FORMAT or payload["key"] != key:
            raise ValueError("compiled reference cache identity mismatch")
        budget_data = _plain_mapping(
            payload["budget"],
            "compiled reference budget",
            frozenset({"max_visual_rows", "max_exceptions"}),
        )
        budget = H3ReferenceBudget(
            cast(int, budget_data["max_visual_rows"]),
            cast(int, budget_data["max_exceptions"]),
        ).validate()
        receipt_data = _plain_mapping(
            payload["receipt"],
            "compiled reference receipt",
            frozenset(
                {
                    "cache_status",
                    "checkpoint_sha256",
                    "format",
                    "input_semantic_items",
                    "input_visual_rows",
                    "method",
                    "output_semantic_items",
                    "output_sha256s",
                    "output_visual_rows",
                    "padding_temporal_tokens",
                    "source_sha256s",
                    "trainable_parameters",
                }
            ),
        )
        receipt = H3CompileReceipt(
            cast(str, receipt_data["format"]),
            H3ReferenceMethod(cast(str, receipt_data["method"])),
            cast(int, receipt_data["input_semantic_items"]),
            cast(int, receipt_data["output_semantic_items"]),
            cast(int, receipt_data["input_visual_rows"]),
            cast(int, receipt_data["output_visual_rows"]),
            tuple(cast(tuple[str, ...], receipt_data["source_sha256s"])),
            tuple(cast(tuple[str, ...], receipt_data["output_sha256s"])),
            cast(str | None, receipt_data["checkpoint_sha256"]),
            cast(str, receipt_data["cache_status"]),  # type: ignore[arg-type]
            cast(int, receipt_data["padding_temporal_tokens"]),
            cast(int, receipt_data["trainable_parameters"]),
        ).validate()
        raw_blocks = payload["blocks"]
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise ValueError("compiled reference cache blocks must be a nonempty list")
        blocks: list[H3CompiledBlock] = []
        for index, raw_block in enumerate(raw_blocks):
            block = _plain_mapping(
                raw_block,
                f"compiled reference block {index}",
                frozenset({"exact", "kind", "latent", "source_ids"}),
            )
            blocks.append(
                H3CompiledBlock(
                    H3ReferenceKind(cast(str, block["kind"])),
                    cast(torch.Tensor, block["latent"]),
                    tuple(cast(tuple[str, ...], block["source_ids"])),
                    cast(bool, block["exact"]),
                ).validate()
            )
        return H3CompiledContext(tuple(blocks), receipt, budget).validate()


class H3DuetProductCache:
    """Maintain an eight-leaf ordered Duet product with logarithmic replacement."""

    def __init__(
        self,
        bridge: LatentHistoryBridge,
        pack: AlignedReferencePack,
        tree: DuetXProductTree,
        history: torch.Tensor,
        source_ids: tuple[str, ...],
        source_fingerprints: tuple[str, ...],
    ) -> None:
        self.bridge = bridge
        self.pack = pack
        self.tree = tree
        self._history = history
        self._source_ids = source_ids
        self._source_fingerprints = source_fingerprints

    @classmethod
    def build(
        cls,
        bridge: LatentHistoryBridge,
        pack: AlignedReferencePack,
        *,
        adapter_sha256: str | None = None,
    ) -> H3DuetProductCache:
        if not isinstance(bridge, LatentHistoryBridge):
            raise ValueError("H3 Duet product cache requires a LatentHistoryBridge")
        pack.validate()
        fingerprint = (
            module_sha256(bridge, domain=b"duet-x-h3-reference-bridge-v1")
            if adapter_sha256 is None
            else _digest(adapter_sha256, "adapter_sha256")
        )
        operators = bridge.encode_operators(pack.history, availability=pack.availability)
        actual_count = len(pack.source_ids)
        source_ids = tuple(
            pack.source_ids[slot] if slot < actual_count else f"padding:{slot}" for slot in range(8)
        )
        source_fingerprints = tuple(
            hashlib.sha256(f"duet-x-h3-source-v1\0{source_id}".encode()).hexdigest()
            for source_id in source_ids
        )
        leaves = tuple(
            AuthoritativeLeaf(
                LeafKey(source_ids[slot], slot, slot),
                source_fingerprints[slot],
                0,
                operators[:, slot],
                (),
            )
            for slot in range(8)
        )
        tree = DuetXProductTree(
            DuetXContract.minimax_h3(adapter_fingerprint=fingerprint),
            dense_steps=operators.shape[2],
            operator_size=operators.shape[3],
            leaves=leaves,
        )
        return cls(
            bridge,
            pack,
            tree,
            pack.history.detach().clone(),
            source_ids,
            source_fingerprints,
        )

    @property
    def root_operator(self) -> torch.Tensor:
        return self.tree.root.dense_operator

    def full_rebuild_root(self) -> torch.Tensor:
        return self.tree.full_rebuild_root().dense_operator

    def replace(self, *, slot: int, latent: torch.Tensor, revision: int) -> CacheUpdate:
        if type(slot) is not int or not 0 <= slot < 8:
            raise ValueError("H3 Duet replacement slot must be in range [0,8)")
        if not bool(self.pack.availability[0, slot].item()):
            raise ValueError("H3 Duet replacement slot is unavailable padding")
        expected_shape = tuple(self._history[:, slot].shape)
        if (
            not isinstance(latent, torch.Tensor)
            or latent.layout != torch.strided
            or tuple(latent.shape) != expected_shape
            or latent.dtype != self._history.dtype
            or latent.device != self._history.device
            or not bool(torch.isfinite(latent).all().item())
        ):
            raise ValueError("H3 Duet replacement latent does not match its aligned slot")
        operator = self.bridge.encode_operators(latent[:, None])[:, 0]
        leaf = AuthoritativeLeaf(
            LeafKey(self._source_ids[slot], slot, slot),
            self._source_fingerprints[slot],
            revision,
            operator,
            (),
        )
        update = self.tree.replace(leaf)
        self._history[:, slot] = latent
        return update

    def materialize(self) -> torch.Tensor:
        last_slot = int(self.pack.availability[0].to(torch.int64).sum().item()) - 1
        anchor = self._history[:, last_slot]
        return self.bridge.materialize_operator(
            self.root_operator.to(device=anchor.device),
            anchor=anchor,
        )


__all__ = ("CompiledReferenceCache", "H3DuetProductCache")
