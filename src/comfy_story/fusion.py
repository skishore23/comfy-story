from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import ceil, sqrt
from typing import TypeAlias, cast

import torch
from torch import nn

Leaf: TypeAlias = str
Tree: TypeAlias = Leaf | tuple["Tree", "Tree"]

LEAVES = ("prompt", "event", "material", "acoustics")
FUSION_TREES: dict[str, Tree] = {
    "left_comb": ((("prompt", "event"), "material"), "acoustics"),
    "event_material": (("prompt", ("event", "material")), "acoustics"),
    "balanced": (("prompt", "event"), ("material", "acoustics")),
    "event_material_room": ("prompt", (("event", "material"), "acoustics")),
    "right_comb": ("prompt", ("event", ("material", "acoustics"))),
}
PENTAGON_PATHS = (
    ("left_comb", "balanced", "right_comb"),
    ("left_comb", "event_material", "event_material_room", "right_comb"),
)


def tree_leaves(tree: Tree) -> tuple[str, ...]:
    if isinstance(tree, str):
        return (tree,)
    return tree_leaves(tree[0]) + tree_leaves(tree[1])


def tree_label(tree: Tree) -> str:
    if isinstance(tree, str):
        return tree
    return f"({tree_label(tree[0])} {tree_label(tree[1])})"


def validate_fusion_tree(tree: Tree) -> None:
    if tree_leaves(tree) != LEAVES:
        raise ValueError(f"fusion tree must preserve leaf order {LEAVES!r}")


@dataclass(frozen=True)
class TamariRotation:
    source: str
    target: str
    location: tuple[int, ...]

    def label(self) -> str:
        location = "root" if not self.location else ".".join(map(str, self.location))
        return f"{self.source} --alpha@{location}--> {self.target}"


def _right_rotations(tree: Tree, location: tuple[int, ...] = ()) -> dict[Tree, tuple[int, ...]]:
    if isinstance(tree, str):
        return {}
    left, right = tree
    rotations: dict[Tree, tuple[int, ...]] = {}
    if isinstance(left, tuple):
        x, y = left
        rotations[(x, (y, right))] = location
    for child, child_location in _right_rotations(left, (*location, 0)).items():
        rotations[(child, right)] = child_location
    for child, child_location in _right_rotations(right, (*location, 1)).items():
        rotations[(left, child)] = child_location
    return rotations


def tamari_rotations() -> tuple[TamariRotation, ...]:
    reverse = {tree: name for name, tree in FUSION_TREES.items()}
    rotations = []
    for source, tree in FUSION_TREES.items():
        for target, location in _right_rotations(tree).items():
            rotations.append(TamariRotation(source, reverse[target], location))
    return tuple(sorted(rotations, key=lambda item: (item.source, item.target)))


def tamari_neighbors(name: str) -> tuple[str, ...]:
    return tuple(rotation.target for rotation in tamari_rotations() if rotation.source == name)


def rebracketing_path(source: str, destination: str, *, directed: bool = False) -> tuple[str, ...]:
    if source not in FUSION_TREES or destination not in FUSION_TREES:
        raise ValueError("source and destination must be named Tamari bracketings")
    if source == destination:
        return (source,)
    adjacency: dict[str, set[str]] = {name: set() for name in FUSION_TREES}
    for rotation in tamari_rotations():
        adjacency[rotation.source].add(rotation.target)
        if not directed:
            adjacency[rotation.target].add(rotation.source)
    queue: deque[tuple[str, ...]] = deque([(source,)])
    visited = {source}
    while queue:
        path = queue.popleft()
        for neighbor in sorted(adjacency[path[-1]]):
            if neighbor == destination:
                return (*path, neighbor)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((*path, neighbor))
    raise ValueError(f"no legal Tamari path from {source!r} to {destination!r}")


class BinaryTokenFusion(nn.Module):
    def __init__(self, channels: int, *, hidden_channels: int = 0):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if hidden_channels < 0:
            raise ValueError("hidden_channels must be non-negative")
        self.hidden_channels = hidden_channels or channels * 2
        self.candidate = nn.Sequential(
            nn.Linear(channels * 2, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, channels),
        )
        self.gate = nn.Linear(channels * 2, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape or left.ndim != 3:
            raise ValueError("fusion requires matching SoundToken[B,T,C] boundaries")
        joined = torch.cat((left, right), dim=-1)
        gate = self.gate(joined).sigmoid()
        return cast(
            torch.Tensor,
            self.norm(gate * left + (1 - gate) * right + self.candidate(joined)),
        )


class LearnedAssociator(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        output = nn.Linear(channels * expansion, channels)
        self.net = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.SiLU(),
            output,
        )
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("associator requires SoundToken[B,T,C]")
        return cast(torch.Tensor, tokens + self.net(self.norm(tokens)))


class AssociatorBank(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        edges: dict[str, LearnedAssociator] = {}
        for rotation in tamari_rotations():
            edges[self.edge_key(rotation.source, rotation.target)] = LearnedAssociator(channels)
            edges[self.edge_key(rotation.target, rotation.source)] = LearnedAssociator(channels)
        self.edges = nn.ModuleDict(edges)

    @staticmethod
    def edge_key(source: str, destination: str) -> str:
        return f"{source}__to__{destination}"

    def transport(self, tokens: torch.Tensor, path: Sequence[str]) -> torch.Tensor:
        if not path:
            raise ValueError("transport path cannot be empty")
        result = tokens
        for source, destination in pairwise(path):
            key = self.edge_key(source, destination)
            if key not in self.edges:
                raise ValueError(f"{source!r} -> {destination!r} is not a local rotation")
            result = cast(torch.Tensor, self.edges[key](result))
        return result


class RotationConditionedAssociator(nn.Module):
    """Shared low-rank residual transport for every oriented Tamari rotation.

    The output projection starts at zero, so enabling the module does not perturb an
    existing model until its transport parameters have been trained.  A separate rank
    embedding identifies each directed edge while the low-rank projections are shared.
    """

    def __init__(self, channels: int, rank: int):
        super().__init__()
        if rank <= 0:
            raise ValueError("associator rank must be positive")
        self.norm = nn.LayerNorm(channels)
        self.down = nn.Linear(channels, rank, bias=False)
        self.rotation_embedding = nn.Embedding(len(self.oriented_edges()), rank)
        self.up = nn.Linear(rank, channels, bias=False)
        nn.init.zeros_(self.up.weight)
        self._edge_indices = {
            self.edge_key(source, destination): index
            for index, (source, destination) in enumerate(self.oriented_edges())
        }

    @staticmethod
    def edge_key(source: str, destination: str) -> str:
        return f"{source}__to__{destination}"

    @staticmethod
    def oriented_edges() -> tuple[tuple[str, str], ...]:
        forward = tuple((rotation.source, rotation.target) for rotation in tamari_rotations())
        reverse = tuple((target, source) for source, target in forward)
        return forward + reverse

    def forward(self, tokens: torch.Tensor, source: str, destination: str) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("associator requires SoundToken[B,T,C]")
        key = self.edge_key(source, destination)
        try:
            edge_index = self._edge_indices[key]
        except KeyError as exc:
            raise ValueError(f"{source!r} -> {destination!r} is not a local rotation") from exc
        index = torch.tensor(edge_index, device=tokens.device)
        rotation = self.rotation_embedding(index)[None, None]
        update = self.up(torch.nn.functional.silu(self.down(self.norm(tokens)) + rotation))
        return cast(torch.Tensor, tokens + update)

    def transport(self, tokens: torch.Tensor, path: Sequence[str]) -> torch.Tensor:
        if not path:
            raise ValueError("transport path cannot be empty")
        result = tokens
        for source, destination in pairwise(path):
            result = self(result, source, destination)
        return result


@dataclass(frozen=True)
class FusionSpec:
    name: str = "balanced"

    @property
    def tree(self) -> Tree:
        try:
            tree = FUSION_TREES[self.name]
        except KeyError as exc:
            raise ValueError(f"unknown bracketing {self.name!r}") from exc
        validate_fusion_tree(tree)
        return tree

    def label(self) -> str:
        return tree_label(self.tree)


class TamariFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        spec: FusionSpec,
        *,
        shared_nodes: bool = False,
        hidden_channels: int = 0,
    ):
        super().__init__()
        self.spec = spec
        self.shared_nodes = shared_nodes
        node_count = 1 if shared_nodes else 3
        self.nodes = nn.ModuleList(
            BinaryTokenFusion(channels, hidden_channels=hidden_channels) for _ in range(node_count)
        )

    def forward(
        self,
        prompt: torch.Tensor,
        event: torch.Tensor,
        material: torch.Tensor,
        acoustics: torch.Tensor,
        *,
        bracketing: str | None = None,
    ) -> torch.Tensor:
        streams: Mapping[str, torch.Tensor] = {
            "prompt": prompt,
            "event": event,
            "material": material,
            "acoustics": acoustics,
        }
        tree = self.spec.tree if bracketing is None else FusionSpec(bracketing).tree
        node_index = 0

        def compile_tree(subtree: Tree) -> torch.Tensor:
            nonlocal node_index
            if isinstance(subtree, str):
                return streams[subtree]
            left = compile_tree(subtree[0])
            right = compile_tree(subtree[1])
            node = self.nodes[0] if self.shared_nodes else self.nodes[node_index]
            node_index += 1
            return cast(torch.Tensor, node(left, right))

        return compile_tree(tree)


class MatrixSemigroupFusion(nn.Module):
    """Associative-by-construction fusion over learned matrix operators.

    Each typed token is encoded as a homogeneous rigid-motion matrix: an
    orthogonal rotation from a Cayley transform plus a bounded translation. Internal
    nodes use ordered matrix multiplication, so parentheses cannot change the product
    apart from floating-point roundoff. The representation is non-commutative and
    order-sensitive; rotations preserve norm and translations grow at most linearly.
    """

    def __init__(
        self,
        channels: int,
        spec: FusionSpec,
        *,
        operator_size: int = 0,
        update_scale: float = 0.25,
        initialization_std: float = 0.005,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if operator_size < 0 or operator_size == 1:
            raise ValueError("operator_size must be zero or at least two")
        if update_scale <= 0:
            raise ValueError("update_scale must be positive")
        if initialization_std <= 0:
            raise ValueError("initialization_std must be positive")
        self.spec = spec
        self.shared_nodes = True
        self.channels = channels
        self.operator_size = operator_size or ceil(sqrt(2 * channels))
        self.update_scale = update_scale
        self.initialization_std = initialization_std
        operator_channels = self.operator_size**2
        self.input_norm = nn.LayerNorm(channels)
        self.encoder = nn.Linear(channels, operator_channels)
        nn.init.normal_(self.encoder.weight, std=self.initialization_std)
        nn.init.zeros_(self.encoder.bias)
        self.decoder = nn.Linear(operator_channels, channels)
        self.output_norm = nn.LayerNorm(channels)
        self.register_buffer("identity", torch.eye(self.operator_size), persistent=False)

    def encode_stream(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode one local token stream as independently transferable operators."""
        if tokens.ndim < 3 or tokens.shape[-1] != self.channels:
            raise ValueError(f"fusion requires [...,T,{self.channels}] token boundaries")
        update = self.encoder(self.input_norm(tokens)).tanh()
        update = update.reshape(*tokens.shape[:-1], self.operator_size, self.operator_size)
        # The Cayley transform produces the rotation block. A bounded
        # translation makes first-order additive information available while the
        # homogeneous matrix product keeps exact associativity and order.
        update = update.float()
        rotation_update = update[..., :-1, :-1]
        skew = self.update_scale * (rotation_update - rotation_update.transpose(-1, -2))
        rotation_identity = torch.eye(
            self.operator_size - 1, device=tokens.device, dtype=torch.float32
        )
        rotation_identity = torch.broadcast_to(rotation_identity, skew.shape)
        rotation = torch.linalg.solve(rotation_identity - skew, rotation_identity + skew)
        translation = self.update_scale * update[..., :-1, -1:]
        top = torch.cat((rotation, translation), dim=-1)
        bottom = torch.zeros(
            *top.shape[:-2],
            1,
            self.operator_size,
            device=tokens.device,
            dtype=torch.float32,
        )
        bottom[..., 0, -1] = 1
        return torch.cat((top, bottom), dim=-2)

    def identity_operator_like(self, operator: torch.Tensor) -> torch.Tensor:
        """Return the exact missing-stream identity at an operator boundary."""
        self._validate_operator(operator)
        identity = cast(torch.Tensor, self.identity).to(
            device=operator.device, dtype=operator.dtype
        )
        return torch.broadcast_to(identity, operator.shape)

    def mask_operator(self, operator: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        """Replace unavailable batch elements of one encoded stream by identity."""
        self._validate_operator(operator)
        expected = (operator.shape[0],)
        if availability.dtype is not torch.bool or tuple(availability.shape) != expected:
            raise ValueError(f"availability must be bool[{expected[0]}]")
        mask = availability.reshape(availability.shape[0], *([1] * (operator.ndim - 1)))
        return torch.where(mask, operator, self.identity_operator_like(operator))

    def combine_ordered_operators(self, operators: Sequence[torch.Tensor]) -> torch.Tensor:
        """Combine local or partial operators with an order-preserving balanced tree."""
        if not operators:
            raise ValueError("ordered operator fusion requires at least one operator")
        reference = operators[0]
        self._validate_operator(reference)
        for operator in operators[1:]:
            self._validate_operator(operator)
            if operator.shape != reference.shape:
                raise ValueError("fusion requires matching operator boundaries")
        level = torch.stack(tuple(operators), dim=0)
        while level.shape[0] > 1:
            pair_count = level.shape[0] // 2
            left = level[: 2 * pair_count : 2]
            right = level[1 : 2 * pair_count : 2]
            combined = torch.matmul(right, left)
            if level.shape[0] % 2:
                combined = torch.cat((combined, level[-1:]), dim=0)
            level = combined
        return level[0]

    def decode_operator(self, operator: torch.Tensor) -> torch.Tensor:
        """Decode a centralized or hierarchically aggregated operator state."""
        self._validate_operator(operator)
        decoded = self.decoder(operator.flatten(start_dim=-2))
        return cast(torch.Tensor, self.output_norm(decoded))

    def _validate_operator(self, operator: torch.Tensor) -> None:
        expected = (self.operator_size, self.operator_size)
        if operator.ndim < 4 or tuple(operator.shape[-2:]) != expected:
            raise ValueError(
                "operator boundary must have shape "
                f"[...,T,{self.operator_size},{self.operator_size}]"
            )

    def fuse_ordered_tensor(
        self,
        streams: torch.Tensor,
        availability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse typed streams with a parallel, order-preserving tree reduction."""
        if streams.ndim != 4 or streams.shape[-1] != self.channels:
            raise ValueError(f"ordered fusion requires streams[B,S,T,{self.channels}]")
        batch, stream_count, _, _ = streams.shape
        if stream_count < 1:
            raise ValueError("ordered fusion requires at least one stream")
        encoded = self.encode_stream(streams)
        operators = list(encoded.unbind(dim=1))
        if availability is not None:
            expected = (batch, stream_count)
            if availability.dtype is not torch.bool or tuple(availability.shape) != expected:
                raise ValueError(f"availability must be bool[{expected[0]},{expected[1]}]")
            operators = [
                self.mask_operator(operator, availability[:, index])
                for index, operator in enumerate(operators)
            ]
        return self.decode_operator(self.combine_ordered_operators(operators))

    def fuse_ordered(
        self,
        streams: Sequence[torch.Tensor],
        availability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse any non-empty ordered stream sequence at a shared token boundary.

        Missing streams contribute the exact matrix identity.  The availability signal
        is deliberately not learned here: downstream models can condition on it without
        weakening the associative no-op contract.
        """
        if not streams:
            raise ValueError("ordered fusion requires at least one stream")
        reference = streams[0]
        if any(stream.shape != reference.shape for stream in streams[1:]):
            raise ValueError("fusion requires matching SoundToken[B,T,C] boundaries")
        stacked = torch.stack(tuple(streams), dim=1)
        return self.fuse_ordered_tensor(stacked, availability)

    def forward(
        self,
        prompt: torch.Tensor,
        event: torch.Tensor,
        material: torch.Tensor,
        acoustics: torch.Tensor,
        *,
        bracketing: str | None = None,
    ) -> torch.Tensor:
        streams = (prompt, event, material, acoustics)
        if any(stream.shape != prompt.shape for stream in streams[1:]):
            raise ValueError("fusion requires matching SoundToken[B,T,C] boundaries")
        _ = self.spec.tree if bracketing is None else FusionSpec(bracketing).tree
        return self.fuse_ordered(streams)
