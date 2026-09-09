import pytest
import torch

from comfy_story.memory import MatrixSemigroupFusion as PublicMatrixSemigroupFusion
from comfy_story.memory.fusion import (
    FUSION_TREES,
    LEAVES,
    BinaryTokenFusion,
    FusionSpec,
    MatrixSemigroupFusion,
    RotationConditionedAssociator,
    TamariFusion,
    rebracketing_path,
    tamari_rotations,
    tree_leaves,
)


def test_five_trees_preserve_typed_leaf_order() -> None:
    assert len(FUSION_TREES) == 5
    assert len(tamari_rotations()) == 5
    assert all(tree_leaves(tree) == LEAVES for tree in FUSION_TREES.values())


def test_audio_fusion_and_path() -> None:
    fusion = TamariFusion(16, FusionSpec("balanced"))
    streams = [torch.randn(2, 8, 16) for _ in range(4)]
    assert fusion(*streams).shape == streams[0].shape
    assert rebracketing_path("left_comb", "right_comb") in (
        ("left_comb", "balanced", "right_comb"),
        ("left_comb", "event_material", "event_material_room", "right_comb"),
    )


def test_shared_fusion_reuses_one_node_for_all_internal_nodes() -> None:
    fusion = TamariFusion(16, FusionSpec("balanced"), shared_nodes=True)
    streams = [torch.randn(2, 8, 16) for _ in range(4)]

    assert len(fusion.nodes) == 1
    assert fusion(*streams).shape == streams[0].shape
    assert set(fusion.state_dict()) == {
        "nodes.0.candidate.0.weight",
        "nodes.0.candidate.0.bias",
        "nodes.0.candidate.2.weight",
        "nodes.0.candidate.2.bias",
        "nodes.0.gate.weight",
        "nodes.0.gate.bias",
        "nodes.0.norm.weight",
        "nodes.0.norm.bias",
    }


def test_binary_fusion_supports_an_explicit_parameter_matching_bottleneck() -> None:
    fusion = BinaryTokenFusion(16, hidden_channels=11)
    left = torch.randn(2, 3, 16)
    right = torch.randn(2, 3, 16)

    assert fusion.hidden_channels == 11
    assert fusion(left, right).shape == left.shape
    assert fusion.candidate[0].weight.shape == (11, 32)
    assert fusion.candidate[2].weight.shape == (16, 11)

    with pytest.raises(ValueError, match="hidden_channels"):
        BinaryTokenFusion(16, hidden_channels=-1)


def test_matrix_semigroup_is_tree_invariant_order_sensitive_and_parameter_efficient() -> None:
    assert PublicMatrixSemigroupFusion is MatrixSemigroupFusion
    torch.manual_seed(17)
    fusion = MatrixSemigroupFusion(32, FusionSpec("balanced"))
    baseline = TamariFusion(32, FusionSpec("balanced"), shared_nodes=True)
    streams = [torch.randn(2, 8, 32) for _ in range(4)]

    outputs = {name: fusion(*streams, bracketing=name) for name in FUSION_TREES}
    reference = outputs["left_comb"]
    assert fusion.operator_size == 8
    assert all(torch.allclose(reference, value, atol=2e-6, rtol=2e-6) for value in outputs.values())
    assert not torch.allclose(reference, fusion(streams[0], streams[2], streams[1], streams[3]))
    assert sum(parameter.numel() for parameter in fusion.parameters()) < sum(
        parameter.numel() for parameter in baseline.parameters()
    )


def test_matrix_semigroup_exposes_validated_operator_gain() -> None:
    fusion = MatrixSemigroupFusion(
        16,
        FusionSpec(),
        update_scale=0.5,
        initialization_std=0.05,
    )

    assert fusion.update_scale == 0.5
    assert fusion.initialization_std == 0.05
    with pytest.raises(ValueError, match="initialization_std"):
        MatrixSemigroupFusion(16, FusionSpec(), initialization_std=0.0)


def test_matrix_semigroup_rejects_mismatched_sound_tokens() -> None:
    fusion = MatrixSemigroupFusion(16, FusionSpec())
    streams = [torch.randn(2, 8, 16) for _ in range(4)]

    with torch.no_grad(), pytest.raises(ValueError, match="matching SoundToken"):
        fusion(streams[0], streams[1][:, :-1], streams[2], streams[3])


def test_matrix_semigroup_fuses_six_streams_with_exact_missing_identities() -> None:
    torch.manual_seed(29)
    fusion = MatrixSemigroupFusion(16, FusionSpec("balanced"), operator_size=6)
    streams = [torch.randn(2, 8, 16) for _ in range(6)]
    available = torch.tensor(
        [[True, True, False, True, False, True], [True, True, False, True, False, True]]
    )

    masked = fusion.fuse_ordered(streams, available)
    reduced = fusion.fuse_ordered([streams[index] for index in (0, 1, 3, 5)])

    assert torch.allclose(masked, reduced, atol=2e-6, rtol=2e-6)


def test_matrix_semigroup_ordered_fusion_preserves_audio_api_and_validates_masks() -> None:
    fusion = MatrixSemigroupFusion(16, FusionSpec("balanced"))
    streams = [torch.randn(2, 8, 16) for _ in range(4)]
    keys = tuple(fusion.state_dict())

    ordered = fusion.fuse_ordered(streams)
    assert torch.equal(fusion(*streams), ordered)
    assert torch.equal(ordered, fusion.fuse_ordered_tensor(torch.stack(streams, dim=1)))
    assert tuple(fusion.state_dict()) == keys
    with pytest.raises(ValueError, match="at least one"):
        fusion.fuse_ordered([])
    with pytest.raises(ValueError, match="availability"):
        fusion.fuse_ordered(streams, torch.ones(2, 4))
    with pytest.raises(ValueError, match="availability"):
        fusion.fuse_ordered(streams, torch.ones(2, 3, dtype=torch.bool))


def test_matrix_semigroup_public_operator_api_supports_edge_aggregation() -> None:
    torch.manual_seed(37)
    fusion = MatrixSemigroupFusion(16, FusionSpec("balanced"), operator_size=6)
    streams = [torch.randn(2, 8, 16) for _ in range(8)]

    operators = [fusion.encode_stream(stream) for stream in streams]
    centralized = fusion.decode_operator(fusion.combine_ordered_operators(operators))
    edge_products = [
        fusion.combine_ordered_operators(operators[:4]),
        fusion.combine_ordered_operators(operators[4:]),
    ]
    hierarchical = fusion.decode_operator(fusion.combine_ordered_operators(edge_products))

    assert torch.equal(centralized, fusion.fuse_ordered(streams))
    assert torch.allclose(hierarchical, centralized, atol=2e-6, rtol=2e-6)


def test_matrix_semigroup_public_operator_api_validates_boundaries_and_missing_streams() -> None:
    fusion = MatrixSemigroupFusion(16, FusionSpec("balanced"), operator_size=6)
    tokens = torch.randn(2, 8, 16)
    operator = fusion.encode_stream(tokens)
    availability = torch.tensor([True, False])

    masked = fusion.mask_operator(operator, availability)
    identity = fusion.identity_operator_like(operator)

    assert torch.equal(masked[0], operator[0])
    assert torch.equal(masked[1], identity[1])
    with pytest.raises(ValueError, match="at least one"):
        fusion.combine_ordered_operators([])
    with pytest.raises(ValueError, match="matching operator"):
        fusion.combine_ordered_operators([operator, operator[:, :-1]])
    with pytest.raises(ValueError, match="operator boundary"):
        fusion.decode_operator(torch.randn(2, 8, 5, 6))
    with pytest.raises(ValueError, match="availability"):
        fusion.mask_operator(operator, torch.ones(2))


def test_matrix_semigroup_remains_finite_over_hundreds_of_streams() -> None:
    torch.manual_seed(31)
    fusion = MatrixSemigroupFusion(16, FusionSpec("balanced"), operator_size=6)
    streams = [torch.randn(1, 3, 16) for _ in range(256)]

    encoded = fusion.encode_stream(streams[0])
    rotation = encoded[..., :-1, :-1]
    identity = torch.eye(5).expand_as(rotation)
    output = fusion.fuse_ordered(streams)

    assert torch.isfinite(output).all()
    assert torch.equal(encoded[..., -1, :-1], torch.zeros_like(encoded[..., -1, :-1]))
    assert torch.equal(encoded[..., -1, -1], torch.ones_like(encoded[..., -1, -1]))
    assert torch.allclose(
        rotation.transpose(-1, -2) @ rotation,
        identity,
        atol=2e-5,
        rtol=2e-5,
    )


def test_rotation_conditioned_associator_is_identity_initialized_and_path_sensitive() -> None:
    associator = RotationConditionedAssociator(channels=16, rank=3)
    tokens = torch.randn(2, 8, 16)
    path = rebracketing_path("balanced", "right_comb")

    assert torch.equal(associator.transport(tokens, path), tokens)
    with torch.no_grad():
        associator.up.weight.fill_(0.1)
        associator.rotation_embedding.weight[0].zero_()
        associator.rotation_embedding.weight[1].fill_(1.0)
    balanced_to_right = associator.transport(tokens, path)
    left_to_balanced = associator.transport(tokens, rebracketing_path("left_comb", "balanced"))
    assert not torch.equal(balanced_to_right, tokens)
    assert not torch.equal(balanced_to_right, left_to_balanced)
