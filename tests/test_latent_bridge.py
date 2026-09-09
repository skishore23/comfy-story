from __future__ import annotations

import pytest
import torch

from comfy_story.latent_bridge import LatentHistoryBridge
from comfy_story.ltx_bridge import LTXLatentHistoryBridge
from comfy_story.minimax_h3_bridge import MiniMaxH3LatentHistoryBridge


def test_ltx_bridge_keeps_historical_class_and_state_dict_contract() -> None:
    bridge = LTXLatentHistoryBridge()

    assert type(bridge).__module__ == "comfy_story.ltx_bridge"
    assert type(bridge).__qualname__ == "LTXLatentHistoryBridge"
    assert tuple(bridge.state_dict()) == (
        "fusion.input_norm.weight",
        "fusion.input_norm.bias",
        "fusion.encoder.weight",
        "fusion.encoder.bias",
        "fusion.decoder.weight",
        "fusion.decoder.bias",
        "fusion.output_norm.weight",
        "fusion.output_norm.bias",
        "residual.weight",
    )


def test_ltx_wrapper_matches_generic_bridge_for_the_same_weights() -> None:
    torch.manual_seed(101)
    generic = LatentHistoryBridge(128, operator_size=4)
    wrapper = LTXLatentHistoryBridge(operator_size=4)
    wrapper.load_state_dict(generic.state_dict(), strict=True)
    history = torch.randn(1, 3, 128, 1, 2, 2)

    generic_output = generic(history, anchor=history[:, -1])
    wrapper_output = wrapper(history, anchor=history[:, -1])

    torch.testing.assert_close(wrapper_output.core_operator, generic_output.core_operator)
    torch.testing.assert_close(wrapper_output.core_latent, generic_output.core_latent)
    torch.testing.assert_close(wrapper_output.per_reference_rmse, generic_output.per_reference_rmse)
    torch.testing.assert_close(wrapper_output.uncertainty, generic_output.uncertainty)


def test_minimax_bridge_uses_the_native_24_channel_boundary() -> None:
    bridge = MiniMaxH3LatentHistoryBridge(operator_size=4)
    history = torch.randn(1, 8, 24, 1, 2, 2)

    output = bridge(history, anchor=history[:, -1])

    assert bridge.fusion.channels == 24
    assert output.core_latent.shape == (1, 24, 1, 2, 2)
    assert output.core_operator.shape == (1, 4, 4, 4)


def test_minimax_bridge_rejects_an_ltx_history() -> None:
    bridge = MiniMaxH3LatentHistoryBridge(operator_size=4)
    history = torch.randn(1, 8, 128, 1, 2, 2)

    with pytest.raises(ValueError, match="history channels must equal 24"):
        bridge(history, anchor=history[:, -1])
