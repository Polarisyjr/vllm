# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config import SpeculativeConfig
from vllm.v1.drafter.config import DrafterConfig


def test_target_of_drafter_engines_names_no_drafter():
    spec = SpeculativeConfig(
        num_speculative_tokens=3, drafter_endpoint=["a:7700", "b:7700"]
    )
    assert spec.method == "remote"
    # Nothing of a local drafter: no draft KV, no drafter rows in the batch.
    assert not spec.use_eagle() and not spec.uses_draft_model()
    assert spec.max_num_new_slots_for_drafting == 0

    with pytest.raises(ValueError, match="drafter engines say"):
        SpeculativeConfig(
            method="eagle3",
            model="some/head",
            num_speculative_tokens=3,
            drafter_endpoint="a:7700",
        )
    with pytest.raises(ValueError, match="requires drafter_endpoint"):
        SpeculativeConfig(method="remote", num_speculative_tokens=3)


def test_exported_layers_name_the_compiled_graphs():
    def compile_hash(layers):
        spec = SpeculativeConfig(num_speculative_tokens=3, drafter_endpoint="a:7700")
        spec.drafter_hidden_layers = layers
        return spec.compute_hash()

    # A target that exports hidden states returns them from its forward pass.
    assert compile_hash(None) == compile_hash(())
    assert compile_hash((2, 18, 33)) != compile_hash(None)
    assert compile_hash((2, 18, 33)) != compile_hash((2, 10, 18, 26, 34))


def test_engine_device():
    def config(device):
        return DrafterConfig(
            method="draft_model", model="m", num_speculative_tokens=3, device=device
        )

    assert config("cuda:1").device == "cuda:1"
    with pytest.raises(ValueError, match="cuda:<index>"):
        config("cuda")
