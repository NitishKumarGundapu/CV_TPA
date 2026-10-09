import torch
from torch import nn

from vit_fewshot.data import stratified_indices
from vit_fewshot.model_hf import LoRALinear


def test_stratified_indices_are_balanced_and_disjoint():
    labels = [0, 0, 0, 1, 1, 1]
    first = stratified_indices(labels, 1, seed=1)
    second = stratified_indices(labels, 1, seed=2, excluded=set(first))
    assert len(first) == len(second) == 2
    assert set(first).isdisjoint(second)
    assert sorted(labels[index] for index in first) == [0, 1]


def test_lora_starts_as_base_layer_and_trains_adapters_only():
    base = nn.Linear(4, 3)
    inputs = torch.randn(2, 4)
    expected = base(inputs)
    layer = LoRALinear(base, rank=2, alpha=4)
    assert torch.allclose(layer(inputs), expected)
    assert not layer.weight.requires_grad
    assert layer.lora_a.requires_grad and layer.lora_b.requires_grad
