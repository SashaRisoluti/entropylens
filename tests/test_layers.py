"""B2: drop-in layers leave output and gradients bitwise unchanged.
Each subclass carries its table strategy (A5 applied, Difetto 3 closed)."""
import pytest
import torch
import torch.nn as nn

from entropylens import layers


def test_b2_output_bitwise_identical():
    torch.manual_seed(3)
    lin1, lin2 = nn.Linear(12, 24), nn.Linear(24, 3)
    plain = nn.Sequential(lin1, nn.ReLU(), lin2)
    tracked = nn.Sequential(lin1, layers.EntropyReLU(), lin2)
    x = torch.randn(50, 12, generator=torch.Generator().manual_seed(4))
    assert torch.equal(plain(x), tracked(x))


def test_b2_gradients_bitwise_identical():
    x = torch.randn(50, 12, generator=torch.Generator().manual_seed(5))

    def grads(act):
        torch.manual_seed(6)
        model = nn.Sequential(nn.Linear(12, 24), act, nn.Linear(24, 3))
        model(x).sum().backward()
        return [p.grad.clone() for p in model.parameters()]

    for g0, g1 in zip(grads(nn.ReLU()), grads(layers.EntropyReLU())):
        assert torch.equal(g0, g1)


def test_subclass_strategies_follow_table():
    assert layers.EntropyReLU().strategy == "relu"
    assert layers.EntropyGELU().strategy == "relu"
    assert layers.EntropySiLU().strategy == "relu"
    assert layers.EntropyLeakyReLU().strategy == "relu"
    assert layers.EntropySigmoid().strategy == "relu"
    assert layers.EntropySoftmax().strategy == "relu"
    assert layers.EntropyTanh().strategy == "abs"


def test_generic_wrapper_requires_strategy_for_unmapped():
    with pytest.raises(Exception, match="explicit"):
        layers.EntropyActivation(nn.Identity())
    wrapped = layers.EntropyActivation(nn.Identity(), strategy="abs")
    assert wrapped.strategy == "abs"


def test_results_available_after_forward():
    act = layers.EntropyReLU(track_history=True)
    model = nn.Sequential(nn.Linear(10, 8), act)
    model(torch.randn(200, 10))
    assert act.last_batch is not None and act.last_batch.n == 200
    assert act.last_per_example is not None
    model(torch.randn(200, 10))
    assert len(act.history) == 2


def test_conv_shaped_input_skipped_with_warning():
    act = layers.EntropyReLU()
    with pytest.warns(UserWarning, match="unsupported"):
        y = act(torch.randn(4, 3, 8, 8))
    assert y.shape == (4, 3, 8, 8)
    assert act.last_batch is None
