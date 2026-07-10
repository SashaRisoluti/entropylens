"""B1: bitwise output invariance with tracker attached. B3: autograd
untouched (bitwise-equal gradients). C3: conv layers skipped with warning."""
import warnings

import pytest
import torch
import torch.nn as nn

from entropylens.hooks import EntropyTracker


def _mlp():
    torch.manual_seed(7)
    return nn.Sequential(
        nn.Linear(20, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.Tanh(),
        nn.Linear(16, 2),
    )


def test_b1_output_bitwise_invariant():
    model = _mlp()
    x = torch.randn(64, 20, generator=torch.Generator().manual_seed(0))
    out_plain = model(x)
    tracker = EntropyTracker(model).attach()
    out_tracked = model(x)
    tracker.detach()
    out_after = model(x)
    assert torch.equal(out_plain, out_tracked), "tracker altered the output"
    assert torch.equal(out_plain, out_after), "detach left residue"


def test_b3_gradients_bitwise_invariant():
    x = torch.randn(64, 20, generator=torch.Generator().manual_seed(1))
    y = torch.randint(0, 2, (64,), generator=torch.Generator().manual_seed(2))

    def grads(with_tracker):
        model = _mlp()
        tracker = EntropyTracker(model).attach() if with_tracker else None
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        if tracker:
            tracker.detach()
        return [p.grad.clone() for p in model.parameters()]

    for g0, g1 in zip(grads(False), grads(True)):
        assert torch.equal(g0, g1), "tracker altered gradients"


def test_b3_entropy_tensors_detached():
    model = _mlp()
    with EntropyTracker(model) as tracker:
        model(torch.randn(32, 20))
    for rec in tracker.records.values():
        assert not rec[-1].batch.entropy.requires_grad


def test_records_collected_per_activation_layer():
    model = _mlp()
    with EntropyTracker(model) as tracker:
        model(torch.randn(128, 20))
    # ReLU and Tanh tracked (mapped); Linear layers tracked batch-wise only
    names = list(tracker.records)
    assert len(names) >= 2
    rec = next(iter(tracker.records.values()))[-1]
    assert rec.batch.n == 128


def test_c3_conv_skipped_with_warning():
    model = nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(), nn.Flatten(),
                          nn.Linear(4 * 6 * 6, 2))
    with pytest.warns(UserWarning, match="conv|4D|unsupported"):
        with EntropyTracker(model) as tracker:
            model(torch.randn(8, 1, 8, 8))
    # conv/relu-on-4D produced no records; the final Linear (2D) did
    for name, recs in tracker.records.items():
        assert all(r.batch is not None for r in recs)


def test_context_manager_detaches_on_exit():
    model = _mlp()
    n_hooks_before = sum(len(m._forward_hooks) for m in model.modules())
    with EntropyTracker(model):
        pass
    n_hooks_after = sum(len(m._forward_hooks) for m in model.modules())
    assert n_hooks_before == n_hooks_after
