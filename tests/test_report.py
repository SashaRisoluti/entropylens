"""A6: the report exposes ONLY the common normalized [0,1] scale by
default; raw nats are available under an accessor that names the bound
they are relative to (non-comparable scales made explicit)."""
import torch
import torch.nn as nn

from entropylens.hooks import EntropyTracker


def _tracked_report(n=256):
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(),
                          nn.Linear(32, 16), nn.Tanh())
    with EntropyTracker(model) as t:
        model(torch.randn(n, 10, generator=torch.Generator().manual_seed(12)))
    return t.report()


def test_summary_is_normalized_and_bounded():
    rep = _tracked_report()
    s = rep.summary()
    assert len(s) >= 2
    for name, row in s.items():
        assert 0.0 <= row["batch_wise_normalized_mean"] <= 1.0
        if row["per_example_normalized_mean"] is not None:
            assert 0.0 <= row["per_example_normalized_mean"] <= 1.0
        assert "reliable_fraction" in row
        assert "per_example_sem" in row


def test_depth_trend_ordered_and_normalized():
    rep = _tracked_report()
    trend = rep.depth_trend()
    assert [t[0] for t in trend] == list(rep.summary().keys())
    assert all(0.0 <= v <= 1.0 for _, v in trend)


def test_raw_values_require_explicit_accessor_and_carry_bounds():
    rep = _tracked_report()
    raw = rep.raw()
    for name, row in raw.items():
        assert "batch_wise_nats" in row and "batch_wise_max_nats" in row
        # the two metrics never appear side by side without their bounds
        if row.get("per_example_nats") is not None:
            assert "per_example_max_nats" in row


def test_unreliable_layers_marked():
    torch.manual_seed(13)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    with EntropyTracker(model) as t:
        model(torch.randn(10, 4))  # N=10 < 90: unreliable for B=32
    s = t.report().summary()
    for row in s.values():
        assert row["reliable_fraction"] == 0.0
