"""viz: post-hoc training visualization. Honesty rules under test:
cross-layer views use ONLY the normalized [0,1] scale (A6/Difetto 4) and
unreliable estimates are masked or excluded, never shown as ordinary
values (Difetto 5). Data functions are tested exactly; figure functions
are smoke-tested on the Agg backend."""
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import torch
import torch.nn as nn

from entropylens import viz
from entropylens.hooks import EntropyTracker


def _tracked(n=256, steps=3):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(),
                          nn.Linear(16, 8), nn.Tanh())
    tracker = EntropyTracker(model).attach()
    for s in range(steps):
        with torch.no_grad():
            model(torch.randn(n, 10, generator=torch.Generator().manual_seed(s)))
    tracker.detach()
    return tracker


def test_layer_trend_data_shapes_and_scale():
    t = _tracked(steps=3)
    data = viz.layer_trend_data(t)
    assert len(data) == 4  # all leaf modules tracked
    for name, (means, rel_frac) in data.items():
        assert len(means) == 3 and len(rel_frac) == 3
        assert all(0.0 <= v <= 1.0 for v in means)   # normalized scale only
        assert all(f == 1.0 for f in rel_frac)       # N=256: reliable


def test_neuron_matrix_and_reliability_mask():
    t = _tracked(steps=3)
    values, reliable = viz.neuron_matrix(t, "1")  # ReLU layer, M=16
    assert values.shape == (16, 3) and reliable.shape == (16, 3)
    assert bool(reliable.all())

    t_small = _tracked(n=10, steps=2)  # N=10 < 90: unreliable for B=32
    _, reliable_small = viz.neuron_matrix(t_small, "1")
    assert not bool(reliable_small.any())


def test_top_neurons_ranked_reliable_only():
    t = _tracked()
    ranked, excluded = viz.top_neurons(t, k=5)
    assert len(ranked) == 5 and excluded == 0
    vals = [r["normalized"] for r in ranked]
    assert vals == sorted(vals, reverse=True)
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(set(r) >= {"layer", "neuron", "normalized"} for r in ranked)

    t_small = _tracked(n=10, steps=1)
    ranked_small, excluded_small = viz.top_neurons(t_small, k=5)
    assert ranked_small == [] and excluded_small == 16 + 8 + 16 + 8


def test_plot_functions_smoke():
    t = _tracked()
    fig1, ax1 = viz.plot_layer_trend(t)
    assert len(ax1.get_lines()) >= 4
    fig2, ax2 = viz.plot_neuron_heatmap(t, "1")
    assert ax2.get_images(), "heatmap must contain an image"
    import matplotlib.pyplot as plt
    plt.close(fig1); plt.close(fig2)


def test_dashboard_saves_png(tmp_path):
    t = _tracked()
    out = tmp_path / "dash.png"
    fig = viz.dashboard(t, k=3, path=str(out))
    assert out.exists() and out.stat().st_size > 0
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_empty_tracker_raises():
    model = nn.Sequential(nn.Linear(4, 4))
    t = EntropyTracker(model)
    with pytest.raises(ValueError, match="no tracked records"):
        viz.layer_trend_data(t)
