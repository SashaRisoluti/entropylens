"""Demo: entropy visualization during training (entropylens.viz).

Same synthetic task as train_experiment.py. The tracker stays attached for
the whole run and tracks ONE no_grad forward per epoch, so each epoch is
one step in the plots. Output: experiments/entropy_dashboard.png with the
layer trend, per-neuron heatmaps and the top-k highest-entropy neurons.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from entropylens.hooks import EntropyTracker  # noqa: E402
from entropylens import viz  # noqa: E402


def main():
    torch.manual_seed(0)
    n = 2000
    x = torch.cat([torch.randn(n // 2, 20) + 1.0,
                   torch.randn(n // 2, 20) - 1.0])
    y = torch.cat([torch.ones(n // 2, dtype=torch.long),
                   torch.zeros(n // 2, dtype=torch.long)])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
    x, y = x[perm], y[perm]

    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                          nn.Linear(64, 32), nn.ReLU(),
                          nn.Linear(32, 2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # track only the two ReLU layers: activation entropy is the target.
    # The tracker is attached ONLY around the end-of-epoch eval forward,
    # so each epoch contributes exactly one step; records accumulate on
    # the same tracker object across epochs.
    tracker = EntropyTracker(
        model, layer_filter=lambda name, m: isinstance(m, nn.ReLU))
    for epoch in range(30):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        tracker.attach()
        with torch.no_grad():
            model(x)
        tracker.detach()

    ranked, excluded = viz.top_neurons(tracker, k=5)
    print(f"top-5 neurons at final epoch ({excluded} unreliable excluded):")
    for i, e in enumerate(ranked, 1):
        print(f"  {i}. layer '{e['layer']}' neuron {e['neuron']}: "
              f"{e['normalized']:.4f}")

    out = Path(__file__).resolve().parent / "entropy_dashboard.png"
    viz.dashboard(tracker, k=5, path=str(out))
    print(f"dashboard saved: {out}")


if __name__ == "__main__":
    main()
