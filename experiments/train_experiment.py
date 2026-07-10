"""D1 (checklist): repeat the training experiment of 04_ARCHITECTURE_DRAFT
with the corrected estimator, comparing against the old defective one on
the SAME run, to see whether the Difetto 2 fix changes the empirical
reading (old observation: batch-wise entropy ROSE from ~2.0 to ~2.3 nats).
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from entropylens.hooks import EntropyTracker  # noqa: E402


def old_minmax_entropy(values, bins=32):
    """The defective old-draft estimator (Difetto 2), local copy for
    comparison only — NOT part of the library."""
    x = values.float().flatten()
    lo, hi = x.min().item(), x.max().item()
    if hi <= lo:
        return 0.0
    counts = torch.histc(x, bins=bins, min=lo, max=hi)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def main():
    torch.manual_seed(0)
    # synthetic binary classification: two gaussian blobs
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

    print(f"{'epoch':>5} {'loss':>8} {'H_new(norm)':>12} {'H_old(nats)':>12}")
    for epoch in range(30):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

        with torch.no_grad(), EntropyTracker(model) as t:
            model(x)
        rep = t.report().summary()
        first_relu = next(k for k in rep if "1" in k)  # first ReLU layer
        h_new = rep[first_relu]["batch_wise_normalized_mean"]
        # old estimator on the same activations, first hidden layer
        h1 = model[1](model[0](x)).detach()
        h_old = sum(old_minmax_entropy(h1[:, j]) for j in
                    range(h1.shape[1])) / h1.shape[1]
        print(f"{epoch:>5} {loss.item():>8.4f} {h_new:>12.4f} {h_old:>12.4f}")


if __name__ == "__main__":
    main()
