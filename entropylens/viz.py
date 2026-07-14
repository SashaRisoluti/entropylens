"""Post-hoc training visualization (optional extra: matplotlib).

Feed an EntropyTracker that stayed attached across training and ran one
tracked forward per epoch: each tracked forward is one "step" on the x
axis. Answers "which layer / neuron has the highest entropy, and how did
it move during training".

Honesty rules (inherited from MATH_DECISIONS.md, enforced here):
- cross-layer views use ONLY the normalized [0,1] scale (A6/Difetto 4);
  raw nats across layers of different width are not comparable;
- unreliable estimates (small N, degenerate) are masked in heatmaps and
  EXCLUDED from rankings, never shown as ordinary values (Difetto 5).

matplotlib is imported lazily: the rest of the library works without it.
"""
import math

import numpy as np

from ._compat import to_numpy as _np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "entropylens.viz requires matplotlib "
            "(pip install matplotlib).") from e
    return plt


def _records(source):
    """Accept an EntropyTracker or its .records dict."""
    rec = getattr(source, "records", source)
    if not rec:
        raise ValueError(
            "no tracked records: keep the tracker attached during training "
            "and run one forward per epoch before plotting")
    return rec


# ------------------------------------------------------------- data layer
def layer_trend_data(source):
    """Per-layer training trend on the comparable scale.

    Returns {layer_name: (means, reliable_fracs)} where means[s] is the
    mean normalized entropy over neurons at step s and reliable_fracs[s]
    the fraction of neurons whose estimate is certified by the flag.
    """
    out = {}
    for name, recs in _records(source).items():
        means = [float(_np(r.batch.normalized).mean()) for r in recs]
        fracs = [float(_np(r.batch.reliable).mean()) for r in recs]
        out[name] = (means, fracs)
    return out


def neuron_matrix(source, layer):
    """(values, reliable) numpy arrays of shape (M, steps) for one layer:
    per-neuron normalized entropy and its reliability mask."""
    recs = _records(source)
    if layer not in recs:
        raise KeyError(f"layer {layer!r} not tracked; "
                       f"tracked: {list(recs)}")
    values = np.stack([_np(r.batch.normalized) for r in recs[layer]], axis=1)
    reliable = np.stack([_np(r.batch.reliable) for r in recs[layer]], axis=1)
    return values, reliable


def top_neurons(source, k=5, step=-1):
    """The k highest-entropy neurons across ALL layers at one step.

    Ranks normalized entropy among RELIABLE neurons only; unreliable ones
    are excluded and counted (returned as the second element) — an
    unreliable high value is noise, not a finding (Difetto 5).
    Returns ([{layer, neuron, normalized}, ...] sorted desc, n_excluded).
    """
    entries, excluded = [], 0
    for name, recs in _records(source).items():
        b = recs[step].batch
        norm = _np(b.normalized)
        rel = _np(b.reliable)
        for j in range(norm.size):
            if bool(rel[j]):
                entries.append({"layer": name, "neuron": j,
                                "normalized": float(norm[j])})
            else:
                excluded += 1
    entries.sort(key=lambda e: e["normalized"], reverse=True)
    return entries[:k], excluded


# ------------------------------------------------------------ figure layer
def plot_layer_trend(source, ax=None):
    """One line per layer: mean normalized entropy vs step. Steps where
    some neurons are unreliable get an 'x' marker. The layer highest at
    the final step is named in the title."""
    plt = _plt()
    data = layer_trend_data(source)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    for name, (means, fracs) in data.items():
        steps = range(len(means))
        (line,) = ax.plot(steps, means, marker="o", markersize=3,
                          label=name)
        bad = [s for s, f in zip(steps, fracs) if f < 1.0]
        if bad:
            ax.plot(bad, [means[s] for s in bad], "x", markersize=9,
                    color=line.get_color())
    top = max(data, key=lambda n: data[n][0][-1])
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("tracked forward (step / epoch)")
    ax.set_ylabel("normalized entropy [0, 1]")
    ax.set_title(f"highest at final step: '{top}' "
                 f"({data[top][0][-1]:.3f}); x = partly unreliable")
    ax.legend(loc="best", fontsize=8)
    return fig, ax


def plot_neuron_heatmap(source, layer, ax=None):
    """Neurons x steps heatmap of normalized entropy for one layer;
    unreliable cells are greyed out, not colored."""
    plt = _plt()
    import numpy as np
    values, reliable = neuron_matrix(source, layer)
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6, max(2.0, 0.18 * values.shape[0])))
    else:
        fig = ax.figure
    masked = np.ma.masked_array(values, mask=~reliable)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap,
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xlabel("step / epoch")
    ax.set_ylabel("neuron")
    ax.set_title(f"layer '{layer}' — normalized entropy "
                 "(grey = unreliable)")
    fig.colorbar(im, ax=ax, label="normalized entropy [0, 1]")
    return fig, ax


def dashboard(source, k=5, path=None):
    """Combined figure: layer trend on top, one per-neuron heatmap per
    layer, and the top-k highest-entropy neurons at the final step.
    If path is given the figure is also saved there."""
    plt = _plt()
    recs = _records(source)
    layers = list(recs)
    n_rows = 1 + len(layers) + 1
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(9, 3.2 * (1 + len(layers)) + 1.6))
    plot_layer_trend(source, ax=axes[0])
    for ax, name in zip(axes[1:-1], layers):
        plot_neuron_heatmap(source, name, ax=ax)
    ranked, excluded = top_neurons(source, k=k)
    lines = [f"top {len(ranked)} neurons at final step "
             f"({excluded} unreliable excluded):"]
    lines += [f"  {i + 1}. layer '{e['layer']}' neuron {e['neuron']}: "
              f"{e['normalized']:.4f}" for i, e in enumerate(ranked)]
    axes[-1].axis("off")
    axes[-1].text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                  family="monospace", fontsize=9,
                  transform=axes[-1].transAxes)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig
