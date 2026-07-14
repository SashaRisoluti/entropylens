"""Aggregation and presentation (checklist A6 / Difetto 4).

Default view is ALWAYS the normalized [0,1] scale, where batch-wise and
per-example entropies are comparable by construction (each divided by its
own attainable bound). Raw nats exist only behind .raw(), which returns
each value WITH the bound it is relative to, so different scales are
never shown side by side implicitly.

Numeric note (E6 refactor): aggregation moved from torch to numpy — Bessel
std semantics preserved (ddof=1, pinned by test), but values can differ
from the pre-refactor torch reductions by <= 1 ulp float32 (summation
order, same caveat class as D7); reliable_fraction now accumulates in
float64 (strictly more accurate).
"""
import math

from ._compat import to_numpy as _np


class EntropyReport:
    def __init__(self, records):
        # records: dict[str, list[LayerRecord]] in registration order
        self._records = records

    def _last(self):
        return {name: recs[-1] for name, recs in self._records.items()
                if recs}

    def summary(self):
        out = {}
        for name, rec in self._last().items():
            b = rec.batch
            row = {
                "batch_wise_normalized_mean": float(_np(b.normalized).mean()),
                "batch_wise_normalized_std": float(_np(b.normalized).std(ddof=1))
                if _np(b.normalized).size > 1 else 0.0,
                "reliable_fraction": float(_np(b.reliable).mean()),
                "n": b.n,
                "per_example_normalized_mean": None,
                "per_example_sem": None,
            }
            p = rec.per_example
            if p is not None and bool(_np(p.valid).any()):
                vals = _np(p.normalized)[_np(p.valid)]
                row["per_example_normalized_mean"] = float(vals.mean())
                # SEM of the batch mean: the only honest per-example
                # uncertainty (each row's H is deterministic, D4)
                row["per_example_sem"] = (
                    float(vals.std(ddof=1) / math.sqrt(vals.size))
                    if vals.size > 1 else 0.0)
                row["per_example_invalid_rows"] = int((~_np(p.valid)).sum())
            out[name] = row
        return out

    def depth_trend(self):
        """(layer_name, batch_wise_normalized_mean) in registration order:
        normalized values are the only cross-layer-comparable ones (A6)."""
        return [(name, row["batch_wise_normalized_mean"])
                for name, row in self.summary().items()]

    def raw(self):
        """Raw nats, each value paired with the bound it is relative to.
        Values with different bounds are NOT directly comparable
        (Difetto 4): compare .summary() normalized values instead."""
        out = {}
        for name, rec in self._last().items():
            b = rec.batch
            row = {
                "batch_wise_nats": float(_np(b.entropy).mean()),
                "batch_wise_max_nats": math.log(min(b.n, b.bins)),
                "per_example_nats": None,
            }
            p = rec.per_example
            if p is not None and bool(_np(p.valid).any()):
                row["per_example_nats"] = float(_np(p.entropy)[_np(p.valid)].mean())
                row["per_example_max_nats"] = math.log(p.m)
            out[name] = row
        return out
