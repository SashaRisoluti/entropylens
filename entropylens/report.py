"""Aggregation and presentation (checklist A6 / Difetto 4).

Default view is ALWAYS the normalized [0,1] scale, where batch-wise and
per-example entropies are comparable by construction (each divided by its
own attainable bound). Raw nats exist only behind .raw(), which returns
each value WITH the bound it is relative to, so different scales are
never shown side by side implicitly.
"""
import math

import torch


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
                "batch_wise_normalized_mean": float(b.normalized.mean()),
                "batch_wise_normalized_std": float(b.normalized.std())
                if b.normalized.numel() > 1 else 0.0,
                "reliable_fraction": float(b.reliable.float().mean()),
                "n": b.n,
                "per_example_normalized_mean": None,
                "per_example_sem": None,
            }
            p = rec.per_example
            if p is not None and bool(p.valid.any()):
                vals = p.normalized[p.valid]
                row["per_example_normalized_mean"] = float(vals.mean())
                # SEM of the batch mean: the only honest per-example
                # uncertainty (each row's H is deterministic, D4)
                row["per_example_sem"] = (
                    float(vals.std() / math.sqrt(vals.numel()))
                    if vals.numel() > 1 else 0.0)
                row["per_example_invalid_rows"] = int((~p.valid).sum())
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
                "batch_wise_nats": float(b.entropy.mean()),
                "batch_wise_max_nats": math.log(min(b.n, b.bins)),
                "per_example_nats": None,
            }
            p = rec.per_example
            if p is not None and bool(p.valid.any()):
                row["per_example_nats"] = float(p.entropy[p.valid].mean())
                row["per_example_max_nats"] = math.log(p.m)
            out[name] = row
        return out
