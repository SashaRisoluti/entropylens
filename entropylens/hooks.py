"""Non-invasive tracking via forward hooks (mode 1: observe an existing
model without modifying it).

Invariants (checklist B1/B3): the hook never modifies the output (it
returns None) and computes everything under torch.no_grad() on detached
tensors. Conv-shaped activations (>2D) are skipped with a warning (C3:
spatial assumption unvalidated in v1, see MATH_DECISIONS.md D5).
"""
import warnings

import torch

from . import core, strategies
from .record import LayerRecord


class EntropyTracker:
    """Attach entropy tracking to an existing model without modifying it.

    tracker = EntropyTracker(model).attach()
    output = model(x)          # bitwise identical to untracked forward
    tracker.report()           # aggregated view (report.EntropyReport)
    tracker.detach()

    strategies_override: dict mapping module name (str, as in
    model.named_modules()) or module type to a per-example strategy
    ('relu'|'abs'). Modules with no table entry and no override get
    batch-wise tracking only (with a one-time warning).
    """

    def __init__(self, model, *, bins=32, q_lo=0.01, q_hi=0.99,
                 strategies_override=None, layer_filter=None):
        self.model = model
        self.bins = bins
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.overrides = dict(strategies_override or {})
        self.layer_filter = layer_filter
        self.records: dict[str, list[LayerRecord]] = {}
        self._handles = []
        self._warned: set[str] = set()

    def _strategy(self, name, module):
        if name in self.overrides:
            return self.overrides[name]
        if type(module) in self.overrides:
            return self.overrides[type(module)]
        try:
            return strategies.strategy_for(module)
        except strategies.UnmappedActivationError:
            if name not in self._warned:
                self._warned.add(name)
                warnings.warn(
                    f"entropylens: no per-example strategy for layer "
                    f"'{name}' ({type(module).__name__}); tracking "
                    "batch-wise entropy only. Pass strategies_override "
                    "to enable per-example tracking.")
            return None

    def _hook(self, name):
        def hook(module, args, output):
            if not torch.is_tensor(output):
                return None
            with torch.no_grad():
                act = output.detach()
                if act.dim() > 2:
                    if name not in self._warned:
                        self._warned.add(name)
                        warnings.warn(
                            f"entropylens: layer '{name}' produced a "
                            f"{act.dim()}D activation (conv feature map?); "
                            "unsupported/unvalidated in v1, layer skipped "
                            "(MATH_DECISIONS.md D5, Q4).")
                    return None
                if act.dim() < 2:
                    return None
                batch = core.batch_wise_entropy(
                    act, bins=self.bins, q_lo=self.q_lo, q_hi=self.q_hi)
                strat = self._strategy(name, module)
                per_ex = (core.per_example_entropy(act, strat)
                          if strat else None)
                self.records.setdefault(name, []).append(
                    LayerRecord(batch, per_ex))
            return None
        return hook

    def attach(self):
        if self._handles:
            raise RuntimeError("already attached")
        for name, module in self.model.named_modules():
            if name == "" or len(list(module.children())) > 0:
                continue  # leaves only
            if self.layer_filter is not None and not self.layer_filter(name, module):
                continue
            self._handles.append(
                module.register_forward_hook(self._hook(name)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        return self

    def report(self):
        from .report import EntropyReport
        return EntropyReport(self.records)

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()
        return False
