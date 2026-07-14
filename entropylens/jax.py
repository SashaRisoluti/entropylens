"""JAX/Flax capture (checklist E5): entropy from flax capture_intermediates.

Design (MATH_DECISIONS.md D8): capture_intermediates returns every module's
__call__ output from the SAME forward pass; the 'intermediates' collection
never enters the grad path, and outputs/grads are bitwise-invariant with
and without capture — including under jax.jit (exp8 G1-G3, see D9 for the
empirical scoping). Captured Dense outputs are PRE-activation (flax
activations are functions, not modules), so no strategy table is possible:
per-example entropy requires an explicit strategy_map by path (D3: no
silent default); a 2D capture without an entry is tracked batch-wise only
WITH a one-time warning per path (hooks/C2 convention, D8 as amended).
Structure notes (exp8): '__call__' markers recur at every nesting level
and are stripped recursively; leaves are 1-tuples of arrays (first element
taken; multi-output modules are out of v1 scope).
"""
import warnings

import numpy as np

from . import core_np
from .record import LayerRecord

try:
    import flax.linen  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "entropylens.jax requires jax and flax: "
        "pip install 'entropylens[jax]'") from e


def capture(model, variables, x, **apply_kwargs):
    """One forward pass with intermediates: returns (outputs, intermediates)."""
    y, mutated = model.apply(
        variables, x, capture_intermediates=True,
        mutable=["intermediates"], **apply_kwargs)
    return y, mutated["intermediates"]


def _flatten(intermediates):
    """{'Dense_0': {'__call__': (arr,)}} -> {'Dense_0': arr}; root skipped."""
    flat = {}

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path if k == "__call__" else path + (k,))
        elif isinstance(node, tuple):
            if node and path:   # root __call__ arrives with empty path
                flat["/".join(path)] = node[0]
        elif path:
            flat["/".join(path)] = node

    walk(dict(intermediates), ())
    return flat


class JaxEntropyTracker:
    """Entropy tracker for flax intermediates.

    Feed it the intermediates from capture() (or a raw flax
    capture_intermediates collection): records are appended per observe()
    call, one per 2D captured array. Per-example entropy requires an
    explicit strategy_map entry per path; without it the path is tracked
    batch-wise only (one-time warning per path).
    """

    def __init__(self, *, strategy_map=None, bins=32, q_lo=0.01, q_hi=0.99):
        self.strategy_map = dict(strategy_map or {})
        self.bins = bins
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.records = {}
        self._warned = set()

    def observe(self, intermediates):
        """Record entropy for every 2D captured array; appends one record
        per path per call."""
        for name, arr in _flatten(intermediates).items():
            act = np.asarray(arr)
            if act.ndim > 2:
                if name not in self._warned:
                    self._warned.add(name)
                    warnings.warn(
                        f"entropylens: capture '{name}' is {act.ndim}D "
                        f"(conv feature map?); unsupported/unvalidated in "
                        f"v1, skipped (MATH_DECISIONS.md D5, Q4).")
                continue
            if act.ndim < 2:
                continue
            strat = self.strategy_map.get(name)
            if strat is None and name not in self._warned:
                self._warned.add(name)
                warnings.warn(
                    f"entropylens: no strategy_map entry for capture "
                    f"'{name}'; tracking batch-wise entropy only. Flax "
                    f"captures are PRE-activation, so no default is safe "
                    f"(MATH_DECISIONS.md D8) — add strategy_map="
                    f"{{'{name}': 'relu'|'abs'}} for per-example tracking.")
            batch = core_np.batch_wise_entropy(
                act, bins=self.bins, q_lo=self.q_lo, q_hi=self.q_hi)
            per_ex = (core_np.per_example_entropy(act, strat)
                      if strat else None)
            self.records.setdefault(name, []).append(
                LayerRecord(batch, per_ex))
        return self

    def report(self):
        from .report import EntropyReport
        return EntropyReport(self.records)
