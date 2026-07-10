"""Drop-in activation wrappers (mode 2 of 01_PROJECT_GOAL.md).

Each wrapper applies the wrapped activation UNCHANGED and computes entropy
on the detached output under no_grad (B2/B3). Per-example strategy comes
from the A5 table per subclass; the generic wrapper demands an explicit
strategy for unmapped modules (Difetto 3).
"""
import warnings

import torch
import torch.nn as nn

from . import core, strategies


class EntropyActivation(nn.Module):
    def __init__(self, activation, strategy=None, *, bins=32,
                 q_lo=0.01, q_hi=0.99, name=None, track_history=False):
        super().__init__()
        self.activation = activation
        if strategy is None:
            strategy = strategies.strategy_for(activation)  # may raise
        if strategy not in core._PER_EXAMPLE_STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        self.strategy = strategy
        self.bins = bins
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.name = name or type(activation).__name__
        self.track_history = track_history
        self.history: list[tuple[core.BatchEntropy, core.PerExampleEntropy]] = []
        self.last_batch: core.BatchEntropy | None = None
        self.last_per_example: core.PerExampleEntropy | None = None
        self._warned = False

    def forward(self, x):
        y = self.activation(x)
        with torch.no_grad():
            act = y.detach()
            if act.dim() > 2:
                if not self._warned:
                    self._warned = True
                    warnings.warn(
                        f"entropylens: {self.name} received a {act.dim()}D "
                        "activation; unsupported/unvalidated in v1, entropy "
                        "skipped (MATH_DECISIONS.md D5, Q4).")
            elif act.dim() == 2:
                self.last_batch = core.batch_wise_entropy(
                    act, bins=self.bins, q_lo=self.q_lo, q_hi=self.q_hi)
                self.last_per_example = core.per_example_entropy(
                    act, self.strategy)
                if self.track_history:
                    self.history.append(
                        (self.last_batch, self.last_per_example))
        return y

    def extra_repr(self):
        return f"strategy={self.strategy}, bins={self.bins}"


def _subclass(cls_name, act_factory):
    class _Wrapper(EntropyActivation):
        def __init__(self, **kwargs):
            super().__init__(act_factory(), **kwargs)
    _Wrapper.__name__ = _Wrapper.__qualname__ = cls_name
    return _Wrapper


EntropyReLU = _subclass("EntropyReLU", nn.ReLU)
EntropyReLU6 = _subclass("EntropyReLU6", nn.ReLU6)
EntropyGELU = _subclass("EntropyGELU", nn.GELU)
EntropySiLU = _subclass("EntropySiLU", nn.SiLU)
EntropyLeakyReLU = _subclass("EntropyLeakyReLU", nn.LeakyReLU)
EntropySigmoid = _subclass("EntropySigmoid", nn.Sigmoid)
EntropySoftmax = _subclass("EntropySoftmax", lambda: nn.Softmax(dim=-1))
EntropyTanh = _subclass("EntropyTanh", nn.Tanh)
EntropySoftsign = _subclass("EntropySoftsign", nn.Softsign)
