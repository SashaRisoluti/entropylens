"""Activation -> negative-handling strategy table (checklist A5 / C2).

This table is the deliverable that closes Difetto 3: the strategy is
derived from the SEMANTICS of the wrapped activation, never from a fixed
default. Full rationale and numeric evidence: MATH_DECISIONS.md D3.

Lookup is by EXACT type: a subclass may change semantics, so it must be
mapped (or given a strategy) explicitly.
"""
import torch.nn as nn

# type -> (strategy, theoretical motivation)
DEFAULT_STRATEGIES = {
    nn.ReLU: ("relu", "outputs are nonnegative by construction; the "
              "strategy is a no-op"),
    nn.ReLU6: ("relu", "outputs are nonnegative by construction; the "
               "strategy is a no-op"),
    nn.Softplus: ("relu", "outputs are strictly positive; the strategy "
                  "is a no-op"),
    nn.Sigmoid: ("relu", "outputs lie in (0,1); the strategy is a no-op"),
    nn.Softmax: ("relu", "output is already a probability distribution; "
                 "renormalization is the identity, entropy is computed "
                 "directly on it"),
    nn.GELU: ("relu", "negative range is bounded (>= -0.170) while the "
              "positive range is unbounded: sign encodes 'nearly off', "
              "not a direction. CAVEAT: for inputs concentrated near 0 "
              "(sigma~1) the discarded negative mass reaches ~15%; it "
              "falls quickly with input scale (2.6% at sigma=3)."),
    nn.SiLU: ("relu", "negative range is bounded (>= -0.278) while the "
              "positive range is unbounded: sign encodes 'nearly off'. "
              "CAVEAT: discarded negative mass ~24% at sigma=1, "
              "~7% at sigma=3."),
    nn.LeakyReLU: ("relu", "negative outputs are the input scaled by the "
                   "leak factor (default 0.01): treated as 'off'; the "
                   "discarded mass is proportional to the slope. For "
                   "large slopes (>0.1) consider an explicit choice."),
    nn.Tanh: ("abs", "sign is a direction, magnitude is the intensity; "
              "~50% of output mass is negative, so discarding it would "
              "destroy half the signal"),
    nn.Softsign: ("abs", "sign is a direction, magnitude is the "
                  "intensity (same argument as Tanh)"),
    nn.Hardtanh: ("abs", "sign is a direction, magnitude is the "
                  "intensity (same argument as Tanh)"),
}


class UnmappedActivationError(ValueError):
    pass


def strategy_for(module):
    """Strategy for an activation module, by exact type (A5 table).

    Raises UnmappedActivationError for unmapped types (e.g. Linear or
    custom modules): ~50% of a generic linear output's mass is negative
    and the sign semantics are unknown, so no default is safe
    (MATH_DECISIONS.md D3) — the caller must choose explicitly.
    """
    entry = DEFAULT_STRATEGIES.get(type(module))
    if entry is None:
        known = ", ".join(t.__name__ for t in DEFAULT_STRATEGIES)
        raise UnmappedActivationError(
            f"no default negative-handling strategy for "
            f"{type(module).__name__}: pass an explicit strategy "
            f"('relu' = negatives mean 'off', 'abs' = sign is a "
            f"direction). Mapped types: {known}.")
    return entry[0]
