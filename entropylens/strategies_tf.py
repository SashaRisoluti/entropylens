"""Keras activation -> negative-handling strategy table (checklist E4).

Mirror of entropylens/strategies.py (MATH_DECISIONS.md D3) for Keras 3
(tensorflow>=2.16). Lookup levels:
1. exact layer class (ReLU, LeakyReLU, Softmax);
2. the layer's `activation` attribute by function identity
   (Dense(activation=...), layers.Activation(...)).
`activations.linear` (Dense default) is deliberately unmapped: ~50% of a
generic linear output's mass is negative with unknown sign semantics (D3)
-> the caller must choose explicitly. Tables are built lazily so importing
this module without using it never touches tensorflow internals.
"""
from .strategies import UnmappedActivationError

_LAYER_TABLE = None
_FN_TABLE = None


def _ensure_tables():
    global _LAYER_TABLE, _FN_TABLE
    if _LAYER_TABLE is not None:
        return
    from tensorflow.keras import activations, layers

    _LAYER_TABLE = {
        layers.ReLU: ("relu", "outputs are nonnegative by construction; "
                      "the strategy is a no-op"),
        layers.LeakyReLU: ("relu", "negative outputs are the input scaled "
                           "by the leak factor: treated as 'off'; the "
                           "discarded mass is proportional to the slope. "
                           "For large slopes (>0.1) consider an explicit "
                           "choice."),
        layers.Softmax: ("relu", "output is already a probability "
                         "distribution; renormalization is the identity, "
                         "entropy is computed directly on it"),
    }
    _FN_TABLE = {
        activations.relu: ("relu", "outputs are nonnegative by "
                           "construction; the strategy is a no-op"),
        activations.relu6: ("relu", "outputs are nonnegative by "
                            "construction; the strategy is a no-op"),
        activations.softplus: ("relu", "outputs are strictly positive; "
                               "the strategy is a no-op"),
        activations.sigmoid: ("relu", "outputs lie in (0,1); the strategy "
                              "is a no-op"),
        activations.softmax: ("relu", "output is already a probability "
                              "distribution; renormalization is the "
                              "identity, entropy is computed directly on it"),
        activations.gelu: ("relu", "negative range is bounded (>= -0.170) "
                           "while the positive range is unbounded: sign "
                           "encodes 'nearly off'. CAVEAT: for inputs "
                           "concentrated near 0 the discarded negative "
                           "mass reaches ~15% (D3)."),
        activations.silu: ("relu", "negative range is bounded (>= -0.278) "
                           "while the positive range is unbounded: sign "
                           "encodes 'nearly off'. CAVEAT: discarded "
                           "negative mass ~24% at sigma=1 (D3)."),
        activations.tanh: ("abs", "sign is a direction, magnitude is the "
                           "intensity; ~50% of output mass is negative, so "
                           "discarding it would destroy half the signal"),
        activations.softsign: ("abs", "sign is a direction, magnitude is "
                               "the intensity (same argument as Tanh)"),
    }
    # swish may alias silu in Keras 3; add defensively if distinct
    swish = getattr(activations, "swish", None)
    if swish is not None and swish not in _FN_TABLE:
        _FN_TABLE[swish] = _FN_TABLE[activations.silu]


def strategy_for_keras(layer):
    """Strategy for a Keras layer, by exact class or activation identity.

    Raises UnmappedActivationError when neither matches (e.g. Dense with
    the default linear activation) — no silent default (D3).
    """
    _ensure_tables()
    entry = _LAYER_TABLE.get(type(layer))
    if entry is None:
        act = getattr(layer, "activation", None)
        if act is not None:
            entry = _FN_TABLE.get(act)
    if entry is None:
        known = ", ".join(sorted(
            [t.__name__ for t in _LAYER_TABLE]
            + [f.__name__ for f in _FN_TABLE]))
        raise UnmappedActivationError(
            f"no default negative-handling strategy for Keras layer "
            f"'{getattr(layer, 'name', '?')}' ({type(layer).__name__}): "
            f"pass an explicit strategy ('relu' = negatives mean 'off', "
            f"'abs' = sign is a direction). Mapped: {known}.")
    return entry[0]
