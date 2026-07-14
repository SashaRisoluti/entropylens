"""TensorFlow/Keras tracker (checklist E3): observation by auxiliary model.

Design (MATH_DECISIONS.md D8): the user's model is NEVER modified — no
hooks, no wrappers. A lazily built auxiliary keras.Model maps the model's
inputs to every eligible layer output; observe(x) runs it with
training=False and records entropy from the numpy activations via core_np.
Invariance (D9): the user's forward/backward never sees the tracker
(exp7 G2/G3), but the observation is an EXTRA inference-mode pass —
BatchNorm/Dropout behave as in inference (exp7 G5). Subclassed models
(no symbolic model.inputs) are unsupported in v1. Known edge (v1, out of
scope): `layer.output` raises on layers with multiple inbound nodes
(shared layers) or multiple outputs — such models need a layer_filter. Nested
keras.Model layers are skipped with a one-time warning (top-level layers only
in v1).
"""
import warnings

import numpy as np

from . import core_np, strategies_tf
from .record import LayerRecord
from .strategies import UnmappedActivationError

try:
    from tensorflow import keras
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "entropylens.tf requires tensorflow>=2.16: "
        "pip install 'entropylens[tf]'") from e


class TFEntropyTracker:
    """Entropy tracker for functional/Sequential Keras models.

    Unlike the torch EntropyTracker (which records on every forward while
    attached), records are appended ONLY when observe(x) is called — one
    extra inference-mode forward per call. Memory note: one observe pass
    materializes ALL eligible layer outputs at once (dict-output auxiliary
    model), unlike torch hooks which see one activation at a time. Call
    observe() outside GradientTape (it works inside, but the auxiliary
    pass would be taped needlessly).
    """

    def __init__(self, model, *, bins=32, q_lo=0.01, q_hi=0.99,
                 strategies_override=None, layer_filter=None):
        self.model = model
        self.bins = bins
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.overrides = dict(strategies_override or {})
        self.layer_filter = layer_filter
        self.records = {}
        self._aux = None
        self._by_name = None
        self._warned = set()

    def _build_aux(self):
        inputs = getattr(self.model, "inputs", None)
        if not inputs:
            raise ValueError(
                "TFEntropyTracker requires a built functional or Sequential "
                "Keras model (model.inputs must exist); subclassed models "
                "have no symbolic layer outputs and are unsupported in v1.")
        outputs = {}
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            if isinstance(layer, keras.Model):
                if layer.name not in self._warned:
                    self._warned.add(layer.name)
                    warnings.warn(
                        f"entropylens: layer '{layer.name}' is a nested "
                        f"keras.Model; v1 tracks top-level layers only and "
                        f"cannot capture a submodel's output through the "
                        f"auxiliary model — skipped. Track the submodel "
                        f"with its own TFEntropyTracker if needed.")
                continue
            if (self.layer_filter is not None
                    and not self.layer_filter(layer.name, layer)):
                continue
            outputs[layer.name] = layer.output
        if not outputs:
            raise ValueError("TFEntropyTracker: no eligible layers to track")
        # Keras 3: model.inputs is a list even for single-input models;
        # passing the bare list makes every aux call emit a structure
        # warning when observe() receives a bare array (exp7)
        aux_inputs = inputs[0] if len(inputs) == 1 else inputs
        return keras.Model(aux_inputs, outputs, name="entropylens_aux")

    def _strategy(self, name, layer):
        if name in self.overrides:
            return self.overrides[name]
        if type(layer) in self.overrides:
            return self.overrides[type(layer)]
        try:
            return strategies_tf.strategy_for_keras(layer)
        except UnmappedActivationError:
            if name not in self._warned:
                self._warned.add(name)
                warnings.warn(
                    f"entropylens: no per-example strategy for layer "
                    f"'{name}' ({type(layer).__name__}); tracking "
                    f"batch-wise entropy only. Pass strategies_override "
                    f"to enable per-example tracking.")
            return None

    def observe(self, x):
        """One inference-mode observation pass; appends one record per layer."""
        if self._aux is None:
            self._aux = self._build_aux()
            self._by_name = {l.name: l for l in self.model.layers}
        outs = self._aux(x, training=False)
        by_name = self._by_name
        for name, t in outs.items():
            act = np.asarray(t)
            if act.ndim > 2:
                if name not in self._warned:
                    self._warned.add(name)
                    warnings.warn(
                        f"entropylens: layer '{name}' produced a "
                        f"{act.ndim}D activation (conv feature map?); "
                        f"unsupported/unvalidated in v1, layer skipped "
                        f"(MATH_DECISIONS.md D5, Q4).")
                continue
            if act.ndim < 2:
                continue
            batch = core_np.batch_wise_entropy(
                act, bins=self.bins, q_lo=self.q_lo, q_hi=self.q_hi)
            strat = self._strategy(name, by_name[name])
            per_ex = (core_np.per_example_entropy(act, strat)
                      if strat else None)
            self.records.setdefault(name, []).append(
                LayerRecord(batch, per_ex))
        return self

    def report(self):
        from .report import EntropyReport
        return EntropyReport(self.records)
