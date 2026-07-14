"""Shared record container: the contract report.py and viz.py consume.

Framework-neutral (no torch/tensorflow/jax imports) so every backend
tracker — hooks.EntropyTracker, tf.TFEntropyTracker, jax.JaxEntropyTracker —
can emit the same schema. Fields are duck-typed: `batch` is a BatchEntropy
(torch core or core_np), `per_example` a PerExampleEntropy or None.
"""
from dataclasses import dataclass


@dataclass
class LayerRecord:
    batch: object
    per_example: object
