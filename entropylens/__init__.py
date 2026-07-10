"""entropylens: entropy-based statistical interpretability for PyTorch.

Measures statistical properties of activation distributions
(concentration vs spread), NOT what a neuron semantically represents.
Formulas: MATH_DECISIONS.md. Non-invasiveness guarantees: the tracked
forward pass is bitwise identical and autograd is never touched.
"""
from .core import (BatchEntropy, PerExampleEntropy, batch_wise_entropy,
                   confidence_interval, per_example_entropy,
                   reliability_threshold, REASON_LABELS)
from .hooks import EntropyTracker
from .layers import (EntropyActivation, EntropyGELU, EntropyLeakyReLU,
                     EntropyReLU, EntropyReLU6, EntropySigmoid,
                     EntropySiLU, EntropySoftmax, EntropySoftsign,
                     EntropyTanh)
from .report import EntropyReport
from .strategies import (DEFAULT_STRATEGIES, UnmappedActivationError,
                         strategy_for)

__version__ = "0.1.0"
