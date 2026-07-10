"""B4: core.py is pure tensor math — it never imports or uses torch.nn,
and its public functions run on plain tensors. This test module itself
imports only torch (not torch.nn). The check is on the AST (imports and
attribute access), not on the raw source string: docstrings legitimately
NAME torch.nn when stating this very constraint."""
import ast
import inspect

import torch

from entropylens import core


def test_core_never_imports_or_uses_torch_nn():
    tree = ast.parse(inspect.getsource(core))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torch.nn"), alias.name
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("torch.nn")
        elif isinstance(node, ast.Attribute) and node.attr == "nn":
            v = node.value
            assert not (isinstance(v, ast.Name) and v.id == "torch"), \
                "torch.nn attribute access in core"


def test_core_functions_accept_plain_tensors():
    r = core.batch_wise_entropy(torch.rand(100, 4))
    assert r.entropy.shape == (4,)
    p = core.per_example_entropy(torch.rand(100, 4), "relu")
    assert p.entropy.shape == (100,)
    est, lo, hi = core.confidence_interval(torch.rand(100, 4), n_boot=30)
    assert est.shape == lo.shape == hi.shape == (4,)


def test_public_api_exports():
    import entropylens
    for name in ["batch_wise_entropy", "per_example_entropy",
                 "confidence_interval", "EntropyTracker",
                 "EntropyActivation", "EntropyReLU", "EntropyTanh",
                 "EntropyReport", "strategy_for", "DEFAULT_STRATEGIES"]:
        assert hasattr(entropylens, name), name
