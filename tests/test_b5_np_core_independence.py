"""E2: core_np.py and record.py are framework-neutral — they never import
torch, tensorflow, or jax (AST check, same discipline as B4)."""
import ast
import inspect

import numpy as np

from entropylens import core_np, record

_FORBIDDEN = ("torch", "tensorflow", "jax", "flax", "keras")


def _assert_no_framework_imports(module):
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in _FORBIDDEN, (
                    f"import {alias.name} in {module.__name__}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in _FORBIDDEN, (
                f"from {node.module} import in {module.__name__}")


def test_core_np_never_imports_frameworks():
    _assert_no_framework_imports(core_np)


def test_record_never_imports_frameworks():
    _assert_no_framework_imports(record)


def test_core_np_functions_accept_plain_arrays():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 4)).astype(np.float32)
    b = core_np.batch_wise_entropy(x)
    assert b.entropy.shape == (4,)
    p = core_np.per_example_entropy(x, "relu")
    assert p.entropy.shape == (100,)
    est, lo, hi = core_np.confidence_interval(x, n_boot=30,
                                              rng=np.random.default_rng(1))
    assert est.shape == lo.shape == hi.shape == (4,)
