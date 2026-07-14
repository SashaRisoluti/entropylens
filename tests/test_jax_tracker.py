"""E5: JAX/Flax capture — same-pass intermediates, explicit strategy_map."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
flax = pytest.importorskip("flax")
import jax.numpy as jnp  # noqa: E402
from flax import linen as nn  # noqa: E402

from entropylens.jax import JaxEntropyTracker, capture  # noqa: E402


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(32)(x))
        return nn.Dense(2)(x)


def _setup():
    model = MLP()
    x = jax.random.normal(jax.random.PRNGKey(0), (128, 20))
    params = model.init(jax.random.PRNGKey(7), x)
    return model, params, x


def test_capture_output_bitwise_invariant():
    model, params, x = _setup()
    y_plain = model.apply(params, x)
    y_cap, inter = capture(model, params, x)
    assert bool(jnp.array_equal(y_plain, y_cap))
    assert "Dense_0" in inter


def test_records_schema():
    model, params, x = _setup()
    _, inter = capture(model, params, x)
    tracker = JaxEntropyTracker(strategy_map={"Dense_0": "abs"})
    with pytest.warns(UserWarning, match="batch-wise entropy only"):
        tracker.observe(inter)
    assert set(tracker.records) == {"Dense_0", "Dense_1"}
    rec = tracker.records["Dense_0"][0]
    assert rec.batch.entropy.shape == (32,)
    assert rec.batch.n == 128
    assert rec.per_example.strategy == "abs"
    assert tracker.records["Dense_1"][0].per_example is None  # no map entry


def test_missing_strategy_warns_once_per_path():
    import warnings as w
    model, params, x = _setup()
    _, inter = capture(model, params, x)
    tracker = JaxEntropyTracker()
    with w.catch_warnings(record=True) as rec:
        w.simplefilter("always")
        tracker.observe(inter)
        tracker.observe(inter)
    msgs = [str(r.message) for r in rec
            if "batch-wise entropy only" in str(r.message)]
    # two unmapped paths (Dense_0, Dense_1), each warned exactly once
    assert len(msgs) == 2


def test_report_and_viz_integration():
    from entropylens import viz
    model, params, x = _setup()
    _, inter = capture(model, params, x)
    tracker = JaxEntropyTracker(strategy_map={"Dense_0": "relu",
                                              "Dense_1": "relu"})
    tracker.observe(inter)
    tracker.observe(inter)
    s = tracker.report().summary()
    assert 0.0 <= s["Dense_0"]["batch_wise_normalized_mean"] <= 1.0
    values, _ = viz.neuron_matrix(tracker.records, "Dense_0")
    assert values.shape == (32, 2)


def test_root_call_not_recorded():
    model, params, x = _setup()
    _, inter = capture(model, params, x)
    tracker = JaxEntropyTracker(strategy_map={"Dense_0": "relu",
                                              "Dense_1": "relu"})
    tracker.observe(inter)
    assert "" not in tracker.records and "__call__" not in tracker.records


def test_nested_module_paths_flatten_recursively():
    class Inner(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(8)(x)

    class Outer(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = Inner()(x)
            return nn.Dense(2)(x)

    model = Outer()
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 20))
    params = model.init(jax.random.PRNGKey(7), x)
    _, inter = capture(model, params, x)
    tracker = JaxEntropyTracker(strategy_map={
        "Inner_0/Dense_0": "relu", "Inner_0": "relu", "Dense_0": "relu"})
    tracker.observe(inter)
    # __call__ markers stripped at EVERY level: nested Dense appears under
    # its full path, the Inner module's own output under its path
    assert "Inner_0/Dense_0" in tracker.records
    assert "Inner_0" in tracker.records
    assert "Dense_0" in tracker.records          # Outer's own Dense
    assert "__call__" not in tracker.records
