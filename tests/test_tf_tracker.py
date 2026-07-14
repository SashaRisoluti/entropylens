"""E3: TF tracker — observation by auxiliary model, user model untouched."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from tensorflow import keras  # noqa: E402

from entropylens.tf import TFEntropyTracker  # noqa: E402


def _mlp():
    keras.utils.set_random_seed(7)
    inp = keras.Input((20,))
    h = keras.layers.Dense(32, activation="relu", name="d1")(inp)
    h = keras.layers.Dense(16, activation="tanh", name="d2")(h)
    out = keras.layers.Dense(2, name="d3")(h)
    return keras.Model(inp, out)


def _x():
    return np.random.default_rng(0).standard_normal((128, 20)).astype(np.float32)


def test_e3_user_model_output_invariant():
    model, x = _mlp(), _x()
    y0 = model(x, training=False)
    tracker = TFEntropyTracker(model)
    tracker.observe(x)
    y1 = model(x, training=False)
    assert bool(tf.reduce_all(tf.equal(y0, y1)))


def test_records_schema_matches_torch_contract():
    model, x = _mlp(), _x()
    tracker = TFEntropyTracker(model)
    tracker.observe(x)
    tracker.observe(x)
    assert set(tracker.records) == {"d1", "d2", "d3"}
    recs = tracker.records["d1"]
    assert len(recs) == 2
    assert recs[0].batch.entropy.shape == (32,)
    assert recs[0].batch.n == 128
    assert recs[0].per_example is not None          # relu mapped
    assert recs[0].per_example.strategy == "relu"


def test_unmapped_layer_warns_and_tracks_batchwise_only():
    model, x = _mlp(), _x()
    tracker = TFEntropyTracker(model)
    with pytest.warns(UserWarning, match="batch-wise entropy only"):
        tracker.observe(x)
    assert tracker.records["d3"][0].per_example is None   # linear Dense


def test_strategies_override_by_name():
    model, x = _mlp(), _x()
    tracker = TFEntropyTracker(model, strategies_override={"d3": "abs"})
    tracker.observe(x)
    assert tracker.records["d3"][0].per_example.strategy == "abs"


def test_report_and_viz_integration():
    from entropylens import viz
    model, x = _mlp(), _x()
    tracker = TFEntropyTracker(model)
    tracker.observe(x)
    s = tracker.report().summary()
    assert 0.0 <= s["d1"]["batch_wise_normalized_mean"] <= 1.0
    values, reliable = viz.neuron_matrix(tracker.records, "d1")
    assert values.shape == (32, 1)


def test_subclassed_model_raises():
    class Sub(keras.Model):
        def __init__(self):
            super().__init__()
            self.d = keras.layers.Dense(2)

        def call(self, x):
            return self.d(x)

    with pytest.raises(ValueError, match="subclassed"):
        TFEntropyTracker(Sub()).observe(_x())


def test_conv_4d_skipped_with_warning():
    keras.utils.set_random_seed(7)
    inp = keras.Input((8, 8, 1))
    h = keras.layers.Conv2D(4, 3, name="c1")(inp)
    h = keras.layers.Flatten(name="f")(h)
    out = keras.layers.Dense(2, name="d")(h)
    model = keras.Model(inp, out)
    x = np.random.default_rng(0).standard_normal((16, 8, 8, 1)).astype(np.float32)
    tracker = TFEntropyTracker(model)
    with pytest.warns(UserWarning, match="unsupported/unvalidated in v1"):
        tracker.observe(x)
    assert "c1" not in tracker.records
    assert "d" in tracker.records
