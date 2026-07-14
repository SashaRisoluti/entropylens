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


def test_layer_filter_excludes_layer():
    model, x = _mlp(), _x()
    tracker = TFEntropyTracker(model, layer_filter=lambda n, l: n != "d1")
    tracker.observe(x)
    assert "d1" not in tracker.records and "d2" in tracker.records


def test_strategies_override_by_class_and_name_precedence():
    model, x = _mlp(), _x()
    tracker = TFEntropyTracker(model, strategies_override={
        keras.layers.Dense: "abs", "d1": "relu"})
    tracker.observe(x)
    assert tracker.records["d1"][0].per_example.strategy == "relu"
    assert tracker.records["d3"][0].per_example.strategy == "abs"


def test_unmapped_warning_fires_once_across_observes():
    import warnings as w
    model, x = _mlp(), _x()
    tracker = TFEntropyTracker(model)
    with w.catch_warnings(record=True) as rec:
        w.simplefilter("always")
        tracker.observe(x)
        tracker.observe(x)
    msgs = [str(r.message) for r in rec
            if "batch-wise entropy only" in str(r.message)]
    assert len(msgs) == 1


def test_nested_submodel_skipped_with_warning():
    keras.utils.set_random_seed(7)
    sub_in = keras.Input((20,))
    sub_out = keras.layers.Dense(8, activation="relu", name="sub_d")(sub_in)
    sub = keras.Model(sub_in, sub_out, name="feature_extractor")
    inp = keras.Input((20,))
    h = sub(inp)
    out = keras.layers.Dense(2, name="head")(h)
    model = keras.Model(inp, out)
    tracker = TFEntropyTracker(model)
    with pytest.warns(UserWarning, match="nested"):
        tracker.observe(_x())
    assert "feature_extractor" not in tracker.records
    assert "head" in tracker.records


def test_multi_input_model_observe():
    keras.utils.set_random_seed(7)
    a = keras.Input((4,), name="a")
    b = keras.Input((4,), name="b")
    h = keras.layers.Concatenate(name="cat")([a, b])
    out = keras.layers.Dense(2, name="d")(h)
    model = keras.Model([a, b], out)
    rng = np.random.default_rng(0)
    xa = rng.standard_normal((32, 4)).astype(np.float32)
    xb = rng.standard_normal((32, 4)).astype(np.float32)
    tracker = TFEntropyTracker(model)
    tracker.observe([xa, xb])
    assert "cat" in tracker.records and "d" in tracker.records
