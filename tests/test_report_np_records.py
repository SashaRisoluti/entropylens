"""E6: report and viz consume records produced by core_np (no torch inside
the records) exactly as they consume torch records."""
import numpy as np

from entropylens import core_np
from entropylens.record import LayerRecord
from entropylens.report import EntropyReport
from entropylens import viz


def _np_records():
    rng = np.random.default_rng(0)
    records = {}
    for name in ("l1", "l2"):
        recs = []
        for _ in range(3):
            act = rng.standard_normal((128, 8)).astype(np.float32)
            b = core_np.batch_wise_entropy(act)
            p = core_np.per_example_entropy(act, "abs")
            recs.append(LayerRecord(b, p))
        records[name] = recs
    return records


def test_summary_on_np_records():
    s = EntropyReport(_np_records()).summary()
    assert set(s) == {"l1", "l2"}
    row = s["l1"]
    assert 0.0 <= row["batch_wise_normalized_mean"] <= 1.0
    assert row["n"] == 128
    assert row["per_example_normalized_mean"] is not None


def test_depth_trend_and_raw_on_np_records():
    rep = EntropyReport(_np_records())
    trend = rep.depth_trend()
    assert [name for name, _ in trend] == ["l1", "l2"]
    raw = rep.raw()
    assert raw["l1"]["batch_wise_max_nats"] > 0


def test_viz_data_on_np_records():
    records = _np_records()
    trend = viz.layer_trend_data(records)
    assert set(trend) == {"l1", "l2"}
    means, fracs = trend["l1"]
    assert len(means) == 3
    values, reliable = viz.neuron_matrix(records, "l1")
    assert values.shape == (8, 3)
    assert reliable.dtype == bool or reliable.dtype == np.bool_


def test_top_neurons_on_np_records():
    # covers viz.top_neurons (and therefore dashboard's data path), which
    # iterates b.normalized element-wise — the .numel() hotspot
    ranked, excluded = viz.top_neurons(_np_records(), k=3)
    assert len(ranked) == 3
    assert all(0.0 <= e["normalized"] <= 1.0 for e in ranked)
    assert excluded >= 0


def test_std_and_sem_pin_ddof1():
    import math
    records = _np_records()
    row = EntropyReport(records).summary()["l1"]
    b = records["l1"][-1].batch      # summary() uses the LAST record
    p = records["l1"][-1].per_example
    assert row["batch_wise_normalized_std"] == float(b.normalized.std(ddof=1))
    vals = p.normalized[p.valid]
    assert row["per_example_sem"] == float(vals.std(ddof=1) / math.sqrt(vals.size))
