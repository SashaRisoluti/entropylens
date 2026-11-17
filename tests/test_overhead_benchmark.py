"""Smoke test: benchmarks/overhead.py runs end to end on its tiny
configuration and writes well-formed outputs. No assertion on timing or
memory values (they depend on the machine and would be flaky)."""
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "benchmarks" / "overhead.py"
MODES = ["baseline", "tracked_every_step", "tracked_eval_forward"]


def test_tiny_run_writes_csv_and_markdown(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--tiny", "--out-dir", str(tmp_path)],
        check=True, timeout=300)

    # read_bytes: read_text would translate CRLF and hide it from the check below
    csv_text = (tmp_path / "results.csv").read_bytes().decode("utf-8")
    rows = list(csv.DictReader(csv_text.splitlines()))
    assert [r["mode"] for r in rows] == MODES  # one small model, one batch
    assert {r["model"] for r in rows} == {"small"}
    for column in ("batch_size", "threads", "timed_steps", "repeats",
                   "time_median_ms", "time_q1_ms", "time_q3_ms",
                   "peak_rss_median_mb", "history_bytes"):
        assert all(r[column] != "" for r in rows), column
    assert rows[0]["history_bytes"] == "0"  # baseline never attaches a tracker
    assert int(rows[1]["history_bytes"]) > 0

    md = (tmp_path / "results.md").read_bytes().decode("utf-8")
    for heading in ("## Environment", "## Method", "## Results",
                    "## Not measured"):
        assert heading in md, heading
    assert md.count("| small | 256 |") == 3  # one row in each results table

    for text in (csv_text, md):  # outputs are committed: no local paths, LF only
        assert "\r" not in text
        assert str(Path.home()) not in text and str(tmp_path) not in text
