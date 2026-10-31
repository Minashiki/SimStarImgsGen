"""Memory acceptance for the frame loop. The 5/20 comparison is the check."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.runtime_env import runtime_environment

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "measure_frame_rss.py"
REPORT_PATH = ROOT / "artifacts" / "m3_memory.json"
GIB = 1024**3
MIB = 1024**2


def _child(n_frames: int) -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(n_frames)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    payload = None
    for line in completed.stdout.splitlines():
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            payload = json.loads(text)
    if payload is None:
        raise AssertionError(completed.stdout)
    assert payload["n_frames"] == n_frames
    assert payload["frames"] == n_frames
    return payload


@pytest.mark.slow
def test_frame_loop_peak_rss_is_bounded():
    measured = {n_frames: _child(n_frames) for n_frames in (10, 5, 20)}
    peaks = {n_frames: int(item["peak_wset"]) for n_frames, item in measured.items()}
    margin = (3 * GIB - peaks[10]) / (3 * GIB)
    report = {
        "environment": runtime_environment(),
        "root_seed": 20261029,
        "seed_note": "覆盖 YAML 中的 dataset.seed",
        "peaks_bytes": {str(n_frames): peaks[n_frames] for n_frames in (10, 5, 20)},
        "peaks_gib": {str(n_frames): peaks[n_frames] / GIB for n_frames in (10, 5, 20)},
        "limit_10_bytes": 3 * GIB,
        "span_limit_bytes": 300 * MIB,
        "ten_frame_margin_vs_3gib": margin,
        "worker_budget_note": (
            "workers 按每个 worker 3 GiB 估算。"
            f"10 帧峰值 {peaks[10] / GIB:.3f} GiB，相对 3 GiB 的余量约 {100 * margin:.1f}%。"
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert peaks[10] <= 3 * GIB
    assert abs(peaks[20] - peaks[5]) < 300 * MIB
