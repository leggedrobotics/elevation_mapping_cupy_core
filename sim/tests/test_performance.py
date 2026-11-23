"""Throughput of the mapping pipeline under a realistic sensor load.

The thresholds here are deliberately loose. They exist to catch a catastrophic
regression -- a kernel recompiled every frame, a stray device-to-host copy in
the hot path -- not to police normal variation between machines.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from emsim.runner import RunConfig, run

pytestmark = requires_gpu

# A 160x120 depth frame is ~19k points per frame, comparable to a downsampled
# RealSense stream feeding a robot-centric map.
MIN_INPUT_HZ = 5.0
MAX_EXPORT_MS = 500.0


@pytest.fixture(scope="module")
def benchmarked():
    cfg = RunConfig(scene="mixed", trajectory="spin", n_steps=24, resolution=0.04, map_length=8.0)
    return run(cfg, layers=("elevation", "variance", "is_valid", "inpaint"))


def test_point_cloud_ingestion_keeps_up(benchmarked):
    stats = benchmarked.timings.summary()
    points = int(np.mean(benchmarked.n_points))
    assert points > 10000, f"benchmark should be under real load, got {points} points/frame"
    assert stats["input_hz"] > MIN_INPUT_HZ, (
        f"{stats['input_hz']:.1f} Hz for {points} points/frame "
        f"(mean {stats['input_mean_ms']:.1f} ms, p95 {stats['input_p95_ms']:.1f} ms)"
    )


def test_no_per_frame_stall_after_warmup(benchmarked):
    """Kernel compilation belongs to the first frame, not every frame."""
    per_frame = benchmarked.timings.input_ms
    assert per_frame.size >= 8
    steady = per_frame[3:]
    assert steady.max() < 10 * np.median(steady) + 50.0, (
        f"a late frame stalled: max {steady.max():.1f} ms vs median {np.median(steady):.1f} ms"
    )


def test_layer_export_is_not_pathological(benchmarked):
    stats = benchmarked.timings.summary()
    assert stats["export_mean_ms"] < MAX_EXPORT_MS, (
        f"exporting layers took {stats['export_mean_ms']:.1f} ms on average"
    )


def test_throughput_report(benchmarked, capsys):
    """Not an assertion so much as a record in the test log."""
    stats = benchmarked.timings.summary()
    err = benchmarked.error("elevation", radius=2.5)
    with capsys.disabled():
        print(
            f"\n  scene={benchmarked.config.scene} "
            f"map={benchmarked.config.map_length} m @ {benchmarked.config.resolution} m "
            f"({benchmarked.cell_n}x{benchmarked.cell_n} cells)"
            f"\n  points/frame : {int(np.mean(benchmarked.n_points))}"
            f"\n  ray cast     : {stats['sensor_mean_ms']:7.2f} ms"
            f"\n  map input    : {stats['input_mean_ms']:7.2f} ms  ({stats['input_hz']:.0f} Hz)"
            f"\n  layer export : {stats['export_mean_ms']:7.2f} ms"
            f"\n  accuracy     : {err}"
        )
