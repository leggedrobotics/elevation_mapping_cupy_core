"""The per-measurement calls must not wait for the GPU.

A hidden read-back (a device scalar in an `if`, boolean-mask indexing, `.get()`)
blocks the caller until every queued kernel has finished, so CPU and GPU never
overlap. Each test queues ~100 ms of GPU work and then makes the call: a call
that returns well before the GPU is done did not wait.
"""

from pathlib import Path
import time

import cupy as cp
import numpy as np
import pytest

from elevation_mapping_cupy import elevation_mapping, parameter
from elevation_mapping_cupy.plugins.min_filter import MinFilter

_CONFIG_DIR = Path(__file__).parent.parent / "configs"
_SPIN = cp.RawKernel(
    r"""
extern "C" __global__ void spin(long long cycles) {
  long long start = clock64();
  while (clock64() - start < cycles) {}
}""",
    "spin",
)


@pytest.fixture
def elmap():
    p = parameter.Parameter(
        use_chainer=False,
        weight_file=str(_CONFIG_DIR / "weights.dat"),
        plugin_config_file=str(_CONFIG_DIR / "plugin_config.yaml"),
    )
    p.subscriber_cfg["front_cam"]["channels"] = ["rgb"]
    p.subscriber_cfg["front_cam"]["fusion"] = ["color"]
    p.subscriber_cfg["front_cam"]["data_type"] = "image"
    p.update()
    return elevation_mapping.ElevationMap(p)


def _busy_cycles(ms):
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    _SPIN((1,), (1,), (np.int64(10_000_000),))
    end.record()
    end.synchronize()
    return np.int64(10_000_000 * ms / cp.cuda.get_elapsed_time(start, end))


def _cpu_ms_behind_gpu_work(fn, cycles):
    """CPU time of fn() while ~100 ms of GPU work is queued ahead of it."""
    fn()  # compile kernels outside the measurement
    cp.cuda.Device().synchronize()
    _SPIN((1,), (1,), (cycles,))
    t0 = time.perf_counter()
    fn()
    ms = (time.perf_counter() - t0) * 1e3
    cp.cuda.Device().synchronize()
    return ms


def test_measurement_calls_do_not_wait_for_the_gpu(elmap):
    cycles = _busy_cycles(100)
    rng = np.random.default_rng(0)
    n = 20000
    points = cp.asarray(
        np.stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), rng.uniform(-0.6, -0.4, n)], 1).astype(np.float32)
    )
    eye = np.eye(3, dtype=np.float32)
    pos = [0.0]

    def move_to():
        pos[0] += 0.37
        elmap.move_to(np.array([pos[0], 0.0, 0.5], np.float32), cp.asarray(eye))

    def update(noise):
        return lambda: elmap.update_map_with_kernel(
            points, [], cp.asarray(eye), cp.asarray([pos[0], 0.0, 0.5], cp.float32), noise, 0.0
        )

    look = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float32)
    image = cp.asarray(rng.uniform(0, 255, (1, 120, 160)).astype(np.float32))
    K = cp.asarray([[100, 0, 80], [0, 100, 60], [0, 0, 1]], cp.float32)

    def input_image():
        elmap.input_image(
            image, ["rgb"], None, look, np.array([pos[0], 0.5, 0.0], np.float32), K,
            cp.zeros(5, cp.float32), "radtan", 120, 160,
        )

    calls = {
        "move_to": move_to,
        "move": lambda: elmap.move(np.array([0.12, -0.07, 0.01], np.float32)),
        "update_map_with_kernel": update(0.0),
        "update_map_with_kernel, drift compensation active": update(1e3),
        "input_image": input_image,
    }
    assert elmap.param.enable_drift_compensation
    for name, fn in calls.items():
        assert _cpu_ms_behind_gpu_work(fn, cycles) < 50, f"{name} waited for queued GPU work"


def test_center_cpu_matches_the_device_center_and_float32_rounding(elmap):
    res = np.float32(elmap.resolution)
    for x, y in [(0.0, 0.0), (0.5 * res, -0.5 * res), (1.5 * res, 2.5 * res), (3.21, -7.04), (-0.05, 1e-3)]:
        before = cp.asarray(elmap.center_cpu)
        position = np.array([x, y, 0.3], np.float32)
        # The device computation the host copy replaces.
        delta = cp.asarray(position) - before
        expected = before.copy()
        expected[:2] += cp.around(delta[:2] / elmap.resolution) * elmap.resolution
        expected[2] += delta[2]
        elmap.move_to(position, cp.eye(3, dtype=cp.float32))
        np.testing.assert_array_equal(elmap.center_cpu, cp.asnumpy(expected))
        np.testing.assert_array_equal(elmap.center_cpu, cp.asnumpy(elmap.center))
    elmap.move(np.array([0.26, -0.11, 0.02], np.float32))
    np.testing.assert_array_equal(elmap.center_cpu, cp.asnumpy(elmap.center))


def test_min_filter_readout_does_not_wait_for_the_gpu(elmap):
    rng = np.random.default_rng(0)
    points = cp.asarray(np.stack([rng.uniform(-3, 3, 5000), rng.uniform(-3, 3, 5000), rng.uniform(-0.6, -0.4, 5000)], 1).astype(np.float32))
    elmap.update_map_with_kernel(points, [], cp.eye(3, dtype=cp.float32), cp.asarray([0, 0, 0.5], cp.float32), 0.0, 0.0)

    def readout():
        elmap.plugin_manager.reset_layers()
        elmap.get_map_with_name_ref("min_filter", return_cupy=True)

    assert _cpu_ms_behind_gpu_work(readout, _busy_cycles(100)) < 50, "min_filter waited for queued GPU work"


def _min_filter_reference(h, valid, d, iteration_n):
    """Each pass fills invalid cells with the minimum valid value within d cells of the
    previous pass and stops once every cell is valid. Like the kernel, neighbours are
    flat-index offsets (they wrap across rows) and border cells are never sources."""
    n = h.shape[0]
    pad = d * (n + 1)
    shifts = [n * dy + dx for dy in range(-d, d + 1) for dx in range(-d, d + 1)]
    m, k = h.copy(), valid.copy()
    for _ in range(iteration_n):
        src = np.where(k > 0.5, m, np.inf)
        src[[0, -1], :] = src[:, [0, -1]] = np.inf
        flat = np.pad(src.ravel(), pad, constant_values=np.inf)
        mins = np.min([flat[pad + s : pad + s + n * n] for s in shifts], axis=0).reshape(n, n)
        fill = (valid < 0.5) & np.isfinite(mins)
        m, k = np.where(fill, mins, m), np.where(fill, np.float32(0.6), k)
        if (k > 0.5).all():
            break
    return np.where(k > 0.5, m, np.nan)


@pytest.mark.parametrize("dilation_size, iteration_n", [(1, 5), (1, 30), (2, 30)])
def test_min_filter_is_deterministic_and_spreads_dilation_size_per_pass(dilation_size, iteration_n):
    rng = np.random.default_rng(1)
    n = 64
    h = rng.uniform(-1, 1, (n, n)).astype(np.float32)
    valid = (rng.uniform(size=(n, n)) < 0.02).astype(np.float32)
    em = cp.zeros((3, n, n), cp.float32)
    em[0], em[2] = cp.asarray(h), cp.asarray(valid)
    f = MinFilter(cell_n=n, dilation_size=dilation_size, iteration_n=iteration_n)
    expected = _min_filter_reference(h, valid, dilation_size, iteration_n)
    for _ in range(5):
        np.testing.assert_array_equal(cp.asnumpy(f(em, [], None, [])), expected)
