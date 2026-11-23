"""The one-launch overlap clear and the persistent drift scratch give the old results."""
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

from elevation_mapping_cupy import elevation_mapping, parameter

_PLUGINS = Path(__file__).parent.parent / "configs" / "plugin_config.yaml"


def _map(**overrides):
    p = parameter.Parameter(use_chainer=False, map_length=4.0, resolution=0.04, plugin_config_file=str(_PLUGINS),
                            **overrides)
    p.update()
    return elevation_mapping.ElevationMap(p)


def _reference_clear(em, t):
    """The former slicing implementation of clear_overlap_map."""
    height_min = t[2] - em.param.overlap_clear_range_z
    height_max = t[2] + em.param.overlap_clear_range_z
    near_map = em.elevation_map[:, em.cell_min : em.cell_max, em.cell_min : em.cell_max]
    valid_idx = ~cp.logical_or(near_map[0] < height_min, near_map[0] > height_max)
    near_map[0] = cp.where(valid_idx, near_map[0], 0.0)
    near_map[1] = cp.where(valid_idx, near_map[1], em.initial_variance)
    near_map[2] = cp.where(valid_idx, near_map[2], 0.0)
    valid_idx = ~cp.logical_or(near_map[5] < height_min, near_map[5] > height_max)
    near_map[5] = cp.where(valid_idx, near_map[5], 0.0)
    near_map[6] = cp.where(valid_idx, near_map[6], 0.0)
    em.elevation_map[:, em.cell_min : em.cell_max, em.cell_min : em.cell_max] = near_map


@pytest.mark.parametrize("clear_range_xy, clear_range_z, tz", [(2.0, 0.5, 0.3), (6.0, 2.0, -1.25), (0.5, 0.1, 0.0)])
def test_overlap_clear_matches_the_slicing_version(clear_range_xy, clear_range_z, tz):
    em = _map(overlap_clear_range_xy=clear_range_xy, overlap_clear_range_z=clear_range_z)
    rng = np.random.default_rng(0)
    m = rng.uniform(-3, 3, em.elevation_map.shape).astype(np.float32)
    t = cp.asarray([0.1, -0.2, tz], cp.float32)
    lo, hi = cp.asnumpy(t[2] - clear_range_z), cp.asnumpy(t[2] + clear_range_z)
    # Heights exactly on the bounds, one float either side, and NaNs.
    for layer in (0, 5):
        flat = m[layer].ravel()
        picks = rng.choice(flat.size, 60, replace=False)
        flat[picks[:10]], flat[picks[10:20]] = lo, hi
        flat[picks[20:30]] = np.nextafter(lo, np.float32(-np.inf))
        flat[picks[30:40]] = np.nextafter(hi, np.float32(np.inf))
        flat[picks[40:60]] = np.nan
    em.elevation_map[...] = cp.asarray(m)
    em.clear_overlap_map(t)
    got = cp.asnumpy(em.elevation_map)
    em.elevation_map[...] = cp.asarray(m)
    _reference_clear(em, t)
    np.testing.assert_array_equal(got, cp.asnumpy(em.elevation_map))


def test_overlap_clear_follows_a_changed_range():
    em = _map(overlap_clear_range_z=0.5)
    t = cp.asarray([0.0, 0.0, 0.0], cp.float32)
    em.elevation_map[0] = 1.0
    em.clear_overlap_map(t)  # 1.0 is outside +-0.5: cleared
    c = em.cell_n // 2
    assert float(em.elevation_map[0, c, c]) == 0.0
    em.param.overlap_clear_range_z = 2.0
    em.elevation_map[0] = 1.0
    em.clear_overlap_map(t)  # now inside: kept
    assert float(em.elevation_map[0, c, c]) == 1.0


def test_drift_sums_restart_every_frame():
    """After each frame the persistent sums hold that frame's error counting alone."""
    em = _map(enable_drift_compensation=True)
    rng = np.random.default_rng(0)
    n = 20000
    # Close to the sensor, so fused cells get a variance below the drift-inlier limit.
    pts = cp.asarray(np.stack([rng.uniform(-0.4, 0.4, n), rng.uniform(-0.4, 0.4, n), np.full(n, -0.5)], 1)
                     .astype(np.float32))
    eye, zero = cp.eye(3, dtype=cp.float32), cp.zeros(1, cp.float32)
    counts = []
    for _ in range(4):
        before = em.elevation_map.copy()
        # Copies: add_points_kernel writes cell indices back into its input points.
        em.update_map_with_kernel(pts.copy(), [], eye, cp.asarray([0, 0, 0.5], cp.float32), 1.0, 1.0)
        error, count = cp.zeros(1, cp.float32), cp.zeros(1, cp.float32)
        em.error_counting_kernel(before, pts.copy(), zero, zero, eye, cp.asarray([0, 0, 0.5], cp.float32),
                                 cp.zeros_like(em.new_map), error, count, size=n)
        assert float(em._drift_count[0]) == float(count[0])
        np.testing.assert_allclose(float(em._drift_error[0]), float(error[0]), rtol=1e-4, atol=1e-5)
        counts.append(float(count[0]))
    assert counts[0] == 0 and counts[-1] > 1000  # empty map first, then matches
