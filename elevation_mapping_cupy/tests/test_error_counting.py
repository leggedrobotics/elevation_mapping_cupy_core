"""The error-counting kernel's drift sums against a reference, over full and partial warps."""
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

from elevation_mapping_cupy import elevation_mapping, parameter

_PLUGINS = Path(__file__).parent.parent / "configs" / "plugin_config.yaml"


@pytest.mark.parametrize("n", [1, 31, 32, 1000, 20001])
def test_error_sums_match_the_matched_points(n):
    # The kernel also needs variance < drift_compensation_variance_inlier / 2 and
    # traversability > traversability_inlier; the map below satisfies both.
    p = parameter.Parameter(use_chainer=False, map_length=4.0, resolution=0.04, plugin_config_file=str(_PLUGINS),
                            mahalanobis_thresh=2.0, drift_compensation_variance_inlier=0.05, traversability_inlier=0.1)
    p.update()
    em = elevation_mapping.ElevationMap(p)
    # A flat, valid, confident, traversable map at height 0.
    em.elevation_map[0] = 0.0
    em.elevation_map[1] = 0.01
    em.elevation_map[2] = 1.0
    em.elevation_map[3] = 1.0
    rng = np.random.default_rng(n)
    sensor = np.float32([0.0, 0.0, 0.5])
    # A ring 0.8-1.5 m out: clear of min_valid_distance and inside the map.
    r, a = rng.uniform(0.8, 1.5, n), rng.uniform(0, 2 * np.pi, n)
    world = np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(-0.05, 0.05, n)], 1)
    # Every third point sits far off the surface: counted in layer 4, not in the sums.
    world[::3, 2] = 0.3
    points = cp.asarray((world - sensor).astype(np.float32))
    error, error_cnt = cp.zeros(1, cp.float32), cp.zeros(1, cp.float32)
    new_map = cp.zeros_like(em.new_map)
    zero = cp.zeros(1, cp.float32)
    em.error_counting_kernel(em.elevation_map, points, zero, zero, cp.eye(3, dtype=cp.float32),
                             cp.asarray(sensor), new_map, error, error_cnt, size=n)
    matched = np.abs(world[:, 2]) < 0.01 * p.mahalanobis_thresh
    assert matched.any() or n < 3
    assert float(error_cnt[0]) == matched.sum()
    np.testing.assert_allclose(float(error[0]), world[matched, 2].sum(), rtol=1e-4, atol=1e-4)
    assert float(new_map[3].sum()) == matched.sum()
    assert float(new_map[4].sum()) == n


def test_per_cell_counts_match_one_point_per_launch():
    p = parameter.Parameter(use_chainer=False, map_length=4.0, resolution=0.04, plugin_config_file=str(_PLUGINS),
                            mahalanobis_thresh=2.0, drift_compensation_variance_inlier=0.05, traversability_inlier=0.1)
    p.update()
    em = elevation_mapping.ElevationMap(p)
    em.elevation_map[0] = 0.0
    em.elevation_map[1] = 0.01
    em.elevation_map[2] = 1.0
    em.elevation_map[3] = 1.0
    rng = np.random.default_rng(0)
    # Runs of points in the same cell, as in scan or pixel order, so warps hold
    # groups of every size; every third point is off the surface (layer 4 only).
    r = np.repeat(rng.uniform(0.8, 1.5, 600), rng.integers(1, 15, 600))[:4000]
    a = np.repeat(rng.uniform(0, 2 * np.pi, 600), rng.integers(1, 15, 600))[:len(r)]
    world = np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(-0.01, 0.01, len(r))], 1)
    world[::3, 2] = 0.3
    sensor = np.float32([0.0, 0.0, 0.5])
    points = cp.asarray((world - sensor).astype(np.float32)).ravel()
    zero = cp.zeros(1, cp.float32)

    def count(chunks):
        new_map = cp.zeros_like(em.new_map)
        error, error_cnt = cp.zeros(1, cp.float32), cp.zeros(1, cp.float32)
        for c in chunks:
            em.error_counting_kernel(em.elevation_map, c, zero, zero, cp.eye(3, dtype=cp.float32),
                                     cp.asarray(sensor), new_map, error, error_cnt, size=c.size // 3)
        return cp.asnumpy(new_map), float(error[0]), float(error_cnt[0])

    serial = count([points[i * 3:(i + 1) * 3] for i in range(len(r))])
    parallel = count([points])
    assert serial[0][4].sum() == len(r)
    np.testing.assert_array_equal(parallel[0][3:5], serial[0][3:5])
    assert parallel[2] == serial[2]
    np.testing.assert_allclose(parallel[1], serial[1], rtol=1e-4, atol=1e-5)
