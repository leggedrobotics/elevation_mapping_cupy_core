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


@pytest.mark.parametrize(
    "weight_file, keep, fill",
    [("", False, 0.0), ("", True, 1.0), (str(_PLUGINS.parent / "weights.dat"), True, 0.0)],
    ids=["no-filter", "no-filter-keep", "filter-keep"],
)
def test_traversability_after_shift_and_clear(weight_file, keep, fill):
    # Without a filter nothing rewrites the traversability layer, so cells reset to 0
    # by a shift or clear() never pass traversability_inlier and drift compensation
    # stops. keep_traversability_without_filter resets them to 1.0 instead.
    p = parameter.Parameter(use_chainer=False, map_length=4.0, resolution=0.04, plugin_config_file=str(_PLUGINS),
                            weight_file=weight_file, keep_traversability_without_filter=keep,
                            mahalanobis_thresh=2.0, drift_compensation_variance_inlier=0.05, traversability_inlier=0.1)
    p.update()
    em = elevation_mapping.ElevationMap(p)
    assert (em.traversability_filter is None) == (weight_file == "")

    em.shift_map_xy(cp.array([10, -7]))
    trav = cp.asnumpy(em.elevation_map[3])
    assert (trav[:10] == fill).all() and (trav[:, -7:] == fill).all()
    assert (trav[10:, :-7] == 1.0).all()

    em.clear()
    assert (cp.asnumpy(em.elevation_map[3]) == fill).all()
    if weight_file:
        return

    # A flat, valid, confident map after clear(): the drift inlier count follows the fill.
    em.elevation_map[0] = 0.0
    em.elevation_map[1] = 0.01
    em.elevation_map[2] = 1.0
    sensor = np.float32([0.0, 0.0, 0.5])
    a = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    world = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1)
    points = cp.asarray((world - sensor).astype(np.float32))
    error, error_cnt = cp.zeros(1, cp.float32), cp.zeros(1, cp.float32)
    zero = cp.zeros(1, cp.float32)
    em.error_counting_kernel(em.elevation_map, points, zero, zero, cp.eye(3, dtype=cp.float32),
                             cp.asarray(sensor), cp.zeros_like(em.new_map), error, error_cnt, size=len(a))
    assert float(error_cnt[0]) == (len(a) if keep else 0)
