"""Visibility cleanup: rays from the sensor to a point lower the validity of older cells they pass under."""
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

from elevation_mapping_cupy import elevation_mapping, parameter

SENSOR = (0.0, 0.0, 0.5)
_PLUGINS = Path(__file__).parent.parent / "configs" / "plugin_config.yaml"


@pytest.fixture
def elmap():
    # cleanup_cos_thresh 0 disables the normal check, so the test does not depend on
    # the normals of a narrow pillar.
    p = parameter.Parameter(use_chainer=False, map_length=8.0, resolution=0.04, max_ray_length=10.0,
                            cleanup_cos_thresh=0.0, enable_visibility_cleanup=True,
                            plugin_config_file=str(_PLUGINS))
    p.update()
    em = elevation_mapping.ElevationMap(p)
    # A 0.4 m pillar 1.1 m in front of the sensor, then aged so the cleanup may touch it.
    rng = np.random.default_rng(0)
    n = 3000
    pillar = np.stack([rng.uniform(1.04, 1.16, n), rng.uniform(-0.06, 0.06, n), np.full(n, 0.4)], 1)
    _fuse(em, pillar)
    for _ in range(20):
        em.update_time()
    return em


def _fuse(em, world_points):
    local = (np.asarray(world_points, np.float32) - np.float32(SENSOR)).astype(np.float32)
    em.update_map_with_kernel(cp.asarray(local), [], cp.eye(3, dtype=cp.float32),
                              cp.asarray(SENSOR, cp.float32), 0.0, 0.0)


def _pillar_validity(em):
    c = em.cell_n // 2
    i = int(round(1.1 / em.resolution))
    # The map's row index runs along x and its column along y (see get_idx).
    return float(cp.asnumpy(em.elevation_map[2, c + i, c]))


def _rays_to(x, z, n=400):
    rng = np.random.default_rng(1)
    return np.stack([np.full(n, x), rng.uniform(-0.01, 0.01, n), np.full(n, z)], 1)


def test_rays_under_an_old_obstacle_clear_it(elmap):
    before = _pillar_validity(elmap)
    assert before > 0.5
    _fuse(elmap, _rays_to(2.5, 0.0))  # the ray passes the pillar at ~0.28 m
    assert _pillar_validity(elmap) < before - 0.5


def test_rays_to_points_outside_the_map_clear_cells_before_they_leave(elmap):
    before = _pillar_validity(elmap)
    _fuse(elmap, _rays_to(6.0, -0.5))  # 2 m beyond the map edge; passes the pillar at ~0.32 m
    assert _pillar_validity(elmap) < before - 0.5


def test_rays_of_invalid_points_change_nothing(elmap):
    _fuse(elmap, _rays_to(2.5, 0.0, n=1))  # settle the per-update bookkeeping first
    before = cp.asnumpy(elmap.elevation_map).copy()
    # Far above the height range: the point is rejected, and its ray must not touch any
    # cell either (a marched ray would set upper bounds on the unobserved cells it crosses).
    _fuse(elmap, _rays_to(2.5, 3.0))
    np.testing.assert_array_equal(cp.asnumpy(elmap.elevation_map), before)
