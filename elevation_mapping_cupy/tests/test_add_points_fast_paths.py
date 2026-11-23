"""The float cell index and the warp-summed fusion in add_points_kernel give the old results."""
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

from elevation_mapping_cupy import elevation_mapping, parameter
from elevation_mapping_cupy.kernels.custom_kernels import map_utils

_PLUGINS = Path(__file__).parent.parent / "configs" / "plugin_config.yaml"


@pytest.mark.parametrize("resolution, n", [(0.04, 200), (0.04, 201), (0.05, 400), (0.1, 101), (0.03, 1000)])
def test_cell_index_matches_the_double_expression(resolution, n):
    kernel = cp.ElementwiseKernel(
        "float32 a", "int32 diff",
        f"int ref = a / {resolution!r} + 0.5 * {n}; diff = get_x_idx(a, 0.0f) - ref;",
        "cell_index_check",
        preamble=map_utils(resolution, n, n, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    rng = np.random.default_rng(n)
    # Cell boundaries and their float neighbours, where the float estimate is
    # ambiguous and the double expression decides, plus random coordinates.
    k = np.arange(-2 * n, 2 * n)
    edges = ((k - 0.5 * n) * resolution).astype(np.float32)
    up, down = np.float32(np.inf), np.float32(-np.inf)
    # Every boundary and the two floats either side of it.
    near = np.concatenate([edges] + [np.nextafter(edges, d) for d in (up, down)]
                          + [np.nextafter(np.nextafter(edges, d), d) for d in (up, down)])
    tiny = np.float32([0.0, -0.0, 1e-30, -1e-30, 1e-8, -1e-8])
    wide = rng.uniform(-n * resolution * 2, n * resolution * 2, 2_000_000).astype(np.float32)
    a = cp.asarray(np.concatenate([near, tiny, wide]))
    assert int(cp.count_nonzero(kernel(a))) == 0


def _map(**overrides):
    p = parameter.Parameter(use_chainer=False, map_length=4.0, resolution=0.04, plugin_config_file=str(_PLUGINS),
                            enable_visibility_cleanup=False, **overrides)
    p.update()
    return elevation_mapping.ElevationMap(p)


def _fuse(em, points, one_at_a_time):
    """add_points_kernel on a fresh map; returns the new_map it accumulates."""
    zero = cp.zeros(1, cp.float32)
    eye, t = cp.eye(3, dtype=cp.float32), cp.asarray([0.0, 0.0, 0.5], cp.float32)
    new_map = cp.zeros_like(em.new_map)
    elevation = em.elevation_map.copy()
    p = cp.asarray(points).ravel()
    chunks = [p[i * 3:(i + 1) * 3] for i in range(len(points))] if one_at_a_time else [p]
    for c in chunks:
        em.add_points_kernel(zero, zero, eye, t, em.normal_map, c, elevation, new_map, size=c.size // 3)
    return cp.asnumpy(new_map)


def test_warp_summed_fusion_matches_one_point_per_launch():
    em = _map()
    rng = np.random.default_rng(0)
    # Runs of consecutive points in the same cell (as in scan or pixel order),
    # interleaved with isolated points, so warps hold groups of every size.
    n = 4000
    cells = np.repeat(rng.uniform(-1.4, 1.4, (n // 8, 2)), rng.integers(1, 15, n // 8), axis=0)[:n]
    cells[::7] = rng.uniform(-1.4, 1.4, (len(cells[::7]), 2))
    points = np.column_stack([cells + rng.uniform(-0.01, 0.01, cells.shape), rng.uniform(-0.55, -0.45, len(cells))])
    points = points.astype(np.float32)
    serial = _fuse(em, points, one_at_a_time=True)
    parallel = _fuse(em, points, one_at_a_time=False)
    assert serial[2].sum() > len(points) // 2
    np.testing.assert_array_equal(parallel[2], serial[2])  # point counts: exact
    np.testing.assert_allclose(parallel[:2], serial[:2], rtol=1e-5, atol=1e-6)  # sums: up to float order
