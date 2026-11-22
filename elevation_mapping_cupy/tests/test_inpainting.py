import cupy as cp
import numpy as np

from elevation_mapping_cupy.plugins.inpainting import Inpainting


def _layers(height, valid):
    em = np.zeros((7,) + height.shape, np.float32)
    em[0] = np.where(valid, height, 0.0)
    em[2] = valid
    return cp.asarray(em)


def test_push_pull_keeps_observed_cells_and_fills_holes_smoothly():
    n = 120
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) * 0.05
    plane = 0.3 * x - 0.1 * y + 1.0
    rng = np.random.default_rng(0)
    valid = rng.random((n, n)) > 0.3
    valid[40:60, 50:75] = False  # a 1 x 1.25 m hole
    valid[0, 0] = True
    em = _layers(plane, valid)
    em[0, 0, 0] = cp.nan  # flagged valid but not finite: treated as a hole
    out = cp.asnumpy(Inpainting(method="push_pull")(em, [], None, []))
    keep = valid.copy()
    keep[0, 0] = False
    assert np.array_equal(out[keep], plane[keep])  # bit-exact
    assert np.isfinite(out).all()
    # A membrane fill reproduces a plane inside holes (to the few smoothing
    # steps it gets): scattered dropouts closely, a 1.25 m hole on a 30 % slope
    # to within 2 cm. Not on the outer ring, which has no neighbour outside to
    # continue the slope from (and is the core's padding, never published).
    err = np.abs(out - plane)[1:-1, 1:-1]
    valid, hole = valid[1:-1, 1:-1], np.zeros((n - 2, n - 2), bool)
    hole[39:59, 49:74] = True
    assert err[~valid & ~hole].max() < 0.005
    assert err[hole].max() < 0.02


def test_push_pull_without_observations_returns_the_elevation_layer():
    em = _layers(np.full((30, 30), 2.0, np.float32), np.zeros((30, 30), bool))
    out = Inpainting(method="push_pull")(em, [], None, [])
    assert cp.array_equal(out, em[0])


def test_push_pull_fill_handles_a_transposed_input():
    from elevation_mapping_cupy.plugins.inpainting import push_pull_fill

    rng = np.random.default_rng(1)
    h = cp.asarray(rng.random((40, 70)).astype(np.float32))
    w = cp.asarray(rng.random((40, 70)) > 0.5)
    # Same fill up to float summation order (the kernels visit neighbours in a fixed order).
    cp.testing.assert_allclose(push_pull_fill(h.T, w.T), push_pull_fill(h, w).T, atol=1e-6)
