"""Behaviour of the individual map layers and post-processing plugins.

Accuracy of the elevation layer is covered elsewhere; this module checks that
the *other* outputs mean what they claim once real sensor data has flowed
through the pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from emsim.runner import RunConfig, make_parameter, run

pytestmark = requires_gpu

BASE_LAYERS = ("elevation", "variance", "is_valid", "upper_bound", "is_upper_bound", "time")
NORMAL_LAYERS = ("normal_x", "normal_y", "normal_z")
PLUGIN_LAYERS = ("min_filter", "smooth", "inpaint", "erosion")


@pytest.fixture(scope="module")
def mapped():
    """One sweep over rough terrain with every layer of interest exported."""
    cfg = RunConfig(scene="rough", trajectory="spin", n_steps=16, resolution=0.04, map_length=6.0)
    return run(cfg, layers=BASE_LAYERS + NORMAL_LAYERS + PLUGIN_LAYERS)


@pytest.fixture(scope="module")
def mapped_flat():
    cfg = RunConfig(scene="flat", trajectory="spin", n_steps=16, resolution=0.04, map_length=6.0)
    return run(cfg, layers=BASE_LAYERS + NORMAL_LAYERS)


@pytest.mark.parametrize("layer", BASE_LAYERS + NORMAL_LAYERS + PLUGIN_LAYERS)
def test_layer_exports_with_the_right_shape(layer, mapped):
    data = mapped.layers[layer]
    assert data.shape == (mapped.cell_n, mapped.cell_n)
    assert np.isfinite(data).any(), f"layer '{layer}' is entirely NaN"


def test_is_valid_marks_exactly_the_observed_cells(mapped, viz):
    """``is_valid`` is the mask that ``elevation`` is NaN-filled against."""
    viz("layers", mapped)
    is_valid = mapped.layers["is_valid"] > 0.5
    observed = np.isfinite(mapped.layers["elevation"])
    np.testing.assert_array_equal(is_valid, observed)
    assert 0.5 < observed.mean() < 1.0, "expected partial coverage from a 6 m map"


def test_variance_drops_where_the_sensor_looked(mapped):
    """Observed cells must fall well below the prior; unobserved ones must not."""
    param = make_parameter(mapped.config)
    variance = mapped.layers["variance"]
    observed = np.isfinite(mapped.layers["elevation"])

    assert np.all(variance >= 0.0), "variance must never go negative"
    assert np.median(variance[observed]) < 0.1 * param.initial_variance
    assert np.all(variance[~observed] >= param.initial_variance - 1e-3)
    assert variance.max() <= param.initial_variance + param.time_variance * mapped.config.n_steps + 1e-3


def test_time_layer_advances_and_resets_on_observation(mapped):
    """The time layer counts how long a cell has gone unobserved."""
    time_layer = mapped.layers["time"]
    observed = np.isfinite(mapped.layers["elevation"])
    interval = make_parameter(mapped.config).time_interval

    assert np.all(time_layer >= 0.0)
    # A 360-degree sweep revisits every direction, so many cells were seen recently.
    assert np.median(time_layer[observed]) <= interval * mapped.config.n_steps
    assert time_layer.max() == pytest.approx(interval * mapped.config.n_steps, rel=0.2)


def test_upper_bound_tracks_measured_cells(mapped):
    """Where a cell was measured, its upper bound sits on the measured surface.

    The two are written by different kernels -- ``upper_bound`` takes the fused
    height of the last point to land in the cell, ``elevation`` the average over
    all of them -- so they agree to within the spread of a single frame's points
    rather than exactly.
    """
    elevation = mapped.layers["elevation"]
    upper = mapped.layers["upper_bound"]
    both = np.isfinite(elevation) & np.isfinite(upper)
    assert both.sum() > 1000
    delta = upper[both] - elevation[both]
    assert np.abs(np.median(delta)) < 0.005
    assert np.percentile(np.abs(delta), 99) < 0.05


def test_is_upper_bound_is_a_binary_flag(mapped):
    is_upper = mapped.layers["is_upper_bound"]
    flagged = np.isfinite(is_upper)
    assert flagged.sum() > 1000
    assert set(np.unique(is_upper[flagged])) <= {0.0, 1.0}


def test_only_visibility_cleanup_downgrades_a_measured_cell():
    """A measured cell becomes "upper bound only" solely via visibility cleanup.

    Cleanup erodes a cell's validity in small steps, so a cell can briefly be
    both measured and flagged. Turning cleanup off must remove that entirely --
    which pins the flag to its one legitimate cause.
    """

    def flagged_measured(enable_cleanup: bool) -> int:
        result = run(
            RunConfig(
                scene="wall",
                trajectory="spin",
                n_steps=16,
                resolution=0.04,
                map_length=6.0,
                enable_visibility_cleanup=enable_cleanup,
            ),
            layers=("elevation", "is_upper_bound"),
        )
        elevation, is_upper = result.layers["elevation"], result.layers["is_upper_bound"]
        measured = np.isfinite(elevation) & np.isfinite(is_upper)
        assert measured.sum() > 1000
        return int((is_upper[measured] > 0.5).sum())

    assert flagged_measured(enable_cleanup=False) == 0
    assert flagged_measured(enable_cleanup=True) > 0, "cleanup should reclaim some cells"


def test_normals_point_up_on_flat_ground(mapped_flat):
    nx, ny, nz = (mapped_flat.layers[k] for k in NORMAL_LAYERS)
    observed = np.isfinite(mapped_flat.layers["elevation"])
    # The normal filter needs a neighbourhood, so score the well-covered interior.
    interior = np.zeros_like(observed)
    m = mapped_flat.cell_n // 4
    interior[m:-m, m:-m] = True
    sel = observed & interior & (np.abs(nz) > 1e-6)
    assert sel.sum() > 1000

    norm = np.sqrt(nx[sel] ** 2 + ny[sel] ** 2 + nz[sel] ** 2)
    np.testing.assert_allclose(norm, 1.0, atol=1e-3)
    assert np.median(nz[sel]) > 0.99, "flat ground normals should point straight up"


def test_normals_tilt_on_a_slope():
    result = run(
        RunConfig(scene="slope", trajectory="spin", n_steps=16, resolution=0.04, map_length=6.0),
        layers=("elevation",) + NORMAL_LAYERS,
    )
    nx, nz = result.layers["normal_x"], result.layers["normal_z"]
    X, _ = result.cell_centers()
    on_ramp = np.isfinite(result.layers["elevation"]) & (X > 1.2) & result.mask(2.0)
    assert on_ramp.sum() > 200

    tilt = np.rad2deg(np.arctan2(-np.median(nx[on_ramp]), np.median(nz[on_ramp])))
    assert tilt == pytest.approx(15.0, abs=5.0), f"recovered slope normal tilt {tilt:.1f} deg"


def test_inpainting_fills_holes_without_moving_known_cells(mapped):
    """``inpaint`` must extend coverage while leaving measured cells alone."""
    elevation = mapped.layers["elevation"]
    inpaint = mapped.layers["inpaint"]
    observed = np.isfinite(elevation)

    assert np.isfinite(inpaint).mean() > np.isfinite(elevation).mean(), "inpaint should add coverage"
    # Where a measurement exists, inpainting must agree with it.
    np.testing.assert_allclose(inpaint[observed], elevation[observed], atol=0.02)
    # Filled-in cells should still be plausible terrain, not wild extrapolation.
    filled = np.isfinite(inpaint) & ~observed
    if filled.sum():
        assert np.abs(inpaint[filled] - np.nanmedian(elevation)).max() < 2.0


def test_min_and_smooth_filters_track_the_surface(mapped):
    """Both are height layers, so they must stay near the measured elevation."""
    elevation = mapped.layers["elevation"]
    observed = np.isfinite(elevation)
    for layer in ("min_filter", "smooth"):
        data = mapped.layers[layer]
        sel = observed & np.isfinite(data)
        assert sel.sum() > 1000, f"{layer} produced too few finite cells"
        assert np.abs(np.median(data[sel] - elevation[sel])) < 0.05, layer


def test_erosion_layer_is_bounded(mapped):
    """``erosion`` post-processes traversability, so it lives in [0, 1]."""
    data = mapped.layers["erosion"]
    finite = data[np.isfinite(data)]
    assert finite.size > 0
    assert finite.min() >= -1e-6 and finite.max() <= 1.0 + 1e-6


def test_traversability_layer(mapped):
    """Traversability needs the learned filter, which needs a CUDA-capable torch."""
    if not mapped.traversability_enabled:
        pytest.skip("traversability filter disabled (no CUDA torch/chainer in this environment)")
    result = run(mapped.config, layers=("elevation", "traversability"))
    trav = result.layers["traversability"]
    finite = trav[np.isfinite(trav)]
    assert finite.size > 1000
    assert finite.min() >= -1e-6 and finite.max() <= 1.0 + 1e-6


def test_clear_resets_the_map():
    """After ``clear`` the map must be empty again, and remappable."""
    import cupy as cp

    from elevation_mapping_cupy.elevation_mapping import ElevationMap

    cfg = RunConfig(scene="rough", trajectory="static", n_steps=1, resolution=0.05, map_length=4.0)
    result = run(cfg)
    assert np.isfinite(result.layers["elevation"]).any()

    param = make_parameter(cfg)
    em = ElevationMap(param)
    buf = np.zeros((param.true_cell_n, param.true_cell_n), dtype=np.float32)
    em.clear()
    em.get_map_with_name_ref("elevation", buf)
    assert np.all(np.isnan(buf)), "a cleared map must report no valid cells"
    np.testing.assert_allclose(cp.asnumpy(em.elevation_map[1]), param.initial_variance)
