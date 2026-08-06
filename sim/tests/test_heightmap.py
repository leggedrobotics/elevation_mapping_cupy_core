"""Ground-truth ray casting and map-grid alignment. No GPU required.

The ray-cast sampler is the yardstick every accuracy test is measured against,
so it is validated here against each scene's independently derived closed-form
surface before it is trusted anywhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from emsim.heightmap import (
    AnalyticHeightmap,
    GroundTruthHeightmap,
    map_cell_centers,
    radial_mask,
)

RESOLUTION = 0.04
BOUNDS = (-4.0, 4.0, -4.0, 4.0)

# Scenes whose surface is smooth: ray casting must reproduce it everywhere.
SMOOTH_SCENES = ["flat", "slope", "rough"]
# Scenes with vertical faces: cells straddling an edge legitimately disagree
# with the analytic surface by up to the full step height, so those are scored
# by percentile rather than maximum.
STEPPED_SCENES = ["steps", "gap", "wall", "boxes", "mixed"]


@pytest.mark.parametrize("name", SMOOTH_SCENES)
def test_raycast_matches_analytic_everywhere_on_smooth_terrain(name, built_scene, gt_sampler):
    scene, *_ = built_scene(name)
    sampler = gt_sampler(name, BOUNDS, RESOLUTION)
    gx, gy = np.meshgrid(sampler.xs, sampler.ys, indexing="ij")
    err = np.abs(sampler.heights - scene.analytic_height(gx, gy))
    assert np.all(np.isfinite(sampler.heights)), "every downward ray must hit the ground plane"
    # MuJoCo triangulates height fields, so allow one height-field cell of sag.
    assert err.max() < 0.01, f"{name}: max ray-cast vs analytic error {err.max():.4f} m"


@pytest.mark.parametrize("name", STEPPED_SCENES)
def test_raycast_matches_analytic_away_from_edges(name, built_scene, gt_sampler):
    scene, *_ = built_scene(name)
    sampler = gt_sampler(name, BOUNDS, RESOLUTION)
    gx, gy = np.meshgrid(sampler.xs, sampler.ys, indexing="ij")
    err = np.abs(sampler.heights - scene.analytic_height(gx, gy))
    assert np.all(np.isfinite(sampler.heights))
    assert np.percentile(err, 99) < 0.01, f"{name}: p99 error {np.percentile(err, 99):.4f} m"
    assert err.mean() < 1e-3, f"{name}: mean error {err.mean():.5f} m"


def test_sampler_grid_is_on_the_map_lattice(gt_sampler):
    sampler = gt_sampler("flat", BOUNDS, RESOLUTION)
    # Every cell centre must sit at (k + 0.5) * resolution, the lattice the
    # elevation map's own cells land on.
    for coords in (sampler.xs, sampler.ys):
        k = coords / RESOLUTION - 0.5
        np.testing.assert_allclose(k, np.round(k), atol=1e-9)


def test_height_at_is_exact_at_cell_centres(gt_sampler):
    sampler = gt_sampler("steps", BOUNDS, RESOLUTION)
    gx, gy = np.meshgrid(sampler.xs, sampler.ys, indexing="ij")
    np.testing.assert_array_equal(sampler.height_at(gx, gy), sampler.heights)


def test_height_at_is_nan_outside_the_built_region(gt_sampler):
    sampler = gt_sampler("flat", BOUNDS, RESOLUTION)
    out = sampler.height_at(np.array([0.0, 99.0, -99.0]), np.array([0.0, 0.0, 0.0]))
    assert np.isfinite(out[0])
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_robot_body_is_excluded_from_ground_truth(built_scene):
    """Parking the sensor carrier over the terrain must not change ground truth."""
    import mujoco

    scene, model, data, robot_id = built_scene("flat")
    bounds, res = (-1.0, 1.0, -1.0, 1.0), 0.1

    data.mocap_pos[0] = np.array([50.0, 50.0, 50.0])
    mujoco.mj_forward(model, data)
    away = GroundTruthHeightmap(model, data, bounds, res, bodyexclude=robot_id).heights

    data.mocap_pos[0] = np.array([0.0, 0.0, 0.4])  # directly over the sampled patch
    mujoco.mj_forward(model, data)
    overhead = GroundTruthHeightmap(model, data, bounds, res, bodyexclude=robot_id).heights
    np.testing.assert_allclose(overhead, away)

    # Sanity check the other way: without the exclusion the robot *is* seen.
    seen = GroundTruthHeightmap(model, data, bounds, res, bodyexclude=-1).heights
    assert seen.max() > away.max() + 0.1

    data.mocap_pos[0] = np.array([0.0, 0.0, 1.0])
    mujoco.mj_forward(model, data)


def test_map_cell_centers_layout():
    """Row 0 is max x and column 0 is max y, matching the exported map."""
    n, res, center = 4, 0.5, (1.0, -2.0)
    X, Y = map_cell_centers(center, n, res)
    assert X.shape == Y.shape == (n, n)
    # Rows carry x and descend; columns carry y and descend.
    assert X[0, 0] > X[-1, 0] and np.allclose(X[:, 0], X[:, -1])
    assert Y[0, 0] > Y[0, -1] and np.allclose(Y[0, :], Y[-1, :])
    np.testing.assert_allclose(np.diff(X[:, 0]), -res)
    np.testing.assert_allclose(np.diff(Y[0, :]), -res)
    # The grid is centred on the map centre.
    assert X.mean() == pytest.approx(center[0])
    assert Y.mean() == pytest.approx(center[1])
    # Extent spans exactly n * resolution, inset by half a cell on each side.
    assert X.max() == pytest.approx(center[0] + (n / 2 - 0.5) * res)
    assert X.min() == pytest.approx(center[0] - (n / 2 - 0.5) * res)


def test_map_cell_centers_track_the_centre():
    n, res = 8, 0.25
    X0, Y0 = map_cell_centers((0.0, 0.0), n, res)
    X1, Y1 = map_cell_centers((1.0, -0.5), n, res)
    np.testing.assert_allclose(X1 - X0, 1.0)
    np.testing.assert_allclose(Y1 - Y0, -0.5)


def test_radial_mask():
    n, res = 20, 0.1
    mask = radial_mask(n, res, 0.5)
    assert mask.shape == (n, n)
    assert mask[n // 2, n // 2]
    assert not mask[0, 0]
    # Area of the disc, to within the discretisation.
    assert mask.sum() * res * res == pytest.approx(np.pi * 0.25, rel=0.1)


def test_analytic_sampler_agrees_with_the_scene(built_scene):
    scene, *_ = built_scene("rough")
    sampler = AnalyticHeightmap(scene.analytic_height)
    grid = sampler.sample_map_grid((0.5, -0.5), 16, 0.1)
    X, Y = map_cell_centers((0.5, -0.5), 16, 0.1)
    np.testing.assert_allclose(grid, scene.analytic_height(X, Y))


def test_sample_map_grid_orientation_follows_the_terrain(gt_sampler):
    """A staircase rising along +x must rise toward row 0 of the sampled grid."""
    sampler = gt_sampler("steps", BOUNDS, RESOLUTION)
    grid = sampler.sample_map_grid((1.5, 0.0), 100, RESOLUTION)
    top_rows = np.nanmean(grid[:10])  # largest x
    bottom_rows = np.nanmean(grid[-10:])  # smallest x
    assert top_rows > bottom_rows + 0.1
