"""End-to-end accuracy of ``ElevationMap`` against ray-cast ground truth.

Each case drives the real mapping pipeline over a simulated trajectory and
scores the resulting elevation layer cell-for-cell. Thresholds are set well
above the numbers the pipeline currently achieves, so these catch regressions
rather than encoding today's exact behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from emsim.metrics import compare_maps
from emsim.runner import BodyMotion, RunConfig, run
from emsim.sensor import SensorNoise

pytestmark = requires_gpu

# Scoring radius. Beyond it the camera only ever sees terrain at a grazing
# angle from one direction, so coverage -- not accuracy -- is the limit.
RADIUS = 2.5

# scene -> (max RMSE, max p95 abs error, min coverage) inside RADIUS.
# Scenes with vertical faces get a looser RMSE: a cell straddling a step edge is
# genuinely ambiguous, which shows up in RMSE and max but not in p95.
SCENE_BUDGETS = {
    "flat": (0.010, 0.015, 0.90),
    "slope": (0.010, 0.015, 0.85),
    "rough": (0.015, 0.015, 0.90),
    "steps": (0.030, 0.020, 0.85),
    "boxes": (0.040, 0.020, 0.85),
    "mixed": (0.030, 0.020, 0.85),
}


def spin_config(scene: str, **kwargs) -> RunConfig:
    """A stationary 360-degree sweep: maximum coverage for the fewest frames."""
    defaults = dict(scene=scene, trajectory="spin", n_steps=16, resolution=0.04, map_length=6.0)
    defaults.update(kwargs)
    return RunConfig(**defaults)


@pytest.mark.parametrize("scene", sorted(SCENE_BUDGETS))
def test_elevation_accuracy_per_scene(scene, sim_run, viz):
    max_rmse, max_p95, min_coverage = SCENE_BUDGETS[scene]
    result = sim_run(f"spin::{scene}", spin_config(scene))
    err = result.error("elevation", radius=RADIUS)
    viz("comparison", result, radius=RADIUS)

    assert err.coverage >= min_coverage, f"{scene}: only {err.coverage:.1%} of cells mapped ({err})"
    assert err.rmse <= max_rmse, f"{scene}: {err}"
    assert err.p95 <= max_p95, f"{scene}: {err}"


def test_flat_ground_is_unbiased(sim_run):
    """On a plane the map should not systematically float above or sink below."""
    result = sim_run("spin::flat", spin_config("flat"))
    err = result.error("elevation", radius=RADIUS)
    assert abs(err.bias) < 0.01, f"flat ground bias {err.bias:+.4f} m"


def test_slope_gradient_is_recovered(sim_run, viz):
    """The mapped surface must reproduce the ramp's gradient, not just its height."""
    result = sim_run("spin::slope", spin_config("slope"))
    viz("surface", result, radius=RADIUS)
    est = result.layers["elevation"]
    X, _ = result.cell_centers()
    mask = result.mask(RADIUS) & np.isfinite(est) & (X > 1.2)  # on the ramp proper
    assert mask.sum() > 500

    fit = np.polyfit(X[mask], est[mask], 1)
    assert fit[0] == pytest.approx(np.tan(np.deg2rad(15.0)), abs=0.02), f"gradient {fit[0]:.4f}"


def test_step_heights_are_resolved(sim_run, viz):
    """Every visible stair tread must come out at its true height.

    A 0.8 m camera cannot see over the staircase, so the upper treads are
    legitimately occluded; those must stay unmapped rather than be invented.
    """
    result = sim_run("spin::steps", spin_config("steps"))
    viz("surface", result, radius=RADIUS)
    est, (X, Y) = result.layers["elevation"], result.cell_centers()
    near_centreline = np.abs(Y) < 0.5

    checked = 0
    for i in range(6):
        x_lo, x_hi = 1.0 + i * 0.45, 1.0 + (i + 1) * 0.45
        footprint = near_centreline & (X > x_lo + 0.1) & (X < x_hi - 0.1)
        tread = footprint & np.isfinite(est)
        if tread.sum() < 20:
            # Occluded by the treads in front of it: nothing may be fabricated.
            assert tread.sum() == 0 or np.allclose(
                est[tread], (i + 1) * 0.12, atol=0.05
            ), f"tread {i} partially mapped with wrong heights"
            continue
        assert np.median(est[tread]) == pytest.approx((i + 1) * 0.12, abs=0.02), f"tread {i}"
        checked += 1

    assert checked >= 3, f"expected at least 3 treads in view, checked {checked}"


def test_map_converges_as_frames_accumulate(sim_run, viz):
    """Coverage must grow monotonically and accuracy must not degrade."""
    cfg = spin_config("rough", n_steps=12, record_per_step=True)
    result = sim_run("spin::rough::per_step", cfg)
    viz("convergence", result)
    coverage = [s.error.coverage for s in result.steps]
    assert len(coverage) == 12
    assert coverage[0] < coverage[-1], "coverage should grow as the sweep proceeds"
    assert all(b >= a - 1e-9 for a, b in zip(coverage, coverage[1:])), "coverage must not drop"
    assert result.steps[-1].error.rmse <= result.steps[1].error.rmse + 0.005


def test_unobserved_cells_are_nan(sim_run):
    """Cells the sensor never saw must read NaN, not a fabricated height."""
    result = sim_run("static::flat", spin_config("flat", trajectory="static", n_steps=4))
    est = result.layers["elevation"]
    X, Y = result.cell_centers()
    # A static camera faces +x, so the far -x corner of the map is never seen.
    behind = (X < result.center[0] - 2.0) & (np.abs(Y - result.center[1]) < 1.0)
    assert behind.sum() > 100
    assert np.all(np.isnan(est[behind])), "cells behind the robot must stay unobserved"


def test_sensor_noise_degrades_accuracy_but_stays_bounded(sim_run):
    """Depth noise should show up in the map without destabilising it."""
    clean = sim_run("spin::flat", spin_config("flat"))
    noisy = sim_run(
        "spin::flat::noisy",
        spin_config("flat", noise=SensorNoise(range_relative_std=0.01, range_absolute_std=0.01, seed=3)),
    )
    clean_err = clean.error("elevation", radius=RADIUS)
    noisy_err = noisy.error("elevation", radius=RADIUS)

    assert noisy_err.rmse > clean_err.rmse, "noise should be visible in the map"
    assert noisy_err.rmse < 0.03, f"noise handling degraded: {noisy_err}"
    assert noisy_err.coverage >= clean_err.coverage - 0.05
    assert abs(noisy_err.bias) < 0.015, "zero-mean range noise must not bias the surface"


def test_dropout_costs_coverage_not_accuracy(sim_run):
    result = sim_run(
        "spin::rough::dropout",
        spin_config("rough", noise=SensorNoise(dropout=0.7, seed=5)),
    )
    err = result.error("elevation", radius=RADIUS)
    assert err.coverage > 0.75, f"70% dropout should still cover most cells: {err}"
    assert err.rmse < 0.02, f"{err}"


def test_accuracy_holds_while_walking(sim_run):
    """The stationary sweeps hold the base level; this one does not.

    Translation, heading change, vertical bob and roll/pitch all at once, which
    is how the sensor actually arrives on a legged robot.
    """
    result = sim_run(
        "walk::mixed",
        spin_config("mixed", trajectory="circle", n_steps=24, path_radius=1.2,
                    body_motion=BodyMotion.walking()),
    )
    err = result.error("elevation", radius=2.0)
    assert err.coverage > 0.65, f"{err}"
    assert err.rmse <= 0.030, f"{err}"
    assert abs(err.bias) < 0.02, f"attitude oscillation should not bias the surface: {err}"


def test_ground_truth_window_tracks_the_map_centre(sim_run):
    """Sanity check on the comparison itself: a shifted window must score worse."""
    result = sim_run("spin::steps", spin_config("steps"))
    est = result.layers["elevation"]
    aligned = compare_maps(est, result.ground_truth, result.mask(RADIUS))

    shifted_gt = result.sampler.sample_map_grid(
        (result.center[0] + 0.4, result.center[1]), result.cell_n, result.config.resolution
    )
    shifted = compare_maps(est, shifted_gt, result.mask(RADIUS))
    assert aligned.rmse < shifted.rmse, "a deliberately misaligned ground truth must score worse"
