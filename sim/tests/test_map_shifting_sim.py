"""Robot-centric map shifting, exercised against a world-fixed terrain.

``ElevationMap`` keeps a window centred on the robot and rolls its buffers as the
robot drives. The simulation gives a decisive test of that: the terrain does not
move, so anything already mapped must keep reporting the same height at the same
*world* coordinate no matter where the window has since moved to.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from emsim.heightmap import map_cell_centers
from emsim.metrics import compare_maps
from emsim.runner import RunConfig, run

pytestmark = requires_gpu


def drive_config(scene: str = "steps", **kwargs) -> RunConfig:
    defaults = dict(
        scene=scene,
        trajectory="line",
        n_steps=14,
        path_length=2.6,
        start_xy=(-2.2, 0.0),
        resolution=0.04,
        map_length=6.0,
        record_per_step=True,
    )
    defaults.update(kwargs)
    return RunConfig(**defaults)


@pytest.fixture(scope="module")
def driven(request):
    return run(drive_config())


def test_map_centre_follows_the_robot(driven):
    """The centre tracks the base to within the half-cell that snapping allows."""
    resolution = driven.config.resolution
    assert len(driven.steps) == driven.config.n_steps
    for step in driven.steps:
        delta = step.center[:2] - step.base_position[:2]
        assert np.max(np.abs(delta)) <= resolution / 2 + 1e-6, (
            f"step {step.index}: centre {step.center[:2]} vs base {step.base_position[:2]}"
        )
    # The map really did travel, so the shifting code was exercised.
    travelled = np.abs(driven.steps[-1].center[0] - driven.steps[0].center[0])
    assert travelled > 2.0, f"map centre only moved {travelled:.2f} m"


def test_map_centre_snaps_to_the_cell_lattice(driven):
    """Centres must land on whole-cell multiples, or ground truth would misalign."""
    resolution = driven.config.resolution
    for step in driven.steps:
        k = step.center[:2] / resolution
        np.testing.assert_allclose(k, np.round(k), atol=1e-4)


def test_terrain_stays_put_in_world_coordinates(driven):
    """Heights already mapped must not drift as the window rolls over them."""
    first, last = driven.steps[0], driven.steps[-1]
    n, resolution = driven.cell_n, driven.config.resolution

    X0, Y0 = map_cell_centers(tuple(first.center[:2]), n, resolution)
    X1, Y1 = map_cell_centers(tuple(last.center[:2]), n, resolution)

    # The two windows overlap; index the same world cells in each. Row 0 holds
    # the largest x, so advancing by `shift` cells along +x moves a given world
    # cell from row r of the early window to row r + shift of the late one.
    shift = int(round((last.center[0] - first.center[0]) / resolution))
    assert shift > 0, "expected the map to have advanced along +x"
    early, late = first.elevation[: n - shift], last.elevation[shift:]
    # atol is float32-scale: the map centre is stored in the map's own dtype.
    np.testing.assert_allclose(X0[: n - shift], X1[shift:], atol=1e-5)
    np.testing.assert_allclose(Y0[: n - shift], Y1[shift:], atol=1e-5)

    both = np.isfinite(early) & np.isfinite(late)
    assert both.sum() > 500, f"only {both.sum()} cells survived in both windows"
    drift = np.abs(early[both] - late[both])
    assert np.percentile(drift, 95) < 0.02, f"p95 drift {np.percentile(drift, 95):.4f} m"
    assert np.median(drift) < 0.005, f"median drift {np.median(drift):.4f} m"


def test_accuracy_holds_against_world_fixed_truth_while_driving(driven):
    """Every step's window must match the ground truth sampled at that window."""
    for step in driven.steps[3:]:  # skip the first frames, still filling in
        err = compare_maps(step.elevation, step.ground_truth)
        assert err.coverage > 0.2, f"step {step.index}: {err}"
        assert err.p95 < 0.05, f"step {step.index}: {err}"


def test_world_fixed_features_keep_their_world_position(driven):
    """A stair tread must read the same height at the same world point, always.

    This is the sharpest statement of what shifting has to get right: if the
    buffers rolled by the wrong amount or in the wrong direction, the tread
    would appear to slide across the world as the robot drives.
    """
    resolution = driven.config.resolution
    # (x, y, expected height) for the "steps" scene: treads are 0.12 m apart,
    # 0.45 m deep, starting at x = 1.0. Probe tread centres and the flat approach.
    probes = [(-1.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.225, 0.0, 0.12), (1.675, 0.0, 0.24)]

    hits = 0
    for step in driven.steps:
        X, Y = map_cell_centers(tuple(step.center[:2]), driven.cell_n, resolution)
        for px, py, expected in probes:
            r = int(np.argmin(np.abs(X[:, 0] - px)))
            c = int(np.argmin(np.abs(Y[0, :] - py)))
            if abs(X[r, c] - px) > resolution or abs(Y[r, c] - py) > resolution:
                continue  # probe outside this window
            value = step.elevation[r, c]
            if not np.isfinite(value):
                continue  # not observed yet -- allowed, but never wrong
            assert value == pytest.approx(expected, abs=0.03), (
                f"step {step.index}: ({px}, {py}) read {value:.3f}, expected {expected:.3f}"
            )
            hits += 1

    assert hits > 40, f"only {hits} probe readings landed on observed cells"


def test_driving_over_terrain_matches_a_stationary_sweep():
    """Driving to a spot should map it about as well as sitting there and spinning."""
    driven_here = run(drive_config(scene="rough", n_steps=12, path_length=2.0, start_xy=(-2.0, 0.0)))
    parked = run(
        RunConfig(
            scene="rough",
            trajectory="spin",
            n_steps=12,
            resolution=0.04,
            map_length=6.0,
            start_xy=(0.0, 0.0),
        )
    )
    driven_err = driven_here.error("elevation", radius=1.5)
    parked_err = parked.error("elevation", radius=1.5)
    assert driven_err.rmse < parked_err.rmse + 0.02, f"driven {driven_err} vs parked {parked_err}"
    assert driven_err.coverage > 0.7, f"{driven_err}"
