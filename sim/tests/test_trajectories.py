"""Trajectory generation: paths, body motion, and mapping under full 6-DoF pose.

The path tests need no GPU -- ``make_trajectory`` only wants a scene for its
terrain height. The accuracy cases at the bottom drive the real map.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from emsim import scenes
from emsim.runner import TRAJECTORIES, BodyMotion, RunConfig, make_trajectory, nominal_path

MOVING = ["line", "circle", "figure8"]
STATIONARY = ["static", "spin"]


def poses_for(trajectory="line", scene_name="flat", **kwargs):
    cfg = RunConfig(scene=scene_name, trajectory=trajectory, n_steps=24, **kwargs)
    return cfg, make_trajectory(cfg, scenes.make_scene(scene_name))


@pytest.mark.parametrize("trajectory", TRAJECTORIES)
def test_every_trajectory_yields_valid_poses(trajectory):
    cfg, poses = poses_for(trajectory)
    assert len(poses) == cfg.n_steps
    for pose in poses:
        assert pose.position.shape == (3,) and np.all(np.isfinite(pose.position))
        np.testing.assert_allclose(pose.rotation @ pose.rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(pose.rotation) == pytest.approx(1.0)


def test_unknown_trajectory_is_rejected():
    with pytest.raises(ValueError, match="unknown trajectory"):
        make_trajectory(RunConfig(trajectory="teleport"), scenes.make_scene("flat"))


@pytest.mark.parametrize("trajectory", STATIONARY)
def test_stationary_trajectories_do_not_translate(trajectory):
    _, poses = poses_for(trajectory)
    xy = np.array([p.position[:2] for p in poses])
    assert np.abs(xy - xy[0]).max() < 1e-12


@pytest.mark.parametrize("trajectory", MOVING)
def test_moving_trajectories_translate(trajectory):
    _, poses = poses_for(trajectory)
    xy = np.array([p.position[:2] for p in poses])
    assert np.ptp(xy, axis=0).max() > 1.0, f"{trajectory} barely moved"


def test_spin_sweeps_a_full_turn_without_moving():
    _, poses = poses_for("spin")
    yaws = np.array([p.rpy[2] for p in poses])
    assert yaws.min() == pytest.approx(0.0)
    assert yaws.max() == pytest.approx(2 * np.pi * 23 / 24, abs=1e-9)


def test_circle_translates_and_rotates_together():
    """The case rotation-only and translation-only tests both miss."""
    cfg, poses = poses_for("circle", path_radius=1.5)
    xy = np.array([p.position[:2] for p in poses])
    yaws = np.unwrap([p.rpy[2] for p in poses])

    radii = np.linalg.norm(xy - np.array(cfg.start_xy), axis=1)
    np.testing.assert_allclose(radii, cfg.path_radius, atol=1e-9)
    assert abs(yaws[-1] - yaws[0]) > 5.0, "heading should sweep most of a turn"


def test_figure8_reverses_its_turn_direction():
    """A lemniscate changes yaw rate sign, which a circle never does."""
    _, poses = poses_for("figure8", path_radius=1.5)
    yaw_rate = np.diff(np.unwrap([p.rpy[2] for p in poses]))
    assert yaw_rate.max() > 0 and yaw_rate.min() < 0, "yaw rate should change sign"


def test_base_follows_the_terrain():
    cfg, poses = poses_for("line", scene_name="slope", path_length=3.0, start_xy=(0.0, 0.0))
    scene = scenes.make_scene("slope")
    for pose in poses:
        expected = scene.analytic_height(pose.position[0], pose.position[1]) + cfg.base_height
        assert pose.position[2] == pytest.approx(float(expected), abs=1e-9)


# --------------------------------------------------------------------------- #
# Body motion
# --------------------------------------------------------------------------- #


def test_body_motion_is_off_by_default():
    """Adding the feature must not silently change existing runs."""
    assert not BodyMotion().enabled
    _, poses = poses_for("line")
    rpy = np.array([p.rpy for p in poses])
    np.testing.assert_allclose(rpy[:, :2], 0.0, atol=1e-15)  # no roll, no pitch
    ys = np.array([p.position[1] for p in poses])
    np.testing.assert_allclose(ys, ys[0], atol=1e-15)  # no sway


def test_walking_motion_moves_all_six_degrees_of_freedom():
    motion = BodyMotion.walking()
    cfg, poses = poses_for("circle", body_motion=motion, path_radius=1.5)
    xyz = np.array([p.position for p in poses])
    rpy = np.array([p.rpy for p in poses])

    def spans(observed, amplitude):
        """Peak-to-peak is bounded by 2A, and a discrete sinusoid rarely reaches it.

        With ``cycles`` oscillations over ``n_steps`` samples the phases land
        where they land -- 6 cycles over 24 steps samples every 90 deg, so a
        term offset by 45 deg only ever reaches 0.707 A. Bound it rather than
        re-deriving the sampling.
        """
        peak_to_peak = float(np.ptp(observed))
        assert peak_to_peak <= 2 * amplitude + 1e-9, "amplitude exceeded its configured bound"
        assert peak_to_peak >= 1.4 * amplitude, "motion is far smaller than configured"

    # Translation in x and y from the path, in z from the terrain plus the bob.
    assert np.ptp(xyz[:, 0]) > 1.0 and np.ptp(xyz[:, 1]) > 1.0
    spans(xyz[:, 2], motion.bob)
    # Rotation in all three axes.
    spans(rpy[:, 0], np.deg2rad(motion.roll_deg))
    spans(rpy[:, 1], np.deg2rad(motion.pitch_deg))
    assert np.ptp(np.unwrap(rpy[:, 2])) > 5.0


def test_bob_oscillates_about_the_nominal_height():
    """The bob must be zero-mean, not a constant height offset."""
    cfg, poses = poses_for("static", body_motion=BodyMotion(bob=0.05, cycles=4.0))
    zs = np.array([p.position[2] for p in poses])
    assert np.mean(zs) == pytest.approx(cfg.base_height, abs=0.001)
    # Symmetric about the nominal height, and inside the configured amplitude.
    # Discrete sampling means the peaks themselves need not be reached.
    above = zs.max() - cfg.base_height
    below = cfg.base_height - zs.min()
    assert above == pytest.approx(below, abs=1e-9)
    assert 0.7 * 0.05 <= above <= 0.05 + 1e-9


def test_sway_is_lateral_to_the_heading():
    """Sway is a body-frame displacement, so on a curve it must follow the yaw."""
    # Heading fixed at 0 (+x): sway shows up purely in y.
    _, poses = poses_for("line", body_motion=BodyMotion(sway=0.05, cycles=3.0))
    ys = np.array([p.position[1] for p in poses])
    assert np.ptp(ys) == pytest.approx(0.1, rel=0.15)

    # Heading fixed at 90 deg by spinning to it is awkward, so check the circle:
    # every sway displacement must be perpendicular to that pose's heading.
    cfg, swayed = poses_for("circle", body_motion=BodyMotion(sway=0.05, cycles=3.0))
    _, plain = poses_for("circle")
    for a, b in zip(swayed, plain):
        offset = a.position[:2] - b.position[:2]
        if np.linalg.norm(offset) < 1e-6:
            continue
        heading = np.array([np.cos(b.rpy[2]), np.sin(b.rpy[2])])
        assert abs(float(offset @ heading)) < 1e-9, "sway leaked into the forward direction"


def test_rotation_matrix_matches_the_reported_rpy():
    from emsim.sensor import rpy_to_matrix

    _, poses = poses_for("figure8", body_motion=BodyMotion.walking())
    for pose in poses:
        np.testing.assert_allclose(pose.rotation, rpy_to_matrix(*pose.rpy), atol=1e-12)


def test_body_motion_phases_are_offset():
    """Bob, roll and pitch must not move in lockstep, or the base just heaves."""
    _, poses = poses_for("static", body_motion=BodyMotion.walking())
    zs = np.array([p.position[2] for p in poses])
    rolls = np.array([p.rpy[0] for p in poses])
    pitches = np.array([p.rpy[1] for p in poses])

    def correlation(a, b):
        a, b = a - a.mean(), b - b.mean()
        return abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert correlation(zs, rolls) < 0.95
    assert correlation(rolls, pitches) < 0.95


def test_nominal_path_is_body_motion_free():
    """``nominal_path`` is the path before oscillation, and must stay that way."""
    cfg = RunConfig(trajectory="line", n_steps=16, body_motion=BodyMotion.walking())
    xs, ys, yaws = nominal_path(cfg)
    np.testing.assert_allclose(ys, ys[0], atol=1e-15)
    np.testing.assert_allclose(yaws, 0.0, atol=1e-15)


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #


@requires_gpu
@pytest.mark.parametrize("trajectory", ["circle", "figure8"])
def test_mapping_survives_full_six_dof_motion(trajectory):
    """Translate, rotate, bob and tilt at once -- and still match ground truth.

    Every other accuracy test holds the base level. Here the sensor arrives at a
    different attitude every frame, so a pose-handling error cannot hide behind
    a constant offset.
    """
    from emsim.runner import run

    result = run(
        RunConfig(
            scene="mixed",
            trajectory=trajectory,
            n_steps=24,
            path_radius=1.2,
            resolution=0.04,
            map_length=8.0,
            body_motion=BodyMotion.walking(),
            record_per_step=True,
        )
    )
    err = result.error("elevation", radius=2.0)
    # Lower than the stationary-sweep budgets on purpose: an orbiting base
    # leaves its own starting patch behind, so the final window is only ever
    # partly filled.
    assert err.coverage > 0.65, f"{trajectory}: {err}"
    assert err.rmse < 0.03, f"{trajectory}: {err}"
    assert abs(err.bias) < 0.02, f"{trajectory}: attitude changes should not bias the surface"

    # The base really did use all six degrees of freedom.
    rpy = np.array([s.base_rpy for s in result.steps])
    assert np.ptp(rpy[:, 0]) > np.deg2rad(3.0) and np.ptp(rpy[:, 1]) > np.deg2rad(2.0)


@requires_gpu
def test_body_motion_does_not_wreck_a_stationary_map():
    """Bobbing and tilting in place should cost little against standing still."""
    from emsim.runner import run

    common = dict(scene="rough", trajectory="spin", n_steps=16,
                  resolution=0.04, map_length=6.0)
    still = run(RunConfig(**common))
    shaky = run(RunConfig(body_motion=BodyMotion.walking(), **common))

    still_err = still.error("elevation", radius=2.5)
    shaky_err = shaky.error("elevation", radius=2.5)
    assert shaky_err.coverage > still_err.coverage - 0.1, f"{shaky_err} vs {still_err}"
    assert shaky_err.rmse < still_err.rmse + 0.015, f"{shaky_err} vs {still_err}"


@requires_gpu
def test_map_centre_tracks_a_six_dof_base():
    """Bob must reach the map's z centre, and xy must still snap to the lattice."""
    from emsim.runner import run

    cfg = RunConfig(
        scene="rough", trajectory="line", n_steps=16, path_length=2.0, start_xy=(-1.0, 0.0),
        resolution=0.04, map_length=6.0, body_motion=BodyMotion.walking(), record_per_step=True,
    )
    result = run(cfg)
    for step in result.steps:
        delta_xy = step.center[:2] - step.base_position[:2]
        assert np.max(np.abs(delta_xy)) <= cfg.resolution / 2 + 1e-6
        # z is not quantised, so the centre should follow the bob exactly.
        assert step.center[2] == pytest.approx(step.base_position[2], abs=1e-4)
