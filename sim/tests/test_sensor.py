"""Ray-cast depth sensor: intrinsics, pose conventions and noise. No GPU required."""

from __future__ import annotations

import numpy as np
import pytest

from emsim.sensor import (
    R_BODY_FROM_OPTICAL,
    CameraIntrinsics,
    DepthSensor,
    SensorNoise,
    camera_pose,
    rot_y,
    rot_z,
)


def test_ray_directions_are_unit_and_centred():
    intr = CameraIntrinsics(width=65, height=49, fovy_deg=60.0)
    dirs = intr.ray_directions()
    assert dirs.shape == (65 * 49, 3)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0)
    # With odd dimensions the principal point is a real pixel: it looks along +z.
    centre = dirs.reshape(49, 65, 3)[24, 32]
    np.testing.assert_allclose(centre, [0.0, 0.0, 1.0], atol=1e-12)
    # Column index grows to the right (+x), row index downward (+y).
    assert dirs.reshape(49, 65, 3)[24, 64, 0] > 0
    assert dirs.reshape(49, 65, 3)[48, 32, 1] > 0


def test_fovy_matches_the_vertical_extent():
    intr = CameraIntrinsics(width=41, height=41, fovy_deg=90.0)
    dirs = intr.ray_directions().reshape(41, 41, 3)
    # Half the vertical FOV, measured out to the last row centre. The outermost
    # pixel centre sits half a pixel inside the nominal edge.
    half = np.rad2deg(np.arctan2(dirs[40, 20, 1], dirs[40, 20, 2]))
    assert half == pytest.approx(45.0, abs=1.5)


def test_body_from_optical_is_a_rotation():
    R = R_BODY_FROM_OPTICAL
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(R @ [0, 0, 1], [1, 0, 0])  # optical forward -> body forward
    np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, -1])  # optical down -> body down


@pytest.mark.parametrize("tilt", [0.0, 20.0, 45.0, 80.0])
def test_camera_tilt_sign_points_down(tilt):
    R_wc, _ = camera_pose(np.zeros(3), rotation=0.0, tilt_down_deg=tilt)
    axis = R_wc @ [0.0, 0.0, 1.0]  # optical axis in world
    assert axis[2] == pytest.approx(-np.sin(np.deg2rad(tilt)), abs=1e-9)
    assert axis[0] == pytest.approx(np.cos(np.deg2rad(tilt)), abs=1e-9)


def test_camera_pose_applies_yaw_to_the_mount_offset():
    R_wc, t_wc = camera_pose(np.array([1.0, 2.0, 0.5]), rotation=np.pi / 2, tilt_down_deg=30.0,
                             offset_body=(0.3, 0.0, 0.0))
    # Facing +y, so a 0.3 m forward mount offset lands at y + 0.3.
    np.testing.assert_allclose(t_wc, [1.0, 2.3, 0.5], atol=1e-12)
    axis = R_wc @ [0.0, 0.0, 1.0]
    assert axis[1] > 0 and axis[0] == pytest.approx(0.0, abs=1e-12) and axis[2] < 0


def test_rotations_are_orthonormal():
    for R in (rot_z(0.7), rot_y(-1.2)):
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_depth_on_flat_ground_lands_on_the_plane(built_scene):
    _, model, data, robot_id = built_scene("flat")
    sensor = DepthSensor(model, data, CameraIntrinsics(64, 48, 60.0), max_range=8.0,
                         bodyexclude=robot_id)
    R_wc, t_wc = camera_pose(np.array([0.0, 0.0, 0.8]), rotation=0.3, tilt_down_deg=45.0)
    cap = sensor.capture(R_wc, t_wc)

    assert cap.points.shape[0] == sensor.n_rays, "every ray should reach the ground plane"
    world = cap.world_points
    # Points are carried as float32 (the map's dtype), so tolerances are float32-scale.
    np.testing.assert_allclose(world[:, 2], 0.0, atol=1e-5)
    # Ranges are consistent with the reconstructed geometry.
    np.testing.assert_allclose(np.linalg.norm(world - t_wc, axis=1), cap.ranges, rtol=1e-5)


def test_depth_sees_terrain_relief(built_scene):
    scene, model, data, robot_id = built_scene("steps")
    sensor = DepthSensor(model, data, CameraIntrinsics(96, 72, 60.0), max_range=8.0,
                         bodyexclude=robot_id)
    R_wc, t_wc = camera_pose(np.array([0.0, 0.0, 0.8]), rotation=0.0, tilt_down_deg=40.0)
    world = sensor.capture(R_wc, t_wc).world_points
    surface = scene.analytic_height(world[:, 0], world[:, 1])

    # No return may float above the terrain. A hit on a step's vertical riser
    # has, in exact arithmetic, exactly the x of the step edge; float32 storage
    # can round it to the lower step's side, so the ceiling is taken over a
    # sub-millimetre neighbourhood.
    eps = 1e-3
    ceiling = np.maximum.reduce(
        [
            scene.analytic_height(world[:, 0] + dx, world[:, 1] + dy)
            for dx in (-eps, 0.0, eps)
            for dy in (-eps, 0.0, eps)
        ]
    )
    assert np.all(world[:, 2] <= ceiling + 1e-4), "returns must lie on or inside the terrain"
    # Points on horizontal faces sit exactly on the analytic surface.
    on_surface = np.abs(world[:, 2] - surface) < 1e-4
    assert on_surface.mean() > 0.5, "most returns should be on horizontal faces"
    assert world[:, 2].max() > 0.1, "the staircase should be visible from here"


def test_max_range_truncates(built_scene):
    _, model, data, robot_id = built_scene("flat")
    sensor = DepthSensor(model, data, CameraIntrinsics(64, 48, 90.0), max_range=1.5,
                         bodyexclude=robot_id)
    R_wc, t_wc = camera_pose(np.array([0.0, 0.0, 0.8]), rotation=0.0, tilt_down_deg=45.0)
    cap = sensor.capture(R_wc, t_wc)
    assert 0 < cap.points.shape[0] < sensor.n_rays, "far rays should be dropped"
    assert cap.ranges.max() <= 1.5


def test_robot_body_does_not_occlude_itself(built_scene):
    """The sensor carrier sits at the camera origin; it must not block the view."""
    import mujoco

    _, model, data, robot_id = built_scene("flat")
    base = np.array([0.0, 0.0, 0.8])
    data.mocap_pos[0] = base
    mujoco.mj_forward(model, data)

    intr = CameraIntrinsics(48, 36, 60.0)
    R_wc, t_wc = camera_pose(base, rotation=0.0, tilt_down_deg=45.0)
    excluded = DepthSensor(model, data, intr, bodyexclude=robot_id).capture(R_wc, t_wc)
    included = DepthSensor(model, data, intr, bodyexclude=-1).capture(R_wc, t_wc)

    assert excluded.points.shape[0] == intr.width * intr.height
    assert included.ranges.mean() < excluded.ranges.mean(), "shell should be hit when not excluded"

    data.mocap_pos[0] = np.array([0.0, 0.0, 1.0])
    mujoco.mj_forward(model, data)


def test_dropout_thins_the_cloud_reproducibly(built_scene):
    _, model, data, robot_id = built_scene("flat")
    intr = CameraIntrinsics(80, 60, 60.0)
    R_wc, t_wc = camera_pose(np.array([0.0, 0.0, 0.8]), rotation=0.0, tilt_down_deg=45.0)

    def capture(seed):
        noise = SensorNoise(dropout=0.3, seed=seed)
        return DepthSensor(model, data, intr, bodyexclude=robot_id, noise=noise).capture(R_wc, t_wc)

    a, b, c = capture(0), capture(0), capture(1)
    assert a.points.shape[0] == pytest.approx(0.7 * intr.width * intr.height, rel=0.05)
    np.testing.assert_array_equal(a.points, b.points)
    assert a.points.shape != c.points.shape or not np.array_equal(a.points, c.points)


def test_range_noise_perturbs_depth_without_bias(built_scene):
    _, model, data, robot_id = built_scene("flat")
    intr = CameraIntrinsics(80, 60, 60.0)
    R_wc, t_wc = camera_pose(np.array([0.0, 0.0, 0.8]), rotation=0.0, tilt_down_deg=45.0)

    clean = DepthSensor(model, data, intr, bodyexclude=robot_id).capture(R_wc, t_wc)
    noisy = DepthSensor(
        model, data, intr, bodyexclude=robot_id, noise=SensorNoise(range_absolute_std=0.02, seed=7)
    ).capture(R_wc, t_wc)

    delta = noisy.ranges - clean.ranges
    assert delta.std() == pytest.approx(0.02, rel=0.15)
    assert abs(delta.mean()) < 0.005


def test_noise_disabled_by_default_is_a_no_op():
    noise = SensorNoise()
    assert not noise.enabled
    ranges = np.linspace(0.5, 5.0, 100)
    out, keep = noise.apply(ranges)
    np.testing.assert_array_equal(out, ranges)
    assert keep.all()
