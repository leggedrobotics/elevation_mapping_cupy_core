"""Optional LiDAR sensor backend.

The pattern and frame checks are CPU-only; the end-to-end accuracy cases need a
GPU. Everything skips cleanly if ``mujoco-lidar`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from emsim import lidar, scenes

pytestmark = pytest.mark.skipif(
    not lidar.is_available(), reason="optional 'mujoco-lidar' package not installed"
)

SPINNING = ["vlp32", "os128"]
LIVOX = ["mid360", "avia"]


def make_lidar(built_scene, name="mixed", **kwargs):
    _, model, data, robot_id = built_scene(name)
    kwargs.setdefault("max_range", 12.0)
    return lidar.LidarSensor(model=model, data=data, bodyexclude=robot_id, **kwargs)


@pytest.mark.parametrize("pattern", SPINNING + LIVOX + ["hdl64", "grid"])
def test_every_pattern_produces_a_scan(pattern, built_scene):
    sensor = make_lidar(built_scene, pattern=pattern)
    assert sensor.n_rays > 100
    R, t = sensor.pose_for(np.array([0.0, 0.0, 0.8]), rotation=0.0)
    cap = sensor.capture(R, t)
    assert cap.points.shape[1] == 3
    assert cap.points.shape[0] > 100, f"{pattern} returned almost nothing"
    assert cap.n_pixels >= cap.points.shape[0]


@pytest.mark.parametrize("pattern", SPINNING + LIVOX)
def test_no_return_floats_above_the_terrain(pattern, built_scene):
    """Points are in the sensor frame; ``R @ p + t`` must land on or inside it."""
    scene, *_ = built_scene("mixed")
    sensor = make_lidar(built_scene, pattern=pattern, tilt_down_deg=20.0)
    R, t = sensor.pose_for(np.array([0.0, 0.0, 0.8]), rotation=0.7)
    world = sensor.capture(R, t).world_points

    # A hit on a vertical face has, in exact arithmetic, exactly the x or y of
    # the edge; float32 storage can round it to the low side.
    eps = 1e-3
    ceiling = np.maximum.reduce(
        [
            scene.analytic_height(world[:, 0] + dx, world[:, 1] + dy)
            for dx in (-eps, 0.0, eps)
            for dy in (-eps, 0.0, eps)
        ]
    )
    assert np.all(world[:, 2] <= ceiling + 1e-3), "returns must lie on or inside the terrain"


@pytest.mark.parametrize("pattern", SPINNING + LIVOX)
def test_returns_sit_on_the_surface_of_smooth_terrain(pattern, built_scene):
    """On terrain with no vertical faces every return is *on* the surface.

    Scored on ``rough`` rather than ``mixed`` on purpose: a return on a step
    riser is legitimately below the surface height at its own (x, y), and the
    Livox patterns aim a large share of their rays at exactly those faces.
    """
    scene, *_ = built_scene("rough")
    sensor = make_lidar(built_scene, "rough", pattern=pattern, tilt_down_deg=20.0)
    R, t = sensor.pose_for(np.array([0.0, 0.0, 0.8]), rotation=0.7)
    world = sensor.capture(R, t).world_points

    residual = np.abs(world[:, 2] - scene.analytic_height(world[:, 0], world[:, 1]))
    on_surface = residual < 1e-3
    assert on_surface.mean() > 0.95, f"{pattern}: only {on_surface.mean():.1%} on-surface"


def test_ranges_match_point_norms(built_scene):
    sensor = make_lidar(built_scene, pattern="vlp32", tilt_down_deg=20.0)
    R, t = sensor.pose_for(np.array([0.0, 0.0, 0.8]), rotation=0.0)
    cap = sensor.capture(R, t)
    np.testing.assert_allclose(np.linalg.norm(cap.points, axis=1), cap.ranges, rtol=1e-4)
    assert cap.ranges.min() >= sensor.min_range
    assert cap.ranges.max() <= sensor.max_range


def test_robot_shell_is_excluded(built_scene):
    """With the shell parked on the sensor, only the exclusion keeps the view."""
    import mujoco

    _, model, data, robot_id = built_scene("flat")
    base = np.array([0.0, 0.0, 0.8])
    # Co-locate the shell with the sensor, as it is on a real robot.
    data.mocap_pos[scenes.mocap_id(model, scenes.ROBOT_BODY)] = base
    mujoco.mj_forward(model, data)

    excluded = lidar.LidarSensor(model=model, data=data, pattern="vlp32",
                                 bodyexclude=robot_id, tilt_down_deg=20.0, max_range=12.0)
    included = lidar.LidarSensor(model=model, data=data, pattern="vlp32",
                                 bodyexclude=-1, tilt_down_deg=20.0, max_range=12.0)
    R, t = excluded.pose_for(base, rotation=0.0)
    a, b = excluded.capture(R, t), included.capture(R, t)

    assert a.ranges.max() > 2.0, "the ground should be visible out to several metres"
    # Every ray is stopped by the shell it sits inside. The CPU path reports
    # those interior hits at the shell's own half-extents; Warp culls backfaces
    # and reports nothing at all. Either way, the terrain is gone.
    assert b.points.shape[0] == 0 or b.ranges.max() < 0.5, (
        f"the shell should block the view when not excluded, got "
        f"{b.points.shape[0]} returns out to {b.ranges.max() if b.points.shape[0] else 0:.2f} m"
    )

    data.mocap_pos[scenes.mocap_id(model, scenes.ROBOT_BODY)] = np.array([0.0, 0.0, 1.0])
    mujoco.mj_forward(model, data)


def test_livox_patterns_are_non_repetitive(built_scene):
    """Successive Livox scans must differ -- that is the point of the rosette."""
    sensor = make_lidar(built_scene, pattern="mid360", tilt_down_deg=25.0)
    R, t = sensor.pose_for(np.array([0.0, 0.0, 0.8]), rotation=0.0)
    first, second = sensor.capture(R, t), sensor.capture(R, t)
    assert first.points.shape != second.points.shape or not np.allclose(
        first.points, second.points
    )


def test_spinning_patterns_are_deterministic(built_scene):
    sensor = make_lidar(built_scene, pattern="vlp32", tilt_down_deg=20.0)
    R, t = sensor.pose_for(np.array([0.0, 0.0, 0.8]), rotation=0.0)
    np.testing.assert_array_equal(sensor.capture(R, t).points, sensor.capture(R, t).points)


def test_pose_for_applies_yaw_and_tilt(built_scene):
    sensor = make_lidar(built_scene, pattern="vlp32", tilt_down_deg=30.0,
                        mount_offset_body=(0.25, 0.0, 0.1))
    R, t = sensor.pose_for(np.array([1.0, 2.0, 0.8]), rotation=np.pi / 2)
    # Facing +y, so the forward mount offset lands on +y and the z offset on z.
    np.testing.assert_allclose(t, [1.0, 2.25, 0.9], atol=1e-12)
    forward = R @ [1.0, 0.0, 0.0]  # body forward in world
    assert forward[1] > 0 and forward[2] == pytest.approx(-np.sin(np.deg2rad(30.0)), abs=1e-9)


def test_capture_reports_the_site_pose_it_used(built_scene):
    """The capture carries the pose MuJoCo actually placed, not the request."""
    sensor = make_lidar(built_scene, pattern="vlp32", tilt_down_deg=20.0)
    R, t = sensor.pose_for(np.array([0.4, -0.3, 0.8]), rotation=1.1)
    cap = sensor.capture(R, t)
    np.testing.assert_allclose(cap.t_wc, t, atol=1e-9)
    np.testing.assert_allclose(cap.R_wc, R, atol=1e-6)


def test_noise_is_applied_along_the_ray(built_scene):
    from emsim.sensor import SensorNoise

    clean = make_lidar(built_scene, pattern="vlp32", tilt_down_deg=20.0)
    noisy = make_lidar(built_scene, pattern="vlp32", tilt_down_deg=20.0,
                       noise=SensorNoise(range_absolute_std=0.02, seed=11))
    R, t = clean.pose_for(np.array([0.0, 0.0, 0.8]), rotation=0.0)
    a, b = clean.capture(R, t), noisy.capture(R, t)

    assert a.points.shape == b.points.shape
    delta = b.ranges - a.ranges
    assert delta.std() == pytest.approx(0.02, rel=0.2)
    # Perturbing range must not rotate the ray.
    dir_a = a.points / np.linalg.norm(a.points, axis=1, keepdims=True)
    dir_b = b.points / np.linalg.norm(b.points, axis=1, keepdims=True)
    np.testing.assert_allclose(dir_a, dir_b, atol=1e-3)


def test_unknown_pattern_is_rejected(built_scene):
    with pytest.raises(KeyError, match="unknown LiDAR pattern"):
        make_lidar(built_scene, pattern="not_a_lidar")


def test_unknown_backend_is_rejected(built_scene):
    with pytest.raises(ValueError, match="unknown LiDAR backend"):
        make_lidar(built_scene, backend="not_a_backend")


def test_auto_backend_resolves(built_scene):
    """``auto`` picks Warp when CUDA is there, cpu otherwise -- never ``auto``."""
    resolved = lidar.resolve_backend("auto")
    assert resolved in ("warp", "cpu")
    assert resolved == ("warp" if lidar.warp_available() else "cpu")
    sensor = make_lidar(built_scene, pattern="vlp32", backend="auto")
    assert sensor.backend == resolved, "the sensor must record the concrete backend"


def test_default_backend_is_warp_when_cuda_is_present(built_scene, capsys):
    """Guards against a silent fall back to the CPU path.

    Everything here runs on whatever ``auto`` resolves to, and CPU ray casting
    is up to 884x slower on height-field scenes -- so a fallback that nobody
    notices is the failure worth catching.
    """
    sensor = make_lidar(built_scene, pattern="vlp32")  # no backend: take the default
    with capsys.disabled():
        print(f"\n  LiDAR ray-cast backend in use: {sensor.backend}")
    if lidar.warp_available():
        assert sensor.backend == "warp", (
            "warp-lang reports a CUDA device but the default backend resolved to "
            f"'{sensor.backend}'"
        )
    else:
        assert sensor.backend == "cpu"


needs_warp = pytest.mark.skipif(
    not lidar.warp_available(), reason="warp-lang with a CUDA device is not available"
)


@needs_warp
@pytest.mark.parametrize("pattern", ["vlp32", "os128"])
def test_warp_matches_cpu(pattern, built_scene):
    """The GPU path must agree with the reference CPU path, ray for ray."""
    base = np.array([0.0, 0.0, 0.8])
    captures = {}
    for backend in ("cpu", "warp"):
        sensor = make_lidar(built_scene, pattern=pattern, backend=backend, tilt_down_deg=20.0)
        captures[backend] = sensor.capture(*sensor.pose_for(base, rotation=0.7))

    cpu, warp = captures["cpu"], captures["warp"]
    assert cpu.points.shape == warp.points.shape, "the two backends dropped different rays"
    # float32 ray casting on two different devices; agreement is to that precision.
    np.testing.assert_allclose(warp.ranges, cpu.ranges, atol=1e-4)
    np.testing.assert_allclose(warp.points, cpu.points, atol=1e-4)


@needs_warp
def test_warp_is_faster_than_cpu(built_scene):
    """The whole point of the backend. Warp measures ~10x on an Orin."""
    import time

    base = np.array([0.0, 0.0, 0.8])
    timings = {}
    for backend in ("cpu", "warp"):
        sensor = make_lidar(built_scene, pattern="os128", backend=backend, tilt_down_deg=20.0)
        R, t = sensor.pose_for(base, rotation=0.0)
        sensor.capture(R, t)  # warm up: Warp compiles its kernels on first use
        start = time.perf_counter()
        for _ in range(3):
            sensor.capture(R, t)
        timings[backend] = (time.perf_counter() - start) / 3

    speedup = timings["cpu"] / timings["warp"]
    assert speedup > 2.0, (
        f"warp {timings['warp'] * 1e3:.1f} ms vs cpu {timings['cpu'] * 1e3:.1f} ms "
        f"is only {speedup:.1f}x"
    )


def test_lidar_mount_is_a_geomless_mocap_body(built_scene):
    """It must be poseable and must never occlude anything."""
    _, model, data, _ = built_scene("flat")
    mid = scenes.mocap_id(model, scenes.LIDAR_BODY)
    assert mid >= 0
    import mujoco

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scenes.LIDAR_BODY)
    assert model.body_geomnum[body] == 0, "the LiDAR mount must carry no geoms"


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #

# pattern -> (min coverage, max rmse) within 2.5 m after a 16-frame sweep.
LIDAR_BUDGETS = {"vlp32": (0.70, 0.030), "os128": (0.60, 0.025), "avia": (0.80, 0.025)}


@requires_gpu
@pytest.mark.parametrize("pattern", sorted(LIDAR_BUDGETS))
def test_lidar_drives_the_map_to_ground_truth(pattern):
    from emsim.runner import RunConfig, run

    min_coverage, max_rmse = LIDAR_BUDGETS[pattern]
    tilt = 25.0 if pattern in LIVOX else 20.0
    result = run(
        RunConfig(
            scene="mixed",
            sensor="lidar",
            lidar_pattern=pattern,
            lidar_tilt_down_deg=tilt,
            trajectory="spin",
            n_steps=16,
            resolution=0.04,
            map_length=10.0,
            max_range=12.0,
        )
    )
    assert result.sensor_backend == lidar.resolve_backend("auto"), (
        f"the run used '{result.sensor_backend}' rather than the default backend"
    )
    err = result.error("elevation", radius=2.5)
    assert err.coverage >= min_coverage, f"{pattern}: {err}"
    assert err.rmse <= max_rmse, f"{pattern}: {err}"


@requires_gpu
def test_lidar_and_camera_agree_on_the_same_terrain():
    """Two very different samplings of one scene must produce the same surface."""
    from emsim.runner import RunConfig, run

    common = dict(scene="rough", trajectory="spin", n_steps=16, resolution=0.04,
                  map_length=8.0, max_range=12.0)
    cam = run(RunConfig(sensor="camera", **common))
    lid = run(RunConfig(sensor="lidar", lidar_pattern="os128", lidar_tilt_down_deg=20.0, **common))

    both = np.isfinite(cam.layers["elevation"]) & np.isfinite(lid.layers["elevation"])
    assert both.sum() > 2000, "the two sensors should share plenty of mapped cells"
    diff = np.abs(cam.layers["elevation"][both] - lid.layers["elevation"][both])
    assert np.percentile(diff, 95) < 0.05, f"p95 camera-vs-lidar disagreement {np.percentile(diff, 95):.4f} m"
