"""Optional LiDAR sensor backend, built on `mujoco-lidar`_.

A depth camera samples the ground on a dense, near-uniform grid. A real LiDAR
does not: it samples along rings or a non-repetitive rosette, with density
falling off sharply with range and varying strongly with elevation angle. That
is the input `elevation_mapping_cupy` actually receives on a legged robot, so
this backend exists to put the map under that kind of load.

It presents the same interface as :class:`~emsim.sensor.DepthSensor` --
``pose_for`` then ``capture`` returning a :class:`~emsim.sensor.DepthCapture` --
so :func:`emsim.runner.run` does not care which one it is holding.

Frames
------
``mujoco-lidar`` takes the sensor pose from a *site* in the model rather than
as an argument, so :meth:`LidarSensor.capture` poses the geom-free
:data:`~emsim.scenes.LIDAR_BODY` mocap body and lets MuJoCo place the site.
``get_hit_points`` then returns points in the sensor frame, which is exactly
what ``input_pointcloud`` wants alongside the site's world rotation.

.. _mujoco-lidar: https://github.com/discoverse-dev/MuJoCo-LiDAR
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from emsim import scenes
from emsim.sensor import DepthCapture, SensorNoise, rot_y, rot_z


class LidarUnavailable(RuntimeError):
    """Raised when the optional ``mujoco-lidar`` dependency is missing."""


def is_available() -> bool:
    """Whether the optional LiDAR backend can be used."""
    try:
        import mujoco_lidar  # noqa: F401

        return True
    except Exception:  # pragma: no cover - depends on the environment
        return False


def warp_available() -> bool:
    """Whether the GPU (Warp) ray-casting backend can be used.

    Importing ``warp`` is not enough -- it initialises happily with only a CPU
    device, and the backend needs CUDA.
    """
    try:
        import warp as wp

        wp.init()
        return len(wp.get_cuda_devices()) > 0
    except Exception:  # pragma: no cover - depends on the environment
        return False


BACKENDS = ("auto", "cpu", "warp", "taichi", "jax")


def resolve_backend(backend: str) -> str:
    """Turn ``"auto"`` into a concrete backend.

    ``auto`` prefers Warp, which casts roughly an order of magnitude faster than
    the CPU path, and falls back to ``cpu`` when CUDA or ``warp-lang`` is
    missing. Other names pass through untouched.
    """
    if backend != "auto":
        return backend
    return "warp" if warp_available() else "cpu"


#: Scan patterns, by name. Each entry builds a callable returning
#: ``(theta, phi)`` ray angles in the sensor frame. Livox units are
#: non-repetitive: successive calls return a different rosette, which is the
#: whole point of them, so those are stateful generators.
def _pattern_factories() -> Dict[str, Callable[[], Tuple[np.ndarray, np.ndarray]]]:
    from mujoco_lidar import scan_gen

    def livox(name: str):
        gen = scan_gen.LivoxGenerator(name)
        return gen.sample_ray_angles

    return {
        "vlp32": scan_gen.generate_vlp32,
        "hdl64": scan_gen.generate_HDL64,
        "os128": scan_gen.generate_os128,
        "airy96": scan_gen.generate_airy96,
        "mid360": livox("mid360"),
        "avia": livox("avia"),
        "mid70": livox("mid70"),
        "horizon": livox("horizon"),
        "grid": lambda: scan_gen.generate_grid_scan_pattern(512, 64),
    }


PATTERNS = (
    "vlp32", "hdl64", "os128", "airy96", "mid360", "avia", "mid70", "horizon", "grid",
)


@dataclass
class LidarSensor:
    """A LiDAR that scans a MuJoCo scene through ``mujoco-lidar``.

    Args:
        model: ``mujoco.MjModel`` from :func:`emsim.scenes.build_model`.
        data: ``mujoco.MjData`` for that model.
        pattern: One of :data:`PATTERNS`.
        max_range: Returns beyond this are dropped.
        min_range: Returns closer than this are dropped.
        bodyexclude: Body id the scan ignores -- the robot shell, which would
            otherwise swallow every ray from the inside.
        backend: One of :data:`BACKENDS`. ``"auto"`` (the default) uses Warp
            when CUDA is available and falls back to ``"cpu"``. Warp casts
            roughly 10x faster; its kernels compile once (~14 s) and are cached
            in ``~/.cache/warp`` thereafter. Results agree with ``cpu`` to
            float32 precision.
        tilt_down_deg: Downward mount tilt. Several units (the Livox Mid-360
            among them) have a mostly-upward vertical FOV, so a ground-mapping
            mount tilts them forward.
        mount_offset_body: Mount offset in the body frame.
        noise: Range degradation, shared with the depth camera.
    """

    model: object
    data: object
    pattern: str = "vlp32"
    max_range: float = 10.0
    min_range: float = 0.2
    bodyexclude: int = -1
    backend: str = "auto"
    tilt_down_deg: float = 0.0
    mount_offset_body: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    noise: Optional[SensorNoise] = None

    def __post_init__(self) -> None:
        if not is_available():
            raise LidarUnavailable(
                "the LiDAR backend needs the optional 'mujoco-lidar' package "
                "(pixi already declares it; run `pixi install`)"
            )
        from mujoco_lidar import MjLidarWrapper

        if self.backend not in BACKENDS:
            raise ValueError(f"unknown LiDAR backend '{self.backend}'; expected one of {list(BACKENDS)}")
        self.backend = resolve_backend(self.backend)
        if self.pattern not in PATTERNS:
            raise KeyError(f"unknown LiDAR pattern '{self.pattern}'; available: {list(PATTERNS)}")

        self.noise = self.noise or SensorNoise()
        self._angles = _pattern_factories()[self.pattern]
        self._mocap = scenes.mocap_id(self.model, scenes.LIDAR_BODY)
        self._wrapper = MjLidarWrapper(
            self.model,
            site_name=scenes.LIDAR_SITE,
            backend=self.backend,
            cutoff_dist=self.max_range,
            args={"bodyexclude": self.bodyexclude},
        )
        theta, _ = self._angles()
        self._rays_per_scan = int(theta.size)

    @property
    def n_rays(self) -> int:
        """Rays per scan. Livox patterns report the size of one sample window."""
        return self._rays_per_scan

    def pose_for(
        self, base_position: np.ndarray, yaw: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sensor pose for a base at ``base_position`` with heading ``yaw``.

        Unlike the camera there is no optical-frame convention here: the LiDAR
        frame is the body frame (x forward, y left, z up), optionally tilted.
        """
        R_wb = rot_z(yaw)
        R_ws = R_wb @ rot_y(np.deg2rad(self.tilt_down_deg))
        t_ws = np.asarray(base_position, dtype=np.float64) + R_wb @ np.asarray(
            self.mount_offset_body, dtype=np.float64
        )
        return R_ws, t_ws

    def capture(self, R_ws: np.ndarray, t_ws: np.ndarray) -> DepthCapture:
        """Run one scan from pose ``(R_ws, t_ws)``."""
        import mujoco

        R_ws = np.ascontiguousarray(R_ws, dtype=np.float64)
        t_ws = np.ascontiguousarray(t_ws, dtype=np.float64)

        self.data.mocap_pos[self._mocap] = t_ws
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, R_ws.reshape(-1))
        self.data.mocap_quat[self._mocap] = quat
        mujoco.mj_forward(self.model, self.data)

        theta, phi = self._angles()
        ranges = np.asarray(self._wrapper.trace_rays(self.data, theta, phi), dtype=np.float64)
        points = np.asarray(self._wrapper.get_hit_points(), dtype=np.float64)

        hit = (ranges >= self.min_range) & (ranges <= self.max_range)
        ranges_noisy, keep = self.noise.apply(ranges)
        valid = hit & keep & (ranges_noisy > 0)

        # Rescale along each ray so the noise lands on range, not on direction.
        scale = np.divide(
            ranges_noisy, ranges, out=np.ones_like(ranges), where=ranges > 0
        )
        points = points[valid] * scale[valid, None]

        # MuJoCo places the site from the mocap pose, so its rotation is the one
        # the hit points are expressed in -- read it back rather than assuming.
        R_site = np.asarray(self._wrapper.sensor_rotation, dtype=np.float64).reshape(3, 3)
        t_site = np.asarray(self._wrapper.sensor_position, dtype=np.float64)

        return DepthCapture(
            points=points.astype(np.float32),
            R_wc=R_site,
            t_wc=t_site,
            ranges=ranges_noisy[valid],
            n_pixels=int(ranges.size),
        )
