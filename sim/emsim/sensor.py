"""Ray-cast depth sensor.

A pinhole depth camera is simulated with ``mj_multiRay``: all pixel rays share
the camera origin, which is exactly what that entry point is for. This keeps the
sensor exact and GPU/GL-free, so the harness runs headless.

Frames
------
* **body** -- x forward, y left, z up (the usual robot convention).
* **optical** -- x right, y down, z forward (the usual camera convention).

``input_pointcloud`` wants points in the sensor frame plus the rotation and
translation that carry them into the map frame, so :meth:`DepthSensor.capture`
returns optical-frame points alongside ``R_wc`` / ``t_wc``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Optical frame expressed in body axes: +z_opt -> +x_body, +x_opt -> -y_body,
# +y_opt -> -z_body.
R_BODY_FROM_OPTICAL = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(pitch: float) -> np.ndarray:
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics derived from a vertical field of view."""

    width: int = 160
    height: int = 120
    fovy_deg: float = 60.0

    @property
    def fy(self) -> float:
        return 0.5 * self.height / np.tan(0.5 * np.deg2rad(self.fovy_deg))

    @property
    def fx(self) -> float:
        return self.fy  # square pixels

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    def ray_directions(self) -> np.ndarray:
        """Unit ray directions in the optical frame, shape ``(H * W, 3)``."""
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing="xy")
        dirs = np.stack(
            [(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u, dtype=np.float64)],
            axis=-1,
        ).reshape(-1, 3)
        return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


@dataclass
class SensorNoise:
    """Depth degradation applied to the ideal ray-cast ranges."""

    range_relative_std: float = 0.0  # std of range error as a fraction of range
    range_absolute_std: float = 0.0  # constant-term std, metres
    dropout: float = 0.0  # fraction of pixels discarded at random
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @property
    def enabled(self) -> bool:
        return self.range_relative_std > 0 or self.range_absolute_std > 0 or self.dropout > 0

    def apply(self, ranges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return noisy ranges and a keep-mask."""
        keep = np.ones(ranges.shape, dtype=bool)
        if self.dropout > 0:
            keep = self._rng.random(ranges.shape) >= self.dropout
        if self.range_relative_std > 0:
            ranges = ranges * (1.0 + self._rng.normal(0.0, self.range_relative_std, ranges.shape))
        if self.range_absolute_std > 0:
            ranges = ranges + self._rng.normal(0.0, self.range_absolute_std, ranges.shape)
        return ranges, keep


@dataclass
class DepthCapture:
    """One depth frame, ready to hand to ``ElevationMap.input_pointcloud``."""

    points: np.ndarray  # (N, 3) in the optical frame
    R_wc: np.ndarray  # (3, 3) optical -> world
    t_wc: np.ndarray  # (3,) camera origin in world
    ranges: np.ndarray  # (N,) range per returned point
    n_pixels: int  # rays cast, including misses

    @property
    def world_points(self) -> np.ndarray:
        """The same points expressed in world coordinates."""
        return self.points @ self.R_wc.T + self.t_wc


class DepthSensor:
    """A pinhole depth camera that renders by ray casting into a MuJoCo scene."""

    def __init__(
        self,
        model,
        data,
        intrinsics: Optional[CameraIntrinsics] = None,
        max_range: float = 6.0,
        min_range: float = 0.05,
        bodyexclude: int = -1,
        noise: Optional[SensorNoise] = None,
        tilt_down_deg: float = 45.0,
        mount_offset_body: Tuple[float, float, float] = (0.2, 0.0, 0.0),
    ):
        self.model = model
        self.data = data
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        self.bodyexclude = int(bodyexclude)
        self.noise = noise or SensorNoise()
        self.tilt_down_deg = float(tilt_down_deg)
        self.mount_offset_body = mount_offset_body

        self._dirs_cam = self.intrinsics.ray_directions()
        n = self._dirs_cam.shape[0]
        self._geomid = np.zeros(n, dtype=np.int32)
        self._dist = np.zeros(n, dtype=np.float64)

    @property
    def n_rays(self) -> int:
        return self._dirs_cam.shape[0]

    def pose_for(self, base_position: np.ndarray, yaw: float) -> Tuple[np.ndarray, np.ndarray]:
        """Sensor pose for a base at ``base_position`` with heading ``yaw``.

        Mirrors :meth:`emsim.lidar.LidarSensor.pose_for`, so the runner can
        drive either sensor through the same two calls.
        """
        return camera_pose(base_position, yaw, self.tilt_down_deg, self.mount_offset_body)

    def capture(self, R_wc: np.ndarray, t_wc: np.ndarray) -> DepthCapture:
        """Cast the full pixel bundle from pose ``(R_wc, t_wc)``.

        Args:
            R_wc: ``(3, 3)`` rotation taking optical-frame vectors to world.
            t_wc: ``(3,)`` camera origin in world coordinates.
        """
        import mujoco

        R_wc = np.ascontiguousarray(R_wc, dtype=np.float64)
        t_wc = np.ascontiguousarray(t_wc, dtype=np.float64)

        dirs_world = np.ascontiguousarray(self._dirs_cam @ R_wc.T)
        mujoco.mj_multiRay(
            self.model,
            self.data,
            t_wc,
            dirs_world.reshape(-1),
            None,
            1,
            self.bodyexclude,
            self._geomid,
            self._dist,
            None,
            self.n_rays,
            self.max_range,
        )

        ranges = self._dist.copy()
        hit = (self._geomid >= 0) & (ranges >= self.min_range) & (ranges <= self.max_range)
        ranges, keep = self.noise.apply(ranges)
        valid = hit & keep & (ranges > 0)

        points = self._dirs_cam[valid] * ranges[valid, None]
        return DepthCapture(
            points=points.astype(np.float32),
            R_wc=R_wc,
            t_wc=t_wc,
            ranges=ranges[valid],
            n_pixels=self.n_rays,
        )


def camera_pose(
    base_position: np.ndarray,
    yaw: float,
    tilt_down_deg: float = 35.0,
    offset_body: Tuple[float, float, float] = (0.2, 0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Camera pose for a base at ``base_position`` with heading ``yaw``.

    Args:
        base_position: ``(3,)`` robot base origin in world coordinates.
        yaw: Heading in radians.
        tilt_down_deg: Downward tilt of the optical axis. Positive looks at the
            ground; 0 looks at the horizon. (A right-handed rotation about the
            body +y axis takes +x toward -z, so this is ``rot_y(+tilt)``.)
        offset_body: Camera mount offset in the body frame.

    Returns:
        ``(R_wc, t_wc)`` -- optical-to-world rotation and camera origin.
    """
    R_wb = rot_z(yaw)
    R_wc = R_wb @ rot_y(np.deg2rad(tilt_down_deg)) @ R_BODY_FROM_OPTICAL
    t_wc = np.asarray(base_position, dtype=np.float64) + R_wb @ np.asarray(offset_body, dtype=np.float64)
    return R_wc, t_wc
