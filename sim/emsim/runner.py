"""Drive ``ElevationMap`` over a simulated trajectory and score it against truth.

This is the piece that ties the other modules together:

    scene -> MuJoCo model -> ray-cast depth frames -> ElevationMap
                          \\-> ray-cast ground truth -> metrics

``cupy`` is imported lazily inside :func:`run` so that the scene, sensor and
ground-truth modules stay usable (and testable) on a machine with no GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from emsim import scenes
from emsim.heightmap import GroundTruthHeightmap, map_cell_centers, radial_mask
from emsim.metrics import MapError, Timings, compare_maps
from emsim.sensor import CameraIntrinsics, DepthSensor, SensorNoise, camera_pose, rot_z

CONFIG_DIR = Path(__file__).resolve().parents[2] / "elevation_mapping_cupy" / "configs"

TRAJECTORIES = ("static", "spin", "line", "circle")


@dataclass
class RunConfig:
    """Everything that defines one simulation run."""

    scene: str = "steps"

    # Map
    resolution: float = 0.04
    map_length: float = 8.0

    # Trajectory
    trajectory: str = "spin"
    n_steps: int = 24
    path_length: float = 3.0  # "line": distance travelled
    path_radius: float = 1.5  # "circle": radius
    start_xy: Tuple[float, float] = (0.0, 0.0)
    base_height: float = 0.8  # sensor carrier height above the terrain

    # Sensor. The tilt/FOV pair is chosen so the top image row still points
    # 15 deg below the horizon: near-horizontal rays skim across height fields
    # for metres and dominate ray-cast cost without adding usable coverage.
    cam_width: int = 160
    cam_height: int = 120
    cam_fovy_deg: float = 60.0
    cam_tilt_down_deg: float = 45.0  # positive looks at the ground
    cam_offset_body: Tuple[float, float, float] = (0.2, 0.0, 0.0)
    max_range: float = 6.0
    noise: SensorNoise = field(default_factory=SensorNoise)

    # Mapping behaviour (mirrors Parameter fields of the same name)
    enable_visibility_cleanup: bool = True
    enable_drift_compensation: bool = False
    enable_overlap_clearance: bool = True
    position_noise: float = 0.0
    orientation_noise: float = 0.0
    with_semantics: bool = False

    # Bookkeeping
    record_per_step: bool = False
    param_overrides: Dict[str, object] = field(default_factory=dict)

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(self.cam_width, self.cam_height, self.cam_fovy_deg)


@dataclass
class StepRecord:
    """The map as it stood after one trajectory step."""

    index: int
    base_position: np.ndarray
    center: np.ndarray  # (3,) map centre
    elevation: np.ndarray  # exported elevation layer
    ground_truth: np.ndarray  # ground truth for that same window
    error: MapError


@dataclass
class RunResult:
    """Final map state plus the ground truth it should be compared against."""

    config: RunConfig
    scene: scenes.Scene
    center: np.ndarray  # (3,) map centre after the last step
    cell_n: int  # side length of every exported layer
    layers: Dict[str, np.ndarray]
    ground_truth: np.ndarray  # same layout as layers["elevation"]
    poses: List[Tuple[np.ndarray, float]]
    n_points: List[int]
    timings: Timings
    traversability_enabled: bool = False
    steps: List[StepRecord] = field(default_factory=list)
    sampler: Optional[GroundTruthHeightmap] = None

    @property
    def per_step_error(self) -> List[MapError]:
        return [s.error for s in self.steps]

    def mask(self, radius: Optional[float] = None) -> Optional[np.ndarray]:
        """Region-of-interest mask, optionally limited to a radius about the centre."""
        if radius is None:
            return None
        return radial_mask(self.cell_n, self.config.resolution, radius)

    def error(self, layer: str = "elevation", radius: Optional[float] = None) -> MapError:
        """Score one exported layer against ground truth."""
        return compare_maps(self.layers[layer], self.ground_truth, self.mask(radius))

    def cell_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        return map_cell_centers(tuple(self.center[:2]), self.cell_n, self.config.resolution)


def make_trajectory(cfg: RunConfig, scene: scenes.Scene) -> List[Tuple[np.ndarray, float]]:
    """Build the list of ``(base_position, yaw)`` poses for a run."""
    n = cfg.n_steps
    x0, y0 = cfg.start_xy
    if cfg.trajectory == "static":
        xs, ys = np.full(n, x0), np.full(n, y0)
        yaws = np.zeros(n)
    elif cfg.trajectory == "spin":
        xs, ys = np.full(n, x0), np.full(n, y0)
        yaws = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    elif cfg.trajectory == "line":
        xs = x0 + np.linspace(0.0, cfg.path_length, n)
        ys = np.full(n, y0)
        yaws = np.zeros(n)
    elif cfg.trajectory == "circle":
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        xs = x0 + cfg.path_radius * np.cos(theta)
        ys = y0 + cfg.path_radius * np.sin(theta)
        yaws = theta + np.pi  # face the centre of the circle
    else:
        raise ValueError(f"unknown trajectory '{cfg.trajectory}'; available: {TRAJECTORIES}")

    zs = scene.analytic_height(xs, ys) + cfg.base_height
    return [(np.array([x, y, z], dtype=np.float64), float(yaw)) for x, y, z, yaw in zip(xs, ys, zs, yaws)]


def make_parameter(cfg: RunConfig):
    """Build a :class:`~elevation_mapping_cupy.parameter.Parameter` for a run."""
    from elevation_mapping_cupy.parameter import Parameter

    param = Parameter(
        resolution=cfg.resolution,
        map_length=cfg.map_length,
        use_chainer=False,
        weight_file=str(CONFIG_DIR / "weights.dat"),
        plugin_config_file=str(CONFIG_DIR / "plugin_config.yaml"),
        enable_visibility_cleanup=cfg.enable_visibility_cleanup,
        enable_drift_compensation=cfg.enable_drift_compensation,
        enable_overlap_clearance=cfg.enable_overlap_clearance,
    )
    if not cfg.with_semantics:
        # The default config registers "rgb"/"person" semantic layers that a
        # purely geometric run would never write to.
        param.subscriber_cfg = {}
    for key, value in cfg.param_overrides.items():
        if not hasattr(param, key):
            raise AttributeError(f"Parameter has no field '{key}'")
        setattr(param, key, value)
    param.update()
    return param


def ground_truth_bounds(poses: Sequence[Tuple[np.ndarray, float]], map_length: float, margin: float = 0.5):
    """Region the ground-truth sampler must cover for a given trajectory."""
    xy = np.array([p[0][:2] for p in poses])
    half = map_length / 2.0 + margin
    return (
        float(xy[:, 0].min() - half),
        float(xy[:, 0].max() + half),
        float(xy[:, 1].min() - half),
        float(xy[:, 1].max() + half),
    )


def run(
    cfg: RunConfig,
    layers: Sequence[str] = ("elevation",),
    scene: Optional[scenes.Scene] = None,
    sampler: Optional[GroundTruthHeightmap] = None,
) -> RunResult:
    """Execute a full simulated mapping run.

    Args:
        cfg: Run definition.
        layers: Map layers to export once the run finishes.
        scene: Pre-built scene; built from ``cfg.scene`` when omitted.
        sampler: Pre-built ground-truth sampler. Supplying a cached one across
            runs of the same scene saves the (static) ray-cast build.

    Returns:
        A :class:`RunResult` holding the final layers, ground truth and timings.
    """
    import cupy as cp
    from elevation_mapping_cupy.elevation_mapping import ElevationMap

    scene = scene or scenes.make_scene(cfg.scene)
    model, data = scenes.build_model(scene)
    robot_id = scenes.robot_body_id(model)
    poses = make_trajectory(cfg, scene)

    if sampler is None:
        sampler = GroundTruthHeightmap(
            model,
            data,
            ground_truth_bounds(poses, cfg.map_length),
            cfg.resolution,
            bodyexclude=robot_id,
        )
    elif abs(sampler.resolution - cfg.resolution) > 1e-12:
        raise ValueError(
            f"sampler resolution {sampler.resolution} does not match map resolution {cfg.resolution}; "
            "the ground-truth lookup would no longer be cell-exact"
        )

    sensor = DepthSensor(
        model,
        data,
        intrinsics=cfg.intrinsics,
        max_range=cfg.max_range,
        bodyexclude=robot_id,
        noise=cfg.noise,
    )

    param = make_parameter(cfg)
    em = ElevationMap(param)
    cell_n = param.true_cell_n
    scratch = np.zeros((cell_n, cell_n), dtype=np.float32)

    sensor_ms: List[float] = []
    input_ms: List[float] = []
    n_points: List[int] = []
    step_records: List[StepRecord] = []

    import mujoco

    for step, (base_pos, yaw) in enumerate(poses):
        # Park the (visual) sensor carrier so it is excluded consistently.
        data.mocap_pos[0] = base_pos
        mujoco.mj_forward(model, data)

        R_wc, t_wc = camera_pose(base_pos, yaw, cfg.cam_tilt_down_deg, cfg.cam_offset_body)

        t0 = time.perf_counter()
        capture = sensor.capture(R_wc, t_wc)
        sensor_ms.append((time.perf_counter() - t0) * 1e3)
        n_points.append(capture.points.shape[0])

        em.move_to(base_pos, cp.asarray(rot_z(yaw), dtype=param.data_type))

        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        if capture.points.shape[0]:
            em.input_pointcloud(
                cp.asarray(capture.points),
                ["x", "y", "z"],
                cp.asarray(R_wc, dtype=param.data_type),
                cp.asarray(t_wc, dtype=param.data_type),
                cfg.position_noise,
                cfg.orientation_noise,
            )
        em.update_variance()
        em.update_time()
        cp.cuda.Stream.null.synchronize()
        input_ms.append((time.perf_counter() - t0) * 1e3)

        if cfg.record_per_step:
            em.get_map_with_name_ref("elevation", scratch)
            step_center = cp.asnumpy(em.center).astype(np.float64)
            elevation = scratch.astype(np.float64)
            step_gt = sampler.sample_map_grid(tuple(step_center[:2]), cell_n, cfg.resolution)
            step_records.append(
                StepRecord(
                    index=step,
                    base_position=base_pos.copy(),
                    center=step_center,
                    elevation=elevation,
                    ground_truth=step_gt,
                    error=compare_maps(elevation, step_gt),
                )
            )

    export_ms: List[float] = []
    out: Dict[str, np.ndarray] = {}
    for name in layers:
        buf = np.zeros((cell_n, cell_n), dtype=np.float32)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        em.get_map_with_name_ref(name, buf)
        cp.cuda.Stream.null.synchronize()
        export_ms.append((time.perf_counter() - t0) * 1e3)
        out[name] = buf.astype(np.float64)

    center = cp.asnumpy(em.center).astype(np.float64)
    gt = sampler.sample_map_grid(tuple(center[:2]), cell_n, cfg.resolution)

    return RunResult(
        config=cfg,
        scene=scene,
        center=center,
        cell_n=cell_n,
        layers=out,
        ground_truth=gt,
        poses=poses,
        n_points=n_points,
        timings=Timings(np.array(sensor_ms), np.array(input_ms), np.array(export_ms)),
        traversability_enabled=em.traversability_filter is not None,
        steps=step_records,
        sampler=sampler,
    )
