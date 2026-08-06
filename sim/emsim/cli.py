"""Command-line entry point: run a scene and report (or plot) the result.

    pixi run sim --scene steps
    pixi run sim --scene mixed --trajectory line --out report/
    pixi run sim --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from emsim import plotting, scenes
from emsim.lidar import PATTERNS as LIDAR_PATTERNS
from emsim.metrics import MapError
from emsim.runner import RunConfig, RunResult, TRAJECTORIES, run
from emsim.sensor import SensorNoise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emsim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="mixed", choices=sorted(scenes.SCENES), help="terrain to map")
    p.add_argument("--all", action="store_true", help="run every scene in the catalogue")
    p.add_argument(
        "--preview",
        action="store_true",
        help="draw the scene and one frame from each sensor, then exit (no mapping, no GPU)",
    )
    p.add_argument("--sensor", default="camera", choices=("camera", "lidar"))
    p.add_argument("--lidar-pattern", default="vlp32", choices=list(LIDAR_PATTERNS),
                   help="LiDAR scan pattern when --sensor lidar")
    p.add_argument("--lidar-tilt", type=float, default=20.0,
                   help="downward LiDAR mount tilt [deg]")
    p.add_argument("--lidar-backend", default="cpu", choices=("cpu", "warp", "taichi", "jax"),
                   help="mujoco-lidar backend; non-cpu needs the matching extra installed")
    p.add_argument("--trajectory", default="spin", choices=TRAJECTORIES)
    p.add_argument("--steps", type=int, default=24, help="number of frames")
    p.add_argument("--resolution", type=float, default=0.04, help="map resolution [m]")
    p.add_argument("--map-length", type=float, default=8.0, help="map side length [m]")
    p.add_argument("--radius", type=float, default=2.5, help="scoring radius about the centre [m]")
    p.add_argument("--range-noise", type=float, default=0.0, help="relative depth noise std")
    p.add_argument("--dropout", type=float, default=0.0, help="fraction of pixels dropped")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", default="elevation", help="comma-separated layers to export")
    p.add_argument("--out", type=Path, default=None, help="directory for PNG plots")
    p.add_argument(
        "--plots",
        default="comparison",
        help="comma-separated: comparison, layers, surface, convergence, filmstrip, all. "
             "convergence and filmstrip need a moving trajectory and record per-step state.",
    )
    return p


ALL_PLOTS = ("comparison", "layers", "surface", "convergence", "filmstrip")

#: Plots built from per-step snapshots, which the run has to be told to keep.
PER_STEP_PLOTS = {"convergence", "filmstrip"}


def resolve_plots(spec: str) -> List[str]:
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    if "all" in requested:
        return list(ALL_PLOTS)
    unknown = set(requested) - set(ALL_PLOTS)
    if unknown:
        raise SystemExit(f"unknown plot(s) {sorted(unknown)}; available: {list(ALL_PLOTS) + ['all']}")
    return requested


def make_config(args, scene: str, plots: Sequence[str] = ()) -> RunConfig:
    return RunConfig(
        scene=scene,
        trajectory=args.trajectory,
        n_steps=args.steps,
        resolution=args.resolution,
        map_length=args.map_length,
        record_per_step=bool(set(plots) & PER_STEP_PLOTS),
        sensor=args.sensor,
        lidar_pattern=args.lidar_pattern,
        lidar_tilt_down_deg=args.lidar_tilt,
        lidar_backend=args.lidar_backend,
        noise=SensorNoise(
            range_relative_std=args.range_noise, dropout=args.dropout, seed=args.seed
        ),
    )


def write_plots(result: RunResult, out: Path, kinds: Sequence[str], radius: Optional[float]) -> List[Path]:
    """Render the requested figures for one run."""
    written = []
    for kind in kinds:
        target = out / f"{result.scene.name}_{kind}.png"
        if kind == "comparison":
            written.append(plotting.plot_comparison(result, target, radius))
        elif kind == "layers":
            written.append(plotting.plot_layers(result, target))
        elif kind == "surface":
            written.append(plotting.plot_surface(result, target, radius))
        elif kind == "convergence":
            written.append(plotting.plot_convergence(result, target))
        elif kind == "filmstrip":
            written.append(plotting.plot_filmstrip(result, target))
    return written


def preview_scene(name: str, out: Path, resolution: float = 0.05, half_extent: float = 4.0) -> Path:
    """Draw a scene and one frame from each sensor. No mapping, no GPU."""
    import mujoco

    from emsim.heightmap import GroundTruthHeightmap
    from emsim.sensor import CameraIntrinsics, DepthSensor

    scene = scenes.make_scene(name)
    model, data = scenes.build_model(scene)
    robot_id = scenes.robot_body_id(model)
    base = np.array([0.0, 0.0, 0.8])
    data.mocap_pos[scenes.mocap_id(model, scenes.ROBOT_BODY)] = base
    mujoco.mj_forward(model, data)

    bounds = (-half_extent, half_extent, -half_extent, half_extent)
    sampler = GroundTruthHeightmap(model, data, bounds, resolution, bodyexclude=robot_id)

    camera = DepthSensor(model, data, CameraIntrinsics(160, 120, 60.0),
                         max_range=8.0, bodyexclude=robot_id)
    captures = [("depth camera", camera.capture(*camera.pose_for(base, 0.0)))]

    from emsim import lidar as lidar_mod

    if lidar_mod.is_available():
        unit = lidar_mod.LidarSensor(model=model, data=data, pattern="vlp32",
                                     tilt_down_deg=20.0, max_range=12.0, bodyexclude=robot_id)
        captures.append(("lidar vlp32 (one scan)", unit.capture(*unit.pose_for(base, 0.0))))

    return plotting.plot_scene_preview(
        scene, sampler, out / f"{name}_scene.png", captures=captures, base_position=base
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    scene_names = sorted(scenes.SCENES) if args.all else [args.scene]

    if args.preview:
        if args.out is None:
            raise SystemExit("--preview needs --out DIR")
        for name in scene_names:
            print(f"wrote {preview_scene(name, args.out)}")
        return 0

    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    plots = resolve_plots(args.plots) if args.out is not None else []
    if "layers" in plots and len(layers) == 1:
        # A one-layer sheet is not worth drawing; show what the map offers.
        layers = ["elevation", "variance", "is_valid", "upper_bound", "inpaint", "smooth"]

    rows = []
    for name in scene_names:
        result = run(make_config(args, name, plots), layers=layers)
        err: MapError = result.error("elevation", radius=args.radius)
        stats = result.timings.summary()
        rows.append((name, err, stats, int(np.mean(result.n_points))))
        for path in write_plots(result, args.out, plots, args.radius):
            print(f"wrote {path}")

    width = max(len(n) for n in scene_names)
    print(f"\nsensor={args.sensor}"
          + (f" ({args.lidar_pattern}, tilt {args.lidar_tilt:.0f} deg)" if args.sensor == "lidar" else "")
          + f"  trajectory={args.trajectory} steps={args.steps} "
          f"map={args.map_length} m @ {args.resolution} m  scored within r <= {args.radius} m\n")
    print(f"{'scene':<{width}}  {'cover':>6}  {'rmse':>7}  {'mae':>7}  {'bias':>8}  "
          f"{'p95':>7}  {'pts/frame':>9}  {'input':>9}")
    for name, err, stats, points in rows:
        print(f"{name:<{width}}  {err.coverage:6.1%}  {err.rmse:7.4f}  {err.mae:7.4f}  "
              f"{err.bias:+8.4f}  {err.p95:7.4f}  {points:9d}  {stats['input_hz']:6.0f} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
