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
from emsim.metrics import MapError
from emsim.runner import RunConfig, RunResult, TRAJECTORIES, run
from emsim.sensor import SensorNoise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emsim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="mixed", choices=sorted(scenes.SCENES), help="terrain to map")
    p.add_argument("--all", action="store_true", help="run every scene in the catalogue")
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


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    scene_names = sorted(scenes.SCENES) if args.all else [args.scene]
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
    print(f"\ntrajectory={args.trajectory} steps={args.steps} "
          f"map={args.map_length} m @ {args.resolution} m  scored within r <= {args.radius} m\n")
    print(f"{'scene':<{width}}  {'cover':>6}  {'rmse':>7}  {'mae':>7}  {'bias':>8}  "
          f"{'p95':>7}  {'pts/frame':>9}  {'input':>9}")
    for name, err, stats, points in rows:
        print(f"{name:<{width}}  {err.coverage:6.1%}  {err.rmse:7.4f}  {err.mae:7.4f}  "
              f"{err.bias:+8.4f}  {err.p95:7.4f}  {points:9d}  {stats['input_hz']:6.0f} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
