"""Command-line entry point: run a scene and report (or plot) the result.

    pixi run sim --scene steps
    pixi run sim --scene mixed --trajectory line --out report/
    pixi run sim --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from emsim import scenes
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
    return p


def make_config(args, scene: str) -> RunConfig:
    return RunConfig(
        scene=scene,
        trajectory=args.trajectory,
        n_steps=args.steps,
        resolution=args.resolution,
        map_length=args.map_length,
        noise=SensorNoise(
            range_relative_std=args.range_noise, dropout=args.dropout, seed=args.seed
        ),
    )


def plot(result: RunResult, path: Path, radius: Optional[float]) -> None:
    """Write a ground-truth / estimate / error triptych."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    est = result.layers["elevation"]
    truth = result.ground_truth
    error = est - truth
    if radius is not None:
        outside = ~result.mask(radius)
        error = np.where(outside, np.nan, error)

    half = result.cell_n * result.config.resolution / 2.0
    extent = [
        result.center[1] - half, result.center[1] + half,
        result.center[0] - half, result.center[0] + half,
    ]
    # Row 0 is max x and column 0 is max y, so flip both axes for a plot with
    # x up and y left -- the usual top-down view.
    def show(ax, data, title, **kw):
        im = ax.imshow(data, extent=extent, origin="upper", **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("y [m]")
        ax.invert_xaxis()
        plt.colorbar(im, ax=ax, fraction=0.046)

    vmin, vmax = np.nanmin(truth), np.nanmax(truth)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    show(axes[0], truth, "ground truth (ray cast)", vmin=vmin, vmax=vmax, cmap="terrain")
    show(axes[1], est, "elevation_mapping_cupy", vmin=vmin, vmax=vmax, cmap="terrain")
    lim = max(0.02, float(np.nanpercentile(np.abs(error), 99))) if np.isfinite(error).any() else 0.02
    show(axes[2], error, "estimate - truth [m]", vmin=-lim, vmax=lim, cmap="coolwarm")
    axes[0].set_ylabel("x [m]")

    err = result.error("elevation", radius=radius)
    fig.suptitle(f"{result.scene.name}: {result.scene.description}\n{err}", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    scene_names = sorted(scenes.SCENES) if args.all else [args.scene]
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]

    rows = []
    for name in scene_names:
        result = run(make_config(args, name), layers=layers)
        err: MapError = result.error("elevation", radius=args.radius)
        stats = result.timings.summary()
        rows.append((name, err, stats, int(np.mean(result.n_points))))
        if args.out is not None:
            out = args.out / f"{name}.png"
            plot(result, out, args.radius)
            print(f"wrote {out}")

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
