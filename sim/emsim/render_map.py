"""Rendered comparison of an elevation map against ground truth (`pixi run render-map`).

Runs a mapping simulation, then renders, through MuJoCo from the same cameras:
the scene itself, the ray-cast ground truth and the elevation map as one column
per cell (same height colours), and the per-cell error (map - truth). One
figure, an oblique row and a top-down row. Rendering is software (see
:mod:`emsim.render`), so a figure takes roughly half a minute.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

#: name -> (distance, azimuth, elevation) about the map centre.
VIEWS = {"oblique": (7.5, 145.0, -32.0), "top-down": (10.5, 90.0, -89.9)}


def cell_columns(values: np.ndarray, x: np.ndarray, y: np.ndarray, floor: float, resolution: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Box centres and half-sizes for one column per cell, from ``floor`` up to its value.

    Columns are at least 4 mm tall, so a cell at the floor still shows.
    """
    half = np.maximum((np.asarray(values, np.float64) - floor) / 2.0, 0.002)
    pos = np.stack([x, y, floor + half], axis=-1)
    size = np.stack([np.full_like(half, resolution / 2.0), np.full_like(half, resolution / 2.0), half], axis=-1)
    return pos, size


def render_map(result, out: Path, width: int = 960, height: int = 720) -> Path:
    """Write the comparison figure for a finished :func:`emsim.runner.run`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco

    from emsim import render, scenes

    cfg = result.config
    est, gt = result.layers["elevation"], result.ground_truth
    x, y = result.cell_centers()
    err = result.error("elevation")

    model, data = scenes.build_model(result.scene)
    # The robot shell would hide the map centre; this is a picture of the terrain.
    for body in (scenes.ROBOT_BODY, scenes.LIDAR_BODY):
        data.mocap_pos[scenes.mocap_id(model, body)] = [0.0, 0.0, -50.0]
    mujoco.mj_forward(model, data)
    renderer = render.make_renderer(model, height, width, max_geom=est.size + 1000)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [result.center[0], result.center[1], 0.1]
    floor = float(np.nanmin(gt)) - 0.03
    eye = np.eye(3).ravel()

    def frame(values=None, colors=None, mask=None) -> np.ndarray:
        option = mujoco.MjvOption()
        if values is not None:
            option.geomgroup[:] = 0  # columns only, no terrain
        renderer.update_scene(data, camera, scene_option=option)
        if values is not None:
            scn = renderer.scene
            pos, size = cell_columns(values[mask], x[mask], y[mask], floor, cfg.resolution)
            for p, s, c in zip(pos, size, colors[mask]):
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, s, p, eye,
                                    np.asarray(c, np.float32))
                scn.ngeom += 1
        return renderer.render()

    both = np.isfinite(est) & np.isfinite(gt)
    hnorm = matplotlib.colors.Normalize(*np.nanpercentile(gt[np.isfinite(gt)], [1, 99.5]))
    hcmap = matplotlib.colormaps["viridis"]
    elim = max(0.02, float(np.percentile(np.abs(est - gt)[both], 98))) if both.any() else 0.02
    enorm = matplotlib.colors.TwoSlopeNorm(0.0, -elim, elim)
    ecmap = matplotlib.colormaps["RdBu_r"]
    panels = [
        ("MuJoCo scene", lambda: frame()),
        ("Ground truth (ray cast)", lambda: frame(gt, hcmap(hnorm(gt)), np.isfinite(gt))),
        ("Elevation map", lambda: frame(est, hcmap(hnorm(est)), np.isfinite(est))),
        ("Error: map − truth", lambda: frame(est, ecmap(enorm(est - gt)), both)),
    ]

    fig = plt.figure(figsize=(22, 11.5))
    grid = fig.add_gridspec(2, 4, wspace=0.03, hspace=0.08, left=0.01, right=0.94, top=0.9, bottom=0.02)
    for r, (view, (distance, azimuth, elevation)) in enumerate(VIEWS.items()):
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        for c, (title, make) in enumerate(panels):
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(make())
            ax.set_title(f"{title} — {view}", fontsize=12)
            ax.axis("off")
    fig.colorbar(matplotlib.cm.ScalarMappable(hnorm, hcmap), cax=fig.add_axes([0.95, 0.52, 0.009, 0.34]),
                 label="height [m] (truth and map)")
    fig.colorbar(matplotlib.cm.ScalarMappable(enorm, ecmap), cax=fig.add_axes([0.95, 0.06, 0.009, 0.34]),
                 label="map − truth [m]")
    sensor = f"{cfg.lidar_pattern} LiDAR" if cfg.sensor == "lidar" else f"depth camera {cfg.cam_width}×{cfg.cam_height}"
    fig.suptitle(f"{cfg.scene}: {sensor}, {cfg.trajectory} path, {cfg.n_steps} frames, "
                 f"{cfg.map_length:g} m map at {cfg.resolution} m cells    {err}", fontsize=14)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    from emsim import scenes
    from emsim.lidar import PATTERNS
    from emsim.runner import TRAJECTORIES, RunConfig, run

    p = argparse.ArgumentParser(prog="emsim.render_map", description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="mixed", choices=sorted(scenes.SCENES))
    p.add_argument("--sensor", default="lidar", choices=("camera", "lidar"))
    p.add_argument("--lidar-pattern", default="vlp32", choices=list(PATTERNS))
    p.add_argument("--trajectory", default="circle", choices=TRAJECTORIES)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--resolution", type=float, default=0.04)
    p.add_argument("--map-length", type=float, default=8.0)
    p.add_argument("--out", type=Path, default=Path("sim/report"))
    args = p.parse_args(argv)

    cfg = RunConfig(scene=args.scene, sensor=args.sensor, lidar_pattern=args.lidar_pattern,
                    trajectory=args.trajectory, n_steps=args.steps, resolution=args.resolution,
                    map_length=args.map_length)
    result = run(cfg, layers=("elevation",))
    name = args.lidar_pattern if args.sensor == "lidar" else "camera"
    path = render_map(result, args.out / f"{args.scene}_{name}_map_vs_truth.png")
    print(result.error("elevation"))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    from emsim import render

    render.init_gl()
    raise SystemExit(main())
