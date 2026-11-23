"""Figures for simulation runs.

Every function takes a :class:`~emsim.runner.RunResult` and writes a PNG. They
back both the CLI and the ``--viz-dir`` pytest option, so a test can emit a
picture of exactly what it asserted.

Matplotlib is imported lazily and forced onto the Agg backend: the harness runs
headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# Layers measured in metres, drawn on a terrain colour ramp and shared scale.
HEIGHT_LAYERS = {"elevation", "upper_bound", "min_filter", "smooth", "inpaint"}


def _agg_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _extent(result) -> List[float]:
    """Axis extent in world metres for ``imshow`` of an exported layer."""
    half = result.cell_n * result.config.resolution / 2.0
    return [
        result.center[1] - half, result.center[1] + half,
        result.center[0] - half, result.center[0] + half,
    ]


def _draw(ax, data, extent, title, plt, **kw):
    """Top-down view with x up and y to the left, matching the map layout."""
    im = ax.imshow(data, extent=extent, origin="upper", **kw)
    ax.set_title(title, fontsize=9)
    ax.invert_xaxis()  # +y to the left
    plt.colorbar(im, ax=ax, fraction=0.046)
    return im


def _save(fig, path: Path, plt) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_comparison(result, path: Path, radius: Optional[float] = None) -> Path:
    """Ground truth, estimate and signed error, side by side."""
    plt = _agg_pyplot()
    est, truth = result.layers["elevation"], result.ground_truth
    error = est - truth
    if radius is not None:
        error = np.where(result.mask(radius), error, np.nan)

    extent = _extent(result)
    vmin, vmax = np.nanmin(truth), np.nanmax(truth)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    _draw(axes[0], truth, extent, "ground truth (ray cast)", plt,
          vmin=vmin, vmax=vmax, cmap="terrain")
    _draw(axes[1], est, extent, "elevation_mapping_cupy", plt,
          vmin=vmin, vmax=vmax, cmap="terrain")
    lim = max(0.02, float(np.nanpercentile(np.abs(error), 99))) if np.isfinite(error).any() else 0.02
    _draw(axes[2], error, extent, "estimate - truth [m]", plt, vmin=-lim, vmax=lim, cmap="coolwarm")
    for ax in axes:
        ax.set_xlabel("y [m]")
    axes[0].set_ylabel("x [m]")

    err = result.error("elevation", radius=radius)
    fig.suptitle(f"{result.scene.name}: {result.scene.description}\n{err}", fontsize=9)
    return _save(fig, path, plt)


def plot_layers(result, path: Path, layers: Optional[Sequence[str]] = None) -> Path:
    """Every exported layer on one sheet -- what ``test_map_layers`` inspects."""
    plt = _agg_pyplot()
    names = list(layers or result.layers)
    extent = _extent(result)

    heights = [result.layers[n] for n in names if n in HEIGHT_LAYERS]
    if heights:
        stacked = np.concatenate([h[np.isfinite(h)].ravel() for h in heights]) if heights else None
        hmin, hmax = (float(np.min(stacked)), float(np.max(stacked))) if stacked.size else (0.0, 1.0)

    cols = min(4, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.9 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        data = result.layers[name]
        if name in HEIGHT_LAYERS:
            _draw(ax, data, extent, f"{name} [m]", plt, vmin=hmin, vmax=hmax, cmap="terrain")
        else:
            _draw(ax, data, extent, name, plt, cmap="viridis")
        ax.set_xlabel("y [m]")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")

    fig.suptitle(f"{result.scene.name}: exported layers", fontsize=10)
    return _save(fig, path, plt)


def plot_convergence(result, path: Path) -> Path:
    """Coverage and error against frame number.

    Requires ``RunConfig.record_per_step``. This is the picture behind
    ``test_map_converges_as_frames_accumulate``.
    """
    if not result.steps:
        raise ValueError("run with RunConfig(record_per_step=True) to plot convergence")
    plt = _agg_pyplot()

    steps = [s.index for s in result.steps]
    coverage = [s.error.coverage * 100 for s in result.steps]
    rmse = [s.error.rmse for s in result.steps]
    p95 = [s.error.p95 for s in result.steps]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    ax0.plot(steps, coverage, marker="o", ms=3)
    ax0.set_xlabel("frame")
    ax0.set_ylabel("coverage [%]")
    ax0.set_title("map fill-in", fontsize=9)
    ax0.grid(alpha=0.3)

    ax1.plot(steps, rmse, marker="o", ms=3, label="rmse")
    ax1.plot(steps, p95, marker="s", ms=3, label="p95 abs")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("error [m]")
    ax1.set_title("accuracy vs ground truth", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    fig.suptitle(f"{result.scene.name} / {result.config.trajectory}: convergence", fontsize=10)
    return _save(fig, path, plt)


def plot_filmstrip(result, path: Path, n: int = 5) -> Path:
    """Successive map windows along the trajectory, with the robot track drawn on.

    The map is robot-centric, so each frame's window sits at a different world
    position -- this is the picture behind the map-shifting tests.
    """
    if not result.steps:
        raise ValueError("run with RunConfig(record_per_step=True) to plot a filmstrip")
    plt = _agg_pyplot()

    picks = np.unique(np.linspace(0, len(result.steps) - 1, n).astype(int))
    truth = np.concatenate([s.ground_truth[np.isfinite(s.ground_truth)].ravel() for s in result.steps])
    vmin, vmax = float(truth.min()), float(truth.max())

    track = np.array([s.base_position for s in result.steps])
    half = result.cell_n * result.config.resolution / 2.0
    centers = np.array([s.center for s in result.steps])
    # One shared world frame across all panels, so the window is visibly sliding
    # rather than each panel being re-centred on the robot.
    xlim = (centers[:, 0].min() - half, centers[:, 0].max() + half)
    ylim = (centers[:, 1].min() - half, centers[:, 1].max() + half)

    fig, axes = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 4.4), squeeze=False)
    for ax, k in zip(axes[0], picks):
        step = result.steps[k]
        extent = [
            step.center[1] - half, step.center[1] + half,
            step.center[0] - half, step.center[0] + half,
        ]
        im = ax.imshow(step.elevation, extent=extent, origin="upper",
                       vmin=vmin, vmax=vmax, cmap="terrain")
        # Outline of the window itself.
        ax.add_patch(
            plt.Rectangle((extent[0], extent[2]), 2 * half, 2 * half,
                          fill=False, ec="tab:red", lw=1.0, ls="--")
        )
        ax.plot(track[:, 1], track[:, 0], "-", color="0.35", lw=1)
        ax.plot(step.base_position[1], step.base_position[0], "o", color="red", ms=5)
        ax.set_title(f"frame {step.index}  cover {step.error.coverage:.0%}", fontsize=9)
        ax.set_xlabel("y [m]")
        ax.set_ylim(*xlim)
        ax.set_xlim(ylim[1], ylim[0])  # +y to the left
        ax.set_aspect("equal")
    axes[0][0].set_ylabel("x [m]")
    fig.colorbar(im, ax=axes[0], fraction=0.02, label="elevation [m]")

    fig.suptitle(
        f"{result.scene.name} / {result.config.trajectory}: robot-centric window follows the base",
        fontsize=10,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scene_preview(
    scene,
    sampler,
    path: Path,
    captures: Optional[Sequence] = None,
    base_position: Optional[np.ndarray] = None,
    stride: int = 3,
) -> Path:
    """What a scene actually looks like, and what the sensor gets back from it.

    Drawn from the scene's own geometry via the ray-cast height field rather
    than rendered through OpenGL -- there is no working GL stack in this
    environment, and this shows more of what matters anyway: the terrain, the
    sensor origin, and the returns the sensor produced.

    Args:
        scene: The :class:`~emsim.scenes.Scene`.
        sampler: A :class:`~emsim.heightmap.GroundTruthHeightmap` over it.
        captures: Optional ``(label, DepthCapture)`` pairs to overlay.
        base_position: Robot base, drawn as a marker.
    """
    plt = _agg_pyplot()
    X, Y = np.meshgrid(sampler.xs, sampler.ys, indexing="ij")
    Z = sampler.heights
    sl = (slice(None, None, stride), slice(None, None, stride))

    n_extra = len(captures or ())
    fig = plt.figure(figsize=(6.2 * (2 + n_extra), 5.2))
    ncols = 2 + n_extra

    ax = fig.add_subplot(1, ncols, 1, projection="3d")
    ax.plot_surface(X[sl], Y[sl], Z[sl], cmap="terrain", linewidth=0, antialiased=False,
                    rstride=1, cstride=1)
    if base_position is not None:
        ax.scatter(*base_position[:2], base_position[2], color="red", s=40, depthshade=False)
    ax.set_title("terrain", fontsize=9)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_zlim(float(np.nanmin(Z)), max(float(np.nanmax(Z)), float(np.nanmin(Z)) + 0.2))
    ax.view_init(elev=42, azim=-130)

    ax = fig.add_subplot(1, ncols, 2)
    im = ax.pcolormesh(Y, X, Z, cmap="terrain", shading="auto")
    if base_position is not None:
        ax.plot(base_position[1], base_position[0], "o", color="red", ms=7)
    ax.set_title("top-down height [m]", fontsize=9)
    ax.set_xlabel("y [m]"); ax.set_ylabel("x [m]")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, fraction=0.046)

    for i, (label, capture) in enumerate(captures or ()):
        ax = fig.add_subplot(1, ncols, 3 + i)
        world = capture.world_points
        sc = ax.scatter(world[:, 1], world[:, 0], c=world[:, 2], s=0.35,
                        cmap="terrain", vmin=np.nanmin(Z), vmax=np.nanmax(Z))
        ax.plot(capture.t_wc[1], capture.t_wc[0], "o", color="red", ms=7)
        ax.set_title(f"{label}\n{world.shape[0]} returns of {capture.n_pixels} rays", fontsize=9)
        ax.set_xlabel("y [m]"); ax.set_ylabel("x [m]")
        ax.set_xlim(sampler.ys.max(), sampler.ys.min())
        ax.set_ylim(sampler.xs.min(), sampler.xs.max())
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.046)

    fig.suptitle(f"{scene.name}: {scene.description}", fontsize=10)
    return _save(fig, path, plt)


def plot_surface(result, path: Path, radius: Optional[float] = None, stride: int = 2) -> Path:
    """Ground truth and estimate as 3D surfaces -- the clearest view of relief."""
    plt = _agg_pyplot()
    est, truth = result.layers["elevation"], result.ground_truth
    if radius is not None:
        keep = result.mask(radius)
        est = np.where(keep, est, np.nan)
        truth = np.where(keep, truth, np.nan)

    X, Y = result.cell_centers()
    sl = (slice(None, None, stride), slice(None, None, stride))
    vmin, vmax = np.nanmin(truth), np.nanmax(truth)

    fig = plt.figure(figsize=(12, 5))
    for i, (data, title) in enumerate(((truth, "ground truth"), (est, "estimate")), start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.plot_surface(X[sl], Y[sl], data[sl], cmap="terrain", vmin=vmin, vmax=vmax,
                        linewidth=0, antialiased=False, rstride=1, cstride=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_zlim(vmin, max(vmax, vmin + 0.1))
        ax.view_init(elev=38, azim=-125)

    fig.suptitle(f"{result.scene.name}: {result.scene.description}", fontsize=10)
    return _save(fig, path, plt)
