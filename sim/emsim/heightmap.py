"""Top-down ray-cast ground-truth height sampler.

The sampler shoots a vertical ray straight down at every cell centre of a grid
that is *aligned to the elevation map's own cell lattice*, so a ground-truth
window can be compared against an exported map layer cell-for-cell with no
interpolation in between.

Grid alignment
--------------
``ElevationMap`` cell centres sit at ``center + (i + 0.5 - N/2) * resolution``
where ``N = map_length / resolution``. The map centre itself only ever moves in
whole-cell steps (``move_to`` rounds the delta to pixels starting from the
origin), so for even ``N`` every cell centre the map can ever have lands on
``(k + 0.5) * resolution`` for integer ``k``. This module builds its global grid
on exactly that lattice, making the lookup in :meth:`GroundTruthHeightmap.height_at`
an exact integer index rather than an interpolation.

Export convention
-----------------
``ElevationMap.get_map_with_name_ref`` flips both axes on export, so in the array
handed back to the caller row 0 is *maximum* x and column 0 is *maximum* y.
:func:`map_cell_centers` reproduces that layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

# Rays start this far above the tallest geometry in the scene.
RAY_CLEARANCE = 5.0

# MuJoCo splits every height-field quad into two triangles. A ray landing
# exactly on that shared diagonal can miss both and fall through to whatever
# lies beneath -- which happens systematically, because a grid symmetric about
# the origin puts every sample with x == y on a diagonal. A second ray at a tiny
# asymmetric offset repairs it: the offset is 4000x smaller than a 0.04 m map
# cell, so on continuous terrain the two rays agree to well below float32
# precision, and the higher of the two is kept.
EDGE_SAFE_OFFSET = (1.0e-5, 3.7e-6)


def map_cell_centers(
    center_xy: Tuple[float, float], n: int, resolution: float
) -> Tuple[np.ndarray, np.ndarray]:
    """World ``(X, Y)`` of every cell of an exported ``(n, n)`` map layer.

    Args:
        center_xy: Map centre in world coordinates.
        n: Side length in cells of the exported layer (``param.true_cell_n``).
        resolution: Map resolution in metres.

    Returns:
        ``(X, Y)``, each ``(n, n)``. Row 0 holds the largest x, column 0 the
        largest y, matching ``get_map_with_name_ref``'s double flip.
    """
    offsets = (np.arange(n) + 0.5 - n / 2.0) * resolution  # ascending, kernel order
    x = center_xy[0] + offsets[::-1]
    y = center_xy[1] + offsets[::-1]
    return np.meshgrid(x, y, indexing="ij")


def radial_mask(n: int, resolution: float, radius: float) -> np.ndarray:
    """Boolean ``(n, n)`` mask of cells within ``radius`` of the map centre."""
    offsets = (np.arange(n) + 0.5 - n / 2.0) * resolution
    dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
    return dx * dx + dy * dy <= radius * radius


class HeightSampler:
    """Interface shared by the ray-cast and analytic ground-truth sources."""

    def height_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def sample_map_grid(
        self, center_xy: Tuple[float, float], n: int, resolution: float
    ) -> np.ndarray:
        """Ground-truth height laid out like an exported map layer."""
        gx, gy = map_cell_centers(center_xy, n, resolution)
        return self.height_at(gx, gy)


@dataclass
class AnalyticHeightmap(HeightSampler):
    """Ground truth straight from a scene's closed-form surface."""

    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def height_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(np.asarray(x), np.asarray(y)), dtype=np.float64)


class GroundTruthHeightmap(HeightSampler):
    """Ray-cast height field over a fixed region of a MuJoCo scene.

    The whole region is cast once on construction and cached; querying is then a
    pure array lookup. Terrain is static, so one build serves an entire run.
    """

    def __init__(
        self,
        model,
        data,
        bounds: Tuple[float, float, float, float],
        resolution: float,
        ray_start_z: Optional[float] = None,
        bodyexclude: int = -1,
        geomgroup: Optional[np.ndarray] = None,
        no_hit_value: float = np.nan,
        edge_safe: bool = True,
    ):
        """
        Args:
            model: ``mujoco.MjModel``.
            data: ``mujoco.MjData``, already forwarded.
            bounds: ``(xmin, xmax, ymin, ymax)`` region to cover, in metres. The
                built grid is snapped outwards to the ``resolution`` lattice.
            resolution: Cell size in metres. Must match the map resolution for
                the lookup to be exact.
            ray_start_z: Height rays are cast from. Defaults to the top of the
                scene's bounding volume plus :data:`RAY_CLEARANCE`.
            bodyexclude: Body id ignored by the cast (the sensor carrier).
            geomgroup: Optional ``(6,)`` uint8 geom-group filter.
            no_hit_value: Value written where a ray hits nothing.
            edge_safe: Cast the extra :data:`EDGE_SAFE_OFFSET` ray per cell that
                repairs height-field triangle-edge dropouts. Doubles build cost.
        """
        import mujoco

        self.resolution = float(resolution)
        self.bounds = bounds
        xmin, xmax, ymin, ymax = bounds

        # Snap outwards to the (k + 0.5) * resolution lattice.
        self._i0 = int(np.floor(xmin / resolution))
        self._j0 = int(np.floor(ymin / resolution))
        nx = int(np.ceil(xmax / resolution)) - self._i0
        ny = int(np.ceil(ymax / resolution)) - self._j0
        self.xs = (np.arange(self._i0, self._i0 + nx) + 0.5) * resolution
        self.ys = (np.arange(self._j0, self._j0 + ny) + 0.5) * resolution

        if ray_start_z is None:
            # stat.extent is the model's bounding-box diagonal, so one extent
            # above the model centre already clears every geom; the clearance is
            # margin on top of that.
            ray_start_z = float(model.stat.center[2] + model.stat.extent) + RAY_CLEARANCE
        self.ray_start_z = float(ray_start_z)

        offsets = [(0.0, 0.0)] + ([EDGE_SAFE_OFFSET] if edge_safe else [])
        heights = np.full((nx, ny), no_hit_value, dtype=np.float64)
        pnt = np.zeros(3, dtype=np.float64)
        vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        pnt[2] = self.ray_start_z
        n_repairs = 0
        for i, x in enumerate(self.xs):
            for j, y in enumerate(self.ys):
                primary = -np.inf
                best = -np.inf
                for k, (dx, dy) in enumerate(offsets):
                    pnt[0], pnt[1] = x + dx, y + dy
                    dist = mujoco.mj_ray(model, data, pnt, vec, geomgroup, 1, bodyexclude, geomid)
                    height = self.ray_start_z - dist if dist >= 0.0 else -np.inf
                    if k == 0:
                        primary = height
                    best = max(best, height)
                if np.isfinite(best):
                    heights[i, j] = best
                    # A cell the offset ray rescued from falling through the mesh.
                    n_repairs += best > primary + 1e-6
        self.heights = heights
        self.n_rays = nx * ny * len(offsets)
        self.n_edge_repairs = int(n_repairs)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.heights.shape

    def height_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Nearest-cell height lookup; NaN outside the built region."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        ix = np.floor(x / self.resolution).astype(np.int64) - self._i0
        iy = np.floor(y / self.resolution).astype(np.int64) - self._j0
        nx, ny = self.heights.shape
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.full(np.broadcast(x, y).shape, np.nan, dtype=np.float64)
        out[inside] = self.heights[ix[inside], iy[inside]]
        return out
