"""The normal layers must not depend on the traversability filter.

``update_map_with_kernel`` feeds ``update_normal`` the ``traversability_input``
buffer, which holds the dilated upper-bound surface. The learned traversability
filter reads that same buffer, so it is easy to guard the dilation that fills it
behind ``traversability_filter is not None`` -- which leaves the buffer all zeros
whenever the filter is unavailable, and every normal silently becomes (0, 0, 1)
regardless of terrain.

These tests drive the real pipeline with the filter deliberately disabled, which
is the configuration that regressed.
"""

import math
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

from elevation_mapping_cupy import elevation_mapping, parameter

CONFIGS = Path(__file__).parent.parent / "configs"

RESOLUTION = 0.05
MAP_LENGTH = 4.0
#: Sample the plane finer than the grid so every interior cell gets several hits.
POINT_SPACING = 0.02
POINT_HALF_EXTENT = 1.5


def _make_map(weight_file: str):
    """An ElevationMap over a plain geometric configuration.

    ``weight_file=""`` is the supported way to run without the learned filter
    (see ``ElevationMap.__init__``), and is what a host with no CUDA-capable
    torch effectively falls back to.
    """
    param = parameter.Parameter(
        use_chainer=False,
        weight_file=weight_file,
        plugin_config_file=str(CONFIGS / "plugin_config.yaml"),
        resolution=RESOLUTION,
        map_length=MAP_LENGTH,
        enable_visibility_cleanup=False,
        enable_drift_compensation=False,
    )
    # A purely geometric run never writes the default semantic layers.
    param.subscriber_cfg = {}
    param.update()
    return elevation_mapping.ElevationMap(param)


def _plane_points(slope_deg: float) -> cp.ndarray:
    """A dense point cloud on the plane ``z = x * tan(slope)``, in world frame."""
    axis = np.arange(-POINT_HALF_EXTENT, POINT_HALF_EXTENT, POINT_SPACING, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    z = x * math.tan(math.radians(slope_deg))
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    return cp.asarray(points, dtype=cp.float32)


def _feed(em, slope_deg: float):
    """Push one sweep of the plane through the full input path."""
    points = _plane_points(slope_deg)
    R = cp.eye(3, dtype=em.param.data_type)
    t = cp.zeros(3, dtype=em.param.data_type)
    em.input_pointcloud(points, ["x", "y", "z"], R, t, 0.0, 0.0)


def _interior(em, layer: str) -> np.ndarray:
    """A layer cropped to the well-covered middle, where normals have neighbours."""
    data = cp.asnumpy(em.get_map_with_name_ref(layer, return_cupy=True))
    margin = data.shape[0] // 4
    return data[margin:-margin, margin:-margin]


def _recovered_tilt_deg(em) -> float:
    """Slope angle implied by the normal layers, in degrees."""
    nx, nz = _interior(em, "normal_x"), _interior(em, "normal_z")
    valid = np.abs(nz) > 1e-6
    assert valid.sum() > 100, "too few cells carry a normal to measure a tilt"
    return float(np.rad2deg(np.arctan2(-np.median(nx[valid]), np.median(nz[valid]))))


def test_traversability_filter_is_actually_disabled():
    """Guards the premise of the tests below."""
    assert _make_map("").traversability_filter is None


def test_normals_tilt_on_a_slope_without_the_traversability_filter():
    """The regression test: normals must track terrain with the filter absent.

    Against the bug the dilated surface stays all zeros, so every normal is
    (0, 0, 1) and this reads back 0 degrees rather than the true slope.
    """
    em = _make_map("")
    _feed(em, slope_deg=15.0)
    assert _recovered_tilt_deg(em) == pytest.approx(15.0, abs=5.0)


def test_normals_point_up_on_flat_ground_without_the_traversability_filter():
    """The flat case cannot catch the bug on its own, but pins the sign convention.

    (0, 0, 1) is the correct answer here, so this passes either way -- it is
    what makes the slope result above meaningful rather than a scaling artefact.
    """
    em = _make_map("")
    _feed(em, slope_deg=0.0)
    nx, ny, nz = (_interior(em, layer) for layer in ("normal_x", "normal_y", "normal_z"))
    valid = np.abs(nz) > 1e-6
    assert valid.sum() > 100

    norm = np.sqrt(nx[valid] ** 2 + ny[valid] ** 2 + nz[valid] ** 2)
    np.testing.assert_allclose(norm, 1.0, atol=1e-3)
    assert np.median(nz[valid]) > 0.99
