"""Geometry of the map columns rendered by emsim.render_map (no GL needed)."""

import numpy as np

from emsim.render_map import cell_columns


def test_columns_span_floor_to_value_over_one_cell():
    values = np.array([0.3, -0.1, -0.2])
    x, y = np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, -0.5])
    pos, size = cell_columns(values, x, y, floor=-0.2, resolution=0.04)
    np.testing.assert_allclose(pos[:, :2], np.stack([x, y], 1))
    np.testing.assert_allclose(size[:, :2], 0.02)
    # Bottom at the floor, top at the value; a cell at the floor keeps a 4 mm column.
    np.testing.assert_allclose(pos[:, 2] - size[:, 2], [-0.2, -0.2, -0.2])
    np.testing.assert_allclose(pos[:2, 2] + size[:2, 2], [0.3, -0.1])
    np.testing.assert_allclose(size[2, 2], 0.002)
