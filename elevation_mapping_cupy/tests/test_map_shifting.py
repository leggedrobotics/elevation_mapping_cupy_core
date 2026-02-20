"""
Tests for map shifting functionality.

These tests verify that the map shifts correctly when the robot moves.

In the src convention, shift_map_xy(delta_pixel) passes delta_pixel directly
to cp.roll(..., axis=(1, 2)):
  - delta_pixel[0] shifts axis 1 (X dimension in the grid)
  - delta_pixel[1] shifts axis 2 (Y dimension in the grid)
"""

import pytest
import numpy as np
import cupy as cp
from pathlib import Path
from elevation_mapping_cupy import parameter, elevation_mapping

# Get absolute paths to config files
_TEST_DIR = Path(__file__).parent
_CONFIG_DIR = _TEST_DIR.parent / "configs"


@pytest.fixture
def elmap_shift():
    """Create a minimal elevation map for shift testing."""
    p = parameter.Parameter(
        use_chainer=False,
        weight_file=str(_CONFIG_DIR / "weights.dat"),
        plugin_config_file=str(_CONFIG_DIR / "plugin_config.yaml"),
    )
    # Use default resolution (0.1m) and map_length (20m) -> ~200x200 cells
    p.update()
    e = elevation_mapping.ElevationMap(p)
    e.clear()  # Start with clean map
    return e


class TestShiftMapXY:
    """Tests for the shift_map_xy function."""

    def test_shift_x_only_affects_axis1(self, elmap_shift):
        """
        X-only shift (delta_pixel[0]) should only affect axis 1, not axis 2.

        shift_map_xy([5, 0]) should move the marker along axis 1 only.
        """
        center_idx = elmap_shift.cell_n // 2

        elmap_shift.elevation_map[0, center_idx, center_idx] = 1.0

        shift_amount = 5
        elmap_shift.shift_map_xy(cp.array([shift_amount, 0], dtype=cp.float32))

        new_axis1 = center_idx + shift_amount

        assert float(elmap_shift.elevation_map[0, new_axis1, center_idx]) == 1.0, \
            f"Marker should be at ({new_axis1}, {center_idx}) after X shift"

        assert float(elmap_shift.elevation_map[0, center_idx, new_axis1]) == 0.0, \
            f"Marker should NOT be at ({center_idx}, {new_axis1}) - X shift should not affect axis 2"

    def test_shift_y_only_affects_axis2(self, elmap_shift):
        """
        Y-only shift (delta_pixel[1]) should only affect axis 2, not axis 1.

        shift_map_xy([0, 5]) should move the marker along axis 2 only.
        """
        center_idx = elmap_shift.cell_n // 2

        elmap_shift.elevation_map[0, center_idx, center_idx] = 1.0

        shift_amount = 5
        elmap_shift.shift_map_xy(cp.array([0, shift_amount], dtype=cp.float32))

        new_axis2 = center_idx + shift_amount

        assert float(elmap_shift.elevation_map[0, center_idx, new_axis2]) == 1.0, \
            f"Marker should be at ({center_idx}, {new_axis2}) after Y shift"

        assert float(elmap_shift.elevation_map[0, new_axis2, center_idx]) == 0.0, \
            f"Marker should NOT be at ({new_axis2}, {center_idx}) - Y shift should not affect axis 1"

    def test_diagonal_shift(self, elmap_shift):
        """Diagonal shift should affect both axes correctly."""
        center_idx = elmap_shift.cell_n // 2

        elmap_shift.elevation_map[0, center_idx, center_idx] = 1.0

        shift_x, shift_y = 3, 7
        elmap_shift.shift_map_xy(cp.array([shift_x, shift_y], dtype=cp.float32))

        expected_axis1 = center_idx + shift_x
        expected_axis2 = center_idx + shift_y

        assert float(elmap_shift.elevation_map[0, expected_axis1, expected_axis2]) == 1.0, \
            f"Marker should be at ({expected_axis1}, {expected_axis2}) after diagonal shift"

    def test_negative_shift(self, elmap_shift):
        """Negative shifts should work correctly."""
        center_idx = elmap_shift.cell_n // 2

        elmap_shift.elevation_map[0, center_idx, center_idx] = 1.0

        shift_x, shift_y = -5, -3
        elmap_shift.shift_map_xy(cp.array([shift_x, shift_y], dtype=cp.float32))

        expected_axis1 = center_idx + shift_x
        expected_axis2 = center_idx + shift_y

        assert float(elmap_shift.elevation_map[0, expected_axis1, expected_axis2]) == 1.0, \
            f"Marker should be at ({expected_axis1}, {expected_axis2}) after negative shift"

    def test_zero_shift_no_change(self, elmap_shift):
        """Zero shift should not modify the map."""
        center_idx = elmap_shift.cell_n // 2

        elmap_shift.elevation_map[0, center_idx, center_idx] = 1.0
        original_map = elmap_shift.elevation_map.copy()

        elmap_shift.shift_map_xy(cp.array([0, 0], dtype=cp.float32))

        assert cp.allclose(elmap_shift.elevation_map, original_map), \
            "Zero shift should not modify the map"


class TestMoveTo:
    """Tests for the move_to function which uses shift_map_xy internally."""

    def test_move_to_x_positive(self, elmap_shift):
        """
        Robot moving in +X direction should shift map in -X direction.
        This makes the map appear to scroll backward as robot moves forward.
        """
        initial_center = cp.asnumpy(elmap_shift.center.copy())

        center_idx = elmap_shift.cell_n // 2
        elmap_shift.elevation_map[0, center_idx, center_idx] = 1.0

        move_distance = 1.0  # meters
        R = np.eye(3, dtype=np.float32)
        elmap_shift.move_to(np.array([move_distance, 0.0, 0.0], dtype=np.float32), R)

        new_center = cp.asnumpy(elmap_shift.center)
        assert new_center[0] > initial_center[0], \
            "Map center X should increase when robot moves +X"
        assert abs(new_center[1] - initial_center[1]) < 1e-6, \
            "Map center Y should not change for X-only movement"

    def test_move_to_y_positive(self, elmap_shift):
        """
        Robot moving in +Y direction should shift map in -Y direction.
        """
        initial_center = cp.asnumpy(elmap_shift.center.copy())

        move_distance = 1.0
        R = np.eye(3, dtype=np.float32)
        elmap_shift.move_to(np.array([0.0, move_distance, 0.0], dtype=np.float32), R)

        new_center = cp.asnumpy(elmap_shift.center)
        assert new_center[1] > initial_center[1], \
            "Map center Y should increase when robot moves +Y"
        assert abs(new_center[0] - initial_center[0]) < 1e-6, \
            "Map center X should not change for Y-only movement"

    def test_move_to_preserves_relative_data(self, elmap_shift):
        """
        After moving, data in the map should maintain its world position.
        A point 1m ahead in X should remain visible after robot moves 0.5m in X.

        In the src convention, X maps to axis 1.  move_to computes
        delta_pixel from delta[:2] and calls shift_map_xy(-delta_pixel),
        so an X movement shifts axis 1.
        """
        resolution = elmap_shift.resolution
        center_idx = elmap_shift.cell_n // 2

        offset_cells = int(1.0 / resolution)
        marker_axis1 = center_idx + offset_cells
        elmap_shift.elevation_map[0, marker_axis1, center_idx] = 1.0

        R = np.eye(3, dtype=np.float32)
        elmap_shift.move_to(np.array([0.5, 0.0, 0.0], dtype=np.float32), R)

        expected_axis1 = marker_axis1 - int(0.5 / resolution)

        assert float(elmap_shift.elevation_map[0, expected_axis1, center_idx]) == 1.0, \
            "Marker should maintain relative world position after robot movement"


class TestPadValue:
    """Tests for padding behavior after shifts."""

    def test_positive_x_shift_pads_low_axis1(self, elmap_shift):
        """After positive X shift (axis 1), low axis-1 indices should be padded."""
        elmap_shift.elevation_map[0, :, :] = 1.0

        shift_amount = 10
        elmap_shift.shift_map_xy(cp.array([shift_amount, 0], dtype=cp.float32))

        assert cp.all(elmap_shift.elevation_map[0, :shift_amount, :] == 0.0), \
            "Low axis-1 indices should be padded with 0 after positive X shift"

        assert cp.any(elmap_shift.elevation_map[0, shift_amount:, :] != 0.0), \
            "High axis-1 indices should still have data after positive X shift"

    def test_positive_y_shift_pads_low_axis2(self, elmap_shift):
        """After positive Y shift (axis 2), low axis-2 indices should be padded."""
        elmap_shift.elevation_map[0, :, :] = 1.0

        shift_amount = 10
        elmap_shift.shift_map_xy(cp.array([0, shift_amount], dtype=cp.float32))

        assert cp.all(elmap_shift.elevation_map[0, :, :shift_amount] == 0.0), \
            "Low axis-2 indices should be padded with 0 after positive Y shift"

        assert cp.any(elmap_shift.elevation_map[0, :, shift_amount:] != 0.0), \
            "High axis-2 indices should still have data after positive Y shift"


#
# Semantic/image fusion was intentionally removed from the supported surface of this repo.
# Keep map-shift tests focused on the elevation map core.
