import cupy as cp
import pytest

from elevation_mapping_cupy.kernels import add_points_kernel
from elevation_mapping_cupy.parameter import Parameter

CELL_N = 32


def point_validity(sensor_x, points, from_sensor):
    """is_valid() for sensor-frame points seen from a sensor at (sensor_x, 0, 0.5)."""
    p = Parameter()
    kernel = add_points_kernel(
        p.resolution, CELL_N, CELL_N, p.sensor_noise_factor, p.mahalanobis_thresh,
        p.outlier_variance, p.wall_num_thresh, p.max_ray_length, p.cleanup_step,
        p.min_valid_distance,
        10.0,  # max_height_range out of the way so only the ramp can reject
        p.cleanup_cos_thresh, p.ramped_height_range_a, p.ramped_height_range_b,
        p.ramped_height_range_c,
        enable_visibility_cleanup=False,
        ramped_height_range_from_sensor=from_sensor,
    )
    pts = cp.asarray(points, dtype=cp.float32).ravel()
    center = cp.asarray([sensor_x], dtype=cp.float32)
    kernel(
        center, cp.zeros(1, cp.float32), cp.eye(3, dtype=cp.float32).ravel(),
        cp.asarray([sensor_x, 0.0, 0.5], dtype=cp.float32), cp.zeros((3, CELL_N, CELL_N), cp.float32),
        pts, cp.zeros((7, CELL_N, CELL_N), cp.float32), cp.zeros((5, CELL_N, CELL_N), cp.float32),
        size=len(points),
    )
    return [bool(v) for v in pts.reshape(-1, 3)[:, 1].get()]


# At 3 m range the default ramp allows (3 - 1) * 0.3 + 0.2 = 0.8 m above the sensor.
POINTS = [[3.0, 0.0, 1.5], [3.0, 0.0, 0.5]]


@pytest.mark.parametrize("sensor_x", [0.0, 30.0])
def test_ramp_is_relative_to_sensor(sensor_x):
    assert point_validity(sensor_x, POINTS, from_sensor=True) == [False, True]


def test_legacy_ramp_is_relative_to_map_origin():
    # Default keeps the old behaviour: the ramp distance is measured from the map origin.
    assert point_validity(0.0, POINTS, from_sensor=False) == [False, True]
    assert point_validity(30.0, POINTS, from_sensor=False) == [True, True]
