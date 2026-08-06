"""MuJoCo simulation harness for ``elevation_mapping_cupy_core``.

Builds procedural terrain in MuJoCo, feeds ray-cast depth frames into
``ElevationMap``, and scores the result against a top-down ray-cast
ground-truth height map.

Only :mod:`emsim.runner` needs CuPy; scenes, the sensor and the ground-truth
sampler work on CPU alone.
"""

from emsim.heightmap import (
    AnalyticHeightmap,
    GroundTruthHeightmap,
    HeightSampler,
    map_cell_centers,
    radial_mask,
)
from emsim.metrics import MapError, Timings, compare_maps
from emsim.scenes import SCENES, Box, HField, Scene, build_model, make_scene, robot_body_id
from emsim.sensor import CameraIntrinsics, DepthCapture, DepthSensor, SensorNoise, camera_pose

__all__ = [
    "AnalyticHeightmap",
    "Box",
    "CameraIntrinsics",
    "DepthCapture",
    "DepthSensor",
    "GroundTruthHeightmap",
    "HField",
    "HeightSampler",
    "MapError",
    "SCENES",
    "Scene",
    "SensorNoise",
    "Timings",
    "build_model",
    "camera_pose",
    "compare_maps",
    "make_scene",
    "map_cell_centers",
    "radial_mask",
    "robot_body_id",
]
