"""Shared fixtures for the MuJoCo simulation tests.

Simulation runs are expensive (kernel compilation, ray-cast ground truth), so
they are memoised for the whole session and shared across the tests that assert
on them.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import pytest

from emsim import scenes
from emsim.heightmap import GroundTruthHeightmap


def _cupy_available() -> bool:
    try:
        import cupy as cp

        cp.zeros(1).sum()
        return True
    except Exception:  # pragma: no cover - depends on the host
        return False


HAS_CUPY = _cupy_available()

requires_gpu = pytest.mark.skipif(not HAS_CUPY, reason="requires a working CuPy/CUDA installation")


@pytest.fixture(scope="session")
def built_scene() -> Callable[[str], tuple]:
    """Memoised ``name -> (scene, model, data, robot_body_id)``."""
    cache: Dict[str, tuple] = {}

    def build(name: str):
        if name not in cache:
            scene = scenes.make_scene(name)
            model, data = scenes.build_model(scene)
            cache[name] = (scene, model, data, scenes.robot_body_id(model))
        return cache[name]

    return build


@pytest.fixture(scope="session")
def gt_sampler(built_scene) -> Callable[..., GroundTruthHeightmap]:
    """Memoised ray-cast ground truth per ``(scene, bounds, resolution)``."""
    cache: Dict[tuple, GroundTruthHeightmap] = {}

    def build(name: str, bounds=(-5.0, 5.0, -5.0, 5.0), resolution: float = 0.04):
        key = (name, bounds, resolution)
        if key not in cache:
            _, model, data, robot_id = built_scene(name)
            cache[key] = GroundTruthHeightmap(
                model, data, bounds, resolution, bodyexclude=robot_id
            )
        return cache[key]

    return build


@pytest.fixture(scope="session")
def sim_run() -> Callable[..., object]:
    """Memoised simulation runs.

    Args of the returned callable:
        key: Cache key. Two calls with the same key return the same result, so
            the key must capture everything that distinguishes the run.
        cfg: The :class:`~emsim.runner.RunConfig` to execute.
        layers: Map layers to export.
    """
    cache: Dict[str, object] = {}

    def execute(key: str, cfg, layers: Sequence[str] = ("elevation",)):
        if key not in cache:
            from emsim.runner import run

            cache[key] = run(cfg, layers=layers)
        return cache[key]

    return execute
