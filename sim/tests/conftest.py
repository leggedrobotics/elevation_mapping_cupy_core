"""Shared fixtures for the MuJoCo simulation tests.

Simulation runs are expensive (kernel compilation, ray-cast ground truth), so
they are memoised for the whole session and shared across the tests that assert
on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import pytest

from emsim import scenes
from emsim.heightmap import GroundTruthHeightmap


def pytest_addoption(parser):
    parser.addoption(
        "--viz-dir",
        action="store",
        default=None,
        metavar="DIR",
        help="write a figure per test that has one, into DIR. Off by default.",
    )


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


@pytest.fixture
def viz(request) -> Callable[..., Optional[Path]]:
    """Emit a figure for the current test, if ``--viz-dir`` was given.

    A no-op otherwise, so tests can call it unconditionally::

        viz("comparison", result)

    ``kind`` is any of the ``plot_*`` functions in :mod:`emsim.plotting`.
    """
    out = request.config.getoption("--viz-dir")

    def emit(kind: str, result, **kwargs) -> Optional[Path]:
        if out is None:
            return None
        from emsim import plotting

        fn = getattr(plotting, f"plot_{kind}")
        # Node ids contain path separators and brackets; keep them out of filenames.
        stem = request.node.name.replace("/", "_").replace("[", "-").replace("]", "")
        return fn(result, Path(out) / f"{stem}_{kind}.png", **kwargs)

    return emit


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
