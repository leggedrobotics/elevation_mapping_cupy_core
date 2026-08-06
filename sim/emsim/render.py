"""Rendered views of a scene through MuJoCo's own rasteriser.

This is genuinely awkward to get working headless on a Jetson, so the setup is
done here once rather than left to the caller:

* Tegra's EGL exposes no usable ``EGL_PLATFORM_DEVICE_EXT`` display and is
  GLES-only, while MuJoCo's context asks for desktop ``EGL_OPENGL_BIT``. So the
  render has to go through Mesa's software rasteriser (``mesalib``), selected by
  pointing the EGL loader at Mesa's ICD and overriding the Gallium driver to
  ``llvmpipe`` -- otherwise Mesa tries the Tegra KMS nodes and reports
  ``kmsro: driver missing``.
* Only one of the enumerated EGL devices actually yields a working context, and
  which one is not knowable up front, so :func:`init_gl` walks them.

All of that must happen *before* ``mujoco`` is imported, because MuJoCo picks
its GL backend at import time. Call :func:`init_gl` first, or use the
``pixi run render`` task which sets the environment up front.

Rendering is software, so expect a second or two per frame. It is for looking at
scenes, not for producing sensor data -- the sensors are ray casts and never
touch OpenGL.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: Camera presets: (name, azimuth, elevation, distance, lookat).
#: A low camera is deliberate: these terrains have decimetre relief over metres,
#: which flat-shades into invisibility from a high angle.
VIEWS: Sequence[Tuple[str, float, float, float, Tuple[float, float, float]]] = (
    ("oblique", 145.0, -11.0, 6.0, (0.9, 0.0, 0.15)),
    ("front", 180.0, -7.0, 5.0, (1.2, 0.0, 0.2)),
    ("top", 90.0, -89.0, 9.0, (0.5, 0.0, 0.0)),
)

_MESA_ENV = {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe",
    "GALLIUM_DRIVER": "llvmpipe",
}


def init_gl(env_prefix: Optional[str] = None) -> None:
    """Point the EGL loader at Mesa's software rasteriser.

    Must run before ``mujoco`` is imported. Idempotent.

    Args:
        env_prefix: Conda/pixi environment prefix holding Mesa. Defaults to
            ``CONDA_PREFIX``.
    """
    prefix = Path(env_prefix or os.environ.get("CONDA_PREFIX", ""))
    for key, value in _MESA_ENV.items():
        os.environ[key] = value
    if prefix:
        vendor = prefix / "share" / "glvnd" / "egl_vendor.d"
        if vendor.is_dir():
            os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = str(vendor)
        lib = prefix / "lib"
        if lib.is_dir():
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            if str(lib) not in existing.split(":"):
                os.environ["LD_LIBRARY_PATH"] = f"{lib}:{existing}" if existing else str(lib)


def pick_egl_device() -> Optional[int]:
    """Index of the first EGL device that yields a desktop-GL context.

    MuJoCo caches its EGL display on the first context it builds, so a failed
    attempt cannot be retried in-process -- the device has to be chosen up
    front. This runs the same handshake MuJoCo does (device display, initialise,
    bind desktop GL, choose a config, create a context) and reports the first
    device that survives all of it. Several devices initialise happily and only
    fail at context creation, so the whole sequence matters.

    Returns:
        A device index, or None if none worked.
    """
    import ctypes

    from mujoco.egl import egl_ext as EGL

    attributes = (
        EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8, EGL.EGL_BLUE_SIZE, 8,
        EGL.EGL_ALPHA_SIZE, 8, EGL.EGL_DEPTH_SIZE, 24, EGL.EGL_STENCIL_SIZE, 8,
        EGL.EGL_COLOR_BUFFER_TYPE, EGL.EGL_RGB_BUFFER,
        EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
        EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT, EGL.EGL_NONE,
    )
    attribute_array = (EGL.EGLint * len(attributes))(*attributes)

    for index, device in enumerate(EGL.eglQueryDevicesEXT()):
        try:
            display = EGL.eglGetPlatformDisplayEXT(EGL.EGL_PLATFORM_DEVICE_EXT, device, None)
            if display == EGL.EGL_NO_DISPLAY or not EGL.eglInitialize(display, None, None):
                continue
            if not EGL.eglBindAPI(EGL.EGL_OPENGL_API):
                continue
            configs = (EGL.EGLConfig * 1)()
            count = EGL.EGLint()
            if not EGL.eglChooseConfig(display, attribute_array, configs, 1, ctypes.pointer(count)):
                continue
            if count.value < 1:
                continue
            context = EGL.eglCreateContext(display, configs[0], EGL.EGL_NO_CONTEXT, None)
            if not context:
                continue
            EGL.eglDestroyContext(display, context)
            return index
        except Exception:  # noqa: BLE001 - a device that errors is simply not the one
            continue
    return None


def make_renderer(model, height: int = 720, width: int = 1080):
    """Build a ``mujoco.Renderer`` on whichever EGL device works.

    Raises:
        RuntimeError: If no EGL device yields a usable context.
    """
    import mujoco

    if "MUJOCO_EGL_DEVICE_ID" not in os.environ:
        device = pick_egl_device()
        if device is None:
            raise RuntimeError(
                "no EGL device produced a desktop-OpenGL context.\n"
                "Rendering needs Mesa's software rasteriser: check that `mesalib` is "
                "installed and that this ran via `pixi run render`, which sets "
                "LD_LIBRARY_PATH and __EGL_VENDOR_LIBRARY_DIRS before the process starts."
            )
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device)
    return mujoco.Renderer(model, height, width)


def render_scene(
    scene_name: str,
    out_dir: Path,
    views: Sequence[str] = ("oblique",),
    base_position: Optional[np.ndarray] = None,
    height: int = 720,
    width: int = 1080,
) -> List[Path]:
    """Render one scene from the named :data:`VIEWS` presets."""
    import mujoco
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from emsim import scenes

    scene = scenes.make_scene(scene_name)
    model, data = scenes.build_model(scene)
    base = np.array([0.0, 0.0, 0.9]) if base_position is None else np.asarray(base_position)
    data.mocap_pos[scenes.mocap_id(model, scenes.ROBOT_BODY)] = base
    data.mocap_pos[scenes.mocap_id(model, scenes.LIDAR_BODY)] = base
    mujoco.mj_forward(model, data)

    renderer = make_renderer(model, height, width)
    presets = {v[0]: v for v in VIEWS}
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name in views:
        if name not in presets:
            raise KeyError(f"unknown view '{name}'; available: {sorted(presets)}")
        _, azimuth, elevation, distance, lookat = presets[name]
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.lookat[:] = lookat
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        renderer.update_scene(data, camera)
        path = out_dir / f"{scene_name}_render_{name}.png"
        plt.imsave(path, renderer.render())
        written.append(path)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    from emsim import scenes

    parser = argparse.ArgumentParser(prog="emsim.render", description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="mixed", choices=sorted(scenes.SCENES))
    parser.add_argument("--all", action="store_true", help="render every scene")
    parser.add_argument("--views", default="oblique",
                        help="comma-separated: " + ", ".join(v[0] for v in VIEWS) + ", or all")
    parser.add_argument("--out", type=Path, default=Path("sim/report"))
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1080)
    args = parser.parse_args(argv)

    views = [v[0] for v in VIEWS] if args.views == "all" else [
        s.strip() for s in args.views.split(",") if s.strip()
    ]
    for name in (sorted(scenes.SCENES) if args.all else [args.scene]):
        for path in render_scene(name, args.out, views, height=args.height, width=args.width):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    init_gl()
    raise SystemExit(main())
