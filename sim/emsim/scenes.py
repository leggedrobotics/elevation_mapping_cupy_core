"""Procedural MuJoCo terrain scenes with analytically known ground truth.

Every scene is built from primitives whose top surface can be written down in
closed form, so ``Scene.analytic_height`` is an independent reference for the
ray-cast :class:`~emsim.heightmap.GroundTruthHeightmap`.

Two primitive families are used:

* **Axis-aligned boxes** for terrain with vertical discontinuities (steps, gaps,
  walls). The analytic height is ``max`` over the boxes covering ``(x, y)``.
* **Height fields** for smoothly varying terrain (slopes, rough ground). The
  analytic height is the generating function itself; MuJoCo triangulates the
  field, so the two agree to within the height-field spacing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Height fields are sampled this finely (m). Below a typical 0.04 m map
# resolution so MuJoCo's triangulation error stays well under a map cell, but
# not so fine that ray casting bogs down: MuJoCo walks the field cell by cell,
# so cost scales with cells-per-ray.
HFIELD_SPACING = 0.03

# Half-extent of every height field (m). Beyond it the ground plane takes over.
HFIELD_RADIUS = 5.0

# Mocap body carrying the (virtual) sensor. Excluded from every ray cast.
ROBOT_BODY = "robot"


@dataclass(frozen=True)
class Box:
    """An axis-aligned box. ``pos`` is the centre, ``size`` the half-extents."""

    pos: Tuple[float, float, float]
    size: Tuple[float, float, float]
    rgba: Tuple[float, float, float, float] = (0.55, 0.55, 0.6, 1.0)

    @property
    def top(self) -> float:
        return self.pos[2] + self.size[2]

    def covers(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean mask of the query points inside the box footprint."""
        return (np.abs(x - self.pos[0]) <= self.size[0]) & (np.abs(y - self.pos[1]) <= self.size[1])

    def to_mjcf(self, name: str) -> str:
        return (
            f'    <geom name="{name}" type="box" '
            f'pos="{self.pos[0]:.6g} {self.pos[1]:.6g} {self.pos[2]:.6g}" '
            f'size="{self.size[0]:.6g} {self.size[1]:.6g} {self.size[2]:.6g}" '
            f'rgba="{self.rgba[0]} {self.rgba[1]} {self.rgba[2]} {self.rgba[3]}"/>'
        )


@dataclass
class HField:
    """A MuJoCo height field plus the function that generated it."""

    name: str
    radius_x: float
    radius_y: float
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    spacing: float = HFIELD_SPACING
    base: float = 0.5

    def __post_init__(self) -> None:
        self.ncol = int(round(2 * self.radius_x / self.spacing)) + 1
        self.nrow = int(round(2 * self.radius_y / self.spacing)) + 1
        xs = np.linspace(-self.radius_x, self.radius_x, self.ncol)
        ys = np.linspace(-self.radius_y, self.radius_y, self.nrow)
        # Row-major, rows along +y and columns along +x, starting at the
        # (-radius_x, -radius_y) corner -- MuJoCo's height-field convention.
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        heights = np.asarray(self.fn(gx, gy), dtype=np.float64)
        if heights.min() < -1e-9:
            raise ValueError(f"height field '{self.name}' must be non-negative, got min {heights.min()}")
        self.elevation = max(float(heights.max()), 1e-3)
        self.data = (heights / self.elevation).astype(np.float32)

    def to_mjcf_asset(self) -> str:
        return (
            f'    <hfield name="{self.name}" nrow="{self.nrow}" ncol="{self.ncol}" '
            f'size="{self.radius_x:.6g} {self.radius_y:.6g} {self.elevation:.6g} {self.base:.6g}"/>'
        )

    def to_mjcf_geom(self) -> str:
        return (
            f'    <geom name="{self.name}" type="hfield" hfield="{self.name}" '
            f'pos="0 0 0" rgba="0.5 0.55 0.5 1"/>'
        )


@dataclass
class Scene:
    """A terrain description: MJCF plus its closed-form surface."""

    name: str
    description: str
    boxes: List[Box] = field(default_factory=list)
    hfield: Optional[HField] = None
    ground_z: float = 0.0
    bounds: Tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0)

    def analytic_height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Terrain height at ``(x, y)``, broadcasting over arrays."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        h = np.full(np.broadcast(x, y).shape, self.ground_z, dtype=np.float64)
        if self.hfield is not None:
            inside = (np.abs(x) <= self.hfield.radius_x) & (np.abs(y) <= self.hfield.radius_y)
            h = np.where(inside, np.maximum(h, self.hfield.fn(x, y)), h)
        for box in self.boxes:
            h = np.where(box.covers(x, y), np.maximum(h, box.top), h)
        return h

    def to_mjcf(self) -> str:
        assets = [self.hfield.to_mjcf_asset()] if self.hfield else []
        geoms = [self.hfield.to_mjcf_geom()] if self.hfield else []
        geoms += [b.to_mjcf(f"box_{i}") for i, b in enumerate(self.boxes)]
        asset_block = "  <asset>\n" + "\n".join(assets) + "\n  </asset>\n" if assets else ""
        return f"""<mujoco model="emsim_{self.name}">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81"/>
{asset_block}  <worldbody>
    <light pos="0 0 6" dir="0 0 -1"/>
    <geom name="ground" type="plane" pos="0 0 {self.ground_z:.6g}" size="30 30 0.1" rgba="0.4 0.4 0.45 1"/>
{chr(10).join(geoms)}
    <body name="{ROBOT_BODY}" mocap="true" pos="0 0 1">
      <geom name="robot_shell" type="box" size="0.3 0.16 0.12" rgba="0.85 0.25 0.2 1"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


# --------------------------------------------------------------------------- #
# Scene catalogue
# --------------------------------------------------------------------------- #


def flat() -> Scene:
    """Featureless ground plane. The baseline accuracy case."""
    return Scene(name="flat", description="Featureless ground plane at z = 0.")


def steps(step_height: float = 0.12, step_depth: float = 0.45, n: int = 6, x0: float = 1.0) -> Scene:
    """A staircase ascending along +x. Exercises sharp vertical discontinuities."""
    boxes = []
    for i in range(n):
        top = (i + 1) * step_height
        boxes.append(
            Box(
                pos=(x0 + (i + 0.5) * step_depth, 0.0, top / 2.0),
                size=(step_depth / 2.0, 1.5, top / 2.0),
                rgba=(0.6, 0.5, 0.45, 1.0),
            )
        )
    return Scene(
        name="steps",
        description=f"{n} steps of {step_height:.2f} m rise / {step_depth:.2f} m run along +x.",
        boxes=boxes,
    )


def gap(platform_z: float = 0.35, gap_width: float = 0.7) -> Scene:
    """Two raised platforms separated by a gap down to ground level."""
    half = 1.6
    x_near = gap_width / 2.0 + half
    boxes = [
        Box(pos=(-x_near, 0.0, platform_z / 2.0), size=(half, 1.8, platform_z / 2.0)),
        Box(pos=(+x_near, 0.0, platform_z / 2.0), size=(half, 1.8, platform_z / 2.0)),
    ]
    return Scene(
        name="gap",
        description=f"Two {platform_z:.2f} m platforms with a {gap_width:.2f} m gap between them.",
        boxes=boxes,
    )


def wall(height: float = 1.2, x: float = 2.0) -> Scene:
    """A vertical wall across the field of view. Exercises occlusion handling."""
    return Scene(
        name="wall",
        description=f"A {height:.2f} m wall at x = {x:.2f} m occluding the terrain behind it.",
        boxes=[Box(pos=(x, 0.0, height / 2.0), size=(0.1, 2.5, height / 2.0), rgba=(0.7, 0.3, 0.3, 1.0))],
    )


def boxes(n: int = 14, seed: int = 0) -> Scene:
    """A deterministic clutter field of boxes of varied size."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        # Keep a clear disc around the origin so the robot always starts on flat ground.
        while True:
            px, py = rng.uniform(-3.5, 3.5, size=2)
            if px * px + py * py > 1.0:
                break
        h = rng.uniform(0.06, 0.45)
        sx, sy = rng.uniform(0.12, 0.4, size=2)
        out.append(Box(pos=(px, py, h / 2.0), size=(sx, sy, h / 2.0)))
    return Scene(name="boxes", description=f"{n} randomly placed boxes (seed {seed}).", boxes=out)


def slope(angle_deg: float = 15.0, x0: float = 0.8, height_cap: float = 1.0) -> Scene:
    """A constant-gradient ramp rising along +x, then flat."""
    slope_k = float(np.tan(np.deg2rad(angle_deg)))

    def fn(x, y):
        return np.clip(slope_k * (x - x0), 0.0, height_cap)

    return Scene(
        name="slope",
        description=f"{angle_deg:.0f} deg ramp starting at x = {x0:.2f} m, capped at {height_cap:.2f} m.",
        hfield=HField(name="slope", radius_x=HFIELD_RADIUS, radius_y=HFIELD_RADIUS, fn=fn),
    )


def rough(amplitude: float = 0.07) -> Scene:
    """Smooth undulating ground from a fixed sum of sinusoids."""

    def fn(x, y):
        z = (
            0.55 * np.sin(1.3 * x) * np.cos(0.9 * y)
            + 0.30 * np.sin(2.7 * x + 0.6) * np.sin(2.1 * y - 0.4)
            + 0.15 * np.cos(4.1 * x - 1.1) * np.cos(3.3 * y + 0.9)
        )
        return amplitude * (z + 1.0)  # shifted to be non-negative

    return Scene(
        name="rough",
        description=f"Undulating sinusoidal terrain, peak-to-peak ~{2 * amplitude:.2f} m.",
        hfield=HField(name="rough", radius_x=HFIELD_RADIUS, radius_y=HFIELD_RADIUS, fn=fn),
    )


def mixed() -> Scene:
    """Rough ground with a staircase and clutter on top. The stress case."""
    base = rough(amplitude=0.04)
    st = steps(step_height=0.10, step_depth=0.40, n=5, x0=1.2)
    cl = boxes(n=6, seed=3)
    return Scene(
        name="mixed",
        description="Undulating ground with a staircase along +x and scattered boxes.",
        boxes=st.boxes + cl.boxes,
        hfield=base.hfield,
    )


SCENES: Dict[str, Callable[[], Scene]] = {
    "flat": flat,
    "steps": steps,
    "gap": gap,
    "wall": wall,
    "boxes": boxes,
    "slope": slope,
    "rough": rough,
    "mixed": mixed,
}


def make_scene(name: str) -> Scene:
    """Build a scene from the catalogue by name."""
    try:
        return SCENES[name]()
    except KeyError:
        raise KeyError(f"unknown scene '{name}'; available: {sorted(SCENES)}") from None


def build_model(scene: Scene):
    """Compile ``scene`` into a ``(MjModel, MjData)`` pair with height fields filled in."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(scene.to_mjcf())
    if scene.hfield is not None:
        model.hfield_data[:] = scene.hfield.data.ravel()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def robot_body_id(model) -> int:
    """Body id of the sensor carrier, for exclusion from ray casts."""
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_BODY)
