"""Scene construction and closed-form surface definitions. No GPU required."""

from __future__ import annotations

import numpy as np
import pytest

from emsim import scenes

ALL_SCENES = sorted(scenes.SCENES)


@pytest.mark.parametrize("name", ALL_SCENES)
def test_scene_compiles(name, built_scene):
    scene, model, data, robot_id = built_scene(name)
    assert scene.name == name
    assert model.ngeom >= 2  # ground plane + robot shell at minimum
    assert robot_id >= 0
    assert model.body_mocapid[robot_id] >= 0, "sensor carrier must be a mocap body"


@pytest.mark.parametrize("name", ALL_SCENES)
def test_analytic_height_broadcasts(name):
    scene = scenes.make_scene(name)
    x, y = np.meshgrid(np.linspace(-4, 4, 17), np.linspace(-4, 4, 13), indexing="ij")
    h = scene.analytic_height(x, y)
    assert h.shape == x.shape
    assert np.all(np.isfinite(h))
    assert np.all(h >= scene.ground_z - 1e-12), "terrain must never dip below the ground plane"
    assert np.isscalar(float(scene.analytic_height(np.float64(0.0), np.float64(0.0))))


def test_box_geometry():
    box = scenes.Box(pos=(1.0, -2.0, 0.25), size=(0.5, 0.25, 0.25))
    assert box.top == pytest.approx(0.5)
    inside = box.covers(np.array([1.0, 1.49, 1.51]), np.array([-2.0, -2.0, -2.0]))
    assert list(inside) == [True, True, False]


def test_steps_profile_matches_construction():
    scene = scenes.steps(step_height=0.1, step_depth=0.5, n=4, x0=1.0)
    # A point in the middle of step i sits at (i + 1) * step_height.
    for i in range(4):
        x = 1.0 + (i + 0.5) * 0.5
        assert scene.analytic_height(np.array(x), np.array(0.0)) == pytest.approx((i + 1) * 0.1)
    # Before the staircase and beside it, ground level.
    assert scene.analytic_height(np.array(0.0), np.array(0.0)) == pytest.approx(0.0)
    assert scene.analytic_height(np.array(1.25), np.array(3.0)) == pytest.approx(0.0)


def test_gap_has_a_gap():
    scene = scenes.gap(platform_z=0.35, gap_width=0.7)
    assert scene.analytic_height(np.array(0.0), np.array(0.0)) == pytest.approx(0.0)
    assert scene.analytic_height(np.array(-1.5), np.array(0.0)) == pytest.approx(0.35)
    assert scene.analytic_height(np.array(1.5), np.array(0.0)) == pytest.approx(0.35)


def test_slope_gradient():
    scene = scenes.slope(angle_deg=15.0, x0=0.8)
    k = np.tan(np.deg2rad(15.0))
    for x in (1.0, 1.5, 2.0):
        assert scene.analytic_height(np.array(x), np.array(0.0)) == pytest.approx(k * (x - 0.8))
    assert scene.analytic_height(np.array(0.0), np.array(0.0)) == pytest.approx(0.0)


def test_hfield_normalisation():
    hf = scenes.HField(name="t", radius_x=1.0, radius_y=1.0, fn=lambda x, y: 0.25 * (x + 1.0), spacing=0.1)
    assert hf.data.min() >= 0.0 and hf.data.max() == pytest.approx(1.0)
    assert hf.data.shape == (hf.nrow, hf.ncol)
    # Recovering heights from the normalised payload reproduces the generator.
    xs = np.linspace(-1.0, 1.0, hf.ncol)
    np.testing.assert_allclose(hf.data[0] * hf.elevation, 0.25 * (xs + 1.0), atol=1e-6)


def test_hfield_rejects_negative_heights():
    with pytest.raises(ValueError, match="non-negative"):
        scenes.HField(name="bad", radius_x=1.0, radius_y=1.0, fn=lambda x, y: x, spacing=0.5)


def test_unknown_scene_name():
    with pytest.raises(KeyError, match="unknown scene"):
        scenes.make_scene("does_not_exist")


def test_boxes_scene_is_deterministic_and_leaves_origin_clear():
    a, b = scenes.boxes(n=8, seed=1), scenes.boxes(n=8, seed=1)
    assert [x.pos for x in a.boxes] == [x.pos for x in b.boxes]
    assert [x.pos for x in scenes.boxes(n=8, seed=2).boxes] != [x.pos for x in a.boxes]
    assert a.analytic_height(np.array(0.0), np.array(0.0)) == pytest.approx(0.0)
