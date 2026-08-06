# MuJoCo simulation harness

A closed-loop test bench for `elevation_mapping_cupy_core`: it builds procedural
terrain in MuJoCo, feeds ray-cast depth frames into `ElevationMap`, and scores
the resulting map cell-for-cell against a top-down ray-cast ground-truth height
map.

```
scene (MJCF)  ──►  depth frames (mj_multiRay)  ──►  ElevationMap  ──┐
      │                                                             ├─►  metrics
      └────────►  ground truth (mj_ray, straight down)  ────────────┘
```

Nothing renders. Both the sensor and the ground truth are ray casts, so the
whole harness runs headless on CPU apart from the mapping itself.

## Running it

```bash
pixi run test-sim
```

```bash
pixi run sim --all --out sim/report
```

`pixi run sim --help` lists the knobs (scene, trajectory, resolution, map size,
depth noise, dropout).

## Visualising it

Two ways in, both writing PNGs — nothing needs a display.

**From the CLI**, with `--out DIR` and `--plots`:

```bash
pixi run sim --scene mixed --trajectory line --steps 12 --plots all --out sim/report
```

**From the tests**, with `--viz-dir`. Every test that has something worth
looking at emits a figure named after itself, so you get a picture of exactly
what was asserted:

```bash
pixi run pytest sim/tests --viz-dir sim/report
```

| Plot | Shows | Test it illustrates |
| --- | --- | --- |
| `comparison` | Ground truth, estimate, signed error side by side | `test_elevation_accuracy_per_scene` |
| `layers` | Every exported layer on one sheet | `test_is_valid_marks_exactly_the_observed_cells` |
| `surface` | Ground truth vs estimate as 3D surfaces | `test_step_heights_are_resolved`, `test_slope_gradient_is_recovered` |
| `convergence` | Coverage and error against frame number | `test_map_converges_as_frames_accumulate` |
| `filmstrip` | The robot-centric window sliding across a fixed world | `test_world_fixed_features_keep_their_world_position` |

`convergence` and `filmstrip` need per-step snapshots, so they imply
`RunConfig(record_per_step=True)`; the CLI sets that for you when you ask for
them. They also want a moving trajectory (`--trajectory line` or `circle`) to
show anything interesting.

To add a figure to a test, take the `viz` fixture and call it — it is a no-op
unless `--viz-dir` was passed, so it is safe to leave in place:

```python
def test_something(sim_run, viz):
    result = sim_run("key", cfg)
    viz("comparison", result, radius=2.5)
```

`kind` is any `plot_*` function in `emsim/plotting.py`.

### Seeing the scene itself

```bash
pixi run sim --all --preview --out sim/report
```

Draws each scene as a 3D surface and a top-down height map, plus one depth frame
and one LiDAR scan overlaid as point clouds — which side by side is the clearest
statement of how differently the two sensors sample the same ground. No GPU, no
mapping.

It is drawn from the scene's own geometry rather than rendered through OpenGL.
There is no working GL stack here: the Jetson's EGL vendor driver is not visible
to this environment's `libEGL`, and Mesa's software EGL fails too. For this
purpose the height field plus the sensor returns is more informative than a
shaded screenshot anyway.

## Layout

| Module | Responsibility |
| --- | --- |
| `emsim/scenes.py` | Procedural terrain as MJCF, each with a closed-form surface |
| `emsim/heightmap.py` | Top-down ray-cast ground truth; map-grid alignment |
| `emsim/sensor.py` | Pinhole depth camera via `mj_multiRay`, with optional noise |
| `emsim/lidar.py` | Optional LiDAR backend: real scan patterns via `mujoco-lidar` |
| `emsim/runner.py` | Drives `ElevationMap` over a trajectory, collects timings |
| `emsim/metrics.py` | RMSE / MAE / bias / p95 / coverage against ground truth |
| `emsim/plotting.py` | Figures: comparison, layers, surface, convergence, filmstrip |
| `emsim/cli.py` | `python -m emsim.cli` |

Only `runner.py` needs CuPy. Scenes, the sensor and the ground-truth sampler are
pure CPU, so the majority of the tests run anywhere.

## Sensors

Two backends, same interface (`pose_for` then `capture` -> `DepthCapture`), so
the run loop does not care which it holds. Pick with `RunConfig(sensor=...)` or
`--sensor`.

**`camera`** (default) -- a pinhole depth camera, dense and short-range. Every
one of its 19 200 rays returns, all within about 3 m.

**`lidar`** -- real scan patterns through [mujoco-lidar][mjlidar]: `vlp32`,
`hdl64`, `os128`, `airy96`, Livox `mid360` / `avia` / `mid70` / `horizon`, and a
plain `grid`. Sparse, ring-structured and long-range: a VLP-32 returns ~53 000
of 120 000 rays, spread over tens of metres. Livox patterns are non-repetitive,
so successive scans sample different points -- which is the property that makes
them worth testing against.

That difference is the point. The camera hands the map a dense patch; the LiDAR
hands it sparse rings whose density falls off sharply with range, which is what
the package actually receives on a robot.

A level-mounted spinning unit puts most of its rings above the horizon and
covers only ~12% of a 2.5 m disc. `lidar_tilt_down_deg` fixes that; the default
of 20 deg takes coverage to ~85%:

| pattern | tilt | coverage r<=2.5 m | rmse | returns/scan |
| --- | --- | --- | --- | --- |
| `vlp32` | 0 deg | 11.7% | 0.0230 | 26 250 |
| `vlp32` | 20 deg | 85.4% | 0.0102 | 52 709 |
| `os128` | 10 deg | 75.2% | 0.0075 | 106 922 |
| `mid360` | 30 deg | 74.4% | 0.0124 | 5 117 |
| `avia` | 25 deg | 96.5% | 0.0094 | 20 642 |

The `cpu` backend is `mj_multiRay` underneath, the same call the depth camera
uses, so it buys scan patterns rather than speed. `mujoco-lidar` also offers
`warp`, `taichi` and `jax` backends (`lidar_backend=`) which are much faster on
large scans, but each needs its own package installed.

[mjlidar]: https://github.com/discoverse-dev/MuJoCo-LiDAR

## Scenes

| Scene | What it exercises |
| --- | --- |
| `flat` | Baseline accuracy and bias on a featureless plane |
| `slope` | Gradient recovery on a 15° ramp |
| `rough` | Smoothly undulating ground |
| `steps` | Sharp vertical discontinuities; self-occlusion |
| `gap` | A hole the sensor cannot see into |
| `wall` | Occlusion of terrain behind an obstacle |
| `boxes` | Deterministic clutter field |
| `mixed` | Rough ground + staircase + clutter |

Terrain with vertical faces is built from axis-aligned boxes; smooth terrain
from height fields. Either way `Scene.analytic_height` gives the surface in
closed form, which is what validates the ray-cast sampler in
`tests/test_heightmap.py` before any accuracy test relies on it.

## How ground truth lines up with the map

`ElevationMap` cell centres sit at `center + (i + 0.5 - N/2) * resolution`, and
the centre only ever moves in whole-cell steps, so every cell centre the map can
have lands on the `(k + 0.5) * resolution` lattice. `GroundTruthHeightmap` builds
its grid on exactly that lattice, so comparing a window is integer indexing —
no interpolation between the estimate and the truth.

`get_map_with_name_ref` flips both axes on export (row 0 is maximum x, column 0
maximum y); `map_cell_centers` reproduces that layout.

Two wrinkles worth knowing about:

- **Height-field triangle edges.** MuJoCo splits each height-field quad into two
  triangles, and a ray landing exactly on the shared diagonal can miss both and
  fall through. A grid symmetric about the origin puts every `x == y` sample on
  such a diagonal, so the sampler casts a second ray at a 10 µm asymmetric
  offset and keeps the higher hit. `GroundTruthHeightmap.n_edge_repairs` counts
  how often that mattered.
- **Near-horizontal rays.** MuJoCo walks height fields cell by cell, so rays that
  skim across one cost far more than rays that hit. The default camera tilt and
  FOV keep the top image row 15° below the horizon, which is worth roughly a
  13× speedup on height-field scenes and costs no usable coverage.

## CUDA on Jetson

The environment is self-contained — no `sudo`, no writing into `/usr/local/cuda`.
The one external prerequisite is JetPack itself (this was developed against
JetPack 6 / R36.4.7, CUDA 12.6), which supplies the driver and CUDA runtime.

Two pieces need care on aarch64:

- **CuPy** comes from stock PyPI (`cupy-cuda12x`), whose aarch64 wheels work
  against the JetPack CUDA runtime.
- **PyTorch** — needed only by the traversability filter, which calls `.cuda()`
  — has no CUDA-capable aarch64 wheel on stock PyPI. It comes from NVIDIA's
  Jetson index, `https://pypi.jetson-ai-lab.io/jp6/cu126`, scoped to the
  `linux-aarch64` target so the manifest still resolves on x86. That wheel links
  against cuDSS, which JetPack does not ship, so `nvidia-cudss-cu12` is pulled
  from PyPI and its lib directory added to `LD_LIBRARY_PATH` in
  `[target.linux-aarch64.activation.env]`.

If the traversability filter cannot be loaded, `ElevationMap` logs a warning and
runs without it. `RunResult.traversability_enabled` reports the state, and the
one test that genuinely depends on the filter (`test_traversability_layer`)
skips rather than passing vacuously.

## A defect this harness caught

`update_map_with_kernel` ends with `self.update_normal(self.traversability_input)`,
but a local change had moved the dilation that fills `traversability_input`
inside the `if self.traversability_filter is not None:` branch. With the filter
loaded the normals were fine; with it unavailable that buffer stayed all zeros
and `normal_x`/`normal_y`/`normal_z` silently became `(0, 0, 1)` everywhere
regardless of terrain — an undeclared dependency of the normal layers on the
traversability filter, with no warning beyond the one about the filter itself.

The dilation does not depend on the learned filter, and upstream runs it
unconditionally, so it now sits outside the guard and only the
`traversability_filter(...)` call remains behind it — fixed separately in #4,
which this branch assumes. The two normals tests in `tests/test_map_layers.py`
assert the real behaviour in both configurations: the harness recovers ~15° on a
15° ramp and `normal_z ≈ 1.0` on flat ground whether or not the filter is
loaded.

## Tests

| File | Needs GPU | Covers |
| --- | --- | --- |
| `test_scenes.py` | no | MJCF compiles; closed-form surfaces are what they claim |
| `test_heightmap.py` | no | Ray-cast truth vs analytic; grid alignment; body exclusion |
| `test_sensor.py` | no | Camera intrinsics, pose conventions, range/dropout noise |
| `test_lidar.py` | mixed | Scan patterns, frames, body exclusion, per-pattern accuracy |
| `test_elevation_accuracy.py` | yes | Per-scene RMSE/p95/coverage; gradients; step heights; convergence; noise |
| `test_map_shifting_sim.py` | yes | Centre tracking; world-fixed features survive shifting |
| `test_map_layers.py` | yes | `is_valid`, `variance`, `time`, `upper_bound`, plugins, `clear` |
| `test_performance.py` | yes | Ingestion throughput; no per-frame stalls |

Accuracy thresholds sit well above what the pipeline currently achieves — they
are there to catch regressions, not to pin down today's exact numbers.
