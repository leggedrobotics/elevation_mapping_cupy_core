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

Both the sensor and the ground truth are ray casts, so the whole harness runs
headless on CPU apart from the mapping itself. Rendering is only ever used for
looking at scenes, never for producing data.

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

It is drawn from the scene's own geometry, which shows the sensor returns
against the terrain. For an actual rendered view, see below.

### Rendered views

```bash
pixi run render --all --views all --out sim/report
```

Renders through MuJoCo's own rasteriser: `oblique`, `front` and `top` presets,
about a second per frame. The cameras sit deliberately low — these terrains have
decimetre relief over metres, which flat-shades into invisibility from above.

Getting this working headless on a Jetson takes some setup, which is why it has
its own module and pixi task:

- Tegra's EGL exposes no usable `EGL_PLATFORM_DEVICE_EXT` display and is
  GLES-only, while MuJoCo's context asks for desktop `EGL_OPENGL_BIT`. So the
  render goes through Mesa's software rasteriser (`mesalib`), with the Gallium
  driver forced to `llvmpipe` — otherwise Mesa tries the Tegra KMS nodes and
  reports `kmsro: driver missing`.
- Several EGL devices are enumerated and only one yields a working context.
  MuJoCo caches its EGL display on the first context it builds, so a failed
  attempt cannot be retried in-process. `emsim.render.pick_egl_device` therefore
  runs the whole handshake itself first and sets `MUJOCO_EGL_DEVICE_ID`.
- `LD_LIBRARY_PATH` has to point at Mesa *before the process starts*, since the
  dynamic loader reads it once. That is what the `render` pixi task is for;
  running `python -m emsim.render` outside it will not find Mesa.

Rendering is for looking at scenes, not for producing data: the sensors and the
ground truth are ray casts and never touch OpenGL.

Rendered PNGs are gitignored.

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
| `emsim/render.py` | Rendered views through MuJoCo's rasteriser (headless via Mesa) |
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

### Ray-cast backends

`lidar_backend` defaults to `"auto"`: Warp when CUDA is available, `cpu`
otherwise. The `cpu` backend is `mj_multiRay` underneath — the same call the
depth camera uses — so it buys scan patterns rather than speed. `taichi` and
`jax` are also selectable but need their own packages.

Per-scan cost of a VLP-32 (120 000 rays) on an Orin, and the resulting maps are
identical to float32 precision:

| scene | cpu | warp | |
| --- | --- | --- | --- |
| `flat` | 27.9 ms | 11.0 ms | 2.5x |
| `boxes` | 36.4 ms | 12.8 ms | 2.9x |
| `wall` | 33.2 ms | 11.3 ms | 2.9x |
| `gap` | 38.3 ms | 10.4 ms | 3.7x |
| `steps` | 40.3 ms | 10.6 ms | 3.8x |
| `mixed` | 133.3 ms | 14.9 ms | 8.9x |
| `rough` | 204.0 ms | 15.6 ms | 13.1x |
| `slope` | **11 016 ms** | 12.5 ms | **884x** |

`slope` is the case that makes Warp effectively mandatory rather than merely
nice. MuJoCo's CPU height-field ray cast walks the grid cell by cell, so a ray
skimming along a flat height field crosses thousands of cells before it exits —
and a 360-degree LiDAR aims a whole ring's worth of rays exactly like that. At
11 s per scan a 24-frame run takes four and a half minutes on CPU and under a
second on Warp. Warp builds a BVH instead, so it barely notices.

Warp compiles its kernels once (~14 s), then loads them from `~/.cache/warp` in
under 2 ms.

[mjlidar]: https://github.com/discoverse-dev/MuJoCo-LiDAR

## Trajectories and body motion

`RunConfig.trajectory` sets the nominal path:

| kind | motion |
| --- | --- |
| `static` | fixed pose |
| `spin` | rotate in place through a full turn |
| `line` | translate along +x |
| `circle` | translate and rotate together, facing the centre |
| `figure8` | a lemniscate: heading sweeps back and forth, yaw rate reverses sign |

`RunConfig.body_motion` then adds what a legged base actually does on top of
that path — vertical bob, lateral sway (in the *body* frame, so it follows the
heading), and roll/pitch. All amplitudes default to zero, so it changes nothing
unless asked for:

```python
RunConfig(trajectory="circle", body_motion=BodyMotion.walking())
```

`BodyMotion.walking()` uses amplitudes a trotting quadruped shows: 4 cm bob,
3 cm sway, 4 deg roll, 3 deg pitch, six cycles over the run. The four terms are
quarter-cycle out of phase, so the attitude traces a loop instead of heaving up
and down in lockstep.

This matters because the sensor is bolted to the base: with body motion on,
every frame arrives from a different attitude, so a pose-handling error cannot
hide behind a constant offset. The full base rotation — not just heading — is
what gets handed to `ElevationMap.move_to`.

One caveat when reading amplitudes back: a sinusoid sampled at `n_steps` points
does not generally land on its peaks. Six cycles over 24 steps samples every
90 deg, so a term offset by 45 deg only ever reaches 0.707 of its amplitude.
The tests bound the observed span rather than asserting the nominal value.

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
| `test_trajectories.py` | mixed | Paths, body motion, mapping under full 6-DoF pose |
| `test_map_shifting_sim.py` | yes | Centre tracking; world-fixed features survive shifting |
| `test_map_layers.py` | yes | `is_valid`, `variance`, `time`, `upper_bound`, plugins, `clear` |
| `test_performance.py` | yes | Ingestion throughput; no per-frame stalls |

Accuracy thresholds sit well above what the pipeline currently achieves — they
are there to catch regressions, not to pin down today's exact numbers.
