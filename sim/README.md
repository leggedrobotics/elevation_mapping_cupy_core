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

The CLI prints an accuracy table and, with `--out`, writes a ground-truth /
estimate / error triptych per scene. `pixi run sim --help` lists the knobs
(scene, trajectory, resolution, map size, depth noise, dropout).

## Layout

| Module | Responsibility |
| --- | --- |
| `emsim/scenes.py` | Procedural terrain as MJCF, each with a closed-form surface |
| `emsim/heightmap.py` | Top-down ray-cast ground truth; map-grid alignment |
| `emsim/sensor.py` | Pinhole depth camera via `mj_multiRay`, with optional noise |
| `emsim/runner.py` | Drives `ElevationMap` over a trajectory, collects timings |
| `emsim/metrics.py` | RMSE / MAE / bias / p95 / coverage against ground truth |
| `emsim/cli.py` | `python -m emsim.cli` |

Only `runner.py` needs CuPy. Scenes, the sensor and the ground-truth sampler are
pure CPU, so the majority of the tests run anywhere.

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

## Known gaps in this environment

- **Traversability filter disabled.** `get_filter_torch` calls `.cuda()`, and
  there is no CUDA-capable PyTorch wheel for this Jetson in a conda environment,
  so `weights.dat` fails to load and `ElevationMap` runs without the learned
  filter. This matches the existing unit-test suite. `RunResult.traversability_enabled`
  reports the state and the affected tests skip rather than pass vacuously.
- **Normal layers are not tested.** `update_map_with_kernel` feeds
  `update_normal` the `traversability_input` buffer, which is only populated
  inside the `traversability_filter is not None` branch. With the filter off that
  buffer stays all zeros, so `normal_x`/`normal_y`/`normal_z` come out as
  `(0, 0, 1)` everywhere regardless of terrain. The two tests in
  `tests/test_map_layers.py` assert the correct behaviour and skip until that is
  fixed — they pass no judgement on the current output.

## Tests

| File | Needs GPU | Covers |
| --- | --- | --- |
| `test_scenes.py` | no | MJCF compiles; closed-form surfaces are what they claim |
| `test_heightmap.py` | no | Ray-cast truth vs analytic; grid alignment; body exclusion |
| `test_sensor.py` | no | Intrinsics, pose conventions, range/dropout noise |
| `test_elevation_accuracy.py` | yes | Per-scene RMSE/p95/coverage; gradients; step heights; convergence; noise |
| `test_map_shifting_sim.py` | yes | Centre tracking; world-fixed features survive shifting |
| `test_map_layers.py` | yes | `is_valid`, `variance`, `time`, `upper_bound`, plugins, `clear` |
| `test_performance.py` | yes | Ingestion throughput; no per-frame stalls |

Accuracy thresholds sit well above what the pipeline currently achieves — they
are there to catch regressions, not to pin down today's exact numbers.
