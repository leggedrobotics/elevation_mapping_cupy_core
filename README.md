# elevation_mapping_cupy_core

A standalone, ROS-free Python package extracted from [elevation_mapping_cupy](https://github.com/leggedrobotics/elevation_mapping_cupy) (ETH Zurich / Takahiro Miki).

The upstream project is a full ROS2 node for real-time GPU-accelerated elevation mapping on legged robots. This package strips all ROS dependencies and exposes only the core Python library, making it usable in:

- **Simulation** pipelines (Isaac Sim, Gazebo, custom envs)
- **Inference** pipelines that consume elevation or semantic maps
- Any **standalone Python** code that needs GPU-backed elevation mapping without a ROS runtime

## What's included

- `ElevationMap` — core CuPy-based elevation map with GPU kernels for point cloud fusion, traversability filtering, and map shifting
- `SemanticMap` — semantic layer management with pluggable fusion strategies (Bayesian, class-average, class-max, etc.)
- `PluginManager` — post-processing plugins (inpainting, erosion, smooth filter, traversability, PCA features, etc.)
- `Parameter` — unified configuration dataclass
- Custom CUDA kernels for fast point cloud ingestion and map updates

## Installation

```bash
pip install elevation-mapping-cupy-core
```

Requires a CUDA 12.x environment with CuPy installed (`cupy-cuda12x`).

## Quick start

```python
from elevation_mapping_cupy import ElevationMap, Parameter

param = Parameter(resolution=0.04, map_length=8.0)
em = ElevationMap(param)

# Feed a point cloud (N, 3) numpy array
em.input(points, R, t, position_noise, orientation_noise)

# Retrieve the elevation layer as a numpy array
elevation = em.get_map_with_name("elevation")
```

## Development and simulation testing

A [pixi](https://pixi.sh) environment pins the full toolchain (CuPy, MuJoCo,
CUDA-enabled PyTorch, pytest) — no `sudo`, no system installs beyond JetPack /
the CUDA runtime itself:

```bash
pixi run test-all
```

`sim/` holds a MuJoCo test bench that drives `ElevationMap` over procedural
terrain with a ray-cast depth camera and scores the result against a top-down
ray-cast ground-truth height map:

```bash
pixi run sim --all --out sim/report
```

See [sim/README.md](sim/README.md) for the scene catalogue, how ground truth is
aligned to the map grid, and what each test covers.

## License

MIT — Copyright (c) 2022 ETH Zurich, Takahiro Miki. See [LICENSE](LICENSE).
