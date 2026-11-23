"""Mapping speed under each sensor load, as a markdown table (`pixi run bench`).

"input" is one frame end to end: host-to-device copy of the points, fusion,
variance and time update, with the GPU synchronised on both sides. The first
WARMUP frames are dropped (kernel compilation). Layer readout is left out: the
runner reads each layer once, cold, so it would time plugin setup.
"""

from __future__ import annotations

import argparse
import platform

import cupy as cp
import mujoco
import numpy as np

from emsim.runner import RunConfig, run

WARMUP = 3
# (label, sensor, lidar pattern)
SENSORS = [
    ("depth camera 160x120", "camera", "vlp32"),
    ("VLP-32", "lidar", "vlp32"),
    ("OS-128", "lidar", "os128"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", default="mixed")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--resolution", type=float, default=0.04)
    p.add_argument("--map-lengths", default="8,16", help="comma-separated map side lengths [m]")
    args = p.parse_args()
    if args.steps <= WARMUP:
        p.error(f"--steps must exceed the {WARMUP} warm-up frames")

    dev = cp.cuda.runtime.getDeviceProperties(0)
    print(f"{platform.node()}: {dev['name'].decode()}, CUDA runtime {cp.cuda.runtime.runtimeGetVersion()}, "
          f"cupy {cp.__version__}, mujoco {mujoco.__version__}; scene={args.scene}, {args.steps} frames, "
          f"{args.resolution} m cells\n")
    print("| sensor | map | points/frame | input p50 | input p95 | rmse |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    # One throwaway run so NVRTC compilation is not charged to the first row.
    run(RunConfig(scene=args.scene, n_steps=WARMUP + 1), layers=("elevation",))
    for length in (float(x) for x in args.map_lengths.split(",")):
        for label, sensor, pattern in SENSORS:
            cfg = RunConfig(scene=args.scene, trajectory="circle", n_steps=args.steps, sensor=sensor,
                            lidar_pattern=pattern, resolution=args.resolution, map_length=length)
            r = run(cfg, layers=("elevation",))
            steady = r.timings.input_ms[WARMUP:]
            cells = r.cell_n
            print(f"| {label} | {length:.0f} m ({cells}x{cells}) | {int(np.mean(r.n_points))} "
                  f"| {np.median(steady):.2f} ms | {np.percentile(steady, 95):.2f} ms "
                  f"| {r.error('elevation', radius=2.5).rmse:.4f} |",
                  flush=True)


if __name__ == "__main__":
    main()
