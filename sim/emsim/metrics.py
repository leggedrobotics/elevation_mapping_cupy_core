"""Error metrics for an estimated elevation map against ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class MapError:
    """Cell-wise agreement between an estimated layer and ground truth.

    Coverage is reported separately from accuracy on purpose: a map that only
    fills in the easy cells can post an excellent RMSE, so a test needs to pin
    both down.
    """

    n_total: int  # cells in the comparison region
    n_valid: int  # cells with both an estimate and ground truth
    coverage: float  # n_valid / n_total
    mae: float
    rmse: float
    bias: float  # mean signed error (estimate - truth)
    p95: float  # 95th percentile absolute error
    max_abs: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"coverage={self.coverage:6.1%}  rmse={self.rmse:7.4f}  mae={self.mae:7.4f}  "
            f"bias={self.bias:+7.4f}  p95={self.p95:7.4f}  max={self.max_abs:7.4f}  "
            f"(n={self.n_valid}/{self.n_total})"
        )


def compare_maps(
    estimate: np.ndarray,
    truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> MapError:
    """Compare two same-shaped layers, ignoring NaN on either side.

    Args:
        estimate: Estimated heights; NaN marks an unobserved cell.
        truth: Ground-truth heights; NaN marks a cell outside the sampled region.
        mask: Optional boolean region of interest.

    Returns:
        A :class:`MapError`. Accuracy fields are NaN when nothing overlaps.
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if estimate.shape != truth.shape:
        raise ValueError(f"shape mismatch: estimate {estimate.shape} vs truth {truth.shape}")

    region = np.ones(estimate.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    # Cells that could ever be scored: inside the region and with known truth.
    scorable = region & np.isfinite(truth)
    valid = scorable & np.isfinite(estimate)

    n_total = int(scorable.sum())
    n_valid = int(valid.sum())
    if n_valid == 0:
        nan = float("nan")
        return MapError(n_total, 0, 0.0, nan, nan, nan, nan, nan)

    err = estimate[valid] - truth[valid]
    abs_err = np.abs(err)
    return MapError(
        n_total=n_total,
        n_valid=n_valid,
        coverage=n_valid / n_total if n_total else 0.0,
        mae=float(abs_err.mean()),
        rmse=float(np.sqrt(np.mean(err**2))),
        bias=float(err.mean()),
        p95=float(np.percentile(abs_err, 95)),
        max_abs=float(abs_err.max()),
    )


@dataclass
class Timings:
    """Wall-clock cost of the mapping pipeline, in milliseconds per frame."""

    sensor_ms: np.ndarray
    input_ms: np.ndarray
    export_ms: np.ndarray

    def summary(self) -> Dict[str, float]:
        def stats(name: str, a: np.ndarray) -> Dict[str, float]:
            if a.size == 0:
                return {f"{name}_mean_ms": float("nan"), f"{name}_p95_ms": float("nan")}
            return {
                f"{name}_mean_ms": float(np.mean(a)),
                f"{name}_p95_ms": float(np.percentile(a, 95)),
            }

        out: Dict[str, float] = {}
        out.update(stats("sensor", self.sensor_ms))
        out.update(stats("input", self.input_ms))
        out.update(stats("export", self.export_ms))
        mean_input = out["input_mean_ms"]
        out["input_hz"] = 1000.0 / mean_input if mean_input and np.isfinite(mean_input) and mean_input > 0 else float("nan")
        return out
