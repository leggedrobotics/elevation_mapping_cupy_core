#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
from typing import List
import cupyx.scipy.ndimage as ndimage
import numpy as np
import cv2 as cv
import logging

_LOGGER = logging.getLogger(__name__)

from .plugin_manager import PluginBase

# Push-pull hole filling (Gortler et al. 1996), all on the GPU. Pull: each coarse
# cell is the weighted mean of its 2x2 children, weight min(1, sum). Push: each
# fine cell blends towards the bilinearly upsampled coarse level by 1 - weight,
# then a few Jacobi steps of Laplace smoothing on the same blend (a cascadic
# multigrid for a membrane fill). Observed cells (weight 1) keep their height.
_KERNELS = cp.RawModule(
    code=r"""
extern "C" __global__ void pull(const float* h, const float* w, int fr, int fc,
                                float* hc, float* wc, int cr, int cc) {
  int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cr || j >= cc) return;
  float sw = 0.f, swh = 0.f;
  for (int di = 0; di < 2; ++di)
    for (int dj = 0; dj < 2; ++dj) {
      int y = 2 * i + di, x = 2 * j + dj;
      if (y < fr && x < fc) { float k = w[y * fc + x]; sw += k; swh += k * h[y * fc + x]; }
    }
  hc[i * cc + j] = sw > 0.f ? swh / sw : 0.f;
  wc[i * cc + j] = fminf(sw, 1.f);
}

// out = w * h + (1 - w) * (bilinear upsample of the coarse level `src`)
extern "C" __global__ void push(const float* h, const float* w, int r, int c,
                                const float* src, int cr, int cc, float* out) {
  int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= r || j >= c) return;
  // Fine cell centre in coarse cell coordinates, clamped to the coarse grid.
  float y = fminf(fmaxf(0.5f * i - 0.25f, 0.f), cr - 1.f), x = fminf(fmaxf(0.5f * j - 0.25f, 0.f), cc - 1.f);
  int y0 = (int)y, x0 = (int)x, y1 = min(y0 + 1, cr - 1), x1 = min(x0 + 1, cc - 1);
  float ty = y - y0, tx = x - x0;
  float up = (1.f - ty) * ((1.f - tx) * src[y0 * cc + x0] + tx * src[y0 * cc + x1])
           + ty * ((1.f - tx) * src[y1 * cc + x0] + tx * src[y1 * cc + x1]);
  float k = w[i * c + j];
  out[i * c + j] = k * h[i * c + j] + (1.f - k) * up;
}

// out = w * h + (1 - w) * (mean of the 4 neighbours in `src`, edges clamped)
extern "C" __global__ void smooth(const float* h, const float* w, int r, int c,
                                  const float* src, float* out) {
  int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= r || j >= c) return;
  float k = w[i * c + j];
  float nb = 0.25f * (src[max(i - 1, 0) * c + j] + src[min(i + 1, r - 1) * c + j]
                    + src[i * c + max(j - 1, 0)] + src[i * c + min(j + 1, c - 1)]);
  out[i * c + j] = k * h[i * c + j] + (1.f - k) * nb;
}
"""
)
_SMOOTH_STEPS = 4  # per level, even: the result ends in the push output


def push_pull_fill(height: cp.ndarray, weight: cp.ndarray) -> cp.ndarray:
    """Fills cells with weight 0 smoothly from the cells with weight 1.

    Asynchronous (no host read-back). Returns float32; zeros if no weight is
    non-zero.
    """
    pull, push, smooth = (_KERNELS.get_function(f) for f in ("pull", "push", "smooth"))
    h = cp.where(weight > 0, height, 0).astype(cp.float32)
    w = weight.astype(cp.float32)
    levels = [(h, w)]
    while h.shape[0] > 1 or h.shape[1] > 1:
        hc = cp.empty(((h.shape[0] + 1) // 2, (h.shape[1] + 1) // 2), cp.float32)
        wc = cp.empty_like(hc)
        pull(_grid(hc.shape), _BLOCK, (h, w, *_dims(h), hc, wc, *_dims(hc)))
        h, w = hc, wc
        levels.append((h, w))
    filled = levels[-1][0]
    for h, w in reversed(levels[:-1]):
        out, tmp = cp.empty_like(h), cp.empty_like(h)
        push(_grid(h.shape), _BLOCK, (h, w, *_dims(h), filled, *_dims(filled), out))
        for _ in range(_SMOOTH_STEPS // 2):
            smooth(_grid(h.shape), _BLOCK, (h, w, *_dims(h), out, tmp))
            smooth(_grid(h.shape), _BLOCK, (h, w, *_dims(h), tmp, out))
        filled = out
    return filled


_BLOCK = (16, 16, 1)


def _dims(a):
    return np.int32(a.shape[0]), np.int32(a.shape[1])


def _grid(shape):
    return ((shape[1] + 15) // 16, (shape[0] + 15) // 16, 1)


class Inpainting(PluginBase):
    """
    This class is used for inpainting, a process of reconstructing lost or deteriorated parts of images and videos.

    Args:
        cell_n (int): The number of cells. Default is 100.
        method (str): The inpainting method: 'telea' or 'ns' (Navier-Stokes), both
            OpenCV on the CPU, or 'push_pull', a smooth fill on the GPU that never
            waits for it (no 8-bit height quantisation either). Default is 'telea'.
        **kwargs (): Additional keyword arguments.
    """

    def __init__(self, cell_n: int = 100, method: str = "telea", **kwargs):
        super().__init__()
        self.push_pull = method == "push_pull"
        if method == "telea":
            self.method = cv.INPAINT_TELEA
        elif method == "ns":  # Navier-Stokes
            self.method = cv.INPAINT_NS
        else:  # default method
            self.method = cv.INPAINT_TELEA

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        valid_layer = elevation_map[2]
        if self.push_pull:
            elevation = elevation_map[0]
            valid = (valid_layer > 0.5) & cp.isfinite(elevation)
            filled = push_pull_fill(elevation, valid)
            # No valid cell at all: the elevation layer as is, like the OpenCV path.
            return cp.where(valid.any(), filled, elevation)
        mask_np = cp.asnumpy((valid_layer < 0.5).astype("uint8"))
        elevation = elevation_map[0]
        finite_elevation = cp.isfinite(elevation)
        valid_mask = cp.logical_and(valid_layer > 0.5, finite_elevation)

        if (mask_np < 1).any():
            if not cp.any(valid_mask):
                return elevation

            h_valid = elevation[valid_mask]
            h_max = float(cp.asnumpy(h_valid.max()))
            h_min = float(cp.asnumpy(h_valid.min()))
            denom = h_max - h_min
            if denom <= 1e-6:
                _LOGGER.warning(
                    "Inpainting detected near-flat terrain (h_min=%.3f, h_max=%.3f); broadcasting height.",
                    h_min,
                    h_max,
                )
                filled = cp.full(elevation.shape, h_max, dtype=cp.float32)
            else:
                # Replace NaNs with the minimum elevation value.
                safe_elevation = cp.where(finite_elevation, elevation, h_min)
                scaled = cp.asnumpy((safe_elevation - h_min) * 255.0 / denom).astype("uint8")
                dst = cv.inpaint(scaled, mask_np, 1, self.method)
                h_inpainted = dst.astype(np.float32) * denom / 255.0 + h_min
                filled = cp.asarray(h_inpainted, dtype=cp.float32)

            # Ensure already-valid cells mirror the authoritative elevation layer.
            filled = cp.where(valid_mask, elevation, filled)
            return filled.astype(cp.float64)
        else:
            return elevation_map[0]
