import re
from typing import List

import cupy as cp
import cv2 as cv
import numpy as np

from .plugin_manager import PluginBase

# Name of the auxiliary semantic layer that tracks camera observations.
_SEM_OBSERVED_LAYER = "_sem_observed"


class JointInpainting(PluginBase):
    """Jointly inpaints elevation and semantic layers from cells that have
    BOTH geometric (LiDAR) AND semantic (camera) observations.

    Traditional elevation mapping treats geometric and semantic inpainting
    separately.  Geometric inpainting fills elevation for cells without LiDAR
    observations, using all LiDAR-observed cells as sources — even if those
    cells lack camera coverage.  Semantic inpainting then tries to fill
    semantics into unobserved cells but can be blocked by thin bands of
    observed-but-invalid semantic data (e.g. the top of an occluding box).

    This plugin solves both problems by defining a single "trusted source" set:
    cells that have been observed by BOTH LiDAR (is_valid == 1) AND camera
    (_sem_observed >= 0.5).  Everything else is inpainted from those trusted
    sources — guaranteeing that propagated geometry always carries semantics
    and vice versa.

    An optional ``mask_dilation`` parameter (in grid cells) expands the
    inpainting mask into the jointly-observed region.  This erodes thin bands
    of unreliable data at sensor boundaries (e.g. the top edge of a box seen
    at a steep angle) so that the inpainting sources are cells with confident
    observations further from the boundary.

    Returns:
        The inpainted elevation layer (2-D, cell_n × cell_n).
        As a side effect, matched semantic layers are also inpainted in-place.

    Args:
        cell_n: Grid dimension (width = height).
        semantic_classes: List of regex patterns to match semantic layer names
            for inpainting.  Defaults to ``[".*"]`` (all non-internal layers).
        method: OpenCV inpainting method, ``'telea'`` or ``'ns'``.
        inpaint_radius: Radius of circular neighbourhood for inpainting
            (in grid cells).
        mask_dilation: Number of cells to dilate the inpainting mask.  This
            eats into the observed border, removing thin unreliable bands
            at observation boundaries.  0 = no dilation (strict).
    """

    def __init__(
        self,
        cell_n: int = 100,
        semantic_classes: list = None,
        method: str = "telea",
        inpaint_radius: int = 3,
        mask_dilation: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.semantic_classes = semantic_classes if semantic_classes is not None else [".*"]
        self.inpaint_radius = inpaint_radius
        self.mask_dilation = mask_dilation
        if method == "telea":
            self.method = cv.INPAINT_TELEA
        elif method == "ns":
            self.method = cv.INPAINT_NS
        else:
            self.method = cv.INPAINT_TELEA

    def _get_matching_semantic_indices(self, layer_names: List[str]) -> List[int]:
        """Return indices of semantic layers whose names match any configured
        regex pattern.  Internal layers (starting with ``_``) are excluded."""
        indices = []
        for i, name in enumerate(layer_names):
            if name.startswith("_"):
                continue
            if any(re.match(pattern, name) for pattern in self.semantic_classes):
                indices.append(i)
        return indices

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        rotation,
        elements_to_shift,
        *args,
    ) -> cp.ndarray:
        """Jointly inpaint elevation and semantics from cells observed by both
        LiDAR and camera.

        Args:
            elevation_map: Core layers ``(N, cell_n, cell_n)``.
                Index 0 = elevation, 2 = is_valid.
            layer_names: Names of core layers.
            plugin_layers: Layers from other plugins.
            plugin_layer_names: Plugin layer names.
            semantic_map: Semantic map ``(N_sem, cell_n, cell_n)``.
            semantic_layer_names: Semantic layer names.
            rotation: Robot base rotation matrix.
            elements_to_shift: Shift tracking elements.

        Returns:
            Inpainted elevation layer (2-D float64 array, cell_n × cell_n).
        """
        # ------------------------------------------------------------------ #
        # 1. Build the joint observation mask
        # ------------------------------------------------------------------ #
        is_valid_gpu = elevation_map[2]  # LiDAR observed cells

        has_obs_layer = _SEM_OBSERVED_LAYER in semantic_layer_names
        if has_obs_layer:
            obs_idx = semantic_layer_names.index(_SEM_OBSERVED_LAYER)
            sem_observed_gpu = semantic_map[obs_idx]
            jointly_observed = (is_valid_gpu >= 0.5) & (sem_observed_gpu >= 0.5)
        else:
            # Fallback: only use is_valid (no camera info available yet)
            jointly_observed = (is_valid_gpu >= 0.5)

        # Mask: cells that are NOT jointly observed → need inpainting
        inpaint_mask_gpu = ~jointly_observed

        # ------------------------------------------------------------------ #
        # 1b. Dilate the mask to erode thin unreliable observation borders.
        #     At sensor boundaries (e.g. the top edge of an occluding box
        #     seen at a steep camera angle) the camera may assign low-quality
        #     semantics.  Dilating the mask eats into those border cells so
        #     the inpainting sources are confidently-observed cells further
        #     from the boundary.
        # ------------------------------------------------------------------ #
        if self.mask_dilation > 0:
            mask_cpu_raw = cp.asnumpy(inpaint_mask_gpu.astype(np.uint8))
            kernel = cv.getStructuringElement(
                cv.MORPH_ELLIPSE,
                (2 * self.mask_dilation + 1, 2 * self.mask_dilation + 1),
            )
            mask_dilated = cv.dilate(mask_cpu_raw, kernel, iterations=1)
            inpaint_mask_gpu = cp.asarray(mask_dilated.astype(bool))

        # If everything is observed, nothing to do
        if not cp.any(inpaint_mask_gpu):
            return elevation_map[0].copy().astype(np.float64)

        mask_cpu = cp.asnumpy(inpaint_mask_gpu.astype(np.uint8))

        # ------------------------------------------------------------------ #
        # 2. Inpaint elevation
        # ------------------------------------------------------------------ #
        elev_cpu = cp.asnumpy(elevation_map[0])

        # Only inpaint if there are valid source cells
        if (mask_cpu < 1).any():
            h_valid = elev_cpu[mask_cpu < 1]
            h_max = float(h_valid.max())
            h_min = float(h_valid.min())
            h_range = h_max - h_min
            if h_range < 1e-6:
                h_range = 1.0  # Avoid division by zero for flat maps

            h_normalized = ((elev_cpu - h_min) * 255.0 / h_range).clip(0, 255).astype(np.uint8)
            h_inpainted_u8 = cv.inpaint(h_normalized, mask_cpu, self.inpaint_radius, self.method)
            h_inpainted = h_inpainted_u8.astype(np.float32) * h_range / 255.0 + h_min
            elevation_result = cp.asarray(h_inpainted, dtype=np.float64)
        else:
            elevation_result = elevation_map[0].copy().astype(np.float64)

        # ------------------------------------------------------------------ #
        # 3. Inpaint semantic layers in-place
        # ------------------------------------------------------------------ #
        sem_indices = self._get_matching_semantic_indices(semantic_layer_names)

        for idx in sem_indices:
            sem_layer_gpu = semantic_map[idx]
            layer_cpu = cp.asnumpy(sem_layer_gpu)

            layer_clamped = np.clip(layer_cpu, 0.0, 1.0)
            layer_u8 = (layer_clamped * 255).astype(np.uint8)

            inpainted_u8 = cv.inpaint(
                layer_u8, mask_cpu, self.inpaint_radius, self.method
            )
            inpainted_f32 = inpainted_u8.astype(np.float32) / 255.0

            inpainted_gpu = cp.asarray(inpainted_f32, dtype=sem_layer_gpu.dtype)
            semantic_map[idx] = cp.where(inpaint_mask_gpu, inpainted_gpu, sem_layer_gpu)

        return elevation_result
