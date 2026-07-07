#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
import string

from .fusion_manager import FusionBase


def exponential_correspondences_to_map_kernel(resolution, width, height, alpha, confidence_threshold):
    exponential_correspondences_to_map_kernel = cp.ElementwiseKernel(
        in_params="raw U sem_map, raw U map_idx, raw U image_mono, raw U confidence, raw U uv_correspondence, raw B valid_correspondence, raw U image_height, raw U image_width",
        out_params="raw U new_sem_map",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            int cell_idx = get_map_idx(i, 0);
            if (valid_correspondence[cell_idx]){
                int cell_idx_2 = get_map_idx(i, 1);
                int idx = int(uv_correspondence[cell_idx]) + int(uv_correspondence[cell_idx_2]) * image_width;
                
                // Check confidence threshold
                float conf = confidence[idx];
                if (conf >= ${confidence_threshold}) {
                    // Above threshold: fuse with exponential smoothing
                    new_sem_map[get_map_idx(i, map_idx)] = sem_map[get_map_idx(i, map_idx)] * (1-${alpha}) +  ${alpha} * image_mono[idx];
                } else {
                    // Below threshold: keep existing value (no fusion)
                    new_sem_map[get_map_idx(i, map_idx)] = sem_map[get_map_idx(i, map_idx)];
                }
            }else{
                new_sem_map[get_map_idx(i, map_idx)] = sem_map[get_map_idx(i, map_idx)];
            }

            """
        ).substitute(alpha=alpha, confidence_threshold=confidence_threshold),
        name="exponential_correspondences_to_map_kernel",
    )
    return exponential_correspondences_to_map_kernel


class ImageExponential(FusionBase):
    def __init__(self, params, *args, **kwargs):
        # super().__init__(fusion_params, *args, **kwargs)
        # print("Initialize fusion kernel")
        self.name = "image_exponential"
        self.cell_n = params.cell_n
        self.resolution = params.resolution
        
        # Read confidence threshold from params, default to 0.5
        self.confidence_threshold = getattr(params, 'confidence_fusion_threshold', 0.5)

        self.exponential_correspondences_to_map_kernel = exponential_correspondences_to_map_kernel(
            resolution=self.resolution, width=self.cell_n, height=self.cell_n, alpha=0.7,
            confidence_threshold=self.confidence_threshold,
        )

    def __call__(
        self,
        sem_map_idx,
        image,
        confidence,
        j,
        uv_correspondence,
        valid_correspondence,
        image_height,
        image_width,
        semantic_map,
        new_map,
        aux_layer_idx=None,  # unused; API compatibility with fusion_manager
    ):
        # Handle missing confidence: create all-ones array (no filtering)
        if confidence is None:
            confidence = cp.ones((1, int(image_height), int(image_width)), dtype=cp.float32)
        
        self.exponential_correspondences_to_map_kernel(
            semantic_map,
            sem_map_idx,
            image[j],
            confidence[j] if confidence.shape[0] > j else confidence[0],
            uv_correspondence,
            valid_correspondence,
            image_height,
            image_width,
            new_map,
            size=int(self.cell_n * self.cell_n),
        )
        semantic_map[sem_map_idx] = new_map[sem_map_idx]
