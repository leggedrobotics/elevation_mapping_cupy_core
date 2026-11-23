#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string
from typing import List

from .plugin_manager import PluginBase


class MinFilter(PluginBase):
    """This is a filter to fill in invalid cells with minimum values around.

    Args:
        cell_n (int): width of the elevation map.
        dilation_size (int): The size of the patch to search for minimum value for each iteration.
        iteration_n (int): The number of iteration to repeat the same filter.
        **kwargs ():
    """

    def __init__(self, cell_n: int = 100, dilation_size: int = 5, iteration_n: int = 5, **kwargs):
        super().__init__()
        self.iteration_n = iteration_n
        self.width = cell_n
        self.height = cell_n
        self.min_filtered = cp.zeros((self.width, self.height))
        self.min_filtered_mask = cp.zeros((self.width, self.height))
        # Each pass reads the previous pass's buffers and writes the other pair, so a pass
        # spreads exactly dilation_size cells whatever the block scheduling. unfilled[it] is
        # set when pass `it` leaves a cell invalid; once a pass leaves none, the later passes
        # only copy (the old early exit, decided on the GPU instead of a host read-back).
        self.min_filter_kernel = cp.ElementwiseKernel(
            in_params="raw U mask, raw U map, raw U map_mask, int32 it",
            out_params="raw U newmap, raw U newmask, raw int32 unfilled",
            preamble=string.Template(
                """
                __device__ int get_map_idx(int idx, int layer_n) {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }

                __device__ int get_relative_map_idx(int idx, int dx, int dy, int layer_n) {
                    const int layer = ${width} * ${height};
                    const int relative_idx = idx + ${width} * dy + dx;
                    return layer * layer_n + relative_idx;
                }
                __device__ bool is_inside(int idx) {
                    int idx_x = idx / ${width};
                    int idx_y = idx % ${width};
                    if (idx_x <= 0 || idx_x >= ${width} - 1) {
                        return false;
                    }
                    if (idx_y <= 0 || idx_y >= ${height} - 1) {
                        return false;
                    }
                    return true;
                }
                """
            ).substitute(width=self.width, height=self.height),
            operation=string.Template(
                """
                U h = map[get_map_idx(i, 0)];
                U m = map_mask[get_map_idx(i, 0)];
                U valid = mask[get_map_idx(i, 0)];
                if (valid < 0.5 && (it == 0 || unfilled[it - 1])) {
                    U min_value = 1000000.0;
                    for (int dy = -${dilation_size}; dy <= ${dilation_size}; dy++) {
                        for (int dx = -${dilation_size}; dx <= ${dilation_size}; dx++) {
                            int idx = get_relative_map_idx(i, dx, dy, 0);
                            if (!is_inside(idx)) {continue;}
                            U valid = map_mask[idx];
                            U value = map[idx];
                            if(valid > 0.5 && value < min_value) {
                                min_value = value;
                            }
                        }
                    }
                    if (min_value < 1000000 - 1) {
                        h = min_value;
                        m = 0.6;
                    }
                }
                newmap[get_map_idx(i, 0)] = h;
                newmask[get_map_idx(i, 0)] = m;
                if (m < 0.5) {
                    unfilled[it] = 1;
                }
                """
            ).substitute(dilation_size=dilation_size),
            name="min_filter_kernel",
        )

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
        m, k = elevation_map[0].copy(), elevation_map[2].copy()
        next_m, next_k = cp.empty_like(m), cp.empty_like(k)
        unfilled = cp.zeros(max(self.iteration_n, 1), dtype=cp.int32)
        for it in range(self.iteration_n):
            self.min_filter_kernel(
                elevation_map[2], m, k, cp.int32(it), next_m, next_k, unfilled, size=(self.width * self.height)
            )
            m, k, next_m, next_k = next_m, next_k, m, k
        self.min_filtered, self.min_filtered_mask = m, k
        min_filtered = cp.where(self.min_filtered_mask > 0.5, self.min_filtered.copy(), cp.nan)
        return min_filtered
