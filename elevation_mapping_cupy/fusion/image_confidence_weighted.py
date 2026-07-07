#
# Confidence-weighted image fusion.
#
# Unlike image_exponential (fixed alpha + hard confidence gate), each map cell
# accumulates an evidence weight `w` in a companion semantic layer. Every
# observation fuses with an effective alpha of conf / (conf + w):
#   - a fresh cell (w = 0) takes the observation fully,
#   - a cell built from confident history resists low-confidence updates
#     proportionally (no threshold cliff),
#   - exponential forgetting (conf_decay) plus a weight cap (conf_weight_cap)
#     bound how entrenched a cell can get, so repeated confident observations
#     always win within ~cap frames after a scene change.
# This is a 1-D Kalman-style update with confidence as inverse variance and
# exponential forgetting.
#
import cupy as cp
import string

from .fusion_manager import FusionBase


def confidence_weighted_correspondences_to_map_kernel(
    resolution, width, height, conf_floor, conf_decay, weight_cap
):
    kernel = cp.ElementwiseKernel(
        in_params=(
            "raw U sem_map, raw U map_idx, raw U weight_idx, raw U image_mono, "
            "raw U confidence, raw U uv_correspondence, raw B valid_correspondence, "
            "raw U image_height, raw U image_width"
        ),
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
            int vi = get_map_idx(i, map_idx);
            int wi = get_map_idx(i, weight_idx);
            if (valid_correspondence[cell_idx]){
                int cell_idx_2 = get_map_idx(i, 1);
                int idx = int(uv_correspondence[cell_idx]) + int(uv_correspondence[cell_idx_2]) * image_width;

                float conf = (float)confidence[idx];
                if (conf >= ${conf_floor}) {
                    // Decayed accumulated evidence, then Kalman-style blend.
                    // Guard conf==0 && w==0 (possible when conf_floor is 0.0):
                    // 0/0 would poison the cell with NaN.
                    float w = (float)sem_map[wi] * ${conf_decay};
                    float denom = conf + w;
                    float a = (denom > 0.0f) ? (conf / denom) : 0.0f;
                    float v_old = (float)sem_map[vi];
                    new_sem_map[vi] = v_old + a * ((float)image_mono[idx] - v_old);
                    float w_new = w + conf;
                    if (w_new > ${weight_cap}) { w_new = ${weight_cap}; }
                    new_sem_map[wi] = w_new;
                } else {
                    // Below floor: keep value and weight untouched
                    new_sem_map[vi] = sem_map[vi];
                    new_sem_map[wi] = sem_map[wi];
                }
            }else{
                new_sem_map[vi] = sem_map[vi];
                new_sem_map[wi] = sem_map[wi];
            }
            """
        ).substitute(conf_floor=conf_floor, conf_decay=conf_decay, weight_cap=weight_cap),
        name="confidence_weighted_correspondences_to_map_kernel",
    )
    return kernel


class ImageConfidenceWeighted(FusionBase):
    def __init__(self, params, *args, **kwargs):
        self.name = "image_confidence_weighted"
        self.cell_n = params.cell_n
        self.resolution = params.resolution

        self.conf_floor = getattr(params, "conf_floor", 0.2)
        self.conf_decay = getattr(params, "conf_decay", 0.97)
        self.weight_cap = getattr(params, "conf_weight_cap", 4.0)

        self.kernel = confidence_weighted_correspondences_to_map_kernel(
            resolution=self.resolution,
            width=self.cell_n,
            height=self.cell_n,
            conf_floor=self.conf_floor,
            conf_decay=self.conf_decay,
            weight_cap=self.weight_cap,
        )
        self._warned_no_weight_layer = False

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
        aux_layer_idx=None,
    ):
        if aux_layer_idx is None:
            if not self._warned_no_weight_layer:
                print(
                    "[image_confidence_weighted] No weight layer index provided; "
                    "skipping fusion (semantic_map.update_layers_image should create it)."
                )
                self._warned_no_weight_layer = True
            return

        # Missing confidence: treat as fully confident (degrades to a decayed
        # running average, still no threshold cliff)
        if confidence is None:
            confidence = cp.ones((1, int(image_height), int(image_width)), dtype=cp.float32)

        self.kernel(
            semantic_map,
            sem_map_idx,
            cp.uint64(aux_layer_idx),
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
        semantic_map[aux_layer_idx] = new_map[aux_layer_idx]
