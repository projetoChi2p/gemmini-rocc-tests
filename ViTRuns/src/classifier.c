#include "../include/gemmini.h"

void compute_classification_head(
    int hidden_dim, 
    int output_size,
    const elem_t * cls_token_output, // Pointer to the 0-th row of Encoder Output
    const elem_t * head_w,
    const acc_t * head_b,
    elem_t * ln_out,
    elem_t * final_logits)
{
    // 1. LayerNorm on CLS Token
    // Input is just 1 row (1xHidden)
    tiled_norm_auto(1, hidden_dim,
        (acc_t*)cls_token_output, ln_out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    // 2. Final Linear Projection
    // (1, Hidden) @ (Hidden, Output) -> (1, Output)
    tiled_matmul_auto(1, output_size, hidden_dim,
        ln_out, head_w, head_b, final_logits,
        hidden_dim, output_size, output_size, output_size,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, false, false, 0, WS);
}