//#include "classifier.h"
#include "include/gemmini.h"
#include <stdbool.h>

void compute_classification_head(
    int seq_len, int hidden_dim, int output_size,
    const elem_t * encoder_output,
    const elem_t * pool_vector,
    const elem_t * head_w,
    const acc_t * head_b,
    elem_t * pool_out_buf,
    elem_t * ln_out_buf,
    elem_t * final_logits)
{
    // 1. Average Pooling (Weighted Sum)
    acc_scale_t pool_scale = (acc_scale_t)(1.0f / seq_len);
    
    tiled_matmul_auto(1, hidden_dim, seq_len,
        (const elem_t *)pool_vector, encoder_output,
        NULL, pool_out_buf,
        seq_len, hidden_dim, 0, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, pool_scale, 0,
        false, false, false, false, false, 0, WS);
    
    // 2. Layer Normalization
    tiled_norm_auto(1, hidden_dim,
        (acc_t*)pool_out_buf, ln_out_buf,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);
        
    // 3. Final Linear Projection (Logits)
    tiled_matmul_auto(1, output_size, hidden_dim,
        ln_out_buf, head_w, head_b, final_logits,
        hidden_dim, output_size, output_size, output_size,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, false, false, 0, WS);
}