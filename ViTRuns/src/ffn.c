//#include "../inc/ffn.h"
#include "include/gemmini.h"
#include "../inc/utils.h" // For cpu_gelu_approx if you uncomment it

void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,
        elem_t * out_buf, acc_t * out_buf_acc)
{
    // 1. FC1 (Linear + ReLU)
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        input, ff1_w, ff1_b, out_buf,
        hidden_dim, expansion_dim, expansion_dim, expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        RELU, 
        ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        true, false, false, false, false, 0, WS);

    gemmini_fence();

    // 2. FC2
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
        out_buf, ff2_w, ff2_b, out_buf_acc,
        expansion_dim, hidden_dim, hidden_dim, hidden_dim, 
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, 
        false, 
        false, 0, WS);

    gemmini_fence();

    // 3. Norm
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, 
        out_buf,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    gemmini_fence();

    // 4. Residual Add
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        out_buf, input, out,
        false, WS);

    gemmini_fence();
}