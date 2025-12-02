// File: transformer_layers_debug.c (FFN Update)

//#include "../inc/ffn.h"
#include "include/gemmini_nn.h"
#include "../inc/utils.h" 

// ==========================================
// 1. HELPER FUNCTIONS (From your update)
// ==========================================

// --- Fast Tanh (using Schraudolph Exp) ---
// tanh(x) = (e^2x - 1) / (e^2x + 1)
static inline float fast_tanh(float x) {
    float abs_x = (x < 0.0f) ? -x : x;
    
    // Saturation check: tanh(x) is approx 1.0 for x > 4.0
    if (abs_x > 4.0f) return (x < 0.0f) ? -1.0f : 1.0f;

    float e2x = fast_exp_schraudolph(2.0f * abs_x);
    float t = (e2x - 1.0f) / (e2x + 1.0f);
    
    return (x < 0.0f) ? -t : t;
}

// --- BERT GELU Approximation ---
// Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void cpu_gelu_tanh(int rows, int cols, elem_t * input, elem_t * output) {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEF = 0.044715f;

    for (int i = 0; i < rows * cols; i++) {
        float x = (float)input[i];
        
        // Inner term: 0.79788 * (x + 0.044715 * x^3)
        float inner = SQRT_2_OVER_PI * (x + COEF * x * x * x);
        
        float tanh_res = fast_tanh(inner);
        
        output[i] = (elem_t)(0.5f * x * (1.0f + tanh_res));
    }
}

// ==========================================
// 2. FFN MODULE
// ==========================================

void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,
        elem_t * out_buf, acc_t * out_buf_acc
        
        #ifdef DEBUG
        // Debug Pointers
        ,const elem_t * expected_fc1,
        const elem_t * expected_fc2
        #endif
    )
{
    // --- 1. FC1 (Linear Only) ---
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        input, ff1_w, ff1_b, out_buf,
        hidden_dim, expansion_dim, expansion_dim, expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 
        ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        true, false, false, false, false, 0, WS);

    gemmini_fence();

    // --- 1.5 GELU Activation (CPU) ---
    cpu_gelu_tanh(seq_len, expansion_dim, out_buf, out_buf);

    // DEBUG: Check FC1 + GELU
    #ifdef DEBUG
    if (expected_fc1 != NULL) print_error_histogram("L0 FFN FC1+GELU", seq_len, expansion_dim, out_buf, expected_fc1);
    #endif

    // --- 2. FC2 ---
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
        out_buf, ff2_w, ff2_b, out_buf_acc,
        expansion_dim, hidden_dim, hidden_dim, hidden_dim, 
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, 
        false, 
        false, 0, WS);

    gemmini_fence();

    // DEBUG: Check FC2
    // Note: Since FC2 output is in acc_t* (out_buf_acc) but print_error_histogram usually expects elem_t*,
    // you might need to cast or normalize. Assuming the debugger handles this or we check the normalized version later.
    // However, usually we check the output *after* normalization for stability, or we can check the accumulator here 
    // if the debugger supports it. Assuming standard elem_t check:
    #ifdef DEBUG
    if (expected_fc2 != NULL) print_error_histogram("L0 FFN FC2", seq_len, hidden_dim, (elem_t*)out_buf_acc, expected_fc2);
    #endif

    // --- 3. Norm ---
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, 
        out_buf,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    gemmini_fence();

    // --- 4. Residual Add ---
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        out_buf, input, out,
        false, WS);

    gemmini_fence();
}