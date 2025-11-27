//#include "../inc/ffn.h"
#include "include/gemmini.h"
#include "../inc/utils.h" // For cpu_gelu_approx if you uncomment it

// --- 3. Fast Tanh (using Schraudolph Exp) ---
// tanh(x) = (e^2x - 1) / (e^2x + 1)
static inline float fast_tanh(float x) {
    float abs_x = (x < 0.0f) ? -x : x;
    
    // Saturation check: tanh(x) is approx 1.0 for x > 4.0
    if (abs_x > 4.0f) return (x < 0.0f) ? -1.0f : 1.0f;

    float e2x = fast_exp_schraudolph(2.0f * abs_x);
    float t = (e2x - 1.0f) / (e2x + 1.0f);
    
    return (x < 0.0f) ? -t : t;
}

// --- 4. BERT GELU Approximation ---
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
        NO_ACTIVATION, 
        ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        true, false, false, false, false, 0, WS);

    gemmini_fence();

    cpu_gelu_tanh(seq_len, expansion_dim, out_buf, out_buf);

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