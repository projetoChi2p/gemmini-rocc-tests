#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// ==========================================
// 1. BAREMETAL MATH HELPERS
// ==========================================
// Optimized integer-based approximations to avoid linking heavy math libraries

static inline float my_exp_ffn(float x) {
    if (x <= -88.0f) return 0.0f;
    if (x >= 88.0f) x = 88.0f;
    union { float f; int32_t i; } converter;
    converter.i = (int32_t)(12102203.0f * x + 1064986824);
    return converter.f;
}

static inline float my_tanh_ffn(float x) {
    float abs_x = (x < 0.0f) ? -x : x;
    if (abs_x > 4.0f) return (x < 0.0f) ? -1.0f : 1.0f;
    float e2x = my_exp_ffn(2.0f * abs_x);
    float t = (e2x - 1.0f) / (e2x + 1.0f);
    return (x < 0.0f) ? -t : t;
}

static inline int my_round_ffn(float x) {
    return (int)(x + (x >= 0 ? 0.5f : -0.5f));
}

// ==========================================
// 2. CPU ACTIVATION & NORMALIZATION
// ==========================================

// GELU Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void cpu_gelu_quantized(int rows, int cols, elem_t * input, elem_t * output) {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEF = 0.044715f;
    
    // Heuristic: Input Int8 range maps to approx [-6.0, 6.0]
    const float range_max = 6.0f;
    const float input_scale = range_max / (float) elem_t_max ; 
    const float output_scale = (float) elem_t_max / range_max; 

    int size = rows * cols;
    for (int i = 0; i < size; i++) {
        float x = (float)input[i] * input_scale;
        
        // Approx calculation
        float inner = SQRT_2_OVER_PI * (x + COEF * x * x * x);
        float tanh_res = my_tanh_ffn(inner);
        float res = 0.5f * x * (1.0f + tanh_res);
        
        // Re-quantize
        int out_val = my_round_ffn(res * output_scale);
        if (out_val > elem_t_max) out_val = elem_t_max;
        if (out_val < elem_t_min) out_val = elem_t_min;
        
        output[i] = (elem_t)out_val;
    }
}

void cpu_layernorm_quantized(int rows, int cols, elem_t * data) {
    for(int i=0; i<rows; i++) {
        float sum=0, sq=0;
        
        // 1. Mean & Variance
        for(int j=0; j<cols; j++) {
            float v = (float)data[i*cols + j];
            sum += v; 
            sq += v*v;
        }
        float mean = sum/cols;
        float var_term = (sq/cols) - (mean*mean);
        if (var_term < 0) var_term = 0; // Safety clamp
        
        float std = sqrtf(var_term + 1e-5); 
        
        // 2. Normalize & Scale
        for(int j=0; j<cols; j++) {
            float v = (float)data[i*cols + j];
            float n = (v - mean)/std * 20.0f; 
            
            // Use explicit rounding helper
            int res = my_round_ffn(n);
            
            if(res > elem_t_max) res=elem_t_max; 
            if(res < elem_t_min) res=elem_t_min;
            data[i*cols+j] = (elem_t)res;
        }
    }
}

// ==========================================
// 3. FFN MODULE IMPLEMENTATION
// ==========================================

void ffn_quantized(
    int hidden_dim, 
    int expansion_dim, 
    int seq_len,
    const elem_t * input, 
    elem_t * out, // Final output
    const elem_t * ff1_w, const elem_t * ff2_w,
    const acc_t * ff1_b, const acc_t * ff2_b,
    elem_t * out_buf,       // Scratchpad for FC1 output
    // Scales
    float scale_ff1,
    float scale_ff2
    )
{
    // --- 1. FC1 (Linear) ---
    // Input: [Seq, Hidden] -> Output: [Seq, Expansion]
    tiled_matmul_auto(
        seq_len, expansion_dim, hidden_dim,
        input, ff1_w, ff1_b, out_buf,
        hidden_dim, expansion_dim, expansion_dim, expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 
        (acc_scale_t)scale_ff1, 
        0, 
        true,
        false, false, false, false, 0, WS
    );

    #ifdef DEBUG
    if (global_layer_index == 0 && debug_inference) {
    // FIX: Size is Seq * Expansion (not Hidden)
    verify_tensor("Layer 0 FC1", out_buf, (elem_t*)debug_layer0_fc1, seq_len * expansion_dim, TOLERANCE);
    }
    #endif

    gemmini_fence();

    // --- 2. GELU Activation ---
    cpu_gelu_quantized(seq_len, expansion_dim, out_buf, out_buf);

    
    #ifdef DEBUG
    if (global_layer_index == 0 && debug_inference) {
    // FIX: Size is Seq * Expansion
    verify_tensor("Layer 0 GELU", out_buf, (elem_t*)debug_layer0_gelu, seq_len * expansion_dim, TOLERANCE);
    }
    #endif


    // --- 3. FC2 (Linear) ---
    // Input: [Seq, Expansion] -> Output: [Seq, Hidden]
    tiled_matmul_auto(
        seq_len, hidden_dim, expansion_dim,
        out_buf, ff2_w, ff2_b, out,
        expansion_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 
        (acc_scale_t)scale_ff2, 
        0, 
        true,
        false, false, false, false, 0, WS
    );

    #ifdef DEBUG
    if (global_layer_index == 0 && debug_inference) {
    // Size is Seq * Hidden (Correct)
    verify_tensor("Layer 0 FC2", out, (elem_t*)debug_layer0_fc2, seq_len * hidden_dim, TOLERANCE);
    }
    #endif

    gemmini_fence();

    // --- 4. Residual Add ---
    // Output = Output + Input (Skip Connection)
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        out, input, out,
        false, WS);

    gemmini_fence();

    // --- 5. LayerNorm ---
    // Post-Norm: Output = LN(Output)
    #ifdef CPU_LAYERNORM
        cpu_layernorm_quantized(seq_len, hidden_dim, out);
    #else
        tiled_norm_auto(
            seq_len, hidden_dim, 
            (acc_t*)out,    // Input (Accumulator/Int32)
            (elem_t*)out,  // Output (Int8)
            ACC_SCALE_IDENTITY,
            LAYERNORM, WS
        );
    #endif
}