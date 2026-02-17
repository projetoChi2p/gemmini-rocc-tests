#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "profiler.h"

// ==========================================
// 1. HELPER FUNCTIONS (Mode Dependent)
// ==========================================

#ifdef QUANTIZED
    // --- QUANTIZED HELPERS (Int8) ---
    // Optimized integer-based approximations
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

    void cpu_gelu_compute(int rows, int cols, elem_t * input, elem_t * output) {
        const float SQRT_2_OVER_PI = 0.7978845608f;
        const float COEF = 0.044715f;
        
        // Heuristic: Input Int8 range maps to approx [-6.0, 6.0]
        const float range_max = RANGE_GELU_OUT;
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

    void cpu_layernorm_compute(int rows, int cols, elem_t * data) {
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
                float n = (v - mean)/std * (float)RANGE_LN_OUT; 
                
                // Use explicit rounding helper
                int res = my_round_ffn(n);
                
                if(res > elem_t_max) res=elem_t_max; 
                if(res < elem_t_min) res=elem_t_min;
                data[i*cols+j] = (elem_t)res;
            }
        }
    }

#else
    // --- FP32 HELPERS ---
    // #include "../inc/utils.h" // Ensure this is linked or included in main.c

    static inline float fast_tanh(float x) {
        float abs_x = (x < 0.0f) ? -x : x;
        if (abs_x > 4.0f) return (x < 0.0f) ? -1.0f : 1.0f;

        // Assuming fast_exp_schraudolph is defined in the shared scope (utils.c)
        float e2x = fast_exp_approx(2.0f * abs_x);
        float t = (e2x - 1.0f) / (e2x + 1.0f);
        return (x < 0.0f) ? -t : t;
    }

    void cpu_gelu_compute(int rows, int cols, elem_t * input, elem_t * output) {
        const float SQRT_2_OVER_PI = 0.7978845608f;
        const float COEF = 0.044715f;

        for (int i = 0; i < rows * cols; i++) {
            float x = (float)input[i];
            float inner = SQRT_2_OVER_PI * (x + COEF * x * x * x);
            float tanh_res = fast_tanh(inner);
            output[i] = (elem_t)(0.5f * x * (1.0f + tanh_res));
        }
    }
    
    // Fallback for Norm in FP32 if Tiled Norm expects Acc input but we give Elem
    void cpu_layernorm_compute(int rows, int cols, elem_t * data) {
        for(int i=0; i<rows; i++) {
            float sum=0, sq=0;
            for(int j=0; j<cols; j++) {
                float v = data[i*cols + j];
                sum += v; sq += v*v;
            }
            float mean = sum/cols;
            float std = sqrtf((sq/cols) - mean*mean + 1e-5);
            for(int j=0; j<cols; j++) {
                data[i*cols + j] = (data[i*cols + j] - mean) / std;
            }
        }
    }
#endif

// ==========================================
// 2. UNIFIED FFN MODULE
// ==========================================

void compute_ffn(
    int hidden_dim, int expansion_dim, int seq_len,
    int layer_idx,           // For profiling
    const elem_t * input, 
    elem_t * out, // Final output buffer
    
    // Weights
    const elem_t * ff1_w, const elem_t * ff2_w,
    const acc_t * ff1_b, const acc_t * ff2_b,
    
    // Scratchpads
    elem_t * norm_buf,      // Pre-LN output
    elem_t * fc1_buf,       // FC1 output
    elem_t * gelu_buf,      // GELU output
    elem_t * fc2_buf,       // FC2 output
    elem_t * resadd_buf,    // Residual output
    acc_t * out_buf_acc,    // Intermediate (FC2 Acc - used in legacy FP32, now largely unused in unified post-norm)

    float scale_ff1,
    float scale_ff2
    )
{
    uint64_t op_start, op_end;
    
    // --- 0. Pre-LayerNorm (match PyTorch: norm inside FFN)
    // Normalize input into norm_buf, then use it as FC1 input.
    op_start = read_cycles();
    #ifdef QUANTIZED
        #ifdef CPU_LAYERNORM
            memcpy(norm_buf, input, seq_len * hidden_dim * sizeof(elem_t));
            cpu_layernorm_compute(seq_len, hidden_dim, norm_buf);
        #else
            tiled_norm_auto(
                seq_len, hidden_dim,
                (acc_t*)input,
                norm_buf,
                ACC_SCALE_IDENTITY,
                LAYERNORM,
                WS
            );
        #endif
    #else
        tiled_norm_auto(
            seq_len, hidden_dim,
            (acc_t*)input,
            norm_buf,
            ACC_SCALE_IDENTITY,
            LAYERNORM,
            WS
        );
    #endif
    op_end = read_cycles();
    if (debug_inference && g_profiling_enabled) {
        g_profile.encoder.layers[layer_idx].ffn.layernorm = op_end - op_start;
    }

    #ifdef DEBUG
        if (global_layer_index == 0 && debug_inference) {
            verify_tensor("Layer 0 FFN Norm", norm_buf, (elem_t*)debug_layer0_ln2, seq_len * hidden_dim, TOLERANCE);
            // exit(0);
        }
    #endif

    // --- 1. FC1 (Linear) ---
    // Input: [Seq, Hidden] -> Output: [Seq, Expansion]
    op_start = read_cycles();
    tiled_matmul_auto(
        seq_len, expansion_dim, hidden_dim,
        norm_buf, ff1_w, ff1_b, fc1_buf,
        hidden_dim, expansion_dim, expansion_dim, expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        #ifdef REPLACE_RELU
            RELU, 
        #else
            NO_ACTIVATION,
        #endif
        scale_ff1, 
        0, 
        true, false, false, false, false, 0, WS
    );

    #ifdef DEBUG
    #ifndef REPLACE_RELU
        if (global_layer_index == 0 && debug_inference) {
            verify_tensor("Layer 0 FC1", fc1_buf, (elem_t*)debug_layer0_fc1, seq_len * expansion_dim, TOLERANCE);
        }
    #endif
    #endif

    gemmini_fence();
    op_end = read_cycles();
    if (debug_inference && g_profiling_enabled) {
        g_profile.encoder.layers[layer_idx].ffn.fc1 = op_end - op_start;
    }

    // --- 2. GELU Activation ---
    op_start = read_cycles();
    #ifndef REPLACE_RELU
        cpu_gelu_compute(seq_len, expansion_dim, fc1_buf, gelu_buf);
    #endif
    op_end = read_cycles();
    if (debug_inference && g_profiling_enabled) {
        g_profile.encoder.layers[layer_idx].ffn.gelu = op_end - op_start;
    }

    #ifdef DEBUG
        if (global_layer_index == 0 && debug_inference) {
            verify_tensor("Layer 0 GELU", gelu_buf, (elem_t*)debug_layer0_gelu, seq_len * expansion_dim, TOLERANCE);
        }
    #endif


    // --- 3. FC2 (Linear) ---
    // Input: [Seq, Expansion] -> Output: [Seq, Hidden]
    // Unification Note: Both modes now write directly to 'out' (elem_t) to support the Residual->Norm flow.
    // Legacy FP32 used 'out_buf_acc' here, but that requires Norm->Residual topology.
    
    op_start = read_cycles();
    tiled_matmul_auto(
        seq_len, hidden_dim, expansion_dim,
        gelu_buf, ff2_w, ff2_b, fc2_buf,
        expansion_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 
        scale_ff2, 
        0, 
        true, false, false, 
        false, // Do not accumulate (write to elem_t)
        false, 0, WS
    );

    #ifdef DEBUG
        if (global_layer_index == 0 && debug_inference) {
            verify_tensor("Layer 0 FC2", fc2_buf, (elem_t*)debug_layer0_fc2, seq_len * hidden_dim, TOLERANCE);
        }
    #endif

    gemmini_fence();
    op_end = read_cycles();
    if (debug_inference && g_profiling_enabled) {
        g_profile.encoder.layers[layer_idx].ffn.fc2 = op_end - op_start;
    }

    // --- 4. Residual Add ---
    // Topology: Pre-LN within FFN; residual after FC2
    op_start = read_cycles();
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        fc2_buf, input, resadd_buf,
        false, WS);

    // copy final result back to out
    memcpy(out, resadd_buf, seq_len * hidden_dim * sizeof(elem_t));
    
    gemmini_fence();
    op_end = read_cycles();
    if (debug_inference && g_profiling_enabled) {
        g_profile.encoder.layers[layer_idx].ffn.residual_add = op_end - op_start;
    }
}