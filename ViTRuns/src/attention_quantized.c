#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>


#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// Include your verification helper
// (Ensure utils_quant.c has the definition of verify_tensor)
#include "utils_quant.c" 

// ==========================================
// 1. BAREMETAL MATH HELPERS
// ==========================================
static inline float my_exp_att(float x) {
    if (x <= -88.0f) return 0.0f;
    if (x >= 88.0f) x = 88.0f;
    union { float f; int32_t i; } converter;
    converter.i = (int32_t)(12102203.0f * x + 1064986824);
    return converter.f;
}

// ==========================================
// 2. CPU SOFTMAX
// ==========================================
void cpu_softmax_quantized(int rows, int cols, elem_t * matrix, float input_scale) {
    for (int i = 0; i < rows; i++) {
        float row_buf[cols]; 
        float max_val = -3.40282e+38F;

        for (int j = 0; j < cols; j++) {
            float val = (float)matrix[i * cols + j] * input_scale;
            row_buf[j] = val;
            if (val > max_val) max_val = val;
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            float exp_val = my_exp_att(row_buf[j] - max_val);
            row_buf[j] = exp_val;
            sum_exp += exp_val;
        }

        float inv_sum = 127.0f / (sum_exp + 1e-6f); 
        for (int j = 0; j < cols; j++) {
            int quant_val = (int)(row_buf[j] * inv_sum);
            if (quant_val > 127) quant_val = 127;
            if (quant_val < 0) quant_val = 0;
            matrix[i * cols + j] = (elem_t)quant_val;
        }
    }
}

// ==========================================
// 3. ATTENTION MODULE
// ==========================================
void attention_quantized(
        int hidden_dim, 
        int num_heads, 
        int seq_len,
        const elem_t * input, 
        elem_t * out, 
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b, const acc_t * Wo_b,
        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf,
        float scale_q, float scale_k, float scale_v, float scale_wo, 
        float score_scaling, 
        float scale_scores_matmul 
    )
{
    int head_dim = hidden_dim / num_heads;

    // --- 1. Compute Q, K, V Projections ---
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim,
        input, Wq, Wq_b, Q_buf,
        hidden_dim, hidden_dim, hidden_dim, hidden_dim, 
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)scale_q, 0, true, false, false, false, false, 0, WS);
    
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim,
        input, Wk, Wk_b, K_buf,
        hidden_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)scale_k, 0, true, false, false, false, false, 0, WS);

    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim,
        input, Wv, Wv_b, V_buf,
        hidden_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)scale_v, 0, true, false, false, false, false, 0, WS);

    gemmini_fence();

    #ifdef DEBUG
    if (global_layer_index == 0 && debug_inference) {
        // [DEBUG] Verify Q, K, V (Assuming Layer 0 arrays are available)
        // Note: This verifies ONLY if we are in Layer 0. 
        // Ideally, pass a 'layer_idx' arg, or we rely on the caller to enable/disable via DEBUG macro
        // Since we don't have layer_idx here, we'll verify against Layer 0 arrays blindly if DEBUG is on.
        // This assumes this function is called for Layer 0 first.
        verify_tensor("Attn: Q Proj", Q_buf, (elem_t*)debug_layer0_q, seq_len * hidden_dim, TOLERANCE);
        verify_tensor("Attn: V Proj", V_buf, (elem_t*)debug_layer0_v, seq_len * hidden_dim, TOLERANCE);
    }
    #endif 

    // --- 2. Calculate Attention Scores (Q * K^T) ---
    for (int h = 0; h < num_heads; h++) {
        tiled_matmul_auto(seq_len, seq_len, head_dim,
            Q_buf + h*head_dim, 
            K_buf + h*head_dim, 
            NULL, 
            attn_buf + h*seq_len*seq_len, 
            hidden_dim, hidden_dim, seq_len, seq_len, 
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            #ifdef CPU_SOFTMAX
                NO_ACTIVATION, (acc_scale_t)scale_scores_matmul, 0, false, 
            #else
                SOFTMAX, (acc_scale_t)scale_scores_matmul, (acc_scale_t)0.05, false, 
            #endif
            false, true, false, false, 0, WS);
    }
    gemmini_fence();

    #ifdef DEBUG
    if (global_layer_index == 0 && debug_inference) {
        // [DEBUG] Verify Head 0 Scores
        //verify_tensor("Attn: Scores (Head 0)", attn_buf, (elem_t*)debug_layer0_scores_head0, seq_len * seq_len, TOLERANCE);
    }
    #endif 

    // --- 3. Softmax ---
    for (int h = 0; h < num_heads; h++) {
        #ifdef CPU_SOFTMAX
            cpu_softmax_quantized(seq_len, seq_len, attn_buf + h*seq_len*seq_len, score_scaling);
        #endif
    }

    #ifdef DEBUG
    if (global_layer_index == 0 && debug_inference) {
        // [DEBUG] Verify Head 0 Probs
        verify_tensor("Attn: Probs (Head 0)", attn_buf, (elem_t*)debug_layer0_probs_head0, seq_len * seq_len, TOLERANCE);
    }
    #endif 

    // --- 4. Context Aggregation (Probs * V) ---
    float context_scale = CONTEXT_SCALE; // Adjusted for HW/SW match
    
    for (int h = 0; h < num_heads; h++) {
        tiled_matmul_auto(seq_len, head_dim, seq_len,
            attn_buf + h*seq_len*seq_len, 
            V_buf + h*head_dim, 
            NULL, 
            out_buf + h*head_dim, 
            seq_len, hidden_dim, 0, hidden_dim, 
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, (acc_scale_t)context_scale, 0, false, 
            false, false, false, false, 0, WS);
    }
    gemmini_fence();

    #ifdef DEBUG
    if (global_layer_index == 0 && debug_inference) {
        // [DEBUG] Verify Head 0 Context
        // Note: out_buf holds interleaved heads [Seq, Hidden]. Head 0 is at offset 0, 64, 128...
        // The debug tensor is packed [Seq, HeadDim]. Stride verification needed.
        // For simplicity, we create a temporary packed buffer for verification
        elem_t temp_head0[seq_len * head_dim];
        for(int r=0; r<seq_len; r++) {
            for(int c=0; c<head_dim; c++) {
                temp_head0[r*head_dim + c] = out_buf[r*hidden_dim + c];
            }
        }
        verify_tensor("Attn: Context (Head 0)", temp_head0, (elem_t*)debug_layer0_context_head0, seq_len * head_dim, TOLERANCE);
    }
    #endif 

    // --- 5. Output Projection (Wo) ---
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim,
        out_buf, Wo, Wo_b, out,
        hidden_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)scale_wo, 0, 
        true, false, false, false, false, 0, WS);
        
    gemmini_fence();
}