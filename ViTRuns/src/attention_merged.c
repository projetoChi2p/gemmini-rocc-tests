#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "profiler.h"



// ==========================================
// 2. UNIFIED ATTENTION MODULE
// ==========================================

void compute_attention(
    // Common Params
    int hidden_dim, 
    int num_heads, 
    int seq_len,
    int layer_idx,           // For profiling
    const elem_t * input, 
    elem_t * out,           // FP32: Now this is just the output of Wo (Linear)
    
    
    // Weights
    const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
    const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b, const acc_t * Wo_b,
    
    // Scratchpads
    elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
    elem_t * scores_buf, elem_t * probs_buf,
    elem_t * norm_buf, elem_t * context_buf,
    elem_t * wo_buf, elem_t * resadd_buf,

    // Scales (Quantized vs FP32)
    
    float scale_q, float scale_k, float scale_v, float scale_wo, 
    float score_scaling_softmax, // Input scale for Softmax
    float scale_scores_matmul,    // Output scale for MatMul

    float score_scaling_factor,   // Single float scaling
    float context_scaling_factor

    )
{

    // =========================================================================
    // QUANTIZED IMPLEMENTATION (Int8) - Self Attention Only
    // =========================================================================
    
    int head_dim = hidden_dim / num_heads;

    // Separate buffers for each stage to avoid aliasing
    elem_t * norm_out_ptr = norm_buf;
    
    uint64_t op_start, op_end;
    
    // --- 0. Pre-LayerNorm ---
    op_start = read_cycles();
    #ifdef QUANTIZED
        #ifdef CPU_LAYERNORM
            // Pre-LN inside attention: normalize input into out_buf workspace
            memcpy(norm_out_ptr, input, seq_len * hidden_dim * sizeof(elem_t));
            cpu_layernorm(seq_len, hidden_dim, norm_out_ptr);
        #else
            // HW LayerNorm: input -> out_buf
            tiled_norm_auto(
                seq_len,
                hidden_dim,
                (acc_t*)input,
                norm_out_ptr,
                ACC_SCALE_IDENTITY,
                LAYERNORM,
                WS
            );
        #endif
    #else
        // FP32: normalize input into out_buf
        tiled_norm_auto(seq_len, hidden_dim, 
            (acc_t*)input, 
            norm_out_ptr, 
            ACC_SCALE_IDENTITY, LAYERNORM, WS);
    #endif
    op_end = read_cycles();
    if (debug_inference && g_profiling_enabled) {
        g_profile.encoder.layers[layer_idx].attention.layernorm = op_end - op_start;
    }

    #ifdef DEBUG
        if (global_layer_index == 0 && debug_inference) {
            verify_tensor("Layer 0 Norm", norm_out_ptr, (elem_t*)debug_layer0_ln1, seq_len * hidden_dim, TOLERANCE);
            // exit(0);
        }
    #endif

    // --- 1. Compute Q, K, V Projections ---
    op_start = read_cycles();
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim,
        norm_out_ptr, Wq, Wq_b, Q_buf,
        hidden_dim, hidden_dim, hidden_dim, hidden_dim, 
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)scale_q, 0, true, false, false, false, false, 0, WS);
    
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim,
        norm_out_ptr, Wk, Wk_b, K_buf,
        hidden_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)scale_k, 0, true, false, false, false, false, 0, WS);

    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim,
        norm_out_ptr, Wv, Wv_b, V_buf,
        hidden_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)scale_v, 0, true, false, false, false, false, 0, WS);

    gemmini_fence();
    op_end = read_cycles();
    if (debug_inference && g_profiling_enabled) {
        g_profile.encoder.layers[layer_idx].attention.qkv_projections = op_end - op_start;
    }

    #ifdef DEBUG
    if (global_layer_index == 0 && debug_inference) {
        verify_tensor("Attn: Q Proj", Q_buf, (elem_t*)debug_layer0_q, seq_len * hidden_dim, TOLERANCE);
        verify_tensor("Attn: V Proj", V_buf, (elem_t*)debug_layer0_v, seq_len * hidden_dim, TOLERANCE);
    }
    #endif 

    // --- 2. Calculate Attention Scores (Q * K^T) ---
    op_start = read_cycles();
    for (int h = 0; h < num_heads; h++) {
        #ifdef QUANTIZED
            tiled_matmul_auto(seq_len, seq_len, head_dim,
                Q_buf + h*head_dim, 
                K_buf + h*head_dim, 
                NULL, 
                scores_buf + h*seq_len*seq_len, 
                hidden_dim, hidden_dim, 0, seq_len, 
                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                #ifdef CPU_SOFTMAX
                    NO_ACTIVATION, (acc_scale_t)SCORE_SCALING_FACTOR, 0, false, 
                #else
                    SOFTMAX, (acc_scale_t)SCORE_SCALING_FACTOR, (acc_scale_t)0.05, false, 
                #endif
                false, true, false, false, 0, WS);
                
        #else 
            tiled_matmul_auto(seq_len, seq_len, head_dim,
                Q_buf + h*head_dim, 
                K_buf + h*head_dim, 
                NULL, 
                scores_buf + h*seq_len*seq_len,
                hidden_dim, hidden_dim, 0, seq_len,
                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                NO_ACTIVATION, SCALE_DK, 0, false, 
                false, true, false, false, 0, WS);

        #endif
    }
    gemmini_fence();
    op_end = read_cycles();
    if (debug_inference && g_profiling_enabled) {
        g_profile.encoder.layers[layer_idx].attention.attention_scores = op_end - op_start;
    }

    #ifdef DEBUG
    #ifdef CPU_SOFTMAX
    if (global_layer_index == 0 && debug_inference) {
        verify_tensor("Attn: Scores (Head 0) - Q * K^T", scores_buf, (elem_t*)debug_layer0_scores_head0, seq_len * seq_len, TOLERANCE);
    }
    #endif
    #endif

    // --- 3. Softmax ---
    op_start = read_cycles();
    for (int h = 0; h < num_heads; h++) {
        elem_t * scores_matrix = scores_buf + h * seq_len * seq_len;
        elem_t * probs_matrix  = probs_buf + h * seq_len * seq_len;

        #ifdef QUANTIZED
            #ifdef CPU_SOFTMAX
                memcpy(probs_matrix, scores_matrix, seq_len * seq_len * sizeof(elem_t));
                cpu_softmax_quantized(seq_len, seq_len, probs_matrix, 1.0/8.0);
            #else
                // HW softmax already wrote into scores buffer when enabled; copy to probs buffer
                memcpy(probs_matrix, scores_matrix, seq_len * seq_len * sizeof(elem_t));
            #endif
        #else
            memcpy(probs_matrix, scores_matrix, seq_len * seq_len * sizeof(elem_t));
            cpu_softmax(seq_len, seq_len, probs_matrix);
            
        #endif
    }
    op_end = read_cycles();
    if (debug_inference && g_profiling_enabled) {
        g_profile.encoder.layers[layer_idx].attention.softmax = op_end - op_start;
    }

    #ifdef DEBUG
    if (global_layer_index == 0 && debug_inference) {
        verify_tensor("Attn: Probs (Head 0) - SOFTMAX", probs_buf, (elem_t*)debug_layer0_probs_head0, seq_len * seq_len, TOLERANCE);
        //exit(0);
    }
    #endif

    // --- 4. Context Aggregation (Probs * V) ---
    // Use provided scaling to prevent saturation in quantized mode
    op_start = read_cycles();
    float context_scale = context_scaling_factor; 
    #ifndef QUANTIZED
        context_scale = ACC_SCALE_IDENTITY;
    #endif
    
    for (int h = 0; h < num_heads; h++) {
        tiled_matmul_auto(
            seq_len,      // dim_I: How many rows we are calculating
            head_dim,     // dim_J: How many columns this head has
            seq_len,      // dim_K: The dot-product dimension

            // Pointer math finds the "Top Left" corner of the head's slice
            probs_buf + (h * seq_len * seq_len), 
            V_buf + (h * head_dim), 
            NULL, 
            context_buf + (h * head_dim), 

            seq_len,      // stride_A: Rows in Probs are seq_len apart
            hidden_dim,   // stride_B: Rows in V are hidden_dim apart! <--- FIX
            0, 
            hidden_dim,   // stride_C: Rows in Out are hidden_dim apart! <--- FIX
            
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, (acc_scale_t)context_scale, 0,
            false, false, false, false, false, 0, WS);
    }
    gemmini_fence();
    op_end = read_cycles();
    if (debug_inference && g_profiling_enabled) {
        g_profile.encoder.layers[layer_idx].attention.context_aggregation = op_end - op_start;
    }

    #ifdef DEBUG
    if (global_layer_index == 0 && debug_inference ) {
        verify_tensor("Attn: Context (Head 0)", context_buf, (elem_t*)debug_layer0_context_head0, seq_len * head_dim, TOLERANCE);
        //exit(0);
    }
    #endif

    acc_scale_t scale_temp = (acc_scale_t)scale_wo; // Adjust for HW/SW match

    // --- 5. Output Projection (Wo) ---
    op_start = read_cycles();
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim,
        context_buf, Wo, Wo_b, wo_buf,
        hidden_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, scale_temp, 0, 
        true, false, false, false, false, 0, WS);
    
    gemmini_fence();
    op_end = read_cycles();
    if (debug_inference && g_profiling_enabled) {
        g_profile.encoder.layers[layer_idx].attention.output_projection = op_end - op_start;
    }

    // Debug: Verify Raw Attention Output
        #ifdef DEBUG
        if (global_layer_index == 0 && debug_inference) {
            verify_tensor("Layer 0 Wo Proj", wo_buf, (elem_t*)debug_layer0_proj, seq_len * hidden_dim, TOLERANCE);
            //exit(0);
        }

        #endif

    // 2a. Residual Add
    op_start = read_cycles();
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        /*input A: */ wo_buf, 
        /*input B: */ input, 
        /*output B:*/ resadd_buf,
        false, WS);

    memcpy(out, resadd_buf, seq_len * hidden_dim * sizeof(elem_t));
    
    gemmini_fence();
    op_end = read_cycles();
    if (debug_inference && g_profiling_enabled) {
        g_profile.encoder.layers[layer_idx].attention.residual_add = op_end - op_start;
    }

    // Debug: Verify LN Output
    #ifdef DEBUG
        if (global_layer_index == 0 && debug_inference) {
            verify_tensor("Layer 0 Resadd 1", out, (elem_t*)debug_layer0_res1, seq_len * hidden_dim, TOLERANCE);
            //exit(1);    
        }
    #endif

}