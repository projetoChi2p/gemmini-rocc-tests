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
    float context_scaling_factor,
    float scale_act_ln1,
    float scale_act_res1

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
            
            // NEW: Pass the weights and the output scale
            cpu_layernorm(seq_len, hidden_dim, norm_out_ptr, 
                          scale_act_ln1);
        #else
            // HW LayerNorm: input -> out_buf
            acc_t * promoted_input = (acc_t*)resadd_buf; // Borrow an unused 32-bit buffer temporarily
            for(int i = 0; i < seq_len * hidden_dim; i++) {
                promoted_input[i] = (acc_t)input[i];
            }
            tiled_norm_auto(
                seq_len,
                hidden_dim,
                promoted_input, // Pass the physically promoted array
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
        verify_tensor("Attn: K Proj", K_buf, (elem_t*)debug_layer0_k, seq_len * hidden_dim, TOLERANCE);
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
                    NO_ACTIVATION, (acc_scale_t)scale_scores_matmul*SCALE_DK, 0, false, 
                #else
                    SOFTMAX, (acc_scale_t)scale_scores_matmul*SCALE_DK, (acc_scale_t)0.05, false, 
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
    // exit(0);
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
                // Dequantize scores_int with their FP scale (score_scaling_softmax = s["score"]),
                // NOT with the matmul acc_scale.  These are different values.
                cpu_softmax_fp32_hybrid(seq_len, seq_len, probs_matrix, scale_scores_matmul/SCALE_DK);
                
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
        display_tensor_distribution_histogram("Attn: Probs (Head 0) - SOFTMAX", probs_buf, seq_len * seq_len);
        display_tensor_distribution_histogram("Attn: Probs (Head 0) - SOFTMAX (Expected)", debug_layer0_probs_head0, seq_len * seq_len);
        // print_tensor(probs_buf, seq_len, seq_len, "Attn: Probs (Head 0) - SOFTMAX");
        // exit(0);
    }
    #endif

    // --- 4. Context Aggregation (Probs * V) ---
    // IMPORTANT: probs_buf holds INTEGER values in [0, Q_MAX=127], NOT floats in [0,1].
    // The Gemmini acc_scale must therefore be:
    //   context_scale = s["v"] / (Q_MAX * s["context"])
    //                 = scales_v / (127 * scales_context_output)
    // If scales_context[l] in your param file is the raw activation scale s["context"],
    // you are missing the 1/Q_MAX factor and the s["v"] numerator.
    op_start = read_cycles();
    float context_scale = context_scaling_factor; 
    #ifndef QUANTIZED
        context_scale = ACC_SCALE_IDENTITY;
    #endif
    
    for (int h = 0; h < num_heads; h++) {
        tiled_matmul_auto(
            seq_len,      // dim_I: Rows (tokens)
            head_dim,     // dim_J: Columns per head (64)
            seq_len,      // dim_K: Reduction dim (the probabilities)

            // Matrix A: Probs (This is head-specific, often contiguous if h is outer)
            probs_buf + (h * seq_len * seq_len), 
            
            // Matrix B: V_buf (Offset by head_dim, but jump by hidden_dim to next row)
            V_buf + (h * head_dim), 
            
            NULL, 
            
            // Matrix C: Output (Write into the specific head slot of the interleaved buffer)
            context_buf + (h * head_dim), 

            seq_len,      // stride_A: Distance between rows in probs
            hidden_dim,   // stride_B: Distance between rows in V (Crucial!)
            0, 
            hidden_dim,   // stride_C: Distance between rows in context_buf
            
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
        // 1. Create a contiguous temporary buffer for Head 0
        elem_t debug_contiguous_head0[seq_len * head_dim];
        
        // 2. Extract Head 0 from the interleaved context_buf
        for (int i = 0; i < seq_len; i++) {
            memcpy(&debug_contiguous_head0[i * head_dim], 
                   &context_buf[i * hidden_dim], // Jump by hidden_dim to stay on Head 0
                   head_dim * sizeof(elem_t));
        }

        // 3. Verify the contiguous buffer
        verify_tensor("Attn: Context (Head 0)", debug_contiguous_head0, 
                      (elem_t*)debug_layer0_context_head0, seq_len * head_dim, TOLERANCE);

        display_tensor_distribution_histogram("Attn: Context (Head 0)", debug_contiguous_head0, seq_len * head_dim);
        // display_tensor_distribution_histogram("Attn: Context (Head 0) Expected", debug_layer0_context_head0, seq_len * head_dim);

        // exit(0);
    }
    #endif

    // Pass M_wo to the hardware, NOT scale_wo
    acc_scale_t scale_temp = (acc_scale_t)scale_wo ; // Combine the output scaling with the context scaling to maintain HW/SW match
    // tiled_matmul_auto(..., scale_temp, ...);
    // acc_scale_t scale_temp = (acc_scale_t)scale_wo; // Adjust for HW/SW match

    // printf("Debug:  scale_wo = %d.%06d\n", (int)(scale_wo), (int)((scale_wo - (int)scale_wo) * 1000000)); // Print scale_wo for debugging
        
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
            display_tensor_distribution_histogram("Layer 0 Wo Proj Distribution", wo_buf, seq_len * hidden_dim);
            display_tensor_distribution_histogram("Layer 0 Wo Proj Distribution (expected)", debug_layer0_proj, seq_len * hidden_dim);
            // exit(0);
        }

        #endif

    float residual_input_scale;
    if (layer_idx == 0) {
        // Layer 0 input comes from the Embedding block
        residual_input_scale = scales_act_embed[layer_idx] / scales_act_res1[layer_idx]; 
    } else {
        // Layer 1+ input comes from the previous layer's ResAdd 2
        residual_input_scale = scales_act_res2[layer_idx - 1] / scales_act_res1[layer_idx];
    }

    // 2a. Residual Add
    op_start = read_cycles();
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,               // Scale A: wo_buf (Already in target scale)
        (acc_scale_t)residual_input_scale, // Scale B: Align the residual input
        ACC_SCALE_IDENTITY,                // Output scale (1.0, just write it)
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
            // exit(1);    
        }
    #endif

}