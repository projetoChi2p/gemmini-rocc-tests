#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h> // for abs

// Include verification helper
#include "utils_quant.c"
uint64_t encoder_decoder_quantized(
        // Dimensions
        int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads,
        int seq_len, int num_layers,

        // Data Pointers
        const elem_t * input,    // [Seq, Hidden]
        const elem_t * enc_out,  // [Seq, Hidden] (NULL for Encoder)
        elem_t * out,            // [Seq, Hidden] Output Buffer

        // Weights (Self Attn)
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b, const acc_t * Wo_b,
        
        // Weights (Cross Attn)
        const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,
        const acc_t * Wq_cross_b, const acc_t * Wk_cross_b, const acc_t * Wv_cross_b, const acc_t * Wo_cross_b,

        // Weights (FFN)
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        // Scratchpad Buffers
        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, 
        elem_t * resadd1_buf, elem_t * resadd2_buf, 
        
        // Constants
        float score_scaling_factor, 

        // Scales
        const float * scales_q, const float * scales_k, const float * scales_v, const float * scales_wo,
        const float * scales_scores, const float * scales_ff1, const float * scales_ff2
    )
{
    const bool is_encoder = (enc_out == NULL);

    // --- STRIDE LOGIC (Standard ViT) ---
    int hidden_dim_compressed = hidden_dim; 

    size_t stride_W_attn = (size_t)hidden_dim * hidden_dim_compressed;
    size_t stride_b_attn = (size_t)hidden_dim_compressed; 
    
    size_t stride_Wo_attn = (size_t)hidden_dim_compressed * hidden_dim;
    size_t stride_Wo_b    = (size_t)hidden_dim;

    size_t stride_ff1_w = (size_t)hidden_dim * expansion_dim;
    size_t stride_ff2_w = (size_t)expansion_dim * hidden_dim;
    size_t stride_ff1_b = (size_t)expansion_dim;
    size_t stride_ff2_b = (size_t)hidden_dim;

    uint64_t start = read_cycles();
    const elem_t * layer_in = input;

    for (int l = 0; l < num_layers; l++) {
        global_layer_index = l;
        
        #ifdef DEBUG
        printf("\n=== Layer %d Start ===\n", l);
        #endif

        gemmini_fence(); 

        // --- 1. Self Attention ---
        attention_quantized(hidden_dim, num_heads, seq_len, 
            layer_in, 
            out_buf, // Result of Wo Projection
            Wq, Wk, Wv, Wo,
            Wq_b, Wk_b, Wv_b, Wo_b,
            Q_buf, K_buf, V_buf, attn_buf, resadd2_buf,
            scales_q[l], scales_k[l], scales_v[l], scales_wo[l], 
            score_scaling_factor, scales_scores[l]
        );

        #ifdef DEBUG
        if (l == 0) {
            verify_tensor("Layer 0 Q", Q_buf, (elem_t*)debug_layer0_q, seq_len * hidden_dim, TOLERANCE);
            verify_tensor("Layer 0 Wo Proj", out_buf, (elem_t*)debug_layer0_proj, seq_len * hidden_dim, TOLERANCE);
        }
        #endif

        gemmini_fence(); 

        // --- 2. Residual Add + LN ---
        tiled_resadd_auto(seq_len, hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
            out_buf, layer_in, resadd1_buf,
            false, WS);
            
        gemmini_fence();

        #ifdef DEBUG
        if (l == 0) {
            verify_tensor("Layer 0 ResAdd1", resadd1_buf, (elem_t*)debug_layer0_res1, seq_len * hidden_dim, TOLERANCE);
        }
        #endif

        cpu_layernorm_quantized(seq_len, hidden_dim, resadd1_buf);
        
        #ifdef DEBUG
        if (l == 0) {
            verify_tensor("Layer 0 LN1", resadd1_buf, (elem_t*)debug_layer0_ln1, seq_len * hidden_dim, TOLERANCE);
        }
        #endif

        // --- 3. FFN ---
        ffn_quantized(hidden_dim, expansion_dim, seq_len,
            resadd1_buf,
            out, // Final Result
            ff1_w, ff2_w,
            ff1_b, ff2_b,
            out_buf, // Scratchpad
            scales_ff1[l], scales_ff2[l]
        );

        #ifdef DEBUG
        if (l == 0) {
            // FIX: Check 'out', not 'out_buf'
            verify_tensor("Layer 0 Out", out, (elem_t*)debug_layer0_out, seq_len * hidden_dim, TOLERANCE);
        }
        #endif

        // --- 4. Advance Pointers ---
        Wq += stride_W_attn; Wk += stride_W_attn; Wv += stride_W_attn; 
        Wo += stride_Wo_attn; 
        
        Wq_b += stride_b_attn; Wk_b += stride_b_attn; Wv_b += stride_b_attn; 
        Wo_b += stride_Wo_b;   
        
        ff1_w += stride_ff1_w; ff2_w += stride_ff2_w; 
        ff1_b += stride_ff1_b; ff2_b += stride_ff2_b;

        // Update Input for Next Layer
        layer_in = out;

        gemmini_fence();
    }

    uint64_t end = read_cycles();
    return end - start;
}