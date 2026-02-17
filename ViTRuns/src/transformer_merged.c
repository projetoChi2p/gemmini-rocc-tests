#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// ==========================================
// UNIFIED TRANSFORMER LOOP
// ==========================================

uint64_t compute_transformer_blocks(
    // 1. Dimensions
    int hidden_dim, 
    int expansion_dim, 
    int num_heads, 
    int cross_num_heads, // FP32 Decoder only
    int seq_len, 
    int num_layers,
    
    // FP32 Specific
    #ifndef QUANTIZED
    int compression_factor,
    #endif

    // 2. Data Pointers
    const elem_t * input,    
    const elem_t * enc_out,  // NULL if Encoder
    elem_t * out,            // Final Output Buffer

    // 3. Weights (Self Attn)
    const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
    const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b, const acc_t * Wo_b,
    
    // 4. Weights (Cross Attn - Optional/FP32)
    const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,
    const acc_t * Wq_cross_b, const acc_t * Wk_cross_b, const acc_t * Wv_cross_b, const acc_t * Wo_cross_b,

    // 5. Weights (FFN)
    const elem_t * ff1_w, const elem_t * ff2_w,
    const acc_t * ff1_b, const acc_t * ff2_b,

    // 6. Scratchpad Buffers
    elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
    elem_t * attn_scores_buf, elem_t * attn_probs_buf,
    elem_t * attn_norm_buf, elem_t * attn_context_buf,
    elem_t * attn_wo_buf, elem_t * attn_resadd_buf,

    elem_t * ffn_norm_buf, elem_t * ffn_fc1_buf,
    elem_t * ffn_gelu_buf, elem_t * ffn_fc2_buf,
    elem_t * ffn_resadd_buf,

    acc_t * out_buf_acc,    // Accumulator Scratch
    
    // 7. Constants & Scales
    float score_scaling_factor,
    float score_scaling_softmax,

    #ifdef QUANTIZED    
        const float * scales_q, const float * scales_k, const float * scales_v, const float * scales_wo,
        const float * scales_scores, const float * scales_ff1, const float * scales_ff2,
    #endif

    float context_scaling_factor

)
{
    const bool is_encoder = (enc_out == NULL);

    // --- STRIDE CALCULATION ---
    int hidden_dim_compressed = hidden_dim;
    #ifndef QUANTIZED
    if (compression_factor > 1) hidden_dim_compressed /= compression_factor;
    #endif

    size_t stride_W_attn = (size_t)hidden_dim * hidden_dim_compressed;
    size_t stride_Wo_attn = (size_t)hidden_dim_compressed * hidden_dim;
    size_t stride_b_attn = (size_t)hidden_dim_compressed; 
    size_t stride_Wo_b    = (size_t)hidden_dim;

    size_t stride_ff1_w = (size_t)hidden_dim * expansion_dim;
    size_t stride_ff2_w = (size_t)expansion_dim * hidden_dim;
    size_t stride_ff1_b = (size_t)expansion_dim;
    size_t stride_ff2_b = (size_t)hidden_dim;

    uint64_t start = read_cycles();
    const elem_t * layer_in = input;

    for (int l = 0; l < num_layers; l++) {
        
        // --- DEBUG SETUP ---
        #ifdef DEBUG
            global_layer_index = l;
            if (l == 0 && debug_inference) printf("\n=== Layer %d Start ===\n", l);
        #endif

        gemmini_fence(); 

        // ====================================================================
        // STEP 1: SELF ATTENTION
        // ====================================================================
        // Flow: Input -> QKV -> Scores -> Context -> Linear Proj -> out_buf
        
        compute_attention(hidden_dim, num_heads, seq_len, 
            layer_in, 
            attn_resadd_buf, // Final attention output (post-residual)

            Wq, Wk, Wv, Wo,
            Wq_b, Wk_b, Wv_b, Wo_b,
            Q_buf, K_buf, V_buf,
            attn_scores_buf, attn_probs_buf,
            attn_norm_buf, attn_context_buf,
            attn_wo_buf, attn_resadd_buf,
            
            #ifdef QUANTIZED
                scales_q[l], scales_k[l], scales_v[l], scales_wo[l], 
                score_scaling_softmax, scales_scores[l],
            #else
                ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
                score_scaling_factor, score_scaling_factor,
            #endif
            score_scaling_factor, context_scaling_factor
            
        );

        

        // ====================================================================
        // STEP 2: RESIDUAL + LAYERNORM (Self Attn)
        // ====================================================================
        // Flow: out_buf + layer_in -> resadd1_buf -> Norm -> ffn_input_ptr

        // Flow: out_buf -> Norm -> norm_out_ptr + layer_in -> ffn_input_ptr

        //if (l == 0 && debug_inference) ffn_input_ptr = debug_layer0_res1; // Output of ResAdd becomes input to FFN, which will write to 'out'

        // ====================================================================
        // STEP 4: FEED FORWARD NETWORK
        // ====================================================================
        // Flow: ffn_input_ptr -> FC1 -> GELU -> FC2 -> Residual -> Norm -> out
        
        compute_ffn(
            hidden_dim, expansion_dim, seq_len,
            attn_resadd_buf,
            out,

            ff1_w, ff2_w, 
            ff1_b, ff2_b,
            
            ffn_norm_buf,
            ffn_fc1_buf,
            ffn_gelu_buf,
            ffn_fc2_buf,
            ffn_resadd_buf,
            out_buf_acc,
            
            #ifdef QUANTIZED
                scales_ff1[l], scales_ff2[l]
            #else   
                ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY    
            #endif
        );

        #ifdef DEBUG
            if (l == 0 && debug_inference) verify_tensor("Layer 0 Out", out, (elem_t*)debug_layer0_out, seq_len * hidden_dim, TOLERANCE);
            //if (l == 1 && debug_inference) verify_tensor("Layer 1 Out", out, (elem_t*)debug_layer1_out, seq_len * hidden_dim, TOLERANCE);
            //if (l == 2 && debug_inference) verify_tensor("Layer 2 Out", out, (elem_t*)debug_layer2_out, seq_len * hidden_dim, TOLERANCE);
            //if (l == 3 && debug_inference) verify_tensor("Layer 3 Out", out, (elem_t*)debug_layer3_out, seq_len * hidden_dim, TOLERANCE);
            //exit(0);
        #endif

        // ====================================================================
        // STEP 5: POINTER UPDATES
        // ====================================================================

        Wq += stride_W_attn; Wk += stride_W_attn; Wv += stride_W_attn; 
        Wo += stride_Wo_attn; 
        
        Wq_b += stride_b_attn; Wk_b += stride_b_attn; Wv_b += stride_b_attn; 
        Wo_b += stride_Wo_b;   
        
        ff1_w += stride_ff1_w; ff2_w += stride_ff2_w; 
        ff1_b += stride_ff1_b; ff2_b += stride_ff2_b;

        layer_in = out; // Output becomes input for next layer

        gemmini_fence();
    }

    uint64_t end = read_cycles();
    return end - start;
}