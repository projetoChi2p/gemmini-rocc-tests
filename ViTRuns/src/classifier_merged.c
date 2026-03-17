#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"


void compute_classifier(
    int hidden_dim, int num_classes,
    const elem_t * encoder_output, 
    elem_t * final_logits,         
    const elem_t * head_w, const acc_t * head_b,
    elem_t * ln_output_buf, 
    float scale_act_ln_final,
    float scale
    #ifdef DISTILLATION
        , const elem_t * head_w_dist, const acc_t * head_b_dist
    #endif
) {
    // Persistent buffers to prevent stack overflow
    static elem_t logits_cls[1024] row_align(1); 
    #ifdef DISTILLATION
        static elem_t logits_dist[1024] row_align(1);
        static elem_t dist_token_scratch[2048] row_align(1);
    #endif

    // --- STEP 1: NORMALIZE CLS TOKEN ---
    // Topology: We must move the CLS token (Row 0) to a clean buffer first.
    memcpy(ln_output_buf, encoder_output, hidden_dim * sizeof(elem_t));

    #ifdef QUANTIZED
        #ifdef CPU_LAYERNORM
            cpu_layernorm(1, hidden_dim, ln_output_buf, scale_act_ln_final);
        #else
            // HW Norm expects acc_t input. If elem_t is int8, this requires a cast/conversion.
            // For stability, we assume ln_output_buf is already treated as the target.
            tiled_norm_auto(1, hidden_dim, (acc_t*)ln_output_buf, ln_output_buf,
                            ACC_SCALE_IDENTITY, LAYERNORM, WS);
        #endif
    #else
        // FP32 Mode: Standard LayerNorm
        tiled_norm_auto(1, hidden_dim, (acc_t*)ln_output_buf, ln_output_buf,
                        ACC_SCALE_IDENTITY, LAYERNORM, WS);
    #endif

    gemmini_fence();

    #ifdef DEBUG
    if (debug_inference) {
        verify_tensor("CLS Token Norm", ln_output_buf, (elem_t*)debug_final_ln, hidden_dim, TOLERANCE);
        display_tensor_distribution_histogram("CLS Token After LN", ln_output_buf, hidden_dim);
        display_tensor_distribution_histogram("CLS Token After LN (Expected)", debug_final_ln, hidden_dim);
        // exit(0);
    }
    #endif

    // --- STEP 2: CLS PROJECTION ---
    acc_scale_t proj_scale = ACC_SCALE_IDENTITY;
    #ifdef QUANTIZED    
        proj_scale = (acc_scale_t)SCALE_HEAD; // Example scale for classifier head (depends on final activation range)
    #endif

    tiled_matmul_auto(1, num_classes, hidden_dim,
        ln_output_buf, head_w, head_b, logits_cls,
        hidden_dim, num_classes, num_classes, num_classes,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, proj_scale, 0, 
        true,  // repeating_bias: True for most ViT heads
        false, false, false, false, 0, WS);
        
    gemmini_fence();

    #ifdef DISTILLATION
        // --- STEP 3: PROCESS DISTILLATION TOKEN ---
        // The Dist token is at index 1 (encoder_output + 1*hidden_dim)
        memcpy(dist_token_scratch, encoder_output + hidden_dim, hidden_dim * sizeof(elem_t));
        
        #ifdef QUANTIZED
            #ifdef CPU_LAYERNORM
                cpu_layernorm_cls(dist_token_scratch, hidden_dim);
            #else
                tiled_norm_auto(1, hidden_dim, (acc_t*)dist_token_scratch, dist_token_scratch,
                                ACC_SCALE_IDENTITY, LAYERNORM, WS);
            #endif
        #else
            tiled_norm_auto(1, hidden_dim, (acc_t*)dist_token_scratch, dist_token_scratch,
                            ACC_SCALE_IDENTITY, LAYERNORM, WS);
        #endif

        gemmini_fence();
        
        tiled_matmul_auto(1, num_classes, hidden_dim,
            dist_token_scratch, head_w_dist, head_b_dist, logits_dist,
            hidden_dim, num_classes, num_classes, num_classes,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, proj_scale, 0, true,
            false, false, false, false, 0, WS);

        gemmini_fence();
    #endif

    // --- STEP 4: OUTPUT FUSION ---
    for (int i = 0; i < num_classes; i++) {
        #ifdef DISTILLATION
            // Sum logits (Late Fusion)
            // Use a float/int32 accumulator to prevent mid-sum overflow
            float sum_val = (float)logits_cls[i] + (float)logits_dist[i];
            
            #ifdef QUANTIZED
                if (sum_val > elem_t_max) sum_val = elem_t_max;
                if (sum_val < elem_t_min) sum_val = elem_t_min;
            #endif
            
            final_logits[i] = (elem_t)sum_val;
        #else
            final_logits[i] = logits_cls[i];
        #endif
    }

    // softmax logit
    

    gemmini_fence();
}