#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// Helper for rounding
static inline int my_round_cls(float x) {
    return (int)(x + (x >= 0 ? 0.5f : -0.5f));
}

// 1. CPU LayerNorm for CLS Token
void cpu_layernorm_cls(elem_t * cls_token, int dim) {
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    // Mean & Variance
    for(int j = 0; j < dim; j++) {
        float v = cls_token[j];
        sum += v;
        sq_sum += v * v;
    }
    
    float mean = sum / dim;
    float var = (sq_sum / dim) - (mean * mean);
    if (var < 0) var = 0;
    float std = sqrtf(var + 1e-5f); 
    
    // Scale (20.0 factor matches training quantization)
    float scale_factor = 20.0f / std;

    for(int j = 0; j < dim; j++) {
        float v = cls_token[j];
        float n = (v - mean) * scale_factor;
        
        int res = my_round_cls(n);
        if (res > 127) res = 127;
        if (res < -128) res = -128;
        
        cls_token[j] = (elem_t)res;
    }
}

void classifier_head_deit_quantized(
    int hidden_dim, int num_classes,
    const elem_t * encoder_output, // [Total_Seq, Hidden]
    elem_t * final_logits,         // [1, Classes]
    const elem_t * w, const acc_t * b, float scale // CLS Head Weights
    #ifdef DISTILLATION
        ,
        const elem_t * w_d, const acc_t * b_d           // Dist Head Weights
    #endif
) {
    // Buffers for LayerNorm and Logits
    static elem_t cls_buf[512];
    static elem_t logits_cls[10];  

    #ifdef DISTILLATION
    static elem_t dist_buf[512];
    static elem_t logits_dist[10];
    #endif

    // --- 1. Process CLS Token (Row 0) ---
    memcpy(cls_buf, encoder_output, hidden_dim * sizeof(elem_t));
    
    #ifdef CPU_LAYERNORM
        cpu_layernorm_cls(cls_buf, hidden_dim);
    #else
        tiled_norm_auto(
            1, hidden_dim, 
            (acc_t*)cls_buf,    // Input (Accumulator/Int32)
            (elem_t*)cls_buf,   // Output (Int8)
            ACC_SCALE_IDENTITY,
            LAYERNORM, WS
        );
    #endif

    tiled_matmul_auto(1, num_classes, hidden_dim,
        cls_buf, w, b, logits_cls,
        hidden_dim, num_classes, num_classes, num_classes,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)scale, 0, true,
        false, false, false, false, 0, WS);
        
    gemmini_fence();

    #ifdef DISTILLATION
    // --- 2. Process Distillation Token (Row 1) ---
    // Pointer math: encoder_output + 1*hidden_dim (Assuming sequence is [CLS, DIST, ...])
    memcpy(dist_buf, encoder_output + hidden_dim, hidden_dim * sizeof(elem_t));
    
    #ifdef CPU_LAYERNORM
        cpu_layernorm_cls(dist_buf, hidden_dim);
    #else
        tiled_norm_auto(
            1, hidden_dim, 
            (acc_t*)dist_buf,    // Input (Accumulator/Int32)
            (elem_t*)dist_buf,   // Output (Int8)
            ACC_SCALE_IDENTITY,
            LAYERNORM, WS
        );
    #endif
    
    tiled_matmul_auto(1, num_classes, hidden_dim,
        dist_buf, w_d, b_d, logits_dist,
        hidden_dim, num_classes, num_classes, num_classes,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)scale, 0, true,
        false, false, false, false, 0, WS);

    gemmini_fence();
    #endif

    // --- 3. Output / Fusion ---
    for (int i = 0; i < num_classes; i++) {
        #ifdef DISTILLATION
            // Late Fusion: Sum logits (Standard DeiT Approach)
            int sum = (int)logits_cls[i] + (int)logits_dist[i];
            
            // Optional: Clamp to int8 range to prevent overflow artifacts
            if (sum > 127) sum = 127;
            if (sum < -128) sum = -128;
            
            final_logits[i] = (elem_t)sum;
        #else
            // Standard ViT: Just return CLS logits
            final_logits[i] = logits_cls[i];
        #endif
    }
}