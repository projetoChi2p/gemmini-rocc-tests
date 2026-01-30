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
    const elem_t * w, const acc_t * b, float scale, // CLS Head Weights
    const elem_t * w_d, const acc_t * b_d           // Dist Head Weights (Assuming same scale)
) {
    // Buffers for LayerNorm
    static elem_t cls_buf[512];
    static elem_t dist_buf[512];
    
    // Temporary Logits
    static elem_t logits_cls[10];  
    static elem_t logits_dist[10];

    // --- 1. Process CLS Token (Row 0) ---
    memcpy(cls_buf, encoder_output, hidden_dim * sizeof(elem_t));
    cpu_layernorm_cls(cls_buf, hidden_dim);
    
    tiled_matmul_auto(1, num_classes, hidden_dim,
        cls_buf, w, b, logits_cls,
        hidden_dim, num_classes, num_classes, num_classes,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)scale, 0, true,
        false, false, false, false, 0, WS);
        
    gemmini_fence();

    // --- 2. Process Distillation Token (Row 1) ---
    // Pointer math: encoder_output + 1*hidden_dim
    memcpy(dist_buf, encoder_output + hidden_dim, hidden_dim * sizeof(elem_t));
    cpu_layernorm_cls(dist_buf, hidden_dim);
    
    tiled_matmul_auto(1, num_classes, hidden_dim,
        dist_buf, w_d, b_d, logits_dist,
        hidden_dim, num_classes, num_classes, num_classes,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)scale, 0, true,
        false, false, false, false, 0, WS);

    gemmini_fence();

    // --- 3. Average ---
    for (int i = 0; i < num_classes; i++) {
        int sum = (int)logits_cls[i] + (int)logits_dist[i];
        final_logits[i] = (elem_t)sum;
    }
}