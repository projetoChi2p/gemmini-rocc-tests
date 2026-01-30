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

// 2. Classifier Head
void classifier_head_quantized(
    int seq_len, 
    int hidden_dim, 
    int num_classes,
    const elem_t * input_seq,   // [Seq, Hidden]
    elem_t * output_scores,     // [1, Classes]
    const elem_t * w,           // Weights [Hidden, Classes]
    const acc_t * b,            // Bias [Classes]
    float scale                 // Scale
) {
    // A. Extract CLS Token (Row 0)
    static elem_t cls_buf[512]; 
    if (hidden_dim > 512) return;
    
    memcpy(cls_buf, input_seq, hidden_dim * sizeof(elem_t));

    // B. LayerNorm (CPU)
    cpu_layernorm_cls(cls_buf, hidden_dim);
    
    // C. Linear Projection (Explicit Tiled MatMul)
    // Operation: [1, Hidden] x [Hidden, Classes] = [1, Classes]
    tiled_matmul_auto(
        1, num_classes, hidden_dim,
        cls_buf,        // A (Input)
        w,              // B (Weights)
        b,              // Bias
        output_scores,  // D (Output)
        
        // Strides
        hidden_dim,     // A Stride (Row major)
        num_classes,    // B Stride (Row major: hidden * classes)
        num_classes,    // D Stride
        num_classes,    // C Stride
        
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 
        (acc_scale_t)scale, 
        0, 
        true, // Repeating Bias
        false, false, false, false, 0, WS
    );
        
    gemmini_fence();
}