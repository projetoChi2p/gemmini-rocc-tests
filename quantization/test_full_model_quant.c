#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "includes/quantized_full_model_params.h"

// ==========================================
// 1. HELPERS (Self-Contained)
// ==========================================
static inline float my_exp(float x) {
    if (x <= -88.0f) return 0.0f;
    if (x >= 88.0f) x = 88.0f;
    union { float f; int32_t i; } converter;
    converter.i = (int32_t)(12102203.0f * x + 1064986824);
    return converter.f;
}

static inline float my_tanh(float x) {
    float abs_x = (x < 0.0f) ? -x : x;
    if (abs_x > 4.0f) return (x < 0.0f) ? -1.0f : 1.0f;
    float e2x = my_exp(2.0f * abs_x);
    float t = (e2x - 1.0f) / (e2x + 1.0f);
    return (x < 0.0f) ? -t : t;
}

static inline int my_round(float x) {
    return (int)(x + (x >= 0 ? 0.5f : -0.5f));
}

void cpu_layernorm(int rows, int cols, elem_t * data) {
    for(int i=0; i<rows; i++) {
        float sum=0, sq=0;
        for(int j=0; j<cols; j++) {
            float v = data[i*cols + j];
            sum += v; sq += v*v;
        }
        float mean = sum/cols;
        float std = sqrtf(sq/cols - mean*mean + 1e-5);
        for(int j=0; j<cols; j++) {
            float v = data[i*cols + j];
            float n = (v - mean)/std * 20.0f; 
            int res = my_round(n);
            if(res > 127) res=127; if(res<-128) res=-128;
            data[i*cols+j] = (elem_t)res;
        }
    }
}

void cpu_gelu(int size, elem_t * data) {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEF = 0.044715f;
    const float scale = 6.0f / 127.0f;
    
    for (int i = 0; i < size; i++) {
        float x = (float)data[i] * scale;
        float inner = SQRT_2_OVER_PI * (x + COEF * x * x * x);
        float res = 0.5f * x * (1.0f + my_tanh(inner));
        int out = my_round(res / scale);
        if(out > 127) out=127; if(out<-128) out=-128;
        data[i] = (elem_t)out;
    }
}

void cpu_softmax(int rows, int cols, elem_t * data, float input_scale) {
    for (int i = 0; i < rows; i++) {
        float row_buf[cols]; 
        float max_val = -10000.0f;
        for (int j = 0; j < cols; j++) {
            float val = (float)data[i*cols+j] * input_scale;
            row_buf[j] = val;
            if (val > max_val) max_val = val;
        }
        float sum = 0;
        for (int j = 0; j < cols; j++) {
            float e = my_exp(row_buf[j] - max_val);
            row_buf[j] = e;
            sum += e;
        }
        float inv_sum = 127.0f / (sum + 1e-6f);
        for (int j = 0; j < cols; j++) {
            int res = (int)(row_buf[j] * inv_sum);
            if(res>127) res=127; if(res<0) res=0;
            data[i*cols+j] = (elem_t)res;
        }
    }
}

// ==========================================
// 2. ViT FORWARD PASS
// ==========================================

// Global Buffers (Static to fit in BSS)
static elem_t input_seq[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t temp_buf[TOTAL_SEQ_LEN][EXPANSION_DIM] row_align(1); // Largest size needed
static elem_t resid_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t attn_scores[NUM_HEADS][TOTAL_SEQ_LEN][TOTAL_SEQ_LEN] row_align(1);
static elem_t attn_context[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);

// QKV Buffers
static elem_t q_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t k_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t v_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);

void vit_forward(elem_t * final_logits) {
    int seq_len = TOTAL_SEQ_LEN; // 17
    
    // --- 1. EMBEDDING ---
    // A. Project
    tiled_matmul_auto(SEQ_LEN, HIDDEN_DIM, PATCH_DIM,
        (elem_t*)img_patches, (elem_t*)w_embed, (acc_t*)b_embed, (elem_t*)input_seq + HIDDEN_DIM,
        PATCH_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)S_EMBED, 0, true, false, false, false, false, 0, WS);
        
    // B. Copy CLS to Row 0
    memcpy(input_seq, cls_token, HIDDEN_DIM);
    
    // C. Add Pos Embed
    tiled_resadd_auto(seq_len, HIDDEN_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t*)input_seq, (elem_t*)pos_embed, (elem_t*)input_seq,
        false, WS);
        
    // Save Residual 1
    memcpy(resid_buf, input_seq, sizeof(input_seq));
    
    // --- 2. ATTENTION ---
    // Q, K, V
    tiled_matmul_auto(seq_len, HIDDEN_DIM, HIDDEN_DIM, (elem_t*)input_seq, (elem_t*)wq, (acc_t*)bq, (elem_t*)q_buf,
        HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)S_Q, 0, true, false, false, false, false, 0, WS);
        
    tiled_matmul_auto(seq_len, HIDDEN_DIM, HIDDEN_DIM, (elem_t*)input_seq, (elem_t*)wk, (acc_t*)bk, (elem_t*)k_buf,
        HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)S_K, 0, true, false, false, false, false, 0, WS);
        
    tiled_matmul_auto(seq_len, HIDDEN_DIM, HIDDEN_DIM, (elem_t*)input_seq, (elem_t*)wv, (acc_t*)bv, (elem_t*)v_buf,
        HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)S_V, 0, true, false, false, false, false, 0, WS);

    gemmini_fence();
    
    // Scores
    int head_dim = HIDDEN_DIM / NUM_HEADS;
    for(int h=0; h<NUM_HEADS; h++) {
        tiled_matmul_auto(seq_len, seq_len, head_dim,
            (elem_t*)q_buf + h*head_dim, (elem_t*)k_buf + h*head_dim, NULL, (elem_t*)attn_scores + h*seq_len*seq_len,
            HIDDEN_DIM, HIDDEN_DIM, seq_len, seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, (acc_scale_t)S_SCORES, 0, false, false, true, false, false, 0, WS);
    }
    gemmini_fence();
    
    // Softmax
    for(int h=0; h<NUM_HEADS; h++) {
        cpu_softmax(seq_len, seq_len, (elem_t*)attn_scores + h*seq_len*seq_len, SCORE_SCALING);
    }
    
    // Context
    float ctx_scale = 1.0f/127.0f;
    for(int h=0; h<NUM_HEADS; h++) {
        tiled_matmul_auto(seq_len, head_dim, seq_len,
            (elem_t*)attn_scores + h*seq_len*seq_len, (elem_t*)v_buf + h*head_dim, NULL, (elem_t*)attn_context + h*head_dim,
            seq_len, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, (acc_scale_t)ctx_scale, 0, false, false, false, false, false, 0, WS);
    }
    gemmini_fence();
    
    // Output Projection
    tiled_matmul_auto(seq_len, HIDDEN_DIM, HIDDEN_DIM, (elem_t*)attn_context, (elem_t*)wo, (acc_t*)bo, (elem_t*)input_seq,
        HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)S_WO, 0, true, false, false, false, false, 0, WS);
        
    // ResAdd 1 (Input + Residual)
    tiled_resadd_auto(seq_len, HIDDEN_DIM, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t*)input_seq, (elem_t*)resid_buf, (elem_t*)input_seq, false, WS);
        
    // LN 1
    cpu_layernorm(seq_len, HIDDEN_DIM, (elem_t*)input_seq);
    
    // Save Residual 2
    memcpy(resid_buf, input_seq, sizeof(input_seq));
    
    // --- 3. FFN ---
    // FC1
    tiled_matmul_auto(seq_len, EXPANSION_DIM, HIDDEN_DIM, (elem_t*)input_seq, (elem_t*)w_fc1, (acc_t*)b_fc1, (elem_t*)temp_buf,
        HIDDEN_DIM, EXPANSION_DIM, EXPANSION_DIM, EXPANSION_DIM, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)S_FC1, 0, true, false, false, false, false, 0, WS);
        
    // GELU
    cpu_gelu(seq_len * EXPANSION_DIM, (elem_t*)temp_buf);
    
    // FC2
    tiled_matmul_auto(seq_len, HIDDEN_DIM, EXPANSION_DIM, (elem_t*)temp_buf, (elem_t*)w_fc2, (acc_t*)b_fc2, (elem_t*)input_seq,
        EXPANSION_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)S_FC2, 0, true, false, false, false, false, 0, WS);
        
    // ResAdd 2
    tiled_resadd_auto(seq_len, HIDDEN_DIM, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t*)input_seq, (elem_t*)resid_buf, (elem_t*)input_seq, false, WS);
        
    // LN 2
    cpu_layernorm(seq_len, HIDDEN_DIM, (elem_t*)input_seq);
    
    // --- 4. HEAD ---
    // Extract CLS (Row 0)
    static elem_t cls_vec[HIDDEN_DIM];
    memcpy(cls_vec, input_seq, HIDDEN_DIM);
    
    // LN Head
    cpu_layernorm(1, HIDDEN_DIM, cls_vec);
    
    // Proj
    tiled_matmul_auto(1, NUM_CLASSES, HIDDEN_DIM, cls_vec, (elem_t*)w_head, (acc_t*)b_head, final_logits,
        HIDDEN_DIM, NUM_CLASSES, NUM_CLASSES, NUM_CLASSES, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)S_HEAD, 0, true, false, false, false, false, 0, WS);
        
    gemmini_fence();
}

int main() {
    gemmini_flush(0);
    printf("=== Full Mini-ViT Model Test ===\n");
    
    static elem_t result_logits[NUM_CLASSES] row_align(1);
    
    uint64_t start = read_cycles();
    vit_forward(result_logits);
    uint64_t end = read_cycles();
    
    printf("Cycles: %llu\n", end - start);
    
    int errors = 0;
    for(int i=0; i<NUM_CLASSES; i++) {
        elem_t got = result_logits[i];
        elem_t exp = ((elem_t*)expected_output)[i];
        if(abs(got - exp) > 4) { // Tolerance 4
            printf("Err Class %d: Got %d Exp %d\n", i, got, exp);
            errors++;
        }
    }
    
    if(errors == 0) printf("SUCCESS: Full Model Verified!\n");
    else printf("FAIL: %d errors\n", errors);
    
    return 0;
}