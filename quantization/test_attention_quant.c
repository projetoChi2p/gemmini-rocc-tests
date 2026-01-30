#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "includes/quantized_attention_params.h"

// --- BAREMETAL HELPERS ---
static inline float my_exp(float x) {
    if (x <= -88.0f) return 0.0f;
    if (x >= 88.0f) x = 88.0f;
    union { float f; int32_t i; } converter;
    converter.i = (int32_t)(12102203.0f * x + 1064986824);
    return converter.f;
}

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
            float exp_val = my_exp(row_buf[j] - max_val);
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

// --- ATTENTION MODULE ---
void attention(int hidden_dim, int num_heads, int seq_len,
        const elem_t * input, elem_t * out, 
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b, const acc_t * Wo_b,
        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf,
        float scale_q, float scale_k, float scale_v, float scale_wo, 
        float score_scaling, float scale_scores_matmul // <--- NEW PARAM
    )
{
    int head_dim = hidden_dim / num_heads;

    // 1. Projections
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

    // 2. Scores (Q * K^T)
    // We now apply scale_scores_matmul to prevent overflow in attn_buf (int8)
    for (int h = 0; h < num_heads; h++) {
        tiled_matmul_auto(seq_len, seq_len, head_dim,
            Q_buf + h*head_dim, K_buf + h*head_dim, 
            NULL, 
            attn_buf + h*seq_len*seq_len,
            hidden_dim, hidden_dim, seq_len, seq_len, 
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, 
            (acc_scale_t)scale_scores_matmul, // <--- CRITICAL FIX: SCALE APPLIED HERE
            0, false, 
            false, true, // Transpose B (K)
            false, false, 0, WS);
    }
    gemmini_fence();

    // 3. Softmax
    for (int h = 0; h < num_heads; h++) {
        cpu_softmax_quantized(seq_len, seq_len, attn_buf + h*seq_len*seq_len, score_scaling);
    }

    // 4. Context (Probs * V)
    float context_scale = 1.0f / 127.0f;
    for (int h = 0; h < num_heads; h++) {
        tiled_matmul_auto(seq_len, head_dim, seq_len,
            attn_buf + h*seq_len*seq_len, V_buf + h*head_dim, 
            NULL, 
            out_buf + h*head_dim,
            seq_len, hidden_dim, hidden_dim, hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, (acc_scale_t)context_scale, 0, false, 
            false, false, 
            false, false, 0, WS);
    }
    gemmini_fence();

    // 5. Output Projection
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim,
        out_buf, Wo, Wo_b, out,
        hidden_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, (acc_scale_t)scale_wo, 0, true, false, false, false, false, 0, WS);
        
    gemmini_fence();
}

int main() {
    gemmini_flush(0);
    printf("=== Quantized Attention Test (Scaled Scores) ===\n");

    static elem_t q_buf[SEQ_LEN][HIDDEN_DIM] row_align(1);
    static elem_t k_buf[SEQ_LEN][HIDDEN_DIM] row_align(1);
    static elem_t v_buf[SEQ_LEN][HIDDEN_DIM] row_align(1);
    static elem_t attn_buf[NUM_HEADS][SEQ_LEN][SEQ_LEN] row_align(1);
    static elem_t out_buf[SEQ_LEN][HIDDEN_DIM] row_align(1);
    static elem_t final_out[SEQ_LEN][HIDDEN_DIM] row_align(1);

    uint64_t start = read_cycles();

    attention(HIDDEN_DIM, NUM_HEADS, SEQ_LEN,
        (elem_t*)inp, (elem_t*)final_out,
        (elem_t*)wq, (elem_t*)wk, (elem_t*)wv, (elem_t*)wo,
        (acc_t*)bq, (acc_t*)bk, (acc_t*)bv, (acc_t*)bo,
        (elem_t*)q_buf, (elem_t*)k_buf, (elem_t*)v_buf,
        (elem_t*)attn_buf, (elem_t*)out_buf,
        SCALE_Q, SCALE_K, SCALE_V, SCALE_WO, SCORE_SCALING, 
        SCALE_SCORES // <--- PASS NEW PARAM
    );

    uint64_t end = read_cycles();
    printf("Cycles: %llu\n", end - start);

    // Verify
    int errors = 0;
    int tolerance = 5; 

    for (int i = 0; i < SEQ_LEN * HIDDEN_DIM; i++) {
        elem_t prod = ((elem_t*)final_out)[i];
        elem_t exp = ((elem_t*)expected_output)[i];
        
        if (abs(prod - exp) > tolerance) {
            if (errors < 5) printf("Err %d: Got %d Exp %d (Diff %d)\n", i, prod, exp, prod-exp);
            errors++;
        }
    }

    if (errors == 0) printf("SUCCESS\n");
    else printf("FAIL: %d errors\n", errors);

    return 0;
}