// File: transformer_layers_debug.c (Partial Update)

#include "include/gemmini_nn.h"
#include <math.h>
#include <float.h>

// ==========================================
// 1. HELPER FUNCTIONS (From your update)
// ==========================================

// Optimized for range [-87, 87]. Very fast, no loops.
static inline float fast_exp_schraudolph(float x) {
    if (x <= -88.0f) return 0.0f;
    if (x >= 88.0f) x = 88.0f;
    union { float f; int i; } converter;
    converter.i = (int)(12102203.0f * x + 1064986824);
    return converter.f;
}

void cpu_softmax(int rows, int cols, elem_t * matrix) {
    for (int i = 0; i < rows; i++) {
        // A. Find Max
        float max_val = -FLT_MAX;
        for (int j = 0; j < cols; j++) {
            float val = (float)matrix[i * cols + j];
            if (val > max_val) max_val = val;
        }

        // B. Exponentiate (x - max)
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = (float)matrix[i * cols + j] - max_val;
            float exp_val = fast_exp_schraudolph(val);
            matrix[i * cols + j] = (elem_t)exp_val;
            sum_exp += exp_val;
        }

        // C. Normalize
        float inv_sum = 1.0f / (sum_exp + 1e-6f); 
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] *= inv_sum;
        }
    }
}

void cpu_scale_matrix(int rows, int cols, elem_t * matrix, float scale) {
    int size = rows * cols;
    for (int i = 0; i < size; i++) {
        matrix[i] = (elem_t)(matrix[i] * scale);
    }
}

// ==========================================
// 2. ATTENTION MODULE
// ==========================================

void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,
        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b, const acc_t * Wo_b,
        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc, float score_scaling_factor
        
        #ifdef DEBUG
        // Debug Pointers (Kept from original signature)
        ,const elem_t * expected_Q, const elem_t * expected_K, const elem_t * expected_V,
        const elem_t * expected_scores, const elem_t * expected_probs
        #endif
    )
{
    int hidden_dim_compressed = hidden_dim / compression_factor;
    int hidden_dim_per_head = hidden_dim_compressed / num_heads;
    if (compression_factor < 0) {
        hidden_dim_compressed = hidden_dim;
        hidden_dim_per_head = (hidden_dim_compressed / 12) * (-compression_factor);
    }

    // --- 1. Compute Q, K, V ---
    const int qkv_matmuls_n = 3;
    const elem_t * qkv_weights[] = {Wq, Wk, Wv};
    const elem_t * qkv_ins[] = {input, enc_out, enc_out};
    const acc_t * qkv_bs[] = {Wq_b, Wk_b, Wv_b};
    elem_t * qkv_outs[] = {Q_buf, K_buf, V_buf};

    for (int i = 0; i < qkv_matmuls_n; i++) {
        tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
            qkv_ins[i], qkv_weights[i], qkv_bs[i], qkv_outs[i],
            hidden_dim, hidden_dim, 0, hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, false, false, false, false, 0, WS);
    }
    gemmini_fence();

    // DEBUG: Check Q, K, V
    #ifdef DEBUG
    if (expected_Q != NULL) {
        print_error_histogram("L0 Q Matrix", seq_len, hidden_dim, Q_buf, expected_Q);
        print_error_histogram("L0 K Matrix", seq_len, hidden_dim, K_buf, expected_K);
        print_error_histogram("L0 V Matrix", seq_len, hidden_dim, V_buf, expected_V);
    }
    #endif

    // --- 2. Compute Q * K^T (Scores) & Scale ---
    acc_scale_t gemmini_scale = ACC_SCALE_IDENTITY; 

    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = Q_buf + head * hidden_dim_per_head;
        const elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf + head * seq_len * seq_len;

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            A, B, NULL, C,
            hidden_dim, hidden_dim, 0, seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, gemmini_scale, 0, false, false, true, false, false, 0, WS);

        // Apply Scaling (CPU)
        cpu_scale_matrix(seq_len, seq_len, C, score_scaling_factor);
    }
    gemmini_fence();

    // DEBUG: Check Scores (Pre-Softmax)
    // Note: If expected_scores from python are Raw (unscaled), this might show a diff, 
    // but usually Pytorch hooks capture the input to Softmax.
    #ifdef DEBUG
    if (expected_scores != NULL) print_error_histogram("L0 Scores (Scaled)", seq_len, seq_len, attn_buf, expected_scores);
    #endif

    // --- 3. Softmax (CPU Approximation) ---
    for (int head = 0; head < num_heads; head++) {
        elem_t * scores_matrix = attn_buf + head * seq_len * seq_len;
        cpu_softmax(seq_len, seq_len, scores_matrix);
    }

    // DEBUG: Check Probs (Post-Softmax)
    #ifdef DEBUG
    if (expected_probs != NULL) print_error_histogram("L0 Probs (Softmaxed)", seq_len, seq_len, attn_buf, expected_probs);
    #endif

    // --- 4. Compute Context (Probs * V) ---
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = attn_buf + head * seq_len * seq_len;
        const elem_t * B = V_buf + head * hidden_dim_per_head;
        elem_t * C = out_buf + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            A, B, NULL, C,
            seq_len, hidden_dim, 0, hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, false, false, false, false, 0, WS);
    }
    gemmini_fence();

    // --- 5. Output Projection ---
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        out_buf, Wo, Wo_b, out_buf_acc,
        hidden_dim, hidden_dim, 0, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, false, false, true, false, 0, WS);

    gemmini_fence();

    // --- 6. Norm + Residual ---
    tiled_norm_auto(seq_len, hidden_dim, (acc_t*)out_buf_acc, out_buf,
        ACC_SCALE_IDENTITY, LAYERNORM, WS);
    
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        input, out_buf, resadd_out, false, WS);
    
    gemmini_fence();
}