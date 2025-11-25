// File: transformer_layers_debug.c

#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_nn.h"
#include <math.h>
#include <float.h> 

// --- 1. Advanced Histogram Debugging Tool ---

#define HIST_BAR_WIDTH 40
#define HIST_BINS 7

// Bins: [0-1e-6), [1e-6, 1e-5), [1e-5, 1e-4), [1e-4, 1e-3), [1e-3, 1e-2), [1e-2, 1e-1), [> 1e-1]
const float BIN_THRESHOLDS[HIST_BINS] = {0.000001f, 0.00001f, 0.0001f, 0.001f, 0.01f, 0.1f, FLT_MAX};
const char* BIN_LABELS[HIST_BINS]     = {"< 1e-6", "1e-6  ", "1e-5  ", "1e-4  ", "1e-3  ", "1e-2  ", "> 1e-1"};

void print_error_histogram(const char * step_name, int rows, int cols, 
                           const elem_t * calculated, const elem_t * expected) {
    
    int counts[HIST_BINS] = {0};
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int total_elements = rows * cols;
    int errors_above_threshold = 0;
    float threshold = 0.01f; // Threshold for "FAIL" judgement

    // 1. Collect Stats
    for (int i = 0; i < total_elements; i++) {
        float c_val = (float)calculated[i];
        float e_val = (float)expected[i];
        float diff = fabs(c_val - e_val);

        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;

        if (diff > threshold) errors_above_threshold++;

        // Binning
        for (int b = 0; b < HIST_BINS; b++) {
            if (diff < BIN_THRESHOLDS[b]) {
                counts[b]++;
                break;
            }
        }
    }

    float mae = sum_diff / total_elements;

    // 2. Print Header
    printf("\n=== DEBUG: %s ===\n", step_name);
    printf("  Dims: %dx%d | Max Diff: %.6f | MAE: %.6f\n", rows, cols, max_diff, mae);
    
    // 3. Print Histogram
    printf("  Error Distribution (Log Scale):\n");
    for (int b = 0; b < HIST_BINS; b++) {
        // Calculate bar length
        int bar_len = (int)((float)counts[b] / total_elements * HIST_BAR_WIDTH);
        
        printf("    %s : ", BIN_LABELS[b]);
        for (int k = 0; k < bar_len; k++) printf("#");
        if (counts[b] > 0 && bar_len == 0) printf("."); // Dot for non-zero but small count
        
        // Print count and percentage
        printf(" (%d - %.1f%%)\n", counts[b], (float)counts[b]/total_elements * 100.0f);
    }

    // 4. Final Judgement
    if (errors_above_threshold > 0) {
        printf("  [FAIL] %d elements have errors > %.2f\n", errors_above_threshold, threshold);
    } else {
        printf("  [PASS] All errors within tolerance.\n");
    }
    printf("--------------------------------------------------\n");
}

// --- Helper Functions (Exp/GELU/Softmax) ---

float gelu_exp_approx(float x) {
    if (x < -5.0f) return 0.001f * (x + 10.0f);
    else if (x < -2.0f) { float xp = x + 3.5f; return 0.030197f + 0.028374f * xp + 0.013903f * xp * xp + 0.004534f * xp * xp * xp; }
    else if (x < 0.0f) { float x2 = x * x; float x3 = x2 * x; float x4 = x2 * x2; return 1.0f + x + 0.5f * x2 + 0.1666667f * x3 + 0.0416664f * x4; }
    else if (x < 2.0f) { float x2 = x * x; float x3 = x2 * x; float x4 = x2 * x2; return 1.0f + x + 0.5f * x2 + 0.1666667f * x3 + 0.0416664f * x4; }
    else if (x < 5.0f) { float xp = x - 3.5f; return 16.444647f + 16.444647f * xp + 8.222323f * xp * xp + 2.740774f * xp * xp * xp; }
    else { return 100.0f + 50.0f * (x - 5.0f); }
}

void cpu_gelu_approx(int rows, int cols, elem_t * input, elem_t * output) {
    for (int i = 0; i < rows * cols; i++) {
        float x = (float)input[i];
        float exp_val = gelu_exp_approx(-1.702f * x);
        float sigmoid = 1.0f / (1.0f + exp_val);
        output[i] = (elem_t)(x * sigmoid);
    }
}

// --- Attention Layer ---

void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,
        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b, const acc_t * Wo_b,
        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc,
        
        // Debug Pointers
        const elem_t * expected_Q, const elem_t * expected_K, const elem_t * expected_V,
        const elem_t * expected_scores, const elem_t * expected_probs)
{
    int hidden_dim_compressed = hidden_dim / compression_factor;
    int hidden_dim_per_head = hidden_dim_compressed / num_heads;
    if (compression_factor < 0) {
        hidden_dim_compressed = hidden_dim;
        hidden_dim_per_head = (hidden_dim_compressed / 12) * (-compression_factor);
    }

    // 1. Q, K, V
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

    if (expected_Q != NULL) {
        print_error_histogram("L0 Q Matrix", seq_len, hidden_dim, Q_buf, expected_Q);
        print_error_histogram("L0 K Matrix", seq_len, hidden_dim, K_buf, expected_K);
        print_error_histogram("L0 V Matrix", seq_len, hidden_dim, V_buf, expected_V);
    }

    // 2. Scores
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
    }
    gemmini_fence();

    if (expected_scores != NULL) print_error_histogram("L0 Scores", seq_len, seq_len, attn_buf, expected_scores);

    // 3. Probs (No Softmax)
    if (expected_probs != NULL) print_error_histogram("L0 Probs", seq_len, seq_len, attn_buf, expected_probs);

    // 4. Context
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

    // 5. Output Projection
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        out_buf, Wo, Wo_b, out_buf_acc,
        hidden_dim, hidden_dim, 0, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, false, false, true, false, 0, WS);

    gemmini_fence();

    // 6. Norm + Residual (Corrected for Aliasing)
    tiled_norm_auto(seq_len, hidden_dim, (acc_t*)out_buf_acc, out_buf,
        ACC_SCALE_IDENTITY, LAYERNORM, WS);
    
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        input, out_buf, resadd_out, false, WS);
    
    gemmini_fence();
}

// --- FFN Layer ---
void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf, acc_t * out_buf_acc,
        
        // Debug Pointers
        const elem_t * expected_fc1,
        const elem_t * expected_fc2)
{
    // 1. FC1 (Matmul Only)
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        input, ff1_w, ff1_b, out_buf,
        hidden_dim, expansion_dim, expansion_dim, expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        RELU, 
        ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        true, false, false, false, false, 0, WS);

    gemmini_fence();

    // 1.5 Manual ReLU (Changed from GELU to match your model update)
    // Note: You switched to ReLU in python model. 
    // If you are using Gemmini's RELU opcode, put it in Step 1.
    // If you are doing it manually in C, use a simple loop.
    // Assuming manual for safety/debug parity:
    /*for (int i = 0; i < seq_len * expansion_dim; i++) {
        if (out_buf[i] < 0) out_buf[i] = 0;
    }*/

    if (expected_fc1 != NULL) print_error_histogram("L0 FFN FC1+ReLU", seq_len, expansion_dim, out_buf, expected_fc1);

    // 2. FC2
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
        out_buf, ff2_w, ff2_b, out_buf_acc,
        expansion_dim, hidden_dim, hidden_dim, hidden_dim, 
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, 
        false, // full_C = false (Overwrite)
        false, 0, WS);

    gemmini_fence();

    if (expected_fc2 != NULL) print_error_histogram("L0 FFN FC2", seq_len, hidden_dim, (elem_t*)out_buf_acc, expected_fc2);

    // --- STEP 3 & 4 FIX: Buffer Aliasing ---

    // 3. Norm
    // Write to 'out_buf' (temp) instead of 'out' to protect 'input'
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, 
        out_buf, // <--- TARGET IS NOW TEMP BUFFER
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    gemmini_fence();

    // 4. Residual Add
    // Add 'input' + 'out_buf' -> 'out'
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        out_buf, // A: Normalized FFN Output
        input,   // B: Original Input (Residual) - Safe now!
        out,     // C: Final Output Destination
        /*relu=*/ false,
        WS);

    gemmini_fence();
}
// --- Encoder/Decoder Loop ---

uint64_t encoder_decoder(
        int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads,
        int seq_len, int compression_factor, int num_layers,

        const elem_t * input, const elem_t * enc_out, elem_t * out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,

        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        const acc_t * Wo_b,
        const acc_t * Wq_cross_b, const acc_t * Wk_cross_b, const acc_t * Wv_cross_b,
        const acc_t * Wo_cross_b,

        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc,
        elem_t * resadd1_buf, elem_t * resadd2_buf,
        
        // Debug pointers
        const elem_t ** expected_attn_outputs,
        const elem_t ** expected_ffn_outputs,

        const elem_t * debug_l0_q, const elem_t * debug_l0_k, const elem_t * debug_l0_v,
        const elem_t * debug_l0_scores, const elem_t * debug_l0_probs,
        const elem_t * debug_l0_ffn_fc1, const elem_t * debug_l0_ffn_fc2)
{
    // ... [Strides logic unchanged] ...
    const bool is_encoder = enc_out == NULL;
    int hidden_dim_compressed = hidden_dim / compression_factor;
    if (compression_factor < 0) hidden_dim_compressed = hidden_dim;
    
    size_t stride_W_attn = hidden_dim * hidden_dim_compressed;
    size_t stride_Wo_attn = hidden_dim_compressed * hidden_dim;
    size_t stride_b_attn = hidden_dim_compressed; 
    size_t stride_b_Wo = hidden_dim;
    size_t stride_ff1_w = hidden_dim * expansion_dim;
    size_t stride_ff2_w = expansion_dim * hidden_dim;
    size_t stride_ff1_b = expansion_dim;
    size_t stride_ff2_b = hidden_dim;

    uint64_t start = read_cycles();
    const elem_t * current_input = input;

    for (int l = 0; l < num_layers; l++) {
        // Debug pointers (L0 only)
        const elem_t * d_q = (l == 0) ? debug_l0_q : NULL;
        const elem_t * d_k = (l == 0) ? debug_l0_k : NULL;
        const elem_t * d_v = (l == 0) ? debug_l0_v : NULL;
        const elem_t * d_s = (l == 0) ? debug_l0_scores : NULL;
        const elem_t * d_p = (l == 0) ? debug_l0_probs : NULL;
        const elem_t * d_f1 = (l == 0) ? debug_l0_ffn_fc1 : NULL;
        const elem_t * d_f2 = (l == 0) ? debug_l0_ffn_fc2 : NULL;

        // 1. Attention
        attention(hidden_dim, expansion_dim, num_heads, seq_len, compression_factor,
            current_input, current_input,
            out, resadd1_buf,
            Wq, Wk, Wv, Wo,
            Wq_b, Wk_b, Wv_b, Wo_b,
            Q_buf, K_buf, V_buf,
            attn_buf, out_buf, out_buf_acc,
            d_q, d_k, d_v, d_s, d_p);
            
        if (expected_attn_outputs != NULL) {
            char buf[32];
            sprintf(buf, "Layer %d Attention", l);
            print_error_histogram(buf, seq_len, hidden_dim, resadd1_buf, expected_attn_outputs[l]);
        }

        // 2. FFN
        const elem_t * ffn_input = resadd1_buf;
        ffn(hidden_dim, expansion_dim, seq_len,
            ffn_input, out,
            ff1_w, ff2_w, ff1_b, ff2_b,
            out_buf, out_buf_acc,
            d_f1, d_f2);

        if (expected_ffn_outputs != NULL) {
            char buf[32];
            sprintf(buf, "Layer %d FFN", l);
            print_error_histogram(buf, seq_len, hidden_dim, out, expected_ffn_outputs[l]);
        }

        // Pointer Update
        Wq += stride_W_attn; Wk += stride_W_attn; Wv += stride_W_attn; Wo += stride_Wo_attn;
        Wq_b += stride_b_attn; Wk_b += stride_b_attn; Wv_b += stride_b_attn; Wo_b += stride_b_Wo;
        ff1_w += stride_ff1_w; ff2_w += stride_ff2_w; ff1_b += stride_ff1_b; ff2_b += stride_ff2_b;

        current_input = out;
    }

    uint64_t end = read_cycles();
    return end - start;
}