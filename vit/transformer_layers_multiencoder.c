// File: transformer_layers.c

#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_nn.h"
#include <math.h>
#include <float.h> 

// --- Helper Functions ---

// 1. Self-contained Exponential Approximation
float gelu_exp_approx(float x) {
    if (x < -5.0f) {
        return 0.001f * (x + 10.0f);
    }
    else if (x < -2.0f) {
        float xp = x + 3.5f; 
        return 0.030197f + 0.028374f * xp + 0.013903f * xp * xp + 0.004534f * xp * xp * xp;
    }
    else if (x < 0.0f) {
        float x2 = x * x;
        float x3 = x2 * x;
        float x4 = x2 * x2;
        return 1.0f + x + 0.5f * x2 + 0.1666667f * x3 + 0.0416664f * x4;
    }
    else if (x < 2.0f) {
        float x2 = x * x;
        float x3 = x2 * x;
        float x4 = x2 * x2;
        return 1.0f + x + 0.5f * x2 + 0.1666667f * x3 + 0.0416664f * x4;
    }
    else if (x < 5.0f) {
        float xp = x - 3.5f; 
        return 16.444647f + 16.444647f * xp + 8.222323f * xp * xp + 2.740774f * xp * xp * xp;
    }
    else {
        return 100.0f + 50.0f * (x - 5.0f); 
    }
}

// 2. GELU Approximation (Sigmoid Method)
// Formula: x * sigmoid(1.702 * x) = x / (1 + exp(-1.702 * x))
void cpu_gelu_approx(int rows, int cols, elem_t * input, elem_t * output) {
    for (int i = 0; i < rows * cols; i++) {
        float x = (float)input[i];
        
        // Calculate exponent: -1.702 * x
        float exp_val = gelu_exp_approx(-1.702f * x);
        
        // Sigmoid result
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
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc)
{
    int hidden_dim_compressed = hidden_dim / compression_factor;
    int hidden_dim_per_head = hidden_dim_compressed / num_heads;
    if (compression_factor < 0) {
        hidden_dim_compressed = hidden_dim;
        hidden_dim_per_head = (hidden_dim_compressed / 12) * (-compression_factor);
    }

    // 1. Compute Q, K, V
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

    // 2. Compute Q * K^T (Scores)
    // FIX: Removed scaling factor (1/sqrt(dk)). Set scale to 1.0.
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

    // 3. Softmax
    // FIX: Removed Softmax to match Python model
    
    // 4. Compute Context * V
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

    // 5. Output Projection (Wo)
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        out_buf, Wo, Wo_b, out_buf_acc,
        hidden_dim, hidden_dim, 0, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, false, false, true, false, 0, WS);

    gemmini_fence();

    // 6. Norm + Residual
    // FIX: Buffer Aliasing. 
    // We write Norm result to 'out_buf' (temp) instead of 'out' (which might be 'input').
    tiled_norm_auto(seq_len, hidden_dim, (acc_t*)out_buf_acc, out_buf,
        ACC_SCALE_IDENTITY, LAYERNORM, WS);

    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        input,   // Original Input (Safe now)
        out_buf, // Normalized Attention Output
        resadd_out, // Final destination
        false,
        WS);

    gemmini_fence();
}


// --- FFN Layer ---
void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf, acc_t * out_buf_acc)
{
    // 1. FC1 (Linear Only)
    // FIX: Changed IGELU to NO_ACTIVATION
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        input, ff1_w, ff1_b, out_buf,
        hidden_dim, expansion_dim, expansion_dim, expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        RELU, 
        ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        true, false, false, false, false, 0, WS);

    gemmini_fence();

    /*for (int i = 0; i < seq_len * expansion_dim; i++) {
        if (out_buf[i] < 0) out_buf[i] = 0;
    }*/


    // 1.5 Manual GELU
    // FIX: Added CPU GELU Approximation
    // cpu_gelu_approx(seq_len, expansion_dim, out_buf, out_buf);

    // 2. FC2
    // FIX: Corrected Strides (Expansion -> Hidden)
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
        out_buf, ff2_w, ff2_b, out_buf_acc,
        expansion_dim, hidden_dim, hidden_dim, hidden_dim, // <--- FIX: Strides C/D are Hidden
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, 
        false, // <--- FIX: full_C should be false (overwrite)
        false, 0, WS);

    gemmini_fence();

    // 3. Norm
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, 
        out_buf,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    gemmini_fence();

    // 4. Residual Add
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        out_buf,
        input,
        out,
        false,
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
        elem_t * resadd1_buf, elem_t * resadd2_buf)
{
    const bool is_encoder = enc_out == NULL;
    
    // Calculate strides for pointer arithmetic
    int hidden_dim_compressed = hidden_dim / compression_factor;
    if (compression_factor < 0) {
        hidden_dim_compressed = hidden_dim;
    }

    // Strides
    size_t stride_W_attn = hidden_dim * hidden_dim_compressed;
    size_t stride_Wo_attn = hidden_dim_compressed * hidden_dim;
    size_t stride_b_attn = hidden_dim_compressed; 
    size_t stride_b_Wo = hidden_dim;
    size_t stride_ff1_w = hidden_dim * expansion_dim;
    size_t stride_ff2_w = expansion_dim * hidden_dim;
    size_t stride_ff1_b = expansion_dim;
    size_t stride_ff2_b = hidden_dim;

    uint64_t start = read_cycles();
    uint64_t start_enc_layer, end_enc_layer;

    const elem_t * current_input = input;

    for (int l = 0; l < num_layers; l++) {
        start_enc_layer = read_cycles();

        // 1. Self Attention
        attention(hidden_dim, expansion_dim, num_heads, seq_len, compression_factor,
            current_input, current_input,
            out, resadd1_buf,
            Wq, Wk, Wv, Wo,
            Wq_b, Wk_b, Wv_b, Wo_b,
            Q_buf, K_buf, V_buf,
            attn_buf, out_buf, out_buf_acc);

        // 2. Cross Attention (Decoder only)
        const elem_t * ffn_input = resadd1_buf;
        
        if (!is_encoder) {
            attention(hidden_dim, expansion_dim, cross_num_heads, seq_len, compression_factor,
                resadd1_buf, enc_out,
                out, resadd2_buf,
                Wq_cross, Wk_cross, Wv_cross, Wo_cross,
                Wq_cross_b, Wk_cross_b, Wv_cross_b, Wo_cross_b,
                Q_buf, K_buf, V_buf,
                attn_buf, out_buf, out_buf_acc);
            
            ffn_input = resadd2_buf;
        }

        // 3. Feed Forward
        ffn(hidden_dim, expansion_dim, seq_len,
            ffn_input,
            out,
            ff1_w, ff2_w,
            ff1_b, ff2_b,
            out_buf, out_buf_acc);

        // Advance pointers
        Wq += stride_W_attn; Wk += stride_W_attn; Wv += stride_W_attn; Wo += stride_Wo_attn;
        Wq_b += stride_b_attn; Wk_b += stride_b_attn; Wv_b += stride_b_attn; Wo_b += stride_b_Wo;

        if (!is_encoder) {
            Wq_cross += stride_W_attn; Wk_cross += stride_W_attn; Wv_cross += stride_W_attn; Wo_cross += stride_Wo_attn;
            Wq_cross_b += stride_b_attn; Wk_cross_b += stride_b_attn; Wv_cross_b += stride_b_attn; Wo_cross_b += stride_b_Wo;
        }

        ff1_w += stride_ff1_w; ff2_w += stride_ff2_w; ff1_b += stride_ff1_b; ff2_b += stride_ff2_b;

        // The output of this layer becomes the input of the next
        current_input = out;
        
        end_enc_layer = read_cycles();
        // printf("Layer %d done: %llu cycles\n", l+1, end_enc_layer - start_enc_layer);
    }

    uint64_t end = read_cycles();
    return end - start;
}