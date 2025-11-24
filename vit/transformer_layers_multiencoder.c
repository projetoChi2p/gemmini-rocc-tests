#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_nn.h"
#include <math.h>
#include <float.h> 

// --- Helper Functions (Unchanged) ---

// Fast approximation for exp(x)
float exp_approx(float x) {
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

void cpu_softmax(int rows, int cols, const elem_t * input, elem_t * output) {
    for (int i = 0; i < rows; i++) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < cols; j++) {
            float val = (float)input[i * cols + j];
            if (val > max_val) {
                max_val = val;
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = (float)input[i * cols + j] - max_val;
            float exp_val = exp_approx(val);
            sum_exp += exp_val;
        }

        for (int j = 0; j < cols; j++) {
            float val = (float)input[i * cols + j] - max_val;
            float exp_val = exp_approx(val);
            output[i * cols + j] = (elem_t)(exp_val / sum_exp);
        }
    }
}

// --- Attention and FFN Layers (Unchanged logic, just reused) ---

void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,

        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,

        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        const acc_t * Wo_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc)
{
    int hidden_dim_compressed = hidden_dim / compression_factor;
    int hidden_dim_per_head = hidden_dim_compressed / num_heads;

    if (compression_factor < 0) {
        hidden_dim_compressed = hidden_dim;
        hidden_dim_per_head = (hidden_dim_compressed / 12) * (-compression_factor);
    }

    const int qkv_matmuls_n = 3;
    for (int i = 0; i < qkv_matmuls_n; i++) {
        const elem_t * qkv_weights[] = {Wq, Wk, Wv};
        const elem_t * qkv_ins[] = {input, enc_out, enc_out};
        const acc_t * qkv_bs[] = {Wq_b, Wk_b, Wv_b}; // <-- FIX: Changed Wk_b to Wv_b
        elem_t * qkv_outs[] = {Q_buf, K_buf, V_buf};

        const elem_t * qkv_w = qkv_weights[i];
        const elem_t * qkv_in = qkv_ins[i];
        const acc_t * qkv_b = qkv_bs[i];
        elem_t * qkv_out = qkv_outs[i];

        tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
            /*A=*/ qkv_in, /*B=*/ qkv_w,
            /*D=*/ qkv_b, /*C=*/ qkv_out,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);
    }

    gemmini_fence();

    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = Q_buf + head * hidden_dim_per_head;
        const elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf + head * seq_len * seq_len;

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ true,
            false, false,
            0,
            WS);
    }

    gemmini_fence();

    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = attn_buf + head * seq_len * seq_len;
        const elem_t * B = V_buf + head * hidden_dim_per_head;
        elem_t * C = out_buf + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/seq_len, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);
    }

    gemmini_fence();

    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        /*A=*/ out_buf, /*B=*/ Wo,
        /*D=*/ Wo_b, /*C=*/ out_buf_acc,
        /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ false,
        false, /*transpose_B=*/ false,
        true, false,
        0,
        WS);

    gemmini_fence();

    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        input,
        out,
        resadd_out,
        /*relu=*/ false,
        WS);

    gemmini_fence();
}


void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf, acc_t * out_buf_acc)
{
    // out = FF1(input)
    // out = GELU(out)
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        /*A=*/ input, /*B=*/ ff1_w,
        /*D=*/ ff1_b, /*C=*/ out_buf,
        /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        WS);

    gemmini_fence();

    // out_buf_acc = FF2(out)
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
        /*A=*/ out_buf, /*B=*/ ff2_w,
        /*D=*/ ff2_b, /*C=*/ out_buf_acc,
        /*stride_A=*/expansion_dim, /*stride_B=*/hidden_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        true, false,
        0,
        WS);

    gemmini_fence();

    // out = LN(out_buf_acc)
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    gemmini_fence();

    // out = out + input
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        out,
        input,
        out,
        /*relu=*/ false,
        WS);

    gemmini_fence();
}

// --- MODIFIED Encoder/Decoder Loop ---

uint64_t encoder_decoder(
        int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads,
        int seq_len, int compression_factor, int num_layers, // Added num_layers

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

    // Weight sizes (elements)
    // Attention weights: hidden_dim * hidden_dim_compressed
    // Attention biases: hidden_dim_compressed (or hidden_dim for Wo_b)
    size_t stride_W_attn = hidden_dim * hidden_dim_compressed;
    size_t stride_Wo_attn = hidden_dim_compressed * hidden_dim;
    size_t stride_b_attn = hidden_dim_compressed; 
    size_t stride_b_Wo = hidden_dim;

    // FFN weights: hidden_dim * expansion_dim
    size_t stride_ff1_w = hidden_dim * expansion_dim;
    size_t stride_ff2_w = expansion_dim * hidden_dim;
    size_t stride_ff1_b = expansion_dim;
    size_t stride_ff2_b = hidden_dim;

    uint64_t start = read_cycles();
    uint64_t start_enc_layer, end_enc_layer;

    // Pointer to the input of the current layer
    const elem_t * current_input = input;

    for (int l = 0; l < num_layers; l++) {
        start_enc_layer = read_cycles();

        // 1. Self Attention
        // Input: current_input
        // Output: resadd1_buf (Intermediate result of Attn + Residual)
        attention(hidden_dim, expansion_dim, num_heads, seq_len, compression_factor,
            current_input, current_input, // self attention: enc_out = input
            out, resadd1_buf,
            Wq, Wk, Wv, Wo,
            Wq_b, Wk_b, Wv_b, Wo_b,
            Q_buf, K_buf, V_buf,
            attn_buf, out_buf, out_buf_acc);

        // 2. Cross Attention (Decoder only)
        // Input: resadd1_buf (from self attn)
        // Output: resadd2_buf (or skip)
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
        // Input: ffn_input (resadd1 or resadd2)
        // Output: out (Final result of this layer)
        ffn(hidden_dim, expansion_dim, seq_len,
            ffn_input,
            out,
            ff1_w, ff2_w,
            ff1_b, ff2_b,
            out_buf, out_buf_acc);

        // Advance pointers for the next layer
        Wq += stride_W_attn;
        Wk += stride_W_attn;
        Wv += stride_W_attn;
        Wo += stride_Wo_attn;

        Wq_b += stride_b_attn;
        Wk_b += stride_b_attn;
        Wv_b += stride_b_attn;
        Wo_b += stride_b_Wo;

        if (!is_encoder) {
            Wq_cross += stride_W_attn;
            Wk_cross += stride_W_attn;
            Wv_cross += stride_W_attn;
            Wo_cross += stride_Wo_attn;

            Wq_cross_b += stride_b_attn;
            Wk_cross_b += stride_b_attn;
            Wv_cross_b += stride_b_attn;
            Wo_cross_b += stride_b_Wo;
        }

        ff1_w += stride_ff1_w;
        ff2_w += stride_ff2_w;
        ff1_b += stride_ff1_b;
        ff2_b += stride_ff2_b;

        // The output of this layer becomes the input of the next
        current_input = out;
        end_enc_layer = read_cycles();

        printf("Completed layer %d/%d\n", l + 1, num_layers);
        printf("  Cycles for this layer: %llu\n", end_enc_layer - start_enc_layer);
    }

    uint64_t end = read_cycles();

    return end - start;
}