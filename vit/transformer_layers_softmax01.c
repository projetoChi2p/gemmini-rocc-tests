// transformer_layers.c
#include "includes/transformer_layers.h" // Inclui nosso novo cabeçalho

// Includes originais necessários para a implementação
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_nn.h"

// Includes for new FP32 functions
#include <math.h>
#include <float.h> 

// --------------------------------------------------------------------------
// High-Precision Piecewise Exponential Approximation (FP32)
// --------------------------------------------------------------------------
// Strategy: Range Reduction.
// We use the identity: e^x = e^(k + dx) = e^k * e^dx
// 1. We find the integer 'k' closest to x.
// 2. We look up e^k from a hardcoded table.
// 3. We calculate e^dx using a 5th-order Taylor series (very accurate for small dx).
// --------------------------------------------------------------------------

static inline float taylor_5_horner(float x) {
    // Taylor Series for e^x around 0: 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120
    // Implemented using Horner's Method for speed and stability:
    // (((((c5 * x + c4) * x + c3) * x + c2) * x + c1) * x + c0)
    
    const float c0 = 1.0f;
    const float c1 = 1.0f;
    const float c2 = 0.5f;
    const float c3 = 0.166666667f;
    const float c4 = 0.041666667f;
    const float c5 = 0.008333333f;

    return x * (x * (x * (x * (x * c5 + c4) + c3) + c2) + c1) + c0;
}

float exp_high_prec(float x) {
    // Pre-calculated constants for e^k
    const float E_POS_2  = 7.389056099f;
    const float E_POS_1  = 2.718281828f;
    const float E_0      = 1.0f;
    const float E_NEG_1  = 0.367879441f;
    const float E_NEG_2  = 0.135335283f;
    const float E_NEG_3  = 0.049787068f;
    const float E_NEG_4  = 0.018315639f;
    const float E_NEG_5  = 0.006737947f;
    const float E_NEG_6  = 0.002478752f;
    const float E_NEG_7  = 0.000911882f;
    const float E_NEG_8  = 0.000335463f;
    const float E_NEG_9  = 0.000123410f;
    const float E_NEG_10 = 0.000045400f;

    float center;
    float scale;

    // Range reduction logic (The "Large Piecewise" part)
    // We select the integer center closest to x to minimize |dx|
    if (x > -0.5f) {
        if (x > 1.5f)       { center = 2.0f; scale = E_POS_2; } 
        else if (x > 0.5f)  { center = 1.0f; scale = E_POS_1; }
        else                { center = 0.0f; scale = E_0;     }
    } 
    else if (x > -5.5f) {
        if (x > -1.5f)      { center = -1.0f; scale = E_NEG_1; }
        else if (x > -2.5f) { center = -2.0f; scale = E_NEG_2; }
        else if (x > -3.5f) { center = -3.0f; scale = E_NEG_3; }
        else if (x > -4.5f) { center = -4.0f; scale = E_NEG_4; }
        else                { center = -5.0f; scale = E_NEG_5; }
    } 
    else {
        if (x > -6.5f)      { center = -6.0f; scale = E_NEG_6; }
        else if (x > -7.5f) { center = -7.0f; scale = E_NEG_7; }
        else if (x > -8.5f) { center = -8.0f; scale = E_NEG_8; }
        else if (x > -9.5f) { center = -9.0f; scale = E_NEG_9; }
        else if (x > -10.5f){ center = -10.0f; scale = E_NEG_10;}
        else { 
            return 0.0f; 
        }
    }

    float dx = x - center; 
    float poly_part = taylor_5_horner(dx);
    return scale * poly_part;
}

// --------------------------------------------------------------------------
// Softmax using the High-Precision Exp
// --------------------------------------------------------------------------
void cpu_softmax(int rows, int cols, const elem_t * input, elem_t * output) {
    for (int i = 0; i < rows; i++) {
        // 1. Find max for numerical stability
        float max_val = -FLT_MAX;
        for (int j = 0; j < cols; j++) {
            float val = (float)input[i * cols + j];
            if (val > max_val) {
                max_val = val;
            }
        }

        // 2. Compute exp(x - max) and sum
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = (float)input[i * cols + j] - max_val;
            sum_exp += exp_high_prec(val);
        }

        // 3. Normalize and Write Output
        float inv_sum = 1.0f / sum_exp; // Calculate inverse for multiplication
        for (int j = 0; j < cols; j++) {
            float val = (float)input[i * cols + j] - max_val;
            output[i * cols + j] = (elem_t)(exp_high_prec(val) * inv_sum);
        }
    }
}


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

    // 1. Calculate Q, K, V
    const int qkv_matmuls_n = 3;
    for (int i = 0; i < qkv_matmuls_n; i++) {
        const elem_t * qkv_weights[] = {Wq, Wk, Wv};
        const elem_t * qkv_ins[] = {input, enc_out, enc_out};
        const acc_t * qkv_bs[] = {Wq_b, Wk_b, Wk_b};
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

    // 2. Calculate Raw Attention Scores (attn = Q * K^T)
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

    // --- 3. SCALING, ANALYSIS, AND SOFTMAX ---
    
    // Standard Attention Scaling Factor
    float inv_scale = 1.0f / sqrtf((float)hidden_dim_per_head);

    for (int head = 0; head < num_heads; head++) {
        elem_t * C = attn_buf + head * seq_len * seq_len;
        int total_elements = seq_len * seq_len;

        // --- STEP A: Apply Scaling ---
        for (int i = 0; i < total_elements; i++) {
            C[i] = C[i] * inv_scale;
        }

        // --- STEP B: DEBUG ANALYSIS (Inserted Here) ---
        // We analyze the data AFTER scaling but BEFORE Softmax
        // to see exactly what 'exp_approx' will be receiving.
        
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        float sum_val = 0.0f;
        
        // Buckets matching your exp_approx regions
        int bin_neg_huge   = 0; // < -5
        int bin_neg_med    = 0; // -5 to -2
        int bin_center     = 0; // -2 to +2 (High Precision)
        int bin_pos_med    = 0; // +2 to +5
        int bin_pos_huge   = 0; // > +5

        for (int i = 0; i < total_elements; i++) {
            float val = (float)C[i];
            
            // Range Stats
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum_val += val;

            // Histogram Stats
            if (val < -5.0f)      bin_neg_huge++;
            else if (val < -2.0f) bin_neg_med++;
            else if (val < 2.0f)  bin_center++;
            else if (val < 5.0f)  bin_pos_med++;
            else                  bin_pos_huge++;
        }

        float mean_val = sum_val / total_elements;

        // Print using your integer-based format
        printf("=== Head %d Analysis ===\n", head);
        printf("  Range: [%de-3, %de-3]\n", (int)(min_val * 1000), (int)(max_val * 1000));
        printf("  Mean:  %de-3\n", (int)(mean_val * 1000));
        printf("  Distribution:\n");
        printf("    < -5.0:   %d\n", bin_neg_huge);
        printf("    -5 to -2: %d\n", bin_neg_med);
        printf("    -2 to +2: %d (High Precision Zone)\n", bin_center);
        printf("    +2 to +5: %d\n", bin_pos_med);
        printf("    > +5.0:   %d\n", bin_pos_huge);
        printf("========================\n");
        // --- END ANALYSIS ---

        // --- STEP C: Run Softmax ---
        cpu_softmax(seq_len, seq_len, C, C);
    }

    // 4. Calculate Final Attention Output (out = attn * V)
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

    // 5. Projection (out_buf_acc = out_buf * Wo)
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

    // 6. Layer Norm
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    // 7. Residual Add
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

uint64_t encoder_decoder(
        int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads,
        int seq_len, int compression_factor,

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

    uint64_t start = read_cycles();

    attention(hidden_dim, expansion_dim, num_heads, seq_len, compression_factor,
        input, input,
        out, resadd1_buf,
        Wq, Wk, Wv, Wo,

        Wq_b, Wk_b, Wv_b,
        Wo_b,

        Q_buf, K_buf, V_buf,
        attn_buf, out_buf, out_buf_acc);

    if (!is_encoder) {
        attention(hidden_dim, expansion_dim, cross_num_heads, seq_len, compression_factor,
            resadd1_buf, enc_out,
            out, resadd2_buf,
            Wq_cross, Wk_cross, Wv_cross, Wo_cross,

            Wq_cross_b, Wk_cross_b, Wv_cross_b,
            Wo_cross_b,

            Q_buf, K_buf, V_buf,
            attn_buf, out_buf, out_buf_acc);
    }

    ffn(hidden_dim, expansion_dim, seq_len,
        is_encoder ? resadd1_buf : resadd2_buf,
        out,
        ff1_w, ff2_w,
        ff1_b, ff2_b,
        out_buf, out_buf_acc);

    uint64_t end = read_cycles();

    return end - start;
}