// File: transformer_layers_debug.c

#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_nn.h"
#include <math.h>
#include <float.h> 

// --- Helper Functions (Unchanged) ---

// --- 1. Self-contained Exponential Approximation ---
// (Same as the one in your transformer_layers.c)
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

// --- 2. GELU Approximation (Sigmoid Method) ---
// Formula: x * sigmoid(1.702 * x) = x / (1 + exp(-1.702 * x))
// This avoids erff() entirely.
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
        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b, const acc_t * Wo_b,
        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc,
        
        // NEW: Debug pointers (pass NULL if not checking)
        const elem_t * expected_Q, const elem_t * expected_K, const elem_t * expected_V,
        const elem_t * expected_scores, const elem_t * expected_probs)
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

    // >>> DEBUG 1: Verify Q, K, V <<<
    if (expected_Q != NULL) {
        printf("\n  --- Deep Debug: Q/K/V ---\n");
        verify_matrix("L0 Q Matrix", seq_len, hidden_dim, Q_buf, expected_Q);
        verify_matrix("L0 K Matrix", seq_len, hidden_dim, K_buf, expected_K);
        verify_matrix("L0 V Matrix", seq_len, hidden_dim, V_buf, expected_V);
    }

    // 2. Compute Q * K^T (Scores)
    // Note: Gemmini Accumulation is usually integer. Scaling factor 1/sqrt(dk) 
    // must be applied here via the `scale` parameter of matmul.
    float d_k = (float)hidden_dim_per_head;
    //float scale_factor = 1.0f / sqrtf(d_k);
    acc_scale_t gemmini_scale = (acc_scale_t)ACC_SCALE_IDENTITY;

    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = Q_buf + head * hidden_dim_per_head;
        const elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf + head * seq_len * seq_len;

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            A, B, NULL, C,
            hidden_dim, hidden_dim, 0, seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, 
            gemmini_scale, // Apply 1/sqrt(dk) here!
            0, false, false, true /* Transpose B */, false, false, 0, WS);
    }
    gemmini_fence();

    // >>> DEBUG 2: Verify Scores (Before Softmax) <<<
    if (expected_scores != NULL) {
        verify_matrix("L0 Scores (Scaled)", seq_len, seq_len, attn_buf, expected_scores);
    }

    // 3. Softmax (Crucial Step often missed in HW offload)
    // Note: attn_buf is overwritten with probabilities
    /*for (int head = 0; head < num_heads; head++) {
        elem_t * C = attn_buf + head * seq_len * seq_len;
        cpu_softmax(seq_len, seq_len, C, C);
    }*/

    // >>> DEBUG 3: Verify Probs (After Softmax) <<<
    if (expected_probs != NULL) {
        verify_matrix("L0 Probs (Softmax)", seq_len, seq_len, attn_buf, expected_probs);
    }

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

    // 6a. Norm 
    tiled_norm_auto(seq_len, hidden_dim, 
        (acc_t*)out_buf_acc, 
        out_buf, // <--- TARGET IS NOW INTERNAL BUFFER
        ACC_SCALE_IDENTITY, LAYERNORM, WS);
    
    gemmini_fence(); // Ensure Norm is done

    // 6b. Residual Add
    // Add 'input' (original residual) + 'out_buf' (normalized attention output)
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        input,    // A: Original Input (Safe now)
        out_buf,  // B: Normalized Attention Result
        resadd_out, // C: Final Output buffer
        false, WS);
    
    gemmini_fence();
}


void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf, acc_t * out_buf_acc,
        
        // NEW: Debug Pointers
        const elem_t * expected_fc1,
        const elem_t * expected_fc2)
{
    // out = FF1(input)
    // out = GELU(out)
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        input, ff1_w, ff1_b, out_buf,
        hidden_dim, expansion_dim, expansion_dim, expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, // <--- CHANGED FROM IGELU
        ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        true, false, false, false, false, 0, WS);

    gemmini_fence();

    cpu_gelu_approx(seq_len, expansion_dim, out_buf, out_buf);

    // >>> DEBUG FFN 1: Verify FC1 + GELU Output <<<
    if (expected_fc1 != NULL) {
        verify_matrix("L0 FFN FC1+GELU", seq_len, expansion_dim, out_buf, expected_fc1);
    }

    // out_buf_acc = FF2(out)
    // 2. FC2
    // out_buf_acc = out_buf @ ff2_w + ff2_b
    // Dimensions: (Seq, Exp) @ (Exp, Hidden) -> (Seq, Hidden)
    tiled_matmul_auto(
        seq_len,        // dim_I (Rows of A)
        hidden_dim,     // dim_J (Cols of B / Width of Output)
        expansion_dim,  // dim_K (Cols of A / Rows of B)
        
        out_buf,        // A
        ff2_w,          // B
        ff2_b,          // D
        out_buf_acc,    // C
        
        // --- STRIDES ---
        expansion_dim,  // stride_A (Input width)
        hidden_dim,     // stride_B (Weight width)
        hidden_dim,     // stride_D (Bias width)   <-- WAS expansion_dim (FIXED)
        hidden_dim,     // stride_C (Output width) <-- WAS expansion_dim (FIXED)
        
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 
        ACC_SCALE_IDENTITY, 
        0,
        true,           // repeating_bias
        false,          // transpose_B
        false,          // transpose_A
        false,          // full_C (accumulate) - Make sure this is intended (usually false for overwrite)
        false,          // low_D
        0,
        WS
    );

    gemmini_fence();

    // >>> DEBUG FFN 2: Verify FC2 Output <<<
    if (expected_fc2 != NULL) {
        // Note: out_buf_acc might be acc_t (float or int32), expected is elem_t.
        // If compilation fails here due to pointer types, cast out_buf_acc to (elem_t*)
        // assuming they are memory-compatible in your current configuration.
        verify_matrix("L0 FFN FC2", seq_len, hidden_dim, (elem_t*)out_buf_acc, expected_fc2);
    }

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
        
        // NEW: Debug pointers (pass NULL if not debugging)
        const elem_t ** expected_attn_outputs,
        const elem_t ** expected_ffn_outputs,

        const elem_t * debug_l0_q, const elem_t * debug_l0_k, const elem_t * debug_l0_v,
        const elem_t * debug_l0_scores, const elem_t * debug_l0_probs,
        
        // NEW: FFN Internal Debug Pointers
        const elem_t * debug_l0_ffn_fc1,
        const elem_t * debug_l0_ffn_fc2)
{
    // ... [Keep variable setup unchanged] ...
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
        // 1. Self Attention
        // Determine if we debug this layer
        const elem_t * d_q = (l == 0) ? debug_l0_q : NULL;
        const elem_t * d_k = (l == 0) ? debug_l0_k : NULL;
        const elem_t * d_v = (l == 0) ? debug_l0_v : NULL;
        const elem_t * d_s = (l == 0) ? debug_l0_scores : NULL;
        const elem_t * d_p = (l == 0) ? debug_l0_probs : NULL;

        const elem_t * d_f1 = (l == 0) ? debug_l0_ffn_fc1 : NULL;
        const elem_t * d_f2 = (l == 0) ? debug_l0_ffn_fc2 : NULL;

        attention(hidden_dim, expansion_dim, num_heads, seq_len, compression_factor,
            current_input, current_input,
            out, resadd1_buf,
            Wq, Wk, Wv, Wo,
            Wq_b, Wk_b, Wv_b, Wo_b,
            Q_buf, K_buf, V_buf,
            attn_buf, out_buf, out_buf_acc,
            // PASS DEBUG POINTERS
            d_q, d_k, d_v, d_s, d_p);
            
        // --- VERIFY STEP 2A: ATTENTION OUTPUT ---
        if (expected_attn_outputs != NULL) {
            char buf[32];
            sprintf(buf, "Layer %d Attention", l);
            // resadd1_buf contains the final output of the Attention Block (Norm(Attn) + Residual)
            verify_matrix(buf, seq_len, hidden_dim, resadd1_buf, expected_attn_outputs[l]);
        }

        // 2. Cross Attention (Skip for Encoder)
        const elem_t * ffn_input = resadd1_buf;

        // 3. Feed Forward
        ffn(hidden_dim, expansion_dim, seq_len,
            ffn_input,
            out,
            ff1_w, ff2_w,
            ff1_b, ff2_b,
            out_buf, out_buf_acc,
            d_f1, d_f2);

        // --- VERIFY STEP 2B: FFN OUTPUT ---
        if (expected_ffn_outputs != NULL) {
            char buf[32];
            sprintf(buf, "Layer %d FFN", l);
            // 'out' contains the final output of the FFN Block (Norm(FFN) + Residual)
            verify_matrix(buf, seq_len, hidden_dim, out, expected_ffn_outputs[l]);
        }

        // Pointer arithmetic (Unchanged)
        Wq += stride_W_attn;
        Wk += stride_W_attn;
        Wv += stride_W_attn;
        Wo += stride_Wo_attn;

        Wq_b += stride_b_attn;
        Wk_b += stride_b_attn;
        Wv_b += stride_b_attn;
        Wo_b += stride_b_Wo;
        ff1_w += stride_ff1_w;
        ff2_w += stride_ff2_w;
        ff1_b += stride_ff1_b;
        ff2_b += stride_ff2_b;

        current_input = out;
    }

    uint64_t end = read_cycles();
    return end - start;
}