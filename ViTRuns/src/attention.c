//#include "attention.h"
#include "include/gemmini.h"
#include <stddef.h> // for NULL

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

    // 3. Compute Context * V
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

    // 4. Output Projection (Wo)
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        out_buf, Wo, Wo_b, out_buf_acc,
        hidden_dim, hidden_dim, 0, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, false, false, true, false, 0, WS);

    gemmini_fence();

    // 5. Norm + Residual
    tiled_norm_auto(seq_len, hidden_dim, (acc_t*)out_buf_acc, out_buf,
        ACC_SCALE_IDENTITY, LAYERNORM, WS);

    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        input, out_buf, resadd_out,
        false, WS);

    gemmini_fence();
}