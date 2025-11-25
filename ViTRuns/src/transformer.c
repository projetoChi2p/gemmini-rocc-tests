#include "../inc/transformer.h"
#include "../inc/attention.h"
#include "../inc/ffn.h"
#include "include/gemmini.h"
#include <stdio.h> // for NULL

uint64_t encoder_decoder(
        int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads,
        int seq_len, int compression_factor, int num_layers,
        const elem_t * input, const elem_t * enc_out, elem_t * out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,
        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b, const acc_t * Wo_b,
        const acc_t * Wq_cross_b, const acc_t * Wk_cross_b, const acc_t * Wv_cross_b, const acc_t * Wo_cross_b,
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
    }

    uint64_t end = read_cycles();
    return end - start;
}