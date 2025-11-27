#ifndef ATTENTION_H
#define ATTENTION_H

#include "include/gemmini_nn.h"

void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,
        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b, const acc_t * Wo_b,
        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc, float score_scaling_factor);

#endif