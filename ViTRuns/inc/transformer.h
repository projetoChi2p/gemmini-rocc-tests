#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "include/gemmini_nn.h"
#include <stdint.h>

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
        elem_t * resadd1_buf, elem_t * resadd2_buf);

#endif