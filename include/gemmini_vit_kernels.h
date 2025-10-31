#ifndef VIT_KERNELS_H
#define VIT_KERNELS_H

/*
 * vit_kernels.h
 *
 * This header declares the high-level functions for each
 * block of the Vision Transformer.
 *
 * It ASSUMES that the following have been defined *before*
 * it is included:
 * 1. Base headers like <stdint.h>
 * 2. Gemmini types (elem_t, acc_t, full_t, ACC_T, uint64_t)
 */

// Include the global model parameters
#include "include/gemmini_vit_params.h"
#include "include/gemmini_testutils.h"

// --- 0. From img2patch.c ---
void run_patch_extraction(
    const elem_t image_in[IMG_H][IMG_W][IMG_C], 
    elem_t patches_out[NUM_PATCHES][PATCH_DIM]
);

// --- 1. From patch_embedding.c ---
void run_patch_embedding(
    const elem_t patches_in[NUM_PATCHES][PATCH_DIM], 
    const elem_t weights_in[PATCH_DIM][HIDDEN_DIM], 
    const elem_t bias_in[HIDDEN_DIM], // Changed type
    elem_t embed_out[NUM_PATCHES][HIDDEN_DIM],
    uint64_t* gemmini_cycles
) ;

void add_cls_and_pos_embed(
    const elem_t patch_embeds[NUM_PATCHES][HIDDEN_DIM],
    const elem_t cls_token_weights[1][HIDDEN_DIM],
    const elem_t pos_embed_weights[SEQ_LEN][HIDDEN_DIM],
    elem_t encoder_input_out[SEQ_LEN][HIDDEN_DIM]
);

// --- 2. From transformer.c ---
// (This prototype is unchanged)
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
    elem_t * resadd1_buf, elem_t * resadd2_buf
);

// --- 3. From classification_head.c ---
void run_classification_head(
    const elem_t token_in[1][HIDDEN_DIM], 
    const elem_t weights_in[HIDDEN_DIM][NUM_CLASSES], 
    const elem_t bias_in[NUM_CLASSES], // Changed type
    elem_t logits_out[1][NUM_CLASSES],
    uint64_t* gemmini_cycles
);

#endif // VIT_KERNELS_H