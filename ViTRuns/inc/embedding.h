#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "include/gemmini_nn.h"

void compute_patch_embeddings(
    int seq_len, int hidden_dim, int patch_dim,
    const elem_t * current_patches,
    const elem_t * patch_embed_w,
    const acc_t * patch_embed_b,
    const elem_t * pos_embed_data,
    elem_t * embed_out_buf,     // Temporary buffer
    elem_t * final_input_buf    // Final output to encoder
);

#endif