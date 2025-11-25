//#include "../inc/embedding.h"
#include "include/gemmini.h"
#include <stdbool.h>

void compute_patch_embeddings(
    int seq_len, int hidden_dim, int patch_dim,
    const elem_t * current_patches,
    const elem_t * patch_embed_w,
    const acc_t * patch_embed_b,
    const elem_t * pos_embed_data,
    elem_t * embed_out_buf,
    elem_t * final_input_buf) 
{
    // 1. Patch Projection (Convolution via Matmul)
    tiled_matmul_auto(seq_len, hidden_dim, patch_dim,
        current_patches, patch_embed_w, patch_embed_b, embed_out_buf,
        patch_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, false, false, 0, WS);

    // 2. Add Position Embeddings
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        embed_out_buf, pos_embed_data, final_input_buf,
        false, WS);
}