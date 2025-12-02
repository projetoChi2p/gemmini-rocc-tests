#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "include/gemmini_nn.h"

void compute_patch_embeddings(
    int patch_seq_len,   
    int hidden_dim, 
    int patch_dim,       // NOW: This must equal (Channels * Patch_Size * Patch_Size)
    const elem_t * current_patches, // NOW: Must be the output of extract_patches_from_image
    const elem_t * patch_embed_w,
    const acc_t * patch_embed_b,
    const elem_t * pos_embed_data, 
    const elem_t * cls_token_data, // NOTE: If your model doesn't use CLS token, pass NULL and handle logic
    elem_t * temp_patch_buf,       
    elem_t * final_input_buf
    
    #ifdef DEBUG
    // Debug Pointer
    ,const elem_t * expected_embedding_output
    #endif
    );

#endif