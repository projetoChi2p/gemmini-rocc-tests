#include "../include/gemmini.h"
#include <string.h> // For memcpy

void compute_patch_embeddings(
    int patch_seq_len,   // 16
    int hidden_dim, 
    int patch_dim,
    const elem_t * current_patches,
    const elem_t * patch_embed_w,
    const acc_t * patch_embed_b,
    const elem_t * pos_embed_data, // Size: (16+1) * Hidden
    const elem_t * cls_token_data, // Size: 1 * Hidden
    elem_t * temp_patch_buf,       // Size: 16 * Hidden
    elem_t * final_input_buf)      // Size: 17 * Hidden
{
    // 1. Project Patches (Input -> Temp Buffer)
    // Dimensions: [16, Hidden]
    tiled_matmul_auto(patch_seq_len, hidden_dim, patch_dim,
        current_patches, patch_embed_w, patch_embed_b, temp_patch_buf,
        patch_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, false, false, 0, WS);

    // 2. Construct Sequence: [CLS] + [Patches]
    size_t row_size = hidden_dim * sizeof(elem_t);

    // A. Copy CLS token to Row 0
    memcpy(final_input_buf, cls_token_data, row_size);

    // B. Copy Patches to Rows 1..16
    // Note: 'final_input_buf + hidden_dim' advances the pointer by 1 row
    memcpy(final_input_buf + hidden_dim, temp_patch_buf, patch_seq_len * row_size);

    // 3. Add Position Embeddings
    // Applies to the whole sequence (17 rows)
    int total_seq_len = patch_seq_len + 1;
    
    tiled_resadd_auto(total_seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        final_input_buf, pos_embed_data, final_input_buf,
        false, WS);
}