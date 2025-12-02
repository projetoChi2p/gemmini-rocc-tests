// File: transformer_layers_debug.c (Embedding Update)

#include "include/gemmini_nn.h"
#include <string.h> // For memcpy

// ==========================================
// 3. EMBEDDING MODULE
// ==========================================

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
    )      
{
    // 1. Project Patches (Input -> Temp Buffer)
    // The Gemmini MatMul handles the size increase automatically via 'patch_dim'
    tiled_matmul_auto(patch_seq_len, hidden_dim, patch_dim,
        current_patches, patch_embed_w, patch_embed_b, temp_patch_buf,
        patch_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, false, false, 0, WS);
    
    gemmini_fence();

    // 2. Construct Sequence logic
    // NOTE: Your previous main() example implies NO CLS token (SEQ_LEN matches patches).
    // If your Python model uses a CLS token, you need to adjust SEQ_LEN in main to be +1.
    // Based on your main.c provided earlier, it seems you are mapping directly.
    
    // Scenario A: WITH CLS Token (Standard ViT)
    if (cls_token_data != NULL) {
        size_t row_size = hidden_dim * sizeof(elem_t);
        // A. Copy CLS token to Row 0
        memcpy(final_input_buf, cls_token_data, row_size);
        // B. Copy Patches to Rows 1..N
        memcpy(final_input_buf + hidden_dim, temp_patch_buf, patch_seq_len * row_size);
    } 
    // Scenario B: WITHOUT CLS Token (Simple/Mini ViT often skips this)
    else {
        // Just copy the projected patches directly to final buffer
        memcpy(final_input_buf, temp_patch_buf, patch_seq_len * hidden_dim * sizeof(elem_t));
    }

    // 3. Add Position Embeddings
    // Adjust total_seq_len based on whether CLS was added
    int total_seq_len = (cls_token_data != NULL) ? (patch_seq_len + 1) : patch_seq_len;
    
    tiled_resadd_auto(total_seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        final_input_buf, pos_embed_data, final_input_buf,
        false, WS);
        
    gemmini_fence();

    // DEBUG: Check Final Embedding Output
    #ifdef DEBUG
    if (expected_embedding_output != NULL) {
        print_error_histogram("Embedding + Pos Output", total_seq_len, hidden_dim, final_input_buf, expected_embedding_output);
    }
    #endif
}
