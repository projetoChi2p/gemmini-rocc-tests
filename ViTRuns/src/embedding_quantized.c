#include "include/gemmini_nn.h"
#include <string.h> // For memcpy

// ==========================================
// 3. EMBEDDING MODULE (QUANTIZED)
// ==========================================


// --- DUT: Function Under Test (Provided by you) ---
void compute_patch_embeddings_quantized(
    int patch_seq_len,   
    int hidden_dim, 
    int patch_dim,       
    const elem_t * current_patches, 
    const elem_t * patch_embed_w,
    const acc_t * patch_embed_b,
    float scale_embed,              
    const elem_t * pos_embed_data, 
    const elem_t * cls_token_data,  
    elem_t * temp_patch_buf,       
    elem_t * final_input_buf
    )      
{
    // 1. Project Patches (Input -> Temp Buffer)
    tiled_matmul_auto(patch_seq_len, hidden_dim, patch_dim,
        current_patches, patch_embed_w, patch_embed_b, temp_patch_buf,
        patch_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 
        (acc_scale_t)scale_embed, 
        0,
        true, false, false, false, false, 0, WS);
    
    gemmini_fence();

    // 2. Construct Sequence (CLS Concatenation)
    if (cls_token_data != NULL) {
        size_t row_size = hidden_dim * sizeof(elem_t);
        
        // A. Copy CLS token to Row 0
        memcpy(final_input_buf, cls_token_data, row_size);
        
        // B. Copy Projected Patches to Rows 1..N
        // Note: Pointer arithmetic 'final_input_buf + hidden_dim' advances by hidden_dim elements
        memcpy(final_input_buf + hidden_dim, temp_patch_buf, patch_seq_len * row_size);
    } 

    // 3. Add Position Embeddings
    int total_seq_len = patch_seq_len + 1;
    
    tiled_resadd_auto(total_seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        final_input_buf, pos_embed_data, final_input_buf,
        false, WS);
        
    gemmini_fence();
}