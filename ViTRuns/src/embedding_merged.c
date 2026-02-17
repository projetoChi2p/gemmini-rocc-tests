#include "include/gemmini_nn.h"
#include <stdbool.h>
#include <string.h> // For memcpy

static void reorder_patches_hwc_to_chw(int patch_seq_len, int patch_size, int channels,
                                       const elem_t *src, elem_t *dst) {
    int patch_area = patch_size * patch_size;
    int patch_dim = patch_area * channels;

    for (int p = 0; p < patch_seq_len; p++) {
        const elem_t *src_patch = src + (p * patch_dim);
        elem_t *dst_patch = dst + (p * patch_dim);

        for (int c = 0; c < channels; c++) {
            elem_t *dst_c = dst_patch + (c * patch_area);

            for (int y = 0; y < patch_size; y++) {
                const elem_t *src_row = src_patch + (y * patch_size * channels) + c;

                for (int x = 0; x < patch_size; x++) {
                    dst_c[y * patch_size + x] = src_row[x * channels];
                }
            }
        }
    }
}

// ==========================================
// UNIFIED EMBEDDING MODULE
// Supports: FP32, Quantized (Int8), CLS, Distillation
// ==========================================

void compute_patch_embeddings(
    // 1. Dimensions
    int patch_seq_len,   
    int hidden_dim, 
    int patch_dim,
    int patch_size,
    int channels,
    bool input_is_hwc,
    
    // 2. Input & Weights
    const elem_t * current_patches, 
    const elem_t * patch_embed_w,
    const acc_t * patch_embed_b,
    
    // 3. Quantization Scale (Only used if QUANTIZED is defined)
    #ifdef QUANTIZED
    float scale_embed,              
    #endif

    // 4. Token Data
    const elem_t * pos_embed_data, 
    const elem_t * cls_token_data,
    #ifdef DISTILLATION
    const elem_t * dist_token_data,
    #endif

    // 5. Buffers
    elem_t * patch_reorder_buf,
    elem_t * temp_patch_buf,
    elem_t * final_input_buf
    )      
{
    // --- STEP 1: Project Patches (Input -> Temp Buffer) ---
    // Calculate the correct accumulation scale
    acc_scale_t matmul_scale;
    
    #ifdef QUANTIZED
        matmul_scale = (acc_scale_t)scale_embed;
    #else
        matmul_scale = ACC_SCALE_IDENTITY; // Typically 1.0f for FP32
    #endif

    const elem_t *patches_for_matmul = current_patches;

    if (input_is_hwc) {
        reorder_patches_hwc_to_chw(patch_seq_len, patch_size, channels,
                                   current_patches, patch_reorder_buf);
        patches_for_matmul = patch_reorder_buf;
    }

    // The Gemmini MatMul handles the size increase automatically via 'patch_dim'
    tiled_matmul_auto(patch_seq_len, hidden_dim, patch_dim,
        patches_for_matmul, patch_embed_w, patch_embed_b, temp_patch_buf,
        patch_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, matmul_scale, 0,
        true, false, false, false, false, 0, WS);
        
    gemmini_fence();

    // --- STEP 2: Construct Sequence [CLS, (DIST), Patches...] ---
    // We use a flexible pointer strategy to support any combination of tokens
    
    size_t row_size_bytes = hidden_dim * sizeof(elem_t);
    int current_token_count = 0; // Tracks where we are writing in the destination buffer

    // A. Copy CLS Token (Index 0)
    if (cls_token_data != NULL) {
        // Copy CLS to the start of final buffer
        memcpy(final_input_buf + (current_token_count * hidden_dim), cls_token_data, row_size_bytes);
        current_token_count++;
    }

    // B. Copy Distillation Token (Optional)
    #ifdef DISTILLATION
    if (dist_token_data != NULL) {
        memcpy(final_input_buf + (current_token_count * hidden_dim), dist_token_data, row_size_bytes);
        current_token_count++;
    } 
    #endif

    // C. Copy Projected Patches
    // We append the patches after whatever tokens (0, 1, or 2) were added above
    // temp_patch_buf contains [patch_seq_len x hidden_dim] data
    memcpy(final_input_buf + (current_token_count * hidden_dim), 
           temp_patch_buf, 
           patch_seq_len * row_size_bytes);

    // --- STEP 3: Add Position Embeddings ---
    // Total sequence length = (Number of Special Tokens) + (Number of Patches)
    int total_seq_len = current_token_count + patch_seq_len;
    
    tiled_resadd_auto(total_seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        final_input_buf, pos_embed_data, final_input_buf,
        false, WS);
        
    gemmini_fence();
}