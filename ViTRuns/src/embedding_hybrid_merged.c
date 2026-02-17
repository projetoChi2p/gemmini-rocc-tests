#include "include/gemmini.h"
#include <string.h> // for memcpy

void compute_hybrid_embeddings(
    // 1. Dimensions
    int batch_size, int in_channels,
    int in_row_dim, int in_col_dim,
    int out_channels,            // Hidden Dim (e.g., 192)
    int out_row_dim, int out_col_dim,
    int stride, int padding, int kernel_dim,
    
    // 2. Weights & Data
    const elem_t * input,
    const elem_t * weights_mat,  // Flattened weights
    const acc_t * bias,
    
    // 3. Scale (Quantized Only)
    #ifdef QUANTIZED
        float scale_conv,              
    #endif

    // 4. Tokens
    const elem_t * pos_embed_data, 
    const elem_t * cls_token_data,
    #ifdef DISTILLATION
        const elem_t * dist_token_data, 
    #endif

    // 5. Buffers
    elem_t * conv_output_buf,    // Intermediate Buffer (Output of Conv)
    elem_t * final_input_buf     // Final Sequence Buffer
) {
    
    // --- Step 0: Determine Conv Scale ---
    acc_scale_t internal_conv_scale;
    #ifdef QUANTIZED
        internal_conv_scale = (acc_scale_t)scale_conv;
    #else
        internal_conv_scale = ACC_SCALE_IDENTITY; // Typically 1.0f
    #endif

    // --- Step 1: Hardware Accelerated Convolutional Stem ---
    tiled_conv_auto(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, 1, 1, padding, kernel_dim,
        false, false, false, false, false,
        (elem_t*)input,
        (elem_t*)weights_mat,
        (acc_t*)bias,
        (elem_t*)conv_output_buf,
        NO_ACTIVATION, 
        internal_conv_scale, 
        1, 1, 0, // FIX: pool_size=1, pool_stride=1, pool_padding=0 to prevent HW division by zero
        WS
    );
    
    gemmini_fence();

    // --- Step 2: Construct Sequence: [CLS, (DIST), Patches...] ---
    size_t row_size = out_channels * sizeof(elem_t);
    int current_idx = 0;
    int n_patches = out_row_dim * out_col_dim;

    // A. Copy CLS Token
    if (cls_token_data != NULL) {
        memcpy(final_input_buf + (current_idx * out_channels), cls_token_data, row_size);
        current_idx++;
    }

    // B. Copy Distillation Token (Optional)
    #ifdef DISTILLATION
    if (dist_token_data != NULL) {
        memcpy(final_input_buf + (current_idx * out_channels), dist_token_data, row_size);
        current_idx++;
    } 
    #endif

    // C. Copy Convolutional Patches
    memcpy(final_input_buf + (current_idx * out_channels), conv_output_buf, n_patches * row_size);

    // --- Step 3: Add Position Embeddings ---
    int total_seq_len = n_patches + current_idx;
    
    // FIX: Replaced MVIN_SCALE_IDENTITY with ACC_SCALE_IDENTITY 
    tiled_resadd_auto(total_seq_len, out_channels,
        ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        final_input_buf, pos_embed_data, final_input_buf,
        false, WS);
        
    gemmini_fence();
}