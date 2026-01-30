#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// Include Generated Params
#include "includes/quantized_embedding_params.h"

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

int main() {
    gemmini_flush(0);
    printf("=== Quantized Embedding Test ===\n");
    printf("Patches: %d, Patch Dim: %d, Hidden: %d\n", PATCH_SEQ_LEN, PATCH_DIM, HIDDEN_DIM);

    // Buffers
    static elem_t temp_patch_buf[PATCH_SEQ_LEN][HIDDEN_DIM] row_align(1);
    
    // Final buffer size = (Seq + 1) * Hidden
    static elem_t final_input_buf[PATCH_SEQ_LEN + 1][HIDDEN_DIM] row_align(1);

    uint64_t start = read_cycles();

    // Call the DUT
    compute_patch_embeddings_quantized(
        PATCH_SEQ_LEN,
        HIDDEN_DIM,
        PATCH_DIM,
        (const elem_t*)current_patches,
        (const elem_t*)patch_embed_w,
        (const acc_t*)patch_embed_b,
        SCALE_EMBED, // Defined in header
        (const elem_t*)pos_embed_data,
        (const elem_t*)cls_token_data,
        (elem_t*)temp_patch_buf,
        (elem_t*)final_input_buf
    );

    uint64_t end = read_cycles();
    printf("Cycles: %llu\n", end - start);

    // Verify
    int errors = 0;
    int total_elements = (PATCH_SEQ_LEN + 1) * HIDDEN_DIM;

    // We check the final output which includes Proj + Concat + PosEmbed
    for (int i = 0; i < total_elements; i++) {
        elem_t prod = ((elem_t*)final_input_buf)[i];
        elem_t exp = ((elem_t*)expected_output)[i];

        if (prod != exp) {
            if (errors < 5) {
                printf("[FAIL] Idx %d: Got %d, Exp %d\n", i, prod, exp);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("SUCCESS: Embedding pipeline matches Golden Reference.\n");
        return 0;
    } else {
        printf("FAILURE: Found %d errors.\n", errors);
        return 1;
    }
}