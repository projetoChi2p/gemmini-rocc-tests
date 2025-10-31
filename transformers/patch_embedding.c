// patch_embedding.c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define BAREMETAL 1
#include "include/gemmini.h"

// --- 1. Include Global Model Params ---
#include "include/gemmini_vit_params.h"

#define CHECK_RESULT 1
#define NO_BIAS 1

// --- 2. Define ACC_T type ---
#define FULL_BIAS_WIDTH 1
#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

// --- 3. Define *this file's local* KERNEL dimensions ---
// This block is ESSENTIAL. It overrides any other definitions.
#undef MAT_DIM_I
#undef MAT_DIM_K
#undef MAT_DIM_J
#define MAT_DIM_I NUM_PATCHES // 64
#define MAT_DIM_K PATCH_DIM   // 64
#define MAT_DIM_J HIDDEN_DIM  // 64

// --- 4. Include helpers and kernel declarations ---
#include "include/gemmini_vit.h" // Debug print helpers
#include "include/gemmini_vit_kernels.h" // Kernel function prototypes

// --- 5. Add LOCAL verification functions ---
// These are compiled with this file's local macros (MAT_DIM_I = 64)
// This is the fix for the "4160 vs 4096" error.
static inline void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

static inline void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], acc_scale_t scale) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      full_t scaled = ACC_SCALE(full[r][c], scale);
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled;
#endif
    }
} 

static inline int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

/**
 * @brief Runs the patch embedding GEMM.
 * (This is function 1/2 from this file)
 */
void run_patch_embedding(
    const elem_t patches_in[MAT_DIM_I][MAT_DIM_K], 
    const elem_t weights_in[MAT_DIM_K][MAT_DIM_J], 
    const ACC_T bias_in[MAT_DIM_I][MAT_DIM_J], 
    elem_t embed_out[MAT_DIM_I][MAT_DIM_J],
    uint64_t* gemmini_cycles
) {
    printf("  [Step 1] Running Patch Embedding (Gemmini WS)...\n");
    printf("    (A) Patches: (%d, %d)\n", MAT_DIM_I, MAT_DIM_K);
    printf("    (B) Weights: (%d, %d)\n", MAT_DIM_K, MAT_DIM_J);
    printf("    (C) Output:  (%d, %d)\n", MAT_DIM_I, MAT_DIM_J);
    
    unsigned long start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)patches_in, (elem_t*)weights_in, 
            NO_BIAS ? NULL : (void*)bias_in, (void*)embed_out,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);

    unsigned long end = read_cycles();
    *gemmini_cycles = end - start;
    printf("  [Step 1] Cycles: %llu\n", *gemmini_cycles);

#if CHECK_RESULT == 1
    // Run CPU verification
    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

    printf("    Verifying Step 1 (CPU vs Gemmini)...\n");
    full_matmul((elem_t (*)[MAT_DIM_K])patches_in, (elem_t (*)[MAT_DIM_J])weights_in, 
                (ACC_T (*)[MAT_DIM_J])bias_in, gold_full);
    full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);

    if (!full_is_equal(embed_out, gold)) {
        printf("    !!! FAILURE: Patch Embedding Mismatch !!!\n");
        return;
    } else {
        printf("    ... Success.\n");
    }
#endif
}


/**
 * @brief Adds the [CLS] token and positional embeddings. (CPU)
 * (This is function 2/2 from this file)
 */
void add_cls_and_pos_embed(
    const elem_t patch_embeds[NUM_PATCHES][HIDDEN_DIM],
    const elem_t cls_token_weights[1][HIDDEN_DIM],
    const elem_t pos_embed_weights[SEQ_LEN][HIDDEN_DIM],
    elem_t encoder_input_out[SEQ_LEN][HIDDEN_DIM])
{
    printf("  [Step 2] Adding [CLS] token & Pos-Embed (CPU)...\n");
    
    // 1. Copy [CLS] token to row 0
    memcpy(encoder_input_out[0], cls_token_weights[0], HIDDEN_DIM * sizeof(elem_t));

    // 2. Copy patch embeddings to rows 1 through N
    memcpy(encoder_input_out[1], patch_embeds, NUM_PATCHES * HIDDEN_DIM * sizeof(elem_t));

    // 3. Add positional embeddings (element-wise)
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            full_t sum = (full_t)encoder_input_out[i][j] + (full_t)pos_embed_weights[i][j];
            if (sum > elem_t_max) sum = elem_t_max;
            if (sum < elem_t_min) sum = elem_t_min; // Handle signed types
            encoder_input_out[i][j] = (elem_t)sum;
        }
    }
}