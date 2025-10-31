// See LICENSE for license details.
//
// This file implements the ViT Classification Head (Step 4)

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

// Define BAREMETAL to use the 64x64x64 dimensions from previous examples
#define BAREMETAL 1

// --- STEP 1: Include base headers for types (elem_t, acc_t, ...) ---
#include "include/gemmini.h"

#define CHECK_RESULT 1
#define NO_BIAS 1

// --- STEP 2: Define ACC_T type ---
#define FULL_BIAS_WIDTH 1
#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif


// --- STEP 4: Include the helper functions ---
// (These will now be compiled using the 1x64x64 dimensions)
#include "include/gemmini_vit.h"

// classification_head.c
// Implements the final classification GEMM kernel.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdio.h>

#define BAREMETAL 1

// --- 1. Include base headers ---
#include "include/gemmini_vit_params.h"
#include "include/gemmini.h"

#define CHECK_RESULT 1
#define NO_BIAS 1

// --- 2. Define ACC_T type ---
#define FULL_BIAS_WIDTH 1
#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

// --- 4. Include helpers and kernel declarations ---
#include "include/gemmini_vit.h" // Your inline helper functions
#include "include/gemmini_vit_kernels.h" // The header declaring our function


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
 * @brief Runs the final classification head GEMM.
 * (This is the function declared in vit_kernels.h)
 */
void run_classification_head(
    const elem_t token_in[MAT_DIM_I][MAT_DIM_K], 
    const elem_t weights_in[MAT_DIM_K][MAT_DIM_J], 
    const ACC_T bias_in[MAT_DIM_I][MAT_DIM_J], 
    elem_t logits_out[MAT_DIM_I][MAT_DIM_J],
    uint64_t* gemmini_cycles
) {
    printf("  [Step 4] Running Classification Head (Gemmini WS)...\n");
    printf("    (A) Token:   (%d, %d)\n", MAT_DIM_I, MAT_DIM_K);
    printf("    (B) Weights: (%d, %d)\n", MAT_DIM_K, MAT_DIM_J);
    printf("    (C) Logits:  (%d, %d)\n", MAT_DIM_I, MAT_DIM_J);

    unsigned long start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)token_in, (elem_t*)weights_in, 
            NO_BIAS ? NULL : (void*)bias_in, (void*)logits_out,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);

    unsigned long end = read_cycles();
    *gemmini_cycles = end - start;
    printf("  [Step 4] Cycles: %llu\n", *gemmini_cycles);

#if CHECK_RESULT == 1
    // Run CPU verification
    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

    printf("    Verifying Step 4 (CPU vs Gemmini)...\n");
    full_matmul((elem_t (*)[MAT_DIM_K])token_in, (elem_t (*)[MAT_DIM_J])weights_in, 
                (ACC_T (*)[MAT_DIM_J])bias_in, gold_full);
    full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);

    if (!full_is_equal(logits_out, gold)) {
        printf("    !!! FAILURE: Classification Head Mismatch !!!\n");
        exit(1);
    } else {
        printf("    ... Success.\n");
    }
#endif
}

// The main() function has been removed.

/*
int main() {

// #if defined(FAST) || !defined(HAS_NORMALIZATIONS)
//     printf("Softmax activation not supported, exiting.\n");
//     exit(0);
// #endif


#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    printf("Classification Head Test (Logits Output)\n");
    printf("Input Token (A) Dims: %dx%d (I x K)\n", MAT_DIM_I, MAT_DIM_K);
    printf("Classifier (B) Dims: %dx%d (K x J)\n", MAT_DIM_K, MAT_DIM_J);
    printf("Output Logits (C) Dims: %dx%d (I x J)\n", MAT_DIM_I, MAT_DIM_J);
    printf("----------------------------------------\n");

    gemmini_flush(0);

    // A: The input [class] token vector (after LayerNorm)
    static elem_t class_token_in[MAT_DIM_I][MAT_DIM_K] row_align(1);
    // B: The classifier weights
    static elem_t classifier_weights[MAT_DIM_K][MAT_DIM_J] row_align(1);
    // C: The final output (probabilities after softmax)
    static elem_t output_probs[MAT_DIM_I][MAT_DIM_J] row_align(1);
    // D: The classifier bias
    static ACC_T classifier_bias[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

    // Gold standard for checking result
    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];
// --- Inside main() ---

#if CHECK_RESULT == 1
    // *** FIX: ADDED INITIALIZATION BACK ***
    printf("Initializing input matrices...\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        class_token_in[i][j] = (rand() % 7) - 3; // -3 to 3
      }
    }

    for (size_t i = 0; i < MAT_DIM_K; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        classifier_weights[i][j] = (rand() % 7) - 3; // -3 to 3
      }
    }

    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        classifier_bias[i][j] = NO_BIAS ? 0 : (rand() % 3) - 1; // -1 to 1
      }
    }
    
    // *** DEBUG PRINTS ***
    printf("\n--- DEBUG: Initialized Matrices ---\n");
    print_A(class_token_in);
    print_B(classifier_weights);
    
    // Print Bias (Matrix D)
    printf("--- Printing D (Bias) (%dx%d matrix) ---\n", MAT_DIM_I, MAT_DIM_J);
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
        printf("Row %lu: [ ", (unsigned long)i);
        size_t j_to_print = MAT_DIM_J > 32 ? 32 : MAT_DIM_J;
        for (size_t j = 0; j < j_to_print; ++j) {
            // Print as a signed integer, as ACC_T is signed
            printf("%d ", (int)classifier_bias[i][j]);
        }
        if (MAT_DIM_J > j_to_print) printf("...");
        printf("]\n");
    }
    printf("--------------------------------------\n\n");
    // *** END DEBUG PRINTS ***


    printf("Starting slow CPU matmul (for verification)\n");
    unsigned long cpu_start = read_cycles();

    // 1. Run the *simple* C matmul to get raw logits
    full_matmul(class_token_in, classifier_weights, classifier_bias, gold_full);

    // 2. Scale the logits 
    full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);

    unsigned long cpu_end = read_cycles();
    printf("Cycles taken: %lu\n", cpu_end - cpu_start);
#endif

    printf("Starting gemmini matmul (WS mode)\n");
    unsigned long start = read_cycles();

    // 3. Run Gemmini, but *turn off* the activation
    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)class_token_in, (elem_t*)classifier_weights, 
            NO_BIAS ? NULL : &classifier_bias[0][0], (elem_t*)output_probs,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, // <--- We are checking raw logits
            ACC_SCALE_IDENTITY, 
            0, // <--- No BERT scale
            false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS); // Run on Gemmini

    unsigned long end = read_cycles();
    printf("Cycles taken: %lu\n", end - start);

#if CHECK_RESULT == 1
    printf("Verifying result (logits vs logits)...\n");
    if (!full_is_equal(output_probs, gold)) {
        printf("!!! FAILURE !!!\n");
        printf("C (Gemmini Output):\n");
        full_printMatrix(output_probs);
        printf("Gold (CPU Output):\n");
        full_printMatrix(gold);
        printf("\n");
        exit(1);
    } else {
        printf("--- SUCCESS ---\n");
        printf("Final Output Logits (pre-softmax):\n");
        full_printMatrix(output_probs);
    }
#endif

    exit(0);
}
*/