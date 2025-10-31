/*
 * gemmini_vit.h
 *
 * This header contains common *debug printing* helper functions.
 *
 * IMPORTANT: Verification functions (matmul, matscale, is_equal)
 * are NOT in this file. They must be defined locally in each
 * .c file that needs them to ensure correct macro expansion.
 */

#ifndef GEMMINI_VIT_H
#define GEMMINI_VIT_H

#include <stdio.h> // For printf

#include "include/gemmini_vit_params.h"

// --- Debug Print Helpers ---

static inline void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  // Print a smaller subset to avoid flooding the console
  size_t print_dim_I = MAT_DIM_I < 16 ? MAT_DIM_I : 16;
  size_t print_dim_J = MAT_DIM_J < 16 ? MAT_DIM_J : 16;
  printf("--- Printing top-left %lux%lu of %dx%d matrix ---\n", 
            (unsigned long)print_dim_I, (unsigned long)print_dim_J,
            MAT_DIM_I, MAT_DIM_J);
  for (size_t i = 0; i < print_dim_I; ++i) {
    for (size_t j = 0; j < print_dim_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
  printf("-------------------------------------------------\n");
}

// (You can add print_A and print_B here if you want)

#endif // GEMMINI_VIT_H