#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "includes/matmul_quant_data.h"

int main() {
    gemmini_flush(0);
    printf("=== Gemmini Fixed Quantized Test ===\n");
    // Note: We print the float scale here to verify
    printf("Testing Scale: %f (Shift approx: %d)\n", TEST_SCALE, TEST_SHIFT);

    static elem_t C_result[TEST_DIM][TEST_DIM] row_align(1);

    // --- DIRECT HARDWARE CALL ---
    tiled_matmul_auto(
        TEST_DIM, TEST_DIM, TEST_DIM,
        (elem_t*)A_mat, (elem_t*)B_mat, (acc_t*)D_bias, (elem_t*)C_result,
        TEST_DIM, TEST_DIM, TEST_DIM, TEST_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION,
        
        // [CRITICAL FIX] Use the Float Scale, not Identity
        (acc_scale_t)TEST_SCALE, 
        
        // [CRITICAL FIX] Set bert_scale to 0 (unless doing int32->int8 input scaling)
        0, 
        
        true, // repeating_bias -> Broadcasts D_bias (1x16) across rows
        false, false,
        false, false,
        0,
        WS
    );

    gemmini_fence();

    // --- Verify Results ---
    int errors = 0;
    for (int i = 0; i < TEST_DIM; i++) {
        for (int j = 0; j < TEST_DIM; j++) {
            elem_t produced = C_result[i][j];
            elem_t expected = C_gold[i][j];

            // Allow for off-by-one rounding differences if Python/Hardware rounding differs slightly
            if (produced != expected) {
                if (errors < 10) { 
                     printf("Error [%d][%d]: Exp %d, Got %d (Diff: %d)\n", 
                           i, j, expected, produced, produced - expected);
                }
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("SUCCESS: Output matches Golden reference!\n");
        return 0;
    } else {
        printf("FAILURE: Found %d errors.\n", errors);
        return 1;
    }
}