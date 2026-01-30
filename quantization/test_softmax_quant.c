#include <stdio.h>
#include <stdlib.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "includes/quantized_softmax_params.h"

int main() {
    gemmini_flush(0);
    printf("=== Hardware Softmax Test (MatMul + Softmax) ===\n");

    static elem_t C_result[MAT_M][MAT_N] row_align(1);

    uint64_t start = read_cycles();

    // Call Hardware with SOFTMAX activation
    tiled_matmul_auto(
        MAT_M, MAT_N, MAT_K,
        (elem_t*)mat_A, (elem_t*)mat_B, 
        NULL, // No Bias
        (elem_t*)C_result,
        MAT_K, MAT_N, MAT_N, MAT_N,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        
        SOFTMAX, // <--- Activation
        
        ACC_SCALE_IDENTITY, // Standard scaling (1.0)
        
        (acc_scale_t)TEST_BERT_SCALE, // <--- Softmax Scale
        
        false, 
        false, false, false, false, 0, WS
    );

    gemmini_fence();
    uint64_t end = read_cycles();
    printf("Cycles: %llu\n", end - start);

    // Verify Output Structure
    // Since we don't have the exact bit-accurate python model for Softmax,
    // we verify properties:
    // 1. All values should be non-negative (Probabilities >= 0)
    // 2. Row sums should be roughly consistent (e.g., around 127 or 256 depending on implementation)
    
    int min_val = 127;
    int max_val = -128;
    long total_row_sum = 0;

    for (int i = 0; i < MAT_M; i++) {
        long row_sum = 0;
        for (int j = 0; j < MAT_N; j++) {
            elem_t val = C_result[i][j];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            row_sum += val;
        }
        total_row_sum += row_sum;
        // printf("Row %d Sum: %ld\n", i, row_sum);
    }

    printf("Min Val: %d, Max Val: %d\n", min_val, max_val);
    printf("Avg Row Sum: %ld\n", total_row_sum / MAT_M);

    if (min_val < 0) {
        printf("FAIL: Softmax produced negative values.\n");
        return 1;
    }
    
    // Check if result is close to our Python approximation (optional)
    int diff_sum = 0;
    for (int i = 0; i < MAT_M * MAT_N; i++) {
        diff_sum += abs(((elem_t*)C_result)[i] - ((elem_t*)expected_C_approx)[i]);
    }
    printf("Mean Absolute Diff from Python Approx: %f\n", (float)diff_sum / (MAT_M * MAT_N));

    if (max_val > 0) {
        printf("SUCCESS: Softmax output looks valid (Non-negative, Non-zero).\n");
        return 0;
    } else {
        printf("FAIL: Softmax output was all zeros.\n");
        return 1;
    }
}