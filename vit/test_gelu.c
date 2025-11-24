#include <stdio.h>
#include <string.h>
#include <math.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_nn.h"

// Include the generated data
#include "includes/gelu_test_params.h"

// Output Buffer
static elem_t actual_output[TEST_SEQ_LEN][TEST_EXPANSION_DIM];

#define VERIFY_EPSILON 0.5f // Slightly higher tolerance for quantization/approx

// --- 1. Self-contained Exponential Approximation ---
// (Same as the one in your transformer_layers.c)
float gelu_exp_approx(float x) {
    if (x < -5.0f) {
        return 0.001f * (x + 10.0f);
    }
    else if (x < -2.0f) {
        float xp = x + 3.5f; 
        return 0.030197f + 0.028374f * xp + 0.013903f * xp * xp + 0.004534f * xp * xp * xp;
    }
    else if (x < 0.0f) {
        float x2 = x * x;
        float x3 = x2 * x;
        float x4 = x2 * x2;
        return 1.0f + x + 0.5f * x2 + 0.1666667f * x3 + 0.0416664f * x4;
    }
    else if (x < 2.0f) {
        float x2 = x * x;
        float x3 = x2 * x;
        float x4 = x2 * x2;
        return 1.0f + x + 0.5f * x2 + 0.1666667f * x3 + 0.0416664f * x4;
    }
    else if (x < 5.0f) {
        float xp = x - 3.5f; 
        return 16.444647f + 16.444647f * xp + 8.222323f * xp * xp + 2.740774f * xp * xp * xp;
    }
    else {
        return 100.0f + 50.0f * (x - 5.0f); 
    }
}

// --- 2. GELU Approximation (Sigmoid Method) ---
// Formula: x * sigmoid(1.702 * x) = x / (1 + exp(-1.702 * x))
// This avoids erff() entirely.
void cpu_gelu_approx(int rows, int cols, elem_t * input, elem_t * output) {
    for (int i = 0; i < rows * cols; i++) {
        float x = (float)input[i];
        
        // Calculate exponent: -1.702 * x
        float exp_val = gelu_exp_approx(-1.702f * x);
        
        // Sigmoid result
        float sigmoid = 1.0f / (1.0f + exp_val);
        
        output[i] = (elem_t)(x * sigmoid);
    }
}
void verify_result(int rows, int cols, elem_t * calc, const elem_t * exp) {
    int errors = 0;
    float max_diff = 0.0f;
    int size = rows * cols;

    for (int i = 0; i < size; i++) {
        float c = (float)calc[i];
        float e = (float)exp[i];
        float diff = fabs(c - e);
        
        if (diff > max_diff) max_diff = diff;

        if (diff > VERIFY_EPSILON) {
            if (errors < 10) { // Limit error prints
                printf("[FAIL] Index %d: Calc %.4f vs Exp %.4f (Diff: %.4f)\n", 
                       i, c, e, diff);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("[PASS] Max Diff: %.5f\n", max_diff);
    } else {
        printf("[FAIL] Total Errors: %d / %d (Max Diff: %.5f)\n", errors, size, max_diff);
    }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      return 1;
    }
#endif
    gemmini_flush(0);

    printf("--- Starting GELU Unit Test ---\n");
    printf("Dims: %d x %d (Input) -> %d x %d (Output)\n", 
           TEST_SEQ_LEN, TEST_HIDDEN_DIM, TEST_SEQ_LEN, TEST_EXPANSION_DIM);

    uint64_t start = read_cycles();

    // Replicating the FFN FC1 call
    // out = IGELU(input * W + b)
    tiled_matmul_auto(
        TEST_SEQ_LEN,        // dim_I (Rows of A)
        TEST_EXPANSION_DIM,  // dim_J (Cols of B)
        TEST_HIDDEN_DIM,     // dim_K (Cols of A / Rows of B)
        
        test_input,          // A
        test_w,              // B
        test_b,              // D (Bias)
        actual_output,       // C (Output)

        TEST_HIDDEN_DIM,     // stride_A
        TEST_EXPANSION_DIM,  // stride_B
        TEST_EXPANSION_DIM,  // stride_D
        TEST_EXPANSION_DIM,  // stride_C

        MVIN_SCALE_IDENTITY, // Scale A
        MVIN_SCALE_IDENTITY, // Scale B
        MVIN_SCALE_IDENTITY, // Scale D
        
        NO_ACTIVATION,               // <--- The key operation
        
        ACC_SCALE_IDENTITY,  // output scale
        ACC_SCALE_IDENTITY,  // bert_scale (often used for quantization adjustment)
        
        true,                // repeating_bias
        false,               // transpose_B
        false, false,        // transpose_A, full_C
        false,
        0,                   // low_D
        WS                   // weight_stationary
    );

    gemmini_fence();

    cpu_gelu_approx(TEST_SEQ_LEN, TEST_EXPANSION_DIM, (elem_t*)actual_output, (elem_t*)actual_output);
    uint64_t end = read_cycles();

    printf("Cycles: %llu\n", end - start);

    // Verification
    verify_result(TEST_SEQ_LEN, TEST_EXPANSION_DIM, 
                  (elem_t*)actual_output, expected_output);

    return 0;
}