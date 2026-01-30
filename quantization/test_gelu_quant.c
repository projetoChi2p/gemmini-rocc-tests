#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "include/gemmini_nn.h"
#include "includes/quantized_gelu_params.h"

// ==========================================
// BAREMETAL MATH HELPERS (No libm required)
// ==========================================

// 1. Fast Exponential Approximation (Schraudolph method)
// Valid for inputs roughly -87 to +87.
// Relies on IEEE 754 float representation.
static inline float my_exp(float x) {
    if (x <= -88.0f) return 0.0f;
    if (x >= 88.0f) x = 88.0f;
    
    // Magic conversion: 2^i ~= (1 + x/2^L)^(2^L)
    union { float f; int32_t i; } converter;
    converter.i = (int32_t)(12102203.0f * x + 1064986824);
    return converter.f;
}

// 2. Fast Tanh using my_exp
// tanh(x) = (e^2x - 1) / (e^2x + 1)
static inline float my_tanh(float x) {
    float abs_x = (x < 0.0f) ? -x : x;
    
    // Optimization: tanh(x) saturates to 1.0 quickly
    if (abs_x > 4.0f) return (x < 0.0f) ? -1.0f : 1.0f;

    float e2x = my_exp(2.0f * abs_x);
    float t = (e2x - 1.0f) / (e2x + 1.0f);
    
    return (x < 0.0f) ? -t : t;
}

// 3. Round Float to Nearest Integer
static inline int my_round(float x) {
    if (x >= 0.0f) {
        return (int)(x + 0.5f);
    } else {
        return (int)(x - 0.5f);
    }
}

// ==========================================
// GELU TEST
// ==========================================

void cpu_gelu_test(int size, const elem_t * input, elem_t * output) {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEF = 0.044715f;
    
    // Matches Python logic
    const float range_max = 6.0f;
    const float input_scale = range_max / 127.0f; 
    const float output_scale = 127.0f / range_max; 

    for (int i = 0; i < size; i++) {
        // 1. Dequantize
        float x = (float)input[i] * input_scale;
        
        // 2. GELU
        // inner = 0.797 * (x + 0.044 * x^3)
        float x_cubed = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEF * x_cubed);
        
        float tanh_res = my_tanh(inner);
        float res = 0.5f * x * (1.0f + tanh_res);
        
        // 3. Re-quantize using standalone round
        int out_val = my_round(res * output_scale);
        
        if (out_val > 127) out_val = 127;
        if (out_val < -128) out_val = -128;
        output[i] = (elem_t)out_val;
    }
}

int main() {
    printf("=== CPU Quantized GELU Test (Standalone) ===\n");

    static elem_t actual_output[GELU_DIM];

    // 1. Run CPU GELU
    cpu_gelu_test(GELU_DIM, (const elem_t*)gelu_input, actual_output);

    // 2. Verify
    int errors = 0;
    // Approximation might differ slightly from Python's perfect exp/tanh
    // Allow slightly higher tolerance (+/- 2 integers)
    int tolerance = 2; 

    for (int i = 0; i < GELU_DIM; i++) {
        elem_t inp = ((elem_t*)gelu_input)[i];
        elem_t prod = actual_output[i];
        elem_t exp = ((elem_t*)expected_output)[i];

        // Check distance
        int diff = (prod > exp) ? (prod - exp) : (exp - prod);

        if (diff > tolerance) {
            if (errors < 10) {
                printf("[FAIL] Inp %d: Got %d, Exp %d (Diff %d)\n", inp, prod, exp, diff);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("SUCCESS: CPU GELU matches Golden Reference (Tol %d).\n", tolerance);
        return 0;
    } else {
        printf("FAILURE: Found %d errors.\n", errors);
        return 1;
    }
}