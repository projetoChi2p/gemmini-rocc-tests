#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "includes/quantized_ffn_params.h"

// --- CPU HELPER (Local) ---
// (Same as before)
static inline float my_exp(float x) {
    if (x <= -88.0f) return 0.0f;
    if (x >= 88.0f) x = 88.0f;
    union { float f; int32_t i; } converter;
    converter.i = (int32_t)(12102203.0f * x + 1064986824);
    return converter.f;
}
static inline float my_tanh(float x) {
    float abs_x = (x < 0.0f) ? -x : x;
    if (abs_x > 4.0f) return (x < 0.0f) ? -1.0f : 1.0f;
    float e2x = my_exp(2.0f * abs_x);
    float t = (e2x - 1.0f) / (e2x + 1.0f);
    return (x < 0.0f) ? -t : t;
}
static inline int my_round(float x) {
    return (int)(x + (x >= 0 ? 0.5f : -0.5f));
}
void cpu_gelu_quantized_local(int rows, int cols, elem_t * input, elem_t * output) {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEF = 0.044715f;
    const float range_max = 6.0f;
    const float input_scale = range_max / 127.0f; 
    const float output_scale = 127.0f / range_max; 

    int size = rows * cols;
    for (int i = 0; i < size; i++) {
        float x = (float)input[i] * input_scale;
        float inner = SQRT_2_OVER_PI * (x + COEF * x * x * x);
        float tanh_res = my_tanh(inner);
        float res = 0.5f * x * (1.0f + tanh_res);
        int out_val = my_round(res * output_scale);
        if (out_val > 127) out_val = 127;
        if (out_val < -128) out_val = -128;
        output[i] = (elem_t)out_val;
    }
}
void cpu_layernorm_fallback(int rows, int cols, elem_t * data) {
    for(int i=0; i<rows; i++) {
        float sum=0, sq=0;
        for(int j=0; j<cols; j++) {
            float v = data[i*cols + j];
            sum += v; sq += v*v;
        }
        float mean = sum/cols;
        float std = sqrtf(sq/cols - mean*mean + 1e-5);
        for(int j=0; j<cols; j++) {
            float v = data[i*cols + j];
            float n = (v - mean)/std * 20.0f; 
            int res = (int)(n + (n>0?0.5:-0.5));
            if(res > 127) res=127; if(res<-128) res=-128;
            data[i*cols+j] = (elem_t)res;
        }
    }
}

int main() {
    gemmini_flush(0);
    printf("=== Quantized FFN Test (Explicit Tiled MatMul) ===\n");

    // Buffers
    static elem_t out_buf[SEQ_LEN][EXPANSION_DIM] row_align(1);
    static elem_t output[SEQ_LEN][HIDDEN_DIM] row_align(1);

    uint64_t start = read_cycles();

    // 1. FC1 (Using tiled_matmul_auto to force repeating_bias = true)
    tiled_matmul_auto(
        SEQ_LEN, EXPANSION_DIM, HIDDEN_DIM,
        (elem_t*)ffn_input, (elem_t*)w1, (acc_t*)b1, (elem_t*)out_buf,
        HIDDEN_DIM, EXPANSION_DIM, EXPANSION_DIM, EXPANSION_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 
        (acc_scale_t)SCALE_FC1, 
        0, 
        true, // <--- CRITICAL FIX: REPEATING BIAS = TRUE
        false, false, false, false, 0, WS
    );
    
    // 2. GELU (CPU)
    cpu_gelu_quantized_local(SEQ_LEN, EXPANSION_DIM, (elem_t*)out_buf, (elem_t*)out_buf);

    // 3. FC2 (Using tiled_matmul_auto)
    tiled_matmul_auto(
        SEQ_LEN, HIDDEN_DIM, EXPANSION_DIM,
        (elem_t*)out_buf, (elem_t*)w2, (acc_t*)b2, (elem_t*)output,
        EXPANSION_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 
        (acc_scale_t)SCALE_FC2, 
        0, 
        true, // <--- CRITICAL FIX: REPEATING BIAS = TRUE
        false, false, false, false, 0, WS
    );

    // 4. ResAdd
    tiled_resadd_auto(SEQ_LEN, HIDDEN_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t*)output, (elem_t*)ffn_input, (elem_t*)output,
        false, WS);
        
    // 5. LayerNorm
    cpu_layernorm_fallback(SEQ_LEN, HIDDEN_DIM, (elem_t*)output);
    
    uint64_t end = read_cycles();
    printf("Cycles: %llu\n", end - start);

    // Verify
    int errors = 0;
    int tolerance = 2; // Allow small drift

    for (int i = 0; i < SEQ_LEN * HIDDEN_DIM; i++) {
        elem_t prod = ((elem_t*)output)[i];
        elem_t exp = ((elem_t*)expected_output)[i];
        
        if (abs(prod - exp) > tolerance) {
            if (errors < 5) printf("Err %d: Got %d Exp %d (Diff %d)\n", i, prod, exp, prod-exp);
            errors++;
        }
    }

    if (errors == 0) printf("SUCCESS\n");
    else printf("FAIL: %d errors\n", errors);

    return 0;
}