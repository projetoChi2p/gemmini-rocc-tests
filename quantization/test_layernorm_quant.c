#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "includes/quantized_layernorm_params.h"

int main() {
    gemmini_flush(0);
    printf("=== Hardware LayerNorm Test ===\n");

    static elem_t ln_output[LN_ROWS][LN_COLS] row_align(1);

    // Call Hardware LayerNorm
    tiled_norm_auto(
        LN_ROWS, LN_COLS,
        (acc_t*)ln_input,    // Input (Accumulator/Int32)
        (elem_t*)ln_output,  // Output (Int8)
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS
    );

    gemmini_fence();

    // Just verify it produced valid data (not all zeros)
    int non_zeros = 0;
    for (int i = 0; i < LN_ROWS * LN_COLS; i++) {
        if (((elem_t*)ln_output)[i] != 0) non_zeros++;
    }

    printf("Non-Zero Outputs: %d / %d\n", non_zeros, LN_ROWS*LN_COLS);
    
    if (non_zeros > 0) printf("SUCCESS: Hardware LayerNorm produced data.\n");
    else printf("FAIL: Output was all zeros.\n");

    return 0;
}