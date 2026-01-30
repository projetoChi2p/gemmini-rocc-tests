#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "includes/quantized_resadd_params.h"

int main() {
    gemmini_flush(0);
    printf("=== Hardware ResAdd Test (A*%.1f + B*%.1f) ===\n", VAL_SCALE_A, VAL_SCALE_B);

    static elem_t C_result[MAT_M][MAT_N] row_align(1);

    uint64_t start = read_cycles();

    // Call Hardware ResAdd
    // Note: acc_scale_t cast converts the float to fixed-point multiplier
    tiled_resadd_auto(
        MAT_M, MAT_N,
        (acc_scale_t)VAL_SCALE_A, 
        (acc_scale_t)VAL_SCALE_B, 
        (acc_scale_t)VAL_SCALE_C,
        (elem_t*)mat_A, 
        (elem_t*)mat_B, 
        (elem_t*)C_result,
        false, // No ReLU
        WS
    );

    gemmini_fence();
    uint64_t end = read_cycles();
    printf("Cycles: %llu\n", end - start);

    // Verify
    int errors = 0;
    
    for (int i = 0; i < MAT_M * MAT_N; i++) {
        elem_t prod = ((elem_t*)C_result)[i];
        elem_t exp = ((elem_t*)expected_C)[i];

        if (prod != exp) {
            if (errors < 5) {
                printf("[FAIL] Idx %d: A=%d, B=%d -> Got %d, Exp %d\n", 
                       i, ((elem_t*)mat_A)[i], ((elem_t*)mat_B)[i], prod, exp);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("SUCCESS: ResAdd matches Golden Reference.\n");
        return 0;
    } else {
        printf("FAILURE: Found %d errors.\n", errors);
        return 1;
    }
}