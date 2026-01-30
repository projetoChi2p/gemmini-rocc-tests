#ifndef __VIT_UTILS_QUANT__
#define __VIT_UTILS_QUANT__

// ==========================================
// HELPER: Tensor Verification
// ==========================================
// Returns true if PASS, false if FAIL
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

bool verify_tensor(const char* name, elem_t* hw_ptr, elem_t* sw_ptr, int size, int tolerance) {
    printf("\n--- VERIFY: %s ---\n", name);
    printf("   Sample [0]: HW %d vs SW %d\n", hw_ptr[0], sw_ptr[0]);
    printf("   Sample [1]: HW %d vs SW %d\n", hw_ptr[1], sw_ptr[1]);

    int errs = 0;
    int max_diff = 0;
    
    // Histogram buckets: [0], [1-2], [3-5], [6-10], [>10]
    int hist[5] = {0, 0, 0, 0, 0};
    
    for (int i = 0; i < size; i++) {
        int diff = abs(hw_ptr[i] - sw_ptr[i]);
        if (diff > max_diff) max_diff = diff;

        // Fill Histogram
        if (diff == 0)      hist[0]++;
        else if (diff <= 2) hist[1]++;
        else if (diff <= 5) hist[2]++;
        else if (diff <= 10) hist[3]++;
        else                hist[4]++;

        // Check Tolerance
        if (diff > tolerance) {
            if (errs < 10) { // Limit detailed prints
                printf("   [FAIL] Idx %d: HW %d vs SW %d (Diff %d)\n", i, hw_ptr[i], sw_ptr[i], diff);
            }
            errs++;
        }
    }

    // Print Histogram
    printf("   Diff Distribution:\n");
    printf("     0:    %d\n", hist[0]);
    printf("     1-2:  %d\n", hist[1]);
    printf("     3-5:  %d\n", hist[2]);
    printf("     6-10: %d\n", hist[3]);
    printf("     >10:  %d\n", hist[4]);
    printf("   Max Diff: %d\n", max_diff);

    if (errs == 0) {
        printf("   >> PASS\n");
        return true;
    } else {
        printf("   >> FAIL (%d errors > tol %d)\n", errs, tolerance);
        return false;
    }
}

// ==========================================
// HELPER: Argmax
// ==========================================
int find_max_index(elem_t * scores, int size) {
    int max_idx = 0;
    elem_t max_val = scores[0];
    for (int i = 1; i < size; i++) {
        if (scores[i] > max_val) {
            max_val = scores[i];
            max_idx = i;
        }
    }
    return max_idx;
}

#endif