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


void print_results_summary(int total_samples, int correct_predictions, int top3_correct_predictions, int top5_correct_predictions, uint64_t total_cycles) {
    double accuracy = (double)correct_predictions / total_samples * 100.0;
    double accuracy_top3 = (double)top3_correct_predictions / total_samples * 100.0;
    double accuracy_top5 = (double)top5_correct_predictions / total_samples * 100.0;
    double avg_cycles = (double)total_cycles / total_samples;

    printf("\n==========================================\n");
    printf("       INFERENCE RESULTS SUMMARY          \n");
    printf("==========================================\n");
    printf(" Total Samples     : %d\n", total_samples);
    printf(" Correct Predictions: %d\n", correct_predictions);
    printf(" Top-3 Predictions: %d\n", top3_correct_predictions);
    printf(" Top-5 Predictions: %d\n", top5_correct_predictions);
    
#ifdef ELEM_T_IS_FLOAT
    printf(" Accuracy          : %.2f%%\n", accuracy);
#else
    printf(" Accuracy          : %d%%\n", (int)accuracy);
#endif
    // === REPORTING ===
    printf("\n--- Results ---\n");
    printf("Top-1 Accuracy: (%d/%d)\n", 
            correct_predictions, NUM_INFERENCES);

    printf("Top-3 Accuracy: (%d/%d)\n", 
            top3_correct_predictions, NUM_INFERENCES);
           
    printf("Top-5 Accuracy: (%d/%d)\n", 
            top5_correct_predictions, NUM_INFERENCES);

    printf("------------------------------------------\n");
    printf(" Total Cycles      : %llu\n", total_cycles);
    printf(" Avg Cycles/Inf    : %.0f\n", avg_cycles);
    printf("==========================================\n");
}

int is_in_top_k(elem_t* arr, int size, int target_idx, int k) {
    elem_t target_score = arr[target_idx];
    int count_greater = 0;

    for (int i = 0; i < size; i++) {
        if (i == target_idx) continue; // Skip comparing with itself

        // If another class has a strictly higher score, increment count
        if (arr[i] > target_score) {
            count_greater++;
        }
        
        // Optimization: If we already found k elements bigger, it's definitely not top-k
        if (count_greater >= k) {
            return 0; // False
        }
    }

    // If fewer than k items are larger, then target is in the top k
    return 1; // True
}

#endif