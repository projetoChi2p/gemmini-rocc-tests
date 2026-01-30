#ifndef __VIT_UTILS__
#define __VIT_UTILS__

#include "../inc/utils.h"
#include <float.h>
#include <stdio.h>

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

void cpu_gelu_approx(int rows, int cols, elem_t * input, elem_t * output) {
    for (int i = 0; i < rows * cols; i++) {
        float x = (float)input[i];
        float exp_val = gelu_exp_approx(-1.702f * x);
        float sigmoid = 1.0f / (1.0f + exp_val);
        output[i] = (elem_t)(x * sigmoid);
    }
}

int find_max_index(elem_t* arr, int size) {
    int max_idx = 0;
    elem_t max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// check_top_k.c
#include <stdbool.h>

// Returns 1 (true) if the value at 'target_idx' is among the top 'k' values in 'arr'
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

void print_results_summary(int total_samples, int correct_predictions, int top5_correct_predictions, uint64_t total_cycles) {
    double accuracy = (double)correct_predictions / total_samples * 100.0;
    double avg_cycles = (double)total_cycles / total_samples;

    printf("\n==========================================\n");
    printf("       INFERENCE RESULTS SUMMARY          \n");
    printf("==========================================\n");
    printf(" Total Samples     : %d\n", total_samples);
    printf(" Correct Predictions: %d\n", correct_predictions);
    
#ifdef ELEM_T_IS_FLOAT
    printf(" Accuracy          : %.2f%%\n", accuracy);
#else
    printf(" Accuracy          : %d%%\n", (int)accuracy);
#endif
    // === REPORTING ===
    printf("\n--- Results ---\n");
    printf("Top-1 Accuracy: (%d/%d)\n", 
            correct_predictions, NUM_INFERENCES);
           
    printf("Top-5 Accuracy: (%d/%d)\n", 
            top5_correct_predictions, NUM_INFERENCES);

    printf("------------------------------------------\n");
    printf(" Total Cycles      : %llu\n", total_cycles);
    printf(" Avg Cycles/Inf    : %.0f\n", avg_cycles);
    printf("==========================================\n");
}


#ifdef DEBUG
#define HIST_BAR_WIDTH 40
#define HIST_BINS 7

// Bins: [0-1e-6), [1e-6, 1e-5), [1e-5, 1e-4), [1e-4, 1e-3), [1e-3, 1e-2), [1e-2, 1e-1), [> 1e-1]
const float BIN_THRESHOLDS[HIST_BINS] = {0.000001f, 0.00001f, 0.0001f, 0.001f, 0.01f, 0.1f, FLT_MAX};
const char* BIN_LABELS[HIST_BINS]     = {"< 1e-6", "1e-6  ", "1e-5  ", "1e-4  ", "1e-3  ", "1e-2  ", "> 1e-1"};
void print_error_histogram(const char * step_name, int rows, int cols, 
                           const elem_t * calculated, const elem_t * expected) {
    
    int counts[HIST_BINS] = {0};
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int total_elements = rows * cols;
    int errors_above_threshold = 0;
    float threshold = 0.01f; // Threshold for "FAIL" judgement

    // 1. Collect Stats
    for (int i = 0; i < total_elements; i++) {
        float c_val = (float)calculated[i];
        float e_val = (float)expected[i];
        float diff = fabs(c_val - e_val);

        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;

        if (diff > threshold) errors_above_threshold++;

        // Binning
        for (int b = 0; b < HIST_BINS; b++) {
            if (diff < BIN_THRESHOLDS[b]) {
                counts[b]++;
                break;
            }
        }
    }

    float mae = sum_diff / total_elements;

    // 2. Print Header
    printf("\n=== DEBUG: %s ===\n", step_name);
    printf("  Dims: %dx%d | Max Diff: %.6f | MAE: %.6f\n", rows, cols, max_diff, mae);
    
    // 3. Print Histogram
    printf("  Error Distribution (Log Scale):\n");
    for (int b = 0; b < HIST_BINS; b++) {
        // Calculate bar length
        int bar_len = (int)((float)counts[b] / total_elements * HIST_BAR_WIDTH);
        
        printf("    %s : ", BIN_LABELS[b]);
        for (int k = 0; k < bar_len; k++) printf("#");
        if (counts[b] > 0 && bar_len == 0) printf("."); // Dot for non-zero but small count
        
        // Print count and percentage
        printf(" (%d - %.1f%%)\n", counts[b], (float)counts[b]/total_elements * 100.0f);
    }

    // 4. Final Judgement
    if (errors_above_threshold > 0) {
        printf("  [FAIL] %d elements have errors > %.2f\n", errors_above_threshold, threshold);
    } else {
        printf("  [PASS] All errors within tolerance.\n");
    }
    printf("--------------------------------------------------\n");
}
#endif // DEBUG

#endif // __VIT_UTILS__
