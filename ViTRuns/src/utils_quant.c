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
    //printf("   Sample [0]: HW %d vs SW %d\n", hw_ptr[0], sw_ptr[0]);
    //printf("   Sample [1]: HW %d vs SW %d\n", hw_ptr[1], sw_ptr[1]);

    #ifdef QUANTIZED
        int scale = 1;
    #else
        int scale = 100; // Scale for printing floats as integers (e.g., 3 decimal places)
        int log_scale = 2;
    #endif

    int errs = 0;
    int max_diff = 0;
    
    // Histogram buckets: [0], [1-2], [3-5], [6-10], [11-12], [13-15], [16-20], [>20]
    int hist[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    for (int i = 0; i < size; i++) {
        int diff = abs(hw_ptr[i]*scale - sw_ptr[i]*scale);
        if (diff > max_diff) max_diff = diff;

        if (diff > tolerance) {
            errs++;
        }

        // Fill Histogram
        if (diff == 0)      hist[0]++;
        else if (diff <= 2) hist[1]++;
        else if (diff <= 5) hist[2]++;
        else if (diff <= 10) hist[3]++;
        else if (diff <= 12) hist[4]++;
        else if (diff <= 15) hist[5]++;
        else if (diff <= 20) hist[6]++;
        else                 hist[7]++;
    }

    

    if (errs > 0) {
        // Print Histogram
        #ifdef QUANTIZED
        printf("===========================================\n");
        printf("   Diff Distribution:\n");
        printf("     0:    %d\n", hist[0]);
        printf("     1-2:  %d\n", hist[1]);
        printf("     3-5:  %d\n", hist[2]);
        printf("     6-10: %d\n", hist[3]);
        printf("     11-12: %d\n", hist[4]);
        printf("     13-15: %d\n", hist[5]);
        printf("     16-20: %d\n", hist[6]);
        printf("     >20:  %d\n", hist[7]);
        printf("   Max Diff: %d\n", max_diff);
        printf("===========================================\n");
        printf("   First 10 Mismatches (HW vs SW):\n");
        #else
        printf("===========================================\n");
        printf("   Diff Distribution:\n");
        printf("     0e-%d:    %d\n", log_scale, hist[0]);
        printf("     1-2e-%d:  %d\n", log_scale, hist[1]);
        printf("     3-5e-%d:  %d\n", log_scale, hist[2]);
        printf("     6-10e-%d: %d\n", log_scale, hist[3]);
        printf("     11-12e-%d: %d\n", log_scale, hist[4]);
        printf("     13-15e-%d: %d\n", log_scale, hist[5]);
        printf("     16-20e-%d: %d\n", log_scale, hist[6]);
        printf("     >20e-%d:  %d\n", log_scale, hist[7]);
        printf("   Max Diff: %de-%d\n", max_diff, log_scale);
        printf("===========================================\n");
        printf("   First 10 Mismatches (HW vs SW):\n");
        int printed = 0;
        for (int i = 0; i < size && printed < 10; i++) {
            int diff = abs(hw_ptr[i]*scale - sw_ptr[i]*scale);
            if (diff > tolerance) {
                printf("     Index %d: HW %de-%d vs SW %de-%d (Diff: %de-%d)\n", i, (int)(hw_ptr[i]*scale), log_scale, (int)(sw_ptr[i]*scale), log_scale, diff, log_scale);
                printed++;
            }
        }
        printf("===========================================\n");
        #endif
    }

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
// ==========================================
// VALUE HISTOGRAM (for debugging actual values)
// ==========================================
// ==========================================
// VALUE HISTOGRAM (for debugging actual values)
// ==========================================
void print_value_histogram(const char* name, elem_t* data, int size, int num_bins) {
    printf("\n=== VALUE HISTOGRAM: %s ===\n", name);
    printf("Size: %d elements\n", size);
    
    // Find min and max values
    int min_val = INT_MAX;
    int max_val = INT_MIN;
    int64_t sum = 0;
    
    for (int i = 0; i < size; i++) {
        int val = data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    
    double avg = (double)sum / size;
    
    printf("Value range: [%d, %d]\n", min_val, max_val);
    printf("Average value: %.2f\n", avg);
    
    // Create bins
    int bin_width = (max_val - min_val + 1) / num_bins;
    if (bin_width == 0) bin_width = 1;
    
    int bins[num_bins];
    for (int i = 0; i < num_bins; i++) bins[i] = 0;
    
    // Count values into bins
    for (int i = 0; i < size; i++) {
        int val = data[i];
        int bin_idx = (val - min_val) / bin_width;
        if (bin_idx >= num_bins) bin_idx = num_bins - 1;
        bins[bin_idx]++;
    }
    
    // Print histogram
    printf("\nValue distribution (%d bins):\n", num_bins);
    for (int i = 0; i < num_bins; i++) {
        int start = min_val + i * bin_width;
        int end = start + bin_width - 1;
        if (i == num_bins - 1) end = max_val;
        
        printf("[%4d, %4d]: %6d", start, end, bins[i]);
        
        // Simple bar chart
        int bar_len = (bins[i] * 50) / size;
        for (int j = 0; j < bar_len; j++) printf("#");
        printf("\n");
    }
    
    // Print most common values
    printf("\n--- Most Common Values ---\n");
    
    // Simple approach: just look for values that appear frequently
    // For a better approach, you'd need to sort, but let's keep it simple
    int common_count = 0;
    printf("Looking for values with >1%% frequency...\n");
    for (int val = min_val; val <= max_val && common_count < 10; val++) {
        int count = 0;
        for (int i = 0; i < size; i++) {
            if (data[i] == val) count++;
        }
        if (count * 100 > size) {  // More than 1% of values
            printf("  Value %d: %d times (%.1f%%)\n", 
                   val, count, (float)count/size*100);
            common_count++;
        }
    }
    
    printf("================================\n");
}

// Simpler version with fixed bins for layer norm values
void print_simple_value_hist(const char* name, elem_t* data, int size) {
    printf("\n=== SIMPLE VALUE HIST: %s ===\n", name);
    
    // Common bins for layer norm debugging
    // Since layer norm outputs are often in a specific range
    int bins[10] = {0};  // < -10, -10 to -5, -5 to -2, -2 to -1, -1 to 0, 
                        // 0 to 1, 1 to 2, 2 to 5, 5 to 10, > 10
    
    for (int i = 0; i < size; i++) {
        int val = data[i];
        if (val < -10) bins[0]++;
        else if (val < -5) bins[1]++;
        else if (val < -2) bins[2]++;
        else if (val < -1) bins[3]++;
        else if (val < 0) bins[4]++;
        else if (val == 0) bins[5]++;  // Exactly 0
        else if (val <= 1) bins[6]++;
        else if (val <= 2) bins[7]++;
        else if (val <= 5) bins[8]++;
        else bins[9]++;
    }
    
    printf("Value ranges:\n");
    printf("  < -10:  %d\n", bins[0]);
    printf("  -10:-5: %d\n", bins[1]);
    printf("  -5:-2:  %d\n", bins[2]);
    printf("  -2:-1:  %d\n", bins[3]);
    printf("  -1:0:   %d\n", bins[4]);
    printf("  =0:     %d\n", bins[5]);
    printf("  0:1:    %d\n", bins[6]);
    printf("  1:2:    %d\n", bins[7]);
    printf("  2:5:    %d\n", bins[8]);
    printf("  >5:     %d\n", bins[9]);
    
    // Show actual unique values (first 20 unique ones)
    printf("\nFirst 20 unique values found:\n");
    int unique_count = 0;
    int printed_vals[20] = {0};
    
    for (int i = 0; i < size && unique_count < 20; i++) {
        int val = data[i];
        int already_printed = 0;
        
        // Check if we already printed this value
        for (int j = 0; j < unique_count; j++) {
            if (printed_vals[j] == val) {
                already_printed = 1;
                break;
            }
        }
        
        if (!already_printed) {
            printed_vals[unique_count++] = val;
            printf("  %d", val);
            if (unique_count % 10 == 0) printf("\n");
        }
    }
    printf("\n");
}

// Minimal: just show the actual HW and SW values
void print_actual_values(const char* name, elem_t* hw, elem_t* sw, int size, int max_to_print) {
    printf("\n=== ACTUAL VALUES: %s ===\n", name);
    printf("Showing first %d mismatches:\n", max_to_print);
    
    int printed = 0;
    for (int i = 0; i < size && printed < max_to_print; i++) {
        if (hw[i] != sw[i]) {
            printf("Idx %4d: HW = %6d, SW = %6d\n", i, hw[i], sw[i]);
            printed++;
        }
    }
    
    // Also show what values are actually in the tensor
    printf("\nHW value frequency (top 10):\n");
    
    // Simple frequency counter for HW values
    int unique_count = 0;
    struct { int val; int count; } freqs[50] = {{0,0}};  // Track up to 50 unique values
    
    for (int i = 0; i < size; i++) {
        int val = hw[i];
        int found = 0;
        
        for (int j = 0; j < unique_count; j++) {
            if (freqs[j].val == val) {
                freqs[j].count++;
                found = 1;
                break;
            }
        }
        
        if (!found && unique_count < 50) {
            freqs[unique_count].val = val;
            freqs[unique_count].count = 1;
            unique_count++;
        }
    }
    
    // Print most frequent values
    for (int i = 0; i < unique_count && i < 10; i++) {
        printf("  Value %d: %d times\n", freqs[i].val, freqs[i].count);
    }
}


#define HISTOGRAM_BINS 10
#define HISTOGRAM_MAX_LEN 100
void display_tensor_distribution_histogram(const char* name, elem_t* data, int size) {
    printf("\n--- VALUE DISTRIBUTION: %s ---\n", name);
    
    // find min and max for range
    elem_t min_val = data[0];
    elem_t max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    printf("Value Range: [%d, %d]\n", min_val, max_val);

    int bins[HISTOGRAM_BINS] = {0};
    float step = (float)(max_val - min_val) / HISTOGRAM_BINS; // Avoid division by zero
    for (int i = 0; i < size; i++) {
        elem_t val = data[i];
        for (int b = 0; b < HISTOGRAM_BINS; b++) {
            if (val <= min_val + step * (b + 1)) {
                bins[b]++;
                break;
            }
        }
    }

    int histogram_length = 0;
    for (int b = 0; b < HISTOGRAM_BINS; b++) {
        if (bins[b] > histogram_length) histogram_length = bins[b];
    }

    for (int b = 0; b < HISTOGRAM_BINS; b++) {
        if (bins[b] == 0) continue; // Skip empty bins
        printf("  Bin %d [%d, %d]: %d \t", b, (int)(min_val + step * b), (int)(min_val + step * (b + 1)), bins[b]);
        int bar_length = (bins[b] * HISTOGRAM_MAX_LEN) / histogram_length; // Scale to max 50 chars
        for (int i = 0; i < bar_length; i++) {
            printf("*");
        }
        printf("\n");
    }
}

void print_tensor(elem_t* data, int rows, int cols, const char* name) {
    printf("\n--- TENSOR: %s ---\n", name);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%6d ", data[r * cols + c]);
        }
        printf("\n");
    }
}

#endif

