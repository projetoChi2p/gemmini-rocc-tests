#include "../inc/utils.h"
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

void print_results_summary(int total_samples, int correct_predictions, uint64_t total_cycles) {
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

    printf("------------------------------------------\n");
    printf(" Total Cycles      : %llu\n", total_cycles);
    printf(" Avg Cycles/Inf    : %.0f\n", avg_cycles);
    printf("==========================================\n");
}