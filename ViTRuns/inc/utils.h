#ifndef UTILS_H
#define UTILS_H

#include <stdint.h> // For uint64_t
#include "include/gemmini_nn.h"

// Approximations
float gelu_exp_approx(float x);
void cpu_gelu_approx(int rows, int cols, elem_t * input, elem_t * output);

// Helpers
int find_max_index(elem_t* arr, int size);
void print_results_summary(int total_samples, int correct_predictions, uint64_t total_cycles);

#endif