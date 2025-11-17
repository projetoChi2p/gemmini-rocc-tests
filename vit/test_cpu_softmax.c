#include <stdio.h>
#include <string.h>
#include "include/gemmini.h" // Para elem_t
#include "includes/cpu_softmax_test_data.h"
#include "includes/test_utils.h"

#include "transformer_layers.c" // Incluindo impl para testar

static elem_t buffer[NUM_HEADS * SEQ_LEN * SEQ_LEN];
static elem_t output[NUM_HEADS * SEQ_LEN * SEQ_LEN];

int main() {
    printf("=== Teste Unitario: CPU Softmax ===\n");
    
    // Flatten and copy input data from 4D array to 1D buffer
    // raw_scores[num_heads][1][seq_len][seq_len] -> buffer[num_heads * seq_len * seq_len]
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < SEQ_LEN; j++) {
                buffer[h * SEQ_LEN * SEQ_LEN + i * SEQ_LEN + j] = raw_scores[h][0][i][j];
            }
        }
    }

    // Execute softmax for each head
    // Each head has SEQ_LEN rows of SEQ_LEN columns
    int total_rows = NUM_HEADS * SEQ_LEN;
    cpu_softmax(total_rows, SEQ_LEN, buffer, output);
    printf("");

    // Flatten expected_probs for comparison
    elem_t expected_flat[NUM_HEADS * SEQ_LEN * SEQ_LEN];
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < SEQ_LEN; j++) {
                expected_flat[h * SEQ_LEN * SEQ_LEN + i * SEQ_LEN + j] = expected_probs[h][0][i][j];
            }
        }
    }

    bool pass = check_matrix("Softmax Output", 
                 (float*)output, 
                 (float*)expected_flat, 
                 total_rows, SEQ_LEN, 
                 0.5f); 

    printf("Test %s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
/*
// Test with known values
void main() {
    float test_input[] = {1.0f, 2.0f, 3.0f};
    float test_output[3];
    
    cpu_softmax(1, 3, test_input, test_output);
    
    printf("Input: [%de-6, %de-6, %de-6]\n", 
        elem_t_to_floats(test_input[0]*1000000), 
        elem_t_to_floats(test_input[1]*1000000), 
        elem_t_to_floats(test_input[2]*1000000)
    );
    printf("Output: [%de-6, %de-6, %de-6]\n", 
        elem_t_to_floats(test_output[0]*1000000), 
        elem_t_to_floats(test_output[1]*1000000), 
        elem_t_to_floats(test_output[2]*1000000)
    );
    
    // Expected roughly: [0.0900, 0.2447, 0.6652]
}*/