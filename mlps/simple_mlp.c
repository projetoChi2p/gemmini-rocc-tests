#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// Include our new, auto-generated parameter file
#include "simple_mlp_params.h"

// Helper function to find the prediction
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


int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      return 1;
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type = WS;

    /*
    if (argc < 2) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        return 0;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        return 1;
    }*/

    bool check = false;
    /*
    if (argc < 3) {
        check = false;
    } else if (strcmp(argv[2], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        return 1;
    }*/

    // This network has 3 layers, so 3 cycle counts
    uint64_t cycles[3] = {0};
    uint64_t start, end;

    printf("--- Starting SimpleMLP Inference --- \n");
    printf("");
    printf("Batch size: %d, Input: %d, Hidden1: %d, Hidden2: %d, Output: %d\n",
        BATCH_SIZE, INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);

    /*printf("input: \n");
    for (int i = 0; i < 1; i ++){
        for (int j = 0; j < INPUT_SIZE; j ++){
            printf("%x ", elem_t_to_floats(input_mat[i][j]));
            if ((j + 1) % 28 == 0){
                printf("\n");
            }
        }
    }*/
    
    /* Layer 0: fc1 (Input -> Hidden1) */
    start = read_cycles();

    // M=BATCH_SIZE, N=HIDDEN1_SIZE, K=INPUT_SIZE
    // C[M, N] = A[M, K] * B[K, N]
    // A = input_mat, B = weights0, C = inter_results0
    tiled_matmul_nn_auto(BATCH_SIZE, HIDDEN1_SIZE, INPUT_SIZE,
        input_mat, weights0, NULL, inter_results0,
        RELU, 1.0f, false, // Apply RELU activation
        tiled_matmul_type, check, "layer_0_fc1");

    end = read_cycles();
    cycles[0] = end-start;

    /*printf("intermediate 0: \n");
    for (int i = 0; i < 1; i ++){
        for (int j = 0; j < HIDDEN1_SIZE; j ++){
            printf("%x ", elem_t_to_floats(inter_results0[i][j]));
        }
    }*/

    /* Layer 1: fc2 (Hidden1 -> Hidden2) */
    start = read_cycles();

    // M=BATCH_SIZE, N=HIDDEN2_SIZE, K=HIDDEN1_SIZE
    // A = inter_results0, B = weights1, C = inter_results1
    tiled_matmul_nn_auto(BATCH_SIZE, HIDDEN2_SIZE, HIDDEN1_SIZE,
        inter_results0, weights1, NULL, inter_results1,
        RELU, 1.0f, false, // Apply RELU activation
        tiled_matmul_type, check, "layer_1_fc2");

    end = read_cycles();
    cycles[1] = end-start;
    
    /*printf("intermediate 1: \n");
    for (int i = 0; i < 1; i ++){
        for (int j = 0; j < HIDDEN2_SIZE; j ++){
            printf("%x ", elem_t_to_floats(inter_results1[i][j]));
        }
    }*/

    /* Layer 2: fc3 (Hidden2 -> Output) */
    start = read_cycles();

    // M=BATCH_SIZE, N=OUTPUT_SIZE, K=HIDDEN2_SIZE
    // A = inter_results1, B = weights2, C = output_results
    tiled_matmul_nn_auto(BATCH_SIZE, OUTPUT_SIZE, HIDDEN2_SIZE,
        inter_results1, weights2, NULL, output_results,
        NO_ACTIVATION, 1.0f, false, // No activation on the final logits
        tiled_matmul_type, check, "layer_2_fc3");

    end = read_cycles();
    cycles[2] = end-start;

    /*for (int i = 0; i < 1; i ++){
        for (int j = 0; j < OUTPUT_SIZE; j ++){
            printf("%x ", elem_t_to_floats(output_results[i][j]));
        }
    }*/

    printf("--- Inference Finished --- \n");

    // --- Print Cycle Counts ---
    uint64_t overall_cycles = 0;
    for(int cyc = 0; cyc < 3 ; cyc++){
        overall_cycles += cycles[cyc];
        printf("Cycles taken in layer %d: %llu\n", cyc, cycles[cyc]);
    }
    printf("Overall cycles taken: %llu\n", overall_cycles);

    // --- Verify Prediction ---
    // We only do this for BATCH_SIZE = 1
    if (BATCH_SIZE == 1) {
        int prediction = find_max_index((elem_t*)output_results, OUTPUT_SIZE);
        printf("\n--- Verification --- \n");
        printf("Prediction: %d\n", prediction);
        printf("Ground Truth: %d\n", ground_truth);

        if (prediction == ground_truth) {
            printf("SUCCESS: Prediction matches ground truth!\n");
        } else {
            printf("FAILURE: Prediction does not match ground truth.\n");
        }
    } else {
        printf("\nSet BATCH_SIZE=1 in 02-export.py to verify prediction.\n");
    }

    return 0;
}
