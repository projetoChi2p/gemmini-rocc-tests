#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

// Include the generated header with weight definitions
#include "includes/cnnt3_sat6_params.h"

#define BATCH_SIZE 1
#define TOLERANCE 1

// ==========================================
// 1. DEBUG HELPER
// ==========================================

bool verify_tensor(const char* name, elem_t* hw_ptr, elem_t* sw_ptr, int size, int tolerance) {
    printf("\n--- VERIFY: %s ---\n", name);
    printf("   Sample [0]: HW %d vs SW %d\n", hw_ptr[0], sw_ptr[0]);
    printf("   Sample [1]: HW %d vs SW %d\n", hw_ptr[1], sw_ptr[1]);

    #ifdef QUANTIZED
        int scale = 1;
    #else
        int scale = 100; // Scale for printing floats as integers (e.g., 3 decimal places)
    #endif

    int errs = 0;
    int max_diff = 0;
    
    // Histogram buckets: [0], [1-2], [3-5], [6-10], [>10]
    int hist[5] = {0, 0, 0, 0, 0};
    
    for (int i = 0; i < size; i++) {
        // Calculate diff (handle potential float vs int types via casting/scaling if needed)
        int val_hw = (int)(hw_ptr[i] * scale);
        int val_sw = (int)(sw_ptr[i] * scale);
        int diff = abs(val_hw - val_sw);
        
        if (diff > max_diff) max_diff = diff;

        // Fill Histogram
        if (diff == 0)      hist[0]++;
        else if (diff <= 2) hist[1]++;
        else if (diff <= 5) hist[2]++;
        else if (diff <= 10) hist[3]++;
        else                hist[4]++;

        // Check Tolerance
        if (diff > tolerance * scale) { // Scale tolerance to match value scaling
            if (errs < 10) { // Limit detailed prints
                #ifdef QUANTIZED
                    printf("   [FAIL] Idx %d: HW %d vs SW %d (Diff %d)\n", i, val_hw, val_sw, diff);
                #else
                    printf("   [FAIL] Idx %d: HW %d vs SW %d (Diff %d)\n", i, val_hw, val_sw, diff);
                #endif
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
// 2. HELPER FUNCTIONS
// ==========================================

elem_t argmax(int size, elem_t * input) {
    elem_t max_val = input[0];
    int max_index = 0;
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_index = i;
        }
    }
    return max_index;
}

// Helper to re-arrange weights for Gemmini (Out, K, K, In) -> (K*K*In, Out)
void flatten_weights(int out_channels, int kernel_dim, int in_channels,
        elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
        elem_t weights_mat[kernel_dim * kernel_dim * in_channels][out_channels]) {

    for (int outc = 0; outc < out_channels; outc++) {
        for (int krow = 0; krow < kernel_dim; krow++) {
            for (int kcol = 0; kcol < kernel_dim; kcol++) {
                for (int inc = 0; inc < in_channels; inc++) {
                    int wmatrow = krow * kernel_dim * in_channels +
                        kcol * in_channels +
                        inc;
                    weights_mat[wmatrow][outc] = weights[outc][krow][kcol][inc];
                }
            }
        }
    }
}

// Global Average Pooling: (Batch, H, W, C) -> (Batch, C)
void global_average_pool(int batch_size, int rows, int cols, int channels,
                         elem_t input[batch_size][rows][cols][channels],
                         elem_t output[batch_size][channels]) {
    
    int num_elements = rows * cols;
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            int32_t sum = 0;
            for (int r = 0; r < rows; r++) {
                for (int col = 0; col < cols; col++) {
                    sum += input[b][r][col][c];
                }
            }
            output[b][c] = (elem_t)(sum / num_elements);
        }
    }
}

// ==========================================
// 3. MEMORY ALLOCATION
// ==========================================

// Flattened Weight Buffers (Gemmini format)
static elem_t w1_flat[L1_KERNEL*L1_KERNEL*IN_CHANNELS][L1_OUT_CH];
static elem_t w2_flat[L2_KERNEL*L2_KERNEL*L1_OUT_CH][L2_OUT_CH];
static elem_t w3_flat[L3_KERNEL*L3_KERNEL*L2_OUT_CH][L3_OUT_CH];

// Accumulator Biases (Promoted from elem_t to acc_t)
static acc_t b1_acc[L1_OUT_CH];
static acc_t b2_acc[L2_OUT_CH];
static acc_t b3_acc[L3_OUT_CH];

// Intermediate Feature Maps
static elem_t l1_out[BATCH_SIZE][L1_OUT_DIM][L1_OUT_DIM][L1_OUT_CH];
static elem_t l2_out[BATCH_SIZE][L2_OUT_DIM][L2_OUT_DIM][L2_OUT_CH];
static elem_t l3_out[BATCH_SIZE][L3_OUT_DIM][L3_OUT_DIM][L3_OUT_CH];

// Final Output
static elem_t final_preds[BATCH_SIZE][NUM_CLASSES];

// ==========================================
// 4. MAIN EXECUTION
// ==========================================

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif
    int errors = 0;
    int correct_predictions = 0;

    gemmini_flush(0);
    printf("Initializing SAT6 Network (Batch Size: %d)...\n", BATCH_SIZE);

    // ------------------------------------------------
    // 1. One-time Setup (Weights & Biases)
    // ------------------------------------------------
    
    // Flatten Weights
    //flatten_weights(L1_OUT_CH, L1_KERNEL, IN_CHANNELS, conv1_w, w1_flat);
    //flatten_weights(L2_OUT_CH, L2_KERNEL, L1_OUT_CH, conv2_w, w2_flat);
    //flatten_weights(L3_OUT_CH, L3_KERNEL, L2_OUT_CH, conv3_w, w3_flat);

    // Promote Biases
    //for(int i=0; i<L1_OUT_CH; i++) b1_acc[i] = (acc_t)conv1_b[i];
    //for(int i=0; i<L2_OUT_CH; i++) b2_acc[i] = (acc_t)conv2_b[i];
    //for(int i=0; i<L3_OUT_CH; i++) b3_acc[i] = (acc_t)conv3_b[i];

    // ------------------------------------------------
    // 2. Inference Loop
    // ------------------------------------------------
    // Loop over the 64 images available in the header
    for (int i = 0; i < 64; i++) {
        
        elem_t * current_image_ptr = (elem_t*)all_input_images[i];
        int ground_truth = all_ground_truths[i];

        if (i == 0) printf("Starting Inference Loop...\n");
        // uint64_t start = read_cycles();

        // ------------------------------------------------
        // LAYER 1: Conv + ReLU
        // ------------------------------------------------
        tiled_conv_auto(
            BATCH_SIZE, IN_DIM, IN_DIM, IN_CHANNELS,
            L1_OUT_CH, L1_OUT_DIM, L1_OUT_DIM,
            L1_STRIDE, 1, 1, L1_PAD, L1_KERNEL,
            false, false, false, false, false,

            (elem_t*)current_image_ptr,
            (elem_t*)conv1_w,
            (acc_t*)conv1_b,
            (elem_t*)l1_out,

            RELU, ACC_SCALE_IDENTITY, 0, 0, 0, WS
        );

        #ifdef DEBUG_CNNT3
        if (i == 0) {
            verify_tensor("Layer 1 Output", 
                (elem_t*)l1_out, (elem_t*)debug_l1_out, 
                BATCH_SIZE * L1_OUT_DIM * L1_OUT_DIM * L1_OUT_CH, TOLERANCE);
        }
        #endif

        // ------------------------------------------------
        // LAYER 2: Conv + ReLU
        // ------------------------------------------------
        tiled_conv_auto(
            BATCH_SIZE, L1_OUT_DIM, L1_OUT_DIM, L1_OUT_CH,
            L2_OUT_CH, L2_OUT_DIM, L2_OUT_DIM,
            L2_STRIDE, 1, 1, L2_PAD, L2_KERNEL,
            false, false, false, false, false,

            (elem_t*)l1_out,
            (elem_t*)conv2_w,
            (acc_t*)conv2_b,
            (elem_t*)l2_out,

            RELU, ACC_SCALE_IDENTITY, 0, 0, 0, WS
        );

        #ifdef DEBUG_CNNT3
        if (i == 0) {
            verify_tensor("Layer 2 Output", 
                (elem_t*)l2_out, (elem_t*)debug_l2_out, 
                BATCH_SIZE * L2_OUT_DIM * L2_OUT_DIM * L2_OUT_CH, TOLERANCE);
        }
        #endif

        // ------------------------------------------------
        // LAYER 3: Conv + ReLU
        // ------------------------------------------------
        tiled_conv_auto(
            BATCH_SIZE, L2_OUT_DIM, L2_OUT_DIM, L2_OUT_CH,
            L3_OUT_CH, L3_OUT_DIM, L3_OUT_DIM,
            L3_STRIDE, 1, 1, L3_PAD, L3_KERNEL,
            false, false, false, false, false,

            (elem_t*)l2_out,
            (elem_t*)conv3_w,
            (acc_t*)conv3_b,
            (elem_t*)l3_out,

            RELU, ACC_SCALE_IDENTITY, 0, 0, 0, WS
        );

        #ifdef DEBUG_CNNT3
        if (i == 0) {
            verify_tensor("Layer 3 Output", 
                (elem_t*)l3_out, (elem_t*)debug_l3_out, 
                BATCH_SIZE * L3_OUT_DIM * L3_OUT_DIM * L3_OUT_CH, TOLERANCE);
        }
        #endif

        // ------------------------------------------------
        // GLOBAL POOLING
        // ------------------------------------------------
        global_average_pool(BATCH_SIZE, L3_OUT_DIM, L3_OUT_DIM, L3_OUT_CH, 
                            l3_out, final_preds);

        #ifdef DEBUG_CNNT3
        if (i == 0) {
            verify_tensor("Final Output", 
                (elem_t*)final_preds, (elem_t*)debug_final_out, 
                BATCH_SIZE * NUM_CLASSES, TOLERANCE);
        }
        #endif

        // uint64_t end = read_cycles();
        
        // ------------------------------------------------
        // Validation 
        // ------------------------------------------------
        int prediction = argmax(NUM_CLASSES, final_preds[0]);
        
        if (prediction != ground_truth) {
            if (errors < 10) // Limit the number of printed errors
                printf("Mismatch at image %d: Predicted %d vs Ground Truth %d\n", 
                    i, prediction, ground_truth);
            errors++;
        } else {
            // printf("Correct prediction for image %d: Predicted %d\n", i, prediction);
            correct_predictions++;
        }
    }

    printf("Total mismatches: %d / 64\n", errors);
    printf("Total correct predictions: %d / 64\n", correct_predictions);

    return 0;
}