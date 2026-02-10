#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

// Include the generated header
#include "includes/cnnt3_sat6_quant_params.h"

#define BATCH_SIZE 1
#define TOLERANCE 1
#define QUANTIZED 

#include "utils.c"
// ==========================================
// 3. MEMORY ALLOCATION
// ==========================================
// Add this helper function back (copied from conv.c)

// Global Buffers for Flattened Weights
static elem_t w1_mat[L1_KERNEL*L1_KERNEL*IN_CHANNELS][L1_OUT_CH];
static elem_t w2_mat[L2_KERNEL*L2_KERNEL*L1_OUT_CH][L2_OUT_CH];
static elem_t w3_mat[L3_KERNEL*L3_KERNEL*L2_OUT_CH][L3_OUT_CH];

// Intermediate Feature Maps
static elem_t l1_out[BATCH_SIZE][L1_OUT_DIM][L1_OUT_DIM][L1_OUT_CH];
static elem_t l2_out[BATCH_SIZE][L2_OUT_DIM][L2_OUT_DIM][L2_OUT_CH];
static elem_t l3_out[BATCH_SIZE][L3_OUT_DIM][L3_OUT_DIM][L3_OUT_CH];

// Final Output
static elem_t final_preds[BATCH_SIZE][NUM_CLASSES];

#undef DEBUG_CNNT3

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
    
    flatten_weights(L1_OUT_CH, L1_KERNEL, IN_CHANNELS, conv1_w, w1_mat);
    flatten_weights(L2_OUT_CH, L2_KERNEL, L1_OUT_CH, conv2_w, w2_mat);
    flatten_weights(L3_OUT_CH, L3_KERNEL, L2_OUT_CH, conv3_w, w3_mat);

    gemmini_flush(0);
    printf("Initializing SAT6 Network (Quantized, Batch Size: %d)...\n", BATCH_SIZE);

    // ------------------------------------------------
    // Inference Loop
    // ------------------------------------------------
    for (int i = 0; i < 64; i++) {
        elem_t * current_image_ptr = (elem_t*)all_input_images[i];
        
        int ground_truth = all_ground_truths[i];

        if (i == 0) printf("Starting Inference Loop...\n");

        // ------------------------------------------------
        // LAYER 1: Conv + ReLU (Scaled)
        // ------------------------------------------------
        tiled_conv_auto(
            BATCH_SIZE, IN_DIM, IN_DIM, IN_CHANNELS,
            L1_OUT_CH, L1_OUT_DIM, L1_OUT_DIM,
            L1_STRIDE, 1, 1, L1_PAD, L1_KERNEL,
            false, false, false, false, false,

            (elem_t*)current_image_ptr,
            (elem_t*)w1_mat,   // Pass flattened weights
            (acc_t*)conv1_b,    // Pass header pointer directly (Must be 32-bit in header)
            (elem_t*)l1_out,

            RELU, 
            (acc_scale_t)L1_SCALE, 
            0, 0, 0, 
            WS
        );

        #ifdef DEBUG_CNNT3
        if (i == 0) {
            verify_tensor("Layer 1 Output (Quantized)", 
                (elem_t*)l1_out, (elem_t*)debug_l1_out, 
                BATCH_SIZE * L1_OUT_DIM * L1_OUT_DIM * L1_OUT_CH, TOLERANCE);
            //display_tensor("Layer 1 Output (Quantized)", (elem_t*)l1_out, BATCH_SIZE * L1_OUT_DIM * L1_OUT_DIM * L1_OUT_CH);
            display_tensor_distribution_histogram("Layer 1 Output Distribution (Quantized)", 
                (elem_t*)l1_out, BATCH_SIZE * L1_OUT_DIM * L1_OUT_DIM * L1_OUT_CH);

            display_tensor_distribution_histogram("Layer 1 Output Distribution (Quantized) - Debug", 
                (elem_t*)debug_l1_out, BATCH_SIZE * L1_OUT_DIM * L1_OUT_DIM * L1_OUT_CH);
        }
        #endif

        // ------------------------------------------------
        // LAYER 2: Conv + ReLU (Scaled)
        // ------------------------------------------------
        tiled_conv_auto(
            BATCH_SIZE, L1_OUT_DIM, L1_OUT_DIM, L1_OUT_CH,
            L2_OUT_CH, L2_OUT_DIM, L2_OUT_DIM,
            L2_STRIDE, 1, 1, L2_PAD, L2_KERNEL,
            false, false, false, false, false,

            (elem_t*)l1_out,
            (elem_t*)w2_mat,   // Pass flattened weights
            (acc_t*)conv2_b,    // Pass header pointer directly
            (elem_t*)l2_out,

            RELU, 
            (acc_scale_t)L2_SCALE,
            0, 0, 0, 
            WS
        );

        #ifdef DEBUG_CNNT3
        if (i == 0) {
            verify_tensor("Layer 2 Output (Quantized)", 
                (elem_t*)l2_out, (elem_t*)debug_l2_out, 
                BATCH_SIZE * L2_OUT_DIM * L2_OUT_DIM * L2_OUT_CH, TOLERANCE);
            display_tensor_distribution_histogram("Layer 2 Output Distribution (Quantized)", 
                (elem_t*)l2_out, BATCH_SIZE * L2_OUT_DIM * L2_OUT_DIM * L2_OUT_CH);
        }
        #endif

        // ------------------------------------------------
        // LAYER 3: Conv + ReLU (Scaled)
        // ------------------------------------------------
        tiled_conv_auto(
            BATCH_SIZE, L2_OUT_DIM, L2_OUT_DIM, L2_OUT_CH,
            L3_OUT_CH, L3_OUT_DIM, L3_OUT_DIM,
            L3_STRIDE, 1, 1, L3_PAD, L3_KERNEL,
            false, false, false, false, false,

            (elem_t*)l2_out,
            (elem_t*)w3_mat,   // Pass flattened weights
            (acc_t*)conv3_b,    // Pass header pointer directly
            (elem_t*)l3_out,

            RELU, 
            (acc_scale_t)L3_SCALE,
            0, 0, 0, 
            WS
        );

        #ifdef DEBUG_CNNT3
        if (i == 0) {
            verify_tensor("Layer 3 Output (Quantized)", 
                (elem_t*)l3_out, (elem_t*)debug_l3_out, 
                BATCH_SIZE * L3_OUT_DIM * L3_OUT_DIM * L3_OUT_CH, TOLERANCE);
            display_tensor_distribution_histogram("Layer 3 Output Distribution (Quantized)", 
                (elem_t*)l3_out, BATCH_SIZE * L3_OUT_DIM * L3_OUT_DIM * L3_OUT_CH);
        }
        #endif

        // ------------------------------------------------
        // GLOBAL POOLING
        // ------------------------------------------------
        global_average_pool(BATCH_SIZE, L3_OUT_DIM, L3_OUT_DIM, L3_OUT_CH, 
                            l3_out, final_preds);

        #ifdef DEBUG_CNNT3
        if (i == 0) {
            verify_tensor("Final Output (Quantized)", 
                (elem_t*)final_preds, (elem_t*)debug_final_out, 
                BATCH_SIZE * NUM_CLASSES, TOLERANCE);
        }
            
            //display_tensor("Final Predictions (Quantized)", (elem_t*)final_preds, BATCH_SIZE * NUM_CLASSES);
        #endif

        
        // ------------------------------------------------
        // Validation 
        // ------------------------------------------------
        int prediction = argmax(NUM_CLASSES, final_preds[0]);
        
        if (prediction != ground_truth) {
            if (errors < 10) 
                printf("Mismatch at image %d: Predicted %d vs Ground Truth %d\n", 
                    i, prediction, ground_truth);
            errors++;
        } else {
            //printf("Correct prediction for image %d: Predicted %d\n", i, prediction);
            correct_predictions++;
        }
    }

    printf("Total mismatches: %d / 64\n", errors);
    printf("Correct predictions: %d / 64 (%.2f%%)\n", correct_predictions, (float)correct_predictions / 64.0 * 100.0);
    
    return 0;
}