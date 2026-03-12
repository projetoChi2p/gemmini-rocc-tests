
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define BATCH_SIZE 1


#include "utils_refact.c"
#ifdef ELEM_T_IS_FLOAT
#include "includes/cnnt2_cifar100_params.h"
#else
#include "includes/cnnt2_cifar100_quant_params.h"
#endif  


// --- Memory Allocation ---
static elem_t current_image_ptr[3072];
static elem_t conv1_w_flat[3*3*3][16];
static elem_t layer0_conv1_out[4096];
static elem_t conv2_w_flat[3*3*16][32];
static elem_t layer1_conv2_out[2048];
static elem_t conv3_w_flat[3*3*32][16];
static elem_t layer2_conv3_out[256];
static elem_t conv4_w_flat[3*3*16][100];
static elem_t layer3_conv4_out[400];
static elem_t layer4_global_pool_out[100];

int main() {
    #ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
    #endif
    flatten_weights(16, 3, 3, (elem_t*)conv1_w, (elem_t*)conv1_w_flat);
    flatten_weights(32, 3, 16, (elem_t*)conv2_w, (elem_t*)conv2_w_flat);
    flatten_weights(16, 3, 32, (elem_t*)conv3_w, (elem_t*)conv3_w_flat);
    flatten_weights(100, 3, 16, (elem_t*)conv4_w, (elem_t*)conv4_w_flat);
    gemmini_flush(0);
    printf("Initializing Network...\n");
    int correct_predictions = 0;
    int errors = 0;
    for (int i = 0; i < NUM_INFERENCES; i++) {
        int ground_truth = all_ground_truths[i];
        elem_t* current_image_ptr = (elem_t*)all_input_images[i];
        tiled_conv_auto(1, 32, 32, 3, 16, 16, 16, 2, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)current_image_ptr, (elem_t*)conv1_w_flat, (acc_t*)conv1_b, (elem_t*)layer0_conv1_out, RELU, (acc_scale_t)CONV1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV1_AVAILABLE)
        if (i == 0) {
            verify_array("conv1", 4096, (elem_t*)layer0_conv1_out, (elem_t*)debug_conv1);
        }
#endif
        tiled_conv_auto(1, 16, 16, 16, 32, 8, 8, 2, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer0_conv1_out, (elem_t*)conv2_w_flat, (acc_t*)conv2_b, (elem_t*)layer1_conv2_out, RELU, (acc_scale_t)CONV2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV2_AVAILABLE)
        if (i == 0) {
            verify_array("conv2", 2048, (elem_t*)layer1_conv2_out, (elem_t*)debug_conv2);
        }
#endif
        tiled_conv_auto(1, 8, 8, 32, 16, 4, 4, 2, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer1_conv2_out, (elem_t*)conv3_w_flat, (acc_t*)conv3_b, (elem_t*)layer2_conv3_out, RELU, (acc_scale_t)CONV3_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV3_AVAILABLE)
        if (i == 0) {
            verify_array("conv3", 256, (elem_t*)layer2_conv3_out, (elem_t*)debug_conv3);
        }
#endif
        tiled_conv_auto(1, 4, 4, 16, 100, 2, 2, 2, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer2_conv3_out, (elem_t*)conv4_w_flat, (acc_t*)conv4_b, (elem_t*)layer3_conv4_out, RELU, (acc_scale_t)CONV4_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV4_AVAILABLE)
        if (i == 0) {
            verify_array("conv4", 400, (elem_t*)layer3_conv4_out, (elem_t*)debug_conv4);
        }
#endif
        global_average_pool(1, 2, 2, 100, (elem_t*)layer3_conv4_out, (elem_t*)layer4_global_pool_out);
#if defined(DEBUG) && defined(DEBUG_GLOBAL_POOL_AVAILABLE)
        if (i == 0) {
            verify_array("global_pool", 100, (elem_t*)layer4_global_pool_out, (elem_t*)debug_global_pool);
        }
#endif
        int prediction = argmax(NUM_CLASSES, layer4_global_pool_out);
        if (prediction == ground_truth) {
            correct_predictions++;
        } else {
            errors++;
        }

    }
    printf("Total mismatches: %d / %d\n", errors, NUM_INFERENCES);
    printf("Correct predictions: %d / %d (%.2f%%)\n", correct_predictions, NUM_INFERENCES, (float)((correct_predictions * 100.0) / NUM_INFERENCES));
    printf("Done\n");
    return 0;
}