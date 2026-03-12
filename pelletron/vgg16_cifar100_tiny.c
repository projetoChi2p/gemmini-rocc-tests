
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
#include "includes/vgg16_cifar100_tiny_params.h"
#else
#include "includes/vgg16_cifar100_tiny_quant_params.h"
#endif


// --- Memory Allocation ---
static elem_t current_image_ptr[3072];
static elem_t conv1_1_w_flat[3*3*3][4];
static elem_t layer0_conv1_1_out[4096];
static elem_t conv1_2_w_flat[3*3*4][4];
static elem_t layer1_conv1_2_out[4096];
static elem_t layer2_pool1_out[1024];
static elem_t conv2_1_w_flat[3*3*4][8];
static elem_t layer3_conv2_1_out[2048];
static elem_t conv2_2_w_flat[3*3*8][8];
static elem_t layer4_conv2_2_out[2048];
static elem_t layer5_pool2_out[512];
static elem_t conv3_1_w_flat[3*3*8][16];
static elem_t layer6_conv3_1_out[1024];
static elem_t conv3_2_w_flat[3*3*16][16];
static elem_t layer7_conv3_2_out[1024];
static elem_t conv3_3_w_flat[3*3*16][16];
static elem_t layer8_conv3_3_out[1024];
static elem_t layer9_pool3_out[256];
static elem_t conv4_1_w_flat[3*3*16][32];
static elem_t layer10_conv4_1_out[512];
static elem_t conv4_2_w_flat[3*3*32][32];
static elem_t layer11_conv4_2_out[512];
static elem_t conv4_3_w_flat[3*3*32][32];
static elem_t layer12_conv4_3_out[512];
static elem_t layer13_pool4_out[128];
static elem_t conv5_1_w_flat[3*3*32][32];
static elem_t layer14_conv5_1_out[128];
static elem_t conv5_2_w_flat[3*3*32][32];
static elem_t layer15_conv5_2_out[128];
static elem_t conv5_3_w_flat[3*3*32][32];
static elem_t layer16_conv5_3_out[128];
static elem_t layer17_global_pool_out[32];
static elem_t layer18_classifier_out[100];

int main() {
    #ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
    #endif
    flatten_weights(4, 3, 3, (elem_t*)conv1_1_w, (elem_t*)conv1_1_w_flat);
    flatten_weights(4, 3, 4, (elem_t*)conv1_2_w, (elem_t*)conv1_2_w_flat);
    flatten_weights(8, 3, 4, (elem_t*)conv2_1_w, (elem_t*)conv2_1_w_flat);
    flatten_weights(8, 3, 8, (elem_t*)conv2_2_w, (elem_t*)conv2_2_w_flat);
    flatten_weights(16, 3, 8, (elem_t*)conv3_1_w, (elem_t*)conv3_1_w_flat);
    flatten_weights(16, 3, 16, (elem_t*)conv3_2_w, (elem_t*)conv3_2_w_flat);
    flatten_weights(16, 3, 16, (elem_t*)conv3_3_w, (elem_t*)conv3_3_w_flat);
    flatten_weights(32, 3, 16, (elem_t*)conv4_1_w, (elem_t*)conv4_1_w_flat);
    flatten_weights(32, 3, 32, (elem_t*)conv4_2_w, (elem_t*)conv4_2_w_flat);
    flatten_weights(32, 3, 32, (elem_t*)conv4_3_w, (elem_t*)conv4_3_w_flat);
    flatten_weights(32, 3, 32, (elem_t*)conv5_1_w, (elem_t*)conv5_1_w_flat);
    flatten_weights(32, 3, 32, (elem_t*)conv5_2_w, (elem_t*)conv5_2_w_flat);
    flatten_weights(32, 3, 32, (elem_t*)conv5_3_w, (elem_t*)conv5_3_w_flat);
    gemmini_flush(0);
    printf("Initializing Network...\n");
    int correct_predictions = 0;
    int errors = 0;
    for (int i = 0; i < NUM_INFERENCES; i++) {
        int ground_truth = all_ground_truths[i];
        elem_t* current_image_ptr = (elem_t*)all_input_images[i];
        tiled_conv_auto(1, 32, 32, 3, 4, 32, 32, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)current_image_ptr, (elem_t*)conv1_1_w_flat, (acc_t*)conv1_1_b, (elem_t*)layer0_conv1_1_out, RELU, (acc_scale_t)CONV1_1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV1_1_AVAILABLE)
        if (i == 0) {
            verify_array("conv1_1", 4096, (elem_t*)layer0_conv1_1_out, (elem_t*)debug_conv1_1);
        }
#endif
        tiled_conv_auto(1, 32, 32, 4, 4, 32, 32, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer0_conv1_1_out, (elem_t*)conv1_2_w_flat, (acc_t*)conv1_2_b, (elem_t*)layer1_conv1_2_out, RELU, (acc_scale_t)CONV1_2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV1_2_AVAILABLE)
        if (i == 0) {
            verify_array("conv1_2", 4096, (elem_t*)layer1_conv1_2_out, (elem_t*)debug_conv1_2);
        }
#endif
        tiled_maxpool_auto(1, 4, 32, 32, 2, 2, 0, (elem_t*)layer1_conv1_2_out, (elem_t*)layer2_pool1_out, WS);
#if defined(DEBUG) && defined(DEBUG_POOL1_AVAILABLE)
        if (i == 0) {
            verify_array("pool1", 1024, (elem_t*)layer2_pool1_out, (elem_t*)debug_pool1);
        }
#endif
        tiled_conv_auto(1, 16, 16, 4, 8, 16, 16, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer2_pool1_out, (elem_t*)conv2_1_w_flat, (acc_t*)conv2_1_b, (elem_t*)layer3_conv2_1_out, RELU, (acc_scale_t)CONV2_1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV2_1_AVAILABLE)
        if (i == 0) {
            verify_array("conv2_1", 2048, (elem_t*)layer3_conv2_1_out, (elem_t*)debug_conv2_1);
        }
#endif
        tiled_conv_auto(1, 16, 16, 8, 8, 16, 16, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer3_conv2_1_out, (elem_t*)conv2_2_w_flat, (acc_t*)conv2_2_b, (elem_t*)layer4_conv2_2_out, RELU, (acc_scale_t)CONV2_2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV2_2_AVAILABLE)
        if (i == 0) {
            verify_array("conv2_2", 2048, (elem_t*)layer4_conv2_2_out, (elem_t*)debug_conv2_2);
        }
#endif
        tiled_maxpool_auto(1, 8, 16, 16, 2, 2, 0, (elem_t*)layer4_conv2_2_out, (elem_t*)layer5_pool2_out, WS);
#if defined(DEBUG) && defined(DEBUG_POOL2_AVAILABLE)
        if (i == 0) {
            verify_array("pool2", 512, (elem_t*)layer5_pool2_out, (elem_t*)debug_pool2);
        }
#endif
        tiled_conv_auto(1, 8, 8, 8, 16, 8, 8, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer5_pool2_out, (elem_t*)conv3_1_w_flat, (acc_t*)conv3_1_b, (elem_t*)layer6_conv3_1_out, RELU, (acc_scale_t)CONV3_1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV3_1_AVAILABLE)
        if (i == 0) {
            verify_array("conv3_1", 1024, (elem_t*)layer6_conv3_1_out, (elem_t*)debug_conv3_1);
        }
#endif
        tiled_conv_auto(1, 8, 8, 16, 16, 8, 8, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer6_conv3_1_out, (elem_t*)conv3_2_w_flat, (acc_t*)conv3_2_b, (elem_t*)layer7_conv3_2_out, RELU, (acc_scale_t)CONV3_2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV3_2_AVAILABLE)
        if (i == 0) {
            verify_array("conv3_2", 1024, (elem_t*)layer7_conv3_2_out, (elem_t*)debug_conv3_2);
        }
#endif
        tiled_conv_auto(1, 8, 8, 16, 16, 8, 8, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer7_conv3_2_out, (elem_t*)conv3_3_w_flat, (acc_t*)conv3_3_b, (elem_t*)layer8_conv3_3_out, RELU, (acc_scale_t)CONV3_3_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV3_3_AVAILABLE)
        if (i == 0) {
            verify_array("conv3_3", 1024, (elem_t*)layer8_conv3_3_out, (elem_t*)debug_conv3_3);
        }
#endif
        tiled_maxpool_auto(1, 16, 8, 8, 2, 2, 0, (elem_t*)layer8_conv3_3_out, (elem_t*)layer9_pool3_out, WS);
#if defined(DEBUG) && defined(DEBUG_POOL3_AVAILABLE)
        if (i == 0) {
            verify_array("pool3", 256, (elem_t*)layer9_pool3_out, (elem_t*)debug_pool3);
        }
#endif
        tiled_conv_auto(1, 4, 4, 16, 32, 4, 4, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer9_pool3_out, (elem_t*)conv4_1_w_flat, (acc_t*)conv4_1_b, (elem_t*)layer10_conv4_1_out, RELU, (acc_scale_t)CONV4_1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV4_1_AVAILABLE)
        if (i == 0) {
            verify_array("conv4_1", 512, (elem_t*)layer10_conv4_1_out, (elem_t*)debug_conv4_1);
        }
#endif
        tiled_conv_auto(1, 4, 4, 32, 32, 4, 4, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer10_conv4_1_out, (elem_t*)conv4_2_w_flat, (acc_t*)conv4_2_b, (elem_t*)layer11_conv4_2_out, RELU, (acc_scale_t)CONV4_2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV4_2_AVAILABLE)
        if (i == 0) {
            verify_array("conv4_2", 512, (elem_t*)layer11_conv4_2_out, (elem_t*)debug_conv4_2);
        }
#endif
        tiled_conv_auto(1, 4, 4, 32, 32, 4, 4, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer11_conv4_2_out, (elem_t*)conv4_3_w_flat, (acc_t*)conv4_3_b, (elem_t*)layer12_conv4_3_out, RELU, (acc_scale_t)CONV4_3_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV4_3_AVAILABLE)
        if (i == 0) {
            verify_array("conv4_3", 512, (elem_t*)layer12_conv4_3_out, (elem_t*)debug_conv4_3);
        }
#endif
        tiled_maxpool_auto(1, 32, 4, 4, 2, 2, 0, (elem_t*)layer12_conv4_3_out, (elem_t*)layer13_pool4_out, WS);
#if defined(DEBUG) && defined(DEBUG_POOL4_AVAILABLE)
        if (i == 0) {
            verify_array("pool4", 128, (elem_t*)layer13_pool4_out, (elem_t*)debug_pool4);
        }
#endif
        tiled_conv_auto(1, 2, 2, 32, 32, 2, 2, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer13_pool4_out, (elem_t*)conv5_1_w_flat, (acc_t*)conv5_1_b, (elem_t*)layer14_conv5_1_out, RELU, (acc_scale_t)CONV5_1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV5_1_AVAILABLE)
        if (i == 0) {
            verify_array("conv5_1", 128, (elem_t*)layer14_conv5_1_out, (elem_t*)debug_conv5_1);
        }
#endif
        tiled_conv_auto(1, 2, 2, 32, 32, 2, 2, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer14_conv5_1_out, (elem_t*)conv5_2_w_flat, (acc_t*)conv5_2_b, (elem_t*)layer15_conv5_2_out, RELU, (acc_scale_t)CONV5_2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV5_2_AVAILABLE)
        if (i == 0) {
            verify_array("conv5_2", 128, (elem_t*)layer15_conv5_2_out, (elem_t*)debug_conv5_2);
        }
#endif
        tiled_conv_auto(1, 2, 2, 32, 32, 2, 2, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer15_conv5_2_out, (elem_t*)conv5_3_w_flat, (acc_t*)conv5_3_b, (elem_t*)layer16_conv5_3_out, RELU, (acc_scale_t)CONV5_3_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV5_3_AVAILABLE)
        if (i == 0) {
            verify_array("conv5_3", 128, (elem_t*)layer16_conv5_3_out, (elem_t*)debug_conv5_3);
        }
#endif
        global_average_pool(1, 2, 2, 32, (elem_t*)layer16_conv5_3_out, (elem_t*)layer17_global_pool_out);
#if defined(DEBUG) && defined(DEBUG_GLOBAL_POOL_AVAILABLE)
        if (i == 0) {
            verify_array("global_pool", 32, (elem_t*)layer17_global_pool_out, (elem_t*)debug_global_pool);
        }
#endif
        tiled_matmul_auto(1, 100, 32, (elem_t*)layer17_global_pool_out, (elem_t*)classifier_w, (acc_t*)classifier_b, (elem_t*)layer18_classifier_out, 32, 100, 100, 100, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, NO_ACTIVATION, (acc_scale_t)CLASSIFIER_SCALE, ACC_SCALE_IDENTITY, true, false, false, false, false, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CLASSIFIER_AVAILABLE)
        if (i == 0) {
            verify_array("classifier", 100, (elem_t*)layer18_classifier_out, (elem_t*)debug_classifier);
        }
#endif
        int prediction = argmax(NUM_CLASSES, layer18_classifier_out);
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