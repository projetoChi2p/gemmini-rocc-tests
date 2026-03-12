
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
#include "includes/resnet18_cifar10_tiny_params.h"
#else
#include "includes/resnet18_cifar10_tiny_quant_params.h"
#endif


// --- Memory Allocation ---
static elem_t current_image_ptr[3072];
static elem_t conv1_w_flat[3*3*3][4];
static elem_t layer0_conv1_out[4096];
static elem_t layer1_0_conv1_w_flat[3*3*4][4];
static elem_t layer1_layer1_0_conv1_out[4096];
static elem_t layer1_0_conv2_w_flat[3*3*4][4];
static elem_t layer2_layer1_0_conv2_out[4096];
static elem_t layer3_add_out[4096];
static elem_t layer1_1_conv1_w_flat[3*3*4][4];
static elem_t layer4_layer1_1_conv1_out[4096];
static elem_t layer1_1_conv2_w_flat[3*3*4][4];
static elem_t layer5_layer1_1_conv2_out[4096];
static elem_t layer6_add_1_out[4096];
static elem_t layer2_0_conv1_w_flat[3*3*4][8];
static elem_t layer7_layer2_0_conv1_out[2048];
static elem_t layer2_0_conv2_w_flat[3*3*8][8];
static elem_t layer8_layer2_0_conv2_out[2048];
static elem_t layer2_0_shortcut_0_w_flat[1*1*4][8];
static elem_t layer9_layer2_0_shortcut_0_out[2048];
static elem_t layer10_add_2_out[2048];
static elem_t layer2_1_conv1_w_flat[3*3*8][8];
static elem_t layer11_layer2_1_conv1_out[2048];
static elem_t layer2_1_conv2_w_flat[3*3*8][8];
static elem_t layer12_layer2_1_conv2_out[2048];
static elem_t layer13_add_3_out[2048];
static elem_t layer3_0_conv1_w_flat[3*3*8][16];
static elem_t layer14_layer3_0_conv1_out[1024];
static elem_t layer3_0_conv2_w_flat[3*3*16][16];
static elem_t layer15_layer3_0_conv2_out[1024];
static elem_t layer3_0_shortcut_0_w_flat[1*1*8][16];
static elem_t layer16_layer3_0_shortcut_0_out[1024];
static elem_t layer17_add_4_out[1024];
static elem_t layer3_1_conv1_w_flat[3*3*16][16];
static elem_t layer18_layer3_1_conv1_out[1024];
static elem_t layer3_1_conv2_w_flat[3*3*16][16];
static elem_t layer19_layer3_1_conv2_out[1024];
static elem_t layer20_add_5_out[1024];
static elem_t layer4_0_conv1_w_flat[3*3*16][32];
static elem_t layer21_layer4_0_conv1_out[512];
static elem_t layer4_0_conv2_w_flat[3*3*32][32];
static elem_t layer22_layer4_0_conv2_out[512];
static elem_t layer4_0_shortcut_0_w_flat[1*1*16][32];
static elem_t layer23_layer4_0_shortcut_0_out[512];
static elem_t layer24_add_6_out[512];
static elem_t layer4_1_conv1_w_flat[3*3*32][32];
static elem_t layer25_layer4_1_conv1_out[512];
static elem_t layer4_1_conv2_w_flat[3*3*32][32];
static elem_t layer26_layer4_1_conv2_out[512];
static elem_t layer27_add_7_out[512];
static elem_t layer28_global_pool_out[32];
static elem_t layer29_classifier_out[10];

int main() {
    #ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
    #endif
    flatten_weights(4, 3, 3, (elem_t*)conv1_w, (elem_t*)conv1_w_flat);
    flatten_weights(4, 3, 4, (elem_t*)layer1_0_conv1_w, (elem_t*)layer1_0_conv1_w_flat);
    flatten_weights(4, 3, 4, (elem_t*)layer1_0_conv2_w, (elem_t*)layer1_0_conv2_w_flat);
    flatten_weights(4, 3, 4, (elem_t*)layer1_1_conv1_w, (elem_t*)layer1_1_conv1_w_flat);
    flatten_weights(4, 3, 4, (elem_t*)layer1_1_conv2_w, (elem_t*)layer1_1_conv2_w_flat);
    flatten_weights(8, 3, 4, (elem_t*)layer2_0_conv1_w, (elem_t*)layer2_0_conv1_w_flat);
    flatten_weights(8, 3, 8, (elem_t*)layer2_0_conv2_w, (elem_t*)layer2_0_conv2_w_flat);
    flatten_weights(8, 1, 4, (elem_t*)layer2_0_shortcut_0_w, (elem_t*)layer2_0_shortcut_0_w_flat);
    flatten_weights(8, 3, 8, (elem_t*)layer2_1_conv1_w, (elem_t*)layer2_1_conv1_w_flat);
    flatten_weights(8, 3, 8, (elem_t*)layer2_1_conv2_w, (elem_t*)layer2_1_conv2_w_flat);
    flatten_weights(16, 3, 8, (elem_t*)layer3_0_conv1_w, (elem_t*)layer3_0_conv1_w_flat);
    flatten_weights(16, 3, 16, (elem_t*)layer3_0_conv2_w, (elem_t*)layer3_0_conv2_w_flat);
    flatten_weights(16, 1, 8, (elem_t*)layer3_0_shortcut_0_w, (elem_t*)layer3_0_shortcut_0_w_flat);
    flatten_weights(16, 3, 16, (elem_t*)layer3_1_conv1_w, (elem_t*)layer3_1_conv1_w_flat);
    flatten_weights(16, 3, 16, (elem_t*)layer3_1_conv2_w, (elem_t*)layer3_1_conv2_w_flat);
    flatten_weights(32, 3, 16, (elem_t*)layer4_0_conv1_w, (elem_t*)layer4_0_conv1_w_flat);
    flatten_weights(32, 3, 32, (elem_t*)layer4_0_conv2_w, (elem_t*)layer4_0_conv2_w_flat);
    flatten_weights(32, 1, 16, (elem_t*)layer4_0_shortcut_0_w, (elem_t*)layer4_0_shortcut_0_w_flat);
    flatten_weights(32, 3, 32, (elem_t*)layer4_1_conv1_w, (elem_t*)layer4_1_conv1_w_flat);
    flatten_weights(32, 3, 32, (elem_t*)layer4_1_conv2_w, (elem_t*)layer4_1_conv2_w_flat);
    gemmini_flush(0);
    printf("Initializing Network...\n");
    int correct_predictions = 0;
    int errors = 0;
    for (int i = 0; i < NUM_INFERENCES; i++) {
        int ground_truth = all_ground_truths[i];
        elem_t* current_image_ptr = (elem_t*)all_input_images[i];
        tiled_conv_auto(1, 32, 32, 3, 4, 32, 32, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)current_image_ptr, (elem_t*)conv1_w_flat, (acc_t*)conv1_b, (elem_t*)layer0_conv1_out, RELU, (acc_scale_t)CONV1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CONV1_AVAILABLE)
        if (i == 0) {
            verify_array("conv1", 4096, (elem_t*)layer0_conv1_out, (elem_t*)debug_conv1);
        }
#endif
        tiled_conv_auto(1, 32, 32, 4, 4, 32, 32, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer0_conv1_out, (elem_t*)layer1_0_conv1_w_flat, (acc_t*)layer1_0_conv1_b, (elem_t*)layer1_layer1_0_conv1_out, RELU, (acc_scale_t)LAYER1_0_CONV1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER1_0_CONV1_AVAILABLE)
        if (i == 0) {
            verify_array("layer1_0_conv1", 4096, (elem_t*)layer1_layer1_0_conv1_out, (elem_t*)debug_layer1_0_conv1);
        }
#endif
        tiled_conv_auto(1, 32, 32, 4, 4, 32, 32, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer1_layer1_0_conv1_out, (elem_t*)layer1_0_conv2_w_flat, (acc_t*)layer1_0_conv2_b, (elem_t*)layer2_layer1_0_conv2_out, NO_ACTIVATION, (acc_scale_t)LAYER1_0_CONV2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER1_0_CONV2_AVAILABLE)
        if (i == 0) {
            verify_array("layer1_0_conv2", 4096, (elem_t*)layer2_layer1_0_conv2_out, (elem_t*)debug_layer1_0_conv2);
        }
#endif
        tiled_resadd_auto(4096, 1, (acc_scale_t)ADD_SCALE_A, (acc_scale_t)ADD_SCALE_B, ACC_SCALE_IDENTITY, (elem_t*)layer2_layer1_0_conv2_out, (elem_t*)layer0_conv1_out, (elem_t*)layer3_add_out, true, WS);
#if defined(DEBUG) && defined(DEBUG_ADD_AVAILABLE)
        if (i == 0) {
            verify_array("add", 4096, (elem_t*)layer3_add_out, (elem_t*)debug_add);
        }
#endif
        tiled_conv_auto(1, 32, 32, 4, 4, 32, 32, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer3_add_out, (elem_t*)layer1_1_conv1_w_flat, (acc_t*)layer1_1_conv1_b, (elem_t*)layer4_layer1_1_conv1_out, RELU, (acc_scale_t)LAYER1_1_CONV1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER1_1_CONV1_AVAILABLE)
        if (i == 0) {
            verify_array("layer1_1_conv1", 4096, (elem_t*)layer4_layer1_1_conv1_out, (elem_t*)debug_layer1_1_conv1);
        }
#endif
        tiled_conv_auto(1, 32, 32, 4, 4, 32, 32, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer4_layer1_1_conv1_out, (elem_t*)layer1_1_conv2_w_flat, (acc_t*)layer1_1_conv2_b, (elem_t*)layer5_layer1_1_conv2_out, NO_ACTIVATION, (acc_scale_t)LAYER1_1_CONV2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER1_1_CONV2_AVAILABLE)
        if (i == 0) {
            verify_array("layer1_1_conv2", 4096, (elem_t*)layer5_layer1_1_conv2_out, (elem_t*)debug_layer1_1_conv2);
        }
#endif
        tiled_resadd_auto(4096, 1, (acc_scale_t)ADD_1_SCALE_A, (acc_scale_t)ADD_1_SCALE_B, ACC_SCALE_IDENTITY, (elem_t*)layer5_layer1_1_conv2_out, (elem_t*)layer3_add_out, (elem_t*)layer6_add_1_out, true, WS);
#if defined(DEBUG) && defined(DEBUG_ADD_1_AVAILABLE)
        if (i == 0) {
            verify_array("add_1", 4096, (elem_t*)layer6_add_1_out, (elem_t*)debug_add_1);
        }
#endif
        tiled_conv_auto(1, 32, 32, 4, 8, 16, 16, 2, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer6_add_1_out, (elem_t*)layer2_0_conv1_w_flat, (acc_t*)layer2_0_conv1_b, (elem_t*)layer7_layer2_0_conv1_out, RELU, (acc_scale_t)LAYER2_0_CONV1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER2_0_CONV1_AVAILABLE)
        if (i == 0) {
            verify_array("layer2_0_conv1", 2048, (elem_t*)layer7_layer2_0_conv1_out, (elem_t*)debug_layer2_0_conv1);
        }
#endif
        tiled_conv_auto(1, 16, 16, 8, 8, 16, 16, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer7_layer2_0_conv1_out, (elem_t*)layer2_0_conv2_w_flat, (acc_t*)layer2_0_conv2_b, (elem_t*)layer8_layer2_0_conv2_out, NO_ACTIVATION, (acc_scale_t)LAYER2_0_CONV2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER2_0_CONV2_AVAILABLE)
        if (i == 0) {
            verify_array("layer2_0_conv2", 2048, (elem_t*)layer8_layer2_0_conv2_out, (elem_t*)debug_layer2_0_conv2);
        }
#endif
        tiled_conv_auto(1, 32, 32, 4, 8, 16, 16, 2, 1, 1, 0, 1, false, false, false, false, false, (elem_t*)layer6_add_1_out, (elem_t*)layer2_0_shortcut_0_w_flat, (acc_t*)layer2_0_shortcut_0_b, (elem_t*)layer9_layer2_0_shortcut_0_out, NO_ACTIVATION, (acc_scale_t)LAYER2_0_SHORTCUT_0_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER2_0_SHORTCUT_0_AVAILABLE)
        if (i == 0) {
            verify_array("layer2_0_shortcut_0", 2048, (elem_t*)layer9_layer2_0_shortcut_0_out, (elem_t*)debug_layer2_0_shortcut_0);
        }
#endif
        tiled_resadd_auto(2048, 1, (acc_scale_t)ADD_2_SCALE_A, (acc_scale_t)ADD_2_SCALE_B, ACC_SCALE_IDENTITY, (elem_t*)layer8_layer2_0_conv2_out, (elem_t*)layer9_layer2_0_shortcut_0_out, (elem_t*)layer10_add_2_out, true, WS);
#if defined(DEBUG) && defined(DEBUG_ADD_2_AVAILABLE)
        if (i == 0) {
            verify_array("add_2", 2048, (elem_t*)layer10_add_2_out, (elem_t*)debug_add_2);
        }
#endif
        tiled_conv_auto(1, 16, 16, 8, 8, 16, 16, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer10_add_2_out, (elem_t*)layer2_1_conv1_w_flat, (acc_t*)layer2_1_conv1_b, (elem_t*)layer11_layer2_1_conv1_out, RELU, (acc_scale_t)LAYER2_1_CONV1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER2_1_CONV1_AVAILABLE)
        if (i == 0) {
            verify_array("layer2_1_conv1", 2048, (elem_t*)layer11_layer2_1_conv1_out, (elem_t*)debug_layer2_1_conv1);
        }
#endif
        tiled_conv_auto(1, 16, 16, 8, 8, 16, 16, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer11_layer2_1_conv1_out, (elem_t*)layer2_1_conv2_w_flat, (acc_t*)layer2_1_conv2_b, (elem_t*)layer12_layer2_1_conv2_out, NO_ACTIVATION, (acc_scale_t)LAYER2_1_CONV2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER2_1_CONV2_AVAILABLE)
        if (i == 0) {
            verify_array("layer2_1_conv2", 2048, (elem_t*)layer12_layer2_1_conv2_out, (elem_t*)debug_layer2_1_conv2);
        }
#endif
        tiled_resadd_auto(2048, 1, (acc_scale_t)ADD_3_SCALE_A, (acc_scale_t)ADD_3_SCALE_B, ACC_SCALE_IDENTITY, (elem_t*)layer12_layer2_1_conv2_out, (elem_t*)layer10_add_2_out, (elem_t*)layer13_add_3_out, true, WS);
#if defined(DEBUG) && defined(DEBUG_ADD_3_AVAILABLE)
        if (i == 0) {
            verify_array("add_3", 2048, (elem_t*)layer13_add_3_out, (elem_t*)debug_add_3);
        }
#endif
        tiled_conv_auto(1, 16, 16, 8, 16, 8, 8, 2, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer13_add_3_out, (elem_t*)layer3_0_conv1_w_flat, (acc_t*)layer3_0_conv1_b, (elem_t*)layer14_layer3_0_conv1_out, RELU, (acc_scale_t)LAYER3_0_CONV1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER3_0_CONV1_AVAILABLE)
        if (i == 0) {
            verify_array("layer3_0_conv1", 1024, (elem_t*)layer14_layer3_0_conv1_out, (elem_t*)debug_layer3_0_conv1);
        }
#endif
        tiled_conv_auto(1, 8, 8, 16, 16, 8, 8, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer14_layer3_0_conv1_out, (elem_t*)layer3_0_conv2_w_flat, (acc_t*)layer3_0_conv2_b, (elem_t*)layer15_layer3_0_conv2_out, NO_ACTIVATION, (acc_scale_t)LAYER3_0_CONV2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER3_0_CONV2_AVAILABLE)
        if (i == 0) {
            verify_array("layer3_0_conv2", 1024, (elem_t*)layer15_layer3_0_conv2_out, (elem_t*)debug_layer3_0_conv2);
        }
#endif
        tiled_conv_auto(1, 16, 16, 8, 16, 8, 8, 2, 1, 1, 0, 1, false, false, false, false, false, (elem_t*)layer13_add_3_out, (elem_t*)layer3_0_shortcut_0_w_flat, (acc_t*)layer3_0_shortcut_0_b, (elem_t*)layer16_layer3_0_shortcut_0_out, NO_ACTIVATION, (acc_scale_t)LAYER3_0_SHORTCUT_0_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER3_0_SHORTCUT_0_AVAILABLE)
        if (i == 0) {
            verify_array("layer3_0_shortcut_0", 1024, (elem_t*)layer16_layer3_0_shortcut_0_out, (elem_t*)debug_layer3_0_shortcut_0);
        }
#endif
        tiled_resadd_auto(1024, 1, (acc_scale_t)ADD_4_SCALE_A, (acc_scale_t)ADD_4_SCALE_B, ACC_SCALE_IDENTITY, (elem_t*)layer15_layer3_0_conv2_out, (elem_t*)layer16_layer3_0_shortcut_0_out, (elem_t*)layer17_add_4_out, true, WS);
#if defined(DEBUG) && defined(DEBUG_ADD_4_AVAILABLE)
        if (i == 0) {
            verify_array("add_4", 1024, (elem_t*)layer17_add_4_out, (elem_t*)debug_add_4);
        }
#endif
        tiled_conv_auto(1, 8, 8, 16, 16, 8, 8, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer17_add_4_out, (elem_t*)layer3_1_conv1_w_flat, (acc_t*)layer3_1_conv1_b, (elem_t*)layer18_layer3_1_conv1_out, RELU, (acc_scale_t)LAYER3_1_CONV1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER3_1_CONV1_AVAILABLE)
        if (i == 0) {
            verify_array("layer3_1_conv1", 1024, (elem_t*)layer18_layer3_1_conv1_out, (elem_t*)debug_layer3_1_conv1);
        }
#endif
        tiled_conv_auto(1, 8, 8, 16, 16, 8, 8, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer18_layer3_1_conv1_out, (elem_t*)layer3_1_conv2_w_flat, (acc_t*)layer3_1_conv2_b, (elem_t*)layer19_layer3_1_conv2_out, NO_ACTIVATION, (acc_scale_t)LAYER3_1_CONV2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER3_1_CONV2_AVAILABLE)
        if (i == 0) {
            verify_array("layer3_1_conv2", 1024, (elem_t*)layer19_layer3_1_conv2_out, (elem_t*)debug_layer3_1_conv2);
        }
#endif
        tiled_resadd_auto(1024, 1, (acc_scale_t)ADD_5_SCALE_A, (acc_scale_t)ADD_5_SCALE_B, ACC_SCALE_IDENTITY, (elem_t*)layer19_layer3_1_conv2_out, (elem_t*)layer17_add_4_out, (elem_t*)layer20_add_5_out, true, WS);
#if defined(DEBUG) && defined(DEBUG_ADD_5_AVAILABLE)
        if (i == 0) {
            verify_array("add_5", 1024, (elem_t*)layer20_add_5_out, (elem_t*)debug_add_5);
        }
#endif
        tiled_conv_auto(1, 8, 8, 16, 32, 4, 4, 2, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer20_add_5_out, (elem_t*)layer4_0_conv1_w_flat, (acc_t*)layer4_0_conv1_b, (elem_t*)layer21_layer4_0_conv1_out, RELU, (acc_scale_t)LAYER4_0_CONV1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER4_0_CONV1_AVAILABLE)
        if (i == 0) {
            verify_array("layer4_0_conv1", 512, (elem_t*)layer21_layer4_0_conv1_out, (elem_t*)debug_layer4_0_conv1);
        }
#endif
        tiled_conv_auto(1, 4, 4, 32, 32, 4, 4, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer21_layer4_0_conv1_out, (elem_t*)layer4_0_conv2_w_flat, (acc_t*)layer4_0_conv2_b, (elem_t*)layer22_layer4_0_conv2_out, NO_ACTIVATION, (acc_scale_t)LAYER4_0_CONV2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER4_0_CONV2_AVAILABLE)
        if (i == 0) {
            verify_array("layer4_0_conv2", 512, (elem_t*)layer22_layer4_0_conv2_out, (elem_t*)debug_layer4_0_conv2);
        }
#endif
        tiled_conv_auto(1, 8, 8, 16, 32, 4, 4, 2, 1, 1, 0, 1, false, false, false, false, false, (elem_t*)layer20_add_5_out, (elem_t*)layer4_0_shortcut_0_w_flat, (acc_t*)layer4_0_shortcut_0_b, (elem_t*)layer23_layer4_0_shortcut_0_out, NO_ACTIVATION, (acc_scale_t)LAYER4_0_SHORTCUT_0_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER4_0_SHORTCUT_0_AVAILABLE)
        if (i == 0) {
            verify_array("layer4_0_shortcut_0", 512, (elem_t*)layer23_layer4_0_shortcut_0_out, (elem_t*)debug_layer4_0_shortcut_0);
        }
#endif
        tiled_resadd_auto(512, 1, (acc_scale_t)ADD_6_SCALE_A, (acc_scale_t)ADD_6_SCALE_B, ACC_SCALE_IDENTITY, (elem_t*)layer22_layer4_0_conv2_out, (elem_t*)layer23_layer4_0_shortcut_0_out, (elem_t*)layer24_add_6_out, true, WS);
#if defined(DEBUG) && defined(DEBUG_ADD_6_AVAILABLE)
        if (i == 0) {
            verify_array("add_6", 512, (elem_t*)layer24_add_6_out, (elem_t*)debug_add_6);
        }
#endif
        tiled_conv_auto(1, 4, 4, 32, 32, 4, 4, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer24_add_6_out, (elem_t*)layer4_1_conv1_w_flat, (acc_t*)layer4_1_conv1_b, (elem_t*)layer25_layer4_1_conv1_out, RELU, (acc_scale_t)LAYER4_1_CONV1_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER4_1_CONV1_AVAILABLE)
        if (i == 0) {
            verify_array("layer4_1_conv1", 512, (elem_t*)layer25_layer4_1_conv1_out, (elem_t*)debug_layer4_1_conv1);
        }
#endif
        tiled_conv_auto(1, 4, 4, 32, 32, 4, 4, 1, 1, 1, 1, 3, false, false, false, false, false, (elem_t*)layer25_layer4_1_conv1_out, (elem_t*)layer4_1_conv2_w_flat, (acc_t*)layer4_1_conv2_b, (elem_t*)layer26_layer4_1_conv2_out, NO_ACTIVATION, (acc_scale_t)LAYER4_1_CONV2_SCALE, 0, 0, 0, WS);
#if defined(DEBUG) && defined(DEBUG_LAYER4_1_CONV2_AVAILABLE)
        if (i == 0) {
            verify_array("layer4_1_conv2", 512, (elem_t*)layer26_layer4_1_conv2_out, (elem_t*)debug_layer4_1_conv2);
        }
#endif
        tiled_resadd_auto(512, 1, (acc_scale_t)ADD_7_SCALE_A, (acc_scale_t)ADD_7_SCALE_B, ACC_SCALE_IDENTITY, (elem_t*)layer26_layer4_1_conv2_out, (elem_t*)layer24_add_6_out, (elem_t*)layer27_add_7_out, true, WS);
#if defined(DEBUG) && defined(DEBUG_ADD_7_AVAILABLE)
        if (i == 0) {
            verify_array("add_7", 512, (elem_t*)layer27_add_7_out, (elem_t*)debug_add_7);
        }
#endif
        global_average_pool(1, 4, 4, 32, (elem_t*)layer27_add_7_out, (elem_t*)layer28_global_pool_out);
#if defined(DEBUG) && defined(DEBUG_GLOBAL_POOL_AVAILABLE)
        if (i == 0) {
            verify_array("global_pool", 32, (elem_t*)layer28_global_pool_out, (elem_t*)debug_global_pool);
        }
#endif
        tiled_matmul_auto(1, 10, 32, (elem_t*)layer28_global_pool_out, (elem_t*)classifier_w, (acc_t*)classifier_b, (elem_t*)layer29_classifier_out, 32, 10, 10, 10, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, NO_ACTIVATION, (acc_scale_t)CLASSIFIER_SCALE, ACC_SCALE_IDENTITY, true, false, false, false, false, 0, WS);
#if defined(DEBUG) && defined(DEBUG_CLASSIFIER_AVAILABLE)
        if (i == 0) {
            verify_array("classifier", 10, (elem_t*)layer29_classifier_out, (elem_t*)debug_classifier);
        }
#endif
        int prediction = argmax(NUM_CLASSES, layer29_classifier_out);
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