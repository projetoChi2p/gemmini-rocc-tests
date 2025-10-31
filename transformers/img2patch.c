#include <stdint.h>     // For uint8_t, uint32_t
#include <stddef.h>     // For size_t
//#include <stdlib.h>     // For rand(), exit()
#include <stdio.h>      // For printf()
#include <string.h>     // For memcpy()
#include <assert.h>     // For assert()
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

// --- 1. Include Base Gemmini Types ---
#define BAREMETAL 1 // Required by gemmini_testutils.h

// --- 2. Include Global ViT Dimensions ---
#include "include/gemmini_vit_params.h"
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#define FULL_BIAS_WIDTH 1
#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif



// --- 3. Include Kernel Prototypes ---
#include "include/gemmini_vit_kernels.h"

// --- All dimension defines (NUM_PATCHES_H, PATCH_DIM, etc.)
// --- are now correctly inherited from gemmini_vit_params.h

// --- Static Array Declarations ---
// These are only needed if you uncomment the main function for local testing.
// static elem_t image[IMG_H][IMG_W][IMG_C];
// static elem_t patches[NUM_PATCHES][PATCH_DIM];

/**
 * @brief (Helper) Prints the output patch matrix for verification.
 */
void print_patches(elem_t p[NUM_PATCHES][PATCH_DIM]) {
    // Only print a small portion if the matrix is too large
    uint32_t rows_to_print = NUM_PATCHES > 16 ? 16 : NUM_PATCHES;
    uint32_t cols_to_print = PATCH_DIM > 32 ? 32 : PATCH_DIM;

    printf("--- Patches Matrix (Shape: %u, %u) ---\n", NUM_PATCHES, PATCH_DIM);
    printf("--- (Printing top-left %u x %u) ---\n", rows_to_print, cols_to_print);

    for (size_t n = 0; n < rows_to_print; ++n) {
        printf("Patch %3zu: [ ", n);
        for (size_t d = 0; d < cols_to_print; ++d) {
            // Print as integer
            printf("%3d ", (int)p[n][d]);
        }
        if (PATCH_DIM > cols_to_print) printf("...");
        printf("]\n");
    }
    printf("------------------------------------------\n");
}

/**
 * @brief (Helper) Initializes the image with sequential data for testing.
 */
void initialize_image(elem_t img[IMG_H][IMG_W][IMG_C]) {
    printf("Initializing image with test data...\n");
    elem_t counter = 0;
    for (size_t h = 0; h < IMG_H; ++h) {
        for (size_t w = 0; w < IMG_W; ++w) {
            for (size_t c = 0; c < IMG_C; ++c) {
                // Use a simple repeating pattern (0-255)
                img[h][w][c] = counter++;
            }
        }
    }
}

/**
 * @brief Converts an image tensor into a sequence of flattened patches.
 *
 * This function implements the (reshape -> transpose -> reshape) logic
 * by directly copying memory blocks from the source image to the
 * destination patch tensor.
 *
 * @param image_in Source image (H, W, C)
 * @param patches_out Destination patch tensor (N, P*P*C)
 */
void run_patch_extraction(
    const elem_t image_in[IMG_H][IMG_W][IMG_C],
    elem_t patches_out[NUM_PATCHES][PATCH_DIM]
) {
    printf("  [Step 0] Converting image to patches (CPU)...\n");

    // Check divisibility at runtime
    if (IMG_H % PATCH_SIZE != 0 || IMG_W % PATCH_SIZE != 0) {
        printf("ERROR: Image dimensions (%d, %d) not divisible by patch size (%d)\n", 
            IMG_H, IMG_W, PATCH_SIZE);
        return;
    }
    
    // Calculate the size (in bytes) of one contiguous row within a patch.
    // This is (PATCH_SIZE * IMG_C) elements.
    const size_t patch_row_bytes = PATCH_SIZE * IMG_C * sizeof(elem_t);

    for (uint32_t ph = 0; ph < NUM_PATCHES_H; ++ph) { // Iterate over patch rows
        for (uint32_t pw = 0; pw < NUM_PATCHES_W; ++pw) { // Iterate over patch cols

            // Calculate the current patch index 'n' (from 0 to N-1)
            uint32_t n = ph * NUM_PATCHES_W + pw;

            for (uint32_t r = 0; r < PATCH_SIZE; ++r) { // Iterate over rows *within* a patch

                // 1. Calculate Source Pointer
                uint32_t h = ph * PATCH_SIZE + r;
                uint32_t w_start = pw * PATCH_SIZE;
                const elem_t* src_ptr = &image_in[h][w_start][0];

                // 2. Calculate Destination Pointer
                uint32_t d_start = r * PATCH_SIZE * IMG_C;
                elem_t* dest_ptr = &patches_out[n][d_start];
                
                // 3. Copy the contiguous block of memory
                memcpy(dest_ptr, src_ptr, patch_row_bytes);
            }
        }
    }
    printf("  [Step 0] Conversion complete.\n");
}

/*
// --- Main function for local testing ---
int main() {
    // To use this, you must declare the static arrays at the top
    static elem_t image[IMG_H][IMG_W][IMG_C];
    static elem_t patches[NUM_PATCHES][PATCH_DIM];

    uint64_t start = read_cycles();

    // 1. Initialize the input image with test data
    initialize_image(image);
    uint64_t init_IMG_Cycles = read_cycles();

    // 2. Perform the conversion
    run_patch_extraction(image, patches);
    uint64_t conversion_cycles = read_cycles();

    printf("Initialization cycles: %lu\n", init_IMG_Cycles - start);
    printf("Conversion cycles: %lu\n", conversion_cycles - init_IMG_Cycles);

    // 3. Print the results for verification
    print_patches(patches);
    
#if IMG_H == 4 && IMG_W == 4 && IMG_C == 1 && PATCH_SIZE == 2
    printf("\nVerifying 4x4 test case...\n");
    assert(patches[0][0] == 0 && patches[0][1] == 1 && patches[0][2] == 4 && patches[0][3] == 5);
    assert(patches[1][0] == 2 && patches[1][1] == 3 && patches[1][2] == 6 && patches[1][3] == 7);
    assert(patches[2][0] == 8 && patches[2][1] == 9 && patches[2][2] == 12 && patches[2][3] == 13);
    assert(patches[3][0] == 10 && patches[3][1] == 11 && patches[3][2] == 14 && patches[3][3] == 15);
    printf("Verification passed!\n");
#endif

    return 0;
}
*/