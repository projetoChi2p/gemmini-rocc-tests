#ifndef GEMMINI_VIT_PARAMS_MNIST_H
#define GEMMINI_VIT_PARAMS_MNIST_H

// --- 1. Base Image & Patch Dimensions (Adapted for 28x28 MNIST) ---
#define IMG_H 28
#define IMG_W 28
#define IMG_C 1           // MNIST is grayscale
#define PATCH_SIZE 4      // A 4x4 patch size is a good fit for 28x28 (7x7=49 patches)

// --- 2. Derived Patch Dimensions ---
#define NUM_PATCHES_H (IMG_H / PATCH_SIZE) // 7
#define NUM_PATCHES_W (IMG_W / PATCH_SIZE) // 7
#define NUM_PATCHES (NUM_PATCHES_H * NUM_PATCHES_W) // 49
#define PATCH_DIM (PATCH_SIZE * PATCH_SIZE * IMG_C) // 16

// --- 3. Transformer Model Dimensions (Scaled up to be useful) ---
#define HIDDEN_DIM 128     // A more reasonable embedding dimension
#define NUM_CLASSES 10    // MNIST has 10 classes
#define SEQ_LEN (NUM_PATCHES + 1) // 49 patches + 1 [CLS] token = 50
#define NUM_HEADS 4       // 4 heads (64 hidden / 4 heads = 16 dim/head)
#define EXPANSION_DIM (HIDDEN_DIM * 4) // Standard FFN expansion (256)
#define NUM_LAYERS 4      // A small but non-trivial number of layers

// --- 4. Example GEMM Dims (for the initial patch embedding) ---
// This operation projects N patches from PATCH_DIM to HIDDEN_DIM
// GEMM: (N, PATCH_DIM) x (PATCH_DIM, HIDDEN_DIM) -> (N, HIDDEN_DIM)
#define MAT_DIM_I NUM_PATCHES // 49
#define MAT_DIM_K PATCH_DIM   // 16
#define MAT_DIM_J HIDDEN_DIM  // 64

#endif // GEMMINI_VIT_PARAMS_MNIST_H