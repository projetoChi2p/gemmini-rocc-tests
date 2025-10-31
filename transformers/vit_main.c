// vit_main.c
// Main file to run the full ViT pipeline

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define BAREMETAL 1
#define NO_BIAS 1

// --- 1. Include base headers ---
#include "include/gemmini_testutils.h"
#include "include/gemmini.h"
#include "include/gemmini_nn.h" // For tiled_matmul_auto in transformer.c

#include "include/gemmini_vit_params.h"

// --- 3. Define ACC_T type ---
#define FULL_BIAS_WIDTH 1
#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

// --- 4. Define dimensions for helper functions ---
// These macros must be defined *before* including gemmini_vit.h
// We set them to the *largest* matrix dimensions needed (for the encoder)
// so the helper functions (like full_matmul) are compiled correctly.
#undef MAT_DIM_I
#undef MAT_DIM_K
#undef MAT_DIM_J
#define MAT_DIM_I SEQ_LEN    // 65
#define MAT_DIM_K HIDDEN_DIM // 64
#define MAT_DIM_J HIDDEN_DIM // 64 
#include "include/gemmini_vit.h"

// --- 6. Include our new kernel declarations ---
// #include "include/gemmini_vit_kernels.h"

#include "vit_weights.h"
#include "image_data.h"

// --- Static Arrays for all weights and activations ---

// Step 0: Patch Extraction
static elem_t image[IMG_H][IMG_W][IMG_C];
static elem_t patches[NUM_PATCHES][PATCH_DIM]; // 64x64

// Step 1: Patch Embedding
// static elem_t patch_embed_weights[PATCH_DIM][HIDDEN_DIM];
static acc_t patch_embed_bias[NUM_PATCHES][HIDDEN_DIM];
static elem_t patch_embed_out[NUM_PATCHES][HIDDEN_DIM]; // 64x64

// Step 2: Positional Embedding
// static elem_t cls_token_weights[1][HIDDEN_DIM];
// static elem_t pos_embed_weights[SEQ_LEN][HIDDEN_DIM]; // 65x64
static elem_t encoder_input[SEQ_LEN][HIDDEN_DIM];     // 65x64

// Step 3: Encoder
static elem_t encoder_output[SEQ_LEN][HIDDEN_DIM];    // 65x64
// Encoder Layer Weights (re-used for simplicity)
// static elem_t Wqkvo[4][HIDDEN_DIM][HIDDEN_DIM];
static acc_t Wqkvo_b[4][HIDDEN_DIM];
// static elem_t ff_w[2][HIDDEN_DIM][EXPANSION_DIM]; // Note: FFN dimensions
static acc_t ff1_b[EXPANSION_DIM];
static acc_t ff2_b[HIDDEN_DIM];
// static acc_t ff1_w[HIDDEN_DIM][EXPANSION_DIM];
// static acc_t ff2_w[EXPANSION_DIM][HIDDEN_DIM];

// Encoder Buffers
static elem_t QKV_buf[3][SEQ_LEN][HIDDEN_DIM];
static elem_t attn_buf[NUM_HEADS][SEQ_LEN][SEQ_LEN];
static elem_t out_buf[SEQ_LEN][EXPANSION_DIM];
static acc_t out_buf_acc[SEQ_LEN][HIDDEN_DIM];
static elem_t resadd1_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t resadd2_buf[SEQ_LEN][HIDDEN_DIM];

// Step 4: Classification
static elem_t class_token_final[1][HIDDEN_DIM];       // 1x64
// static elem_t classifier_weights[HIDDEN_DIM][NUM_CLASSES];
static acc_t classifier_bias[1][NUM_CLASSES];
static elem_t final_logits[1][NUM_CLASSES];         // 1x64

// --- Helper to initialize all static data ---
void init_weights() {
    
    // Copy the constant data into the non-const static array
    memcpy(image, test_image_data, sizeof(test_image_data));

    // All other weight loops are REMOVED.
    
    // Biases are skipped (NO_BIAS=1)
    printf("Initialization complete. Weights loaded from vit_weights.h\n");
}

// --- ViT-specific Helper Function ---
void extract_cls_token(
    const elem_t encoder_output[SEQ_LEN][HIDDEN_DIM],
    elem_t class_token_out[1][HIDDEN_DIM])
{
    printf("  [Step 3b] Extracting [CLS] token...\n");
    // The [CLS] token is the first row (index 0) of the encoder output
    memcpy(class_token_out, encoder_output, HIDDEN_DIM * sizeof(elem_t));
}


/**
 * @brief Prints the contents of a matrix with a title and shape.
 * * @param title A string to print as the header.
 * @param m     A pointer to the matrix data (must be flat).
 * @param rows  The number of rows in the matrix.
 * @param cols  The number of columns (stride) of the matrix.
 */
void print_matrix(const char* title, elem_t* m, size_t rows, size_t cols) {
    printf("--- %s (Shape: %zu, %zu) ---\n", title, rows, cols);
    
    // Limit how much we print to avoid flooding the console
    size_t rows_to_print = rows > 8 ? 8 : rows; // Print max 8 rows
    size_t cols_to_print = cols > 16 ? 16 : cols; // Print max 16 cols

    for (size_t r = 0; r < rows_to_print; r++) {
        printf("  Row %3zu: [ ", r);
        for (size_t c = 0; c < cols_to_print; c++) {
            // %4d formats the number to take up 4 spaces for alignment
            printf("%4d ", (int)m[r * cols + c]);
        }
        if (cols > cols_to_print) printf("...");
        printf("]\n");
    }
    if (rows > rows_to_print) printf("  ...\n");
    printf("----------------------------------------\n");
}



static inline void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

static inline void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], acc_scale_t scale) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      full_t scaled = ACC_SCALE(full[r][c], scale);
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled;
#endif
    }
} 

static inline int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}




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


/**
 * @brief Runs the patch embedding GEMM.
 * (This is function 1/2 from this file)
 */
int run_patch_embedding(
    const elem_t patches_in[NUM_PATCHES][PATCH_DIM], 
    const elem_t weights_in[PATCH_DIM][HIDDEN_DIM], 
    const ACC_T bias_in[NUM_PATCHES][HIDDEN_DIM], 
    elem_t embed_out[NUM_PATCHES][HIDDEN_DIM],
    uint64_t* gemmini_cycles
) {
    printf("  [Step 1] Running Patch Embedding (Gemmini WS)...\n");
    printf("    (A) Patches: (%d, %d)\n", MAT_DIM_I, MAT_DIM_K);
    printf("    (B) Weights: (%d, %d)\n", MAT_DIM_K, MAT_DIM_J);
    printf("    (C) Output:  (%d, %d)\n", MAT_DIM_I, MAT_DIM_J);
    
    unsigned long start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)patches_in, (elem_t*)weights_in, 
            NO_BIAS ? NULL : (void*)bias_in, (void*)embed_out,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);

    unsigned long end = read_cycles();
    *gemmini_cycles = end - start;
    printf("  [Step 1] Cycles: %llu\n", *gemmini_cycles);

#if CHECK_RESULT == 1
    // Run CPU verification
    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

    printf("    Verifying Step 1 (CPU vs Gemmini)...\n");
    full_matmul((elem_t (*)[MAT_DIM_K])patches_in, (elem_t (*)[MAT_DIM_J])weights_in, 
                (ACC_T (*)[MAT_DIM_J])bias_in, gold_full);
    full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);

    if (!full_is_equal(embed_out, gold)) {
        printf("    !!! FAILURE: Patch Embedding Mismatch !!!\n");
        return;
    } else {
        printf("    ... Success.\n");
    }
#endif
}


/**
 * @brief Adds the [CLS] token and positional embeddings. (CPU)
 * (This is function 2/2 from this file)
 */
void add_cls_and_pos_embed(
    const elem_t patch_embeds[NUM_PATCHES][HIDDEN_DIM],
    const elem_t cls_token_weights[1][HIDDEN_DIM],
    const elem_t pos_embed_weights[SEQ_LEN][HIDDEN_DIM],
    elem_t encoder_input_out[SEQ_LEN][HIDDEN_DIM])
{
    printf("  [Step 2] Adding [CLS] token & Pos-Embed (CPU)...\n");
    
    // 1. Copy [CLS] token to row 0
    memcpy(encoder_input_out[0], cls_token_weights[0], HIDDEN_DIM * sizeof(elem_t));

    // 2. Copy patch embeddings to rows 1 through N
    memcpy(encoder_input_out[1], patch_embeds, NUM_PATCHES * HIDDEN_DIM * sizeof(elem_t));

    // 3. Add positional embeddings (element-wise)
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            full_t sum = (full_t)encoder_input_out[i][j] + (full_t)pos_embed_weights[i][j];
            if (sum > elem_t_max) sum = elem_t_max;
            if (sum < elem_t_min) sum = elem_t_min; // Handle signed types
            encoder_input_out[i][j] = (elem_t)sum;
        }
    }
}

// Note: For self-attention, "enc_out" should be the same as "input".
// Note: "compression_factor" should be 1 for most use cases.
void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,

        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,

        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        const acc_t * Wo_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc)
{
    int hidden_dim_compressed = hidden_dim / compression_factor;
    int hidden_dim_per_head = hidden_dim_compressed / num_heads;

    if (compression_factor < 0) {
        hidden_dim_compressed = hidden_dim;
        hidden_dim_per_head = (hidden_dim_compressed / 12) * (-compression_factor);
    }

    // Q = Wq * input
    // K = Wk * enc_out
    // V = Wv * enc_out
    const int qkv_matmuls_n = 3;
    for (int i = 0; i < qkv_matmuls_n; i++) {
        const elem_t * qkv_weights[] = {Wq, Wk, Wv};
        const elem_t * qkv_ins[] = {input, enc_out, enc_out};
        const acc_t * qkv_bs[] = {Wq_b, Wk_b, Wk_b};
        elem_t * qkv_outs[] = {Q_buf, K_buf, V_buf};

        const elem_t * qkv_w = qkv_weights[i];
        const elem_t * qkv_in = qkv_ins[i];
        const acc_t * qkv_b = qkv_bs[i];
        elem_t * qkv_out = qkv_outs[i];

        tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
            /*A=*/ qkv_in, /*B=*/ qkv_w,
            /*D=*/ qkv_b, /*C=*/ qkv_out,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);
    }

    gemmini_fence();

    // attn = Q * K
    // attn = softmax(attn)
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = Q_buf + head * hidden_dim_per_head;
        const elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf + head * seq_len * seq_len;

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            SOFTMAX, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ true,
            false, false,
            0,
            WS);
    }

    gemmini_fence();

    // out_buf = attn * V
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = attn_buf + head * seq_len * seq_len;
        const elem_t * B = V_buf + head * hidden_dim_per_head;
        elem_t * C = out_buf + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/seq_len, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);
    }

    gemmini_fence();

    // out_buf_acc = out_buf * Wo
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        /*A=*/ out_buf, /*B=*/ Wo,
        /*D=*/ Wo_b, /*C=*/ out_buf_acc,
        /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ false,
        false, /*transpose_B=*/ false,
        true, false,
        0,
        WS);

    gemmini_fence();

    // out = LN(out_buf_acc)
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    // input = out + input
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        input,
        out,
        resadd_out,
        /*relu=*/ false,
        WS);

    gemmini_fence();
}

void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf, acc_t * out_buf_acc)
{
    // out = FF1(input)
    // out = GELU(out)
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        /*A=*/ input, /*B=*/ ff1_w,
        /*D=*/ ff1_b, /*C=*/ out_buf,
        /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        WS);

    gemmini_fence();

    // out_buf_acc = FF2(out)
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
        /*A=*/ out_buf, /*B=*/ ff2_w,
        /*D=*/ ff2_b, /*C=*/ out_buf_acc,
        /*stride_A=*/expansion_dim, /*stride_B=*/hidden_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        true, false,
        0,
        WS);

    gemmini_fence();

    // out = LN(out_buf_acc)
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);

    gemmini_fence();

    // out = out + input
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        out,
        input,
        out,
        /*relu=*/ false,
        WS);

    gemmini_fence();
}

// Note: If "enc_out == NULL", then this will act as an encoder layer.
//   Otherwise, it will act as a decoder layer. If this is an encoder layer,
//   then "cross_num_heads" and all the "W*_cross" args are ignored.
uint64_t encoder_decoder(
        int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads,
        int seq_len, int compression_factor,

        const elem_t * input, const elem_t * enc_out, elem_t * out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,

        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        const acc_t * Wo_b,
        const acc_t * Wq_cross_b, const acc_t * Wk_cross_b, const acc_t * Wv_cross_b,
        const acc_t * Wo_cross_b,

        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc,
        elem_t * resadd1_buf, elem_t * resadd2_buf)
{
    const bool is_encoder = enc_out == NULL;

    uint64_t start = read_cycles();

    attention(hidden_dim, expansion_dim, num_heads, seq_len, compression_factor,
        input, input,
        out, resadd1_buf,
        Wq, Wk, Wv, Wo,

        Wq_b, Wk_b, Wv_b,
        Wo_b,

        Q_buf, K_buf, V_buf,
        attn_buf, out_buf, out_buf_acc);

    if (!is_encoder) {
        attention(hidden_dim, expansion_dim, cross_num_heads, seq_len, compression_factor,
            resadd1_buf, enc_out,
            out, resadd2_buf,
            Wq_cross, Wk_cross, Wv_cross, Wo_cross,

            Wq_cross_b, Wk_cross_b, Wv_cross_b,
            Wo_cross_b,

            Q_buf, K_buf, V_buf,
            attn_buf, out_buf, out_buf_acc);
    }

    ffn(hidden_dim, expansion_dim, seq_len,
        is_encoder ? resadd1_buf : resadd2_buf,
        out,
        ff1_w, ff2_w,
        ff1_b, ff2_b,
        out_buf, out_buf_acc);

    uint64_t end = read_cycles();

    return end - start;
}

#define ENCODER_DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, input, enc_out, output) ({ \
    \
    static const elem_t Wqkvo[4][hidden_dim][hidden_dim]; \
    static const elem_t Wqkvo_cross[4][hidden_dim][hidden_dim]; \
    static const acc_t Wqkvo_b[4][hidden_dim]; \
    static const acc_t Wqkvo_cross_b[4][hidden_dim]; \
    static const elem_t ff_w[2][hidden_dim*expansion_dim]; \
    static const acc_t ff1_b[expansion_dim]; \
    static const acc_t ff2_b[hidden_dim]; \
    \
    static elem_t QKV_buf[3][seq_len][hidden_dim];\
    static elem_t attn_buf[num_heads][seq_len][seq_len];\
    static elem_t out_buf[seq_len][expansion_dim];\
    static acc_t out_buf_acc[seq_len][hidden_dim];\
    static elem_t resadd1_buf[seq_len][hidden_dim];\
    static elem_t resadd2_buf[seq_len][hidden_dim];\
    \
    uint64_t cycles = encoder_decoder( \
            hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, \
            compression_factor, \
            \
            input, enc_out, output, \
            Wqkvo[0], Wqkvo[1], Wqkvo[2], Wqkvo[3],\
            Wqkvo_cross[0], Wqkvo_cross[1], Wqkvo_cross[2], Wqkvo_cross[3],\
            \
            Wqkvo_b[0], Wqkvo_b[1], Wqkvo_b[2], \
            Wqkvo_b[2], \
            Wqkvo_cross_b[0], Wqkvo_cross_b[1], Wqkvo_cross_b[2], \
            Wqkvo_cross_b[3], \
            \
            ff_w[0], ff_w[1], \
            ff1_b, ff2_b, \
            \
            QKV_buf[0], QKV_buf[1], QKV_buf[2], \
            attn_buf, out_buf, out_buf_acc, \
            resadd1_buf, resadd2_buf \
    ); \
    \
    cycles; \
})

#define PRINT_ENCODER_DECODER(name, is_encoder, hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor) { \
    static const elem_t input[seq_len][hidden_dim]; \
    static const elem_t enc_out[seq_len][hidden_dim]; \
    static elem_t output[seq_len][hidden_dim]; \
    \
    char * type_str = is_encoder ? "encoder" : "decoder"; \
    \
    uint64_t cycles = ENCODER_DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, input, is_encoder ? NULL : enc_out, output); \
    \
    printf("%s stats: %s, hidden_dim=%d, expansion_dim=%d, num_heads=%d, cross_num_heads=%d, seq_len=%d, compression_factor=%d\n", \
            name, type_str, hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor); \
    printf("%s cycles: %llu\n\n", name, cycles); \
}


/**
 * @brief Runs the final classification head GEMM.
 * (This is the function declared in vit_kernels.h)
 */
void run_classification_head(
    const elem_t token_in[1][HIDDEN_DIM], 
    const elem_t weights_in[HIDDEN_DIM][NUM_CLASSES], 
    const ACC_T bias_in[1][NUM_CLASSES], 
    elem_t logits_out[1][NUM_CLASSES],
    uint64_t* gemmini_cycles
) {
    printf("  [Step 4] Running Classification Head (Gemmini WS)...\n");
    printf("    (A) Token:   (%d, %d)\n", MAT_DIM_I, MAT_DIM_K);
    printf("    (B) Weights: (%d, %d)\n", MAT_DIM_K, MAT_DIM_J);
    printf("    (C) Logits:  (%d, %d)\n", MAT_DIM_I, MAT_DIM_J);

    unsigned long start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)token_in, (elem_t*)weights_in, 
            NO_BIAS ? NULL : (void*)bias_in, (void*)logits_out,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);

    unsigned long end = read_cycles();
    *gemmini_cycles = end - start;
    printf("  [Step 4] Cycles: %llu\n", *gemmini_cycles);

#if CHECK_RESULT == 1
    // Run CPU verification
    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

    printf("    Verifying Step 4 (CPU vs Gemmini)...\n");
    full_matmul((elem_t (*)[MAT_DIM_K])token_in, (elem_t (*)[MAT_DIM_J])weights_in, 
                (ACC_T (*)[MAT_DIM_J])bias_in, gold_full);
    full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);

    if (!full_is_equal(logits_out, gold)) {
        printf("    !!! FAILURE: Classification Head Mismatch !!!\n");
        exit(1);
    } else {
        printf("    ... Success.\n");
    }
#endif
}



// --- Main Pipeline ---
int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
    gemmini_flush(0);
    uint64_t total_gemmini_cycles = 0;
    uint64_t step_cycles = 0;
    
    printf("--- Starting Full ViT Pipeline ---\n");
    printf("Seq Len: %d, Hidden Dim: %d, Num Patches: %d\n", SEQ_LEN, HIDDEN_DIM, NUM_PATCHES);
    printf("FFN Expansion Dim: %d\n", EXPANSION_DIM);

    // Initialize all static weights and image
    init_weights();

    print_matrix("Initial Image", (elem_t*)image, IMG_H, IMG_W * IMG_C);

    // --- STEP 0: Patch Extraction (CPU) ---
    // Output: `patches` (64, 64)
    run_patch_extraction(image, patches);

    print_matrix("Step 0: Extracted Patches (CPU)", (elem_t*)patches, NUM_PATCHES, PATCH_DIM);
    
    // --- STEP 1: Patch Embedding (Gemmini) ---
    // This is the snippet you posted.
    // It performs: (64, 64) x (64, 64) -> (64, 64)
    // Output: `patch_embed_out`
    #undef MAT_DIM_I
    #undef MAT_DIM_K
    #undef MAT_DIM_J
    #define MAT_DIM_I NUM_PATCHES
    #define MAT_DIM_K PATCH_DIM
    #define MAT_DIM_J HIDDEN_DIM
    run_patch_embedding(patches, patch_embed_weights, patch_embed_bias, 
                        patch_embed_out, &step_cycles);

    print_matrix("Step 1: Patch Embeddings (Gemmini)", (elem_t*)patch_embed_out, NUM_PATCHES, HIDDEN_DIM);

    // --- STEP 2: Add [CLS] & Positional Embeddings (CPU) ---
    // THIS IS THE STEP YOU ARE ASKING FOR.
    // It takes `patch_embed_out` (64, 64) and produces
    // `encoder_input` (65, 64) by:
    //   1. Concatenating the [CLS] token (xclass vector) at row 0.
    //   2. Adding the positional embeddings.
    add_cls_and_pos_embed(patch_embed_out, cls_token_weights, 
                          pos_embed_weights, encoder_input);

    print_matrix("Step 2: Encoder Input (w/ CLS+Pos)", (elem_t*)encoder_input, SEQ_LEN, HIDDEN_DIM);

    // --- STEP 3: Run Encoder Layers ---
    printf("[Step 3] Running %d Encoder Layers...\n", NUM_LAYERS);

    
    elem_t* in_ptr = (elem_t*)encoder_input;
    elem_t* out_ptr = (elem_t*)encoder_output;
    print_matrix("Step 3: Final Encoder Output", (elem_t*)in_ptr, SEQ_LEN, HIDDEN_DIM);
    
    // Redefine dims for the encoder kernels
    #undef MAT_DIM_I
    #undef MAT_DIM_K
    #undef MAT_DIM_J
    #define MAT_DIM_I SEQ_LEN
    #define MAT_DIM_K HIDDEN_DIM
    #define MAT_DIM_J HIDDEN_DIM

    for (int i = 0; i < NUM_LAYERS; i++) {
        printf("  - Running Encoder Layer %d\n", i + 1);

        step_cycles = encoder_decoder(
            HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, 0, SEQ_LEN, 1,
            in_ptr, NULL, out_ptr, // Pass NULL for enc_out (self-attention)
            
            // Self-Attention Weights
            (elem_t*)Wqkvo[0], (elem_t*)Wqkvo[1], (elem_t*)Wqkvo[2], (elem_t*)Wqkvo[3],
            NULL, NULL, NULL, NULL, // No cross-attention
            
            // Self-Attention Biases
            (acc_t*)Wqkvo_b[0], (acc_t*)Wqkvo_b[1], (acc_t*)Wqkvo_b[2], (acc_t*)Wqkvo_b[3],
            NULL, NULL, NULL, NULL, // No cross-attention
            
            // FFN Weights
            (elem_t*)ff1_w, (elem_t*)ff2_w, (acc_t*)ff1_b, (acc_t*)ff2_b,
            
            // Buffers
            (elem_t*)QKV_buf[0], (elem_t*)QKV_buf[1], (elem_t*)QKV_buf[2],
            (elem_t*)attn_buf, (elem_t*)out_buf, (acc_t*)out_buf_acc,
            (elem_t*)resadd1_buf, (elem_t*)resadd2_buf);
        
        total_gemmini_cycles += step_cycles;
        printf("  - Layer %d Cycles: %llu\n", i + 1, step_cycles);

        // Swap pointers for next iteration
        elem_t* temp = in_ptr;
        in_ptr = out_ptr;
        out_ptr = temp;
    }
    // After the loop, `in_ptr` holds the final output


    // --- STEP 4: Classification Head ---
    
    // 4a. Extract the [CLS] token from the final encoder output
    extract_cls_token((elem_t (*)[HIDDEN_DIM])in_ptr, class_token_final);

    print_matrix("Step 4a: Extracted [CLS] Token", (elem_t*)class_token_final, 1, HIDDEN_DIM);

    // 4b. Run the classifier (Gemmini)
    // (A) token: (1, 64), (B) weights: (64, 64) -> (C) logits: (1, 64)
    #undef MAT_DIM_I
    #undef MAT_DIM_K
    #undef MAT_DIM_J
    #define MAT_DIM_I 1
    #define MAT_DIM_K HIDDEN_DIM
    #define MAT_DIM_J NUM_CLASSES
    run_classification_head(class_token_final, classifier_weights, 
                            classifier_bias, final_logits, &step_cycles);
    total_gemmini_cycles += step_cycles;

    print_matrix("Step 4b: Final Logits (Gemmini)", (elem_t*)final_logits, 1, NUM_CLASSES);

    printf("--- ViT Pipeline Complete ---\n");
    printf("Total Gemmini Cycles: %llu\n", total_gemmini_cycles);

    // Find the final predicted class
    int predicted_class = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
       if (final_logits[0][i] > final_logits[0][predicted_class]) {
           predicted_class = i;
       }
    }
    printf("Final Predicted Class: %d\n", predicted_class);
    
    /*printf("Final Logits (top 16):\n");
    for (int i = 0; i < 16 && i < NUM_CLASSES; i++) {
        printf("%d ", (int)final_logits[0][i]);
    }
    printf("\n");*/

    exit(0);
}