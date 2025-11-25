#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

// External Gemmini Headers
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// Modularized Components
// 2. Include your Module Implementations (.c files)
// The compiler treats this as if you wrote all the code in this one file.
#include "src/utils.c"
#include "src/embedding.c"
#include "src/attention.c"
#include "src/ffn.c"
#include "src/transformer.c"
#include "src/classifier.c"

// Weights and Params
#include "includes/classifier_params_large_multilayer.h" 

// --- Buffers (Static Allocation) ---
// Note: These buffers are passed into the functions.
static elem_t embed_out[SEQ_LEN][HIDDEN_DIM];
static elem_t encoder_input[SEQ_LEN][HIDDEN_DIM];
static elem_t encoder_output[SEQ_LEN][HIDDEN_DIM];
static elem_t Q_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t K_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t V_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t attn_buf[NUM_HEADS][SEQ_LEN][SEQ_LEN];
static elem_t out_buf_ffn[SEQ_LEN][EXPANSION_DIM];
static acc_t out_buf_acc_ffn[SEQ_LEN][HIDDEN_DIM];
static elem_t resadd1_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t resadd2_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t pool_out[1][HIDDEN_DIM];
static elem_t ln_out[1][HIDDEN_DIM];
static elem_t final_logits[1][OUTPUT_SIZE];

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      return 1;
    }
#endif

    gemmini_flush(0);
    printf("");
    printf("--- Starting Modularized MiniViT Inference ---\n");
    printf("Running %d inferences...\n", NUM_INFERENCES);

    int correct_predictions = 0;
    uint64_t total_cycles = 0;

    // === INFERENCE LOOP ===
    for (int i = 0; i < NUM_INFERENCES; i++) {
        
        // 1. Setup Input
        const elem_t * current_patches = (const elem_t *)all_input_patches[i];
        int current_ground_truth = all_ground_truths[i];

        uint64_t start_cycles = read_cycles();

        // 2. Pre-Processing (Embedding)
        compute_patch_embeddings(SEQ_LEN, HIDDEN_DIM, PATCH_DIM,
            current_patches, patch_embed_w, patch_embed_b, pos_embed_data,
            (elem_t*)embed_out, (elem_t*)encoder_input);

        // 3. Transformer Encoder Body
        encoder_decoder(
            HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, CROSS_NUM_HEADS,
            SEQ_LEN, COMPRESSION_FACTOR, ENCODER_LAYERS,
            (const elem_t *)encoder_input, NULL, (elem_t *)encoder_output,
            (const elem_t *)Wq, (const elem_t *)Wk, (const elem_t *)Wv, (const elem_t *)Wo,
            (const elem_t *)Wq_cross, (const elem_t *)Wk_cross, (const elem_t *)Wv_cross, (const elem_t *)Wo_cross,
            (const acc_t *)Wq_b, (const acc_t *)Wk_b, (const acc_t *)Wv_b, (const acc_t *)Wo_b,
            (const acc_t *)Wq_cross_b, (const acc_t *)Wk_cross_b, (const acc_t *)Wv_cross_b, (const acc_t *)Wo_cross_b,
            (const elem_t *)ff1_w, (const elem_t *)ff2_w,
            (const acc_t *)ff1_b, (const acc_t *)ff2_b,
            (elem_t *)Q_buf, (elem_t *)K_buf, (elem_t *)V_buf,
            (elem_t *)attn_buf, (elem_t *)out_buf_ffn, (acc_t *)out_buf_acc_ffn,
            (elem_t *)resadd1_buf, (elem_t *)resadd2_buf
        );

        // 4. Post-Processing (Classification)
        compute_classification_head(SEQ_LEN, HIDDEN_DIM, OUTPUT_SIZE,
            (const elem_t *)encoder_output, (const elem_t *)pool_vector,
            head_w, head_b,
            (elem_t*)pool_out, (elem_t*)ln_out, (elem_t*)final_logits);
        
        uint64_t end_cycles = read_cycles();
        total_cycles += (end_cycles - start_cycles);

        // 5. Verification
        int prediction = find_max_index((elem_t*)final_logits, OUTPUT_SIZE);
        if (prediction == current_ground_truth) {
            correct_predictions++;
        }

        if ((i + 1) % 100 == 0) {
            printf("Processed %d / %d samples...\n", i + 1, NUM_INFERENCES);
        }
    } 

    // === REPORTING ===
    print_results_summary(NUM_INFERENCES, correct_predictions, total_cycles);

    return 0;
}