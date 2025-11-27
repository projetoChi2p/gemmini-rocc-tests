#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

//#define DEBUG_STEPS

#define debug_step(step, i) if (i % 1 == 0){ \
            printf("[DEBUG] > Step: %s for inference %d.\n", step, i); \
        } 

// External Gemmini Headers
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// Weights and Params
#include "includes/minivit_cls_params.h" 

// 1. Patch Embedding Buffers
static elem_t temp_patch_embed[SEQ_LEN][HIDDEN_DIM]; // Holds just the 16 patches

// 2. Transformer Buffers (Must hold 17 tokens)
static elem_t encoder_input[TOTAL_SEQ_LEN][HIDDEN_DIM];
static elem_t encoder_output[TOTAL_SEQ_LEN][HIDDEN_DIM];

static elem_t Q_buf[TOTAL_SEQ_LEN][HIDDEN_DIM];
static elem_t K_buf[TOTAL_SEQ_LEN][HIDDEN_DIM];
static elem_t V_buf[TOTAL_SEQ_LEN][HIDDEN_DIM];
static elem_t attn_buf[NUM_HEADS][TOTAL_SEQ_LEN][TOTAL_SEQ_LEN]; // 17x17 Attention Map
static elem_t out_buf_ffn[TOTAL_SEQ_LEN][EXPANSION_DIM];
static acc_t out_buf_acc_ffn[TOTAL_SEQ_LEN][HIDDEN_DIM];
static elem_t resadd1_buf[TOTAL_SEQ_LEN][HIDDEN_DIM];
static elem_t resadd2_buf[TOTAL_SEQ_LEN][HIDDEN_DIM];

// 3. Head Buffers
static elem_t ln_out[1][HIDDEN_DIM];
static elem_t final_logits[1][OUTPUT_SIZE];

// Include Module Implementations (AFTER buffers are defined, if they use them globally, 
// though passing pointers is better style. Assuming your src/*.c files perform the logic provided above)
#include "src/utils.c"
#include "src/embedding.c"
#include "src/attention.c"
#include "src/ffn.c"
#include "src/transformer.c"
#include "src/classifier.c"

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      return 1;
    }
#endif

    gemmini_flush(0);
    printf("--- Starting Modularized MiniViT Inference (CLS Token) ---\n");
    printf("Running %d inferences...\n", NUM_INFERENCES);

    int correct_predictions = 0;
    uint64_t total_cycles = 0;

    // === INFERENCE LOOP ===
    for (int i = 0; i < NUM_INFERENCES; i++) {
        
        // 1. Setup Input
        #ifdef DEBUG_STEPS
            debug_step("Setup Input", i);
        #endif
        const elem_t * current_patches = (const elem_t *)all_input_patches[i];
        int current_ground_truth = all_ground_truths[i];

        uint64_t start_cycles = read_cycles();

        // 2. Pre-Processing (Embedding + CLS Concatenation)
        #ifdef DEBUG_STEPS
            debug_step("Pre-Processing (Embedding)", i);
        #endif
        
        compute_patch_embeddings(
            SEQ_LEN, HIDDEN_DIM, PATCH_DIM,
            current_patches, 
            patch_embed_w, patch_embed_b, 
            pos_embed_data,
            cls_token_data,          // <--- NEW: Passed from header
            (elem_t*)temp_patch_embed, // Temp storage for 16 patches
            (elem_t*)encoder_input     // Destination for 17 tokens
        );

        // 3. Transformer Encoder Body
        #ifdef DEBUG_STEPS
            debug_step("Transformer Encoder Body", i);
        #endif
        
        // Note: Passing TOTAL_SEQ_LEN (17) instead of SEQ_LEN
        encoder_decoder(
            HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, CROSS_NUM_HEADS,
            TOTAL_SEQ_LEN, // <--- 17
            COMPRESSION_FACTOR, ENCODER_LAYERS,
            (const elem_t *)encoder_input, NULL, (elem_t *)encoder_output,
            (const elem_t *)Wq, (const elem_t *)Wk, (const elem_t *)Wv, (const elem_t *)Wo,
            (const elem_t *)Wq_cross, (const elem_t *)Wk_cross, (const elem_t *)Wv_cross, (const elem_t *)Wo_cross,
            (const acc_t *)Wq_b, (const acc_t *)Wk_b, (const acc_t *)Wv_b, (const acc_t *)Wo_b,
            (const acc_t *)Wq_cross_b, (const acc_t *)Wk_cross_b, (const acc_t *)Wv_cross_b, (const acc_t *)Wo_cross_b,
            (const elem_t *)ff1_w, (const elem_t *)ff2_w,
            (const acc_t *)ff1_b, (const acc_t *)ff2_b,
            (elem_t *)Q_buf, (elem_t *)K_buf, (elem_t *)V_buf,
            (elem_t *)attn_buf, (elem_t *)out_buf_ffn, (acc_t *)out_buf_acc_ffn,
            (elem_t *)resadd1_buf, (elem_t *)resadd2_buf, 
            SCORE_SCALING_FACTOR
        );

        // 4. Post-Processing (Classification)
        #ifdef DEBUG_STEPS
            debug_step("Post-Processing (Classification)", i);
        #endif
        
        // Note: We pass encoder_output directly. 
        // It points to index 0, which is the CLS token output.
        // We removed pool_vector because we don't average anymore.
        compute_classification_head(
            HIDDEN_DIM, OUTPUT_SIZE,
            (const elem_t *)encoder_output, // Takes Row 0 (CLS)
            head_w, head_b,
            (elem_t*)ln_out, (elem_t*)final_logits
        );
        
        uint64_t end_cycles = read_cycles();
        total_cycles += (end_cycles - start_cycles);

        // 5. Verification
        #ifdef DEBUG_STEPS
            debug_step("Verification", i);
        #endif
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