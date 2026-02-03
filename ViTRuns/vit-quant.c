#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

//#define DEBUG
#define TOLERANCE 1

//#define DISTILLATION
#define CPU_LAYERNORM
#define CPU_SOFTMAX

int global_layer_index = 0;
bool debug_inference = false;

// Gemmini Headers
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// Model Parameters (Ensure this matches the quantization export)

// #include "includes/deitvit_cifar100_quant_params.h" 
// #include "includes/deitvit_cifar10_quant_params.h" 
// #include "includes/deitvit_mnist_quant_params.h" 
#include "includes/minivit_mnist_quant_params.h"

// Include the verified source modules directly
// (In a real build system, compile these separately and link. 
//  For simplicity here, we include the source to see the full picture.)
#include "src/attention_quantized.c"
#include "src/ffn_quantized.c"
#include "src/embedding_quantized.c"       // From Brick 6
#include "src/classifier_quantized.c" // From Brick 7
#include "src/transformer_quantized.c" // From Final Integration

#include "src/utils_quant.c"            // Verification Helper

// ==========================================
// STATIC BUFFERS (Global to avoid Stack Overflow)
// ==========================================

// 1. Input Buffer
//    Flattened Patch Buffer: [Seq, PatchDim]
//    Note: SEQ_LEN usually refers to number of patches (16)
static elem_t patch_buffer[SEQ_LEN][PATCH_DIM]; 
static elem_t temp_patch_buf[SEQ_LEN][HIDDEN_DIM];

static elem_t encoder_input[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t encoder_output[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);

// 3. Scratchpads for Encoder/Decoder
//    These are reused across layers
static elem_t Q_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t K_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t V_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t attn_buf[NUM_HEADS][TOTAL_SEQ_LEN][TOTAL_SEQ_LEN] row_align(1);
static elem_t out_buf[TOTAL_SEQ_LEN][EXPANSION_DIM] row_align(1); // Max size needed
static elem_t resadd1_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t ln_output_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t resadd2_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);

// 4. Output Buffers
static elem_t final_logits[1][NUM_CLASSES] row_align(1);


// ==========================================
// MAIN INFERENCE LOOP
// ==========================================
int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        return 1;
    }
#endif

    gemmini_flush(0);
    printf("--- Starting Verified Quantized MiniViT ---\n");
    printf("Seq: %d (+1 CLS), Hidden: %d, Heads: %d, Layers: %d\n", 
            SEQ_LEN, HIDDEN_DIM, NUM_HEADS, ENCODER_LAYERS);
           
    int correct_predictions = 0;
    int top3_correct_predictions = 0;
    int top5_correct_predictions = 0;
    uint64_t total_cycles = 0;
    int num_inferences = NUM_INFERENCES; // Defined in header

    for (int i = 0; i < num_inferences; i++) {
        if (i == 0){
            debug_inference = true;
        } else {
            debug_inference = false;
        }
        
        // 1. Prepare Input
        // Copy specific test image patch data to our patch buffer
        // (Assuming header provides all_input_patches as flat array or similar)
        // Adjust this cast based on your exact header structure
        elem_t * current_image_ptr = (elem_t*)all_input_patches[i];
        
        int ground_truth = all_ground_truths[i];

        uint64_t start = read_cycles();

        // --- STEP 1: EMBEDDING ---
        compute_patch_embeddings_quantized(
            SEQ_LEN, HIDDEN_DIM, PATCH_DIM,
            current_image_ptr,
            patch_embed_w, patch_embed_b,
            SCALE_EMBED,
            pos_embed_data, cls_token_data,
            #ifdef DISTILLATION
                dist_token_data,
            #endif
            (elem_t*)temp_patch_buf,
            (elem_t*)encoder_input
        );
        if (i == 0) {
            #ifdef DEBUG
            if (!verify_tensor("Embedding Output", 
                (elem_t*)encoder_input, (elem_t*)debug_embedding_out, 
                TOTAL_SEQ_LEN * HIDDEN_DIM, TOLERANCE)) {}//return 1;
            #endif
        }

        // --- STEP 2: ENCODER LAYERS ---
        encoder_decoder_quantized(
            HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, 0, // 0 Cross heads (Encoder only)
            TOTAL_SEQ_LEN, ENCODER_LAYERS,
            
            (elem_t*)encoder_input, NULL, (elem_t*)encoder_output,
            
            // Weights (Arrays from header)
            Wq, Wk, Wv, Wo,
            Wq_b, Wk_b, Wv_b, Wo_b,
            
            NULL, NULL, NULL, NULL, // No Cross Attn
            NULL, NULL, NULL, NULL,
            
            ff1_w, ff2_w,
            ff1_b, ff2_b,
            
            // Buffers
            (elem_t*)Q_buf, (elem_t*)K_buf, (elem_t*)V_buf,
            (elem_t*)attn_buf, (elem_t*)out_buf,
            (elem_t*)resadd1_buf, (elem_t*)resadd2_buf,
            (elem_t*)ln_output_buf,
            
            SCORE_SCALING_FACTOR,
            
            // Per-Layer Scales
            scales_q, scales_k, scales_v, scales_wo,
            scales_scores, // The fixed score scaling
            scales_ff1, scales_ff2
        );

        // CHECK 3: Final Encoder Output
        if (i == 0) {
            #ifdef DEBUG
            if (!verify_tensor("Final Encoder Output", 
                (elem_t*)encoder_output, (elem_t*)debug_final_encoder_out, 
                TOTAL_SEQ_LEN * HIDDEN_DIM, TOLERANCE)) { // Tolerate small drift accumulation
                printf("!!! Encoder Stack Failed !!!\n");
                return 1;
            }
            #endif
        }

        // --- STEP 3: CLASSIFIER HEAD ---
        // Note: encoder_output contains the sequence. CLS is at index 0.
        // If num_layers is ODD, output might be in 'encoder_output'.
        // If EVEN, it might be in 'encoder_input' pointer depending on ping-pong logic?
        // CHECK encoder_decoder_quantized logic: 
        // It updates 'layer_in' = 'out'.
        // The last 'out' written is the valid one.
        // Our function writes to 'out' (arg 3) at the end of every loop?
        // Wait, the loop writes to 'out', sets 'layer_in' = 'out'.
        // So yes, 'encoder_output' holds the final result.
        
        classifier_head_deit_quantized(
            HIDDEN_DIM, NUM_CLASSES,
            (elem_t*)encoder_output, 
            (elem_t*)final_logits,
            head_w, head_b,
            SCALE_HEAD
            #ifdef DISTILLATION
                ,    
                head_dist_w, head_dist_b
            #endif

        );

        if (i == 0) {
            #ifdef DEBUG
            if (!verify_tensor("Final Logits Output", 
                (elem_t*)final_logits, (elem_t*)debug_final_logits, 
                NUM_CLASSES, TOLERANCE)) { // Tolerate small drift accumulation
                printf("!!! Final Logits Failed !!!\n");
                return 1;
            }
            #endif
            //return 1;
        }

        uint64_t end = read_cycles();
        total_cycles += (end - start);

        // --- STEP 4: VERIFY ---
        int prediction = find_max_index((elem_t*)final_logits, NUM_CLASSES);
        
        if (prediction == ground_truth) {
            correct_predictions++;
        }
        if (is_in_top_k((elem_t*)final_logits, NUM_CLASSES, ground_truth, 3)) {
            // For Top-3 accuracy (if needed)
            top3_correct_predictions++;
        }   
        if (is_in_top_k((elem_t*)final_logits, NUM_CLASSES, ground_truth, 5)) {
            // For Top-5 accuracy (if needed)
            top5_correct_predictions++;
        }   
        
        if ((i+1) % 10 == 0) {
            printf("Processed %d/%d. Correct: %d/%d\n", i+1, num_inferences, 
                  correct_predictions, i+1 );
        }
    }

    /*printf("\n=== FINAL RESULTS ===\n");
    printf("Accuracy:%d/%d\n", 
          correct_predictions, num_inferences);
    printf("Avg Cycles: %llu\n", total_cycles / num_inferences);*/

    print_results_summary(num_inferences, correct_predictions, top3_correct_predictions, top5_correct_predictions, total_cycles);

    return 0;
}