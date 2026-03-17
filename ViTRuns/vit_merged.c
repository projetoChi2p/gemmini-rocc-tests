// ==========================================
// ViT Merged Implementation for Gemmini
// ==========================================
// Build Configuration:
//   This file is configured via Makefile variables MODEL and DATASET
//   Usage: make -f RunViT.mk run MODEL=minivit DATASET=sat6
//
// The build system will automatically select the appropriate parameter
// files based on MODEL and DATASET values defined at compile time.
// See RunViT.mk and include/input_weights.h for configuration details.
// 
// Last modified: Dynamic configuration added for model/dataset selection
// ==========================================

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#endif

// Gemmini Headers
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// ==========================================
// CONFIGURATION TOGGLES
// ==========================================


// [OPTIONAL] Debugging Flags
// #define DEBUG
#define TOLERANCE 1

// [OPTIONAL] Architecture Flags --- Already handled via parameter files
//#define HYBRID_EMBEDDING
//#define DISTILLATION

// [OPTIONAL] CPU Fallbacks
#define CPU_LAYERNORM
#define CPU_SOFTMAX
// #define CPU_RESADD
// Using ReLU instead of GELU in FFN does not affect the correctness of the attention outputs, so it's a safe optional toggle for testing or ablation.
// #define REPLACE_RELU

// #define INPUT_IS_IMAGE // If defined, the embedding module will perform im2patch internally. Otherwise, it expects pre-patchified input.
#define CPU_IM2PATCH // If defined, the im2patch operation will be performed on the CPU instead of Gemmini. Only relevant if INPUT_IS_IMAGE is defined.

// Global Debug State
int global_layer_index = 0;
bool debug_inference = false;

// ==========================================
// PARAMETER LOADING
// ==========================================
#include "include/input_weights.h" // Contains all input data and model weights as C arrays
//#include "includes/minivit_cifar10_params.h" // Example parameter file for MiniViT trained on CIFAR-10. Adjust path as needed.

//#undef DEBUG
// ==========================================
// MODULE INCLUSION
// ==========================================

// Include the merged source files
// (In a real Make system, these would be compiled separately)
#include "src/math.c"
#include "src/utils_quant.c" 
#include "src/profiler.h"
#include "src/profiler.c"
#include "src/embedding_hybrid_merged.c"
#include "src/embedding_merged.c"
#include "src/attention_merged.c"
#include "src/ffn_merged.c"
#include "src/transformer_merged.c"
#include "src/classifier_merged.c"


// ==========================================
// STATIC BUFFERS (Global)
// ==========================================

// 1. Input Buffers
// [Seq, PatchDim] - Aligned for Gemmini DMA
// Note: Buffer usage depends on INPUT_IS_IMAGE flag:
//   - If INPUT_IS_IMAGE: im2patch_buf -> patch_buffer -> temp_patch_buf -> encoder_input
//   - If not: input data -> patch_buffer (if reorder needed) -> temp_patch_buf -> encoder_input
static elem_t im2patch_buf[SEQ_LEN][PATCH_DIM] row_align(1);      // Buffer for image-to-patch conversion
static elem_t patch_buffer[SEQ_LEN][PATCH_DIM] row_align(1);      // Buffer for patch reordering (HWC->CHW)
static elem_t temp_patch_buf[SEQ_LEN][HIDDEN_DIM] row_align(1);   // Buffer for projected patches

// 2. Transformer State Buffers
static elem_t encoder_input[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t encoder_output[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);

// 3. Scratchpads (Reused across layers) - Dedicated buffers per stage
static elem_t Q_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t K_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t V_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);

static elem_t attn_scores_buf[NUM_HEADS][TOTAL_SEQ_LEN][TOTAL_SEQ_LEN] row_align(1);
static elem_t attn_probs_buf[NUM_HEADS][TOTAL_SEQ_LEN][TOTAL_SEQ_LEN] row_align(1);
static elem_t attn_norm_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t attn_context_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t attn_wo_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t attn_resadd_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);

static elem_t ffn_norm_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static float  ffn_norm_in_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1); // For FP32 Norm input if needed
static elem_t ffn_fc1_buf[TOTAL_SEQ_LEN][EXPANSION_DIM] row_align(1);
static float  ffn_fc1_out[TOTAL_SEQ_LEN][EXPANSION_DIM] row_align(1);
static elem_t ffn_gelu_buf[TOTAL_SEQ_LEN][EXPANSION_DIM] row_align(1);
static elem_t ffn_fc2_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static elem_t ffn_resadd_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);

static elem_t ln_output_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);
static acc_t ffn_acc_buf[TOTAL_SEQ_LEN][HIDDEN_DIM] row_align(1);

// 5. Output Buffers
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
    printf("Total Seq Len: %d\n", TOTAL_SEQ_LEN);
    gemmini_flush(0);
    
    #ifdef QUANTIZED
        printf("--- Starting merged ViT (Quantized Int8) ---\n");
    #else
        printf("--- Starting merged ViT (FP32 Verification) ---\n");
    #endif

    printf("Seq: %d, Hidden: %d, Heads: %d, Layers: %d\n", 
            SEQ_LEN, HIDDEN_DIM, NUM_HEADS, ENCODER_LAYERS);
    
    // Print build configuration
    #define STRINGIFY_VALUE(x) #x
    #define TOSTRING_VALUE(x) STRINGIFY_VALUE(x)
    printf("Build Config: MODEL=%s, DATASET=%s\n", 
            TOSTRING_VALUE(MODEL), TOSTRING_VALUE(DATASET));
    
    // Initialize profiler
    profiler_init(ENCODER_LAYERS);
           
    int correct_predictions = 0;
    int expected_logits_errors = 0;
    int top3_correct_predictions = 0;
    int top5_correct_predictions = 0;
    uint64_t total_cycles = 0;

    // Handle inference count definition differences
    #ifndef NUM_INFERENCES
        #define NUM_INFERENCES 100
    #endif
    printf("Running %d inferences...\n", NUM_INFERENCES);
    int num_inferences = NUM_INFERENCES;


    for (int i = 0; i < num_inferences; i++) {
        
        // Setup Debug State
        if (i == 0) debug_inference = true;
        else debug_inference = false;
        
        // 1. Prepare Input
        #ifdef INPUT_IS_IMAGE
            // When INPUT_IS_IMAGE is defined, input is a full image (not pre-patchified)
            // Note: all_input_images array must be defined in parameter file with size [N][IMAGE_SIZE*IMAGE_SIZE*CHANNELS]
            elem_t * current_image_ptr = (elem_t*)all_input_images[i];
        #else
            // When INPUT_IS_IMAGE is not defined, input is pre-patchified data
            // all_input_patches array size: [N][SEQ_LEN][PATCH_DIM]
            elem_t * current_patch_ptr = (elem_t*)all_input_patches[i];
        #endif
        
        int ground_truth = all_ground_truths[i];

        #ifdef DEBUG
        if (i == 0) {
            #ifdef INPUT_IS_IMAGE
                verify_tensor("Input Image", 
                    (elem_t*)current_image_ptr, (elem_t*)current_image_ptr, 
                    IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS, TOLERANCE);
            #else
                verify_tensor("Input Patches", 
                    (elem_t*)current_patch_ptr, (elem_t*)current_patch_ptr, 
                    SEQ_LEN * PATCH_DIM, TOLERANCE);
            #endif
            }
        #endif

        uint64_t start = read_cycles();
        uint64_t phase_start, phase_end;

        // --- STEP 1: EMBEDDING ---
        // merged call handles scaling & token types internally
        phase_start = read_cycles();
        #ifdef HYBRID_EMBEDDING
            compute_hybrid_embeddings(
                1, NUM_CHANNELS,              // FIX: Use IN_CHANNELS exported from Python
                IN_ROW_DIM, IN_COL_DIM,      
                HIDDEN_DIM,                  
                OUT_ROW_DIM, OUT_COL_DIM,    
                STRIDE, PADDING, KERNEL_DIM, 
                
                current_image_ptr,           
                conv_stem_w,                 
                conv_stem_b,                 
                
                #ifdef QUANTIZED
                    SCALE_EMBED,                 
                #endif
                
                pos_embed_data, 
                cls_token_data,
                #ifdef DISTILLATION
                    dist_token_data,
                #endif
                
                (elem_t*)temp_patch_buf,    
                (elem_t*)encoder_input       // Initial input for Layer 0
            );
        #else
            compute_patch_embeddings(
                SEQ_LEN, HIDDEN_DIM, PATCH_DIM,
                PATCH_SIZE, NUM_CHANNELS, false,
                
                // Input Mode (false = already patchified, true = full image)
                #ifdef INPUT_IS_IMAGE
                    true, 
                    IMAGE_SIZE, IMAGE_SIZE,
                    current_image_ptr,          // input_data (full image)
                #else
                    false,                      // input_is_image
                    0,                          // img_height (unused when input_is_image=false)
                    0,                          // img_width (unused when input_is_image=false)
                    current_patch_ptr,          // input_data (patches in this case)
                #endif
                
                patch_embed_w, patch_embed_b,

                // Quantization Scale (Conditional)
                #ifdef QUANTIZED
                    SCALE_EMBED,
                #endif

                pos_embed_data, 
                cls_token_data,
                
                // Distillation Token (Conditional)
                #ifdef DISTILLATION
                    dist_token_data,
                #endif

                #ifdef INPUT_IS_IMAGE
                    (elem_t*)im2patch_buf,  // im2patch_buf (required when input_is_image=true)
                #else
                    NULL,                   // im2patch_buf (unused when input_is_image=false)
                #endif
                (elem_t*)patch_buffer,      // patch_reorder_buf
                (elem_t*)temp_patch_buf,    // temp_patch_buf
                (elem_t*)encoder_input      // final_input_buf
            );
        #endif
        phase_end = read_cycles();
        if (i == 0) g_profile.embedding = phase_end - phase_start;
        
        #ifdef DEBUG
        if (i == 0)
            verify_tensor("Embedding Output", 
                (elem_t*)encoder_input, (elem_t*)debug_embedding_out, 
                TOTAL_SEQ_LEN * HIDDEN_DIM, TOLERANCE);
        #endif

        // --- STEP 2: TRANSFORMER ENCODER ---
        phase_start = read_cycles();
        compute_transformer_blocks(
            HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, 
            0, // Cross heads (Encoder = 0)
            TOTAL_SEQ_LEN, ENCODER_LAYERS,
            
            // FP32 Compression arg
            #ifndef QUANTIZED
            1, 
            #endif

            (elem_t*)encoder_input, NULL, (elem_t*)encoder_output,
            
            // Weights (Self Attn)
            Wq, Wk, Wv, Wo,
            Wq_b, Wk_b, Wv_b, Wo_b,
            
            // Weights (Cross Attn - Unused in Encoder)
            NULL, NULL, NULL, NULL,
            NULL, NULL, NULL, NULL,
            
            // Weights (FFN)
            ff1_w, ff2_w, ff1_b, ff2_b,
            
            // Scratchpads
            (elem_t*)Q_buf, (elem_t*)K_buf, (elem_t*)V_buf,
            (elem_t*)attn_scores_buf, (elem_t*)attn_probs_buf,
            (elem_t*)attn_norm_buf, (elem_t*)attn_context_buf,
            (elem_t*)attn_wo_buf, (elem_t*)attn_resadd_buf,

            (elem_t*)ffn_norm_buf, (float*)ffn_norm_in_buf, 
            (elem_t*)ffn_fc1_buf, (float*) ffn_fc1_out,
            (elem_t*)ffn_gelu_buf, (elem_t*)ffn_fc2_buf,
            (elem_t*)ffn_resadd_buf,

            // Accumulator / Extra Buffers 
            (acc_t*)ffn_acc_buf,

            SCORE_SCALING_FACTOR,
            SCORE_SCALING_FACTOR,
            
            // Per-Layer Scales (Quantized Only)
            #ifdef QUANTIZED
                scales_q, scales_k, scales_v, scales_wo,
                scales_score, scales_ff1, scales_ff2,
            #endif
            scales_context,
            scales_act_ln1,
            scales_act_ln2,
            scales_act_res1,
            scales_act_res2
        );
        phase_end = read_cycles();
        if (i == 0) g_profile.encoder.total = phase_end - phase_start;

        #ifdef DEBUG
            if (i == 0) {
                verify_tensor("Encoder Output", 
                    (elem_t*)encoder_output, (elem_t*)debug_final_encoder_out, 
                    TOTAL_SEQ_LEN * HIDDEN_DIM, TOLERANCE);
            }
        #endif
        // --- STEP 3: CLASSIFIER HEAD ---
        phase_start = read_cycles();
        compute_classifier(
            HIDDEN_DIM, NUM_CLASSES,
            (elem_t*)encoder_output, 
            (elem_t*)final_logits,
            head_w, head_b,
            (elem_t*)ln_output_buf, // Reuse scratchpad for LN
            scales_act_ln_final[0],
            // Quantized Specifics
            #ifdef QUANTIZED
                SCALE_HEAD // Example scale for classifier head (depends on final activation range)
            #else
                ACC_SCALE_IDENTITY
            #endif
                
            #ifdef DISTILLATION
                , head_dist_w, head_dist_b
            #endif
        );
        phase_end = read_cycles();
        if (i == 0) g_profile.classifier = phase_end - phase_start;

        // #ifdef DEBUG
        //     memcpy(final_logits, debug_final_logits, NUM_CLASSES * sizeof(elem_t));
        // #endif

        if (i == 0) {
            #ifdef DEBUG
            if (!verify_tensor("Final Logits Output", 
                (elem_t*)final_logits, (elem_t*)debug_final_logits, 
                NUM_CLASSES, TOLERANCE)) { // Tolerate small drift accumulation
                printf("!!! Final Logits Failed !!!\n");
                //return 1;
            }
            display_tensor_distribution_histogram("Final Logits Distribution", 
                (elem_t*)final_logits, NUM_CLASSES);
            #endif
            
            //return 1;
        }

        #ifdef DEBUG
            printf("Expected final Logits :\n");     
            int expected_class = find_max_index((elem_t*)debug_all_logits_int[i], NUM_CLASSES);
            if (expected_class == ground_truth) {
                // printf("Expected prediction matches ground truth: %d\n", expected_class);
            } else {
                //printf("Expected prediction does NOT match ground truth: Predicted %d, Expected %d\n", expected_class, ground_truth);
                expected_logits_errors ++;
            }

        #endif
        
        uint64_t end = read_cycles();
        if (i == 0) g_profile.total = end - start;
        total_cycles += (end - start);

        // --- STEP 4: VERIFY ---
        int prediction = find_max_index((elem_t*)final_logits, NUM_CLASSES);
        //int real_ground_truth = find_max_index((elem_t*)debug_final_logits, NUM_CLASSES); // +1 if class indices start at 1

        /*if (i == 10){
            return 0;
        } else {
            printf("Inference %d Results:\n", i);
            printf("  Prediction: %d\n", prediction); // +1 if class indices start at 1
            printf("  Expected:   %d\n", ground_truth);
        }*/

        // Display Histogram of top5 logits for first inference
        #ifdef DEBUG
        if (i <= 10) {
            printf("\nTop-5 Logits:\n");
            /*for (int j = 0; j < NUM_CLASSES; j++) {
                printf("  Class %d: %de-2\t", j, (int)(final_logits[0][j]*100));
                for (int k = 0; k < final_logits[0][j]*100/elem_t_max; k += 1) { // Simple histogram bar
                    printf("#");
                }
                printf("\n");
            }*/
            if (prediction == ground_truth) {
                printf("Correct prediction for image %d: Predicted %d\n", i, prediction);
            } else {
                printf("Incorrect prediction for image %d: Predicted %d, Expected %d\n", i, prediction, ground_truth);
            }
        }
        #endif
        
        if (prediction == ground_truth) {
            correct_predictions++;
        }
        
        // Helper function should be available in utils.c
        if (is_in_top_k((elem_t*)final_logits, NUM_CLASSES, ground_truth, 3)) {
            top3_correct_predictions++;
            // printf("Top-3 correct prediction for image %d: Predicted %d, Expected %d\n", i, prediction, ground_truth);
        }   
        if (is_in_top_k((elem_t*)final_logits, NUM_CLASSES, ground_truth, 5)) {
            top5_correct_predictions++;
            // printf("Top-5 correct prediction for image %d: Predicted %d, Expected %d\n", i, prediction, ground_truth);
        }   
        
        
        if ((i+1) % 10 == 0) {
            printf("Processed %d/%d. Correct: %d/%d\n", i+1, num_inferences, 
                  correct_predictions, i+1 );
        }
    }

    print_results_summary(num_inferences, correct_predictions, top3_correct_predictions, top5_correct_predictions, total_cycles);
    
    printf("Expected logits errors (debug vs actual): %d\n", expected_logits_errors);

    // Print profiling report (uses data from first inference)
    // profiler_print_report();
    
    // Cleanup
    profiler_free();

    return 0;
}