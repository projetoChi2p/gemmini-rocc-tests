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
#define TOLERANCE 3

// [OPTIONAL] Architecture Flags --- Already handled via parameter files
//#define HYBRID_EMBEDDING
//#define DISTILLATION

// [OPTIONAL] CPU Fallbacks
#define CPU_LAYERNORM
#define CPU_SOFTMAX
// Using ReLU instead of GELU in FFN does not affect the correctness of the attention outputs, so it's a safe optional toggle for testing or ablation.
// #define REPLACE_RELU

// Global Debug State
int global_layer_index = 0;
bool debug_inference = false;

// ==========================================
// PARAMETER LOADING
// ==========================================

// === QUANTIZED MODELS ===
// -- MiniViT--
#ifndef ELEM_T_IS_FLOAT
#define QUANTIZED
// #include "includes/minivit_sat6_quant_params.h"     // SAT6         MiniViT
// #include "includes/minivit_mnist_quant_params.h"     // MNIST        MiniViT
#include "includes/minivit_cifar10_quant_params.h"   // CIFAR10      MiniViT
// #include "includes/minivit_cifar100_quant_params.h"  // CIFAR100     MiniViT
// -- DeiTViT --
// #include "includes/deitvit_mnist_quant_params.h"     // MNIST        DeiTViT
// #include "includes/deitvit_cifar10_quant_params.h"   // CIFAR10      DeiTViT
// #include "includes/deitvit_cifar100_quant_params.h"  // CIFAR100     DeiTViT

// -- DeiTCNN --
// #include "includes/deitcnn_mnist_quant_params.h"     // MNIST        DeiTCNN
// #include "includes/deitcnn_cifar10_quant_params.h"   // CIFAR10      DeiTCNN
// #include "includes/deitcnn_cifar100_quant_params.h"  // CIFAR100     DeiTCNN
#else
// === NON-QUANTIZED MODELS ===
// -- MiniViT--
// #include "includes/minivit_emnist_params.h"            // SAT6         MiniViT
// #include "includes/minivit_sat6_params.h"            // SAT6         MiniViT
// #include "includes/minivit_mnist_params.h"           // MNIST        MiniViT
// #include "includes/minivit_cifar10_params.h"         // CIFAR10      MiniViT
// #include "includes/minivit_cifar100_params.h"        // CIFAR100     MiniViT
// -- DeiTViT --
// #include "includes/deitvit_mnist_params.h"           // MNIST        DeiTViT
// #include "includes/deitvit_cifar10_params.h"         // CIFAR10      DeiTViT
// #include "includes/deitvit_cifar100_params.h"        // CIFAR100     DeiTViT

// -- DeiTCNN --
// #include "includes/deitcnn_mnist_params.h"           // MNIST        DeiTCNN
// #include "includes/deitcnn_cifar10_params.h"         // CIFAR10      DeiTCNN
// #include "includes/deitcnn_cifar100_params.h"        // CIFAR100     DeiTCNN
#endif
// ==========================================
// MODULE INCLUSION
// ==========================================

// Include the merged source files
// (In a real Make system, these would be compiled separately)
#include "src/math.c"
#include "src/utils_quant.c" 
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
static elem_t patch_buffer[SEQ_LEN][PATCH_DIM] row_align(1);
static elem_t temp_patch_buf[SEQ_LEN][HIDDEN_DIM] row_align(1);

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
static elem_t ffn_fc1_buf[TOTAL_SEQ_LEN][EXPANSION_DIM] row_align(1);
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
           
    int correct_predictions = 0;
    int top3_correct_predictions = 0;
    int top5_correct_predictions = 0;
    uint64_t total_cycles = 0;
    
    // Handle inference count definition differences
    #ifndef NUM_INFERENCES
        #define NUM_INFERENCES 1
    #endif
    printf("Running %d inferences...\n", NUM_INFERENCES);
    int num_inferences = NUM_INFERENCES;


    for (int i = 0; i < num_inferences; i++) {
        
        // Setup Debug State
        if (i == 0) debug_inference = true;
        else debug_inference = false;
        
        // 1. Prepare Input
        elem_t * current_image_ptr = (elem_t*)all_input_patches[i];
        int ground_truth = all_ground_truths[i];

        #ifdef DEBUG
        if (i == 0) {
            verify_tensor("Input Image", 
                (elem_t*)current_image_ptr, (elem_t*)current_image_ptr, 
                SEQ_LEN * HIDDEN_DIM, TOLERANCE); // Exact match expected
            }
        #endif

        uint64_t start = read_cycles();

        // --- STEP 1: EMBEDDING ---
        // merged call handles scaling & token types internally
        // --- STEP 1: EMBEDDING ---
        #ifdef HYBRID_EMBEDDING
            compute_hybrid_embeddings(
                1, NUM_CHANNELS,              
                IN_ROW_DIM, IN_COL_DIM,      
                HIDDEN_DIM,                  
                OUT_ROW_DIM, OUT_COL_DIM,    
                STRIDE, PADDING, KERNEL_DIM, 
                
                current_image_ptr,           
                conv_stem_w,                 
                conv_stem_b,                 
                
                // Conditional Scale Argument
                #ifdef QUANTIZED
                    SCALE_EMBED,                 
                #endif
                
                pos_embed_data, 
                cls_token_data,
                #ifdef DISTILLATION
                    dist_token_data,
                #endif
                
                (elem_t*)temp_patch_buf,    
                (elem_t*)encoder_input      
            );
        #else
            compute_patch_embeddings(
                SEQ_LEN, HIDDEN_DIM, PATCH_DIM,
                PATCH_SIZE, NUM_CHANNELS, false,
                current_image_ptr,
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

                (elem_t*)patch_buffer,
                (elem_t*)temp_patch_buf,
                (elem_t*)encoder_input
            );
        #endif
        
        #ifdef DEBUG
        if (i == 0)
            verify_tensor("Embedding Output", 
                (elem_t*)encoder_input, (elem_t*)debug_embedding_out, 
                TOTAL_SEQ_LEN * HIDDEN_DIM, TOLERANCE);
        #endif

        // --- STEP 2: TRANSFORMER ENCODER ---
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

            (elem_t*)ffn_norm_buf, (elem_t*)ffn_fc1_buf,
            (elem_t*)ffn_gelu_buf, (elem_t*)ffn_fc2_buf,
            (elem_t*)ffn_resadd_buf,

            // Accumulator / Extra Buffers 
            (acc_t*)ffn_acc_buf,

            SCORE_SCALING_FACTOR,
            SCORE_SCALING_FACTOR,
            
            // Per-Layer Scales (Quantized Only)
            #ifdef QUANTIZED
                scales_q, scales_k, scales_v, scales_wo,
                scales_scores, scales_ff1, scales_ff2,
            #endif
            CONTEXT_SCALE
        );

        #ifdef DEBUG
            if (i == 0) {
                verify_tensor("Encoder Output", 
                    (elem_t*)encoder_output, (elem_t*)debug_final_encoder_out, 
                    TOTAL_SEQ_LEN * HIDDEN_DIM, TOLERANCE);
            }
        #endif
        // --- STEP 3: CLASSIFIER HEAD ---
        compute_classifier(
            HIDDEN_DIM, NUM_CLASSES,
            (elem_t*)encoder_output, 
            (elem_t*)final_logits,
            head_w, head_b,
            (elem_t*)ln_output_buf, // Reuse scratchpad for LN
            
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

        
        uint64_t end = read_cycles();
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
        
        if (prediction == ground_truth) {
            correct_predictions++;
        }
        
        // Helper function should be available in utils.c
        if (is_in_top_k((elem_t*)final_logits, NUM_CLASSES, ground_truth, 3)) {
            top3_correct_predictions++;
        }   
        if (is_in_top_k((elem_t*)final_logits, NUM_CLASSES, ground_truth, 5)) {
            top5_correct_predictions++;
        }   
        
        
        if ((i+1) % 10 == 0) {
            printf("Processed %d/%d. Correct: %d/%d\n", i+1, num_inferences, 
                  correct_predictions, i+1 );
        }
    }

    print_results_summary(num_inferences, correct_predictions, top3_correct_predictions, top5_correct_predictions, total_cycles);

    return 0;
}