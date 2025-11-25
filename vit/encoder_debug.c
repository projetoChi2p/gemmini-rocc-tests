#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

void verify_matrix(const char * step_name, int rows, int cols, 
                   const elem_t * calculated, const elem_t * expected);

// Inclui os pesos treinados, a imagem, e o label real
#include "includes/classifier_params_debug.h" 
// Inclui as declarações das funções (attention, ffn, etc)
#include "transformer_layers_debug.c" 

// --- Verification Helper ---
#define VERIFY_EPSILON 1.0f // Tolerance for float comparison

void verify_matrix(const char * step_name, int rows, int cols, 
                   const elem_t * calculated, const elem_t * expected) {
    int errors = 0;
    float max_diff = 0.0f;
    int size = rows * cols;

    printf("  > Verifying %s... ", step_name);

    for (int i = 0; i < size; i++) {
        float c_val = (float)calculated[i];
        float e_val = (float)expected[i];
        float diff = fabs(c_val - e_val);
        
        if (diff > max_diff) max_diff = diff;
        
        if (diff > VERIFY_EPSILON) {
            if (errors < 1) { // Print only the first error details
                printf("\n    [FAIL] Mismatch at index %d: Calc %f vs Exp %f (Diff: %f)", 
                       i, c_val, e_val, diff);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("[PASS] (Max Diff: %.5f)\n", max_diff);
    } else {
        printf("\n    [FAIL] Total Errors: %d / %d\n", errors, size);
    }
}

// Helper para encontrar o dígito previsto
int find_max_index(elem_t* arr, int size) {
    int max_idx = 0;
    elem_t max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// --- Buffers Intermediários ---
// Passo 1: Embedding
static elem_t embed_out[SEQ_LEN][HIDDEN_DIM];
static elem_t encoder_input[SEQ_LEN][HIDDEN_DIM]; // embed_out + pos_embed

// Passo 2: Encoder (e seus buffers internos)
static elem_t encoder_output[SEQ_LEN][HIDDEN_DIM];
static elem_t Q_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t K_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t V_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t attn_buf[NUM_HEADS][SEQ_LEN][SEQ_LEN];
static elem_t out_buf_attn[SEQ_LEN][HIDDEN_DIM];
static acc_t out_buf_acc_attn[SEQ_LEN][HIDDEN_DIM];
static elem_t resadd1_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t resadd2_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t out_buf_ffn[SEQ_LEN][EXPANSION_DIM]; // Buffer maior
static acc_t out_buf_acc_ffn[SEQ_LEN][HIDDEN_DIM];

// Passo 3: Head
static elem_t pool_out[1][HIDDEN_DIM]; // Média Global
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
    enum tiled_matmul_type_t tiled_matmul_type = WS;
    printf("--- Starting MiniViT Classifier Full Inference (Hybrid) --- \n");

    // === PASSO 1: PATCH + POS EMBEDDING ===
    tiled_matmul_auto(SEQ_LEN, HIDDEN_DIM, PATCH_DIM,
        input_patches, patch_embed_w, patch_embed_b, embed_out,
        PATCH_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, false, false, 0, tiled_matmul_type);

    tiled_resadd_auto(SEQ_LEN, HIDDEN_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        embed_out, pos_embed_data, encoder_input,
        false, tiled_matmul_type);

    // >>> VERIFY STEP 1 <<<
    verify_matrix("Embedding + Pos", SEQ_LEN, HIDDEN_DIM, 
                  (elem_t*)encoder_input, debug_embedding_final);
    
    // === PREPARE DEBUG POINTERS FOR ENCODER LOOP ===
    // Construct arrays of pointers to the header variables
    const elem_t * debug_attn_list[] = {
        debug_layer0_attn_out, 
        debug_layer1_attn_out, 
        debug_layer2_attn_out, 
        debug_layer3_attn_out 
    };
    
    const elem_t * debug_ffn_list[] = {
        debug_layer0_ffn_out, 
        debug_layer1_ffn_out, 
        debug_layer2_ffn_out, 
        debug_layer3_ffn_out 
    };

    // === PASSO 2: ENCODER ===
    encoder_decoder(
        HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, CROSS_NUM_HEADS,
        SEQ_LEN, COMPRESSION_FACTOR, ENCODER_LAYERS,
        (const elem_t *)encoder_input,
        NULL, 
        (elem_t *)encoder_output,
        (const elem_t *)Wq, (const elem_t *)Wk, (const elem_t *)Wv, (const elem_t *)Wo,
        (const elem_t *)Wq_cross, (const elem_t *)Wk_cross, (const elem_t *)Wv_cross, (const elem_t *)Wo_cross,
        (const acc_t *)Wq_b, (const acc_t *)Wk_b, (const acc_t *)Wv_b, (const acc_t *)Wo_b,
        (const acc_t *)Wq_cross_b, (const acc_t *)Wk_cross_b, (const acc_t *)Wv_cross_b, (const acc_t *)Wo_cross_b,
        (const elem_t *)ff1_w, (const elem_t *)ff2_w,
        (const acc_t *)ff1_b, (const acc_t *)ff2_b,
        (elem_t *)Q_buf, (elem_t *)K_buf, (elem_t *)V_buf,
        (elem_t *)attn_buf, 
        (elem_t *)out_buf_ffn, 
        (acc_t *)out_buf_acc_ffn,
        (elem_t *)resadd1_buf, 
        (elem_t *)resadd2_buf,
        // NEW ARGUMENTS:
        debug_attn_list,
        debug_ffn_list,
        debug_l0_q, debug_l0_k, debug_l0_v, debug_l0_scores, debug_l0_probs,
        debug_l0_ffn_fc1, debug_l0_ffn_fc2
    );

    // === PASSO 3: CLASSIFICATION HEAD ===
    acc_scale_t pool_scale = (acc_scale_t)(1.0f / SEQ_LEN);

    // 3a. Global Average Pooling
    tiled_matmul_auto(1, HIDDEN_DIM, SEQ_LEN,
        (const elem_t *)pool_vector, (const elem_t *)encoder_output, NULL, (elem_t *)pool_out,
        SEQ_LEN, HIDDEN_DIM, 0, HIDDEN_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, pool_scale, 0,
        false, false, false, false, false, 0, tiled_matmul_type);

    // >>> VERIFY STEP 3a <<<
    verify_matrix("Global Pool", 1, HIDDEN_DIM, (elem_t*)pool_out, debug_global_pool_out);

    // 3b. LayerNorm
    tiled_norm_auto(1, HIDDEN_DIM, (acc_t*)pool_out, (elem_t*)ln_out,
        ACC_SCALE_IDENTITY, LAYERNORM, tiled_matmul_type);

    // >>> VERIFY STEP 3b <<<
    verify_matrix("Final LayerNorm", 1, HIDDEN_DIM, (elem_t*)ln_out, debug_final_ln_out);
        
    // 3c. Final Matmul
    tiled_matmul_auto(1, OUTPUT_SIZE, HIDDEN_DIM,
        ln_out, head_w, head_b, final_logits,
        HIDDEN_DIM, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, false, false, 0, tiled_matmul_type);

    // >>> VERIFY STEP 3c <<<
    verify_matrix("Final Logits", 1, OUTPUT_SIZE, (elem_t*)final_logits, debug_final_logits);

    // --- FINAL CHECK ---
    int prediction = find_max_index((elem_t*)final_logits, OUTPUT_SIZE);
    printf("\n--- Final Result ---\n");
    printf("Predicted: %d | True: %d\n", prediction, ground_truth[0]);

    return 0;
}