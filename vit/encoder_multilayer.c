#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// Inclui os pesos treinados, a imagem, e o label real
#include "includes/classifier_params_multilayer.h" 
// Inclui as declarações das funções (attention, ffn, etc)
#include "transformer_layers_multiencoder.c" 

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
    // 1a. Patch Embed (Matmul: input_patches @ patch_embed_w + patch_embed_b)
    tiled_matmul_auto(SEQ_LEN, HIDDEN_DIM, PATCH_DIM,
        input_patches, patch_embed_w, patch_embed_b, embed_out,
        PATCH_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, false, false, 0, tiled_matmul_type);

    // 1b. Pos Embed (Add: embed_out + pos_embed_data)
    tiled_resadd_auto(SEQ_LEN, HIDDEN_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        embed_out, pos_embed_data, encoder_input,
        false, tiled_matmul_type);
    
    // === PASSO 2: ENCODER ===
    encoder_decoder(
        HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, CROSS_NUM_HEADS,
        SEQ_LEN, COMPRESSION_FACTOR, ENCODER_LAYERS, // <--- UPDATED: Added ENCODER_LAYERS
        (const elem_t *)encoder_input,
        NULL, // Modo Encoder
        (elem_t *)encoder_output,
        (const elem_t *)Wq, (const elem_t *)Wk, (const elem_t *)Wv, (const elem_t *)Wo,
        (const elem_t *)Wq_cross, (const elem_t *)Wk_cross, (const elem_t *)Wv_cross, (const elem_t *)Wo_cross,
        (const acc_t *)Wq_b, (const acc_t *)Wk_b, (const acc_t *)Wv_b, (const acc_t *)Wo_b,
        (const acc_t *)Wq_cross_b, (const acc_t *)Wk_cross_b, (const acc_t *)Wv_cross_b, (const acc_t *)Wo_cross_b,
        (const elem_t *)ff1_w, (const elem_t *)ff2_w,
        (const acc_t *)ff1_b, (const acc_t *)ff2_b,
        (elem_t *)Q_buf, (elem_t *)K_buf, (elem_t *)V_buf,
        (elem_t *)attn_buf, 
        (elem_t *)out_buf_ffn, // Passa o buffer maior
        (acc_t *)out_buf_acc_ffn,
        (elem_t *)resadd1_buf, 
        (elem_t *)resadd2_buf
    );

    // === PASSO 3: CLASSIFICATION HEAD ===
    printf("Step 3: Classification Head...\n");
    uint64_t start = read_cycles();

    // 3a. Global Average Pooling (Simulado com MatMul)
    //     pool_out[1, 64] = pool_vector[1, 16] @ encoder_output[16, 64]
    //     Aplicamos a escala (1.0 / 16) para fazer a média.
    
    // O fator de escala é 1.0 / SEQ_LEN
    acc_scale_t pool_scale = (acc_scale_t)(1.0f / SEQ_LEN);

    tiled_matmul_auto(
        1,          // dim_I (Linhas de A)
        HIDDEN_DIM, // dim_J (Colunas de B)
        SEQ_LEN,    // dim_K (Colunas de A / Linhas de B)
        
        (const elem_t *)pool_vector,
        (const elem_t *)encoder_output,
        NULL, // D (sem bias)
        (elem_t *)pool_out,

        SEQ_LEN,     // stride_A (dim_K)
        HIDDEN_DIM,  // stride_B (dim_J)
        0,           // stride_D
        HIDDEN_DIM,  // stride_C (dim_J)

        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 
        pool_scale, // <-- A MÁGICA: aplica 1.0/SEQ_LEN ao resultado
        0,
        false, // repeating_bias
        false, false,
        false, false,
        0,
        tiled_matmul_type);
    
    gemmini_fence(); // (Adicionado por segurança)

    // 3b. LayerNorm (Agora recebe a entrada correta [1, 64])
    tiled_norm_auto(1, HIDDEN_DIM,
        (acc_t*)pool_out, (elem_t*)ln_out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, tiled_matmul_type);
        
    // 3c. Matmul Final (Head: ln_out @ head_w + head_b)
    tiled_matmul_auto(1, OUTPUT_SIZE, HIDDEN_DIM,
        ln_out, head_w, head_b, final_logits,
        HIDDEN_DIM, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, false, false, 0, tiled_matmul_type);
        
    uint64_t end = read_cycles();

    printf("Simulation took %d Cycles\n", end - start);
    
        // --- FIM DA INFERÊNCIA ---
    printf("--- Inference Finished --- \n");

    // --- 5. Verificação do Resultado ---
    int prediction = find_max_index((elem_t*)final_logits, OUTPUT_SIZE);
    
    printf("\n--- Verification --- \n");
    printf("Prediction: %d\n", prediction);
    printf("Ground Truth: %d\n", ground_truth[0]);



    if (prediction == ground_truth[0]) {
        printf("\n****************************************\n");
        printf("   SUCCESS: Prediction matches ground truth!\n");
        printf("****************************************\n");
    } else {
        printf("\n****************************************\n");
        printf("   FAILURE: Prediction does not match.\n");
        printf("****************************************\n");
    }

    return 0;
}