// test_minivit_classifier.c
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// Inclui os pesos, o lote de imagens e os labels
#include "includes/classifier_params_large.h" 
#include "transformer_layers.c"

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
// (Estes são REUTILIZADOS em cada iteração do loop)
static elem_t embed_out[SEQ_LEN][HIDDEN_DIM];
static elem_t encoder_input[SEQ_LEN][HIDDEN_DIM];
static elem_t encoder_output[SEQ_LEN][HIDDEN_DIM];
static elem_t Q_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t K_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t V_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t attn_buf[NUM_HEADS][SEQ_LEN][SEQ_LEN];
static elem_t out_buf_attn[SEQ_LEN][HIDDEN_DIM];
static acc_t out_buf_acc_attn[SEQ_LEN][HIDDEN_DIM];
static elem_t resadd1_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t resadd2_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t out_buf_ffn[SEQ_LEN][EXPANSION_DIM];
static acc_t out_buf_acc_ffn[SEQ_LEN][HIDDEN_DIM];
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
    enum tiled_matmul_type_t tiled_matmul_type = WS;
    printf("--- Starting MiniViT Classifier Batch Inference --- \n");
    printf("Running %d inferences...\n", NUM_INFERENCES);

    int correct_predictions = 0;
    uint64_t total_cycles = 0;

    // === INÍCIO DO LOOP DE INFERÊNCIA ===
    for (int i = 0; i < NUM_INFERENCES; i++) {
        
        // Seleciona a imagem e o label para esta iteração
        const elem_t * current_patches = (const elem_t *)all_input_patches[i];
        int current_ground_truth = all_ground_truths[i];

        uint64_t start_cycles = read_cycles();

        // === PASSO 1: PATCH + POS EMBEDDING ===
        tiled_matmul_auto(SEQ_LEN, HIDDEN_DIM, PATCH_DIM,
            current_patches, patch_embed_w, patch_embed_b, embed_out,
            PATCH_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
            true, false, false, false, false, 0, tiled_matmul_type);

        tiled_resadd_auto(SEQ_LEN, HIDDEN_DIM,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
            embed_out, pos_embed_data, encoder_input,
            false, tiled_matmul_type);
        
        // === PASSO 2: ENCODER ===
        encoder_decoder(
            HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, CROSS_NUM_HEADS,
            SEQ_LEN, COMPRESSION_FACTOR,
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

        // === PASSO 3: CLASSIFICATION HEAD ===
        acc_scale_t pool_scale = (acc_scale_t)(1.0f / SEQ_LEN);
        tiled_matmul_auto(1, HIDDEN_DIM, SEQ_LEN,
            (const elem_t *)pool_vector, (const elem_t *)encoder_output,
            NULL, (elem_t *)pool_out,
            SEQ_LEN, HIDDEN_DIM, 0, HIDDEN_DIM,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, pool_scale, 0,
            false, false, false, false, false, 0, tiled_matmul_type);
        
        tiled_norm_auto(1, HIDDEN_DIM,
            (acc_t*)pool_out, (elem_t*)ln_out,
            ACC_SCALE_IDENTITY,
            LAYERNORM, tiled_matmul_type);
            
        tiled_matmul_auto(1, OUTPUT_SIZE, HIDDEN_DIM,
            ln_out, head_w, head_b, final_logits,
            HIDDEN_DIM, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
            true, false, false, false, false, 0, tiled_matmul_type);
        
        uint64_t end_cycles = read_cycles();
        total_cycles += (end_cycles - start_cycles);

        // --- Verificação da Iteração ---
        int prediction = find_max_index((elem_t*)final_logits, OUTPUT_SIZE);
        if (prediction == current_ground_truth) {
            correct_predictions++;
        }

        if ((i + 1) % 100 == 0) {
            printf("Processado %d / %d amostras...\n", i + 1, NUM_INFERENCES);
        }

    } // === FIM DO LOOP DE INFERÊNCIA ===

    printf("--- Batch Inference Finished --- \n");
    printf("");
    
    // --- Relatório Final ---
    double accuracy = (double)correct_predictions / NUM_INFERENCES * 100.0;
    double avg_cycles = (double)total_cycles / NUM_INFERENCES;

    printf("\n--- Resultados Finais --- \n");
    printf("Total de Amostras: %d\n", NUM_INFERENCES);
    printf("Corretas:        %d\n", correct_predictions);
    printf("Acuracia:        %de-2%%\n", elem_t_to_floats(accuracy*100));
    printf("Ciclos Totais:   %llu\n", total_cycles);
    printf("Ciclos/Inferencia: %d\n", elem_t_to_floats(avg_cycles));
    printf("");

    return 0;
}