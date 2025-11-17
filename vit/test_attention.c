// test_attention.c
#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// 1. Nossos utilitários e dados
#include "includes/attn_test_data.h"
#include "includes/test_utils.h"

// 2. A implementação C que queremos testar
#include "transformer_layers.c" 

// 3. Buffers temporários que a função 'attention' precisa
static elem_t Q_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t K_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t V_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t attn_buf[NUM_HEADS][SEQ_LEN][SEQ_LEN];
static elem_t out_buf[SEQ_LEN][HIDDEN_DIM];
static acc_t out_buf_acc[SEQ_LEN][HIDDEN_DIM];

// Buffers de saída
static elem_t out_intermediate[SEQ_LEN][HIDDEN_DIM]; // Saída pós-LN
static elem_t resadd_out_final[SEQ_LEN][HIDDEN_DIM]; // Saída pós-Residual Add

int main() {
    gemmini_flush(0);
    printf("=== Teste de Unidade: Funcao C 'attention' ===\n");

    // 4. Chamar a função C
    attention(
        HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, SEQ_LEN,
        COMPRESSION_FACTOR,

        (const elem_t *)input_mat,
        (const elem_t *)input_mat, // self-attention
        
        (elem_t *)out_intermediate,
        (elem_t *)resadd_out_final, // O resultado final que queremos
        
        (const elem_t *)Wq, (const elem_t *)Wk, (const elem_t *)Wv, (const elem_t *)Wo,
        (const acc_t *)Wq_b, (const acc_t *)Wk_b, (const acc_t *)Wv_b, (const acc_t *)Wo_b,

        (elem_t *)Q_buf, (elem_t *)K_buf, (elem_t *)V_buf,
        (elem_t *)attn_buf, (elem_t *)out_buf, (acc_t *)out_buf_acc
    );

    // 5. Verificar o resultado
    // O Python Module retorna a saída pós-residual, então comparamos 'resadd_out_final'
    bool pass = check_matrix("Attention Module", 
                             (float*)resadd_out_final, 
                             (float*)expected_output_mat, 
                             SEQ_LEN, HIDDEN_DIM, 
                             2.0f); // Tolerância de 1e-3
    

    return pass ? 0 : 1;
}