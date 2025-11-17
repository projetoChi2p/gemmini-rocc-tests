// test_qk_matmul.c
#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// 1. Nossos utilitários e dados
#include "includes/qk_matmul_test_data.h" // <-- Novo header
#include "includes/test_utils.h"

// 2. Buffer de Saída
static elem_t attn_buf[NUM_HEADS][SEQ_LEN][SEQ_LEN];

int main() {
    gemmini_flush(0);
    printf("=== Teste de Unidade: Bloco QK_T (APENAS MatMul) ===\n");

    // --- 3. CÓDIGO SENDO TESTADO ---
    for (int head = 0; head < NUM_HEADS; head++) {
        const elem_t * A = (const elem_t *)Q_buf[0] + head * HIDDEN_DIM_PER_HEAD;
        const elem_t * B = (const elem_t *)K_buf[0] + head * HIDDEN_DIM_PER_HEAD;
        elem_t * C = (elem_t *)attn_buf + head * SEQ_LEN * SEQ_LEN;

        tiled_matmul_auto(SEQ_LEN, SEQ_LEN, HIDDEN_DIM_PER_HEAD,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/HIDDEN_DIM, /*stride_B=*/HIDDEN_DIM, 
            /*stride_D=*/0, /*stride_C=*/SEQ_LEN,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            
            NO_ACTIVATION, // <-- MUDANÇA PRINCIPAL: DE SOFTMAX PARA NO_ACTIVATION
            
            /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ true, // Q * K.T
            false, false,
            0,
            WS);
    }
    gemmini_fence();
    // --- FIM DO CÓDIGO SENDO TESTADO ---

    // 4. Verificar os resultados
    bool pass = check_matrix("attn_buf (Q*K.T)", 
                             (float*)attn_buf, 
                             (float*)expected_attn_scores_buf, // <-- Novo gabarito
                             NUM_HEADS * SEQ_LEN, SEQ_LEN, 
                             0.001f); // Tolerância

    if (pass) {
        printf(ANSI_COLOR_GREEN "SUCESSO: O MatMul (Q*K.T) esta correto!" ANSI_COLOR_RESET "\n");
    } else {
        printf(ANSI_COLOR_RED "FALHA: O MatMul (Q*K.T) falhou." ANSI_COLOR_RESET "\n");
    }

    return pass ? 0 : 1;
}