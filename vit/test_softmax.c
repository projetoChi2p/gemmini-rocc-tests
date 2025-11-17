// test_softmax.c
#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// 1. Nossos utilitários e dados
#include "includes/softmax_test_data.h"
#include "includes/test_utils.h"

#define SCORE_DIM SEQ_LEN
// 2. Buffer de Saída
// (O .h exporta (1, 1, 16), mas o C tratará como (1, 16))
static elem_t probs_buf[1][SCORE_DIM]; 

int main() {
    gemmini_flush(0);
    printf("=== Teste de Unidade: Bloco SOFTMAX (isolado) ===\n");

    // --- 3. CÓDIGO SENDO TESTADO ---
    // Executamos (Scores * Identity) com ativação SOFTMAX
    // I=1, J=SCORE_DIM, K=SCORE_DIM
    tiled_matmul_auto(
        1,         // dim_I
        SCORE_DIM, // dim_J
        SCORE_DIM, // dim_K
        
        (const elem_t *)input_scores_mat,
        (const elem_t *)identity_mat,
        NULL, // D (sem bias)
        (elem_t *)probs_buf,

        SCORE_DIM, // stride_A (K)
        SCORE_DIM, // stride_B (J)
        0,         // stride_D
        SCORE_DIM, // stride_C (J)

        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        
        SOFTMAX, // <-- A ÚNICA COISA QUE ESTAMOS TESTANDO
        
        ACC_SCALE_IDENTITY, 0,
        false, // repeating_bias
        false, false,
        false, false,
        0,
        WS);
    
    gemmini_fence();
    // --- FIM DO CÓDIGO SENDO TESTADO ---

    // 4. Verificar os resultados
    bool pass = check_matrix("Softmax(Scores)", 
                             (float*)probs_buf, 
                             (float*)expected_probs_mat,
                             1, SCORE_DIM, // (1) linha, (16) colunas
                             0.1f);

    if (pass) {
        printf(ANSI_COLOR_GREEN "SUCESSO: A ativacao SOFTMAX funciona!" ANSI_COLOR_RESET "\n");
    } else {
        printf(ANSI_COLOR_RED "FALHA: A ativacao SOFTMAX falhou." ANSI_COLOR_RESET "\n");
    }

    return pass ? 0 : 1;
}