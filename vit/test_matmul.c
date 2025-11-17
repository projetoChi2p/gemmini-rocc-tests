// test_matmul.c
#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h" // Para tiled_matmul_auto

// 1. Nossos utilitários e dados
#include "includes/matmul_test_data.h"
#include "includes/test_utils.h" // Para check_matrix

// 2. Buffer de Saída
//    (O .h do Python já removeu a dimensão do batch)
static elem_t output_mat[SEQ_LEN][HIDDEN_DIM];

int main() {
    gemmini_flush(0);
    printf("=== Teste de Unidade: Simples tiled_matmul_auto (QKV) ===\n");

    // 3. Chamar a função C (simulando a primeira chamada em 'attention')
    //    tiled_matmul_auto(M, N, K, A, B, D, C, ...)
    tiled_matmul_auto(
        SEQ_LEN,     // M
        HIDDEN_DIM,  // N (hidden_dim_compressed == hidden_dim)
        HIDDEN_DIM,  // K
        
        (const elem_t *)input_mat,
        (const elem_t *)Wq,
        (const acc_t *)Wq_b,
        (elem_t *)output_mat,

        HIDDEN_DIM,  // stride_A (K)
        HIDDEN_DIM,  // stride_B (N)
        HIDDEN_DIM,  // stride_D (N) <-- CORRIGIDO
        HIDDEN_DIM,  // stride_C (N)

        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        
        true,        // repeating_bias <-- CORRIGIDO
        
        false,       // transpose_B
        false,       // transpose_B
        false, false,
        0,
        WS
    );

    gemmini_fence();

    // 4. Verificar o resultado
    bool pass = check_matrix("MatMul (Q = Wq*Input + Bq)", 
                             (float*)output_mat, 
                             (float*)expected_output_mat, 
                             SEQ_LEN, HIDDEN_DIM, 
                             0.001f);

    if (pass) {
        printf("Sucesso! O bug do bias foi corrigido.\n");
    } else {
        printf("Falha! Verifique os strides e parâmetros.\n");
    }

    return pass ? 0 : 1;
}