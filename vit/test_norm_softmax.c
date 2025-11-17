// test_norm_softmax.c
#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "includes/norm_softmax_test_data.h"
#include "includes/test_utils.h"

// Buffer de saída (elem_t)
static elem_t output_buf[SEQ_LEN][HIDDEN_DIM];

int main() {
    gemmini_flush(0);
    printf("=== Teste de Hipotese: tiled_norm_auto com SOFTMAX ===\n");

    // Chamada Experimental:
    // Tentando usar a unidade de normalização para fazer Softmax
    tiled_norm_auto(
        SEQ_LEN,       // Dimensão I (linhas)
        HIDDEN_DIM,    // Dimensão J (colunas)
        
        (acc_t*)input_acc_mat, // Entrada (Scores)
        (elem_t*)output_buf,   // Saída (Probabilidades)
        
        ACC_SCALE_IDENTITY,    // Scale (C_scale) - tente 1.0 primeiro
        SOFTMAX,               // <--- A GRANDE APOSTA
        WS
    );

    gemmini_fence();

    // Verificar se bate com o PyTorch
    bool pass = check_matrix("Norm->Softmax Output", 
                             (float*)output_buf, 
                             (float*)expected_output_mat, 
                             SEQ_LEN, HIDDEN_DIM, 
                             0.02f); // Tolerância relaxada

    if (pass) {
        printf(ANSI_COLOR_GREEN "SUCESSO: tiled_norm_auto suporta SOFTMAX!\n" ANSI_COLOR_RESET);
    } else {
        printf(ANSI_COLOR_RED "FALHA: tiled_norm_auto nao se comporta como Softmax.\n" ANSI_COLOR_RESET);
        printf("Nota: O hardware pode estar calculando (x-mean)/std em vez de exp(x)/sum.\n");
    }

    return pass ? 0 : 1;
}