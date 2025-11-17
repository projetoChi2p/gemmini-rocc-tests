// test_layernorm.c
#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "includes/ln_test_data.h"
#include "includes/test_utils.h"

// Buffer de saída (elem_t)
static elem_t ln_out_buf[SEQ_LEN][HIDDEN_DIM];

int main() {
    gemmini_flush(0);
    printf("=== Teste de Unidade: Layer Normalization ===\n");

    // Chamada exata do seu código: (acc_t*)input -> (elem_t*)output
    tiled_norm_auto(
        SEQ_LEN, 
        HIDDEN_DIM,
        (acc_t*)input_acc_mat, // Entrada (simulando out_buf_acc)
        (elem_t*)ln_out_buf,   // Saída
        ACC_SCALE_IDENTITY,
        LAYERNORM, 
        WS
    );

    gemmini_fence();

    bool pass = check_matrix("LayerNorm Output", 
                             (float*)ln_out_buf, 
                             (float*)expected_output_mat, 
                             SEQ_LEN, HIDDEN_DIM, 
                             0.01f); // Tolerância

    return pass ? 0 : 1;
}