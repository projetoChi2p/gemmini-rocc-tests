// test_resadd.c
#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "includes/resadd_test_data.h"
#include "includes/test_utils.h"

static elem_t resadd_out_buf[SEQ_LEN][HIDDEN_DIM];

int main() {
    gemmini_flush(0);
    printf("=== Teste de Unidade: Residual Addition ===\n");

    tiled_resadd_auto(
        SEQ_LEN, 
        HIDDEN_DIM,
        MVIN_SCALE_IDENTITY, // Scale A
        MVIN_SCALE_IDENTITY, // Scale B
        ACC_SCALE_IDENTITY,  // Scale C (Output)
        (elem_t*)input_A,
        (elem_t*)input_B,
        (elem_t*)resadd_out_buf,
        false, // ReLU = false
        WS
    );

    gemmini_fence();

    bool pass = check_matrix("ResAdd Output", 
                             (float*)resadd_out_buf, 
                             (float*)expected_sum, 
                             SEQ_LEN, HIDDEN_DIM, 
                             0.001f);

    return pass ? 0 : 1;
}