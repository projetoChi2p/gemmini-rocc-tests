// test_context.c
#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "includes/context_test_data.h"
#include "includes/test_utils.h"

static elem_t out_buf[NUM_HEADS][SEQ_LEN][D_K];

int main() {
    gemmini_flush(0);
    printf("=== Teste Unitario: Context (Probs * V) ===\n");

    for (int head = 0; head < NUM_HEADS; head++) {
        // Ponteiros para a cabeça atual
        const elem_t* A = (const elem_t*)probs_mat + head * SEQ_LEN * SEQ_LEN;
        const elem_t* B = (const elem_t*)v_mat + head * SEQ_LEN * D_K;
        elem_t* C = (elem_t*)out_buf + head * SEQ_LEN * D_K;

        // A (SEQ_LEN x SEQ_LEN) * B (SEQ_LEN x D_K) = C (SEQ_LEN x D_K)
        tiled_matmul_auto(
            SEQ_LEN, D_K, SEQ_LEN,
            A, B, NULL, C,
            SEQ_LEN, HIDDEN_DIM, 0, HIDDEN_DIM, // Strides
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
            false, // bias
            false, false, // transpose
            false, false, 0, WS
        );
    }
    gemmini_fence();

    // Verificar
    // O Python output é (1, H, S, Dk). O C output é igual mas achatado.
    bool pass = check_matrix("Context Output", 
                 (float*)out_buf, 
                 (float*)expected_context, 
                 NUM_HEADS * SEQ_LEN, D_K, 
                 0.001f);

    return pass ? 0 : 1;
}