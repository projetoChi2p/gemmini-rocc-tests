// test_qk.c
#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// 1. Nossos utilitários e dados
#include "includes/qk_test_data.h"
#include "includes/test_utils.h"

// 2. Buffer de Saída (para onde o C irá escrever)
// (NUM_HEADS, SEQ_LEN, SEQ_LEN) -> (4, 16, 16)
static elem_t attn_buf[NUM_HEADS][SEQ_LEN][SEQ_LEN];

int main() {
    gemmini_flush(0);
    printf("=== Teste de Unidade: Bloco QK_T (softmax(Q * K.T)) ===\n");

    // --- 3. CÓDIGO SENDO TESTADO ---
    // (Copiado de transformer_layers.c)
    for (int head = 0; head < NUM_HEADS; head++) {
        // O .h já achatou o batch, então usamos Q_buf[0]
        const elem_t * A = (const elem_t *)Q_buf[0] + head * HIDDEN_DIM_PER_HEAD;
        const elem_t * B = (const elem_t *)K_buf[0] + head * HIDDEN_DIM_PER_HEAD;
        
        // C aponta para o buffer de saída
        elem_t * C = (elem_t *)attn_buf + head * SEQ_LEN * SEQ_LEN;

        tiled_matmul_auto(SEQ_LEN, SEQ_LEN, HIDDEN_DIM_PER_HEAD,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/HIDDEN_DIM, /*stride_B=*/HIDDEN_DIM, 
            /*stride_D=*/0, /*stride_C=*/SEQ_LEN,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ true, // Q * K.T
            false, false,
            0,
            WS);
    }
    gemmini_fence();
    // --- FIM DO CÓDIGO SENDO TESTADO ---

    // 4. Verificar os resultados
    // Comparamos toda a memória (4*16*16) de uma vez
    bool pass = check_matrix("attn_buf (Q*K.T)", 
                             (float*)attn_buf, 
                             (float*)expected_attn_buf, 
                             NUM_HEADS * SEQ_LEN, SEQ_LEN, // (4*16) linhas, (16) colunas
                             0.001f); // Tolerância

    return pass ? 0 : 1;
}