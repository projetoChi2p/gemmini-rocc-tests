// test_qkv.c
#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// 1. Nossos utilitários e dados
#include "includes/qkv_test_data.h"
#include "includes/test_utils.h"

// 2. Buffers de Saída (para onde o C irá escrever)
static elem_t Q_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t K_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t V_buf[SEQ_LEN][HIDDEN_DIM];

int main() {
    gemmini_flush(0);
    printf("=== Teste de Unidade: Bloco QKV da Atencao ===\n");

    // --- 3. CÓDIGO SENDO TESTADO ---
    // (Copiado de transformer_layers.c, COM OS BUGS)

    int hidden_dim_compressed = HIDDEN_DIM / COMPRESSION_FACTOR;

    // Definição de ponteiros (COM BUG 2)
    const elem_t * qkv_weights[] = {Wq, Wk, Wv};
    const elem_t * qkv_ins[] = { (const elem_t *)input_mat, 
                                (const elem_t *)input_mat, 
                                (const elem_t *)input_mat };
    const acc_t * qkv_bs[] = {Wq_b, Wk_b, Wv_b}; 
    elem_t * qkv_outs[] = { (elem_t *)Q_buf, (elem_t *)K_buf, (elem_t *)V_buf };

    const int qkv_matmuls_n = 3;
    for (int i = 0; i < qkv_matmuls_n; i++) {
        const elem_t * qkv_w = qkv_weights[i];
        const elem_t * qkv_in = qkv_ins[i];
        const acc_t * qkv_b = qkv_bs[i];
        elem_t * qkv_out = qkv_outs[i];

        // Chamada de Matmul (COM BUG 1)
        tiled_matmul_auto(SEQ_LEN, hidden_dim_compressed, HIDDEN_DIM,
            /*A=*/ qkv_in, /*B=*/ qkv_w,
            /*D=*/ qkv_b, /*C=*/ qkv_out,
            /*stride_A=*/HIDDEN_DIM, /*stride_B=*/HIDDEN_DIM, 
            /*stride_D=*/0, /*stride_C=*/HIDDEN_DIM,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false, // <-- Bug do Bias
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);
    }
    gemmini_fence();
    // --- FIM DO CÓDIGO SENDO TESTADO ---

    // 4. Verificar os resultados
    bool pass_q = check_matrix("Q_buf", (float*)Q_buf, (float*)expected_Q_mat, 
                               SEQ_LEN, HIDDEN_DIM, 0.001f);
    bool pass_k = check_matrix("K_buf", (float*)K_buf, (float*)expected_K_mat, 
                               SEQ_LEN, HIDDEN_DIM, 0.001f);
    bool pass_v = check_matrix("V_buf", (float*)V_buf, (float*)expected_V_mat, 
                               SEQ_LEN, HIDDEN_DIM, 0.001f);

    if (pass_q && pass_k && pass_v) {
        printf(ANSI_COLOR_GREEN "SUCESSO: Bloco QKV passou!" ANSI_COLOR_RESET "\n");
        return 0;
    } else {
        printf(ANSI_COLOR_RED "FALHA: Bloco QKV falhou." ANSI_COLOR_RESET "\n");
        return 1;
    }
}