// test_ffn.c
#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// 1. Nossos utilitários e dados
#include "includes/ffn_test_data.h"
#include "includes/test_utils.h"

// 2. A implementação C que queremos testar
#include "transformer_layers.c" 

// 3. Buffers temporários que a função 'ffn' precisa
static elem_t out_buf[SEQ_LEN][EXPANSION_DIM];
static acc_t out_buf_acc[SEQ_LEN][HIDDEN_DIM];

// Buffer de saída
// Nota: A função 'ffn' do C escreve o resultado final em 'out'
static elem_t ffn_out_final[SEQ_LEN][HIDDEN_DIM]; 

int main() {
    gemmini_flush(0);
    printf("=== Teste de Unidade: Funcao C 'ffn' ===\n");

    // 4. Chamar a função C
    ffn(
        HIDDEN_DIM, EXPANSION_DIM, SEQ_LEN,
        
        (const elem_t *)input_mat,
        (elem_t *)ffn_out_final, // O resultado final que queremos
        
        (const elem_t *)ff1_w, (const elem_t *)ff2_w,
        (const acc_t *)ff1_b, (const acc_t *)ff2_b,

        (elem_t *)out_buf, (acc_t *)out_buf_acc
    );

    // 5. Verificar o resultado
    bool pass = check_matrix("FFN Module", 
                             (float*)ffn_out_final, 
                             (float*)expected_output_mat, 
                             SEQ_LEN, HIDDEN_DIM, 
                             10.0f);

    return pass ? 0 : 1;
}