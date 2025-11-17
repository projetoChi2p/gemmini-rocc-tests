// test_encoder.c
#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// 1. Nossos utilitários e dados
#include "includes/encoder_test_data.h" // O header que acabamos de gerar
#include "includes/test_utils.h"

// 2. A DECLARAÇÃO C que queremos testar
#include "transformer_layers_softmax01.c" 

// 3. Buffers temporários que a função 'encoder_decoder' precisa
//    Estes são todos os buffers que 'attention' E 'ffn' usam.
static elem_t Q_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t K_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t V_buf[SEQ_LEN][HIDDEN_DIM];
static elem_t attn_buf[NUM_HEADS][SEQ_LEN][SEQ_LEN];
static elem_t out_buf_attn[SEQ_LEN][HIDDEN_DIM]; // Buffer de saída da Attn
static acc_t out_buf_acc_attn[SEQ_LEN][HIDDEN_DIM];
static elem_t resadd1_buf[SEQ_LEN][HIDDEN_DIM]; // Saída da Attn + Res
static elem_t resadd2_buf[SEQ_LEN][HIDDEN_DIM]; // Não usado no encoder

// Buffers para FFN
static elem_t out_buf_ffn[SEQ_LEN][EXPANSION_DIM];
static acc_t out_buf_acc_ffn[SEQ_LEN][HIDDEN_DIM];

// Buffer de saída final
static elem_t encoder_out_final[SEQ_LEN][HIDDEN_DIM]; 

int main() {
    gemmini_flush(0);
    printf("=== Teste de Integracao: Funcao C 'encoder_decoder' ===\n");

    // 4. Chamar a função C
    // Nota: Passamos NULL para 'enc_out' para sinalizar o modo Encoder
    encoder_decoder(
        HIDDEN_DIM, EXPANSION_DIM, NUM_HEADS, CROSS_NUM_HEADS,
        SEQ_LEN, COMPRESSION_FACTOR,

        (const elem_t *)input_mat,
        NULL, // <-- Modo Encoder
        (elem_t *)encoder_out_final, // O resultado final que queremos
        
        // Pesos da Self-Attention
        (const elem_t *)Wq, (const elem_t *)Wk, (const elem_t *)Wv, (const elem_t *)Wo,
        
        // Pesos da Cross-Attention (Dummies)
        (const elem_t *)Wq_cross, (const elem_t *)Wk_cross, (const elem_t *)Wv_cross, (const elem_t *)Wo_cross,

        // Biases da Self-Attention
        (const acc_t *)Wq_b, (const acc_t *)Wk_b, (const acc_t *)Wv_b, (const acc_t *)Wo_b,
        
        // Biases da Cross-Attention (Dummies)
        (const acc_t *)Wq_cross_b, (const acc_t *)Wk_cross_b, (const acc_t *)Wv_cross_b, (const acc_t *)Wo_cross_b,

        // Pesos do FFN
        (const elem_t *)ff1_w, (const elem_t *)ff2_w,
        (const acc_t *)ff1_b, (const acc_t *)ff2_b,

        // Buffers internos
        (elem_t *)Q_buf, (elem_t *)K_buf, (elem_t *)V_buf,
        (elem_t *)attn_buf, 
        (elem_t *)out_buf_attn, // Reutilizado por ffn
        (acc_t *)out_buf_acc_attn, // Reutilizado por ffn
        (elem_t *)resadd1_buf, 
        (elem_t *)resadd2_buf
    );


    // 5. Verificar o resultado
    // Nota: A tolerância aqui pode precisar ser alta!
    // Você usou 10.0f para FFN e 2.0f para Attention.
    // Somando os erros (2x LayerNorm, 1x IGELU), 15.0f pode ser necessário.
    float tolerance = 2.7f;
    printf("Usando tolerancia: %f\n", tolerance);
    
    bool pass = check_matrix("Encoder Output", 
                             (float*)encoder_out_final, 
                             (float*)expected_output_mat, 
                             SEQ_LEN, HIDDEN_DIM, 
                             tolerance); 

    return pass ? 0 : 1;
}