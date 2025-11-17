// transformer_layers.c
#include "includes/transformer_layers.h" 
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_nn.h"
#include <math.h>
#include <float.h> 

// ==========================================
// FERRAMENTAS DE DIAGNÓSTICO (NOVO)
// ==========================================

// Função genérica para imprimir estatísticas de qualquer buffer (Q, K, V, Output)
void analyze_tensor(const char* label, const elem_t* data, int size) {
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    double sum_val = 0.0; // Double para evitar overflow na soma

    for (int i = 0; i < size; i++) {
        float val = (float)data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_val += val;
    }

    float mean_val = (float)(sum_val / size);

    printf("  [DEBUG] %-15s -> Range: [%de-3, %de-3] | Mean: %de-3\n", 
           label, 
           (int)(min_val * 1000), 
           (int)(max_val * 1000), 
           (int)(mean_val * 1000));
}

// ==========================================
// FUNÇÕES AUXILIARES (EXP / SOFTMAX)
// ==========================================

// ... (Sua função exp_approx permanece igual, omiti para economizar espaço) ...
// Se precisar, cole a versão anterior da exp_approx e exp_high_prec aqui.
// Para este exemplo, vou usar a exp_approx simples que você mandou no último snippet.

float exp_approx(float x) {
    if (x < -5.0f) return 0.001f * (x + 10.0f);
    else if (x < -2.0f) { float xp = x + 3.5f; return 0.030197f + 0.028374f * xp + 0.013903f * xp * xp + 0.004534f * xp * xp * xp; }
    else if (x < 0.0f) { float x2 = x*x; float x3 = x2*x; float x4 = x2*x2; return 1.0f + x + 0.5f*x2 + 0.1666667f*x3 + 0.0416664f*x4; }
    else if (x < 2.0f) { float x2 = x*x; float x3 = x2*x; float x4 = x2*x2; return 1.0f + x + 0.5f*x2 + 0.1666667f*x3 + 0.0416664f*x4; }
    else if (x < 5.0f) { float xp = x - 3.5f; return 16.444647f + 16.444647f * xp + 8.222323f * xp * xp + 2.740774f * xp * xp * xp; }
    else return 100.0f + 50.0f * (x - 5.0f);
}

void cpu_softmax(int rows, int cols, const elem_t * input, elem_t * output) {
    for (int i = 0; i < rows; i++) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < cols; j++) {
            float val = (float)input[i * cols + j];
            if (val > max_val) max_val = val;
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = (float)input[i * cols + j] - max_val;
            float exp_val = exp_approx(val);
            sum_exp += exp_val;
        }

        for (int j = 0; j < cols; j++) {
            float val = (float)input[i * cols + j] - max_val;
            float exp_val = exp_approx(val);
            output[i * cols + j] = (elem_t)(exp_val / sum_exp);
        }
    }
}

// ==========================================
// ATTENTION BLOCK
// ==========================================

void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,

        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,

        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b,
        const acc_t * Wo_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc)
{
    int hidden_dim_compressed = hidden_dim / compression_factor;
    int hidden_dim_per_head = hidden_dim_compressed / num_heads;

    if (compression_factor < 0) {
        hidden_dim_compressed = hidden_dim;
        hidden_dim_per_head = (hidden_dim_compressed / 12) * (-compression_factor);
    }
    
    printf("\n--- Starting Attention Block Analysis ---\n");

    // 1. Calculate Q, K, V
    const int qkv_matmuls_n = 3;
    for (int i = 0; i < qkv_matmuls_n; i++) {
        const elem_t * qkv_weights[] = {Wq, Wk, Wv};
        const elem_t * qkv_ins[] = {input, enc_out, enc_out};
        const acc_t * qkv_bs[] = {Wq_b, Wk_b, Wk_b};
        elem_t * qkv_outs[] = {Q_buf, K_buf, V_buf};

        tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
            /*A=*/ qkv_ins[i], /*B=*/ qkv_weights[i],
            /*D=*/ qkv_bs[i], /*C=*/ qkv_outs[i],
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
            false, false, false, false, false, 0, WS);
    }

    gemmini_fence();
    
    // [PROBE 1] Analisar Q, K e V gerados
    // Tamanho = seq_len * hidden_dim_compressed
    int qkv_size = seq_len * hidden_dim_compressed;
    analyze_tensor("Q Matrix", Q_buf, qkv_size);
    analyze_tensor("K Matrix", K_buf, qkv_size);
    analyze_tensor("V Matrix", V_buf, qkv_size);

    // 2. Calculate Raw Attention Scores (attn = Q * K^T)
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = Q_buf + head * hidden_dim_per_head;
        const elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf + head * seq_len * seq_len;

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
            false, false, true, false, false, 0, WS);
    }

    gemmini_fence();

    // --- 3. SCALING, ANALYSIS, AND SOFTMAX ---
    float inv_scale = 1.0f / sqrtf((float)hidden_dim_per_head);
    
    // [PROBE 2] Analisar scores ANTES do scaling (Global)
    // Útil para ver o quão grande os valores saem da Matmul bruta
    analyze_tensor("Attn Raw (Pre)", attn_buf, num_heads * seq_len * seq_len);

    for (int head = 0; head < num_heads; head++) {
        elem_t * C = attn_buf + head * seq_len * seq_len;
        int total_elements = seq_len * seq_len;

        // Apply Scaling
        for (int i = 0; i < total_elements; i++) {
            C[i] = C[i] * inv_scale;
        }
        
        // [PROBE 3] Distribuição detalhada para o Softmax (Sua análise anterior)
        // Esta mantemos detalhada pois é onde ocorre a função exponencial crítica
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        float sum_val = 0.0f;
        int bin_neg_huge = 0, bin_neg_med = 0, bin_center = 0, bin_pos_med = 0, bin_pos_huge = 0;

        for (int i = 0; i < total_elements; i++) {
            float val = (float)C[i];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum_val += val;

            if (val < -5.0f) bin_neg_huge++;
            else if (val < -2.0f) bin_neg_med++;
            else if (val < 2.0f) bin_center++;
            else if (val < 5.0f) bin_pos_med++;
            else bin_pos_huge++;
        }
        
        // Imprime resumo rápido do Head para não poluir muito
        printf("  [DEBUG] Head %d Scaled Input -> Range: [%d, %d] (e-3) | HighPrec: %d/%d\n", 
               head, (int)(min_val*1000), (int)(max_val*1000), bin_center, total_elements);

        cpu_softmax(seq_len, seq_len, C, C);
    }
    
    // [PROBE 4] Verificar se o Softmax produziu probabilidades válidas (0 a 1)
    analyze_tensor("Softmax Output", attn_buf, num_heads * seq_len * seq_len);

    // 4. Calculate Final Attention Output (out = attn * V)
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = attn_buf + head * seq_len * seq_len;
        const elem_t * B = V_buf + head * hidden_dim_per_head;
        elem_t * C = out_buf + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/seq_len, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
            false, false, false, false, false, 0, WS);
    }

    gemmini_fence();
    
    // [PROBE 5] Verificar o vetor de contexto (resultado da atenção aplicada a V)
    analyze_tensor("Context Vector", out_buf, seq_len * hidden_dim_compressed);

    // 5. Projection (out_buf_acc = out_buf * Wo)
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        /*A=*/ out_buf, /*B=*/ Wo,
        /*D=*/ Wo_b, /*C=*/ out_buf_acc,
        /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false, true, false, 0, WS);

    gemmini_fence();
    
    // Nota: Não analisamos out_buf_acc aqui pois é do tipo acc_t. 
    // Analisaremos após o LayerNorm que converte de volta para elem_t.

    // 6. Layer Norm
    tiled_norm_auto(seq_len, hidden_dim,
        (acc_t*)out_buf_acc, (elem_t*)out,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);
        
    // [PROBE 6] Saída final do bloco (após LayerNorm)
    analyze_tensor("Block Output (LN)", out, seq_len * hidden_dim);

    // 7. Residual Add
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        input, out, resadd_out,
        false, WS);

    gemmini_fence();
    printf("--- End Attention Block Analysis ---\n\n");
}

// A função ffn e encoder_decoder permanecem iguais ao anterior
// Apenas certifique-se de manter as declarações no final do arquivo
void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,
        elem_t * out_buf, acc_t * out_buf_acc) {
    // ... (mesmo código anterior) ...
    // Se quiser adicionar probes aqui também:
    // analyze_tensor("FFN Input", input, ...);
    // ...
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        input, ff1_w, ff1_b, out_buf,
        hidden_dim, expansion_dim, expansion_dim, expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        true, false, false, false, false, 0, WS);
    gemmini_fence();
    
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
        out_buf, ff2_w, ff2_b, out_buf_acc,
        expansion_dim, hidden_dim, expansion_dim, expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, true, false, 0, WS);
    gemmini_fence();

    tiled_norm_auto(seq_len, hidden_dim, (acc_t*)out_buf_acc, (elem_t*)out, ACC_SCALE_IDENTITY, LAYERNORM, WS);
    gemmini_fence();

    tiled_resadd_auto(seq_len, hidden_dim, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, out, input, out, false, WS);
    gemmini_fence();
}

uint64_t encoder_decoder(
        int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads,
        int seq_len, int compression_factor,
        const elem_t * input, const elem_t * enc_out, elem_t * out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,
        const acc_t * Wq_b, const acc_t * Wk_b, const acc_t * Wv_b, const acc_t * Wo_b,
        const acc_t * Wq_cross_b, const acc_t * Wk_cross_b, const acc_t * Wv_cross_b, const acc_t * Wo_cross_b,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,
        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf, acc_t * out_buf_acc,
        elem_t * resadd1_buf, elem_t * resadd2_buf)
{
    const bool is_encoder = enc_out == NULL;
    uint64_t start = read_cycles();

    attention(hidden_dim, expansion_dim, num_heads, seq_len, compression_factor,
        input, input, out, resadd1_buf,
        Wq, Wk, Wv, Wo, Wq_b, Wk_b, Wv_b, Wo_b,
        Q_buf, K_buf, V_buf, attn_buf, out_buf, out_buf_acc);

    if (!is_encoder) {
        attention(hidden_dim, expansion_dim, cross_num_heads, seq_len, compression_factor,
            resadd1_buf, enc_out, out, resadd2_buf,
            Wq_cross, Wk_cross, Wv_cross, Wo_cross,
            Wq_cross_b, Wk_cross_b, Wv_cross_b, Wo_cross_b,
            Q_buf, K_buf, V_buf, attn_buf, out_buf, out_buf_acc);
    }

    ffn(hidden_dim, expansion_dim, seq_len,
        is_encoder ? resadd1_buf : resadd2_buf, out,
        ff1_w, ff2_w, ff1_b, ff2_b, out_buf, out_buf_acc);

    uint64_t end = read_cycles();
    return end - start;
}