#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "includes/quantized_classifier_head_params.h"

// --- CPU LayerNorm Helper ---
void cpu_layernorm_cls(elem_t * cls_token, int dim) {
    float sum=0, sq=0;
    for(int j=0; j<dim; j++) {
        float v = cls_token[j];
        sum += v; sq += v*v;
    }
    float mean = sum/dim;
    float std = sqrtf(sq/dim - mean*mean + 1e-5);
    for(int j=0; j<dim; j++) {
        float v = cls_token[j];
        float n = (v - mean)/std * 20.0f; // Scale 20
        int res = (int)(n + (n>0?0.5:-0.5));
        if(res > 127) res=127; if(res<-128) res=-128;
        cls_token[j] = (elem_t)res;
    }
}

// --- DUT: Classifier Head ---
void classifier_head(
    int seq_len, int hidden_dim, int num_classes,
    const elem_t * input_seq,
    elem_t * output_scores,
    const elem_t * w, const acc_t * b,
    float scale
) {
    // 1. Extract CLS (Row 0)
    // We assume input_seq is row-major [Seq, Hidden]
    // So Row 0 is just the first 'hidden_dim' bytes.
    // We copy it to a temp buffer to normalize it safely.
    static elem_t cls_buf[128]; // Max Hidden Dim assumption
    memcpy(cls_buf, input_seq, hidden_dim * sizeof(elem_t));

    // 2. LayerNorm (CPU)
    cpu_layernorm_cls(cls_buf, hidden_dim);
    
    // 3. Linear Projection
    // Input: [1, Hidden]. Weights: [Hidden, Classes]. Output: [1, Classes]
    tiled_matmul_nn_auto(1, num_classes, hidden_dim,
        cls_buf, w, b, output_scores,
        NO_ACTIVATION, (acc_scale_t)scale, 0, WS, false, "head");
        
    gemmini_fence();
}

int main() {
    gemmini_flush(0);
    printf("=== Classifier Head Test ===\n");

    static elem_t output_scores[NUM_CLASSES] row_align(1);

    uint64_t start = read_cycles();

    classifier_head(SEQ_LEN, HIDDEN_DIM, NUM_CLASSES,
        (elem_t*)inp, output_scores,
        (elem_t*)w_head, (acc_t*)b_head,
        SCALE_HEAD
    );

    uint64_t end = read_cycles();
    printf("Cycles: %llu\n", end - start);

    // Verify
    int errors = 0;
    int tolerance = 1;

    for (int i = 0; i < NUM_CLASSES; i++) {
        elem_t prod = output_scores[i];
        elem_t exp = ((elem_t*)expected_output)[i];
        
        if (abs(prod - exp) > tolerance) {
            if (errors < 5) printf("Err Class %d: Got %d Exp %d\n", i, prod, exp);
            errors++;
        }
    }

    if (errors == 0) printf("SUCCESS\n");
    else printf("FAIL: %d errors\n", errors);

    return 0;
}