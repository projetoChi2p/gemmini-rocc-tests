#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "include/gemmini_nn.h"

void compute_classification_head(
    int seq_len, int hidden_dim, int output_size,
    const elem_t * encoder_output,
    const elem_t * pool_vector,
    const elem_t * head_w,
    const acc_t * head_b,
    elem_t * pool_out_buf,   // Temp buffer for pooling result
    elem_t * ln_out_buf,     // Temp buffer for norm result
    elem_t * final_logits    // Final output
);

#endif