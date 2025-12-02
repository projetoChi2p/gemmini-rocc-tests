#ifndef FFN_H
#define FFN_H

#include "include/gemmini_nn.h"

void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,
        elem_t * out_buf, acc_t * out_buf_acc
        
        #ifdef DEBUG
        // Debug Pointers
        ,const elem_t * expected_fc1,
        const elem_t * expected_fc2
        #endif
    );

#endif