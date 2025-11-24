#include <stdio.h>
#include <math.h>
#include <assert.h>

// Assuming exp_approx is declared in transformer_layers.h
#include "transformer_layers.c"
#include "includes/exp_data.h"

#define TOLERANCE 0.01
extern float golden[TEST_SIZE];

float fp32abs(float x){
    if (x < 0) return -x; else return x;
}

void test_exp_approx_basic() {
    float input[TEST_SIZE];
    float output[TEST_SIZE];
    int errors = 0;
    
    // Test small positive values
    for (int i = 0; i < TEST_SIZE; i++) {
        input[i] = (float)i * 0.1f;
    }

    for (int i = 0; i < TEST_SIZE; i++) {
        output[i] = exp_approx(input[i]);
        if (fp32abs(output[i] - golden[i]) > TOLERANCE){
            #ifdef ELEM_T_IS_FLOAT
            printf("Mismatch! index %d: output=%de-6 golden=%de-6 (diff=%de-6)\n", i, 
                elem_t_to_floats(output[i]*1000000), 
                elem_t_to_floats(golden[i]*1000000), 
                elem_t_to_floats(fp32abs(output[i]-golden[i])*1000000)
            );
            //printf("");
            #endif
            errors += 1;
        }
    }
    if (errors == 0)
        printf("test_exp_approx_basic PASSED\n");
    else
        printf("test_exp_approx_basic FAILED (errors: %d)\n", errors);
}
/*
void test_exp_approx_negative() {
    float input[TEST_SIZE];
    float output[TEST_SIZE];
    
    for (int i = 0; i < TEST_SIZE; i++) {
        input[i] = (float)(-i) * 0.1f;
    }
    
    exp_approx(input, output, TEST_SIZE);
    
    for (int i = 0; i < TEST_SIZE; i++) {
        float expected = expf(input[i]);
        assert(fabsf(output[i] - expected) / expected < TOLERANCE);
    }
    printf("test_exp_approx_negative PASSED\n");
}*/

int main() {
    test_exp_approx_basic();
    //test_exp_approx_negative();
    //printf("All tests PASSED\n");
    return 0;
}