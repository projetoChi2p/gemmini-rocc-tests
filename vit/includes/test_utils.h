// test_utils.h
#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <math.h>
#include <stdio.h>
#include <stdbool.h>

// Cores para o terminal
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

// Verifica se duas matrizes são iguais com uma tolerância
// Assume que as matrizes são armazenadas contiguamente na memória (row-major)
static bool check_matrix(const char* layer_name, 
                         const float* actual, 
                         const float* expected, 
                         int rows, int cols, 
                         float tolerance) {
    
    printf("Testing layer: %s [%dx%d]... ", layer_name, rows, cols);
    
    int errors = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < rows * cols; i++) {
        float diff = fabsf(actual[i] - expected[i]);
        if (diff > max_diff) max_diff = diff;

        if (diff > tolerance) {
            if (errors < 5) { // Imprimir apenas os primeiros 5 erros
                if (errors == 0) printf(ANSI_COLOR_RED "FAIL\n" ANSI_COLOR_RESET);
                printf("  Mismatch at index %d (row %d, col %d): C=%de-3 vs Py=%de-3 (diff=%de-3)\n", 
                       i, i / cols, i % cols, elem_t_to_floats(actual[i]*1000), elem_t_to_floats(expected[i]*1000), elem_t_to_floats(diff*1000));
            }
            errors++;
        }else {
            printf("  Correct at index %d (row %d, col %d): C=%de-3 vs Py=%de-3 (diff=%de-3)\n", 
                i, i / cols, i % cols, elem_t_to_floats(actual[i]*1000), elem_t_to_floats(expected[i]*1000), elem_t_to_floats(diff*1000));

        }
    }

    if (errors == 0) {
        printf(ANSI_COLOR_GREEN "PASS" ANSI_COLOR_RESET " (Max diff: %de-3)\n", elem_t_to_floats(1000*max_diff));
        return true;
    } else {
        printf(ANSI_COLOR_RED "  Total Errors: %d" ANSI_COLOR_RESET "\n", errors);
        return false;
    }
}

#endif // TEST_UTILS_H