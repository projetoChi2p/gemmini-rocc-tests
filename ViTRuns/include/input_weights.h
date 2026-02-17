// ==========================================
// DYNAMIC MODEL/DATASET SELECTION
// ==========================================
#ifndef __INPUT_WEIGHTS_H
#define __INPUT_WEIGHTS_H

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define CONCAT_HIDDEN(a, b, c, d) a ## _ ## b ## _ ## d
#define CONCAT(a, b, c, d) CONCAT_HIDDEN(a, b, c, d)

// 1. Determine the suffix
#ifndef ELEM_T_IS_FLOAT
    #define SUFFIX quant_params
#else
    #define SUFFIX params
#endif

// 2. Build the middle part of the filename: "minivit_mnist_params"
#define FILENAME_CORE CONCAT(MODEL, DATASET, _, SUFFIX)

// 3. Use the preprocessor's ability to join strings automatically
// "includes/" "minivit_mnist_params" ".h" becomes "includes/minivit_mnist_params.h"
#define FULL_PATH TOSTRING(ViTRuns/includes/FILENAME_CORE.h)

#include FULL_PATH
#endif