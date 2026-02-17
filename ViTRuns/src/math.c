#ifndef MATH_C
#define MATH_C
#include "include/gemmini_params.h"
#include "include/gemmini_nn.h"

static inline int my_round_ffn(float x) {
    float y = x + 0.5f;
    int i = (int)y; // trunc toward 0
    if (y < 0.0f && y != (float)i) {
        i -= 1; // emulate floor for negative values
    }
    return i;
}

#ifdef QUANTIZED
    void cpu_layernorm(int rows, int cols, elem_t * data) {
        for(int i=0; i<rows; i++) {
            double sum = 0.0;

            // 1. Mean (use double to match numpy's default accumulation)
            for(int j=0; j<cols; j++) {
                double v = (double)data[i*cols + j];
                sum += v;
            }
            double mean = sum / (double)cols;

            // 2. Variance (two-pass, more stable like numpy)
            double var_acc = 0.0;
            for(int j=0; j<cols; j++) {
                double v = (double)data[i*cols + j];
                double d = v - mean;
                var_acc += d * d;
            }
            double var_term = var_acc / (double)cols;
            if (var_term < 0) var_term = 0; // Safety clamp

            double std = sqrt(var_term + 1e-5); 

            // 3. Normalize & Scale
            for(int j=0; j<cols; j++) {
                double v = (double)data[i*cols + j];
                double n = (v - mean) / std * (double)RANGE_LN_OUT;

                // Use explicit rounding helper
                int res = my_round_ffn((float)n);

                if(res > elem_t_max) res=elem_t_max; 
                if(res < elem_t_min) res=elem_t_min;
                data[i*cols+j] = (elem_t)res;
            }
        }
    }
    // --- QUANTIZED HELPERS ---
    static inline float my_exp_att(float x) {
        if (x <= -88.0f) return 0.0f;
        if (x >= 88.0f) x = 88.0f;
        union { float f; int32_t i; } converter;
        converter.i = (int32_t)(12102203.0f * x + 1064986824);
        return converter.f;
    }

    // ==========================================
    // 2. CPU SOFTMAX
    // ==========================================
    void cpu_softmax_quantized(int rows, int cols, elem_t * matrix, float input_scale) {
        for (int i = 0; i < rows; i++) {
            float row_buf[cols]; 
            float max_val = -3.40282e+38F;

            for (int j = 0; j < cols; j++) {
                float val = (float)matrix[i * cols + j] * input_scale;
                row_buf[j] = val;
                if (val > max_val) max_val = val;
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < cols; j++) {
                float exp_val = my_exp_att(row_buf[j] - max_val);
                row_buf[j] = exp_val;
                sum_exp += exp_val;
            }

            float inv_sum = (float) elem_t_max / (sum_exp + 1e-6f); 
            for (int j = 0; j < cols; j++) {
                int quant_val = (int)(row_buf[j] * inv_sum);
                if (quant_val > elem_t_max) quant_val = elem_t_max;
                if (quant_val < 0) quant_val = 0;
                matrix[i * cols + j] = (elem_t)quant_val;
            }
        }
    }


#else

    // Fast exp approximation that doesn't require libm
    static inline float fast_exp_approx(float x) {
        if (x <= -88.0f) return 0.0f;
        if (x >= 88.0f) x = 88.0f;
        union { float f; int i; } u;
        // Schraudolph-style approximation to expf(x)
        u.i = (int)(12102203.0f * x + 1064986824);
        return u.f;
    }

    void cpu_softmax(int rows, int cols, elem_t * matrix) {
        for (int i = 0; i < rows; i++) {
            // A. Find Max
            float max_val = -3.40282347e+38;
            for (int j = 0; j < cols; j++) {
                float val = (float)matrix[i * cols + j];
                if (val > max_val) max_val = val;
            }

            // B. Exponentiate (x - max)
            float sum_exp = 0.0f;
            for (int j = 0; j < cols; j++) {
                float val = (float)matrix[i * cols + j] - max_val;
                float exp_val = fast_exp_approx(val);
                matrix[i * cols + j] = (elem_t)exp_val;
                sum_exp += exp_val;
            }

            // C. Normalize
            float inv_sum = 1.0f / (sum_exp + 1e-6f); 
            for (int j = 0; j < cols; j++) {
                matrix[i * cols + j] *= inv_sum;
            }
        }
    }

    void cpu_scale_matrix(int rows, int cols, elem_t * matrix, float scale) {
        int size = rows * cols;
        for (int i = 0; i < size; i++) {
            matrix[i] = (elem_t)(matrix[i] * scale);
        }
    }

#endif
#endif