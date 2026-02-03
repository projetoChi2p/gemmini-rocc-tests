#ifndef __MATH_QUANT_C__
#define __MATH_QUANT_C__

// ==========================================
// 1. BAREMETAL MATH HELPERS
// ==========================================
// Optimized integer-based approximations to avoid linking heavy math libraries

static inline float my_exp(float x) {
    if (x <= -88.0f) return 0.0f;
    if (x >= 88.0f) x = 88.0f;
    union { float f; int32_t i; } converter;
    converter.i = (int32_t)(12102203.0f * x + 1064986824);
    return converter.f;
}

static inline float my_tanh(float x) {
    float abs_x = (x < 0.0f) ? -x : x;
    if (abs_x > 4.0f) return (x < 0.0f) ? -1.0f : 1.0f;
    float e2x = my_exp(2.0f * abs_x);
    float t = (e2x - 1.0f) / (e2x + 1.0f);
    return (x < 0.0f) ? -t : t;
}

static inline int my_round(float x) {
    return (int)(x + (x >= 0 ? 0.5f : -0.5f));
}

// ==========================================
// 2. CPU ACTIVATION & NORMALIZATION
// ==========================================

// GELU Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/*void cpu_gelu_quantized(int rows, int cols, elem_t * input, elem_t * output) {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEF = 0.044715f;
    
    // Heuristic: Input Int8 range maps to approx [-6.0, 6.0]
    const float range_max = 5.0f;
    const float input_scale = range_max / (float) elem_t_max; 
    const float output_scale = (float) elem_t_max / range_max; 

    int size = rows * cols;
    for (int i = 0; i < size; i++) {
        float x = (float)input[i] * input_scale;
        
        // Approx calculation
        float inner = SQRT_2_OVER_PI * (x + COEF * x * x * x);
        float tanh_res = my_tanh(inner);
        float res = 0.5f * x * (1.0f + tanh_res);
        
        // Re-quantize
        int out_val = my_round(res * output_scale);
        if (out_val > elem_t_max) out_val = elem_t_max;
        if (out_val < elem_t_min) out_val = elem_t_min;
        
        output[i] = (elem_t)out_val;
    }
}*/
void cpu_softmax_quantized(int rows, int cols, elem_t * matrix, float input_scale) {
    for (int i = 0; i < rows; i++) {
        // WARNING: Ensure 'cols' is small enough for stack allocation (< 1024 safe)
        float row_buf[cols]; 
        float max_val = -3.40282e+38F;

        // 1. Dequantize & Find Max
        for (int j = 0; j < cols; j++) {
            float val = (float)matrix[i * cols + j] * input_scale;
            row_buf[j] = val;
            if (val > max_val) max_val = val;
        }

        // 2. Exponentiate & Sum
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            float exp_val = my_exp(row_buf[j] - max_val);
            row_buf[j] = exp_val;
            sum_exp += exp_val;
        }

        // 3. Normalize & Re-quantize
        // Map Probability 1.0 -> elem_t_max (e.g., 32767)
        float inv_sum = (float)elem_t_max / (sum_exp + 1e-6f); 

        for (int j = 0; j < cols; j++) {
            int quant_val = (int)(row_buf[j] * inv_sum);
            
            // Saturation checks
            if (quant_val > elem_t_max) quant_val = elem_t_max;
            if (quant_val < 0) quant_val = 0; // Probabilities must be >= 0

            matrix[i * cols + j] = (elem_t)quant_val;
        }
    }
}

/*void cpu_layernorm_quantized(int rows, int cols, elem_t * data, float scale, float shift) {
    for(int i=0; i<rows; i++) {
        float sum=0, sq=0;
        
        // 1. Mean & Variance
        for(int j=0; j<cols; j++) {
            float v = (float)data[i*cols + j];
            sum += v; 
            sq += v*v;
        }
        float mean = sum/cols;
        float var_term = (sq/cols) - (mean*mean);
        if (var_term < 0) var_term = 0; // Safety clamp
        
        float std = sqrtf(var_term + 1e-5); 
        
        // 2. Normalize & Scale
        for(int j=0; j<cols; j++) {
            float v = (float)data[i*cols + j];
            float n = (v - mean)/std * 20.0f; 
            
            // Use explicit rounding helper
            int res = my_round(n);
            
            if(res > elem_t_max) res=elem_t_max; 
            if(res < elem_t_min) res=elem_t_min;
            data[i*cols+j] = (elem_t)res;
        }
    }
}*/


#endif