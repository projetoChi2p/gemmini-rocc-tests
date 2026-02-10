
// ==========================================
// 1. DEBUG HELPER
// ==========================================
bool verify_tensor(const char* name, elem_t* hw_ptr, elem_t* sw_ptr, int size, int tolerance) {
    printf("\n--- VERIFY: %s ---\n", name);
    printf("   Sample [0]: HW %d vs SW %d\n", hw_ptr[0], sw_ptr[0]);
    printf("   Sample [1]: HW %d vs SW %d\n", hw_ptr[1], sw_ptr[1]);

    int scale = 1;
    int errs = 0;
    int max_diff = 0;
    int hist[5] = {0, 0, 0, 0, 0};
    
    for (int i = 0; i < size; i++) {
        int val_hw = (int)(hw_ptr[i] * scale);
        int val_sw = (int)(sw_ptr[i] * scale);
        int diff = abs(val_hw - val_sw);
        
        if (diff > max_diff) max_diff = diff;

        if (diff == 0)      hist[0]++;
        else if (diff <= 2) hist[1]++;
        else if (diff <= 5) hist[2]++;
        else if (diff <= 10) hist[3]++;
        else                hist[4]++;

        if (diff > tolerance * scale) { 
            if (errs < 5) { // Reduced print spam
                printf("   [FAIL] Idx %d: HW %d vs SW %d (Diff %d)\n", i, val_hw, val_sw, diff);
            }
            errs++;
        }
    }

    printf("   Diff Distribution: \n    0: %d \n  1-2: %d \n  3-5: %d \n 6-10: %d \n  >10: %d\n", hist[0], hist[1], hist[2], hist[3], hist[4]);
    printf("   Max Diff: %d\n", max_diff);

    if (errs == 0) {
        printf("   >> PASS\n");
        return true;
    } else {
        printf("   >> FAIL (%d errors)\n", errs);
        return false;
    }
}

void display_tensor(const char* name, elem_t* data, int size) {
    printf("\n--- TENSOR: %s ---\n", name);
    for (int i = 0; i < size; i++) {
        printf("%d ", data[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");
}

#define HISTOGRAM_BINS 10
#define HISTOGRAM_MAX_LEN 100
void display_tensor_distribution_histogram(const char* name, elem_t* data, int size) {
    printf("\n--- VALUE DISTRIBUTION: %s ---\n", name);
    
    // find min and max for range
    int min_val = data[0];
    int max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    printf("Value Range: [%d, %d]\n", min_val, max_val);

    int bins[HISTOGRAM_BINS] = {0};
    int step = (max_val - min_val) / HISTOGRAM_BINS + 1; // Avoid division by zero
    for (int i = 0; i < size; i++) {
        int val = data[i];
        for (int b = 0; b < HISTOGRAM_BINS; b++) {
            if (val <= min_val + step * (b + 1)) {
                bins[b]++;
                break;
            }
        }
    }

    int histogram_length = 0;
    for (int b = 0; b < HISTOGRAM_BINS; b++) {
        if (bins[b] > histogram_length) histogram_length = bins[b];
    }

    for (int b = 0; b < HISTOGRAM_BINS; b++) {
        printf("  Bin %d [%d, %d]: %d \t", b, min_val + step * b, min_val + step * (b + 1), bins[b]);
        int bar_length = (bins[b] * HISTOGRAM_MAX_LEN) / histogram_length; // Scale to max 50 chars
        for (int i = 0; i < bar_length; i++) {
            printf("*");
        }
        printf("\n");
    }
}

// ==========================================
// 2. HELPER FUNCTIONS
// ==========================================

elem_t argmax(int size, elem_t * input) {
    elem_t max_val = input[0];
    int max_index = 0;
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_index = i;
        }
    }
    return max_index;
}

void global_average_pool(int batch_size, int rows, int cols, int channels,
                         elem_t input[batch_size][rows][cols][channels],
                         elem_t output[batch_size][channels]) {
    int num_elements = rows * cols;
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            int32_t sum = 0;
            for (int r = 0; r < rows; r++) {
                for (int col = 0; col < cols; col++) {
                    sum += input[b][r][col][c];
                }
            }
            // Simple integer average
            output[b][c] = (elem_t)(sum / num_elements);
        }
    }
}

void flatten_weights(int out_channels, int kernel_dim, int in_channels,
        elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
        elem_t weights_mat[kernel_dim * kernel_dim * in_channels][out_channels]) {

    for (int outc = 0; outc < out_channels; outc++) {
        for (int krow = 0; krow < kernel_dim; krow++) {
            for (int kcol = 0; kcol < kernel_dim; kcol++) {
                for (int inc = 0; inc < in_channels; inc++) {
                    int wmatrow = krow * kernel_dim * in_channels +
                        kcol * in_channels +
                        inc;
                    weights_mat[wmatrow][outc] = weights[outc][krow][kcol][inc];
                }
            }
        }
    }
}
