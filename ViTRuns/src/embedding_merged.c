#include "include/gemmini_nn.h"
#include <stdbool.h>
#include <string.h> // For memcpy

// ==========================================
// CPU IM2PATCH - Extract patches from image
// ==========================================
// Extracts non-overlapping patches from an image
// Supports both HWC (Height, Width, Channel) and CHW (Channel, Height, Width) formats
// 
// Parameters:
//   - image: input image data
//   - img_height, img_width: image dimensions
//   - channels: number of channels (e.g., 3 for RGB)
//   - patch_size: size of each patch (assumes square patches)
//   - input_is_hwc: true if image is in HWC format, false for CHW
//   - output_is_hwc: true to output patches in HWC format, false for CHW
//   - patches_out: output buffer for extracted patches [num_patches x patch_dim]
//
// Returns: number of patches extracted
static int cpu_im2patch(
    const elem_t *image,
    int img_height,
    int img_width,
    int channels,
    int patch_size,
    bool input_is_hwc,
    bool output_is_hwc,
    elem_t *patches_out)
{
    int num_patches_h = img_height / patch_size;
    int num_patches_w = img_width / patch_size;
    int num_patches = num_patches_h * num_patches_w;
    int patch_area = patch_size * patch_size;
    int patch_dim = patch_area * channels;
    
    int patch_idx = 0;
    
    // Extract patches row by row
    for (int ph = 0; ph < num_patches_h; ph++) {
        for (int pw = 0; pw < num_patches_w; pw++) {
            elem_t *current_patch = patches_out + (patch_idx * patch_dim);
            
            // Extract one patch
            for (int y = 0; y < patch_size; y++) {
                for (int x = 0; x < patch_size; x++) {
                    int img_y = ph * patch_size + y;
                    int img_x = pw * patch_size + x;
                    
                    for (int c = 0; c < channels; c++) {
                        elem_t pixel_val;
                        
                        // Read from image (handle input format)
                        if (input_is_hwc) {
                            // HWC: [H][W][C]
                            pixel_val = image[img_y * img_width * channels + 
                                            img_x * channels + c];
                        } else {
                            // CHW: [C][H][W]
                            pixel_val = image[c * img_height * img_width + 
                                            img_y * img_width + img_x];
                        }
                        
                        // Write to patch (handle output format)
                        if (output_is_hwc) {
                            // HWC: [patch_size][patch_size][channels]
                            current_patch[y * patch_size * channels + 
                                        x * channels + c] = pixel_val;
                        } else {
                            // CHW: [channels][patch_size][patch_size]
                            current_patch[c * patch_area + 
                                        y * patch_size + x] = pixel_val;
                        }
                    }
                }
            }
            
            patch_idx++;
        }
    }
    
    return num_patches;
}

static void reorder_patches_hwc_to_chw(int patch_seq_len, int patch_size, int channels,
                                       const elem_t *src, elem_t *dst) {
    int patch_area = patch_size * patch_size;
    int patch_dim = patch_area * channels;

    for (int p = 0; p < patch_seq_len; p++) {
        const elem_t *src_patch = src + (p * patch_dim);
        elem_t *dst_patch = dst + (p * patch_dim);

        for (int c = 0; c < channels; c++) {
            elem_t *dst_c = dst_patch + (c * patch_area);

            for (int y = 0; y < patch_size; y++) {
                const elem_t *src_row = src_patch + (y * patch_size * channels) + c;

                for (int x = 0; x < patch_size; x++) {
                    dst_c[y * patch_size + x] = src_row[x * channels];
                }
            }
        }
    }
}

// ==========================================
// UNIFIED EMBEDDING MODULE
// Supports: FP32, Quantized (Int8), CLS, Distillation
// Modes: Direct patches OR Image-to-patches conversion
// ==========================================
// 
// USAGE EXAMPLES:
// 
// 1. With pre-patchified input (original behavior):
//    compute_patch_embeddings(
//        SEQ_LEN, HIDDEN_DIM, PATCH_DIM, PATCH_SIZE, NUM_CHANNELS, false,
//        false, 0, 0,                     // input_is_image=false, img dims unused
//        patches_data,                    // pre-patchified data
//        patch_embed_w, patch_embed_b,
//        #ifdef QUANTIZED scale_embed, #endif
//        pos_embed_data, cls_token_data,
//        #ifdef DISTILLATION dist_token_data, #endif
//        NULL,                            // im2patch_buf unused
//        patch_reorder_buf, temp_patch_buf, final_input_buf
//    );
//
// 2. With full image input (new feature):
//    compute_patch_embeddings(
//        SEQ_LEN, HIDDEN_DIM, PATCH_DIM, PATCH_SIZE, NUM_CHANNELS, true,
//        true, img_height, img_width,     // input_is_image=true, provide img dims
//        image_data,                      // full image (HWC or CHW format)
//        patch_embed_w, patch_embed_b,
//        #ifdef QUANTIZED scale_embed, #endif
//        pos_embed_data, cls_token_data,
//        #ifdef DISTILLATION dist_token_data, #endif
//        im2patch_buffer,                 // required buffer for patch extraction
//        patch_reorder_buf, temp_patch_buf, final_input_buf
//    );
// ==========================================

void compute_patch_embeddings(
    // 1. Dimensions
    int patch_seq_len,   
    int hidden_dim, 
    int patch_dim,
    int patch_size,
    int channels,
    bool input_is_hwc,
    
    // 2. Input Mode & Data
    bool input_is_image,           // NEW: true if input is full image, false if pre-patchified
    int img_height,                // NEW: image height (if input_is_image=true)
    int img_width,                 // NEW: image width (if input_is_image=true)
    const elem_t * input_data,     // RENAMED: can be image or patches depending on mode
    
    // 3. Weights
    const elem_t * patch_embed_w,
    const acc_t * patch_embed_b,
    
    // 4. Quantization Scale (Only used if QUANTIZED is defined)
    #ifdef QUANTIZED
    float scale_embed,              
    #endif

    // 5. Token Data
    const elem_t * pos_embed_data, 
    const elem_t * cls_token_data,
    #ifdef DISTILLATION
    const elem_t * dist_token_data,
    #endif

    // 6. Buffers
    elem_t * im2patch_buf,         // NEW: buffer for im2patch output (if input_is_image=true)
    elem_t * patch_reorder_buf,
    elem_t * temp_patch_buf,
    elem_t * final_input_buf
    )      
{
    // --- STEP 0: Convert Image to Patches (if needed) ---
    const elem_t *patches_input;
    
    if (input_is_image) {
        // Extract patches from image using CPU im2patch
        #ifdef CPU_IM2PATCH
        int extracted_patches = cpu_im2patch(
            input_data, 
            img_height, img_width, 
            channels, 
            patch_size,
            input_is_hwc,      // input format
            input_is_hwc,      // output format (keep same as input)
            im2patch_buf
        );
        #else
        int extracted_patches = im2patch(
        );
        #endif
        
        // Sanity check
        if (extracted_patches != patch_seq_len) {
            // printf("Warning: Expected %d patches, got %d\n", patch_seq_len, extracted_patches);
        }
        
        patches_input = im2patch_buf;
    } else {
        // Input is already patchified
        patches_input = input_data;
    }
    
    // --- STEP 1: Project Patches (Input -> Temp Buffer) ---
    // Calculate the correct accumulation scale
    acc_scale_t matmul_scale;
    
    #ifdef QUANTIZED
        matmul_scale = (acc_scale_t)scale_embed;
    #else
        matmul_scale = ACC_SCALE_IDENTITY; // Typically 1.0f for FP32
    #endif

    const elem_t *patches_for_matmul = patches_input;

    if (input_is_hwc) {
        reorder_patches_hwc_to_chw(patch_seq_len, patch_size, channels,
                                   patches_input, patch_reorder_buf);
        patches_for_matmul = patch_reorder_buf;
    }

    // The Gemmini MatMul handles the size increase automatically via 'patch_dim'
    tiled_matmul_auto(patch_seq_len, hidden_dim, patch_dim,
        patches_for_matmul, patch_embed_w, patch_embed_b, temp_patch_buf,
        patch_dim, hidden_dim, hidden_dim, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, matmul_scale, 0,
        true, false, false, false, false, 0, WS);
        
    gemmini_fence();

    // --- STEP 2: Construct Sequence [CLS, (DIST), Patches...] ---
    // We use a flexible pointer strategy to support any combination of tokens
    
    size_t row_size_bytes = hidden_dim * sizeof(elem_t);
    int current_token_count = 0; // Tracks where we are writing in the destination buffer

    // A. Copy CLS Token (Index 0)
    if (cls_token_data != NULL) {
        // Copy CLS to the start of final buffer
        memcpy(final_input_buf + (current_token_count * hidden_dim), cls_token_data, row_size_bytes);
        current_token_count++;
    }

    // B. Copy Distillation Token (Optional)
    #ifdef DISTILLATION
    if (dist_token_data != NULL) {
        memcpy(final_input_buf + (current_token_count * hidden_dim), dist_token_data, row_size_bytes);
        current_token_count++;
    } 
    #endif

    // C. Copy Projected Patches
    // We append the patches after whatever tokens (0, 1, or 2) were added above
    // temp_patch_buf contains [patch_seq_len x hidden_dim] data
    memcpy(final_input_buf + (current_token_count * hidden_dim), 
           temp_patch_buf, 
           patch_seq_len * row_size_bytes);

    // --- STEP 3: Add Position Embeddings ---
    // Total sequence length = (Number of Special Tokens) + (Number of Patches)
    int total_seq_len = current_token_count + patch_seq_len;
    
    tiled_resadd_auto(total_seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        final_input_buf, pos_embed_data, final_input_buf,
        false, WS);
        
    gemmini_fence();
}