#include "profiler.h"
#include <string.h>

// ==========================================
// GLOBAL PROFILER STATE
// ==========================================

InferenceProfile g_profile;
int g_profiling_enabled = 0;

// ==========================================
// INITIALIZATION & CLEANUP
// ==========================================

void profiler_init(int num_layers) {
    memset(&g_profile, 0, sizeof(InferenceProfile));
    
    if (num_layers > MAX_LAYERS) {
        printf("ERROR: Too many layers (%d > %d). Increase MAX_LAYERS.\n", num_layers, MAX_LAYERS);
        return;
    }
    
    g_profile.encoder.num_layers = num_layers;
    
    g_profiling_enabled = 1;
    printf("\n=== PROFILER INITIALIZED (%d layers) ===\n", num_layers);
}

void profiler_free() {
    // No-op for static allocation
    g_profiling_enabled = 0;
}

void profiler_reset() {
    if (!g_profiling_enabled) return;
    
    int num_layers = g_profile.encoder.num_layers;
    
    memset(&g_profile, 0, sizeof(InferenceProfile));
    g_profile.encoder.num_layers = num_layers;
}

// ==========================================
// REPORTING FUNCTIONS
// ==========================================

void profiler_print_layer_detail(int layer_idx) {
    if (!g_profiling_enabled || layer_idx >= g_profile.encoder.num_layers) {
        return;
    }
    
    TransformerLayerProfile* layer = &g_profile.encoder.layers[layer_idx];
    
    printf("\n--- Layer %d Details ---\n", layer_idx);
    printf("Total: ");
    print_cycles(layer->total);
    printf(" cycles (100.0%%)\n");
    
    // Attention breakdown
    printf("\nAttention:\n");
    printf("  Total:         ");
    print_cycles(layer->attention.total);
    printf(" cycles (%d.%d%%)\n", 
           calc_percentage_x10(layer->attention.total, layer->total) / 10,
           calc_percentage_x10(layer->attention.total, layer->total) % 10);
    
    printf("    LayerNorm:   ");
    print_cycles(layer->attention.layernorm);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->attention.layernorm, layer->attention.total) / 10,
           calc_percentage_x10(layer->attention.layernorm, layer->attention.total) % 10);
    
    printf("    QKV Proj:    ");
    print_cycles(layer->attention.qkv_projections);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->attention.qkv_projections, layer->attention.total) / 10,
           calc_percentage_x10(layer->attention.qkv_projections, layer->attention.total) % 10);
    
    printf("    Attn Scores: ");
    print_cycles(layer->attention.attention_scores);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->attention.attention_scores, layer->attention.total) / 10,
           calc_percentage_x10(layer->attention.attention_scores, layer->attention.total) % 10);
    
    printf("    Softmax:     ");
    print_cycles(layer->attention.softmax);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->attention.softmax, layer->attention.total) / 10,
           calc_percentage_x10(layer->attention.softmax, layer->attention.total) % 10);
    
    printf("    Context Agg: ");
    print_cycles(layer->attention.context_aggregation);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->attention.context_aggregation, layer->attention.total) / 10,
           calc_percentage_x10(layer->attention.context_aggregation, layer->attention.total) % 10);
    
    printf("    Out Proj:    ");
    print_cycles(layer->attention.output_projection);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->attention.output_projection, layer->attention.total) / 10,
           calc_percentage_x10(layer->attention.output_projection, layer->attention.total) % 10);
    
    printf("    Residual:    ");
    print_cycles(layer->attention.residual_add);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->attention.residual_add, layer->attention.total) / 10,
           calc_percentage_x10(layer->attention.residual_add, layer->attention.total) % 10);
    
    // FFN breakdown
    printf("\nFFN:\n");
    printf("  Total:         ");
    print_cycles(layer->ffn.total);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->ffn.total, layer->total) / 10,
           calc_percentage_x10(layer->ffn.total, layer->total) % 10);
    
    printf("    LayerNorm:   ");
    print_cycles(layer->ffn.layernorm);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->ffn.layernorm, layer->ffn.total) / 10,
           calc_percentage_x10(layer->ffn.layernorm, layer->ffn.total) % 10);
    
    printf("    FC1:         ");
    print_cycles(layer->ffn.fc1);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->ffn.fc1, layer->ffn.total) / 10,
           calc_percentage_x10(layer->ffn.fc1, layer->ffn.total) % 10);
    
    printf("    GELU:        ");
    print_cycles(layer->ffn.gelu);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->ffn.gelu, layer->ffn.total) / 10,
           calc_percentage_x10(layer->ffn.gelu, layer->ffn.total) % 10);
    
    printf("    FC2:         ");
    print_cycles(layer->ffn.fc2);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->ffn.fc2, layer->ffn.total) / 10,
           calc_percentage_x10(layer->ffn.fc2, layer->ffn.total) % 10);
    
    printf("    Residual:    ");
    print_cycles(layer->ffn.residual_add);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(layer->ffn.residual_add, layer->ffn.total) / 10,
           calc_percentage_x10(layer->ffn.residual_add, layer->ffn.total) % 10);
}

void profiler_print_report() {
    if (!g_profiling_enabled) {
        printf("Profiler not enabled\n");
        return;
    }
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║           ViT INFERENCE PROFILING REPORT                      ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    // Overall summary
    printf("\n[TOTAL INFERENCE TIME]: ");
    print_cycles(g_profile.total);
    printf(" cycles\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    // Phase 1: Embedding
    printf("\n1. EMBEDDING:          ");
    print_cycles(g_profile.embedding);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(g_profile.embedding, g_profile.total) / 10,
           calc_percentage_x10(g_profile.embedding, g_profile.total) % 10);
    
    // Phase 2: Encoder (Summary)
    printf("\n2. ENCODER:            ");
    print_cycles(g_profile.encoder.total);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(g_profile.encoder.total, g_profile.total) / 10,
           calc_percentage_x10(g_profile.encoder.total, g_profile.total) % 10);
    
    // Per-layer summary
    printf("\n   Per-Layer Breakdown:\n");
    printf("   ─────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < g_profile.encoder.num_layers; i++) {
        TransformerLayerProfile* layer = &g_profile.encoder.layers[i];
        printf("   Layer %d:            ", i);
        print_cycles(layer->total);
        printf(" cycles (%d.%d%% of encoder)\n",
               calc_percentage_x10(layer->total, g_profile.encoder.total) / 10,
               calc_percentage_x10(layer->total, g_profile.encoder.total) % 10);
        
        printf("     Attention:       ");
        print_cycles(layer->attention.total);
        printf(" cycles (%d.%d%%)\n",
               calc_percentage_x10(layer->attention.total, layer->total) / 10,
               calc_percentage_x10(layer->attention.total, layer->total) % 10);
        
        printf("     FFN:             ");
        print_cycles(layer->ffn.total);
        printf(" cycles (%d.%d%%)\n",
               calc_percentage_x10(layer->ffn.total, layer->total) / 10,
               calc_percentage_x10(layer->ffn.total, layer->total) % 10);
    }
    
    // Phase 3: Classifier
    printf("\n3. CLASSIFIER:         ");
    print_cycles(g_profile.classifier);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(g_profile.classifier, g_profile.total) / 10,
           calc_percentage_x10(g_profile.classifier, g_profile.total) % 10);
    
    // Data Movement Analysis
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("\n[DATA MOVEMENT & OVERHEAD ANALYSIS]\n");
    printf("───────────────────────────────────────────────────────────────\n");
    
    // Calculate actual compute cycles (sum of all measured operations)
    uint64_t compute_cycles = 0;
    for (int i = 0; i < g_profile.encoder.num_layers; i++) {
        TransformerLayerProfile* layer = &g_profile.encoder.layers[i];
        
        // Attention compute operations
        compute_cycles += layer->attention.qkv_projections;
        compute_cycles += layer->attention.attention_scores;
        compute_cycles += layer->attention.context_aggregation;
        compute_cycles += layer->attention.output_projection;
        
        // FFN compute operations
        compute_cycles += layer->ffn.fc1;
        compute_cycles += layer->ffn.fc2;
    }
    compute_cycles += g_profile.embedding;
    compute_cycles += g_profile.classifier;
    
    // Calculate housekeeping (non-matmul operations)
    uint64_t housekeeping_cycles = 0;
    for (int i = 0; i < g_profile.encoder.num_layers; i++) {
        TransformerLayerProfile* layer = &g_profile.encoder.layers[i];
        
        // Attention housekeeping
        housekeeping_cycles += layer->attention.layernorm;
        housekeeping_cycles += layer->attention.softmax;
        housekeeping_cycles += layer->attention.residual_add;
        
        // FFN housekeeping
        housekeeping_cycles += layer->ffn.layernorm;
        housekeeping_cycles += layer->ffn.gelu;
        housekeeping_cycles += layer->ffn.residual_add;
    }
    
    // Data movement overhead (unmeasured time)
    uint64_t measured_cycles = compute_cycles + housekeeping_cycles;
    g_profile.data_movement = (g_profile.total > measured_cycles) ? 
                              (g_profile.total - measured_cycles) : 0;
    
    printf("Compute (MatMul):      ");
    print_cycles(compute_cycles);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(compute_cycles, g_profile.total) / 10,
           calc_percentage_x10(compute_cycles, g_profile.total) % 10);
    
    printf("Housekeeping:          ");
    print_cycles(housekeeping_cycles);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(housekeeping_cycles, g_profile.total) / 10,
           calc_percentage_x10(housekeeping_cycles, g_profile.total) % 10);
    printf("  - LayerNorm\n");
    printf("  - Softmax\n");
    printf("  - GELU\n");
    printf("  - Residual Adds\n");
    
    printf("Data Movement:         ");
    print_cycles(g_profile.data_movement);
    printf(" cycles (%d.%d%%)\n",
           calc_percentage_x10(g_profile.data_movement, g_profile.total) / 10,
           calc_percentage_x10(g_profile.data_movement, g_profile.total) % 10);
    printf("  - DMA transfers (mvin/mvout)\n");
    printf("  - Memory synchronization\n");
    printf("  - Fence/barrier operations\n");
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    
    // Detailed layer view (first and last layer)
    if (g_profile.encoder.num_layers > 0) {
        profiler_print_layer_detail(0);
        if (g_profile.encoder.num_layers > 1) {
            profiler_print_layer_detail(g_profile.encoder.num_layers - 1);
        }
    }
    
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                    END OF PROFILING REPORT                     ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
}
