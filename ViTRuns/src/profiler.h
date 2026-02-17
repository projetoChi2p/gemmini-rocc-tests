#ifndef PROFILER_H
#define PROFILER_H

#include <stdint.h>
#include <stdio.h>
#include "include/gemmini.h"

// ==========================================
// PROFILER STRUCTURES
// ==========================================

// Attention sub-operation timings
typedef struct {
    uint64_t layernorm;
    uint64_t qkv_projections;
    uint64_t attention_scores;
    uint64_t softmax;
    uint64_t context_aggregation;
    uint64_t output_projection;
    uint64_t residual_add;
    uint64_t total;
} AttentionProfile;

// FFN sub-operation timings
typedef struct {
    uint64_t layernorm;
    uint64_t fc1;
    uint64_t gelu;
    uint64_t fc2;
    uint64_t residual_add;
    uint64_t total;
} FFNProfile;

// Per-layer transformer profiling
typedef struct {
    AttentionProfile attention;
    FFNProfile ffn;
    uint64_t total;
} TransformerLayerProfile;

#ifndef MAX_LAYERS
#define MAX_LAYERS 12  // Support up to 12 transformer layers
#endif

// Overall transformer encoder profiling
typedef struct {
    TransformerLayerProfile layers[MAX_LAYERS];  // Static array of layer profiles
    int num_layers;
    uint64_t total;
} EncoderProfile;

// Top-level inference profiling
typedef struct {
    uint64_t embedding;
    EncoderProfile encoder;
    uint64_t classifier;
    uint64_t total;
    uint64_t data_movement;  // Calculated: total - (actual computation)
} InferenceProfile;

// ==========================================
// GLOBAL PROFILER STATE
// ==========================================

extern InferenceProfile g_profile;
extern int g_profiling_enabled;

// ==========================================
// PROFILER FUNCTIONS
// ==========================================

// Initialize profiler (no allocation needed, using static arrays)
void profiler_init(int num_layers);

// Free profiler memory (no-op for static allocation)
void profiler_free();

// Reset all counters
void profiler_reset();

// Print comprehensive profiling report
void profiler_print_report();

// Print detailed breakdown for a specific layer
void profiler_print_layer_detail(int layer_idx);

// Helper: Calculate percentage (returns integer percentage * 10 for one decimal place)
static inline int calc_percentage_x10(uint64_t part, uint64_t total) {
    if (total == 0) return 0;
    // Scale to avoid overflow: (part * 1000) / total
    return (int)((part * 1000) / total);
}

// Helper: Print cycles in readable format
static inline void print_cycles(uint64_t cycles) {
    if (cycles >= 1000000000) {
        printf("%llu", (unsigned long long)(cycles / 1000000000));
        printf(".");
        printf("%llu", (unsigned long long)((cycles / 100000000) % 10));
        printf("%llu B", (unsigned long long)((cycles / 10000000) % 10));
    } else if (cycles >= 1000000) {
        printf("%llu", (unsigned long long)(cycles / 1000000));
        printf(".");
        printf("%llu", (unsigned long long)((cycles / 100000) % 10));
        printf("%llu M", (unsigned long long)((cycles / 10000) % 10));
    } else if (cycles >= 1000) {
        printf("%llu", (unsigned long long)(cycles / 1000));
        printf(".");
        printf("%llu", (unsigned long long)((cycles / 100) % 10));
        printf("%llu K", (unsigned long long)((cycles / 10) % 10));
    } else {
        printf("%llu", (unsigned long long)cycles);
    }
}

#endif // PROFILER_H
