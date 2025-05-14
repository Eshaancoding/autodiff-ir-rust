#include <stdio.h>
#include <assert.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <iostream>

// confirmed by lscpu
#ifndef NTHREADS
    #define NTHREADS 8
#endif

// params
#define B 32
#define I 512
#define O 256
#define NUM_TRIALS 50

// 55 MiB

// Initialize randomized matrix
void init_rand(float* mat, size_t n_elem) {
    for (size_t i = 0; i < n_elem; i++) {
        mat[i] = rand() / (float)RAND_MAX;
    }
}

// Timer 
uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1e9 + (uint64_t)start.tv_nsec;
}

// ============= Reverse Skip Caching =============
// check whether it actually inlines
inline void reverse_skip_cache (float* arr, size_t W, size_t H) {
    // cache size usually 64 bytes
    // float is 4 bytes
    for (int i = W*H-16; i >= 0; i -= 16) {  
        float v = arr[min(0, i)]; 
        // O3 might remove this...
        // might have to manually unroll 
    }
}

inline void reverse_skip_cache_multicore (float* arr, size_t W, size_t H) {
    const int w_mini = (W//NTHREADS);

    #pragma omp parallel for num_threads(NTHREADS)
    for (int i = 0; i < NTHREADS; i++) {
        reverse_skip_cache(arr[i*w_mini*H], w_mini, H);
    }
}

// ============= Tiled Dot prod =============
// check whether it actually inlines
inline void tiled_dot_prod () {
    
}

// ============= Dot prod =============
inline void dot_prod (float* A, float* W, float* Res, size_t B, size_t I, size_t O) {
    
};

int main () {
    if ((I % NTHREADS != 0) || (B % NTHREADS != 0)) {
        throw invalid_argument("B & I is not divisible by NTHREADS for reverse skip cache multicore")        
    }
    
    // ============== Alloc =============
    uint64_t start_alloc = timer();
    float* A = (float*)_mm_malloc(B * I * sizeof(float), 64);   // input matrix  (aligned 64; cache length)
    float* W = (float*)_mm_malloc(I * O * sizeof(float), 64);   // weight matrix (aligned 64; cache length)
    float* Res = (float*)_mm_malloc(B * O * sizeof(float), 64); // output matrix (aligned 64; cache length)
    init_rand(A, B * I);
    init_rand(W, I * O);
    uint64_t end_alloc = timer();

    printf("Total time alloc: %f s\n", (end_alloc - start_alloc) * 1e-9);
           
    // ============== Run trials =============
    for (int trials = 0; trials < num_trials; trials++) {
        uint64_t start_comp = timer();

        dot_prod(A, W, Res, B, I, O);

        uint64_t end_comp = timer();
        printf("Trials: %d, Time: %f", trials + 1, (end_comp - start_comp) * 1e-9);    
    }

    // ============== Free variables =============
    _mm_free(A);
    _mm_free(W);
    _mm_free(Res);
}

/*
Command:

gcc -O3 -march=native -Wall -fopenmp -mno-avx512f fullyopt_matmul.c -o matmul
objdump -d -M intel matmul | less
*/