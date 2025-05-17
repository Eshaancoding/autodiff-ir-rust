#include <cerrno>
#include <iostream>
#include <ctime>
#include <cstdint>
#include <stdio.h>
#include <assert.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

// confirmed by lscpu
#ifndef NTHREADS
    #define NTHREADS 8
#endif

// params
#define B 32
#define I 512
#define O 256
#define NUM_TRIALS 50

float temp = 0.0;

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
inline void rs_cache (float* arr, size_t W, size_t H) {
    /* 
    cache size usually 64 bytesfloat is 4 bytes
    for less branch stuff --> we use 4 bytes
    idea: start from the size of L2 cache?
    also do an ablation test on whether rs cache actually works (and skipping works)
    */

    for (int i = W*H-16; i >= 0; i -= 96) {  
        volatile float* vptr = &arr[i]; // tell compiler not to delete the memory loads (for cache)
        float v = vptr[0];
        v = vptr[-16];
        v = vptr[-32];
        v = vptr[-48];
        v = vptr[-64];
        v = vptr[-80];
    }
}

// can't be inlined as OMP parallel needs to declare as a seperate func
void reverse_skip_cache_multicore (float* arr, size_t W, size_t H) {
    const int w_mini = W / NTHREADS;

    #pragma omp parallel for num_threads(NTHREADS)
    for (int i = 0; i < NTHREADS; i++) {
        rs_cache(&arr[i*w_mini*H], w_mini, H);
    }
}

// ============= Tiled Dot prod =============
// check whether it actually inlines
inline void tiled_dot_prod () {
       
}

// ============= Dot prod =============
void dot_prod (float* A, float* W, float* Res, size_t Bsize, size_t Isize, size_t Osize) {
    reverse_skip_cache_multicore(A, B, I);
};

int main () {
    if ((I % NTHREADS != 0) || (B % NTHREADS != 0)) {
        throw std::invalid_argument("B & I is not divisible by NTHREADS for reverse skip cache multicore");
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
    float avg = 0.0;
    for (int trials = 0; trials < NUM_TRIALS; trials++) {
        uint64_t start_comp = timer();

        dot_prod(A, W, Res, B, I, O);

        uint64_t end_comp = timer();
        float time = (end_comp - start_comp) * 1e-9;
        printf("Trials: %d, Time: %f\n", trials + 1, time);    
        avg += time;
    }

    printf("Avg time: %f", avg/NUM_TRIALS);

    // ============== Free variables =============
    _mm_free(A);
    _mm_free(W);
    _mm_free(Res);
}

/*
Command:

g++ -O3 -march=native -Wall -fopenmp -mno-avx512f fullyopt_matmul.c -o matmul
objdump -d -M intel matmul | less
*/