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

#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(dynamic) num_threads(8)")

// params (all should be powers of 2, ideally)
#define B 64
#define I 512
#define O 512

#define O_CACHE 32
#define B_CACHE 

#define NUM_TRIALS 20

// if using __mm256, use 8
// if using __mm512, use 16
#define FPSIZE 8 

// ========= update this: O_CACHE / FPSIZE ========
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
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
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


// ============= Dot prod =============
// function computation that happens at each core
void dot_prod_core (float* A, float* W, float* Res, int B_size, int I_size, int O_size, int core_idx) {
    
}

void dot_prod (float* A, float* W, float* Res, int B_size, int I_size, int O_size) {
    assert(O_size % NTHREADS == 0); // condition is not always guaranteed
    assert(B_size % NTHREADS == 0);
    assert(I_size % 16 == 0);
    
    const int o_cache_size = O_size / NTHREADS;    

    PRAGMA_OMP_PARALLEL_FOR 
    for (int core_idx = 0; core_idx < NTHREADS; core_idx++) {
        dot_prod_core(
            // A - x (input) col-major marix
            A, 
            // W - weight row-major matrix  
            &W[I_size * o_cache_size],
            // Send the result of the array to the dot product core. Each core will write to this array at different memory locations, preventing conflicts
            Res,
            // Batch size stays the same (iterated per each core)
            B_size,
            // Input size stays the same (iterated each per core)
            I_size, 
            // The size of the output is now the o_cache_size, because we split the "W" matrix per each core.
            o_cache_size,
            // core idx
            core_idx
        );
    }
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

        float exec_time = (end_comp - start_comp) * 1e-9;

        double FLOP = 2 * (double)B * I * O;
        float gflops = (float)(FLOP / exec_time / 1e9);       

        printf("Trials: %d, Time: %f GFLOPS: %f\n", trials + 1, exec_time, gflops);    
        avg += exec_time;
    }

    float avg_time = avg / NUM_TRIALS;
    double FLOP = 2 * (double)B * I * O;
    float gflops = (float)(FLOP / avg_time / 1e9);       
    printf("Avg time: %f sec | GFLOPS: %f\n", avg_time, gflops);

    // ============== Free variables =============
    _mm_free(A);
    _mm_free(W);
    _mm_free(Res);
}

/*
Command:

g++ -O3 -march=native -mavx512f -Wall -fopenmp new_matmul.cpp -o matmul && objdump -d -M intel matmul > matmul_objdump.txt && ./matmul

========== Reference result
m=n=k=256 | GFLOPS = 37
m=n=k=256 | GFLOPS = 68
m=n=k=256 | GFLOPS = 62
m=n=k=256 | GFLOPS = 62
m=n=k=256 | GFLOPS = 63
m=n=k=256 | GFLOPS = 69
m=n=k=256 | GFLOPS = 69
m=n=k=256 | GFLOPS = 68
m=n=k=256 | GFLOPS = 70
m=n=k=256 | GFLOPS = 79
m=n=k=256 | GFLOPS = 79
m=n=k=256 | GFLOPS = 89
m=n=k=256 | GFLOPS = 82
m=n=k=256 | GFLOPS = 82
m=n=k=256 | GFLOPS = 81
m=n=k=256 | GFLOPS = 93

I have seen it reach around 200 ish before


*/