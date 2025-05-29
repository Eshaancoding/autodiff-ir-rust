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
#include <random>

// check for num threads; confirmed by lscpu
#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(dynamic) num_threads(8)")

// check accuracy flag
// #define CHECK_ACC 

// params (all should be powers of 2, ideally)
// 1024 by 1024 by 1024 gigves around 250 - 290 GFLOPS
#define B 512
#define I 512
#define O 512

// if check acc flag is up, then these will be reduce to 1
#define NUM_TRIALS 20
#define NUM_MM_PERTRIAL 300

// Cache sizes
#define O_CACHE 16 // prefers bigger o cache rather than b cache
#define B_CACHE 16
#define I_CACHE 128 // I think this was the best performing?

// Initialize randomized matrix (from normal distribution)
void init_rand(float* mat, size_t n_elem) {
    for (size_t i = 0; i < n_elem; i++) {
        if (rand() % 2 == 0) {
            mat[i] = rand() / (float)RAND_MAX;
        } else {
            mat[i] = -rand() / (float)RAND_MAX;
        }
    }
}

// Timer 
uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

void print_m512(__m512 vec) {
    alignas(32) float values[16];  // aligned memory for AVX
    _mm512_storeu_ps(values, vec);  // store vec into values

    std::cout << "Contents of __m512: ";
    for (int i = 0; i < 16; ++i) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;
}

// ============= Dot prod =============
// function computation that happens at each core
void dot_prod_core (float* A, float* W, float* Res, int B_size, int I_size, int O_size, int o_start) {
    // iterate over batch size
    for (int b_start = 0; b_start < B_size; b_start += B_CACHE) {
        // initialize res matrix
        __m512 res [B_CACHE * O_CACHE / 16] = {}; // might register overspill since it's an array

        // load the res matrix with the accumulation from the weight matrix.        
        #pragma GCC unroll 8
        for (int b_res = 0; b_res < B_CACHE; b_res++) {
            #pragma GCC unroll 16
            for (int o_res = 0; o_res < (O_CACHE / 16); o_res++) {
                res[b_res*(O_CACHE/16)+o_res] = _mm512_load_ps(&Res[(b_start + b_res) * O_size + (o_start + o_res*16)]);
            }
        }

        // iterate over input cache, using the FMA instruction for the res matrix
        for (int i_c = 0; i_c < I_CACHE; i_c++) {
            // float* A_tiny = &A[i_c*B_size + b_start];
            // float* W_tiny = &W[i_c*O_size + o_start];
            
            // in output cache, iterate and pack into 8/16 reg
            for (int o_reg = 0; o_reg < O_CACHE; o_reg += 16) {
                __m512 m = _mm512_load_ps(&W[i_c*O_size + o_start + o_reg]);

                #pragma GCC unroll 16
                for (int b_reg = 0; b_reg < B_CACHE; b_reg += 1) {
                    __m512 x = _mm512_set1_ps(A[i_c*B_size + b_start + b_reg]);
                    __m512* res_p = &res[b_reg * (O_CACHE/16) + (o_reg/16)];
                    *res_p = _mm512_fmadd_ps(m, x, *res_p);
                }
            }
        }

        // set res matrix back to Res (and do any other computation if needed as well)
        #pragma GCC unroll 8
        for (int b_res = 0; b_res < B_CACHE; b_res++) {
            #pragma GCC unroll 16
            for (int o_res = 0; o_res < (O_CACHE/16); o_res++) { 
                _mm512_store_ps(
                    &Res[(b_start + b_res) * O_size + (o_start + o_res*16)],
                    res[b_res*(O_CACHE/16)+o_res]
                );
            }
        }
    }
}

void dot_prod (float* A, float* W, float* Res, int B_size, int I_size, int O_size) {
    assert(I_size % I_CACHE == 0);
    assert(O_size % O_CACHE == 0);
    assert(B_size % B_CACHE == 0);
    assert(O_CACHE % 16 == 0);

    // iterate over each input size
    for (int i_start = 0; i_start < I_size; i_start += I_CACHE) {

        // iterate over output size to get cache
        PRAGMA_OMP_PARALLEL_FOR 
        for (int o_start = 0; o_start < O_size; o_start += O_CACHE) {
            dot_prod_core(
                // A - col-major marix
                &A[i_start * B_size], 
                // W - row-major matrix  
                &W[i_start * O_size],
                // Send the result of the array to the dot product core. Each core will write to this array at different memory locations, preventing conflicts
                Res,
                // Sizes of matrix
                B_size,
                I_size, 
                O_size,
                // O accessing offset used per core.
                o_start
            );
        }
    }
};

void matmul_naive (const float* A, const float* W, float* Res, int Bsize, int Isize, int Osize) {
    for (int b = 0; b < Bsize; ++b) {
        for (int o = 0; o < Osize; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < Isize; ++i) {
                // sum += A[b * Isize + i] * W[i * O + o];  // this assumes that A and W are both row major
                sum += A[i * Bsize + b] * W[i * O + o]; // A is col major wise, not row major.
            }
            Res[b * Osize + o] = sum;
        }
    }
}

int main () {
    // ============== Alloc =============
    uint64_t start_alloc = timer();
    float* A = (float*)_mm_malloc(B * I * sizeof(float), 64);   // input matrix  (aligned 64; cache length)
    float* W = (float*)_mm_malloc(I * O * sizeof(float), 64);   // weight matrix (aligned 64; cache length)
    float* Res = (float*)_mm_malloc(B * O * sizeof(float), 64); // output matrix (aligned 64; cache length)
    float* ReC = (float*)_mm_malloc(B * O * sizeof(float), 64); // output matrix (aligned 64; cache length)
    init_rand(A, B * I);
    init_rand(W, I * O);

    uint64_t end_alloc = timer();

    printf("Total time alloc: %f s\n", (end_alloc - start_alloc) * 1e-9);
           
    matmul_naive(A, W, ReC, B, I, O);

    // ============== Run trials =============
    float avg = 0.0;
    for (int trials = 0; trials < NUM_TRIALS; trials++) {
        // reset res
        uint64_t start_comp = timer();
        for (int t = 0; t < NUM_MM_PERTRIAL; t++) {
            dot_prod(A, W, Res, B, I, O); // technically, this doesn't work when repeating trials...
            #ifdef CHECK_ACC
                break; // only allow one trial when checking accuracy
            #endif
        }
        uint64_t end_comp = timer();

        #ifdef CHECK_ACC 
            break; // only allow one trial when checking accuracy.
        #endif

        float exec_time = (end_comp - start_comp) * 1e-9 / NUM_MM_PERTRIAL;

        double FLOP = 2 * (double)B * I * O;
        float gflops = (float)(FLOP / exec_time / 1e9);       

        printf("Trials: %d, Time: %f GFLOPS: %f\n", trials + 1, exec_time, gflops);    
        avg += exec_time;
    }

    #ifndef CHECK_ACC
        float avg_time = avg / NUM_TRIALS;
        double FLOP = 2 * (double)B * I * O;
        float gflops = (float)(FLOP / avg_time / 1e9);       
        printf("\nAvg time: %f sec | GFLOPS: %f\n\n", avg_time, gflops);
    #else
        unsigned int incorrect = 0;
        for (int i = 0; i < B * O; i++) {
            if (abs(Res[i] - ReC[i]) >= 1e-5) incorrect += 1;
        }
        printf("Num correct: %d / %d (%.2f accuracy)", B*O-incorrect, B*O, ((float)(B*O-incorrect)/(B*O) * 100));
    #endif

    // ============== Free variables =============
    _mm_free(A);
    _mm_free(W);
    _mm_free(Res);
    _mm_free(ReC);
}

/*
Command:
g++ -O3 -march=native -mavx512f -Wall -fopenmp new_matmul.cpp -o matmul && objdump -d -M intel matmul > matmul_objdump.txt && ./matmul
*/