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

// confirmed by lscpu
#ifndef NTHREADS
    #define NTHREADS 8
#endif

#define DEBUG 

#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(dynamic) num_threads(8)")

// params (all should be powers of 2, ideally)
#define B 256
#define I 256
#define O 256

#define O_CACHE 8
#define B_CACHE 16

#ifdef DEBUG
    #define NUM_TRIALS 1
#else
    #define NUM_TRIALS 100
#endif

// if using __mm256, use 8
// if using __mm512, use 16
#define FPSIZE 8 

// ========= update this: O_CACHE / FPSIZE ========
float temp = 0.0;

// 55 MiB

// Initialize randomized matrix
void init_rand(float* mat, size_t n_elem) {
    const unsigned int seed = 42;
    std::mt19937 generator(seed); // ensures that the random number generator will be the same throughout iterations

    // Mean = 0.0, Standard deviation = 1.0 (standard normal distribution)
    std::normal_distribution<double> distribution(0.0, 1.0);
   
    srand(time(NULL));
    for (size_t i = 0; i < n_elem; i++) {
        mat[i] = distribution(generator);
    }
}

// Timer 
uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}


void print_m256(__m256 vec) {
    alignas(32) float values[8];  // aligned memory for AVX
    _mm256_storeu_ps(values, vec);  // store vec into values

    std::cout << "Contents of __m256: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;
}

// ============= Dot prod =============
// function computation that happens at each core
void dot_prod_core (float* A, float* W, float* Res, int B_size, int I_size, int O_size, int core_idx) {
    const int w_glob_offset = O_size * core_idx; // global x offset 
    const int total_O = O_size * NTHREADS;

    // maybe set i cache over here?
    for (int o_cache_idx = 0; o_cache_idx < O_size; o_cache_idx += O_CACHE) {
        for (int b_cache_idx = 0; b_cache_idx < B_size; b_cache_idx += B_CACHE) {
            const int w_loc_offset = w_glob_offset + o_cache_idx;        
            // initialize res matrix
            __m256 res [B_CACHE * O_CACHE / 8] = {}; // might register overspill since it's an array

            // Go through I size
            for (int i_idx = 0; i_idx < I_size; i_idx++) {
                /* 
                The fact that this is dependent on I and skipping is kind of pissing me off
                but if I move it earlier, then I would have to initialize the res matrix later... less effective FMAs.
                It's a tradeoff. If I is large (which can be in some cases!), then it might be helpful to use another kernel method. 
                However, I am not concerned with CPU optimization for ML at the moment, as most optimizations exist at the GPU anyways.
                There's other tradeoffs, like splitting the weight matrix... OR isntead splitting the A matrix, etc. etc. etc.
                At GPU there will be multiple types of kernels that will specialize in specific dimensions & operations (if needed)
                */
                float* a_start = &A[B_size*i_idx + b_cache_idx]; 
                float* w_start = &W[total_O*i_idx + o_cache_idx + w_loc_offset];
                 
                for (int w_tiny = 0; w_tiny < (O_CACHE/8); w_tiny += 1) { // change 8 to 16 to _mm512
                    __m256 m = _mm256_load_ps(&w_start[w_tiny*8]);
                    
                    #pragma GCC unroll 16
                    for (int a_tiny = 0; a_tiny < B_CACHE; a_tiny += 1) {
                        __m256 x = _mm256_broadcast_ss(&a_start[a_tiny]);
                        __m256* res_p = &res[w_tiny*B_CACHE + a_tiny];
                        *res_p = _mm256_fmadd_ps(m, x, *res_p);
                    }
                }
            }

            // set res matrix into W (and do any other computation if needed as well)
            for (int i_res = 0; i_res < B_CACHE; i_res++) {
                #pragma GCC unroll 16
                for (int x_res = 0; x_res < (O_CACHE / 8); x_res++) {
                    _mm256_store_ps(
                        &Res[(b_cache_idx + i_res)*total_O + w_loc_offset+(x_res*8)], 
                        res[i_res*(O_CACHE/8)+x_res]
                    );
                }
            }
        }
    }
}

void dot_prod (float* A, float* W, float* Res, int B_size, int I_size, int O_size) {
    const int o_cache_size = O_size / NTHREADS;    

    assert(O_size % NTHREADS == 0); // condition is not always guaranteed
    assert(o_cache_size % O_CACHE == 0);
    assert(O_CACHE % 8 == 0);
    assert(B_size % B_CACHE == 0);

    #ifndef DEBUG
        PRAGMA_OMP_PARALLEL_FOR 
    #endif
    for (int core_idx = 0; core_idx < NTHREADS; core_idx++) {
        dot_prod_core(
            // A - col-major marix
            A, 
            // W - row-major matrix  
            W,
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

void matmul_naive (const float* A, const float* W, float* Res, int Bsize, int Isize, int Osize) {
    for (int b = 0; b < Bsize; ++b) {
        for (int o = 0; o < Osize; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < Isize; ++i) {
                // sum += A[b * Isize + i] * W[i * O + o];  // this assumes that A and W are both row major
                sum += A[i * Bsize + b] * W[i * O + o]; // we are assuming A is col major
            }
            Res[b * Osize + o] = sum;
        }
    }
}

int main () {
    if ((I % NTHREADS != 0) || (B % NTHREADS != 0)) {
        throw std::invalid_argument("B & I is not divisible by NTHREADS for reverse skip cache multicore");
    }
    
    // ============== Alloc =============
    uint64_t start_alloc = timer();
    float* A = (float*)_mm_malloc(B * I * sizeof(float), 64);   // input matrix  (aligned 64; cache length)
    float* W = (float*)_mm_malloc(I * O * sizeof(float), 64);   // weight matrix (aligned 64; cache length)
    float* Res;
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
        if (trials > 0) _mm_free(Res);
        Res = (float*)_mm_malloc(B * O * sizeof(float), 64); // output matrix (aligned 64; cache length)

        uint64_t start_comp = timer();
        dot_prod(A, W, Res, B, I, O);
        uint64_t end_comp = timer();

        float exec_time = (end_comp - start_comp) * 1e-9;

        double FLOP = 2 * (double)B * I * O;
        float gflops = (float)(FLOP / exec_time / 1e9);       

        printf("Trials: %d, Time: %f GFLOPS: %f\n", trials + 1, exec_time, gflops);    
        if (trials > 0) { // first usually is warmup
            avg += exec_time;
        }
    }

    float avg_time = avg / (NUM_TRIALS-1);
    double FLOP = 2 * (double)B * I * O;
    float gflops = (float)(FLOP / avg_time / 1e9);       
    printf("Avg time: %f sec | GFLOPS: %f\n", avg_time, gflops);
    
    printf("\n");
    for (int i = 0; i < 16; i++) {
        printf("Idx %d | Test: %f Correct: %f\n", i, Res[i], ReC[i]);
    }
    printf("...\n");
    printf("Idx last | Test: %f Correct: %f", Res[B * O - 1], ReC[B * O - 1]);

    // ============== Free variables =============
    _mm_free(A);
    _mm_free(W);
    _mm_free(Res);
    _mm_free(ReC);
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


*/