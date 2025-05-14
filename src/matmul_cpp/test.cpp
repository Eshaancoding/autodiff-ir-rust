#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512
#define REPEAT 5000  // Repeat to simulate a "bunch" of multiplications

float A[N][N] __attribute__((aligned(64)));
float B[N][N] __attribute__((aligned(64)));
float C[N][N] __attribute__((aligned(64)));

void initialize_matrices() {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            A[i][j] = (float)(rand()) / RAND_MAX;
            B[i][j] = (float)(rand()) / RAND_MAX;
        }
}

void elw_multiply_avx512() {
    for (int r = 0; r < REPEAT; ++r) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; j += 16) {
                __m512 a = _mm512_load_ps(&A[i][j]);
                __m512 c = _mm512_load_ps(&C[i][j]);
                // c = _mm512_mul_ps(a, _mm512_mul_ps(a, _mm512_mul_ps(a, _mm512_mul_ps(a, _mm512_mul_ps(a, c)))));
                c = _mm512_mul_ps(a, c);
                _mm512_store_ps(&C[i][j], c);
            }
        }
    }
}

int main() {
    srand((unsigned)time(NULL));
    initialize_matrices();

    clock_t start = clock();
    elw_multiply_avx512();
    clock_t end = clock();

    printf("Completed %d element-wise multiplications on %dx%d matrix.\n", REPEAT, N, N);
    printf("Elapsed time: %.4f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Optional: print part of the result
    printf("C[0][0] = %f\n", C[0][0]);

    return 0;
}

// 1000 * 5 inline --> 0.052 seconds
// 5000 --> 0.2157
// okay so clear difference here. nearly 5x
// "kernel fusion" would be beneficial


/**
gcc -O3 -mavx512f test.cpp -o elw_mul_avx512
objdump -d -M intel elw_mul_avx512 | less
*/