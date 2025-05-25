```bash
#!/bin/bash
set -e

THREADS=${1:-16}
export OMP_NUM_THREADS=$THREADS

cat << EOF > matrix_multiply.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 2000

void init_matrix(double mat[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

// 串行版本
void serial_multiply(double A[N][N], double B[N][N], double C[N][N]) {
    clock_t start = clock();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double duration = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("串行版本运行时间: %.2f秒\n", duration);
}

// Parallel for版本
void parallel_for_multiply(double A[N][N], double B[N][N], double C[N][N]) {
    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double duration = omp_get_wtime() - start;
    printf("Parallel for版本运行时间: %.2f秒\n", duration);
}

// Transparent版本
void transparent_multiply(double A[N][N], double B[N][N], double C[N][N]) {
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        #pragma omp taskloop default(none) shared(A,B,C) collapse(2) grainsize(50)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] = 0;
                for (int k = 0; k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
    double duration = omp_get_wtime() - start;
    printf("Transparent版本运行时间: %.2f秒\n", duration);
}

int main() {
    static double A[N][N], B[N][N], C[N][N];
    srand(time(NULL));
    init_matrix(A);
    init_matrix(B);

    printf("当前线程数: %d\n", omp_get_max_threads());

    printf("\n===== 串行版本 =====\n");
    serial_multiply(A, B, C);

    printf("\n===== Parallel for版本 =====\n");
    parallel_for_multiply(A, B, C);

    printf("\n===== Transparent版本 =====\n");
    transparent_multiply(A, B, C);
}
EOF

gcc -fopenmp matrix_multiply.c -o matrix_multiply
./matrix_multiply

rm -f matrix_multiply matrix_multiply.c