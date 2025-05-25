```bash
#!/bin/bash

# 设置线程数，默认为16，可通过命令行参数修改，如 ./script.sh 64
THREADS=${1:-16}

cat << EOF > matrix_multiply.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 1024
#define REPEAT 5

void init_matrix(double *mat) {
    for (int i = 0; i < N*N; i++) {
        mat[i] = (double)rand() / RAND_MAX;
    }
}

int main() {
    double *A = malloc(N*N*sizeof(double));
    double *B = malloc(N*N*sizeof(double));
    double *C = malloc(N*N*sizeof(double));
    init_matrix(A);
    init_matrix(B);

    // 串行版本
    clock_t start = clock();
    for (int r = 0; r < REPEAT; r++) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                double temp = A[i*N + k];
                for (int j = 0; j < N; j++) {
                    C[i*N + j] += temp * B[k*N + j];
                }
            }
        }
    }
    double serial_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("串行版本运行时间: %.3f秒\n", serial_time);

    // 重置结果矩阵
    for (int i = 0; i < N*N; i++) C[i] = 0.0;

    // parallel for版本
    start = clock();
    #pragma omp parallel for collapse(2) num_threads($THREADS)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double temp = A[i*N + k];
            for (int j = 0; j < N; j++) {
                #pragma omp atomic
                C[i*N + j] += temp * B[k*N + j];
            }
        }
    }
    double parallel_for_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("parallel for版本运行时间: %.3f秒 (加速比: %.2fx)\n", parallel_for_time, serial_time/parallel_for_time);

    // 重置结果矩阵
    for (int i = 0; i < N*N; i++) C[i] = 0.0;

    // scope版本
    start = clock();
    #pragma omp parallel default(none) shared(A, B, C) num_threads($THREADS)
    {
        #pragma omp for collapse(2) private(i, k, j) schedule(dynamic)
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                double temp = A[i*N + k];
                #pragma omp scope reduction(+:C[i*N +j])
                for (int j = 0; j < N; j++) {
                    C[i*N + j] += temp * B[k*N + j];
                }
            }
        }
    }
    double scope_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("scope版本运行时间: %.3f秒 (加速比: %.2fx)\n", scope_time, serial_time/scope_time);

    free(A);
    free(B);
    free(C);
    return 0;
}
EOF

gcc -fopenmp -O3 matrix_multiply.c -o matrix_multiply
./matrix_multiply
rm matrix_multiply matrix_multiply.c