```bash
#!/bin/bash

# 设置矩阵大小和线程数
MATRIX_SIZE=2000
THREADS=${1:-16}  # 默认为16线程

# 编译选项
CC=gcc
CFLAGS="-fopenmp -O3 -lm"

cat <<EOF > matrix_multiply.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#define N $MATRIX_SIZE

void initialize_matrix(double *mat) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i*N + j] = (double)rand() / RAND_MAX;
        }
    }
}

// 串行版本
void serial_multiply(double *A, double *B, double *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

// parallel for版本
void parallel_for_multiply(double *A, double *B, double *C) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

// reverse_offload版本1
void reverse_offload_v1(double *A, double *B, double *C) {
    #pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
    {
        #pragma omp target teams distribute parallel for reverse_offload
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A[i*N + k] * B[k*N + j];
                }
                C[i*N + j] = sum;
            }
        }
    }
}

// reverse_offload版本2
void reverse_offload_v2(double *A, double *B, double *C) {
    #pragma omp target enter data map(to: A[0:N*N], B[0:N*N])
    #pragma omp target teams distribute parallel for reverse_offload map(always, from: C[0:N*N])
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    #pragma omp target exit data map(from: C[0:N*N])
}

int main() {
    double *A = malloc(N*N*sizeof(double));
    double *B = malloc(N*N*sizeof(double));
    double *C = malloc(N*N*sizeof(double));
    
    initialize_matrix(A);
    initialize_matrix(B);

    // 串行版本
    double start = omp_get_wtime();
    serial_multiply(A, B, C);
    double serial_time = omp_get_wtime() - start;

    // parallel for版本
    start = omp_get_wtime();
    parallel_for_multiply(A, B, C);
    double parallel_for_time = omp_get_wtime() - start;

    // reverse_offload版本1
    start = omp_get_wtime();
    reverse_offload_v1(A, B, C);
    double rev_offload_v1_time = omp_get_wtime() - start;

    // reverse_offload版本2
    start = omp_get_wtime();
    reverse_offload_v2(A, B, C);
    double rev_offload_v2_time = omp_get_wtime() - start;

    printf("矩阵大小: %dx%d，线程数: %d\n", N, N, $THREADS);
    printf("----------------------------------------\n");
    printf("串行版本运行时间: %.2f秒\n", serial_time);
    printf("parallel for加速比: %.2fx\n", serial_time / parallel_for_time);
    printf("reverse_offload版本1加速比: %.2fx\n", serial_time / rev_offload_v1_time);
    printf("reverse_offload版本2加速比: %.2fx\n", serial_time / rev_offload_v2_time);

    free(A);
    free(B);
    free(C);
    return 0;
}
EOF

# 编译和运行
$CC $CFLAGS matrix_multiply.c -o matrix_multiply
OMP_NUM_THREADS=$THREADS ./matrix_multiply