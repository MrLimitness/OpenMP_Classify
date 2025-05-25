```bash
#!/bin/bash

set -e

# 矩阵维度设置（可调整）
MATRIX_SIZE=512

# 线程数设置（通过参数传入，默认16）
THREADS=${1:-16}

cat << EOF > omp_metadirective_demo.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N $MATRIX_SIZE

void init_matrix(double *mat) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N*N; i++) {
        mat[i] = (double)rand() / RAND_MAX;
    }
}

// 串行版本
void serial_matmul(double *A, double *B, double *C) {
    double start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    printf("串行版本运行时间: %.3f秒\n", omp_get_wtime() - start);
}

// parallel for版本
void parallel_for_matmul(double *A, double *B, double *C) {
    double start = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    printf("parallel for版本运行时间: %.3f秒\n", omp_get_wtime() - start);
}

// metadirective版本
void metadirective_matmul(double *A, double *B, double *C) {
    double start = omp_get_wtime();
    #pragma omp metadirective \
        when(implementation={vendor(gnu)}, device={isa(cpu)}: parallel for schedule(dynamic)) \
        when(implementation={vendor(gnu)}, device={isa(any)}: parallel for schedule(guided)) \
        default(parallel for schedule(static))
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    printf("metadirective版本运行时间: %.3f秒\n", omp_get_wtime() - start);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("请指定运行模式: 1-串行 2-parallel for 3-metadirective 0-全部\n");
        return 1;
    }

    double *A = malloc(N*N*sizeof(double));
    double *B = malloc(N*N*sizeof(double));
    double *C = malloc(N*N*sizeof(double));

    init_matrix(A);
    init_matrix(B);

    int mode = atoi(argv[1]);
    double serial_time = 0;

    if (mode == 1 || mode == 0) {
        printf("\n=== 串行版本 ===\n");
        serial_matmul(A, B, C);
        serial_time = omp_get_wtime() - serial_time;
    }

    if (mode == 2 || mode == 0) {
        printf("\n=== parallel for版本 ===\n");
        double start = omp_get_wtime();
        parallel_for_matmul(A, B, C);
        double t = omp_get_wtime() - start;
        printf("加速比: %.2fX\n", serial_time/t);
    }

    if (mode == 3 || mode == 0) {
        printf("\n=== metadirective版本 ===\n");
        double start = omp_get_wtime();
        metadirective_matmul(A, B, C);
        double t = omp_get_wtime() - start;
        printf("加速比: %.2fX\n", serial_time/t);
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
EOF

# 编译设置
export OMP_NUM_THREADS=$THREADS
echo "当前线程数: $OMP_NUM_THREADS"

# 编译程序
gcc -fopenmp -O3 omp_metadirective_demo.c -o omp_demo

# 运行所有版本（模式0）
./omp_demo 0

# 清理
rm omp_metadirective_demo.c omp_demo