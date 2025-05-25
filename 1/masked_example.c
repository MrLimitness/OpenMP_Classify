```bash
#!/bin/bash

# 设置默认线程数和矩阵大小
THREADS=${1:-16}
SIZE=${2:-1024}

cat <<EOF > omp_masked_example.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N $SIZE

void matrix_mult_serial(double *A, double *B, double *C) {
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
}

void matrix_mult_parallel_for(double *A, double *B, double *C) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
}

void matrix_mult_masked(double *A, double *B, double *C) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < N; ++i) {
                #pragma omp task depend(out: C[i])
                {
                    #pragma omp masked
                    {
                        // 主线程专门处理最后一行的计算
                        if (i == N-1) {
                            for (int k = 0; k < N; ++k)
                                for (int j = 0; j < N; ++j)
                                    C[i*N + j] += A[i*N + k] * B[k*N + j];
                        }
                    }
                    
                    // 其他线程处理其他行
                    if (i != N-1) {
                        for (int k = 0; k < N; ++k)
                            for (int j = 0; j < N; ++j)
                                C[i*N + j] += A[i*N + k] * B[k*N + j];
                    }
                }
            }
        }
    }
}

int main() {
    double *A = malloc(N*N*sizeof(double));
    double *B = malloc(N*N*sizeof(double));
    double *C = malloc(N*N*sizeof(double));

    // 初始化矩阵
    for (int i = 0; i < N*N; ++i) {
        A[i] = (double)rand()/RAND_MAX;
        B[i] = (double)rand()/RAND_MAX;
        C[i] = 0.0;
    }

    double start, end;

    // 串行版本
    start = omp_get_wtime();
    matrix_mult_serial(A, B, C);
    end = omp_get_wtime();
    printf("串行版本运行时间: %.2f秒\n", end - start);
    double serial_time = end - start;

    // 重置结果矩阵
    for (int i = 0; i < N*N; ++i) C[i] = 0.0;

    // parallel for版本
    start = omp_get_wtime();
    matrix_mult_parallel_for(A, B, C);
    end = omp_get_wtime();
    printf("parallel for版本运行时间: %.2f秒\t加速比: %.2f\n", end - start, serial_time/(end - start));

    // 重置结果矩阵
    for (int i = 0; i < N*N; ++i) C[i] = 0.0;

    // masked版本
    start = omp_get_wtime();
    matrix_mult_masked(A, B, C);
    end = omp_get_wtime();
    printf("masked版本运行时间: %.2f秒\t加速比: %.2f\n", end - start, serial_time/(end - start));

    free(A);
    free(B);
    free(C);
    return 0;
}
EOF

# 编译和运行
gcc -fopenmp -O2 -o omp_masked_example omp_masked_example.c
echo "========================================"
echo "当前配置: 矩阵大小 ${SIZE}x${SIZE}, 线程数 ${THREADS}"
OMP_NUM_THREADS=$THREADS ./omp_masked_example
echo "========================================"

rm omp_masked_example omp_masked_example.c