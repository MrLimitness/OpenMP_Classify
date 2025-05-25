```bash
#!/bin/bash

# 设置默认线程数，可通过命令行参数修改，如 ./script.sh 144
THREADS=${1:-16}
MATRIX_SIZE=1024  # 矩阵大小，可根据需要调整

cat << EOF > omp_target_example.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N $MATRIX_SIZE
#define M $MATRIX_SIZE
#define K $MATRIX_SIZE

void matrix_mult_serial(float *A, float *B, float *C) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            float temp = A[i*K + k];
            for (int j = 0; j < M; j++) {
                C[i*M + j] += temp * B[k*M + j];
            }
        }
    }
}

void matrix_mult_parallel_for(float *A, float *B, float *C) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            float temp = A[i*K + k];
            for (int j = 0; j < M; j++) {
                C[i*M + j] += temp * B[k*M + j];
            }
        }
    }
}

void matrix_mult_target(float *A, float *B, float *C) {
    #pragma omp target data map(to: A[0:N*K], B[0:K*M]) map(tofrom: C[0:N*M])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i*K + k] * B[k*M + j];
                }
                C[i*M + j] = sum;
            }
        }
    }
}

int main() {
    float *A = malloc(N*K*sizeof(float));
    float *B = malloc(K*M*sizeof(float));
    float *C_serial = calloc(N*M, sizeof(float));
    float *C_parallel_for = calloc(N*M, sizeof(float));
    float *C_target = calloc(N*M, sizeof(float));

    // 初始化矩阵
    for (int i = 0; i < N*K; i++) A[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < K*M; i++) B[i] = (float)rand()/RAND_MAX;

    // 串行版本
    double start = omp_get_wtime();
    matrix_mult_serial(A, B, C_serial);
    double serial_time = omp_get_wtime() - start;
    printf("串行版本运行时间: %.3f秒\n", serial_time);

    // parallel for版本
    omp_set_num_threads($THREADS);
    start = omp_get_wtime();
    matrix_mult_parallel_for(A, B, C_parallel_for);
    double parallel_for_time = omp_get_wtime() - start;
    printf("Parallel for版本运行时间: %.3f秒 (加速比: %.2fx)\n", 
           parallel_for_time, serial_time/parallel_for_time);

    // target版本
    omp_set_num_threads($THREADS);
    start = omp_get_wtime();
    matrix_mult_target(A, B, C_target);
    double target_time = omp_get_wtime() - start;
    printf("Target版本运行时间: %.3f秒 (加速比: %.2fx)\n", 
           target_time, serial_time/target_time);

    free(A); free(B);
    free(C_serial); free(C_parallel_for); free(C_target);
    return 0;
}
EOF

gcc -fopenmp -O3 omp_target_example.c -o omp_target_example
./omp_target_example