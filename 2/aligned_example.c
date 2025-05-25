#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

#define N 1024
#define M 1024
#define K 1024

double serial_matrix_multiply(double A[N][M], double B[M][K], double C[N][K]) {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < M; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    gettimeofday(&end, NULL);
    double duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    return duration;
}

int main() {
    double A[N][M], B[M][K], C[N][K];
    double duration;

    // 初始化矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[i][j] = rand() % 10;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            B[i][j] = rand() % 10;
        }
    }

    // 串行版本
    duration = serial_matrix_multiply(A, B, C);
    printf("串行版本运行时间: %.6f 秒\n", duration);

    // parallel for 版本
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < M; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    duration = serial_matrix_multiply(A, B, C);
    printf("parallel for 版本运行时间: %.6f 秒\n", duration);

    // aligned 版本
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp simd aligned(A,B,C:64)
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < M; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    duration = serial_matrix_multiply(A, B, C);
    printf("aligned 版本运行时间: %.6f 秒\n", duration);

    return 0;
}