#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#define N 1500  // 定义矩阵大小

// 串行版本
void matrix_multiply_serial(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// parallel for
void matrix_multiply_parallel_for(double **A, double **B, double **C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// 手动tile优化
void matrix_multiply_manual_tile(double **A, double **B, double **C, int n) {
    const int tile_size = 32;
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += tile_size) {
        for (int jj = 0; jj < n; jj += tile_size) {
            for (int kk = 0; kk < n; kk += tile_size) {
                for (int i = ii; i < ii + tile_size && i < n; i++) {
                    for (int j = jj; j < jj + tile_size && j < n; j++) {
                        double sum = (kk == 0) ? 0.0 : C[i][j];
                        for (int k = kk; k < kk + tile_size && k < n; k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
}

int main() {
    int num_threads = 64;
    omp_set_num_threads(num_threads);

    printf("\n矩阵大小: %d x %d\n", N, N);
    printf("运行线程数: %d\n\n", num_threads);

    double **A = (double **)malloc(N * sizeof(double *));
    double **B = (double **)malloc(N * sizeof(double *));
    double **C = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }
    }

    printf("运行串行版本...\n");
    double start = omp_get_wtime();
    matrix_multiply_serial(A, B, C, N);
    double end = omp_get_wtime();
    double serial_time = end - start;
    printf("串行运行时间: %.4f 秒\n", serial_time);

    double checksum_serial = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            checksum_serial += C[i][j];
    printf("串行结果校验和: %.6e\n\n", checksum_serial);

    // 初始化C
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = 0.0;

    printf("运行parallel for版本...\n");
    start = omp_get_wtime();
    matrix_multiply_parallel_for(A, B, C, N);
    end = omp_get_wtime();
    double parallel_time = end - start;
    printf("parallel for运行时间: %.4f 秒\n", parallel_time);
    printf("加速比: %.2f\n", serial_time / parallel_time);

    double checksum_parallel = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            checksum_parallel += C[i][j];
    printf("parallel for结果校验和: %.6e\n", checksum_parallel);
    printf("误差: %.6e\n\n", fabs(checksum_parallel - checksum_serial));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = 0.0;

    printf("运行手动tile版本...\n");
    start = omp_get_wtime();
    matrix_multiply_manual_tile(A, B, C, N);
    end = omp_get_wtime();
    double tile_time = end - start;
    printf("手动tile运行时间: %.4f 秒\n", tile_time);
    printf("加速比: %.2f\n", serial_time / tile_time);

    double checksum_tile = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            checksum_tile += C[i][j];
    printf("手动tile结果校验和: %.6e\n", checksum_tile);
    printf("误差: %.6e\n", fabs(checksum_tile - checksum_serial));

    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}

// 矩阵大小: 1500 x 1500
// 运行线程数: 16

// 运行串行版本...
// 串行运行时间: 7.4599 秒
// 串行结果校验和: 8.425293e+08

// 运行parallel for版本...
// parallel for运行时间: 1.0390 秒
// 加速比: 7.18
// parallel for结果校验和: 8.425293e+08
// 误差: 0.000000e+00

// 运行手动tile版本...
// 手动tile运行时间: 0.9632 秒
// 加速比: 7.74
// 手动tile结果校验和: 8.425293e+08
// 误差: 0.000000e+00


// 矩阵大小: 1500 x 1500
// 运行线程数: 64

// 运行串行版本...
// 串行运行时间: 9.9397 秒
// 串行结果校验和: 8.435140e+08

// 运行parallel for版本...
// parallel for运行时间: 0.4287 秒
// 加速比: 23.19
// parallel for结果校验和: 8.435140e+08
// 误差: 0.000000e+00

// 运行手动tile版本...
// 手动tile运行时间: 0.2347 秒
// 加速比: 42.35
// 手动tile结果校验和: 8.435140e+08
// 误差: 0.000000e+00