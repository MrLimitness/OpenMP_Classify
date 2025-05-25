#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#define N 1024
#define ITERATIONS 10

// 串行版本的矩阵乘法
void matrix_multiply_serial(double **A, double **B, double **C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// parallel for版本的矩阵乘法
void matrix_multiply_parallel_for(double **A, double **B, double **C) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// proc_bind(close)版本的矩阵乘法
void matrix_multiply_proc_bind_close(double **A, double **B, double **C) {
    #pragma omp parallel for proc_bind(close)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// proc_bind(spread)版本的矩阵乘法
void matrix_multiply_proc_bind_spread(double **A, double **B, double **C) {
    #pragma omp parallel for proc_bind(spread)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// proc_bind(master)版本的矩阵乘法
void matrix_multiply_proc_bind_master(double **A, double **B, double **C) {
    #pragma omp parallel for proc_bind(master)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 使用proc_bind(close)和分块优化的矩阵乘法
void matrix_multiply_proc_bind_close_blocked(double **A, double **B, double **C) {
    #define BLOCK_SIZE 64
    
    #pragma omp parallel proc_bind(close)
    {
        #pragma omp for
        for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                    for (int i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                        for (int j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
                            if (kk == 0) C[i][j] = 0.0;
                            for (int k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                                C[i][j] += A[i][k] * B[k][j];
                            }
                        }
                    }
                }
            }
        }
    }
}

// 分配矩阵内存
double** allocate_matrix(int n) {
    double **matrix = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double *)malloc(n * sizeof(double));
    }
    return matrix;
}

// 初始化矩阵
void initialize_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
}

// 释放矩阵内存
void free_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// 验证两个矩阵是否相等
int verify_matrices(double **C1, double **C2, int n) {
    double epsilon = 1e-6;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(C1[i][j] - C2[i][j]) > epsilon) {
                return 0;
            }
        }
    }
    return 1;
}

// 清空矩阵
void clear_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        memset(matrix[i], 0, n * sizeof(double));
    }
}

int main(int argc, char **argv) {
    int num_threads = 64; // 默认线程数
    
    // 解析命令行参数
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    
    // 设置线程数
    omp_set_num_threads(num_threads);
    
    // 输出线程数
    printf("使用线程数: %d\n", num_threads);
    
    // 分配矩阵内存
    double **A = allocate_matrix(N);
    double **B = allocate_matrix(N);
    double **C_serial = allocate_matrix(N);
    double **C_result = allocate_matrix(N);
    
    // 初始化矩阵
    srand(42);
    initialize_matrix(A, N);
    initialize_matrix(B, N);
    
    // 用于计时的变量
    double start_time, end_time;
    double serial_time, parallel_time, speedup;
    
    printf("矩阵大小: %d x %d\n", N, N);
    printf("重复次数: %d\n\n", ITERATIONS);

    // 串行版本
    printf("运行串行版本...\n");
    start_time = omp_get_wtime();
    for (int i = 0; i < ITERATIONS; i++) {
        matrix_multiply_serial(A, B, C_serial);
    }
    end_time = omp_get_wtime();
    serial_time = end_time - start_time;
    printf("串行版本运行时间: %.2f 秒\n", serial_time);
    printf("加速比: 1.00\n\n");

    // parallel for版本
    printf("运行parallel for版本...\n");
    clear_matrix(C_result, N);
    start_time = omp_get_wtime();
    for (int i = 0; i < ITERATIONS; i++) {
        matrix_multiply_parallel_for(A, B, C_result);
    }
    end_time = omp_get_wtime();
    parallel_time = end_time - start_time;
    speedup = serial_time / parallel_time;
    printf("parallel for版本运行时间: %.2f 秒\n", parallel_time);
    printf("加速比: %.2f\n", speedup);
    if (!verify_matrices(C_serial, C_result, N)) {
        printf("警告: parallel for版本结果不正确!\n");
    }
    printf("\n");

    // proc_bind(close)版本
    printf("运行proc_bind(close)版本...\n");
    clear_matrix(C_result, N);
    start_time = omp_get_wtime();
    for (int i = 0; i < ITERATIONS; i++) {
        matrix_multiply_proc_bind_close(A, B, C_result);
    }
    end_time = omp_get_wtime();
    parallel_time = end_time - start_time;
    speedup = serial_time / parallel_time;
    printf("proc_bind(close)版本运行时间: %.2f 秒\n", parallel_time);
    printf("加速比: %.2f\n", speedup);
    if (!verify_matrices(C_serial, C_result, N)) {
        printf("警告: proc_bind(close)版本结果不正确!\n");
    }
    printf("\n");

    // proc_bind(spread)版本
    printf("运行proc_bind(spread)版本...\n");
    clear_matrix(C_result, N);
    start_time = omp_get_wtime();
    for (int i = 0; i < ITERATIONS; i++) {
        matrix_multiply_proc_bind_spread(A, B, C_result);
    }
    end_time = omp_get_wtime();
    parallel_time = end_time - start_time;
    speedup = serial_time / parallel_time;
    printf("proc_bind(spread)版本运行时间: %.2f 秒\n", parallel_time);
    printf("加速比: %.2f\n", speedup);
    if (!verify_matrices(C_serial, C_result, N)) {
        printf("警告: proc_bind(spread)版本结果不正确!\n");
    }
    printf("\n");

    // proc_bind(master)版本
    printf("运行proc_bind(master)版本...\n");
    clear_matrix(C_result, N);
    start_time = omp_get_wtime();
    for (int i = 0; i < ITERATIONS; i++) {
        matrix_multiply_proc_bind_master(A, B, C_result);
    }
    end_time = omp_get_wtime();
    parallel_time = end_time - start_time;
    speedup = serial_time / parallel_time;
    printf("proc_bind(master)版本运行时间: %.2f 秒\n", parallel_time);
    printf("加速比: %.2f\n", speedup);
    if (!verify_matrices(C_serial, C_result, N)) {
        printf("警告: proc_bind(master)版本结果不正确!\n");
    }
    printf("\n");

    // proc_bind(close) + 分块优化版本
    printf("运行proc_bind(close) + 分块优化版本...\n");
    clear_matrix(C_result, N);
    start_time = omp_get_wtime();
    for (int i = 0; i < ITERATIONS; i++) {
        matrix_multiply_proc_bind_close_blocked(A, B, C_result);
    }
    end_time = omp_get_wtime();
    parallel_time = end_time - start_time;
    speedup = serial_time / parallel_time;
    printf("proc_bind(close) + 分块优化版本运行时间: %.2f 秒\n", parallel_time);
    printf("加速比: %.2f\n", speedup);
    if (!verify_matrices(C_serial, C_result, N)) {
        printf("警告: proc_bind(close) + 分块优化版本结果不正确!\n");
    }
    printf("\n");

    // 释放内存
    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C_serial, N);
    free_matrix(C_result, N);
    
    return 0;
}

// 使用线程数: 16
// 矩阵大小: 1024 x 1024
// 重复次数: 10

// 运行串行版本...
// 串行版本运行时间: 20.64 秒
// 加速比: 1.00

// 运行parallel for版本...
// parallel for版本运行时间: 3.17 秒
// 加速比: 6.50

// 运行proc_bind(close)版本...
// proc_bind(close)版本运行时间: 3.27 秒
// 加速比: 6.32

// 运行proc_bind(spread)版本...
// proc_bind(spread)版本运行时间: 4.83 秒
// 加速比: 4.27

// 运行proc_bind(master)版本...
// proc_bind(master)版本运行时间: 2.93 秒
// 加速比: 7.05

// 运行proc_bind(close) + 分块优化版本...
// proc_bind(close) + 分块优化版本运行时间: 2.81 秒
// 加速比: 7.34

// 使用线程数: 64
// 矩阵大小: 1024 x 1024
// 重复次数: 10

// 运行串行版本...
// 串行版本运行时间: 22.42 秒
// 加速比: 1.00

// 运行parallel for版本...
// parallel for版本运行时间: 1.28 秒
// 加速比: 17.46

// 运行proc_bind(close)版本...
// proc_bind(close)版本运行时间: 1.60 秒
// 加速比: 14.04

// 运行proc_bind(spread)版本...
// proc_bind(spread)版本运行时间: 2.02 秒
// 加速比: 11.09

// 运行proc_bind(master)版本...
// proc_bind(master)版本运行时间: 2.13 秒
// 加速比: 10.52

// 运行proc_bind(close) + 分块优化版本...
// proc_bind(close) + 分块优化版本运行时间: 1.72 秒
// 加速比: 13.01