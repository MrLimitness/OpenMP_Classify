#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// 矩阵大小，调整以获得约10秒的运行时间
#define N 1600

// 串行版本
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

// 使用parallel for版本
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

// 使用SIMD指令版本
void matrix_multiply_simd(double **A, double **B, double **C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            #pragma omp simd
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 结合parallel for和SIMD版本
void matrix_multiply_parallel_for_simd(double **A, double **B, double **C) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            #pragma omp simd
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 使用collapse优化嵌套循环和SIMD版本
void matrix_multiply_collapse_simd(double **A, double **B, double **C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            #pragma omp simd
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 使用SIMD指令和reduction版本
void matrix_multiply_simd_reduction(double **A, double **B, double **C) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int num_threads = 64; // 默认线程数
    
    // 如果提供了命令行参数，设置线程数
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    
    // 设置OpenMP线程数
    omp_set_num_threads(num_threads);
    printf("使用 %d 个线程进行测试\n", num_threads);
    
    // 分配内存
    double **A = (double **)malloc(N * sizeof(double *));
    double **B = (double **)malloc(N * sizeof(double *));
    double **C = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }
    
    // 初始化矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
        }
    }
    
    // 测试各个版本并计算加速比
    double start_time, end_time, serial_time;
    
    // 串行版本
    start_time = omp_get_wtime();
    matrix_multiply_serial(A, B, C);
    end_time = omp_get_wtime();
    serial_time = end_time - start_time;
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    
    // parallel for版本
    start_time = omp_get_wtime();
    matrix_multiply_parallel_for(A, B, C);
    end_time = omp_get_wtime();
    printf("parallel for版本运行时间: %.4f 秒, 加速比: %.2f\n", 
           end_time - start_time, serial_time / (end_time - start_time));
    
    // SIMD版本
    start_time = omp_get_wtime();
    matrix_multiply_simd(A, B, C);
    end_time = omp_get_wtime();
    printf("SIMD版本运行时间: %.4f 秒, 加速比: %.2f\n", 
           end_time - start_time, serial_time / (end_time - start_time));
    
    // parallel for + SIMD版本
    start_time = omp_get_wtime();
    matrix_multiply_parallel_for_simd(A, B, C);
    end_time = omp_get_wtime();
    printf("parallel for + SIMD版本运行时间: %.4f 秒, 加速比: %.2f\n", 
           end_time - start_time, serial_time / (end_time - start_time));
    
    // collapse + SIMD版本
    start_time = omp_get_wtime();
    matrix_multiply_collapse_simd(A, B, C);
    end_time = omp_get_wtime();
    printf("collapse + SIMD版本运行时间: %.4f 秒, 加速比: %.2f\n", 
           end_time - start_time, serial_time / (end_time - start_time));
    
    // SIMD + reduction版本
    start_time = omp_get_wtime();
    matrix_multiply_simd_reduction(A, B, C);
    end_time = omp_get_wtime();
    printf("SIMD + reduction版本运行时间: %.4f 秒, 加速比: %.2f\n", 
           end_time - start_time, serial_time / (end_time - start_time));
    
    // 释放内存
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

// 使用 16 个线程进行测试
// 串行版本运行时间: 7.5385 秒
// parallel for版本运行时间: 0.4990 秒, 加速比: 15.11
// SIMD版本运行时间: 6.8873 秒, 加速比: 1.09
// parallel for + SIMD版本运行时间: 0.6457 秒, 加速比: 11.68
// collapse + SIMD版本运行时间: 0.4679 秒, 加速比: 16.11
// SIMD + reduction版本运行时间: 0.4739 秒, 加速比: 15.91


// 使用 64 个线程进行测试
// 串行版本运行时间: 10.9732 秒
// parallel for版本运行时间: 0.6563 秒, 加速比: 16.72
// SIMD版本运行时间: 8.6013 秒, 加速比: 1.28
// parallel for + SIMD版本运行时间: 0.6582 秒, 加速比: 16.67
// collapse + SIMD版本运行时间: 0.6529 秒, 加速比: 16.81
// SIMD + reduction版本运行时间: 0.6525 秒, 加速比: 16.82

// 使用 256 个线程进行测试
// 串行版本运行时间: 6.3665 秒
// parallel for版本运行时间: 0.3260 秒, 加速比: 19.53
// SIMD版本运行时间: 7.4951 秒, 加速比: 0.85
// parallel for + SIMD版本运行时间: 0.3298 秒, 加速比: 19.31
// collapse + SIMD版本运行时间: 0.3255 秒, 加速比: 19.56
// SIMD + reduction版本运行时间: 0.3179 秒, 加速比: 20.03