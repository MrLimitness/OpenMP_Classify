#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

// 定义矩阵大小
#define N 4000

// 串行版本：标准矩阵乘法
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

// 并行版本1：使用parallel for和allocate的矩阵乘法
void matrix_multiply_parallel_allocate(double **A, double **B, double **C, int n) {
    double *A_flat, *B_flat, *C_flat;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // 在并行区域内分配内存
            #pragma omp allocate(A_flat, B_flat, C_flat) allocator(omp_large_cap_mem_alloc)
            A_flat = (double *)malloc(n * n * sizeof(double));
            B_flat = (double *)malloc(n * n * sizeof(double));
            C_flat = (double *)calloc(n * n, sizeof(double));
        }
        
        // 将二维数组转换为一维数组
        #pragma omp for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A_flat[i * n + j] = A[i][j];
                B_flat[i * n + j] = B[i][j];
            }
        }
        
        // 使用 parallel for 进行矩阵乘法
        #pragma omp for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A_flat[i * n + k] * B_flat[k * n + j];
                }
                C_flat[i * n + j] = sum;
            }
        }
        
        // 将结果从一维数组复制回二维数组
        #pragma omp for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = C_flat[i * n + j];
            }
        }
    }
    
    // 释放内存
    free(A_flat);
    free(B_flat);
    free(C_flat);
}

int main() {
    int num_threads = 16; // 可以修改为16/64/144等
    omp_set_num_threads(num_threads);
    
    printf("矩阵大小: %d x %d\n", N, N);
    printf("运行线程数: %d\n\n", num_threads);
    
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
    srand(12345);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }
    }
    
    // 测试串行版本
    printf("运行串行版本...\n");
    double start_time = omp_get_wtime();
    matrix_multiply_serial(A, B, C, N);
    double end_time = omp_get_wtime();
    double serial_time = end_time - start_time;
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    printf("串行版本加速比: 1.00\n\n");
    
    // 计算串行结果的校验和
    double serial_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            serial_checksum += C[i][j];
        }
    }
    printf("串行版本结果校验和: %.6e\n\n", serial_checksum);
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
        }
    }
    
    // 测试parallel for + allocate版本
    printf("运行parallel for + allocate版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_parallel_allocate(A, B, C, N);
    end_time = omp_get_wtime();
    double parallel_allocate_time = end_time - start_time;
    printf("parallel for + allocate版本运行时间: %.4f 秒\n", parallel_allocate_time);
    printf("parallel for + allocate版本加速比: %.2f\n\n", serial_time / parallel_allocate_time);
    
    // 验证结果
    double parallel_allocate_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            parallel_allocate_checksum += C[i][j];
        }
    }
    printf("parallel for + allocate版本结果校验和: %.6e\n", parallel_allocate_checksum);
    printf("parallel for + allocate版本结果误差: %.6e\n\n", fabs(parallel_allocate_checksum - serial_checksum));
    
    // 性能总结
    printf("======= 性能总结 =======\n");
    printf("串行版本:                  %.4f 秒 (加速比: 1.00)\n", serial_time);
    printf("parallel for + allocate版本: %.4f 秒 (加速比: %.2f)\n", parallel_allocate_time, serial_time / parallel_allocate_time);
    
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