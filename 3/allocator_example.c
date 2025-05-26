#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <numa.h>
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

// 并行版本1：使用parallel for和allocator的矩阵乘法
void matrix_multiply_parallel_allocator(double **A, double **B, double **C, int n) {
    #pragma omp parallel for collapse(2) \
        firstprivate(A, B, C) \
        allocator(omp_pteam_mempool_allocator)
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

// 并行版本2：使用parallel for和numa-aware allocator的矩阵乘法
void matrix_multiply_parallel_numa_allocator(double **A, double **B, double **C, int n) {
    #pragma omp parallel for collapse(2) \
        firstprivate(A, B, C) \
        allocator(omp_numa_allocator)
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
    
    // 测试parallel for + allocator版本
    printf("运行parallel for + allocator版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_parallel_allocator(A, B, C, N);
    end_time = omp_get_wtime();
    double parallel_allocator_time = end_time - start_time;
    printf("parallel for + allocator版本运行时间: %.4f 秒\n", parallel_allocator_time);
    printf("parallel for + allocator版本加速比: %.2f\n\n", serial_time / parallel_allocator_time);
    
    // 验证结果
    double parallel_allocator_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            parallel_allocator_checksum += C[i][j];
        }
    }
    printf("parallel for + allocator版本结果校验和: %.6e\n", parallel_allocator_checksum);
    printf("parallel for + allocator版本结果误差: %.6e\n\n", fabs(parallel_allocator_checksum - serial_checksum));
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
        }
    }
    
    // 测试parallel for + numa-aware allocator版本
    printf("运行parallel for + numa-aware allocator版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_parallel_numa_allocator(A, B, C, N);
    end_time = omp_get_wtime();
    double parallel_numa_allocator_time = end_time - start_time;
    printf("parallel for + numa-aware allocator版本运行时间: %.4f 秒\n", parallel_numa_allocator_time);
    printf("parallel for + numa-aware allocator版本加速比: %.2f\n\n", serial_time / parallel_numa_allocator_time);
    
    // 验证结果
    double parallel_numa_allocator_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            parallel_numa_allocator_checksum += C[i][j];
        }
    }
    printf("parallel for + numa-aware allocator版本结果校验和: %.6e\n", parallel_numa_allocator_checksum);
    printf("parallel for + numa-aware allocator版本结果误差: %.6e\n\n", fabs(parallel_numa_allocator_checksum - serial_checksum));
    
    // 性能总结
    printf("======= 性能总结 =======\n");
    printf("串行版本:                        %.4f 秒 (加速比: 1.00)\n", serial_time);
    printf("parallel for + allocator版本:    %.4f 秒 (加速比: %.2f)\n", parallel_allocator_time, serial_time / parallel_allocator_time);
    printf("parallel for + numa-aware allocator版本: %.4f 秒 (加速比: %.2f)\n", parallel_numa_allocator_time, serial_time / parallel_numa_allocator_time);
    
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
