
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

// 并行版本1：使用parallel for的矩阵乘法
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

// 并行版本2：使用aligned的矩阵乘法
void matrix_multiply_aligned(double **A, double **B, double **C, int n) {
    // 创建一维数组用于设备数据传输
    double *A_flat = (double *)_mm_malloc(n * n * sizeof(double), 64);
    double *B_flat = (double *)_mm_malloc(n * n * sizeof(double), 64);
    double *C_flat = (double *)_mm_malloc(n * n * sizeof(double), 64);
    
    // 将二维数组转换为一维数组
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_flat[i * n + j] = A[i][j];
            B_flat[i * n + j] = B[i][j];
            C_flat[i * n + j] = 0.0;
        }
    }
    
    // 使用aligned进行矩阵乘法
    #pragma omp parallel for collapse(2)
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
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = C_flat[i * n + j];
        }
    }
    
    // 释放内存
    _mm_free(A_flat);
    _mm_free(B_flat);
    _mm_free(C_flat);
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
    
    // 测试parallel for版本
    printf("运行parallel for版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_parallel_for(A, B, C, N);
    end_time = omp_get_wtime();
    double parallel_for_time = end_time - start_time;
    printf("parallel for版本运行时间: %.4f 秒\n", parallel_for_time);
    printf("parallel for版本加速比: %.2f\n\n", serial_time / parallel_for_time);
    
    // 验证结果
    double parallel_for_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            parallel_for_checksum += C[i][j];
        }
    }
    printf("parallel for版本结果校验和: %.6e\n", parallel_for_checksum);
    printf("parallel for版本结果误差: %.6e\n\n", fabs(parallel_for_checksum - serial_checksum));
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
        }
    }
    
    // 测试aligned版本
    printf("运行aligned版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_aligned(A, B, C, N);
    end_time = omp_get_wtime();
    double aligned_time = end_time - start_time;
    printf("aligned版本运行时间: %.4f 秒\n", aligned_time);
    printf("aligned版本加速比: %.2f\n\n", serial_time / aligned_time);
    
    // 验证结果
    double aligned_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            aligned_checksum += C[i][j];
        }
    }
    printf("aligned版本结果校验和: %.6e\n", aligned_checksum);
    printf("aligned版本结果误差: %.6e\n\n", fabs(aligned_checksum - serial_checksum));
    
    // 性能总结
    printf("======= 性能总结 =======\n");
    printf("串行版本:             %.4f 秒 (加速比: 1.00)\n", serial_time);
    printf("parallel for版本:     %.4f 秒 (加速比: %.2f)\n", parallel_for_time, serial_time / parallel_for_time);
    printf("aligned版本:          %.4f 秒 (加速比: %.2f)\n", aligned_time, serial_time / aligned_time);
    
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
