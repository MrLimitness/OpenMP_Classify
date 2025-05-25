#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#define N 1000 // 矩阵大小

// 用于计时
double get_time() {
    return omp_get_wtime();
}

// 打印矩阵（用于调试）
void print_matrix(double **matrix, int n) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("...\n");
}

// 初始化矩阵
void init_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

// 版本0：串行版本
void matrix_multiply_serial(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 版本1：使用parallel for
void matrix_multiply_parallel_for(double **A, double **B, double **C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 版本2：使用reverse子句
void matrix_multiply_reverse(double **A, double **B, double **C, int n) {
    #pragma omp parallel for schedule(static) reverse
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 版本3：使用reverse子句和collapse
void matrix_multiply_reverse_collapse(double **A, double **B, double **C, int n) {
    #pragma omp parallel for collapse(2) schedule(static) reverse
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 版本4：使用reverse和动态调度
void matrix_multiply_reverse_dynamic(double **A, double **B, double **C, int n) {
    #pragma omp parallel for schedule(dynamic, 16) reverse
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 版本5：使用reverse和嵌套并行
void matrix_multiply_reverse_nested(double **A, double **B, double **C, int n) {
    #pragma omp parallel
    {
        #pragma omp for schedule(static) reverse
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int thread_num = 4; // 默认线程数

    // 如果提供了命令行参数，使用它设置线程数
    if (argc > 1) {
        thread_num = atoi(argv[1]);
    }
    
    omp_set_num_threads(thread_num);
    
    printf("使用 %d 个线程进行计算\n", thread_num);
    
    // 为矩阵分配内存
    double **A = (double **)malloc(N * sizeof(double *));
    double **B = (double **)malloc(N * sizeof(double *));
    double **C = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }
    
    // 初始化矩阵
    srand(time(NULL));
    init_matrix(A, N);
    init_matrix(B, N);
    
    double start_time, end_time;
    double serial_time = 0.0;
    
    // 测试串行版本
    printf("\n===== 版本0：串行版本 =====\n");
    start_time = get_time();
    matrix_multiply_serial(A, B, C, N);
    end_time = get_time();
    serial_time = end_time - start_time;
    printf("运行时间: %.4f 秒\n", serial_time);
    printf("加速比: %.2f\n", 1.0);
    
    // 测试parallel for版本
    printf("\n===== 版本1：使用parallel for =====\n");
    start_time = get_time();
    matrix_multiply_parallel_for(A, B, C, N);
    end_time = get_time();
    printf("运行时间: %.4f 秒\n", end_time - start_time);
    printf("加速比: %.2f\n", serial_time / (end_time - start_time));
    
    // 测试reverse版本
    printf("\n===== 版本2：使用reverse子句 =====\n");
    start_time = get_time();
    matrix_multiply_reverse(A, B, C, N);
    end_time = get_time();
    printf("运行时间: %.4f 秒\n", end_time - start_time);
    printf("加速比: %.2f\n", serial_time / (end_time - start_time));
    
    // 测试reverse + collapse版本
    printf("\n===== 版本3：使用reverse子句和collapse =====\n");
    start_time = get_time();
    matrix_multiply_reverse_collapse(A, B, C, N);
    end_time = get_time();
    printf("运行时间: %.4f 秒\n", end_time - start_time);
    printf("加速比: %.2f\n", serial_time / (end_time - start_time));
    
    // 测试reverse + dynamic版本
    printf("\n===== 版本4：使用reverse和动态调度 =====\n");
    start_time = get_time();
    matrix_multiply_reverse_dynamic(A, B, C, N);
    end_time = get_time();
    printf("运行时间: %.4f 秒\n", end_time - start_time);
    printf("加速比: %.2f\n", serial_time / (end_time - start_time));
    
    // 测试reverse + nested版本
    printf("\n===== 版本5：使用reverse和嵌套并行 =====\n");
    start_time = get_time();
    matrix_multiply_reverse_nested(A, B, C, N);
    end_time = get_time();
    printf("运行时间: %.4f 秒\n", end_time - start_time);
    printf("加速比: %.2f\n", serial_time / (end_time - start_time));
    
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