#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define SIZE 2000

void initialize_matrix(double **A, double **B, double **C, int n) {
    *A = (double*)malloc(n * n * sizeof(double));
    *B = (double*)malloc(n * n * sizeof(double));
    *C = (double*)malloc(n * n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*A)[i * n + j] = (double)rand() / RAND_MAX;
            (*B)[i * n + j] = (double)rand() / RAND_MAX;
            (*C)[i * n + j] = 0.0;
        }
    }
}

void free_matrix(double *A, double *B, double *C) {
    free(A);
    free(B);
    free(C);
}

// 版本1：串行版本
void matrix_multiply_serial(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// 版本2：parallel for版本
void matrix_multiply_parallel_for(double *A, double *B, double *C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// 版本3：使用interchange指令优化循环顺序 (i,j,k) -> (i,k,j)
void matrix_multiply_interchange_1(double *A, double *B, double *C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        #pragma omp interchange
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// 版本4：使用interchange指令优化循环顺序 (i,j,k) -> (j,i,k)
void matrix_multiply_interchange_2(double *A, double *B, double *C, int n) {
    #pragma omp parallel for
    #pragma omp interchange
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// 版本5：使用interchange指令和collapse指令组合
void matrix_multiply_interchange_collapse(double *A, double *B, double *C, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        #pragma omp interchange
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// 版本6：使用interchange指令和schedule调度策略
void matrix_multiply_interchange_schedule(double *A, double *B, double *C, int n) {
    #pragma omp parallel for schedule(dynamic, 10)
    for (int i = 0; i < n; i++) {
        #pragma omp interchange
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    double *A, *B, *C;
    double start_time, end_time;
    int n = SIZE;
    int num_threads = 64; // 默认线程数
    
    // 检查命令行参数中是否指定了线程数
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    
    omp_set_num_threads(num_threads);
    
    printf("矩阵大小: %d x %d\n", n, n);
    printf("线程数: %d\n", num_threads);
    
    srand(time(NULL));
    
    // 版本1：串行版本
    initialize_matrix(&A, &B, &C, n);
    start_time = omp_get_wtime();
    matrix_multiply_serial(A, B, C, n);
    end_time = omp_get_wtime();
    double serial_time = end_time - start_time;
    printf("串行版本运行时间: %.6f 秒\n", serial_time);
    printf("串行版本加速比: 1.00\n");
    free_matrix(A, B, C);
    
    // 版本2：parallel for版本
    initialize_matrix(&A, &B, &C, n);
    start_time = omp_get_wtime();
    matrix_multiply_parallel_for(A, B, C, n);
    end_time = omp_get_wtime();
    double parallel_for_time = end_time - start_time;
    printf("Parallel for版本运行时间: %.6f 秒\n", parallel_for_time);
    printf("Parallel for版本加速比: %.2f\n", serial_time / parallel_for_time);
    free_matrix(A, B, C);
    
    // 版本3：使用interchange指令优化循环顺序 (i,j,k) -> (i,k,j)
    initialize_matrix(&A, &B, &C, n);
    start_time = omp_get_wtime();
    matrix_multiply_interchange_1(A, B, C, n);
    end_time = omp_get_wtime();
    double interchange_1_time = end_time - start_time;
    printf("Interchange版本1运行时间: %.6f 秒\n", interchange_1_time);
    printf("Interchange版本1加速比: %.2f\n", serial_time / interchange_1_time);
    free_matrix(A, B, C);
    
    // 版本4：使用interchange指令优化循环顺序 (i,j,k) -> (j,i,k)
    initialize_matrix(&A, &B, &C, n);
    start_time = omp_get_wtime();
    matrix_multiply_interchange_2(A, B, C, n);
    end_time = omp_get_wtime();
    double interchange_2_time = end_time - start_time;
    printf("Interchange版本2运行时间: %.6f 秒\n", interchange_2_time);
    printf("Interchange版本2加速比: %.2f\n", serial_time / interchange_2_time);
    free_matrix(A, B, C);
    
    // 版本5：使用interchange指令和collapse指令组合
    initialize_matrix(&A, &B, &C, n);
    start_time = omp_get_wtime();
    matrix_multiply_interchange_collapse(A, B, C, n);
    end_time = omp_get_wtime();
    double interchange_collapse_time = end_time - start_time;
    printf("Interchange+Collapse版本运行时间: %.6f 秒\n", interchange_collapse_time);
    printf("Interchange+Collapse版本加速比: %.2f\n", serial_time / interchange_collapse_time);
    free_matrix(A, B, C);
    
    // 版本6：使用interchange指令和schedule调度策略
    initialize_matrix(&A, &B, &C, n);
    start_time = omp_get_wtime();
    matrix_multiply_interchange_schedule(A, B, C, n);
    end_time = omp_get_wtime();
    double interchange_schedule_time = end_time - start_time;
    printf("Interchange+Schedule版本运行时间: %.6f 秒\n", interchange_schedule_time);
    printf("Interchange+Schedule版本加速比: %.2f\n", serial_time / interchange_schedule_time);
    free_matrix(A, B, C);
    
    return 0;
}

// 线程数: 16
// 串行版本运行时间: 37.792410 秒
// 串行版本加速比: 1.00
// Parallel for版本运行时间: 4.126950 秒
// Parallel for版本加速比: 9.16
// Interchange版本1运行时间: 3.438415 秒
// Interchange版本1加速比: 10.99
// Interchange版本2运行时间: 4.174502 秒
// Interchange版本2加速比: 9.05
// Interchange+Collapse版本运行时间: 3.632303 秒
// Interchange+Collapse版本加速比: 10.40
// Interchange+Schedule版本运行时间: 2.086982 秒
// Interchange+Schedule版本加速比: 18.11

// 线程数: 64
// 串行版本运行时间: 30.147944 秒
// 串行版本加速比: 1.00
// Parallel for版本运行时间: 1.668567 秒
// Parallel for版本加速比: 18.07
// Interchange版本1运行时间: 1.641925 秒
// Interchange版本1加速比: 18.36
// Interchange版本2运行时间: 1.383861 秒
// Interchange版本2加速比: 21.79
// Interchange+Collapse版本运行时间: 1.022791 秒
// Interchange+Collapse版本加速比: 29.48
// Interchange+Schedule版本运行时间: 0.944711 秒
// Interchange+Schedule版本加速比: 31.91