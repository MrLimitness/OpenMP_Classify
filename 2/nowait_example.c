
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#define SIZE 2000

// 设置线程数 - 可以修改这个值
#define NUM_THREADS 128

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

void matrix_multiply_nowait_v1(double **A, double **B, double **C, int n) {
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        #pragma omp for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = C[i][j] * 1.001;
            }
        }
    }
}

void matrix_multiply_nowait_v2(double **A, double **B, double **C, int n) {
    #pragma omp parallel
    {
        #pragma omp for nowait schedule(dynamic)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        #pragma omp single nowait
        {
            printf("矩阵计算完成，开始后处理\n");
        }
        
        #pragma omp for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = C[i][j] * 1.001 + 0.001;
            }
        }
    }
}

void matrix_multiply_nowait_v3(double **A, double **B, double **C, int n) {
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < n/2; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        #pragma omp for nowait
        for (int i = n/2; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        #pragma omp barrier
        
        #pragma omp for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = C[i][j] * 1.002;
            }
        }
    }
}

double** allocate_matrix(int n) {
    double **matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

void free_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void initialize_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (double)(rand() % 100) / 10.0;
        }
    }
}

int main() {
    omp_set_num_threads(NUM_THREADS);
    
    printf("OpenMP nowait指令性能测试\n");
    printf("矩阵大小: %d x %d\n", SIZE, SIZE);
    printf("线程数: %d\n", NUM_THREADS);
    printf("================================\n");
    
    double **A = allocate_matrix(SIZE);
    double **B = allocate_matrix(SIZE);
    double **C = allocate_matrix(SIZE);
    
    initialize_matrix(A, SIZE);
    initialize_matrix(B, SIZE);
    
    double start_time, end_time;
    double serial_time, parallel_time, nowait_time1, nowait_time2, nowait_time3;
    
    // 串行版本
    printf("运行串行版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_serial(A, B, C, SIZE);
    end_time = omp_get_wtime();
    serial_time = end_time - start_time;
    printf("串行版本运行时间: %.3f 秒\n", serial_time);
    
    // parallel for版本
    printf("运行parallel for版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_parallel_for(A, B, C, SIZE);
    end_time = omp_get_wtime();
    parallel_time = end_time - start_time;
    printf("parallel for版本运行时间: %.3f 秒\n", parallel_time);
    printf("parallel for版本加速比: %.2fx\n", serial_time / parallel_time);
    
    // nowait版本1
    printf("运行nowait版本1...\n");
    start_time = omp_get_wtime();
    matrix_multiply_nowait_v1(A, B, C, SIZE);
    end_time = omp_get_wtime();
    nowait_time1 = end_time - start_time;
    printf("nowait版本1运行时间: %.3f 秒\n", nowait_time1);
    printf("nowait版本1加速比: %.2fx\n", serial_time / nowait_time1);
    
    // nowait版本2
    printf("运行nowait版本2...\n");
    start_time = omp_get_wtime();
    matrix_multiply_nowait_v2(A, B, C, SIZE);
    end_time = omp_get_wtime();
    nowait_time2 = end_time - start_time;
    printf("nowait版本2运行时间: %.3f 秒\n", nowait_time2);
    printf("nowait版本2加速比: %.2fx\n", serial_time / nowait_time2);
    
    // nowait版本3
    printf("运行nowait版本3...\n");
    start_time = omp_get_wtime();
    matrix_multiply_nowait_v3(A, B, C, SIZE);
    end_time = omp_get_wtime();
    nowait_time3 = end_time - start_time;
    printf("nowait版本3运行时间: %.3f 秒\n", nowait_time3);
    printf("nowait版本3加速比: %.2fx\n", serial_time / nowait_time3);
    
    printf("================================\n");
    printf("性能对比总结:\n");
    printf("串行版本: %.3f 秒 (加速比: 1.00x)\n", serial_time);
    printf("parallel for版本: %.3f 秒 (加速比: %.2fx)\n", parallel_time, serial_time / parallel_time);
    printf("nowait版本1: %.3f 秒 (加速比: %.2fx)\n", nowait_time1, serial_time / nowait_time1);
    printf("nowait版本2: %.3f 秒 (加速比: %.2fx)\n", nowait_time2, serial_time / nowait_time2);
    printf("nowait版本3: %.3f 秒 (加速比: %.2fx)\n", nowait_time3, serial_time / nowait_time3);
    
    free_matrix(A, SIZE);
    free_matrix(B, SIZE);
    free_matrix(C, SIZE);
    
    return 0;
}
// 矩阵大小: 2000 x 2000
// 线程数: 16
// ================================
// 运行串行版本...
// 串行版本运行时间: 44.713 秒
// 运行parallel for版本...
// parallel for版本运行时间: 2.210 秒
// parallel for版本加速比: 20.23x
// 运行nowait版本1...
// nowait版本1运行时间: 3.416 秒
// nowait版本1加速比: 13.09x
// 运行nowait版本2...
// 矩阵计算完成，开始后处理
// nowait版本2运行时间: 2.182 秒
// nowait版本2加速比: 20.49x
// 运行nowait版本3...
// nowait版本3运行时间: 4.241 秒
// nowait版本3加速比: 10.54x
// ================================
// 性能对比总结:
// 串行版本: 44.713 秒 (加速比: 1.00x)
// parallel for版本: 2.210 秒 (加速比: 20.23x)
// nowait版本1: 3.416 秒 (加速比: 13.09x)
// nowait版本2: 2.182 秒 (加速比: 20.49x)
// nowait版本3: 4.241 秒 (加速比: 10.54x)


// OpenMP nowait指令性能测试
// 矩阵大小: 2000 x 2000
// 线程数: 64
// ================================
// 运行串行版本...
// 串行版本运行时间: 43.095 秒
// 运行parallel for版本...
// parallel for版本运行时间: 1.567 秒
// parallel for版本加速比: 27.50x
// 运行nowait版本1...
// nowait版本1运行时间: 1.066 秒
// nowait版本1加速比: 40.44x
// 运行nowait版本2...
// 矩阵计算完成，开始后处理
// nowait版本2运行时间: 1.011 秒
// nowait版本2加速比: 42.61x
// 运行nowait版本3...
// nowait版本3运行时间: 1.448 秒
// nowait版本3加速比: 29.76x
// ================================
// 性能对比总结:
// 串行版本: 43.095 秒 (加速比: 1.00x)
// parallel for版本: 1.567 秒 (加速比: 27.50x)
// nowait版本1: 1.066 秒 (加速比: 40.44x)
// nowait版本2: 1.011 秒 (加速比: 42.61x)
// nowait版本3: 1.448 秒 (加速比: 29.76x)



// OpenMP nowait指令性能测试
// 矩阵大小: 2000 x 2000
// 线程数: 128
// ================================
// 运行串行版本...
// 串行版本运行时间: 46.452 秒
// 运行parallel for版本...
// parallel for版本运行时间: 1.178 秒
// parallel for版本加速比: 39.43x
// 运行nowait版本1...
// nowait版本1运行时间: 0.988 秒
// nowait版本1加速比: 46.99x
// 运行nowait版本2...
// 矩阵计算完成，开始后处理
// nowait版本2运行时间: 0.678 秒
// nowait版本2加速比: 68.55x
// 运行nowait版本3...
// nowait版本3运行时间: 0.935 秒
// nowait版本3加速比: 49.67x
// ================================
// 性能对比总结:
// 串行版本: 46.452 秒 (加速比: 1.00x)
// parallel for版本: 1.178 秒 (加速比: 39.43x)
// nowait版本1: 0.988 秒 (加速比: 46.99x)
// nowait版本2: 0.678 秒 (加速比: 68.55x)
// nowait版本3: 0.935 秒 (加速比: 49.67x)