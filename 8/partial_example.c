#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 1500 // 矩阵大小
#define CHUNK_SIZE 100

// 初始化矩阵
void initMatrix(double **A, double **B, double **C) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }
    }
}

// 矩阵乘法 - 串行版本
void matrixMultSerial(double **A, double **B, double **C) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double sum = 0.0;
            for(int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// 矩阵乘法 - parallel for版本
void matrixMultParallelFor(double **A, double **B, double **C, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double sum = 0.0;
            for(int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// 矩阵乘法 - partial版本1：使用partial和schedule调度
void matrixMultPartial1(double **A, double **B, double **C, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp for partial(k) schedule(dynamic, CHUNK_SIZE)
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                double sum = 0.0;
                for(int k = 0; k < N; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
}

// 矩阵乘法 - partial版本2：使用partial和collapse
void matrixMultPartial2(double **A, double **B, double **C, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp for partial(j,k) collapse(2)
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                double sum = 0.0;
                for(int k = 0; k < N; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
}

// 矩阵乘法 - partial版本3：结合nowait指令
void matrixMultPartial3(double **A, double **B, double **C, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp for partial(k) nowait
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                double sum = 0.0;
                for(int k = 0; k < N; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
}

// 矩阵乘法 - partial版本4：使用taskloop和partial
void matrixMultPartial4(double **A, double **B, double **C, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop partial(j) grainsize(CHUNK_SIZE)
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < N; j++) {
                    double sum = 0.0;
                    for(int k = 0; k < N; k++) {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int num_threads = 4; // 默认线程数
    
    // 如果提供了命令行参数，则使用它作为线程数
    if(argc > 1) {
        num_threads = atoi(argv[1]);
    }
    
    printf("使用线程数: %d\n", num_threads);
    
    // 分配矩阵内存
    double **A = (double **)malloc(N * sizeof(double *));
    double **B = (double **)malloc(N * sizeof(double *));
    double **C = (double **)malloc(N * sizeof(double *));
    
    for(int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }
    
    // 初始化随机数种子
    srand(time(NULL));
    
    // 初始化矩阵
    initMatrix(A, B, C);
    
    double start_time, end_time, serial_time;
    
    // 运行串行版本
    printf("\n运行串行版本...\n");
    start_time = omp_get_wtime();
    matrixMultSerial(A, B, C);
    end_time = omp_get_wtime();
    serial_time = end_time - start_time;
    printf("串行版本运行时间: %.2f 秒\n", serial_time);
    printf("加速比: 1.00\n");
    
    // 运行parallel for版本
    printf("\n运行parallel for版本...\n");
    start_time = omp_get_wtime();
    matrixMultParallelFor(A, B, C, num_threads);
    end_time = omp_get_wtime();
    printf("parallel for版本运行时间: %.2f 秒\n", end_time - start_time);
    printf("加速比: %.2f\n", serial_time / (end_time - start_time));
    
    // 运行partial版本1
    printf("\n运行partial版本1 (partial + schedule)...\n");
    start_time = omp_get_wtime();
    matrixMultPartial1(A, B, C, num_threads);
    end_time = omp_get_wtime();
    printf("partial版本1运行时间: %.2f 秒\n", end_time - start_time);
    printf("加速比: %.2f\n", serial_time / (end_time - start_time));
    
    // 运行partial版本2
    printf("\n运行partial版本2 (partial + collapse)...\n");
    start_time = omp_get_wtime();
    matrixMultPartial2(A, B, C, num_threads);
    end_time = omp_get_wtime();
    printf("partial版本2运行时间: %.2f 秒\n", end_time - start_time);
    printf("加速比: %.2f\n", serial_time / (end_time - start_time));
    
    // 运行partial版本3
    printf("\n运行partial版本3 (partial + nowait)...\n");
    start_time = omp_get_wtime();
    matrixMultPartial3(A, B, C, num_threads);
    end_time = omp_get_wtime();
    printf("partial版本3运行时间: %.2f 秒\n", end_time - start_time);
    printf("加速比: %.2f\n", serial_time / (end_time - start_time));
    
    // 运行partial版本4
    printf("\n运行partial版本4 (taskloop + partial)...\n");
    start_time = omp_get_wtime();
    matrixMultPartial4(A, B, C, num_threads);
    end_time = omp_get_wtime();
    printf("partial版本4运行时间: %.2f 秒\n", end_time - start_time);
    printf("加速比: %.2f\n", serial_time / (end_time - start_time));
    
    // 释放内存
    for(int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    
    return 0;
}