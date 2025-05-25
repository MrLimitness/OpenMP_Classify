
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define SIZE 1500
#define ITERATIONS 5

// 工具函数：测量程序运行时间
double measure_time(void (*func)(double**, double**, double**), double **a, double **b, double **c) {
    double start = omp_get_wtime();
    func(a, b, c);
    double end = omp_get_wtime();
    return end - start;
}

// 为矩阵分配内存
double** allocate_matrix(int size) {
    double **matrix = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*)malloc(size * sizeof(double));
    }
    return matrix;
}

// 初始化矩阵
void initialize_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

// 释放矩阵内存
void free_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// 1. 串行版本
void matrix_multiply_serial(double **a, double **b, double **c) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                c[i][j] = 0.0;
                for (int k = 0; k < SIZE; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
}

// 2. 使用parallel for的版本
void matrix_multiply_parallel_for(double **a, double **b, double **c) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                c[i][j] = 0.0;
                for (int k = 0; k < SIZE; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
}

// 3. 使用num_threads版本1 - 基本使用
void matrix_multiply_num_threads(double **a, double **b, double **c) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for num_threads(omp_get_max_threads())
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                c[i][j] = 0.0;
                for (int k = 0; k < SIZE; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
}

// 4. 使用num_threads版本2 - 多级嵌套并行
void matrix_multiply_nested_num_threads(double **a, double **b, double **c) {
    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            printf("外层并行区域使用 %d 线程\n", omp_get_num_threads());
        }
        
        for (int iter = 0; iter < ITERATIONS; iter++) {
            #pragma omp for
            for (int i = 0; i < SIZE; i++) {
                #pragma omp parallel for num_threads(omp_get_max_threads() / 4)
                for (int j = 0; j < SIZE; j++) {
                    c[i][j] = 0.0;
                    for (int k = 0; k < SIZE; k++) {
                        c[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
        }
    }
}

// 5. 使用num_threads版本3 - 不同区域不同线程数
void matrix_multiply_varied_num_threads(double **a, double **b, double **c) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // 前半部分使用较多线程
        #pragma omp parallel for num_threads(omp_get_max_threads())
        for (int i = 0; i < SIZE/2; i++) {
            for (int j = 0; j < SIZE; j++) {
                c[i][j] = 0.0;
                for (int k = 0; k < SIZE; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        // 后半部分使用较少线程
        #pragma omp parallel for num_threads(omp_get_max_threads()/2)
        for (int i = SIZE/2; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                c[i][j] = 0.0;
                for (int k = 0; k < SIZE; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
}

// 6. 使用num_threads版本4 - 结合调度策略
void matrix_multiply_num_threads_schedule(double **a, double **b, double **c) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel num_threads(omp_get_max_threads())
        {
            #pragma omp for schedule(dynamic, SIZE/16)
            for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++) {
                    c[i][j] = 0.0;
                    for (int k = 0; k < SIZE; k++) {
                        c[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // 设置线程数（如果提供了命令行参数）
    if (argc > 1) {
        int num_threads = atoi(argv[1]);
        omp_set_num_threads(num_threads);
        printf("设置了 %d 个线程\n", num_threads);
    } else {
        printf("使用默认线程数: %d\n", omp_get_max_threads());
    }
    
    srand(time(NULL));
    
    // 分配矩阵内存
    double **a = allocate_matrix(SIZE);
    double **b = allocate_matrix(SIZE);
    double **c = allocate_matrix(SIZE);
    
    // 初始化矩阵
    initialize_matrix(a, SIZE);
    initialize_matrix(b, SIZE);
    
    printf("矩阵大小: %d x %d, 迭代次数: %d\n\n", SIZE, SIZE, ITERATIONS);
    
    // 测试串行版本
    double time_serial = measure_time(matrix_multiply_serial, a, b, c);
    printf("串行版本运行时间: %.4f 秒\n", time_serial);
    printf("加速比: %.2f\n\n", time_serial / time_serial);
    
    // 测试parallel for版本
    double time_parallel_for = measure_time(matrix_multiply_parallel_for, a, b, c);
    printf("parallel for版本运行时间: %.4f 秒\n", time_parallel_for);
    printf("加速比: %.2f\n\n", time_serial / time_parallel_for);
    
    // 测试num_threads版本1
    double time_num_threads = measure_time(matrix_multiply_num_threads, a, b, c);
    printf("num_threads版本1运行时间: %.4f 秒\n", time_num_threads);
    printf("加速比: %.2f\n\n", time_serial / time_num_threads);
    
    // 测试num_threads版本2
    double time_nested = measure_time(matrix_multiply_nested_num_threads, a, b, c);
    printf("num_threads嵌套版本运行时间: %.4f 秒\n", time_nested);
    printf("加速比: %.2f\n\n", time_serial / time_nested);
    
    // 测试num_threads版本3
    double time_varied = measure_time(matrix_multiply_varied_num_threads, a, b, c);
    printf("num_threads变化版本运行时间: %.4f 秒\n", time_varied);
    printf("加速比: %.2f\n\n", time_serial / time_varied);
    
    // 测试num_threads版本4
    double time_schedule = measure_time(matrix_multiply_num_threads_schedule, a, b, c);
    printf("num_threads调度版本运行时间: %.4f 秒\n", time_schedule);
    printf("加速比: %.2f\n\n", time_serial / time_schedule);
    
    // 释放矩阵内存
    free_matrix(a, SIZE);
    free_matrix(b, SIZE);
    free_matrix(c, SIZE);
    
    return 0;
}

// 使用默认线程数: 128
// 矩阵大小: 1500 x 1500, 迭代次数: 5

// 串行版本运行时间: 36.3203 秒
// 加速比: 1.00

// parallel for版本运行时间: 3.1977 秒
// 加速比: 11.36

// num_threads版本1运行时间: 2.8393 秒
// 加速比: 12.79

// 外层并行区域使用 4 线程
// num_threads嵌套版本运行时间: 12.7291 秒
// 加速比: 2.85

// num_threads变化版本运行时间: 4.5057 秒
// 加速比: 8.06

// num_threads调度版本运行时间: 7.2763 秒
// 加速比: 4.99
