#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#define MATRIX_SIZE 1024
#define ITERATIONS 5

// 工具函数：打印运行时间和加速比
void print_performance(const char* version_name, double serial_time, double version_time) {
    double speedup = serial_time / version_time;
    printf("%s运行时间: %.4f秒, 加速比: %.2f\n", version_name, version_time, speedup);
}

// 初始化矩阵
void init_matrices(double** A, double** B, double** C, int size) {
    *A = (double*)malloc(size * size * sizeof(double));
    *B = (double*)malloc(size * size * sizeof(double));
    *C = (double*)malloc(size * size * sizeof(double));

    for (int i = 0; i < size * size; i++) {
        (*A)[i] = (double)rand() / RAND_MAX;
        (*B)[i] = (double)rand() / RAND_MAX;
        (*C)[i] = 0.0;
    }
}

// 释放矩阵内存
void free_matrices(double* A, double* B, double* C) {
    free(A);
    free(B);
    free(C);
}

// 串行版本
double serial_version(double* A, double* B, double* C, int size) {
    double start_time = omp_get_wtime();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        memset(C, 0, size * size * sizeof(double));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }
    return omp_get_wtime() - start_time;
}

// parallel for版本
double parallel_for_version(double* A, double* B, double* C, int size, int num_threads) {
    omp_set_num_threads(num_threads);
    double start_time = omp_get_wtime();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        memset(C, 0, size * size * sizeof(double));
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }
    return omp_get_wtime() - start_time;
}

// collapse版本1
double collapse_version1(double* A, double* B, double* C, int size, int num_threads) {
    omp_set_num_threads(num_threads);
    double start_time = omp_get_wtime();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        memset(C, 0, size * size * sizeof(double));
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }
    return omp_get_wtime() - start_time;
}

// collapse版本2
double collapse_version2(double* A, double* B, double* C, int size, int num_threads) {
    omp_set_num_threads(num_threads);
    double start_time = omp_get_wtime();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        memset(C, 0, size * size * sizeof(double));
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double sum = 0.0;
                for (int k = 0; k < size; k++) {
                    sum += A[i * size + k] * B[k * size + j];
                }
                C[i * size + j] = sum;
            }
        }
    }
    return omp_get_wtime() - start_time;
}

// collapse版本3：分块
double collapse_version3(double* A, double* B, double* C, int size, int num_threads) {
    omp_set_num_threads(num_threads);
    int block_size = 32;
    double start_time = omp_get_wtime();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        memset(C, 0, size * size * sizeof(double));
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < size; i += block_size) {
            for (int j = 0; j < size; j += block_size) {
                for (int ii = i; ii < i + block_size && ii < size; ii++) {
                    for (int jj = j; jj < j + block_size && jj < size; jj++) {
                        double sum = 0.0;
                        for (int k = 0; k < size; k++) {
                            sum += A[ii * size + k] * B[k * size + jj];
                        }
                        C[ii * size + jj] = sum;
                    }
                }
            }
        }
    }
    return omp_get_wtime() - start_time;
}

// collapse版本4：块外+块内collapse
double collapse_version4(double* A, double* B, double* C, int size, int num_threads) {
    omp_set_num_threads(num_threads);
    int block_size = 32;
    double start_time = omp_get_wtime();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        memset(C, 0, size * size * sizeof(double));
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < size; i += block_size) {
            for (int j = 0; j < size; j += block_size) {
                for (int ii = i; ii < size && ii < i + block_size; ii++) {
                    for (int jj = j; jj < size && jj < j + block_size; jj++) {
                        double sum = 0.0;
                        for (int k = 0; k < size; k++) {
                            sum += A[ii * size + k] * B[k * size + jj];
                        }
                        C[ii * size + jj] = sum;
                    }
                }
            }
        }
    }
    return omp_get_wtime() - start_time;
}

// 主函数
int main(int argc, char* argv[]) {
    int num_threads = 64;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }

    printf("使用线程数: %d\n", num_threads);
    srand(time(NULL));

    double *A, *B, *C;
    init_matrices(&A, &B, &C, MATRIX_SIZE);
    printf("矩阵大小: %d x %d, 迭代次数: %d\n", MATRIX_SIZE, MATRIX_SIZE, ITERATIONS);

    printf("\n运行串行版本...\n");
    double serial_time = serial_version(A, B, C, MATRIX_SIZE);
    print_performance("串行版本", serial_time, serial_time);

    printf("\n运行parallel for版本...\n");
    double time1 = parallel_for_version(A, B, C, MATRIX_SIZE, num_threads);
    print_performance("parallel for版本", serial_time, time1);

    printf("\n运行collapse版本1...\n");
    double time2 = collapse_version1(A, B, C, MATRIX_SIZE, num_threads);
    print_performance("collapse版本1", serial_time, time2);

    printf("\n运行collapse版本2...\n");
    double time3 = collapse_version2(A, B, C, MATRIX_SIZE, num_threads);
    print_performance("collapse版本2", serial_time, time3);

    printf("\n运行collapse版本3...\n");
    double time4 = collapse_version3(A, B, C, MATRIX_SIZE, num_threads);
    print_performance("collapse版本3", serial_time, time4);

    printf("\n运行collapse版本4...\n");
    double time5 = collapse_version4(A, B, C, MATRIX_SIZE, num_threads);
    print_performance("collapse版本4", serial_time, time5);

    free_matrices(A, B, C);
    printf("\n所有测试完成！\n");

    return 0;
}


// 使用线程数: 4
// 矩阵大小: 1024 x 1024, 迭代次数: 5

// 运行串行版本...
// 串行版本运行时间: 25.4315秒, 加速比: 1.00

// 运行parallel for版本...
// parallel for版本运行时间: 7.6121秒, 加速比: 3.34

// 运行collapse版本1...
// collapse版本1运行时间: 5.0875秒, 加速比: 5.00

// 运行collapse版本2...
// collapse版本2运行时间: 5.3319秒, 加速比: 4.77

// 运行collapse版本3...
// collapse版本3运行时间: 4.5331秒, 加速比: 5.61

// 运行collapse版本4...
// collapse版本4运行时间: 5.5794秒, 加速比: 4.56


// 使用线程数: 64
// 矩阵大小: 1024 x 1024, 迭代次数: 5

// 运行串行版本...
// 串行版本运行时间: 20.5345秒, 加速比: 1.00

// 运行parallel for版本...
// parallel for版本运行时间: 1.1215秒, 加速比: 18.31

// 运行collapse版本1...
// collapse版本1运行时间: 1.0964秒, 加速比: 18.73

// 运行collapse版本2...
// collapse版本2运行时间: 0.9144秒, 加速比: 22.46

// 运行collapse版本3...
// collapse版本3运行时间: 1.0659秒, 加速比: 19.27

// 运行collapse版本4...
// collapse版本4运行时间: 0.9642秒, 加速比: 21.30