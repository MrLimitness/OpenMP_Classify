#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>

#define SIZE 1024
#define ITERATIONS 5

// 初始化矩阵
void init_matrices(double* A, double* B, double* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = (double)rand() / RAND_MAX;
            B[i * size + j] = (double)rand() / RAND_MAX;
            C[i * size + j] = 0.0;
        }
    }
}

// 串行版本
double serial_version(double* A, double* B, double* C, int size) {
    double start_time = omp_get_wtime();
    for (int iter = 0; iter < ITERATIONS; iter++) {
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

// Parallel for 版本
double parallel_for_version(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(num_threads);
    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
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

// simdlen 版本 1
double simdlen_version1(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(num_threads);
    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double sum = 0.0;
                #pragma omp simd simdlen(8) reduction(+:sum)
                for (int k = 0; k < size; k++) {
                    sum += A[i * size + k] * B[k * size + j];
                }
                C[i * size + j] = sum;
            }
        }
    }
    return omp_get_wtime() - start_time;
}

// simdlen 版本 2
double simdlen_version2(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(num_threads);
    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double sum = 0.0;
                #pragma omp simd simdlen(16) reduction(+:sum)
                for (int k = 0; k < size; k++) {
                    sum += A[i * size + k] * B[k * size + j];
                }
                C[i * size + j] = sum;
            }
        }
    }
    return omp_get_wtime() - start_time;
}

// simdlen 版本 3
double simdlen_version3(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(num_threads);
    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double sum = 0.0;
                #pragma omp simd simdlen(4) reduction(+:sum)
                for (int k = 0; k < size; k++) {
                    sum += A[i * size + k] * B[k * size + j];
                }
                C[i * size + j] = sum;
            }
        }
    }
    return omp_get_wtime() - start_time;
}

// simdlen 版本 4（分块 + simdlen）
double simdlen_version4(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(num_threads);
    int block_size = 64;

    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (int ii = 0; ii < size; ii += block_size) {
            for (int jj = 0; jj < size; jj += block_size) {
                for (int kk = 0; kk < size; kk += block_size) {
                    for (int i = ii; i < ii + block_size && i < size; i++) {
                        for (int j = jj; j < jj + block_size && j < size; j++) {
                            double sum = 0.0;
                            for (int k = kk; k < kk + block_size; k++) {
                                if (k < size) {
                                    #pragma omp simd simdlen(8) reduction(+:sum)
                                    for (int kk_inner = k; kk_inner < k + 1; kk_inner++) {
                                        sum += A[i * size + kk_inner] * B[kk_inner * size + j];
                                    }
                                }
                            }
                            if (kk == 0) {
                                C[i * size + j] = sum;
                            } else {
                                C[i * size + j] += sum;
                            }
                        }
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

    double* A = (double*)malloc(SIZE * SIZE * sizeof(double));
    double* B = (double*)malloc(SIZE * SIZE * sizeof(double));
    double* C = (double*)malloc(SIZE * SIZE * sizeof(double));
    if (!A || !B || !C) {
        printf("内存分配失败！\n");
        return 1;
    }

    init_matrices(A, B, C, SIZE);
    printf("矩阵大小: %d x %d, 迭代次数: %d\n", SIZE, SIZE, ITERATIONS);

    printf("\n运行串行版本...\n");
    double t_serial = serial_version(A, B, C, SIZE);
    printf("串行版本时间: %.4f 秒，加速比: 1.00\n", t_serial);

    memset(C, 0, SIZE * SIZE * sizeof(double));
    printf("\nparallel for 版本...\n");
    double t_par = parallel_for_version(A, B, C, SIZE, num_threads);
    printf("parallel for 时间: %.4f 秒，加速比: %.2f\n", t_par, t_serial / t_par);

    memset(C, 0, SIZE * SIZE * sizeof(double));
    printf("\nsimdlen(8) 版本...\n");
    double t_v1 = simdlen_version1(A, B, C, SIZE, num_threads);
    printf("simdlen(8) 时间: %.4f 秒，加速比: %.2f\n", t_v1, t_serial / t_v1);

    memset(C, 0, SIZE * SIZE * sizeof(double));
    printf("\nsimdlen(16) 版本...\n");
    double t_v2 = simdlen_version2(A, B, C, SIZE, num_threads);
    printf("simdlen(16) 时间: %.4f 秒，加速比: %.2f\n", t_v2, t_serial / t_v2);

    memset(C, 0, SIZE * SIZE * sizeof(double));
    printf("\nsimdlen(4) 版本...\n");
    double t_v3 = simdlen_version3(A, B, C, SIZE, num_threads);
    printf("simdlen(4) 时间: %.4f 秒，加速比: %.2f\n", t_v3, t_serial / t_v3);

    memset(C, 0, SIZE * SIZE * sizeof(double));
    printf("\nsimdlen + 分块优化版本...\n");
    double t_v4 = simdlen_version4(A, B, C, SIZE, num_threads);
    printf("simdlen+block 时间: %.4f 秒，加速比: %.2f\n", t_v4, t_serial / t_v4);

    free(A);
    free(B);
    free(C);
    printf("\n所有测试完成！\n");

    return 0;
}
// 使用线程数: 16
// 矩阵大小: 1024 x 1024, 迭代次数: 5

// 运行串行版本...
// 串行版本时间: 15.2596 秒，加速比: 1.00

// parallel for 版本...
// parallel for 时间: 1.4962 秒，加速比: 10.20

// simdlen(8) 版本...
// simdlen(8) 时间: 2.6677 秒，加速比: 5.72

// simdlen(16) 版本...
// simdlen(16) 时间: 3.5330 秒，加速比: 4.32

// simdlen(4) 版本...
// simdlen(4) 时间: 3.2281 秒，加速比: 4.73

// simdlen + 分块优化版本...
// simdlen+block 时间: 3.1705 秒，加速比: 4.81


// 使用线程数: 64
// 矩阵大小: 1024 x 1024, 迭代次数: 5

// 运行串行版本...
// 串行版本时间: 18.4593 秒，加速比: 1.00

// parallel for 版本...
// parallel for 时间: 0.9866 秒，加速比: 18.71

// simdlen(8) 版本...
// simdlen(8) 时间: 0.9507 秒，加速比: 19.42

// simdlen(16) 版本...
// simdlen(16) 时间: 0.9553 秒，加速比: 19.32

// simdlen(4) 版本...
// simdlen(4) 时间: 0.8754 秒，加速比: 21.09

// simdlen + 分块优化版本...
// simdlen+block 时间: 0.8475 秒，加速比: 21.78
