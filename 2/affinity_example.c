
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#define MATRIX_SIZE 1024
#define ITERATIONS 5

// 初始化矩阵
void init_matrices(double** A, double** B, double** C, int size) {
    *A = (double*)malloc(size * size * sizeof(double));
    *B = (double*)malloc(size * size * sizeof(double));
    *C = (double*)malloc(size * size * sizeof(double));
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            (*A)[i * size + j] = (double)rand() / RAND_MAX;
            (*B)[i * size + j] = (double)rand() / RAND_MAX;
            (*C)[i * size + j] = 0.0;
        }
    }
}

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
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// parallel for版本
double parallel_for_version(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    omp_set_num_threads(num_threads);
    
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
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// collapse版本1 - 合并两层循环
double collapse_version1(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    omp_set_num_threads(num_threads);
    
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
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// collapse版本2 - 合并并优化循环顺序
double collapse_version2(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    omp_set_num_threads(num_threads);
    
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
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// collapse版本3 - 合并并分块
double collapse_version3(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    omp_set_num_threads(num_threads);
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        memset(C, 0, size * size * sizeof(double));
        
        int block_size = 32;
        
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
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// affinity(close)版本
double affinity_close_version(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    omp_set_num_threads(num_threads);
    
    double sum = 0.0;
    #pragma omp parallel proc_bind(close)
    {
        #pragma omp for reduction(+:sum)
        for (int iter = 0; iter < ITERATIONS; iter++) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    C[i * size + j] = sin(A[i * size + j]) + cos(B[i * size + j]) * tan(0.1);
                    sum += C[i * size + j];
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// affinity(spread)版本
double affinity_spread_version(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    omp_set_num_threads(num_threads);
    
    double sum = 0.0;
    #pragma omp parallel proc_bind(spread)
    {
        #pragma omp for reduction(+:sum)
        for (int iter = 0; iter < ITERATIONS; iter++) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    C[i * size + j] = sin(A[i * size + j]) + cos(B[i * size + j]) * tan(0.1);
                    sum += C[i * size + j];
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// affinity(master)版本
double affinity_master_version(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    omp_set_num_threads(num_threads);
    
    double sum = 0.0;
    #pragma omp parallel proc_bind(master)
    {
        #pragma omp for reduction(+:sum)
        for (int iter = 0; iter < ITERATIONS; iter++) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    C[i * size + j] = sin(A[i * size + j]) + cos(B[i * size + j]) * tan(0.1);
                    sum += C[i * size + j];
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// affinity+places版本
double affinity_places_version(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    omp_set_num_threads(num_threads);
    omp_set_schedule(omp_sched_dynamic, 16);
    
    double sum = 0.0;
    #pragma omp parallel proc_bind(close)
    {
        int tid = omp_get_thread_num();
        int place = omp_get_place_num();
        
        #pragma omp master
        {
            printf("线程分布信息:\n");
        }
        
        #pragma omp critical
        {
            printf("线程 %d 运行在物理位置 %d\n", tid, place);
        }
        
        #pragma omp barrier
        
        #pragma omp for reduction(+:sum) schedule(static)
        for (int iter = 0; iter < ITERATIONS; iter++) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    C[i * size + j] = sin(A[i * size + j]) + cos(B[i * size + j]) * tan(0.1);
                    sum += C[i * size + j];
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main(int argc, char* argv[]) {
    int num_threads = 64;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    printf("使用线程数: %d\n", num_threads);
    
    srand(time(NULL));
    
    double* A;
    double* B;
    double* C;
    
    init_matrices(&A, &B, &C, MATRIX_SIZE);
    
    printf("矩阵大小: %d x %d, 迭代次数: %d\n", MATRIX_SIZE, MATRIX_SIZE, ITERATIONS);
    
    printf("\n运行串行版本...\n");
    double serial_time = serial_version(A, B, C, MATRIX_SIZE);
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    printf("加速比: 1.00\n");
    
    printf("\n运行parallel for版本...\n");
    double parallel_time = parallel_for_version(A, B, C, MATRIX_SIZE, num_threads);
    printf("parallel for版本运行时间: %.4f 秒\n", parallel_time);
    printf("加速比: %.2f\n", serial_time / parallel_time);
    
    printf("\n运行collapse版本1 (合并两层循环)...\n");
    double collapse_time1 = collapse_version1(A, B, C, MATRIX_SIZE, num_threads);
    printf("collapse版本1运行时间: %.4f 秒\n", collapse_time1);
    printf("加速比: %.2f\n", serial_time / collapse_time1);
    
    printf("\n运行collapse版本2 (合并并优化循环顺序)...\n");
    double collapse_time2 = collapse_version2(A, B, C, MATRIX_SIZE, num_threads);
    printf("collapse版本2运行时间: %.4f 秒\n", collapse_time2);
    printf("加速比: %.2f\n", serial_time / collapse_time2);
    
    printf("\n运行collapse版本3 (合并并分块)...\n");
    double collapse_time3 = collapse_version3(A, B, C, MATRIX_SIZE, num_threads);
    printf("collapse版本3运行时间: %.4f 秒\n", collapse_time3);
    printf("加速比: %.2f\n", serial_time / collapse_time3);
    
    printf("\n运行affinity(close)版本...\n");
    double affinity_close_time = affinity_close_version(A, B, C, MATRIX_SIZE, num_threads);
    printf("affinity(close)版本运行时间: %.4f 秒\n", affinity_close_time);
    printf("加速比: %.2f\n", serial_time / affinity_close_time);
    
    printf("\n运行affinity(spread)版本...\n");
    double affinity_spread_time = affinity_spread_version(A, B, C, MATRIX_SIZE, num_threads);
    printf("affinity(spread)版本运行时间: %.4f 秒\n", affinity_spread_time);
    printf("加速比: %.2f\n", serial_time / affinity_spread_time);
    
    printf("\n运行affinity(master)版本...\n");
    double affinity_master_time = affinity_master_version(A, B, C, MATRIX_SIZE, num_threads);
    printf("affinity(master)版本运行时间: %.4f 秒\n", affinity_master_time);
    printf("加速比: %.2f\n", serial_time / affinity_master_time);
    
    printf("\n运行affinity+places版本...\n");
    double affinity_places_time = affinity_places_version(A, B, C, MATRIX_SIZE, num_threads);
    printf("affinity+places版本运行时间: %.4f 秒\n", affinity_places_time);
    printf("加速比: %.2f\n", serial_time / affinity_places_time);
    
    free_matrices(A, B, C);
    
    printf("\n所有测试完成！\n");
    
    return 0;
}


// 使用线程数: 4
// 矩阵大小: 1024 x 1024, 迭代次数: 5

// 运行串行版本...
// 串行版本运行时间: 21.1098 秒
// 加速比: 1.00

// 运行parallel for版本...
// parallel for版本运行时间: 6.9781 秒
// 加速比: 3.03

// 运行collapse版本1 (合并两层循环)...
// collapse版本1运行时间: 7.0613 秒
// 加速比: 2.99

// 运行collapse版本2 (合并并优化循环顺序)...
// collapse版本2运行时间: 4.5922 秒
// 加速比: 4.60

// 运行collapse版本3 (合并并分块)...
// collapse版本3运行时间: 5.6519 秒
// 加速比: 3.73

// 运行affinity(close)版本...
// affinity(close)版本运行时间: 0.0983 秒
// 加速比: 214.70

// 运行affinity(spread)版本...
// affinity(spread)版本运行时间: 0.0696 秒
// 加速比: 303.29

// 运行affinity(master)版本...
// affinity(master)版本运行时间: 0.1180 秒
// 加速比: 178.86

// 运行affinity+places版本...
// 线程 3 运行在物理位置 -1
// 线程分布信息:
// 线程 0 运行在物理位置 -1
// 线程 2 运行在物理位置 -1
// 线程 1 运行在物理位置 -1
// affinity+places版本运行时间: 0.0740 秒
// 加速比: 285.19

// 所有测试完成！

// 使用线程数: 64
// 矩阵大小: 1024 x 1024, 迭代次数: 5

// 运行串行版本...
// 串行版本运行时间: 21.3626 秒
// 加速比: 1.00

// 运行parallel for版本...
// parallel for版本运行时间: 0.7894 秒
// 加速比: 27.06

// 运行collapse版本1 (合并两层循环)...
// collapse版本1运行时间: 0.8777 秒
// 加速比: 24.34

// 运行collapse版本2 (合并并优化循环顺序)...
// collapse版本2运行时间: 1.0581 秒
// 加速比: 20.19

// 运行collapse版本3 (合并并分块)...
// collapse版本3运行时间: 0.8470 秒
// 加速比: 25.22

// 运行affinity(close)版本...
// affinity(close)版本运行时间: 0.0576 秒
// 加速比: 370.88

// 运行affinity(spread)版本...
// affinity(spread)版本运行时间: 0.0531 秒
// 加速比: 402.33

// 运行affinity(master)版本...
// affinity(master)版本运行时间: 0.0803 秒
// 加速比: 266.08

// 运行affinity+places版本...
// 线程 17 运行在物理位置 -1
// 线程 9 运行在物理位置 -1
// 线程 23 运行在物理位置 -1
// 线程 45 运行在物理位置 -1
// 线程 14 运行在物理位置 -1
// 线程 29 运行在物理位置 -1
// 线程 5 运行在物理位置 -1
// 线程 25 运行在物理位置 -1
// 线程 6 运行在物理位置 -1
// 线程 15 运行在物理位置 -1
// 线程 27 运行在物理位置 -1
// 线程 41 运行在物理位置 -1
// 线程 18 运行在物理位置 -1
// 线程 10 运行在物理位置 -1
// 线程 40 运行在物理位置 -1
// 线程 57 运行在物理位置 -1
// 线程 59 运行在物理位置 -1
// 线程 44 运行在物理位置 -1
// 线程 38 运行在物理位置 -1
// 线程 20 运行在物理位置 -1
// 线程 47 运行在物理位置 -1
// 线程分布信息:
// 线程 26 运行在物理位置 -1
// 线程 19 运行在物理位置 -1
// 线程 13 运行在物理位置 -1
// 线程 48 运行在物理位置 -1
// 线程 53 运行在物理位置 -1
// 线程 33 运行在物理位置 -1
// 线程 32 运行在物理位置 -1
// 线程 11 运行在物理位置 -1
// 线程 31 运行在物理位置 -1
// 线程 42 运行在物理位置 -1
// 线程 61 运行在物理位置 -1
// 线程 58 运行在物理位置 -1
// 线程 56 运行在物理位置 -1
// 线程 49 运行在物理位置 -1
// 线程 22 运行在物理位置 -1
// 线程 52 运行在物理位置 -1
// 线程 51 运行在物理位置 -1
// 线程 3 运行在物理位置 -1
// 线程 55 运行在物理位置 -1
// 线程 24 运行在物理位置 -1
// 线程 16 运行在物理位置 -1
// 线程 12 运行在物理位置 -1
// 线程 35 运行在物理位置 -1
// 线程 62 运行在物理位置 -1
// 线程 60 运行在物理位置 -1
// 线程 34 运行在物理位置 -1
// 线程 46 运行在物理位置 -1
// 线程 36 运行在物理位置 -1
// 线程 21 运行在物理位置 -1
// 线程 50 运行在物理位置 -1
// 线程 54 运行在物理位置 -1
// 线程 43 运行在物理位置 -1
// 线程 8 运行在物理位置 -1
// 线程 37 运行在物理位置 -1
// 线程 2 运行在物理位置 -1
// 线程 63 运行在物理位置 -1
// 线程 4 运行在物理位置 -1
// 线程 0 运行在物理位置 -1
// 线程 30 运行在物理位置 -1
// 线程 7 运行在物理位置 -1
// 线程 39 运行在物理位置 -1
// 线程 28 运行在物理位置 -1
// 线程 1 运行在物理位置 -1
// affinity+places版本运行时间: 0.0916 秒
// 加速比: 233.18