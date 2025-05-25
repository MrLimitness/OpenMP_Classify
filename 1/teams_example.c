#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

// 定义矩阵大小
#define N 2048

// 串行版本
double serial_matrix_multiply(double* A, double* B, double* C) {
    printf("执行串行版本...\n");
    double start_time = omp_get_wtime();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    double end_time = omp_get_wtime();
    printf("串行版本执行时间: %.6f 秒\n", end_time - start_time);
    printf("加速比: 1.00\n\n");
    
    return end_time - start_time;
}

// parallel for 版本
double parallel_for_matrix_multiply(double* A, double* B, double* C, int num_threads) {
    printf("执行 parallel for 版本 (线程数: %d)...\n", num_threads);
    double start_time = omp_get_wtime();

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("parallel for 版本执行时间: %.6f 秒\n", elapsed);
    
    return elapsed;
}

// teams 版本1: 基本 teams + distribute + parallel for
double teams_version1(double* A, double* B, double* C, int num_teams, int num_threads) {
    printf("执行 teams 版本1 (团队数: %d, 每团队线程数: %d)...\n", num_teams, num_threads);
    double start_time = omp_get_wtime();

    #pragma omp teams num_teams(num_teams) thread_limit(num_threads)
    #pragma omp distribute parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("teams 版本1执行时间: %.6f 秒\n", elapsed);
    
    return elapsed;
}

// teams 版本2: teams + distribute + parallel for + collapse
double teams_version2(double* A, double* B, double* C, int num_teams, int num_threads) {
    printf("执行 teams 版本2 (团队数: %d, 每团队线程数: %d)...\n", num_teams, num_threads);
    double start_time = omp_get_wtime();

    #pragma omp teams num_teams(num_teams) thread_limit(num_threads)
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("teams 版本2执行时间: %.6f 秒\n", elapsed);
    
    return elapsed;
}

// teams 版本3: teams + distribute + 分块优化
double teams_version3(double* A, double* B, double* C, int num_teams, int num_threads) {
    printf("执行 teams 版本3 (团队数: %d, 每团队线程数: %d)...\n", num_teams, num_threads);
    double start_time = omp_get_wtime();
    
    int BLOCK_SIZE = 64;

    #pragma omp teams num_teams(num_teams) thread_limit(num_threads)
    #pragma omp distribute
    for (int ib = 0; ib < N; ib += BLOCK_SIZE) {
        #pragma omp parallel for num_threads(num_threads)
        for (int jb = 0; jb < N; jb += BLOCK_SIZE) {
            for (int kb = 0; kb < N; kb += BLOCK_SIZE) {
                // 计算块内矩阵乘法
                for (int i = ib; i < ib + BLOCK_SIZE && i < N; i++) {
                    for (int j = jb; j < jb + BLOCK_SIZE && j < N; j++) {
                        double sum = 0.0;
                        for (int k = kb; k < kb + BLOCK_SIZE && k < N; k++) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        if (kb == 0) {
                            C[i * N + j] = sum;
                        } else {
                            C[i * N + j] += sum;
                        }
                    }
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("teams 版本3执行时间: %.6f 秒\n", elapsed);
    
    return elapsed;
}

// teams 版本4: teams + distribute + simd
double teams_version4(double* A, double* B, double* C, int num_teams, int num_threads) {
    printf("执行 teams 版本4 (团队数: %d, 每团队线程数: %d)...\n", num_teams, num_threads);
    double start_time = omp_get_wtime();

    #pragma omp teams num_teams(num_teams) thread_limit(num_threads)
    #pragma omp distribute parallel for 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("teams 版本4执行时间: %.6f 秒\n", elapsed);
    
    return elapsed;
}

int main(int argc, char* argv[]) {
    int num_threads = 64;  // 默认线程数
    int num_teams = 4;     // 默认团队数
    
    // 检查命令行参数
    if (argc >= 2) {
        num_threads = atoi(argv[1]);
    }
    
    if (argc >= 3) {
        num_teams = atoi(argv[2]);
    }
    
    printf("设置线程数: %d, 团队数: %d\n\n", num_threads, num_teams);
    
    // 分配矩阵内存
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)malloc(N * N * sizeof(double));
    double *C_serial = (double*)malloc(N * N * sizeof(double));
    
    // 初始化矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (double)rand() / RAND_MAX;
            B[i * N + j] = (double)rand() / RAND_MAX;
        }
    }
    
    // 执行串行版本
    double serial_time = serial_matrix_multiply(A, B, C_serial);
    
    // 执行 parallel for 版本
    double parallel_time = parallel_for_matrix_multiply(A, B, C, num_threads);
    printf("加速比: %.2f\n\n", serial_time / parallel_time);
    
    // 执行 teams 版本1
    double teams_time1 = teams_version1(A, B, C, num_teams, num_threads);
    printf("加速比: %.2f\n\n", serial_time / teams_time1);
    
    // 执行 teams 版本2
    double teams_time2 = teams_version2(A, B, C, num_teams, num_threads);
    printf("加速比: %.2f\n\n", serial_time / teams_time2);
    
    // 执行 teams 版本3
    double teams_time3 = teams_version3(A, B, C, num_teams, num_threads);
    printf("加速比: %.2f\n\n", serial_time / teams_time3);
    
    // 执行 teams 版本4
    double teams_time4 = teams_version4(A, B, C, num_teams, num_threads);
    printf("加速比: %.2f\n\n", serial_time / teams_time4);
    
    // 释放内存
    free(A);
    free(B);
    free(C);
    free(C_serial);
    
    return 0;
}

// 设置线程数: 16, 团队数: 4

// 执行串行版本...
// 串行版本执行时间: 28.887903 秒
// 加速比: 1.00

// 执行 parallel for 版本 (线程数: 16)...
// parallel for 版本执行时间: 1.476510 秒
// 加速比: 19.56

// 执行 teams 版本1 (团队数: 4, 每团队线程数: 16)...
// teams 版本1执行时间: 1.457353 秒
// 加速比: 19.82

// 执行 teams 版本2 (团队数: 4, 每团队线程数: 16)...
// teams 版本2执行时间: 1.460489 秒
// 加速比: 19.78

// 执行 teams 版本3 (团队数: 4, 每团队线程数: 16)...
// teams 版本3执行时间: 0.539415 秒
// 加速比: 53.55

// 执行 teams 版本4 (团队数: 4, 每团队线程数: 16)...
// teams 版本4执行时间: 2.007844 秒
// 加速比: 14.39


// 设置线程数: 64, 团队数: 4

// 执行串行版本...
// 串行版本执行时间: 28.887751 秒
// 加速比: 1.00

// 执行 parallel for 版本 (线程数: 64)...
// parallel for 版本执行时间: 1.268969 秒
// 加速比: 22.76

// 执行 teams 版本1 (团队数: 4, 每团队线程数: 64)...
// teams 版本1执行时间: 1.223569 秒
// 加速比: 23.61

// 执行 teams 版本2 (团队数: 4, 每团队线程数: 64)...
// teams 版本2执行时间: 1.251688 秒
// 加速比: 23.08

// 执行 teams 版本3 (团队数: 4, 每团队线程数: 64)...
// teams 版本3执行时间: 0.511085 秒
// 加速比: 56.52

// 执行 teams 版本4 (团队数: 4, 每团队线程数: 64)...
// teams 版本4执行时间: 1.205182 秒
// 加速比: 23.97