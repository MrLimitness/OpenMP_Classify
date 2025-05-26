
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

// 定义矩阵大小
#define N 2000

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

// 并行版本2：使用teams distribute并行版本
void matrix_multiply_teams_distribute(double **A, double **B, double **C, int n) {
    #pragma omp teams distribute parallel for
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

// 并行版本3：使用teams distribute dispatch并行版本
void matrix_multiply_teams_distribute_dispatch(double **A, double **B, double **C, int n) {
    #pragma omp teams distribute
    for (int i = 0; i < n; i++) {
        #pragma omp dispatch
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// 并行版本4：使用teams distribute dispatch与collapse结合
void matrix_multiply_teams_distribute_dispatch_collapse(double **A, double **B, double **C, int n) {
    #pragma omp teams distribute collapse(2)
    for (int i = 0; i < n; i += 32) {
        for (int j = 0; j < n; j += 32) {
            #pragma omp dispatch
            for (int ii = i; ii < i + 32 && ii < n; ii++) {
                for (int jj = j; jj < j + 32 && jj < n; jj++) {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        sum += A[ii][k] * B[k][jj];
                    }
                    C[ii][jj] = sum;
                }
            }
        }
    }
}

// 并行版本5：使用teams distribute dispatch schedule结合
void matrix_multiply_teams_distribute_dispatch_schedule(double **A, double **B, double **C, int n) {
    #pragma omp teams distribute
    for (int i = 0; i < n; i++) {
        #pragma omp dispatch schedule(dynamic, 16)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    int num_threads = 64; // 可以修改为16/64/144等
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
    
    // 测试teams distribute版本
    printf("运行teams distribute版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_teams_distribute(A, B, C, N);
    end_time = omp_get_wtime();
    double teams_distribute_time = end_time - start_time;
    printf("teams distribute版本运行时间: %.4f 秒\n", teams_distribute_time);
    printf("teams distribute版本加速比: %.2f\n\n", serial_time / teams_distribute_time);
    
    // 验证结果
    double teams_distribute_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            teams_distribute_checksum += C[i][j];
        }
    }
    printf("teams distribute版本结果校验和: %.6e\n", teams_distribute_checksum);
    printf("teams distribute版本结果误差: %.6e\n\n", fabs(teams_distribute_checksum - serial_checksum));
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
        }
    }
    
    // 测试teams distribute dispatch版本
    printf("运行teams distribute dispatch版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_teams_distribute_dispatch(A, B, C, N);
    end_time = omp_get_wtime();
    double teams_distribute_dispatch_time = end_time - start_time;
    printf("teams distribute dispatch版本运行时间: %.4f 秒\n", teams_distribute_dispatch_time);
    printf("teams distribute dispatch版本加速比: %.2f\n\n", serial_time / teams_distribute_dispatch_time);
    
    // 验证结果
    double teams_distribute_dispatch_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            teams_distribute_dispatch_checksum += C[i][j];
        }
    }
    printf("teams distribute dispatch版本结果校验和: %.6e\n", teams_distribute_dispatch_checksum);
    printf("teams distribute dispatch版本结果误差: %.6e\n\n", fabs(teams_distribute_dispatch_checksum - serial_checksum));
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
        }
    }
    
    // 测试teams distribute dispatch collapse版本
    printf("运行teams distribute dispatch collapse版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_teams_distribute_dispatch_collapse(A, B, C, N);
    end_time = omp_get_wtime();
    double teams_distribute_dispatch_collapse_time = end_time - start_time;
    printf("teams distribute dispatch collapse版本运行时间: %.4f 秒\n", teams_distribute_dispatch_collapse_time);
    printf("teams distribute dispatch collapse版本加速比: %.2f\n\n", serial_time / teams_distribute_dispatch_collapse_time);
    
    // 验证结果
    double teams_distribute_dispatch_collapse_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            teams_distribute_dispatch_collapse_checksum += C[i][j];
        }
    }
    printf("teams distribute dispatch collapse版本结果校验和: %.6e\n", teams_distribute_dispatch_collapse_checksum);
    printf("teams distribute dispatch collapse版本结果误差: %.6e\n\n", fabs(teams_distribute_dispatch_collapse_checksum - serial_checksum));
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
        }
    }
    
    // 测试teams distribute dispatch schedule版本
    printf("运行teams distribute dispatch schedule版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_teams_distribute_dispatch_schedule(A, B, C, N);
    end_time = omp_get_wtime();
    double teams_distribute_dispatch_schedule_time = end_time - start_time;
    printf("teams distribute dispatch schedule版本运行时间: %.4f 秒\n", teams_distribute_dispatch_schedule_time);
    printf("teams distribute dispatch schedule版本加速比: %.2f\n\n", serial_time / teams_distribute_dispatch_schedule_time);
    
    // 验证结果
    double teams_distribute_dispatch_schedule_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            teams_distribute_dispatch_schedule_checksum += C[i][j];
        }
    }
    printf("teams distribute dispatch schedule版本结果校验和: %.6e\n", teams_distribute_dispatch_schedule_checksum);
    printf("teams distribute dispatch schedule版本结果误差: %.6e\n\n", fabs(teams_distribute_dispatch_schedule_checksum - serial_checksum));
    
    // 性能总结
    printf("======= 性能总结 =======\n");
    printf("串行版本:                          %.4f 秒 (加速比: 1.00)\n", serial_time);
    printf("parallel for版本:                  %.4f 秒 (加速比: %.2f)\n", parallel_for_time, serial_time / parallel_for_time);
    printf("teams distribute版本:              %.4f 秒 (加速比: %.2f)\n", teams_distribute_time, serial_time / teams_distribute_time);
    printf("teams distribute dispatch版本:     %.4f 秒 (加速比: %.2f)\n", teams_distribute_dispatch_time, serial_time / teams_distribute_dispatch_time);
    printf("teams distribute dispatch collapse: %.4f 秒 (加速比: %.2f)\n", teams_distribute_dispatch_collapse_time, serial_time / teams_distribute_dispatch_collapse_time);
    printf("teams distribute dispatch schedule: %.4f 秒 (加速比: %.2f)\n", teams_distribute_dispatch_schedule_time, serial_time / teams_distribute_dispatch_schedule_time);
    
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


// 矩阵大小: 2000 x 2000
// 运行线程数: 16

// 运行串行版本...
// 串行版本运行时间: 38.4745 秒
// 串行版本加速比: 1.00

// 串行版本结果校验和: 1.999779e+09

// 运行parallel for版本...
// parallel for版本运行时间: 4.0146 秒
// parallel for版本加速比: 9.58

// parallel for版本结果校验和: 1.999779e+09
// parallel for版本结果误差: 0.000000e+00

// 运行teams distribute版本...
// teams distribute版本运行时间: 3.9799 秒
// teams distribute版本加速比: 9.67

// teams distribute版本结果校验和: 1.999779e+09
// teams distribute版本结果误差: 0.000000e+00

// 运行teams distribute dispatch版本...
// teams distribute dispatch版本运行时间: 28.9088 秒
// teams distribute dispatch版本加速比: 1.33

// teams distribute dispatch版本结果校验和: 1.999779e+09
// teams distribute dispatch版本结果误差: 0.000000e+00

// 运行teams distribute dispatch collapse版本...
// teams distribute dispatch collapse版本运行时间: 20.5043 秒
// teams distribute dispatch collapse版本加速比: 1.88

// teams distribute dispatch collapse版本结果校验和: 1.999779e+09
// teams distribute dispatch collapse版本结果误差: 0.000000e+00

// 运行teams distribute dispatch schedule版本...
// teams distribute dispatch schedule版本运行时间: 26.6164 秒
// teams distribute dispatch schedule版本加速比: 1.45

// teams distribute dispatch schedule版本结果校验和: 1.999779e+09
// teams distribute dispatch schedule版本结果误差: 0.000000e+00

// ======= 性能总结 =======
// 串行版本:                          38.4745 秒 (加速比: 1.00)
// parallel for版本:                  4.0146 秒 (加速比: 9.58)
// teams distribute版本:              3.9799 秒 (加速比: 9.67)
// teams distribute dispatch版本:     28.9088 秒 (加速比: 1.33)
// teams distribute dispatch collapse: 20.5043 秒 (加速比: 1.88)
// teams distribute dispatch schedule: 26.6164 秒 (加速比: 1.45)


// 矩阵大小: 2000 x 2000
// 运行线程数: 64

// 运行串行版本...
// 串行版本运行时间: 43.3563 秒
// 串行版本加速比: 1.00

// 串行版本结果校验和: 1.999779e+09

// 运行parallel for版本...
// parallel for版本运行时间: 1.3708 秒
// parallel for版本加速比: 31.63

// parallel for版本结果校验和: 1.999779e+09
// parallel for版本结果误差: 0.000000e+00

// 运行teams distribute版本...
// teams distribute版本运行时间: 2.6695 秒
// teams distribute版本加速比: 16.24

// teams distribute版本结果校验和: 1.999779e+09
// teams distribute版本结果误差: 0.000000e+00

// 运行teams distribute dispatch版本...
// teams distribute dispatch版本运行时间: 45.1621 秒
// teams distribute dispatch版本加速比: 0.96

// teams distribute dispatch版本结果校验和: 1.999779e+09
// teams distribute dispatch版本结果误差: 0.000000e+00

// 运行teams distribute dispatch collapse版本...
// teams distribute dispatch collapse版本运行时间: 24.4275 秒
// teams distribute dispatch collapse版本加速比: 1.77

// teams distribute dispatch collapse版本结果校验和: 1.999779e+09
// teams distribute dispatch collapse版本结果误差: 0.000000e+00

// 运行teams distribute dispatch schedule版本...
// teams distribute dispatch schedule版本运行时间: 21.9135 秒
// teams distribute dispatch schedule版本加速比: 1.98

// teams distribute dispatch schedule版本结果校验和: 1.999779e+09
// teams distribute dispatch schedule版本结果误差: 0.000000e+00

// ======= 性能总结 =======
// 串行版本:                          43.3563 秒 (加速比: 1.00)
// parallel for版本:                  1.3708 秒 (加速比: 31.63)
// teams distribute版本:              2.6695 秒 (加速比: 16.24)
// teams distribute dispatch版本:     45.1621 秒 (加速比: 0.96)
// teams distribute dispatch collapse: 24.4275 秒 (加速比: 1.77)
// teams distribute dispatch schedule: 21.9135 秒 (加速比: 1.98)