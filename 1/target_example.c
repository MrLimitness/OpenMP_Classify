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

// 并行版本2：使用target的矩阵乘法
void matrix_multiply_target(double **A, double **B, double **C, int n) {
    // 创建一维数组用于设备数据传输
    double *A_flat = (double *)malloc(n * n * sizeof(double));
    double *B_flat = (double *)malloc(n * n * sizeof(double));
    double *C_flat = (double *)malloc(n * n * sizeof(double));
    
    // 将二维数组转换为一维数组
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_flat[i * n + j] = A[i][j];
            B_flat[i * n + j] = B[i][j];
            C_flat[i * n + j] = 0.0;
        }
    }
    
    // 使用target将计算迁移到加速器设备
    #pragma omp target data map(to: A_flat[0:n*n], B_flat[0:n*n]) map(tofrom: C_flat[0:n*n])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A_flat[i * n + k] * B_flat[k * n + j];
                }
                C_flat[i * n + j] = sum;
            }
        }
    }
    
    // 将结果从一维数组复制回二维数组
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = C_flat[i * n + j];
        }
    }
    
    // 释放内存
    free(A_flat);
    free(B_flat);
    free(C_flat);
}

// 并行版本3：使用target与数据分块的矩阵乘法
void matrix_multiply_target_blocking(double **A, double **B, double **C, int n) {
    double *A_flat = (double *)malloc(n * n * sizeof(double));
    double *B_flat = (double *)malloc(n * n * sizeof(double));
    double *C_flat = (double *)malloc(n * n * sizeof(double));
    
    // 将二维数组转换为一维数组
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_flat[i * n + j] = A[i][j];
            B_flat[i * n + j] = B[i][j];
            C_flat[i * n + j] = 0.0;
        }
    }
    
    // 块大小
    const int BLOCK_SIZE = 32;
    
    #pragma omp target data map(to: A_flat[0:n*n], B_flat[0:n*n]) map(tofrom: C_flat[0:n*n])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < n; i += BLOCK_SIZE) {
            for (int j = 0; j < n; j += BLOCK_SIZE) {
                for (int k = 0; k < n; k += BLOCK_SIZE) {
                    // 块矩阵乘法
                    for (int ii = i; ii < i + BLOCK_SIZE && ii < n; ii++) {
                        for (int jj = j; jj < j + BLOCK_SIZE && jj < n; jj++) {
                            double sum = C_flat[ii * n + jj];  // 累加到现有值
                            for (int kk = k; kk < k + BLOCK_SIZE && kk < n; kk++) {
                                sum += A_flat[ii * n + kk] * B_flat[kk * n + jj];
                            }
                            C_flat[ii * n + jj] = sum;
                        }
                    }
                }
            }
        }
    }
    
    // 将结果从一维数组复制回二维数组
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = C_flat[i * n + j];
        }
    }
    
    // 释放内存
    free(A_flat);
    free(B_flat);
    free(C_flat);
}

// 并行版本4：使用target与阶段性计算的矩阵乘法
void matrix_multiply_target_staged(double **A, double **B, double **C, int n) {
    double *A_flat = (double *)malloc(n * n * sizeof(double));
    double *B_flat = (double *)malloc(n * n * sizeof(double));
    double *C_flat = (double *)malloc(n * n * sizeof(double));
    
    // 将二维数组转换为一维数组
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_flat[i * n + j] = A[i][j];
            B_flat[i * n + j] = B[i][j];
            C_flat[i * n + j] = 0.0;
        }
    }
    
    // 划分为2个阶段计算，每个阶段计算一半的行
    const int STAGE_SIZE = n / 2;
    
    // 第一阶段：计算前一半行
    #pragma omp target data map(to: A_flat[0:STAGE_SIZE*n], B_flat[0:n*n]) map(tofrom: C_flat[0:STAGE_SIZE*n])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < STAGE_SIZE; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A_flat[i * n + k] * B_flat[k * n + j];
                }
                C_flat[i * n + j] = sum;
            }
        }
    }
    
    // 第二阶段：计算后一半行
    #pragma omp target data map(to: A_flat[STAGE_SIZE*n:STAGE_SIZE*n], B_flat[0:n*n]) map(tofrom: C_flat[STAGE_SIZE*n:STAGE_SIZE*n])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = STAGE_SIZE; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A_flat[i * n + k] * B_flat[k * n + j];
                }
                C_flat[i * n + j] = sum;
            }
        }
    }
    
    // 将结果从一维数组复制回二维数组
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = C_flat[i * n + j];
        }
    }
    
    // 释放内存
    free(A_flat);
    free(B_flat);
    free(C_flat);
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
    
    // 测试target版本
    printf("运行target版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_target(A, B, C, N);
    end_time = omp_get_wtime();
    double target_time = end_time - start_time;
    printf("target版本运行时间: %.4f 秒\n", target_time);
    printf("target版本加速比: %.2f\n\n", serial_time / target_time);
    
    // 验证结果
    double target_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            target_checksum += C[i][j];
        }
    }
    printf("target版本结果校验和: %.6e\n", target_checksum);
    printf("target版本结果误差: %.6e\n\n", fabs(target_checksum - serial_checksum));
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
        }
    }
    
    // 测试target blocking版本
    printf("运行target blocking版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_target_blocking(A, B, C, N);
    end_time = omp_get_wtime();
    double target_blocking_time = end_time - start_time;
    printf("target blocking版本运行时间: %.4f 秒\n", target_blocking_time);
    printf("target blocking版本加速比: %.2f\n\n", serial_time / target_blocking_time);
    
    // 验证结果
    double target_blocking_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            target_blocking_checksum += C[i][j];
        }
    }
    printf("target blocking版本结果校验和: %.6e\n", target_blocking_checksum);
    printf("target blocking版本结果误差: %.6e\n\n", fabs(target_blocking_checksum - serial_checksum));
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
        }
    }
    
    // 测试target staged版本
    printf("运行target staged版本...\n");
    start_time = omp_get_wtime();
    matrix_multiply_target_staged(A, B, C, N);
    end_time = omp_get_wtime();
    double target_staged_time = end_time - start_time;
    printf("target staged版本运行时间: %.4f 秒\n", target_staged_time);
    printf("target staged版本加速比: %.2f\n\n", serial_time / target_staged_time);
    
    // 验证结果
    double target_staged_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            target_staged_checksum += C[i][j];
        }
    }
    printf("target staged版本结果校验和: %.6e\n", target_staged_checksum);
    printf("target staged版本结果误差: %.6e\n\n", fabs(target_staged_checksum - serial_checksum));
    
    // 性能总结
    printf("======= 性能总结 =======\n");
    printf("串行版本:             %.4f 秒 (加速比: 1.00)\n", serial_time);
    printf("parallel for版本:     %.4f 秒 (加速比: %.2f)\n", parallel_for_time, serial_time / parallel_for_time);
    printf("target版本:           %.4f 秒 (加速比: %.2f)\n", target_time, serial_time / target_time);
    printf("target blocking版本:  %.4f 秒 (加速比: %.2f)\n", target_blocking_time, serial_time / target_blocking_time);
    printf("target staged版本:    %.4f 秒 (加速比: %.2f)\n", target_staged_time, serial_time / target_staged_time);
    
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


// 运行线程数: 16

// 运行串行版本...
// 串行版本运行时间: 34.9910 秒
// 串行版本加速比: 1.00

// 串行版本结果校验和: 1.999779e+09

// 运行parallel for版本...
// parallel for版本运行时间: 4.6120 秒
// parallel for版本加速比: 7.59

// parallel for版本结果校验和: 1.999779e+09
// parallel for版本结果误差: 0.000000e+00

// 运行target版本...
// target版本运行时间: 0.6915 秒
// target版本加速比: 50.60

// target版本结果校验和: 1.999779e+09
// target版本结果误差: 0.000000e+00

// 运行target blocking版本...
// target blocking版本运行时间: 0.3653 秒
// target blocking版本加速比: 95.79

// target blocking版本结果校验和: 1.999779e+09
// target blocking版本结果误差: 0.000000e+00

// 运行target staged版本...
// target staged版本运行时间: 1.0806 秒
// target staged版本加速比: 32.38

// target staged版本结果校验和: 1.999779e+09
// target staged版本结果误差: 0.000000e+00

// ======= 性能总结 =======
// 串行版本:             34.9910 秒 (加速比: 1.00)
// parallel for版本:     4.6120 秒 (加速比: 7.59)
// target版本:           0.6915 秒 (加速比: 50.60)
// target blocking版本:  0.3653 秒 (加速比: 95.79)
// target staged版本:    1.0806 秒 (加速比: 32.38)



// 运行线程数: 64

// 运行串行版本...
// 串行版本运行时间: 34.1737 秒
// 串行版本加速比: 1.00

// 串行版本结果校验和: 1.999779e+09

// 运行parallel for版本...
// parallel for版本运行时间: 1.6871 秒
// parallel for版本加速比: 20.26

// parallel for版本结果校验和: 1.999779e+09
// parallel for版本结果误差: 0.000000e+00

// 运行target版本...
// target版本运行时间: 1.1505 秒
// target版本加速比: 29.70

// target版本结果校验和: 1.999779e+09
// target版本结果误差: 0.000000e+00

// 运行target blocking版本...
// target blocking版本运行时间: 0.4019 秒
// target blocking版本加速比: 85.03

// target blocking版本结果校验和: 1.999779e+09
// target blocking版本结果误差: 0.000000e+00

// 运行target staged版本...
// target staged版本运行时间: 1.1365 秒
// target staged版本加速比: 30.07

// target staged版本结果校验和: 1.999779e+09
// target staged版本结果误差: 0.000000e+00

// ======= 性能总结 =======
// 串行版本:             34.1737 秒 (加速比: 1.00)
// parallel for版本:     1.6871 秒 (加速比: 20.26)
// target版本:           1.1505 秒 (加速比: 29.70)
// target blocking版本:  0.4019 秒 (加速比: 85.03)
// target staged版本:    1.1365 秒 (加速比: 30.07)