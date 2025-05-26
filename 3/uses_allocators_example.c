#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <string.h>

// 定义矩阵大小
#define N 5000

// 串行版本：密集矩阵-向量乘法
void matrix_vector_multiply_serial(double **A, double *x, double *y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
}

// 并行版本1：基本parallel for的矩阵-向量乘法
void matrix_vector_multiply_parallel(double **A, double *x, double *y, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
}

// 并行版本2：使用uses_allocators的矩阵-向量乘法
void matrix_vector_multiply_uses_allocators(double **A, double *x, double *y, int n) {
    // 使用自定义分配器和uses_allocators
    #pragma omp parallel uses_allocators(omp_high_bw_mem_alloc)
    {
        double *local_y = (double *)omp_alloc(n * sizeof(double), omp_high_bw_mem_alloc);
        memset(local_y, 0, n * sizeof(double));
        
        #pragma omp for
        for (int i = 0; i < n; i++) {
            local_y[i] = 0.0;
            for (int j = 0; j < n; j++) {
                local_y[i] += A[i][j] * x[j];
            }
        }
        
        // 将结果复制回全局数组
        #pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                y[i] = local_y[i];
            }
        }
        
        omp_free(local_y, omp_high_bw_mem_alloc);
    }
}

// 并行版本3：使用uses_allocators和分块策略的矩阵-向量乘法
void matrix_vector_multiply_uses_allocators_blocked(double **A, double *x, double *y, int n) {
    const int block_size = 64;
    
    #pragma omp parallel uses_allocators(omp_high_bw_mem_alloc)
    {
        double *local_y = (double *)omp_alloc(n * sizeof(double), omp_high_bw_mem_alloc);
        double *block_x = (double *)omp_alloc(block_size * sizeof(double), omp_high_bw_mem_alloc);
        
        memset(local_y, 0, n * sizeof(double));
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            local_y[i] = 0.0;
            for (int jb = 0; jb < n; jb += block_size) {
                int jlim = (jb + block_size < n) ? jb + block_size : n;
                
                // 预加载块
                for (int j = jb; j < jlim; j++) {
                    block_x[j - jb] = x[j];
                }
                
                // 计算块
                for (int j = jb; j < jlim; j++) {
                    local_y[i] += A[i][j] * block_x[j - jb];
                }
            }
        }
        
        // 合并结果
        #pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                y[i] = local_y[i];
            }
        }
        
        omp_free(block_x, omp_high_bw_mem_alloc);
        omp_free(local_y, omp_high_bw_mem_alloc);
    }
}

// 并行版本4：使用uses_allocators和预分配缓存友好布局的矩阵-向量乘法
void matrix_vector_multiply_uses_allocators_cache_friendly(double **A, double *x, double *y, int n) {
    #pragma omp parallel uses_allocators(omp_high_bw_mem_alloc, omp_large_cap_mem_alloc)
    {
        // 预分配本地结果数组（高带宽内存）
        double *local_y = (double *)omp_alloc(n * sizeof(double), omp_high_bw_mem_alloc);
        
        // 预分配转置矩阵（大容量内存）
        double **A_T = (double **)omp_alloc(n * sizeof(double *), omp_large_cap_mem_alloc);
        for (int i = 0; i < n; i++) {
            A_T[i] = (double *)omp_alloc(n * sizeof(double), omp_large_cap_mem_alloc);
        }
        
        // 单线程转置矩阵以提高缓存命中率
        #pragma omp single
        {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    A_T[j][i] = A[i][j];
                }
            }
        }
        
        // 初始化本地结果
        memset(local_y, 0, n * sizeof(double));
        
        // 使用转置矩阵计算
        #pragma omp for schedule(guided)
        for (int i = 0; i < n; i++) {
            local_y[i] = 0.0;
            for (int j = 0; j < n; j++) {
                local_y[i] += A_T[j][i] * x[j];
            }
        }
        
        // 合并结果
        #pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                y[i] = local_y[i];
            }
        }
        
        // 释放转置矩阵
        for (int i = 0; i < n; i++) {
            omp_free(A_T[i], omp_large_cap_mem_alloc);
        }
        omp_free(A_T, omp_large_cap_mem_alloc);
        omp_free(local_y, omp_high_bw_mem_alloc);
    }
}

int main() {
    int num_threads = 16; // 可以修改为16/64/144等
    omp_set_num_threads(num_threads);
    
    printf("矩阵大小: %d x %d\n", N, N);
    printf("运行线程数: %d\n\n", num_threads);
    
    // 分配内存
    double **A = (double **)malloc(N * sizeof(double *));
    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));
    
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
    }
    
    // 初始化矩阵和向量
    srand(12345);
    for (int i = 0; i < N; i++) {
        x[i] = (double)rand() / RAND_MAX;
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
        }
    }
    
    // 测试串行版本
    printf("运行串行版本...\n");
    double start_time = omp_get_wtime();
    matrix_vector_multiply_serial(A, x, y, N);
    double end_time = omp_get_wtime();
    double serial_time = end_time - start_time;
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    printf("串行版本加速比: 1.00\n\n");
    
    // 计算串行结果的校验和
    double serial_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        serial_checksum += y[i];
    }
    printf("串行版本结果校验和: %.6e\n\n", serial_checksum);
    
    // 重置结果向量
    memset(y, 0, N * sizeof(double));
    
    // 测试基本parallel for版本
    printf("运行基本parallel for版本...\n");
    start_time = omp_get_wtime();
    matrix_vector_multiply_parallel(A, x, y, N);
    end_time = omp_get_wtime();
    double parallel_time = end_time - start_time;
    printf("基本parallel for版本运行时间: %.4f 秒\n", parallel_time);
    printf("基本parallel for版本加速比: %.2f\n\n", serial_time / parallel_time);
    
    // 验证结果
    double parallel_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        parallel_checksum += y[i];
    }
    printf("基本parallel for版本结果校验和: %.6e\n", parallel_checksum);
    printf("基本parallel for版本结果误差: %.6e\n\n", fabs(parallel_checksum - serial_checksum));
    
    // 重置结果向量
    memset(y, 0, N * sizeof(double));
    
    // 测试uses_allocators版本
    printf("运行uses_allocators版本...\n");
    start_time = omp_get_wtime();
    matrix_vector_multiply_uses_allocators(A, x, y, N);
    end_time = omp_get_wtime();
    double uses_allocators_time = end_time - start_time;
    printf("uses_allocators版本运行时间: %.4f 秒\n", uses_allocators_time);
    printf("uses_allocators版本加速比: %.2f\n\n", serial_time / uses_allocators_time);
    
    // 验证结果
    double uses_allocators_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        uses_allocators_checksum += y[i];
    }
    printf("uses_allocators版本结果校验和: %.6e\n", uses_allocators_checksum);
    printf("uses_allocators版本结果误差: %.6e\n\n", fabs(uses_allocators_checksum - serial_checksum));
    
    // 重置结果向量
    memset(y, 0, N * sizeof(double));
    
    // 测试uses_allocators_blocked版本
    printf("运行uses_allocators_blocked版本...\n");
    start_time = omp_get_wtime();
    matrix_vector_multiply_uses_allocators_blocked(A, x, y, N);
    end_time = omp_get_wtime();
    double uses_allocators_blocked_time = end_time - start_time;
    printf("uses_allocators_blocked版本运行时间: %.4f 秒\n", uses_allocators_blocked_time);
    printf("uses_allocators_blocked版本加速比: %.2f\n\n", serial_time / uses_allocators_blocked_time);
    
    // 验证结果
    double uses_allocators_blocked_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        uses_allocators_blocked_checksum += y[i];
    }
    printf("uses_allocators_blocked版本结果校验和: %.6e\n", uses_allocators_blocked_checksum);
    printf("uses_allocators_blocked版本结果误差: %.6e\n\n", fabs(uses_allocators_blocked_checksum - serial_checksum));
    
    // 重置结果向量
    memset(y, 0, N * sizeof(double));
    
    // 测试uses_allocators_cache_friendly版本
    printf("运行uses_allocators_cache_friendly版本...\n");
    start_time = omp_get_wtime();
    matrix_vector_multiply_uses_allocators_cache_friendly(A, x, y, N);
    end_time = omp_get_wtime();
    double uses_allocators_cache_friendly_time = end_time - start_time;
    printf("uses_allocators_cache_friendly版本运行时间: %.4f 秒\n", uses_allocators_cache_friendly_time);
    printf("uses_allocators_cache_friendly版本加速比: %.2f\n\n", serial_time / uses_allocators_cache_friendly_time);
    
    // 验证结果
    double uses_allocators_cache_friendly_checksum = 0.0;
    for (int i = 0; i < N; i++) {
        uses_allocators_cache_friendly_checksum += y[i];
    }
    printf("uses_allocators_cache_friendly版本结果校验和: %.6e\n", uses_allocators_cache_friendly_checksum);
    printf("uses_allocators_cache_friendly版本结果误差: %.6e\n\n", fabs(uses_allocators_cache_friendly_checksum - serial_checksum));
    
    // 性能总结
    printf("======= 性能总结 =======\n");
    printf("串行版本:                        %.4f 秒 (加速比: 1.00)\n", serial_time);
    printf("基本parallel for版本:            %.4f 秒 (加速比: %.2f)\n", parallel_time, serial_time / parallel_time);
    printf("uses_allocators版本:             %.4f 秒 (加速比: %.2f)\n", uses_allocators_time, serial_time / uses_allocators_time);
    printf("uses_allocators_blocked版本:     %.4f 秒 (加速比: %.2f)\n", uses_allocators_blocked_time, serial_time / uses_allocators_blocked_time);
    printf("uses_allocators_cache_friendly版本: %.4f 秒 (加速比: %.2f)\n", uses_allocators_cache_friendly_time, serial_time / uses_allocators_cache_friendly_time);
    
    // 释放内存
    for (int i = 0; i < N; i++) {
        free(A[i]);
    }
    free(A);
    free(x);
    free(y);
    
    return 0;
}