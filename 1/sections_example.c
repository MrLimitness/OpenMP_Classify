#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define N 12000
#define M_PI 3.14159265358979323846

// 模拟繁重计算的函数
void heavy_computation(double* array, int size, int iterations) {
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < size; i++) {
            array[i] = sin(array[i]) * cos(array[i]) * tan(array[i]);
        }
    }
}

// 矩阵加法
void matrix_addition(double* A, double* B, double* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i*size + j] = A[i*size + j] + B[i*size + j];
        }
    }
}

// 矩阵转置
void matrix_transpose(double* A, double* B, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            B[j*size + i] = A[i*size + j];
        }
    }
}

// 一维傅里叶变换 (简化版)
void simple_fft(double* input, double* output_real, double* output_imag, int size) {
    for (int k = 0; k < size; k++) {
        output_real[k] = 0;
        output_imag[k] = 0;
        for (int n = 0; n < size; n++) {
            double angle = 2 * M_PI * k * n / size;
            output_real[k] += input[n] * cos(angle);
            output_imag[k] += -input[n] * sin(angle);
        }
    }
}

// 图像模糊 (简化版)
void image_blur(double* input, double* output, int size) {
    // 简单的3x3模糊核
    const int kernel_size = 3;
    const double kernel[3][3] = {
        {1.0/9, 1.0/9, 1.0/9},
        {1.0/9, 1.0/9, 1.0/9},
        {1.0/9, 1.0/9, 1.0/9}
    };
    
    // 对每个像素应用模糊核
    for (int i = 1; i < size-1; i++) {
        for (int j = 1; j < size-1; j++) {
            double sum = 0.0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += input[(i+ki)*size + (j+kj)] * kernel[ki+1][kj+1];
                }
            }
            output[i*size + j] = sum;
        }
    }
    
    // 处理边界
    for (int i = 0; i < size; i++) {
        output[i] = input[i];                   // 顶部边界
        output[i*size] = input[i*size];         // 左边界
        output[i*size + size-1] = input[i*size + size-1]; // 右边界
        output[(size-1)*size + i] = input[(size-1)*size + i]; // 底部边界
    }
}

// 串行版本
void serial_version(double* A, double* B, double* C, double* D, double* E, double* F, 
                   double* G, double* H, double* I, int size) {
    printf("执行串行版本...\n");
    double start_time = omp_get_wtime();
    
    // 任务1: 矩阵加法
    matrix_addition(A, B, C, size);
    
    // 任务2: 矩阵转置
    matrix_transpose(A, D, size);
    
    // 任务3: 简单FFT
    simple_fft(E, F, G, size);
    
    // 任务4: 图像模糊
    image_blur(H, I, size);
    
    double end_time = omp_get_wtime();
    printf("串行版本执行时间: %.6f 秒\n", end_time - start_time);
    printf("加速比: 1.00\n\n");
    
    return;
}

// 使用parallel for的版本
void parallel_for_version(double* A, double* B, double* C, double* D, double* E, double* F, 
                         double* G, double* H, double* I, int size, int num_threads) {
    printf("执行 parallel for 版本 (线程数: %d)...\n", num_threads);
    double start_time = omp_get_wtime();
    
    // 任务1: 矩阵加法 (使用parallel for)
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i*size + j] = A[i*size + j] + B[i*size + j];
        }
    }
    
    // 任务2: 矩阵转置 (使用parallel for)
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            D[j*size + i] = A[i*size + j];
        }
    }
    
    // 任务3: 简单FFT (使用parallel for)
    #pragma omp parallel for num_threads(num_threads)
    for (int k = 0; k < size; k++) {
        F[k] = 0;
        G[k] = 0;
        for (int n = 0; n < size; n++) {
            double angle = 2 * M_PI * k * n / size;
            F[k] += E[n] * cos(angle);
            G[k] += -E[n] * sin(angle);
        }
    }
    
    // 任务4: 图像模糊 (使用parallel for)
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 1; i < size-1; i++) {
        for (int j = 1; j < size-1; j++) {
            double sum = 0.0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += H[(i+ki)*size + (j+kj)] * (1.0/9.0);
                }
            }
            I[i*size + j] = sum;
        }
    }
    
    // 处理边界
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; i++) {
        I[i] = H[i];                   // 顶部边界
        I[i*size] = H[i*size];         // 左边界
        I[i*size + size-1] = H[i*size + size-1]; // 右边界
        I[(size-1)*size + i] = H[(size-1)*size + i]; // 底部边界
    }
    
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("parallel for 版本执行时间: %.6f 秒\n", elapsed);
    
    return;
}

// 使用sections版本1: 基本sections
void sections_version1(double* A, double* B, double* C, double* D, double* E, double* F, 
                      double* G, double* H, double* I, int size, int num_threads) {
    printf("执行 sections 版本1 (线程数: %d)...\n", num_threads);
    double start_time = omp_get_wtime();
    
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                // 任务1: 矩阵加法
                matrix_addition(A, B, C, size);
            }
            
            #pragma omp section
            {
                // 任务2: 矩阵转置
                matrix_transpose(A, D, size);
            }
            
            #pragma omp section
            {
                // 任务3: 简单FFT
                simple_fft(E, F, G, size);
            }
            
            #pragma omp section
            {
                // 任务4: 图像模糊
                image_blur(H, I, size);
            }
        }
    }
    
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("sections 版本1执行时间: %.6f 秒\n", elapsed);
    
    return;
}

// 使用sections版本2: sections + 内部parallel for
void sections_version2(double* A, double* B, double* C, double* D, double* E, double* F, 
                      double* G, double* H, double* I, int size, int num_threads) {
    printf("执行 sections 版本2 (线程数: %d)...\n", num_threads);
    double start_time = omp_get_wtime();
    
    int threads_per_section = num_threads / 4; // 每个section使用的线程数
    if (threads_per_section < 1) threads_per_section = 1;
    
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                // 任务1: 矩阵加法 (使用内部parallel for)
                #pragma omp parallel for num_threads(threads_per_section)
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        C[i*size + j] = A[i*size + j] + B[i*size + j];
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务2: 矩阵转置 (使用内部parallel for)
                #pragma omp parallel for num_threads(threads_per_section)
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        D[j*size + i] = A[i*size + j];
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务3: 简单FFT (使用内部parallel for)
                #pragma omp parallel for num_threads(threads_per_section)
                for (int k = 0; k < size; k++) {
                    F[k] = 0;
                    G[k] = 0;
                    for (int n = 0; n < size; n++) {
                        double angle = 2 * M_PI * k * n / size;
                        F[k] += E[n] * cos(angle);
                        G[k] += -E[n] * sin(angle);
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务4: 图像模糊 (使用内部parallel for)
                #pragma omp parallel for num_threads(threads_per_section)
                for (int i = 1; i < size-1; i++) {
                    for (int j = 1; j < size-1; j++) {
                        double sum = 0.0;
                        for (int ki = -1; ki <= 1; ki++) {
                            for (int kj = -1; kj <= 1; kj++) {
                                sum += H[(i+ki)*size + (j+kj)] * (1.0/9.0);
                            }
                        }
                        I[i*size + j] = sum;
                    }
                }
                
                // 处理边界
                #pragma omp parallel for num_threads(threads_per_section)
                for (int i = 0; i < size; i++) {
                    I[i] = H[i];
                    I[i*size] = H[i*size];
                    I[i*size + size-1] = H[i*size + size-1];
                    I[(size-1)*size + i] = H[(size-1)*size + i];
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("sections 版本2执行时间: %.6f 秒\n", elapsed);
    
    return;
}

// 使用sections版本3: sections + 调整任务粒度
void sections_version3(double* A, double* B, double* C, double* D, double* E, double* F, 
                      double* G, double* H, double* I, int size, int num_threads) {
    printf("执行 sections 版本3 (线程数: %d)...\n", num_threads);
    double start_time = omp_get_wtime();
    
    // 更细粒度的任务划分，每个section只负责更小的工作量
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp sections
        {
            // 矩阵加法 - 拆分为两部分
            #pragma omp section
            {
                for (int i = 0; i < size/2; i++) {
                    for (int j = 0; j < size; j++) {
                        C[i*size + j] = A[i*size + j] + B[i*size + j];
                    }
                }
            }
            
            #pragma omp section
            {
                for (int i = size/2; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        C[i*size + j] = A[i*size + j] + B[i*size + j];
                    }
                }
            }
            
            // 矩阵转置 - 拆分为两部分
            #pragma omp section
            {
                for (int i = 0; i < size/2; i++) {
                    for (int j = 0; j < size; j++) {
                        D[j*size + i] = A[i*size + j];
                    }
                }
            }
            
            #pragma omp section
            {
                for (int i = size/2; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        D[j*size + i] = A[i*size + j];
                    }
                }
            }
            
            // FFT - 拆分为两部分
            #pragma omp section
            {
                for (int k = 0; k < size/2; k++) {
                    F[k] = 0;
                    G[k] = 0;
                    for (int n = 0; n < size; n++) {
                        double angle = 2 * M_PI * k * n / size;
                        F[k] += E[n] * cos(angle);
                        G[k] += -E[n] * sin(angle);
                    }
                }
            }
            
            #pragma omp section
            {
                for (int k = size/2; k < size; k++) {
                    F[k] = 0;
                    G[k] = 0;
                    for (int n = 0; n < size; n++) {
                        double angle = 2 * M_PI * k * n / size;
                        F[k] += E[n] * cos(angle);
                        G[k] += -E[n] * sin(angle);
                    }
                }
            }
            
            // 图像模糊 - 拆分为两部分
            #pragma omp section
            {
                for (int i = 1; i < size/2; i++) {
                    for (int j = 1; j < size-1; j++) {
                        double sum = 0.0;
                        for (int ki = -1; ki <= 1; ki++) {
                            for (int kj = -1; kj <= 1; kj++) {
                                sum += H[(i+ki)*size + (j+kj)] * (1.0/9.0);
                            }
                        }
                        I[i*size + j] = sum;
                    }
                }
            }
            
            #pragma omp section
            {
                for (int i = size/2; i < size-1; i++) {
                    for (int j = 1; j < size-1; j++) {
                        double sum = 0.0;
                        for (int ki = -1; ki <= 1; ki++) {
                            for (int kj = -1; kj <= 1; kj++) {
                                sum += H[(i+ki)*size + (j+kj)] * (1.0/9.0);
                            }
                        }
                        I[i*size + j] = sum;
                    }
                }
            }
        }
    }
    
    // 处理边界 - 单独处理
    for (int i = 0; i < size; i++) {
        I[i] = H[i];
        I[i*size] = H[i*size];
        I[i*size + size-1] = H[i*size + size-1];
        I[(size-1)*size + i] = H[(size-1)*size + i];
    }
    
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("sections 版本3执行时间: %.6f 秒\n", elapsed);
    
    return;
}

// 使用sections版本4: 嵌套sections
void sections_version4(double* A, double* B, double* C, double* D, double* E, double* F, 
                      double* G, double* H, double* I, int size, int num_threads) {
    printf("执行 sections 版本4 (线程数: %d)...\n", num_threads);
    double start_time = omp_get_wtime();
    
    omp_set_nested(1); // 启用嵌套并行
    
    #pragma omp parallel num_threads(2)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                // 第一组：矩阵运算
                #pragma omp parallel num_threads(num_threads/2)
                {
                    #pragma omp sections
                    {
                        #pragma omp section
                        {
                            // 任务1: 矩阵加法
                            matrix_addition(A, B, C, size);
                        }
                        
                        #pragma omp section
                        {
                            // 任务2: 矩阵转置
                            matrix_transpose(A, D, size);
                        }
                    }
                }
            }
            
            #pragma omp section
            {
                // 第二组：信号处理和图像处理
                #pragma omp parallel num_threads(num_threads/2)
                {
                    #pragma omp sections
                    {
                        #pragma omp section
                        {
                            // 任务3: 简单FFT
                            simple_fft(E, F, G, size);
                        }
                        
                        #pragma omp section
                        {
                            // 任务4: 图像模糊
                            image_blur(H, I, size);
                        }
                    }
                }
            }
        }
    }
    
    omp_set_nested(0); // 禁用嵌套并行
    
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("sections 版本4执行时间: %.6f 秒\n", elapsed);
    
    return;
}

int main(int argc, char* argv[]) {
    int num_threads = 64;  // 默认线程数
    
    // 检查命令行参数
    if (argc >= 2) {
        num_threads = atoi(argv[1]);
    }
    
    printf("设置线程数: %d\n\n", num_threads);
    
    // 分配矩阵和数组内存
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)malloc(N * N * sizeof(double));
    double *D = (double*)malloc(N * N * sizeof(double));
    double *E = (double*)malloc(N * sizeof(double));
    double *F = (double*)malloc(N * sizeof(double));
    double *G = (double*)malloc(N * sizeof(double));
    double *H = (double*)malloc(N * N * sizeof(double));
    double *I = (double*)malloc(N * N * sizeof(double));
    
    // 检查内存分配
    if (!A || !B || !C || !D || !E || !F || !G || !H || !I) {
        printf("内存分配失败！\n");
        exit(1);
    }
    
    // 初始化数据
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i*N + j] = (double)rand() / RAND_MAX;
            B[i*N + j] = (double)rand() / RAND_MAX;
            H[i*N + j] = (double)rand() / RAND_MAX;
        }
        E[i] = (double)rand() / RAND_MAX;
    }
    
    // 保存原始数据副本用于各个版本测试
    double *A_copy = (double*)malloc(N * N * sizeof(double));
    double *B_copy = (double*)malloc(N * N * sizeof(double));
    double *E_copy = (double*)malloc(N * sizeof(double));
    double *H_copy = (double*)malloc(N * N * sizeof(double));
    
    memcpy(A_copy, A, N * N * sizeof(double));
    memcpy(B_copy, B, N * N * sizeof(double));
    memcpy(E_copy, E, N * sizeof(double));
    memcpy(H_copy, H, N * N * sizeof(double));
    
    // 串行版本
    double start_serial = omp_get_wtime();
    serial_version(A, B, C, D, E, F, G, H, I, N);
    double serial_time = omp_get_wtime() - start_serial;
    
    // 重置数据
    memcpy(A, A_copy, N * N * sizeof(double));
    memcpy(B, B_copy, N * N * sizeof(double));
    memcpy(E, E_copy, N * sizeof(double));
    memcpy(H, H_copy, N * N * sizeof(double));
    
    // Parallel For 版本
    double start_pfor = omp_get_wtime();
    parallel_for_version(A, B, C, D, E, F, G, H, I, N, num_threads);
    double pfor_time = omp_get_wtime() - start_pfor;
    printf("加速比: %.2f\n\n", serial_time / pfor_time);
    
    // 重置数据
    memcpy(A, A_copy, N * N * sizeof(double));
    memcpy(B, B_copy, N * N * sizeof(double));
    memcpy(E, E_copy, N * sizeof(double));
    memcpy(H, H_copy, N * N * sizeof(double));
    
    // Sections 版本1
    double start_sec1 = omp_get_wtime();
    sections_version1(A, B, C, D, E, F, G, H, I, N, num_threads);
    double sec1_time = omp_get_wtime() - start_sec1;
    printf("加速比: %.2f\n\n", serial_time / sec1_time);
    
    // 重置数据
    memcpy(A, A_copy, N * N * sizeof(double));
    memcpy(B, B_copy, N * N * sizeof(double));
    memcpy(E, E_copy, N * sizeof(double));
    memcpy(H, H_copy, N * N * sizeof(double));
    
    // Sections 版本2
    double start_sec2 = omp_get_wtime();
    sections_version2(A, B, C, D, E, F, G, H, I, N, num_threads);
    double sec2_time = omp_get_wtime() - start_sec2;
    printf("加速比: %.2f\n\n", serial_time / sec2_time);
    
    // 重置数据
    memcpy(A, A_copy, N * N * sizeof(double));
    memcpy(B, B_copy, N * N * sizeof(double));
    memcpy(E, E_copy, N * sizeof(double));
    memcpy(H, H_copy, N * N * sizeof(double));
    
    // Sections 版本3
    double start_sec3 = omp_get_wtime();
    sections_version3(A, B, C, D, E, F, G, H, I, N, num_threads);
    double sec3_time = omp_get_wtime() - start_sec3;
    printf("加速比: %.2f\n\n", serial_time / sec3_time);
    
    // 重置数据
    memcpy(A, A_copy, N * N * sizeof(double));
    memcpy(B, B_copy, N * N * sizeof(double));
    memcpy(E, E_copy, N * sizeof(double));
    memcpy(H, H_copy, N * N * sizeof(double));
    
    // Sections 版本4
    double start_sec4 = omp_get_wtime();
    sections_version4(A, B, C, D, E, F, G, H, I, N, num_threads);
    double sec4_time = omp_get_wtime() - start_sec4;
    printf("加速比: %.2f\n\n", serial_time / sec4_time);
    
    // 释放内存
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    free(F);
    free(G);
    free(H);
    free(I);
    free(A_copy);
    free(B_copy);
    free(E_copy);
    free(H_copy);
    
    return 0;
}

// 设置线程数: 16

// 执行串行版本...
// 串行版本执行时间: 11.273945 秒
// 加速比: 1.00

// 执行 parallel for 版本 (线程数: 16)...
// parallel for 版本执行时间: 1.582553 秒
// 加速比: 7.12

// 执行 sections 版本1 (线程数: 16)...
// sections 版本1执行时间: 4.910498 秒
// 加速比: 2.30

// 执行 sections 版本2 (线程数: 16)...
// sections 版本2执行时间: 5.022881 秒
// 加速比: 2.24

// 执行 sections 版本3 (线程数: 16)...
// sections 版本3执行时间: 2.549349 秒
// 加速比: 4.42

// 执行 sections 版本4 (线程数: 16)...
// sections 版本4执行时间: 5.056817 秒
// 加速比: 2.23


// 设置线程数: 64

// 执行串行版本...
// 串行版本执行时间: 11.821234 秒
// 加速比: 1.00

// 执行 parallel for 版本 (线程数: 64)...
// parallel for 版本执行时间: 0.708103 秒
// 加速比: 16.69

// 执行 sections 版本1 (线程数: 64)...
// sections 版本1执行时间: 4.525682 秒
// 加速比: 2.61

// 执行 sections 版本2 (线程数: 64)...
// sections 版本2执行时间: 4.629313 秒
// 加速比: 2.55

// 执行 sections 版本3 (线程数: 64)...
// sections 版本3执行时间: 2.970516 秒
// 加速比: 3.98

// 执行 sections 版本4 (线程数: 64)...
// sections 版本4执行时间: 5.050735 秒
// 加速比: 2.34