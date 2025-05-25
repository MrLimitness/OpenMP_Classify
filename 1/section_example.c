#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>

// 定义矩阵大小
#define SIZE 2000

// 全局变量
double A[SIZE][SIZE];
double B[SIZE][SIZE];
double C[SIZE][SIZE];
double D[SIZE][SIZE];
double E[SIZE][SIZE];
double F[SIZE][SIZE];

// 初始化矩阵
void initialize_matrices() {
    int i, j;
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
            C[i][j] = 0.0;
            D[i][j] = 0.0;
            E[i][j] = 0.0;
            F[i][j] = 0.0;
        }
    }
}

// 串行版本：处理多个矩阵计算任务
void serial_process() {
    int i, j, k;
    printf("串行版本开始执行...\n");
    
    // 任务1：矩阵乘法 C = A * B
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            C[i][j] = 0.0;
            for (k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    // 任务2：矩阵转置 D = B^T
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            D[i][j] = B[j][i];
        }
    }
    
    // 任务3：矩阵加法 E = A + B
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            E[i][j] = A[i][j] + B[i][j];
        }
    }
    
    // 任务4：矩阵对角线元素计算
    for (i = 0; i < SIZE; i++) {
        F[i][i] = A[i][i] * B[i][i];
        // 模拟一些额外工作
        usleep(100);
    }
}

// 使用parallel for版本
void parallel_for_process() {
    int i, j, k;
    printf("parallel for版本开始执行...\n");
    
    // 任务1：矩阵乘法 C = A * B
    #pragma omp parallel for private(j, k) shared(A, B, C)
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            C[i][j] = 0.0;
            for (k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    // 任务2：矩阵转置 D = B^T
    #pragma omp parallel for private(j) shared(B, D)
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            D[i][j] = B[j][i];
        }
    }
    
    // 任务3：矩阵加法 E = A + B
    #pragma omp parallel for private(j) shared(A, B, E)
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            E[i][j] = A[i][j] + B[i][j];
        }
    }
    
    // 任务4：矩阵对角线元素计算
    #pragma omp parallel for shared(A, B, F)
    for (i = 0; i < SIZE; i++) {
        F[i][i] = A[i][i] * B[i][i];
        // 模拟一些额外工作
        usleep(100);
    }
}

// 使用sections版本
void sections_process() {
    printf("sections版本开始执行...\n");
    
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                // 任务1：矩阵乘法 C = A * B
                int i, j, k;
                for (i = 0; i < SIZE; i++) {
                    for (j = 0; j < SIZE; j++) {
                        C[i][j] = 0.0;
                        for (k = 0; k < SIZE; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务2：矩阵转置 D = B^T
                int i, j;
                for (i = 0; i < SIZE; i++) {
                    for (j = 0; j < SIZE; j++) {
                        D[i][j] = B[j][i];
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务3：矩阵加法 E = A + B
                int i, j;
                for (i = 0; i < SIZE; i++) {
                    for (j = 0; j < SIZE; j++) {
                        E[i][j] = A[i][j] + B[i][j];
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务4：矩阵对角线元素计算
                int i;
                for (i = 0; i < SIZE; i++) {
                    F[i][i] = A[i][i] * B[i][i];
                    // 模拟一些额外工作
                    usleep(100);
                }
            }
        }
    }
}

// 使用sections+parallel for组合版本
void sections_parallel_for_process() {
    printf("sections+parallel for组合版本开始执行...\n");
    
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                // 任务1：矩阵乘法 C = A * B，内部使用parallel for
                int i, j, k;
                #pragma omp parallel for private(j, k) shared(A, B, C)
                for (i = 0; i < SIZE; i++) {
                    for (j = 0; j < SIZE; j++) {
                        C[i][j] = 0.0;
                        for (k = 0; k < SIZE; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务2：矩阵转置 D = B^T，内部使用parallel for
                int i, j;
                #pragma omp parallel for private(j) shared(B, D)
                for (i = 0; i < SIZE; i++) {
                    for (j = 0; j < SIZE; j++) {
                        D[i][j] = B[j][i];
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务3：矩阵加法 E = A + B，内部使用parallel for
                int i, j;
                #pragma omp parallel for private(j) shared(A, B, E)
                for (i = 0; i < SIZE; i++) {
                    for (j = 0; j < SIZE; j++) {
                        E[i][j] = A[i][j] + B[i][j];
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务4：矩阵对角线元素计算，内部使用parallel for
                int i;
                #pragma omp parallel for shared(A, B, F)
                for (i = 0; i < SIZE; i++) {
                    F[i][i] = A[i][i] * B[i][i];
                    // 模拟一些额外工作
                    usleep(100);
                }
            }
        }
    }
}

// 使用sections+任务分块版本
void sections_chunked_process() {
    printf("sections+任务分块版本开始执行...\n");
    
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                // 任务1：矩阵乘法 C = A * B 前半部分
                int i, j, k;
                for (i = 0; i < SIZE/2; i++) {
                    for (j = 0; j < SIZE; j++) {
                        C[i][j] = 0.0;
                        for (k = 0; k < SIZE; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务1：矩阵乘法 C = A * B 后半部分
                int i, j, k;
                for (i = SIZE/2; i < SIZE; i++) {
                    for (j = 0; j < SIZE; j++) {
                        C[i][j] = 0.0;
                        for (k = 0; k < SIZE; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务2：矩阵转置 D = B^T
                int i, j;
                for (i = 0; i < SIZE; i++) {
                    for (j = 0; j < SIZE; j++) {
                        D[i][j] = B[j][i];
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务3和4：矩阵加法 E = A + B 和对角线计算
                int i, j;
                for (i = 0; i < SIZE; i++) {
                    for (j = 0; j < SIZE; j++) {
                        E[i][j] = A[i][j] + B[i][j];
                    }
                    // 对角线元素计算
                    F[i][i] = A[i][i] * B[i][i];
                    // 模拟一些额外工作
                    usleep(100);
                }
            }
        }
    }
}

// 使用sections+嵌套并行版本
void sections_nested_process() {
    printf("sections+嵌套并行版本开始执行...\n");
    
    omp_set_nested(1); // 允许嵌套并行
    
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                // 任务1：矩阵乘法 C = A * B
                #pragma omp parallel
                {
                    int i, j, k;
                    int tid = omp_get_thread_num();
                    int nthreads = omp_get_num_threads();
                    int chunk = SIZE / nthreads;
                    int start = tid * chunk;
                    int end = (tid == nthreads - 1) ? SIZE : start + chunk;
                    
                    for (i = start; i < end; i++) {
                        for (j = 0; j < SIZE; j++) {
                            C[i][j] = 0.0;
                            for (k = 0; k < SIZE; k++) {
                                C[i][j] += A[i][k] * B[k][j];
                            }
                        }
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务2：矩阵转置 D = B^T
                #pragma omp parallel
                {
                    int i, j;
                    int tid = omp_get_thread_num();
                    int nthreads = omp_get_num_threads();
                    int chunk = SIZE / nthreads;
                    int start = tid * chunk;
                    int end = (tid == nthreads - 1) ? SIZE : start + chunk;
                    
                    for (i = start; i < end; i++) {
                        for (j = 0; j < SIZE; j++) {
                            D[i][j] = B[j][i];
                        }
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务3：矩阵加法 E = A + B
                int i, j;
                for (i = 0; i < SIZE; i++) {
                    for (j = 0; j < SIZE; j++) {
                        E[i][j] = A[i][j] + B[i][j];
                    }
                }
            }
            
            #pragma omp section
            {
                // 任务4：矩阵对角线元素计算
                int i;
                for (i = 0; i < SIZE; i++) {
                    F[i][i] = A[i][i] * B[i][i];
                    // 模拟一些额外工作
                    usleep(100);
                }
            }
        }
    }
    
    omp_set_nested(0); // 重置嵌套并行设置
}

int main(int argc, char *argv[]) {
    int num_threads = 64; // 默认线程数
    
    // 如果提供了命令行参数，使用命令行指定的线程数
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    
    printf("设置OpenMP线程数为: %d\n", num_threads);
    omp_set_num_threads(num_threads);
    
    // 初始化随机数生成器
    srand(time(NULL));
    
    // 初始化矩阵
    initialize_matrices();
    
    double start_time, end_time;
    double serial_time, parallel_for_time, sections_time, sections_parallel_for_time, sections_chunked_time, sections_nested_time;
    
    // 运行串行版本并计时
    start_time = omp_get_wtime();
    serial_process();
    end_time = omp_get_wtime();
    serial_time = end_time - start_time;
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    
    // 重新初始化矩阵
    initialize_matrices();
    
    // 运行parallel for版本并计时
    start_time = omp_get_wtime();
    parallel_for_process();
    end_time = omp_get_wtime();
    parallel_for_time = end_time - start_time;
    printf("parallel for版本运行时间: %.4f 秒, 加速比: %.2f\n", parallel_for_time, serial_time/parallel_for_time);
    
    // 重新初始化矩阵
    initialize_matrices();
    
    // 运行sections版本并计时
    start_time = omp_get_wtime();
    sections_process();
    end_time = omp_get_wtime();
    sections_time = end_time - start_time;
    printf("sections版本运行时间: %.4f 秒, 加速比: %.2f\n", sections_time, serial_time/sections_time);
    
    // 重新初始化矩阵
    initialize_matrices();
    
    // 运行sections+parallel for组合版本并计时
    start_time = omp_get_wtime();
    sections_parallel_for_process();
    end_time = omp_get_wtime();
    sections_parallel_for_time = end_time - start_time;
    printf("sections+parallel for组合版本运行时间: %.4f 秒, 加速比: %.2f\n", sections_parallel_for_time, serial_time/sections_parallel_for_time);
    
    // 重新初始化矩阵
    initialize_matrices();
    
    // 运行sections+任务分块版本并计时
    start_time = omp_get_wtime();
    sections_chunked_process();
    end_time = omp_get_wtime();
    sections_chunked_time = end_time - start_time;
    printf("sections+任务分块版本运行时间: %.4f 秒, 加速比: %.2f\n", sections_chunked_time, serial_time/sections_chunked_time);
    
    // 重新初始化矩阵
    initialize_matrices();
    
    // 运行sections+嵌套并行版本并计时
    start_time = omp_get_wtime();
    sections_nested_process();
    end_time = omp_get_wtime();
    sections_nested_time = end_time - start_time;
    printf("sections+嵌套并行版本运行时间: %.4f 秒, 加速比: %.2f\n", sections_nested_time, serial_time/sections_nested_time);
    
    // 输出各版本加速比对比
    printf("\n各版本加速比对比:\n");
    printf("串行版本: 1.00\n");
    printf("parallel for版本: %.2f\n", serial_time/parallel_for_time);
    printf("sections版本: %.2f\n", serial_time/sections_time);
    printf("sections+parallel for组合版本: %.2f\n", serial_time/sections_parallel_for_time);
    printf("sections+任务分块版本: %.2f\n", serial_time/sections_chunked_time);
    printf("sections+嵌套并行版本: %.2f\n", serial_time/sections_nested_time);
    
    return 0;
}

// 设置OpenMP线程数为: 16
// 串行版本开始执行...
// 串行版本运行时间: 22.3961 秒
// parallel for版本开始执行...
// parallel for版本运行时间: 3.7746 秒, 加速比: 5.93
// sections版本开始执行...
// sections版本运行时间: 32.2134 秒, 加速比: 0.70
// sections+parallel for组合版本开始执行...
// sections+parallel for组合版本运行时间: 28.7752 秒, 加速比: 0.78
// sections+任务分块版本开始执行...
// sections+任务分块版本运行时间: 15.8261 秒, 加速比: 1.42
// sections+嵌套并行版本开始执行...
// sections+嵌套并行版本运行时间: 4.0259 秒, 加速比: 5.56

// 各版本加速比对比:
// 串行版本: 1.00
// parallel for版本: 5.93
// sections版本: 0.70
// sections+parallel for组合版本: 0.78
// sections+任务分块版本: 1.42
// sections+嵌套并行版本: 5.56

// 设置OpenMP线程数为: 64
// 串行版本开始执行...
// 串行版本运行时间: 32.2812 秒
// parallel for版本开始执行...
// parallel for版本运行时间: 1.3119 秒, 加速比: 24.61
// sections版本开始执行...
// sections版本运行时间: 31.3382 秒, 加速比: 1.03
// sections+parallel for组合版本开始执行...
// sections+parallel for组合版本运行时间: 28.9941 秒, 加速比: 1.11
// sections+任务分块版本开始执行...
// sections+任务分块版本运行时间: 13.5519 秒, 加速比: 2.38
// sections+嵌套并行版本开始执行...
// sections+嵌套并行版本运行时间: 2.3086 秒, 加速比: 13.98

// 各版本加速比对比:
// 串行版本: 1.00
// parallel for版本: 24.61
// sections版本: 1.03
// sections+parallel for组合版本: 1.11
// sections+任务分块版本: 2.38
// sections+嵌套并行版本: 13.98