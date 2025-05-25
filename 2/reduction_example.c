
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#define SIZE 200000
#define ITERATIONS 1000

double serial_sum(double* arr, int size) {
    double sum = 0.0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        sum = 0.0;
        for (int i = 0; i < size; i++) {
            sum += sin(arr[i]) * cos(arr[i]);
        }
    }
    return sum;
}

double parallel_for_sum(double* arr, int size) {
    double sum = 0.0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        sum = 0.0;
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
          
            sum += sin(arr[i]) * cos(arr[i]);
        }
    }
    return sum;
}

double reduction_sum(double* arr, int size) {
    double sum = 0.0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < size; i++) {
            sum += sin(arr[i]) * cos(arr[i]);
        }
    }
    return sum;
}

double reduction_tiled_sum(double* arr, int size) {
    double sum = 0.0;
    int chunk_size = 100;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static, chunk_size)
        for (int i = 0; i < size; i++) {
            sum += sin(arr[i]) * cos(arr[i]);
        }
    }
    return sum;
}

double reduction_sections_sum(double* arr, int size) {
    double sum = 0.0;
    int chunk = size / 4;
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        sum = 0.0;
        
        #pragma omp parallel sections reduction(+:sum)
        {
            #pragma omp section
            {
                for (int i = 0; i < chunk; i++) {
                    sum += sin(arr[i]) * cos(arr[i]);
                }
            }
            
            #pragma omp section
            {
                for (int i = chunk; i < 2*chunk; i++) {
                    sum += sin(arr[i]) * cos(arr[i]);
                }
            }
            
            #pragma omp section
            {
                for (int i = 2*chunk; i < 3*chunk; i++) {
                    sum += sin(arr[i]) * cos(arr[i]);
                }
            }
            
            #pragma omp section
            {
                for (int i = 3*chunk; i < size; i++) {
                    sum += sin(arr[i]) * cos(arr[i]);
                }
            }
        }
    }
    return sum;
}

double reduction_nested_sum(double* arr, int size) {
    double sum = 0.0;
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        sum = 0.0;
        
        #pragma omp parallel reduction(+:sum)
        {
            double local_sum = 0.0;
            int tid = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            int items_per_thread = size / num_threads;
            int start = tid * items_per_thread;
            int end = (tid == num_threads-1) ? size : start + items_per_thread;
            
            #pragma omp parallel for reduction(+:local_sum)
            for (int i = start; i < end; i++) {
                local_sum += sin(arr[i]) * cos(arr[i]);
            }
            
            sum += local_sum;
        }
    }
    return sum;
}

int main(int argc, char* argv[]) {
    int num_threads = 64; // 默认线程数
    
    // 检查命令行参数，允许用户指定线程数
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads <= 0) {
            printf("错误: 线程数必须大于0\n");
            return 1;
        }
    }
    
    omp_set_num_threads(num_threads);
    printf("使用 %d 个线程运行\n", num_threads);
    
    // 初始化数组
    double *arr = (double*)malloc(SIZE * sizeof(double));
    for (int i = 0; i < SIZE; i++) {
        arr[i] = (double)rand() / RAND_MAX;
    }
    
    double start_time, end_time, result;
    
    // 串行版本
    start_time = omp_get_wtime();
    result = serial_sum(arr, SIZE);
    end_time = omp_get_wtime();
    double serial_time = end_time - start_time;
    printf("串行版本结果: %.10f, 运行时间: %.6f 秒\n", result, serial_time);
    
    // parallel for版本
    start_time = omp_get_wtime();
    result = parallel_for_sum(arr, SIZE);
    end_time = omp_get_wtime();
    double parallel_for_time = end_time - start_time;
    printf("parallel for版本结果: %.10f, 运行时间: %.6f 秒, 加速比: %.2f\n", 
           result, parallel_for_time, serial_time / parallel_for_time);
    
    // reduction版本
    start_time = omp_get_wtime();
    result = reduction_sum(arr, SIZE);
    end_time = omp_get_wtime();
    double reduction_time = end_time - start_time;
    printf("基本reduction版本结果: %.10f, 运行时间: %.6f 秒, 加速比: %.2f\n", 
           result, reduction_time, serial_time / reduction_time);
    
    // reduction + tiled版本
    start_time = omp_get_wtime();
    result = reduction_tiled_sum(arr, SIZE);
    end_time = omp_get_wtime();
    double reduction_tiled_time = end_time - start_time;
    printf("reduction + 分块版本结果: %.10f, 运行时间: %.6f 秒, 加速比: %.2f\n", 
           result, reduction_tiled_time, serial_time / reduction_tiled_time);
    
    // reduction + sections版本
    start_time = omp_get_wtime();
    result = reduction_sections_sum(arr, SIZE);
    end_time = omp_get_wtime();
    double reduction_sections_time = end_time - start_time;
    printf("reduction + sections版本结果: %.10f, 运行时间: %.6f 秒, 加速比: %.2f\n", 
           result, reduction_sections_time, serial_time / reduction_sections_time);
    
    // reduction + nested版本
    start_time = omp_get_wtime();
    result = reduction_nested_sum(arr, SIZE);
    end_time = omp_get_wtime();
    double reduction_nested_time = end_time - start_time;
    printf("reduction + 嵌套并行版本结果: %.10f, 运行时间: %.6f 秒, 加速比: %.2f\n", 
           result, reduction_nested_time, serial_time / reduction_nested_time);
    
    free(arr);
    return 0;
}


// 使用 16 个线程运行
// 串行版本结果: 70824.6777688284, 运行时间: 8.774176 秒
// parallel for版本结果: 39817.1798307378, 运行时间: 16.915828 秒, 加速比: 0.52
// 基本reduction版本结果: 70824.6777688284, 运行时间: 11.792075 秒, 加速比: 0.74
// reduction + 分块版本结果: 70824.6777688283, 运行时间: 0.424345 秒, 加速比: 20.68
// reduction + sections版本结果: 70824.6777688285, 运行时间: 2.115268 秒, 加速比: 4.15
// reduction + 嵌套并行版本结果: 70824.6777688284, 运行时间: 4.024588 秒, 加速比: 2.18

// 使用 64 个线程运行
// 串行版本结果: 70824.6777688284, 运行时间: 4.566454 秒
// parallel for版本结果: 1106.5630443710, 运行时间: 15.750833 秒, 加速比: 0.29
// 基本reduction版本结果: 70824.6777688284, 运行时间: 0.098820 秒, 加速比: 46.21
// reduction + 分块版本结果: 70824.6777688283, 运行时间: 0.092933 秒, 加速比: 49.14
// reduction + sections版本结果: 70824.6777688285, 运行时间: 3.242663 秒, 加速比: 1.41
// reduction + 嵌套并行版本结果: 70824.6777688284, 运行时间: 10.101645 秒, 加速比: 0.45