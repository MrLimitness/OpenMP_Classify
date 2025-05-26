
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

// 定义数组大小
#define N 2000000000  // 约20亿个整数


// 串行版本：计算直方图
void histogram_serial(int *data, int *hist, int n, int num_bins) {
    for (int i = 0; i < num_bins; i++) {
        hist[i] = 0;
    }
    
    for (int i = 0; i < n; i++) {
        hist[data[i]]++;
    }
}

// 并行版本1：使用parallel for的直方图计算（存在竞争问题）
void histogram_parallel_for(int *data, int *hist, int n, int num_bins) {
    for (int i = 0; i < num_bins; i++) {
        hist[i] = 0;
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        hist[data[i]]++;  // 竞争问题导致结果错误
    }
}

// 并行版本2：使用atomic解决竞争问题
void histogram_atomic(int *data, int *hist, int n, int num_bins) {
    for (int i = 0; i < num_bins; i++) {
        hist[i] = 0;
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        #pragma omp atomic
        hist[data[i]]++;
    }
}

// 并行版本3：使用atomic与分块优化结合
void histogram_atomic_chunked(int *data, int *hist, int n, int num_bins) {
    for (int i = 0; i < num_bins; i++) {
        hist[i] = 0;
    }
    
    // 分块大小
    const int CHUNK_SIZE = 1000;
    
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (int i = 0; i < n; i++) {
        #pragma omp atomic
        hist[data[i]]++;
    }
}

// 并行版本4：使用局部直方图加atomic
void histogram_local_atomic(int *data, int *hist, int n, int num_bins) {
    // 初始化主直方图
    for (int i = 0; i < num_bins; i++) {
        hist[i] = 0;
    }
    
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        
        // 创建线程本地直方图
        int *local_hist = (int *)calloc(num_bins, sizeof(int));
        
        // 每个线程处理自己的数据块
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            local_hist[data[i]]++;
        }
        
        // 合并本地直方图到全局直方图
        for (int i = 0; i < num_bins; i++) {
            if (local_hist[i] > 0) {
                #pragma omp atomic
                hist[i] += local_hist[i];
            }
        }
        
        free(local_hist);
    }
}

// 并行版本5：使用原子操作和局部直方图与动态调度结合
void histogram_local_atomic_dynamic(int *data, int *hist, int n, int num_bins) {
    // 初始化主直方图
    for (int i = 0; i < num_bins; i++) {
        hist[i] = 0;
    }
    
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        
        // 创建线程本地直方图
        int *local_hist = (int *)calloc(num_bins, sizeof(int));
        
        // 每个线程处理自己的数据块
        #pragma omp for schedule(dynamic, 10000)
        for (int i = 0; i < n; i++) {
            local_hist[data[i]]++;
        }
        
        // 合并本地直方图到全局直方图
        for (int i = 0; i < num_bins; i++) {
            if (local_hist[i] > 0) {
                #pragma omp atomic
                hist[i] += local_hist[i];
            }
        }
        
        free(local_hist);
    }
}

int main() {
    int num_threads = 16; // 可以修改为16/64/144等
    omp_set_num_threads(num_threads);
    
    // 定义直方图桶数
    const int NUM_BINS = 256;
    
    printf("数组大小: %d\n", N);
    printf("直方图桶数: %d\n", NUM_BINS);
    printf("运行线程数: %d\n\n", num_threads);
    
    // 分配内存
    int *data = (int *)malloc(N * sizeof(int));
    int *hist = (int *)malloc(NUM_BINS * sizeof(int));
    int *hist_parallel = (int *)malloc(NUM_BINS * sizeof(int));
    
    // 初始化数据
    srand(12345);
    for (int i = 0; i < N; i++) {
        data[i] = rand() % NUM_BINS;
    }
    
    // 测试串行版本
    printf("运行串行版本...\n");
    double start_time = omp_get_wtime();
    histogram_serial(data, hist, N, NUM_BINS);
    double end_time = omp_get_wtime();
    double serial_time = end_time - start_time;
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    printf("串行版本加速比: 1.00\n\n");
    
    // 计算串行结果的校验和
    long long serial_checksum = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        serial_checksum += hist[i];
    }
    printf("串行版本结果校验和: %lld\n\n", serial_checksum);
    
    // 测试parallel for版本（存在竞争问题）
    printf("运行parallel for版本（存在竞争问题）...\n");
    start_time = omp_get_wtime();
    histogram_parallel_for(data, hist_parallel, N, NUM_BINS);
    end_time = omp_get_wtime();
    double parallel_for_time = end_time - start_time;
    printf("parallel for版本运行时间: %.4f 秒\n", parallel_for_time);
    printf("parallel for版本加速比: %.2f\n\n", serial_time / parallel_for_time);
    
    // 验证结果
    long long parallel_for_checksum = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        parallel_for_checksum += hist_parallel[i];
    }
    printf("parallel for版本结果校验和: %lld\n", parallel_for_checksum);
    printf("parallel for版本结果与串行版本差异: %lld\n\n", llabs(parallel_for_checksum - serial_checksum));
    
    // 测试atomic版本
    printf("运行atomic版本...\n");
    start_time = omp_get_wtime();
    histogram_atomic(data, hist_parallel, N, NUM_BINS);
    end_time = omp_get_wtime();
    double atomic_time = end_time - start_time;
    printf("atomic版本运行时间: %.4f 秒\n", atomic_time);
    printf("atomic版本加速比: %.2f\n\n", serial_time / atomic_time);
    
    // 验证结果
    long long atomic_checksum = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        atomic_checksum += hist_parallel[i];
    }
    printf("atomic版本结果校验和: %lld\n", atomic_checksum);
    printf("atomic版本结果与串行版本差异: %lld\n\n", llabs(atomic_checksum - serial_checksum));
    
    // 测试atomic chunked版本
    printf("运行atomic chunked版本...\n");
    start_time = omp_get_wtime();
    histogram_atomic_chunked(data, hist_parallel, N, NUM_BINS);
    end_time = omp_get_wtime();
    double atomic_chunked_time = end_time - start_time;
    printf("atomic chunked版本运行时间: %.4f 秒\n", atomic_chunked_time);
    printf("atomic chunked版本加速比: %.2f\n\n", serial_time / atomic_chunked_time);
    
    // 验证结果
    long long atomic_chunked_checksum = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        atomic_chunked_checksum += hist_parallel[i];
    }
    printf("atomic chunked版本结果校验和: %lld\n", atomic_chunked_checksum);
    printf("atomic chunked版本结果与串行版本差异: %lld\n\n", llabs(atomic_chunked_checksum - serial_checksum));
    
    // 测试local atomic版本
    printf("运行local atomic版本...\n");
    start_time = omp_get_wtime();
    histogram_local_atomic(data, hist_parallel, N, NUM_BINS);
    end_time = omp_get_wtime();
    double local_atomic_time = end_time - start_time;
    printf("local atomic版本运行时间: %.4f 秒\n", local_atomic_time);
    printf("local atomic版本加速比: %.2f\n\n", serial_time / local_atomic_time);
    
    // 验证结果
    long long local_atomic_checksum = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        local_atomic_checksum += hist_parallel[i];
    }
    printf("local atomic版本结果校验和: %lld\n", local_atomic_checksum);
    printf("local atomic版本结果与串行版本差异: %lld\n\n", llabs(local_atomic_checksum - serial_checksum));
    
    // 测试local atomic dynamic版本
    printf("运行local atomic dynamic版本...\n");
    start_time = omp_get_wtime();
    histogram_local_atomic_dynamic(data, hist_parallel, N, NUM_BINS);
    end_time = omp_get_wtime();
    double local_atomic_dynamic_time = end_time - start_time;
    printf("local atomic dynamic版本运行时间: %.4f 秒\n", local_atomic_dynamic_time);
    printf("local atomic dynamic版本加速比: %.2f\n\n", serial_time / local_atomic_dynamic_time);
    
    // 验证结果
    long long local_atomic_dynamic_checksum = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        local_atomic_dynamic_checksum += hist_parallel[i];
    }
    printf("local atomic dynamic版本结果校验和: %lld\n", local_atomic_dynamic_checksum);
    printf("local atomic dynamic版本结果与串行版本差异: %lld\n\n", llabs(local_atomic_dynamic_checksum - serial_checksum));
    
    // 性能总结
    printf("======= 性能总结 =======\n");
    printf("串行版本:                  %.4f 秒 (加速比: 1.00)\n", serial_time);
    printf("parallel for版本(有竞争):  %.4f 秒 (加速比: %.2f) - 结果不正确\n", parallel_for_time, serial_time / parallel_for_time);
    printf("atomic版本:                %.4f 秒 (加速比: %.2f)\n", atomic_time, serial_time / atomic_time);
    printf("atomic chunked版本:        %.4f 秒 (加速比: %.2f)\n", atomic_chunked_time, serial_time / atomic_chunked_time);
    printf("local atomic版本:          %.4f 秒 (加速比: %.2f)\n", local_atomic_time, serial_time / local_atomic_time);
    printf("local atomic dynamic版本:  %.4f 秒 (加速比: %.2f)\n", local_atomic_dynamic_time, serial_time / local_atomic_dynamic_time);
    
    // 释放内存
    free(data);
    free(hist);
    free(hist_parallel);
    
    return 0;
}




// 数组大小: 2000000000
// 直方图桶数: 256
// 运行线程数: 16

// 运行串行版本...
// 串行版本运行时间: 5.8330 秒
// 串行版本加速比: 1.00

// 串行版本结果校验和: 2000000000

// 运行parallel for版本（存在竞争问题）...
// parallel for版本运行时间: 30.2677 秒
// parallel for版本加速比: 0.19

// parallel for版本结果校验和: 976178026
// parallel for版本结果与串行版本差异: 1023821974

// 运行atomic版本...
// atomic版本运行时间: 18.3481 秒
// atomic版本加速比: 0.32

// atomic版本结果校验和: 2000000000
// atomic版本结果与串行版本差异: 0

// 运行atomic chunked版本...
// atomic chunked版本运行时间: 18.0465 秒
// atomic chunked版本加速比: 0.32

// atomic chunked版本结果校验和: 2000000000
// atomic chunked版本结果与串行版本差异: 0

// 运行local atomic版本...
// local atomic版本运行时间: 0.3822 秒
// local atomic版本加速比: 15.26

// local atomic版本结果校验和: 2000000000
// local atomic版本结果与串行版本差异: 0

// 运行local atomic dynamic版本...
// local atomic dynamic版本运行时间: 0.1860 秒
// local atomic dynamic版本加速比: 31.36

// local atomic dynamic版本结果校验和: 2000000000
// local atomic dynamic版本结果与串行版本差异: 0

// ======= 性能总结 =======
// 串行版本:                  5.8330 秒 (加速比: 1.00)
// parallel for版本(有竞争):  30.2677 秒 (加速比: 0.19) - 结果不正确
// atomic版本:                18.3481 秒 (加速比: 0.32)
// atomic chunked版本:        18.0465 秒 (加速比: 0.32)
// local atomic版本:          0.3822 秒 (加速比: 15.26)
// local atomic dynamic版本:  0.1860 秒 (加速比: 31.36)