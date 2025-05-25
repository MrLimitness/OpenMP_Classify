
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define SIZE 100000000  // 数组大小，调整以获得约18秒的运行时间
#define NUM_ITERATIONS 100000000  // 重复计算次数，增加计算量

// 串行版本、parallel for版本和atomic版本的函数声明
void serial_version(int* data, int size, int iterations);
void parallel_for_version(int* data, int size, int iterations);
void atomic_version1(int* data, int size, int iterations);
void atomic_version2(int* data, int size, int iterations);
void atomic_version3(int* data, int size, int iterations);
void atomic_version4(int* data, int size, int iterations);

int main() {
    // 分配内存
    int* data = (int*)malloc(SIZE * sizeof(int));
    if (data == NULL) {
        printf("内存分配失败\n");
        return 1;
    }

    // 初始化数据
    for (int i = 0; i < SIZE; i++) {
        data[i] = rand() % 100;
    }

    // 获取线程数
    int num_threads;
    printf("请输入要使用的线程数: ");
    scanf("%d", &num_threads);
    omp_set_num_threads(num_threads);
    printf("使用 %d 个线程运行\n\n", num_threads);

    // 串行版本
    double start_time = omp_get_wtime();
    serial_version(data, SIZE, NUM_ITERATIONS);
    double serial_time = omp_get_wtime() - start_time;
    printf("串行版本运行时间: %.2f 秒\n", serial_time);
    printf("串行版本加速比: 1.00\n\n");

    // 重置数据
    for (int i = 0; i < SIZE; i++) {
        data[i] = rand() % 100;
    }

    // parallel for版本
    start_time = omp_get_wtime();
    parallel_for_version(data, SIZE, NUM_ITERATIONS);
    double parallel_for_time = omp_get_wtime() - start_time;
    printf("Parallel For版本运行时间: %.2f 秒\n", parallel_for_time);
    printf("Parallel For版本加速比: %.2f\n\n", serial_time / parallel_for_time);

    // 重置数据
    for (int i = 0; i < SIZE; i++) {
        data[i] = rand() % 100;
    }

    // Atomic版本1
    start_time = omp_get_wtime();
    atomic_version1(data, SIZE, NUM_ITERATIONS);
    double atomic_time1 = omp_get_wtime() - start_time;
    printf("Atomic版本1运行时间: %.2f 秒\n", atomic_time1);
    printf("Atomic版本1加速比: %.2f\n\n", serial_time / atomic_time1);

    // 重置数据
    for (int i = 0; i < SIZE; i++) {
        data[i] = rand() % 100;
    }

    // Atomic版本2
    start_time = omp_get_wtime();
    atomic_version2(data, SIZE, NUM_ITERATIONS);
    double atomic_time2 = omp_get_wtime() - start_time;
    printf("Atomic版本2运行时间: %.2f 秒\n", atomic_time2);
    printf("Atomic版本2加速比: %.2f\n\n", serial_time / atomic_time2);

    // 重置数据
    for (int i = 0; i < SIZE; i++) {
        data[i] = rand() % 100;
    }

    // Atomic版本3
    start_time = omp_get_wtime();
    atomic_version3(data, SIZE, NUM_ITERATIONS);
    double atomic_time3 = omp_get_wtime() - start_time;
    printf("Atomic版本3运行时间: %.2f 秒\n", atomic_time3);
    printf("Atomic版本3加速比: %.2f\n\n", serial_time / atomic_time3);

    // 重置数据
    for (int i = 0; i < SIZE; i++) {
        data[i] = rand() % 100;
    }

    // Atomic版本4
    start_time = omp_get_wtime();
    atomic_version4(data, SIZE, NUM_ITERATIONS);
    double atomic_time4 = omp_get_wtime() - start_time;
    printf("Atomic版本4运行时间: %.2f 秒\n", atomic_time4);
    printf("Atomic版本4加速比: %.2f\n\n", serial_time / atomic_time4);

    // 释放内存
    free(data);
    return 0;
}

// 串行版本：计算数组元素出现频率
void serial_version(int* data, int size, int iterations) {
    int histogram[100] = {0};  // 假设数据范围在0-99之间
    
    for (int iter = 0; iter < iterations; iter++) {
        // 重置直方图
        for (int i = 0; i < 100; i++) {
            histogram[i] = 0;
        }
        
        // 计算频率
        for (int i = 0; i < size; i++) {
            histogram[data[i]]++;
        }
        
        // 对直方图做一些操作，增加计算量
        int sum = 0;
        for (int i = 0; i < 100; i++) {
            sum += histogram[i];
        }
    }
}

// Parallel For版本
void parallel_for_version(int* data, int size, int iterations) {
    int histogram[100] = {0};
    
    for (int iter = 0; iter < iterations; iter++) {
        // 重置直方图
        for (int i = 0; i < 100; i++) {
            histogram[i] = 0;
        }
        
        // 这里会有竞争条件，但我们暂时忽略它以展示加速比差异
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            histogram[data[i]]++;
        }
        
        int sum = 0;
        for (int i = 0; i < 100; i++) {
            sum += histogram[i];
        }
    }
}

// Atomic版本1：使用atomic解决竞争条件
void atomic_version1(int* data, int size, int iterations) {
    int histogram[100] = {0};
    
    for (int iter = 0; iter < iterations; iter++) {
        // 重置直方图
        for (int i = 0; i < 100; i++) {
            histogram[i] = 0;
        }
        
        // 使用atomic保证原子性
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            #pragma omp atomic
            histogram[data[i]]++;
        }
        
        int sum = 0;
        for (int i = 0; i < 100; i++) {
            sum += histogram[i];
        }
    }
}

// Atomic版本2：使用atomic和reduction结合
void atomic_version2(int* data, int size, int iterations) {
    int histogram[100] = {0};
    
    for (int iter = 0; iter < iterations; iter++) {
        // 重置直方图
        for (int i = 0; i < 100; i++) {
            histogram[i] = 0;
        }
        
        // 使用atomic保证原子性
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            #pragma omp atomic
            histogram[data[i]]++;
        }
        
        // 使用reduction进行求和
        int sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < 100; i++) {
            sum += histogram[i];
        }
    }
}

// Atomic版本3：使用私有直方图减少竞争
void atomic_version3(int* data, int size, int iterations) {
    int histogram[100] = {0};
    
    for (int iter = 0; iter < iterations; iter++) {
        // 重置直方图
        for (int i = 0; i < 100; i++) {
            histogram[i] = 0;
        }
        
        #pragma omp parallel
        {
            // 每个线程使用私有直方图
            int private_histogram[100] = {0};
            
            // 并行计算私有直方图
            #pragma omp for
            for (int i = 0; i < size; i++) {
                private_histogram[data[i]]++;
            }
            
            // 合并结果到全局直方图
            for (int i = 0; i < 100; i++) {
                if (private_histogram[i] > 0) {
                    #pragma omp atomic
                    histogram[i] += private_histogram[i];
                }
            }
        }
        
        int sum = 0;
        for (int i = 0; i < 100; i++) {
            sum += histogram[i];
        }
    }
}

// Atomic版本4：使用atomic和调度策略结合
void atomic_version4(int* data, int size, int iterations) {
    int histogram[100] = {0};
    
    for (int iter = 0; iter < iterations; iter++) {
        // 重置直方图
        for (int i = 0; i < 100; i++) {
            histogram[i] = 0;
        }
        
        // 使用atomic和块调度，减少线程冲突
        #pragma omp parallel for schedule(static, 1000)
        for (int i = 0; i < size; i++) {
            #pragma omp atomic
            histogram[data[i]]++;
        }
        
        // 并行计算总和
        int sum = 0;
        #pragma omp parallel
        {
            int local_sum = 0;
            
            #pragma omp for nowait
            for (int i = 0; i < 100; i++) {
                local_sum += histogram[i];
            }
            
            #pragma omp atomic
            sum += local_sum;
        }
    }
}
