
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define ARRAY_SIZE 500000000  // 大数组，确保运行时间足够长

// 串行版本
double serial_version() {
    double start_time = omp_get_wtime();
    
    // 初始化数组
    int *array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;  // 随机值填充数组
    }
    
    // 计算所有大于50的元素数量
    long count = 0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (array[i] > 50) {
            count++;
        }
    }
    
    free(array);
    double end_time = omp_get_wtime();
    printf("串行版本结果: %ld\n", count);
    return end_time - start_time;
}

// 使用 parallel for 的版本
double parallel_for_version(int num_threads) {
    double start_time = omp_get_wtime();
    
    // 初始化数组
    int *array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;  // 随机值填充数组
    }
    
    // 计算所有大于50的元素数量，但存在竞争条件
    long count = 0;
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (array[i] > 50) {
            #pragma omp critical
            count++;
        }
    }
    
    free(array);
    double end_time = omp_get_wtime();
    printf("parallel for版本结果 (使用critical): %ld\n", count);
    return end_time - start_time;
}

// 使用 atomic 的版本1 - 基本应用
double atomic_version1(int num_threads) {
    double start_time = omp_get_wtime();
    
    // 初始化数组
    int *array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;  // 随机值填充数组
    }
    
    // 使用atomic替代critical，性能更好
    long count = 0;
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (array[i] > 50) {
            #pragma omp atomic
            count++;
        }
    }
    
    free(array);
    double end_time = omp_get_wtime();
    printf("atomic版本1结果: %ld\n", count);
    return end_time - start_time;
}

// 使用 atomic 的版本2 - 数据分块减少原子操作
double atomic_version2(int num_threads) {
    double start_time = omp_get_wtime();
    
    // 初始化数组
    int *array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;
    }
    
    // 使用局部计数器，减少原子操作次数
    long count = 0;
    #pragma omp parallel num_threads(num_threads)
    {
        long local_count = 0;
        #pragma omp for
        for (int i = 0; i < ARRAY_SIZE; i++) {
            if (array[i] > 50) {
                local_count++;
            }
        }
        
        // 只对最终结果执行一次原子操作
        #pragma omp atomic
        count += local_count;
    }
    
    free(array);
    double end_time = omp_get_wtime();
    printf("atomic版本2结果 (减少原子操作): %ld\n", count);
    return end_time - start_time;
}

// 使用 atomic 的版本3 - 结合reduction
double atomic_version3(int num_threads) {
    double start_time = omp_get_wtime();
    
    // 初始化数组
    int *array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;
    }
    
    // 使用reduction代替atomic计数
    long count = 0;
    #pragma omp parallel for reduction(+:count) num_threads(num_threads)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (array[i] > 50) {
            count++;
        }
    }
    
    free(array);
    double end_time = omp_get_wtime();
    printf("atomic版本3结果 (使用reduction): %ld\n", count);
    return end_time - start_time;
}

// 使用 atomic 的版本4 - 使用多种atomic类型
double atomic_version4(int num_threads) {
    double start_time = omp_get_wtime();
    
    // 初始化数组
    int *array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;
    }
    
    // 使用多种atomic操作类型
    long count_gt50 = 0;  // 大于50的计数
    long count_lt20 = 0;  // 小于20的计数
    long max_value = 0;   // 最大值
    
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (array[i] > 50) {
            #pragma omp atomic
            count_gt50++;
        }
        else if (array[i] < 20) {
            #pragma omp atomic
            count_lt20++;
        }
        
        if (array[i] > max_value) {
            #pragma omp atomic write
            max_value = array[i] > max_value ? array[i] : max_value;
        }
    }
    
    free(array);
    double end_time = omp_get_wtime();
    printf("atomic版本4结果 (多种atomic类型): >50: %ld, <20: %ld, 最大值: %ld\n", 
           count_gt50, count_lt20, max_value);
    return end_time - start_time;
}

int main() {
    srand(time(NULL));
    
    // 获取用户输入的线程数
    int num_threads;
    printf("请输入要使用的线程数: ");
    scanf("%d", &num_threads);
    printf("使用 %d 个线程进行测试\n\n", num_threads);
    
    // 运行各个版本并记录时间
    printf("运行串行版本...\n");
    double serial_time = serial_version();
    printf("串行版本运行时间: %.6f 秒\n\n", serial_time);
    
    printf("运行parallel for版本...\n");
    double parallel_time = parallel_for_version(num_threads);
    printf("parallel for版本运行时间: %.6f 秒\n", parallel_time);
    printf("加速比: %.2f\n\n", serial_time / parallel_time);
    
    printf("运行atomic版本1...\n");
    double atomic_time1 = atomic_version1(num_threads);
    printf("atomic版本1运行时间: %.6f 秒\n", atomic_time1);
    printf("加速比: %.2f\n\n", serial_time / atomic_time1);
    
    printf("运行atomic版本2...\n");
    double atomic_time2 = atomic_version2(num_threads);
    printf("atomic版本2运行时间: %.6f 秒\n", atomic_time2);
    printf("加速比: %.2f\n\n", serial_time / atomic_time2);
    
    printf("运行atomic版本3...\n");
    double atomic_time3 = atomic_version3(num_threads);
    printf("atomic版本3运行时间: %.6f 秒\n", atomic_time3);
    printf("加速比: %.2f\n\n", serial_time / atomic_time3);
    
    printf("运行atomic版本4...\n");
    double atomic_time4 = atomic_version4(num_threads);
    printf("atomic版本4运行时间: %.6f 秒\n", atomic_time4);
    printf("加速比: %.2f\n\n", serial_time / atomic_time4);
    
    printf("性能比较总结:\n");
    printf("串行版本: %.6f 秒 (基准)\n", serial_time);
    printf("parallel for版本: %.6f 秒, 加速比: %.2f\n", parallel_time, serial_time / parallel_time);
    printf("atomic版本1: %.6f 秒, 加速比: %.2f\n", atomic_time1, serial_time / atomic_time1);
    printf("atomic版本2: %.6f 秒, 加速比: %.2f\n", atomic_time2, serial_time / atomic_time2);
    printf("atomic版本3: %.6f 秒, 加速比: %.2f\n", atomic_time3, serial_time / atomic_time3);
    printf("atomic版本4: %.6f 秒, 加速比: %.2f\n", atomic_time4, serial_time / atomic_time4);
    
    return 0;
}
