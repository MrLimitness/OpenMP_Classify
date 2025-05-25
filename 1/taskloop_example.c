#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>
#include <math.h>
// 全局设置
#define SIZE 8000
#define HEAVY_WORK_ITERS 100000

// 工作量不均匀的函数
void do_work(int i) {
    // 模拟工作量不均匀的计算
    // 某些索引（能被10整除的）需要更多计算
    int iterations = (i % 10 == 0) ? HEAVY_WORK_ITERS * 20 : HEAVY_WORK_ITERS;
    double result = 0.0;
    for (int j = 0; j < iterations; j++) {
        result += sin(i * 0.1 + j * 0.01) * cos(i * 0.2 + j * 0.02);
    }
    // 使用易失性变量防止编译器优化掉计算
    volatile double dummy = result;
}

// 串行版本
double serial_version(int *array, int size) {
    double start_time = omp_get_wtime();
    
    for (int i = 0; i < size; i++) {
        do_work(i);
        array[i] = i * 2;
    }
    
    double end_time = omp_get_wtime();
    printf("串行版本运行时间: %.2f 秒\n", end_time - start_time);
    printf("串行版本加速比: 1.00\n");
    return end_time - start_time;
}

// 使用 parallel for 的版本
double parallel_for_version(int *array, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; i++) {
        do_work(i);
        array[i] = i * 2;
    }
    
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("parallel for 版本运行时间: %.2f 秒\n", elapsed);
    return elapsed;
}

// 使用 taskloop 的基本版本
double taskloop_basic_version(int *array, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp single
        {
            #pragma omp taskloop
            for (int i = 0; i < size; i++) {
                do_work(i);
                array[i] = i * 2;
            }
        }
    }
    
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("taskloop 基本版本运行时间: %.2f 秒\n", elapsed);
    return elapsed;
}

// 使用 taskloop 带有 grainsize 的版本
double taskloop_grainsize_version(int *array, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp single
        {
            #pragma omp taskloop grainsize(100)
            for (int i = 0; i < size; i++) {
                do_work(i);
                array[i] = i * 2;
            }
        }
    }
    
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("taskloop grainsize 版本运行时间: %.2f 秒\n", elapsed);
    return elapsed;
}

// 使用 taskloop 带有 nogroup 的版本
double taskloop_nogroup_version(int *array, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp single
        {
            #pragma omp taskloop nogroup
            for (int i = 0; i < size; i++) {
                do_work(i);
                array[i] = i * 2;
            }
        }
    }
    
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("taskloop nogroup 版本运行时间: %.2f 秒\n", elapsed);
    return elapsed;
}

// 使用 taskloop 结合 priority 的版本
double taskloop_priority_version(int *array, int size, int num_threads) {
    double start_time = omp_get_wtime();
    
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp single
        {
            // 将数组分成两部分：工作量大的和工作量小的
            // 优先处理工作量大的部分（能被10整除的索引）
            #pragma omp taskloop grainsize(50) priority(1)
            for (int i = 0; i < size; i++) {
                if (i % 10 == 0) {
                    do_work(i);
                    array[i] = i * 2;
                }
            }
            
            #pragma omp taskloop grainsize(200) priority(0)
            for (int i = 0; i < size; i++) {
                if (i % 10 != 0) {
                    do_work(i);
                    array[i] = i * 2;
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("taskloop priority 版本运行时间: %.2f 秒\n", elapsed);
    return elapsed;
}

int main(int argc, char *argv[]) {
    int num_threads = 4; // 默认线程数
    
    // 检查是否提供了线程数参数
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    
    printf("使用 %d 个线程运行测试\n", num_threads);
    
    int *array = (int*)malloc(SIZE * sizeof(int));
    if (!array) {
        printf("内存分配失败\n");
        return 1;
    }
    
    // 运行并测量串行版本
    double serial_time = serial_version(array, SIZE);
    
    // 运行并测量 parallel for 版本
    double parallel_for_time = parallel_for_version(array, SIZE, num_threads);
    printf("parallel for 版本加速比: %.2f\n", serial_time / parallel_for_time);
    
    // 运行并测量 taskloop 基本版本
    double taskloop_basic_time = taskloop_basic_version(array, SIZE, num_threads);
    printf("taskloop 基本版本加速比: %.2f\n", serial_time / taskloop_basic_time);
    
    // 运行并测量 taskloop grainsize 版本
    double taskloop_grainsize_time = taskloop_grainsize_version(array, SIZE, num_threads);
    printf("taskloop grainsize 版本加速比: %.2f\n", serial_time / taskloop_grainsize_time);
    
    // 运行并测量 taskloop nogroup 版本
    double taskloop_nogroup_time = taskloop_nogroup_version(array, SIZE, num_threads);
    printf("taskloop nogroup 版本加速比: %.2f\n", serial_time / taskloop_nogroup_time);
    
    // 运行并测量 taskloop priority 版本
    double taskloop_priority_time = taskloop_priority_version(array, SIZE, num_threads);
    printf("taskloop priority 版本加速比: %.2f\n", serial_time / taskloop_priority_time);
    
    // 打印最佳版本
    printf("\n性能总结：\n");
    double best_time = parallel_for_time;
    const char* best_version = "parallel for";
    
    if (taskloop_basic_time < best_time) {
        best_time = taskloop_basic_time;
        best_version = "taskloop 基本版本";
    }
    if (taskloop_grainsize_time < best_time) {
        best_time = taskloop_grainsize_time;
        best_version = "taskloop grainsize";
    }
    if (taskloop_nogroup_time < best_time) {
        best_time = taskloop_nogroup_time;
        best_version = "taskloop nogroup";
    }
    if (taskloop_priority_time < best_time) {
        best_time = taskloop_priority_time;
        best_version = "taskloop priority";
    }
    
    printf("最佳性能版本: %s (加速比: %.2f)\n", best_version, serial_time / best_time);
    
    free(array);
    return 0;
}
