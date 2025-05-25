#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <string.h>

// 递归计算斐波那契数列（计算密集型）
long long fibonacci(long long n) {
    if (n < 2) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

void serial_compute(long long* results, int* numbers, int size) {
    for (int i = 0; i < size; i++) {
        results[i] = fibonacci(numbers[i]);
    }
}

void parallel_for_compute(long long* results, int* numbers, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        results[i] = fibonacci(numbers[i]);
    }
}

void task_compute_no_grainsize(long long* results, int* numbers, int size) {
    #pragma omp parallel
    {
        #pragma omp single
        for (int i = 0; i < size; i++) {
            #pragma omp task
            results[i] = fibonacci(numbers[i]);
        }
    }
}

void task_compute_with_manual_grainsize(long long* results, int* numbers, int size, int grainsize) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < size; i += grainsize) {
                int start = i;
                int end = (i + grainsize > size) ? size : i + grainsize;
                #pragma omp task firstprivate(start, end)
                for (int j = start; j < end; j++) {
                    results[j] = fibonacci(numbers[j]);
                }
            }
        }
    }
}

void process_range(long long* results, int* numbers, int start, int end, int grainsize) {
    int range = end - start;
    if (range <= grainsize) {
        for (int i = start; i < end; i++) {
            results[i] = fibonacci(numbers[i]);
        }
    } else {
        int mid = start + range / 2;
        #pragma omp task
        process_range(results, numbers, start, mid, grainsize);

        #pragma omp task
        process_range(results, numbers, mid, end, grainsize);

        #pragma omp taskwait
    }
}

void task_compute_recursive(long long* results, int* numbers, int size, int grainsize) {
    #pragma omp parallel
    {
        #pragma omp single
        process_range(results, numbers, 0, size, grainsize);
    }
}

int main(int argc, char* argv[]) {
    int num_threads = (argc > 1) ? atoi(argv[1]) : 64;
    printf("\n使用线程数：%d\n", num_threads);
    omp_set_num_threads(num_threads);

    int size = 400;
    int* numbers = malloc(size * sizeof(int));
    long long* results = malloc(size * sizeof(long long));
    long long* verify_results = malloc(size * sizeof(long long));

    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        numbers[i] = 30 + rand() % 8;
    }

    printf("\n运行串行版本...\n");
    double start_time = omp_get_wtime();
    serial_compute(verify_results, numbers, size);
    double serial_time = omp_get_wtime() - start_time;
    printf("串行版本运行时间: %.4f 秒\n", serial_time);

    printf("\n运行 parallel for 版本...\n");
    memset(results, 0, size * sizeof(long long));
    start_time = omp_get_wtime();
    parallel_for_compute(results, numbers, size);
    double for_time = omp_get_wtime() - start_time;
    printf("parallel for 运行时间: %.4f 秒, 加速比: %.2f\n", for_time, serial_time / for_time);

    printf("\n运行任务并行(无grainsize)...\n");
    memset(results, 0, size * sizeof(long long));
    start_time = omp_get_wtime();
    task_compute_no_grainsize(results, numbers, size);
    double task_no_grainsize_time = omp_get_wtime() - start_time;
    printf("任务并行(无grainsize)时间: %.4f 秒, 加速比: %.2f\n", task_no_grainsize_time, serial_time / task_no_grainsize_time);

    printf("\n运行任务并行(grainsize=1)...\n");
    memset(results, 0, size * sizeof(long long));
    start_time = omp_get_wtime();
    task_compute_with_manual_grainsize(results, numbers, size, 1);
    double grainsize1_time = omp_get_wtime() - start_time;
    printf("任务并行(grainsize=1)时间: %.4f 秒, 加速比: %.2f\n", grainsize1_time, serial_time / grainsize1_time);

    int medium_grainsize = 5;
    printf("\n运行任务并行(grainsize=%d)...\n", medium_grainsize);
    memset(results, 0, size * sizeof(long long));
    start_time = omp_get_wtime();
    task_compute_with_manual_grainsize(results, numbers, size, medium_grainsize);
    double grainsize_medium_time = omp_get_wtime() - start_time;
    printf("任务并行(grainsize=%d)时间: %.4f 秒, 加速比: %.2f\n", medium_grainsize, grainsize_medium_time, serial_time / grainsize_medium_time);

    int large_grainsize = 10;
    printf("\n运行任务递归(grainsize=%d)...\n", large_grainsize);
    memset(results, 0, size * sizeof(long long));
    start_time = omp_get_wtime();
    task_compute_recursive(results, numbers, size, large_grainsize);
    double recursive_time = omp_get_wtime() - start_time;
    printf("任务递归(grainsize=%d)时间: %.4f 秒, 加速比: %.2f\n", large_grainsize, recursive_time, serial_time / recursive_time);

    free(numbers);
    free(results);
    free(verify_results);
    return 0;
}

// 使用线程数：16

// 运行串行版本...
// 串行版本运行时间: 14.3637 秒

// 运行 parallel for 版本...
// parallel for 运行时间: 1.3424 秒, 加速比: 10.70

// 运行任务并行(无grainsize)...
// 任务并行(无grainsize)时间: 1.2873 秒, 加速比: 11.16

// 运行任务并行(grainsize=1)...
// 任务并行(grainsize=1)时间: 1.2048 秒, 加速比: 11.92

// 运行任务并行(grainsize=5)...
// 任务并行(grainsize=5)时间: 1.1008 秒, 加速比: 13.05

// 运行任务递归(grainsize=10)...
// 任务递归(grainsize=10)时间: 1.6809 秒, 加速比: 8.55


// 使用线程数：64

// 运行串行版本...
// 串行版本运行时间: 16.7469 秒

// 运行 parallel for 版本...
// parallel for 运行时间: 0.5140 秒, 加速比: 32.58

// 运行任务并行(无grainsize)...
// 任务并行(无grainsize)时间: 0.5153 秒, 加速比: 32.50

// 运行任务并行(grainsize=1)...
// 任务并行(grainsize=1)时间: 0.5391 秒, 加速比: 31.06

// 运行任务并行(grainsize=5)...
// 任务并行(grainsize=5)时间: 0.8244 秒, 加速比: 20.31

// 运行任务递归(grainsize=10)...
// 任务递归(grainsize=10)时间: 0.6654 秒, 加速比: 25.17