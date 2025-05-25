#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

// 快速排序的分区函数
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return (i + 1);
}

// 串行快速排序
void quicksort_serial(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort_serial(arr, low, pi - 1);
        quicksort_serial(arr, pi + 1, high);
    }
}

// 使用parallel for的快速排序
void quicksort_parallel_for(int arr[], int low, int high, int depth) {
    if (low < high) {
        if (depth <= 0) {
            // 如果递归深度达到阈值，使用串行排序
            quicksort_serial(arr, low, high);
        } else {
            int pi = partition(arr, low, high);
            
            #pragma omp parallel sections
            {
                #pragma omp section
                quicksort_parallel_for(arr, low, pi - 1, depth - 1);
                
                #pragma omp section
                quicksort_parallel_for(arr, pi + 1, high, depth - 1);
            }
        }
    }
}

// 使用task的快速排序 - 版本1：基本任务并行
void quicksort_task_v1(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        
        #pragma omp task
        quicksort_task_v1(arr, low, pi - 1);
        
        #pragma omp task
        quicksort_task_v1(arr, pi + 1, high);
        
        #pragma omp taskwait
    }
}

// 使用task的快速排序 - 版本2：结合任务切分阈值
void quicksort_task_v2(int arr[], int low, int high, int threshold) {
    if (low < high) {
        if (high - low < threshold) {
            // 小规模问题串行处理
            quicksort_serial(arr, low, high);
        } else {
            int pi = partition(arr, low, high);
            
            #pragma omp task
            quicksort_task_v2(arr, low, pi - 1, threshold);
            
            #pragma omp task
            quicksort_task_v2(arr, pi + 1, high, threshold);
            
            #pragma omp taskwait
        }
    }
}

// 使用task的快速排序 - 版本3：结合任务切分阈值和任务优先级
void quicksort_task_v3(int arr[], int low, int high, int threshold) {
    if (low < high) {
        if (high - low < threshold) {
            // 小规模问题串行处理
            quicksort_serial(arr, low, high);
        } else {
            int pi = partition(arr, low, high);
            int left_size = pi - low;
            int right_size = high - pi;
            
            // 优先处理较大的子数组
            if (left_size > right_size) {
                #pragma omp task priority(1)
                quicksort_task_v3(arr, low, pi - 1, threshold);
                
                #pragma omp task priority(0)
                quicksort_task_v3(arr, pi + 1, high, threshold);
            } else {
                #pragma omp task priority(0)
                quicksort_task_v3(arr, low, pi - 1, threshold);
                
                #pragma omp task priority(1)
                quicksort_task_v3(arr, pi + 1, high, threshold);
            }
            
            #pragma omp taskwait
        }
    }
}

// 使用task的快速排序 - 版本4：结合任务切分和任务池控制
void quicksort_task_v4(int arr[], int low, int high, int threshold, int depth) {
    if (low < high) {
        if (high - low < threshold || depth <= 0) {
            // 小规模问题或深度过大，串行处理
            quicksort_serial(arr, low, high);
        } else {
            int pi = partition(arr, low, high);
            
            #pragma omp task final(depth <= 2) mergeable
            quicksort_task_v4(arr, low, pi - 1, threshold, depth - 1);
            
            #pragma omp task final(depth <= 2) mergeable
            quicksort_task_v4(arr, pi + 1, high, threshold, depth - 1);
            
            #pragma omp taskwait
        }
    }
}

// 生成大型随机数组
int* generate_random_array(int size) {
    int* arr = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % size;
    }
    return arr;
}

// 检查数组是否已排序
int is_sorted(int arr[], int size) {
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i - 1]) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char* argv[]) {
    // 默认线程数为4，可通过命令行参数修改
    int num_threads = 64;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);
    
    // 设置随机种子
    srand(time(NULL));
    
    // 数组大小（调整以获得约18秒的串行运行时间）
    int size = 50000000;
    
    printf("使用 %d 线程进行测试\n", num_threads);
    printf("数组大小: %d\n", size);
    
    // 任务阈值参数
    int threshold = 10000;
    int max_depth = 12;
    
    // 串行版本
    int* arr1 = generate_random_array(size);
    double start_time = omp_get_wtime();
    quicksort_serial(arr1, 0, size - 1);
    double serial_time = omp_get_wtime() - start_time;
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    printf("数组已排序: %s\n", is_sorted(arr1, size) ? "是" : "否");
    free(arr1);
    
    // Parallel For 版本
    int* arr2 = generate_random_array(size);
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        quicksort_parallel_for(arr2, 0, size - 1, 4);
    }
    double parallel_for_time = omp_get_wtime() - start_time;
    printf("Parallel For 版本运行时间: %.4f 秒\n", parallel_for_time);
    printf("加速比: %.2f\n", serial_time / parallel_for_time);
    printf("数组已排序: %s\n", is_sorted(arr2, size) ? "是" : "否");
    free(arr2);
    
    // Task 版本1
    int* arr3 = generate_random_array(size);
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        quicksort_task_v1(arr3, 0, size - 1);
    }
    double task_time1 = omp_get_wtime() - start_time;
    printf("Task 版本1运行时间: %.4f 秒\n", task_time1);
    printf("加速比: %.2f\n", serial_time / task_time1);
    printf("数组已排序: %s\n", is_sorted(arr3, size) ? "是" : "否");
    free(arr3);
    
    // Task 版本2（带阈值）
    int* arr4 = generate_random_array(size);
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        quicksort_task_v2(arr4, 0, size - 1, threshold);
    }
    double task_time2 = omp_get_wtime() - start_time;
    printf("Task 版本2运行时间: %.4f 秒\n", task_time2);
    printf("加速比: %.2f\n", serial_time / task_time2);
    printf("数组已排序: %s\n", is_sorted(arr4, size) ? "是" : "否");
    free(arr4);
    
    // Task 版本3（带优先级）
    int* arr5 = generate_random_array(size);
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        quicksort_task_v3(arr5, 0, size - 1, threshold);
    }
    double task_time3 = omp_get_wtime() - start_time;
    printf("Task 版本3运行时间: %.4f 秒\n", task_time3);
    printf("加速比: %.2f\n", serial_time / task_time3);
    printf("数组已排序: %s\n", is_sorted(arr5, size) ? "是" : "否");
    free(arr5);
    
    // Task 版本4（深度控制）
    int* arr6 = generate_random_array(size);
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        quicksort_task_v4(arr6, 0, size - 1, threshold, max_depth);
    }
    double task_time4 = omp_get_wtime() - start_time;
    printf("Task 版本4运行时间: %.4f 秒\n", task_time4);
    printf("加速比: %.2f\n", serial_time / task_time4);
    printf("数组已排序: %s\n", is_sorted(arr6, size) ? "是" : "否");
    free(arr6);
    
    return 0;
}


// 使用 4 线程进行测试
// 数组大小: 50000000
// 串行版本运行时间: 4.9420 秒
// 数组已排序: 是
// Parallel For 版本运行时间: 4.9424 秒
// 加速比: 1.00
// 数组已排序: 是
// Task 版本1运行时间: 41.1176 秒
// 加速比: 0.12
// 数组已排序: 是
// Task 版本2运行时间: 2.5459 秒
// 加速比: 1.94
// 数组已排序: 是
// Task 版本3运行时间: 2.5110 秒
// 加速比: 1.97
// 数组已排序: 是
// Task 版本4运行时间: 1.4807 秒
// 加速比: 3.34
// 数组已排序: 是


// 使用 16 线程进行测试
// 数组大小: 50000000
// 串行版本运行时间: 4.8100 秒
// 数组已排序: 是
// Parallel For 版本运行时间: 4.7469 秒
// 加速比: 1.01
// 数组已排序: 是
// Task 版本1运行时间: 87.1978 秒
// 加速比: 0.06
// 数组已排序: 是
// Task 版本2运行时间: 0.8567 秒
// 加速比: 5.61
// 数组已排序: 是
// Task 版本3运行时间: 0.7802 秒
// 加速比: 6.17
// 数组已排序: 是
// Task 版本4运行时间: 0.8718 秒
// 加速比: 5.52
// 数组已排序: 是

// 使用 64 线程进行测试
// 数组大小: 50000000
// 串行版本运行时间: 4.7841 秒
// 数组已排序: 是
// Parallel For 版本运行时间: 5.1743 秒
// 加速比: 0.92
// 数组已排序: 是
// Task 版本1运行时间: 108.6462 秒
// 加速比: 0.04
// 数组已排序: 是
// Task 版本2运行时间: 0.5617 秒
// 加速比: 8.52
// 数组已排序: 是
// Task 版本3运行时间: 0.5092 秒
// 加速比: 9.40
// 数组已排序: 是
// Task 版本4运行时间: 0.7203 秒
// 加速比: 6.64
// 数组已排序: 是