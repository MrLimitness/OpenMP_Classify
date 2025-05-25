
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
double get_serial_time();
// 设置线程数（可修改）
#define NUM_THREADS 16

// 计算密集型函数，模拟不均匀工作负载
double compute_work(int i) {
    double sum = 0.0;
    // 工作量随i增加而增加，模拟不均匀负载
    int work_size = 1000 + i * 50;
    for (int j = 0; j < work_size; j++) {
        sum += sin(i * j * 0.001) * cos(j * 0.001);
        sum += sqrt(i + j + 1.0);
    }
    return sum;
}

// 串行版本
void serial_version() {
    printf("=== 串行版本 ===\n");
    double start_time = omp_get_wtime();
    
    double total_sum = 0.0;
    const int N = 8000;
    
    for (int i = 0; i < N; i++) {
        total_sum += compute_work(i);
    }
    
    double end_time = omp_get_wtime();
    double serial_time = end_time - start_time;
    
    printf("计算结果: %.6f\n", total_sum);
    printf("串行运行时间: %.2f 秒\n", serial_time);
    printf("加速比: 1.00\n\n");
}

// parallel for版本
void parallel_for_version() {
    printf("=== Parallel For版本 ===\n");
    double start_time = omp_get_wtime();
    
    double total_sum = 0.0;
    const int N = 8000;
    
    #pragma omp parallel for num_threads(NUM_THREADS) reduction(+:total_sum)
    for (int i = 0; i < N; i++) {
        total_sum += compute_work(i);
    }
    
    double end_time = omp_get_wtime();
    double parallel_time = end_time - start_time;
    
    printf("计算结果: %.6f\n", total_sum);
    printf("Parallel For运行时间: %.2f 秒\n", parallel_time);
    printf("加速比: %.2f\n\n", get_serial_time() / parallel_time);
}

// schedule版本3：使用dynamic调度
void schedule_dynamic_version() {
    printf("=== Schedule Dynamic版本 ===\n");
    double start_time = omp_get_wtime();
    
    double total_sum = 0.0;
    const int N = 8000;
    
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic, 10) reduction(+:total_sum)
    for (int i = 0; i < N; i++) {
        total_sum += compute_work(i);
    }
    
    double end_time = omp_get_wtime();
    double dynamic_time = end_time - start_time;
    
    printf("计算结果: %.6f\n", total_sum);
    printf("Schedule Dynamic运行时间: %.2f 秒\n", dynamic_time);
    printf("加速比: %.2f\n\n", get_serial_time() / dynamic_time);
}

// schedule版本4：使用guided调度
void schedule_guided_version() {
    printf("=== Schedule Guided版本 ===\n");
    double start_time = omp_get_wtime();
    
    double total_sum = 0.0;
    const int N = 8000;
    
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(guided, 5) reduction(+:total_sum)
    for (int i = 0; i < N; i++) {
        total_sum += compute_work(i);
    }
    
    double end_time = omp_get_wtime();
    double guided_time = end_time - start_time;
    
    printf("计算结果: %.6f\n", total_sum);
    printf("Schedule Guided运行时间: %.2f 秒\n", guided_time);
    printf("加速比: %.2f\n\n", get_serial_time() / guided_time);
}

// schedule版本5：使用static调度
void schedule_static_version() {
    printf("=== Schedule Static版本 ===\n");
    double start_time = omp_get_wtime();
    
    double total_sum = 0.0;
    const int N = 8000;
    
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static, 50) reduction(+:total_sum)
    for (int i = 0; i < N; i++) {
        total_sum += compute_work(i);
    }
    
    double end_time = omp_get_wtime();
    double static_time = end_time - start_time;
    
    printf("计算结果: %.6f\n", total_sum);
    printf("Schedule Static运行时间: %.2f 秒\n", static_time);
    printf("加速比: %.2f\n\n", get_serial_time() / static_time);
}

// schedule版本6：使用runtime调度
void schedule_runtime_version() {
    printf("=== Schedule Runtime版本 ===\n");
    // 设置运行时调度为dynamic
    omp_set_schedule(omp_sched_dynamic, 8);
    
    double start_time = omp_get_wtime();
    
    double total_sum = 0.0;
    const int N = 8000;
    
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime) reduction(+:total_sum)
    for (int i = 0; i < N; i++) {
        total_sum += compute_work(i);
    }
    
    double end_time = omp_get_wtime();
    double runtime_time = end_time - start_time;
    
    printf("计算结果: %.6f\n", total_sum);
    printf("Schedule Runtime运行时间: %.2f 秒\n", runtime_time);
    printf("加速比: %.2f\n\n", get_serial_time() / runtime_time);
}

// 全局变量存储串行时间
double g_serial_time = 0.0;

double get_serial_time() {
    return g_serial_time;
}

void measure_serial_time() {
    printf("正在测量串行基准时间...\n");
    double start_time = omp_get_wtime();
    
    double total_sum = 0.0;
    const int N = 8000;
    
    for (int i = 0; i < N; i++) {
        total_sum += compute_work(i);
    }
    
    double end_time = omp_get_wtime();
    g_serial_time = end_time - start_time;
    printf("串行基准时间: %.2f 秒\n\n", g_serial_time);
}

int main() {
    printf("OpenMP Schedule调度策略性能测试\n");
    printf("当前线程数设置: %d\n", NUM_THREADS);
    printf("可在代码顶部修改NUM_THREADS宏来调整线程数\n");
    printf("========================================\n\n");
    
    // 首先测量串行基准时间
    measure_serial_time();
    
    // 运行各个版本
    serial_version();
    parallel_for_version();
    schedule_dynamic_version();
    schedule_guided_version();
    schedule_static_version();
    schedule_runtime_version();
    
    printf("========================================\n");
    printf("测试完成！\n");
    printf("不同schedule策略的性能差异主要体现在负载不均衡的情况下\n");
    printf("Dynamic和Guided通常在工作负载不均匀时表现更好\n");
    
    return 0;
}


// OpenMP Schedule调度策略性能测试
// 当前线程数设置: 16
// 可在代码顶部修改NUM_THREADS宏来调整线程数
// ========================================

// 正在测量串行基准时间...
// 串行基准时间: 67.24 秒

// === 串行版本 ===
// 计算结果: 557767738791.835205
// 串行运行时间: 54.32 秒
// 加速比: 1.00

// === Parallel For版本 ===
// 计算结果: 557767738791.835327
// Parallel For运行时间: 7.34 秒
// 加速比: 9.16

// === Schedule Dynamic版本 ===
// 计算结果: 557767738791.835083
// Schedule Dynamic运行时间: 3.94 秒
// 加速比: 17.05

// === Schedule Guided版本 ===
// 计算结果: 557767738791.835083
// Schedule Guided运行时间: 4.53 秒
// 加速比: 14.86

// === Schedule Static版本 ===
// 计算结果: 557767738791.835205
// Schedule Static运行时间: 4.81 秒
// 加速比: 13.99

// === Schedule Runtime版本 ===
// 计算结果: 557767738791.834961
// Schedule Runtime运行时间: 4.47 秒
// 加速比: 15.06

// ========================================
// 测试完成！
// 不同schedule策略的性能差异主要体现在负载不均衡的情况下
// Dynamic和Guided通常在工作负载不均匀时表现更好