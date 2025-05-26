#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <string.h>

// 模拟复杂的科学计算或数据处理任务
double complex_computation(long i, int complexity_factor) {
    double result = 0.0;
    double base_work = fabs(sin(i * 0.01) * cos(i * 0.023) * tan(i * 0.003));
    long work_amount = (long)(complexity_factor * pow(base_work * 15.0 + 0.5, 3) * 100000);
    work_amount = work_amount > 10000 ? work_amount : 10000;

    for (long j = 0; j < work_amount; j++) {
        result += sin(j * 0.0001) * cos(j * 0.00013) / (1.0 + fabs(sin(j * 0.0021)));
        if (j % 1000 == 0 && base_work > 0.7) {
            for (int k = 0; k < 100; k++) {
                result += sqrt(fabs(result + k)) / (1.0 + k);
            }
        }
    }

    return result;
}

double serial_version(long size, int complexity_factor) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    printf("\n开始串行计算...\n");
    for (long i = 0; i < size; i++) {
        total_result += complex_computation(i, complexity_factor);
        if (i % (size / 10) == 0) {
            printf("串行处理进度: %.1f%%\n", (double)i / size * 100);
        }
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    printf("\n串行版本运行时间: %.4f 秒\n", time_taken);
    printf("串行版本结果总和: %.6e\n", total_result);
    printf("串行版本加速比: 1.00\n");

    return time_taken;
}

void parallel_for_version(long size, int complexity_factor, double serial_time) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    printf("\n开始parallel for计算...\n");
    #pragma omp parallel
    {
        #pragma omp single
        printf("使用 %d 个线程\n", omp_get_num_threads());
    }

    #pragma omp parallel for reduction(+:total_result) schedule(static)
    for (long i = 0; i < size; i++) {
        total_result += complex_computation(i, complexity_factor);
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    printf("parallel for版本(静态调度)运行时间: %.4f 秒\n", time_taken);
    printf("parallel for版本结果总和: %.6e\n", total_result);
    printf("parallel for版本加速比: %.2f\n", serial_time / time_taken);
}

void parallel_for_dynamic_version(long size, int complexity_factor, double serial_time) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    printf("\n开始parallel for(动态调度)计算...\n");

    #pragma omp parallel for reduction(+:total_result) schedule(dynamic, 50)
    for (long i = 0; i < size; i++) {
        total_result += complex_computation(i, complexity_factor);
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    printf("parallel for版本(动态调度)运行时间: %.4f 秒\n", time_taken);
    printf("parallel for版本结果总和: %.6e\n", total_result);
    printf("parallel for版本加速比: %.2f\n", serial_time / time_taken);
}

void parallel_for_guided_version(long size, int complexity_factor, double serial_time) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    printf("\n开始parallel for(guided调度)计算...\n");

    #pragma omp parallel for reduction(+:total_result) schedule(guided, 20)
    for (long i = 0; i < size; i++) {
        total_result += complex_computation(i, complexity_factor);
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    printf("parallel for版本(guided调度)运行时间: %.4f 秒\n", time_taken);
    printf("parallel for版本结果总和: %.6e\n", total_result);
    printf("parallel for版本加速比: %.2f\n", serial_time / time_taken);
}

void taskloop_version1(long size, int complexity_factor, double serial_time) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    printf("\n开始taskloop基本版本计算...\n");

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop reduction(+:total_result)
            for (long i = 0; i < size; i++) {
                total_result += complex_computation(i, complexity_factor);
            }
        }
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    printf("taskloop版本1运行时间: %.4f 秒\n", time_taken);
    printf("taskloop版本1结果总和: %.6e\n", total_result);
    printf("taskloop版本1加速比: %.2f\n", serial_time / time_taken);
}

void taskloop_version2(long size, int complexity_factor, double serial_time) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    printf("\n开始taskloop grainsize版本计算...\n");

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop grainsize(50) reduction(+:total_result)
            for (long i = 0; i < size; i++) {
                total_result += complex_computation(i, complexity_factor);
            }
        }
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    printf("taskloop版本2(grainsize=50)运行时间: %.4f 秒\n", time_taken);
    printf("taskloop版本2结果总和: %.6e\n", total_result);
    printf("taskloop版本2加速比: %.2f\n", serial_time / time_taken);
}

void taskloop_version3(long size, int complexity_factor, double serial_time) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    printf("\n开始taskloop num_tasks版本计算...\n");

    #pragma omp parallel
    {
        #pragma omp single
        {
            int thread_count = omp_get_num_threads();
            #pragma omp taskloop num_tasks(thread_count * 8) reduction(+:total_result)
            for (long i = 0; i < size; i++) {
                total_result += complex_computation(i, complexity_factor);
            }
        }
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    printf("taskloop版本3(num_tasks)运行时间: %.4f 秒\n", time_taken);
    printf("taskloop版本3结果总和: %.6e\n", total_result);
    printf("taskloop版本3加速比: %.2f\n", serial_time / time_taken);
}

void taskloop_version4(long size, int complexity_factor, double serial_time) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    printf("\n开始taskloop版本4(无nogroup)计算...\n");

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop reduction(+:total_result)
            for (long i = 0; i < size; i++) {
                total_result += complex_computation(i, complexity_factor);
            }
        }
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    printf("taskloop版本4(无nogroup)运行时间: %.4f 秒\n", time_taken);
    printf("taskloop版本4结果总和: %.6e\n", total_result);
    printf("taskloop版本4加速比: %.2f\n", serial_time / time_taken);
}

void taskloop_version5(long size, int complexity_factor, double serial_time) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    printf("\n开始taskloop版本5(综合优化)计算...\n");

    #pragma omp parallel
    {
        #pragma omp single
        {
            int thread_count = omp_get_num_threads();
            #pragma omp taskloop grainsize(20) reduction(+:total_result)
            for (long i = 0; i < size; i++) {
                total_result += complex_computation(i, complexity_factor);
            }
        }
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    printf("taskloop版本5(综合优化)运行时间: %.4f 秒\n", time_taken);
    printf("taskloop版本5结果总和: %.6e\n", total_result);
    printf("taskloop版本5加速比: %.2f\n", serial_time / time_taken);
}

void recursive_taskloop_version(long size, int complexity_factor, double serial_time) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    printf("\n开始递归+taskloop混合版本计算...\n");

    void process_range(long start, long end, int depth, double* result) {
        const int MAX_DEPTH = 3;
        long range_size = end - start;

        if (depth >= MAX_DEPTH || range_size <= 100) {
            double local_result = 0.0;
            #pragma omp taskloop reduction(+:local_result)
            for (long i = start; i < end; i++) {
                local_result += complex_computation(i, complexity_factor);
            }
            #pragma omp atomic
            *result += local_result;
        } else {
            long mid = start + range_size / 2;
            #pragma omp task
            process_range(start, mid, depth + 1, result);
            #pragma omp task
            process_range(mid, end, depth + 1, result);
        }
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            process_range(0, size, 0, &total_result);
        }
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    printf("递归+taskloop混合版本运行时间: %.4f 秒\n", time_taken);
    printf("递归+taskloop混合版本结果总和: %.6e\n", total_result);
    printf("递归+taskloop混合版本加速比: %.2f\n", serial_time / time_taken);
}

int main(int argc, char* argv[]) {
    long size = 100;
    int complexity_factor = 5;
    int run_all = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            size = atol(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            complexity_factor = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            omp_set_num_threads(atoi(argv[i + 1]));
            i++;
        } else if (strcmp(argv[i], "-quick") == 0) {
            run_all = 0;
        }
    }

    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    printf("======= OpenMP Taskloop 性能对比测试 =======\n");
    printf("运行线程数: %d\n", num_threads);
    printf("问题规模: %ld\n", size);
    printf("复杂度因子: %d\n\n", complexity_factor);

    double serial_time = serial_version(size, complexity_factor);
    parallel_for_version(size, complexity_factor, serial_time);

    if (run_all) {
        parallel_for_dynamic_version(size, complexity_factor, serial_time);
        parallel_for_guided_version(size, complexity_factor, serial_time);
    }

    taskloop_version1(size, complexity_factor, serial_time);

    if (run_all) {
        taskloop_version2(size, complexity_factor, serial_time);
        taskloop_version3(size, complexity_factor, serial_time);
        taskloop_version4(size, complexity_factor, serial_time);
        taskloop_version5(size, complexity_factor, serial_time);
    }

    recursive_taskloop_version(size, complexity_factor, serial_time);

    printf("\n======= 测试完成 =======\n");
    return 0;
}
// ======= OpenMP Taskloop 性能对比测试 =======
// 运行线程数: 128
// 问题规模: 100
// 复杂度因子: 5


// 开始串行计算...
// 串行处理进度: 0.0%
// 串行处理进度: 10.0%
// 串行处理进度: 20.0%
// 串行处理进度: 30.0%
// 串行处理进度: 40.0%
// 串行处理进度: 50.0%
// 串行处理进度: 60.0%
// 串行处理进度: 70.0%
// 串行处理进度: 80.0%
// 串行处理进度: 90.0%

// 串行版本运行时间: 6.5692 秒
// 串行版本结果总和: -9.084768e+05
// 串行版本加速比: 1.00

// 开始parallel for计算...
// 使用 128 个线程
// parallel for版本(静态调度)运行时间: 0.7935 秒
// parallel for版本结果总和: -9.084768e+05
// parallel for版本加速比: 8.28

// 开始parallel for(动态调度)计算...
// parallel for版本(动态调度)运行时间: 6.0105 秒
// parallel for版本结果总和: -9.084768e+05
// parallel for版本加速比: 1.09

// 开始parallel for(guided调度)计算...
// parallel for版本(guided调度)运行时间: 6.0134 秒
// parallel for版本结果总和: -9.084768e+05
// parallel for版本加速比: 1.09

// 开始taskloop基本版本计算...
// taskloop版本1运行时间: 1.0302 秒
// taskloop版本1结果总和: -9.084768e+05
// taskloop版本1加速比: 6.38

// 开始taskloop grainsize版本计算...
// taskloop版本2(grainsize=50)运行时间: 5.9734 秒
// taskloop版本2结果总和: -9.084768e+05
// taskloop版本2加速比: 1.10

// 开始taskloop num_tasks版本计算...
// taskloop版本3(num_tasks)运行时间: 0.7950 秒
// taskloop版本3结果总和: -9.084768e+05
// taskloop版本3加速比: 8.26

// 开始taskloop版本4(无nogroup)计算...
// taskloop版本4(无nogroup)运行时间: 0.8063 秒
// taskloop版本4结果总和: -9.084768e+05
// taskloop版本4加速比: 8.15

// 开始taskloop版本5(综合优化)计算...
// taskloop版本5(综合优化)运行时间: 5.7971 秒
// taskloop版本5结果总和: -9.084768e+05
// taskloop版本5加速比: 1.13

// 开始递归+taskloop混合版本计算...
// 递归+taskloop混合版本运行时间: 0.8379 秒
// 递归+taskloop混合版本结果总和: -9.084768e+05
// 递归+taskloop混合版本加速比: 7.84

// ======= 测试完成 =======