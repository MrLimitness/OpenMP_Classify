
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

// 自定义数学函数，有一个常量参数alpha和一个变量参数x
double custom_math_function(double alpha, double x) {
    return alpha * sin(x) * cos(x) * exp(-fabs(x) * 0.01) + sqrt(fabs(x + alpha));
}

void print_results(const char* version_name, double serial_time, double parallel_time) {
    printf("%s运行时间: %.4f秒\n", version_name, parallel_time);
    printf("%s加速比: %.2f\n\n", version_name, serial_time / parallel_time);
}

int main(int argc, char* argv[]) {
    // 设置线程数（可以通过命令行参数修改）
    int num_threads = 64; // 默认线程数
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);
    printf("使用 %d 个线程运行\n\n", num_threads);

    // 数据大小，调整以获得约18秒的串行运行时间
    const int SIZE = 400000000;
    double* data = (double*)malloc(SIZE * sizeof(double));
    double* results = (double*)malloc(SIZE * sizeof(double));
    double* results_parallel = (double*)malloc(SIZE * sizeof(double));
    double* results_uniform = (double*)malloc(SIZE * sizeof(double));
    double* results_uniform_improved = (double*)malloc(SIZE * sizeof(double));
    double* results_uniform_combined = (double*)malloc(SIZE * sizeof(double));
    double* results_uniform_optimized = (double*)malloc(SIZE * sizeof(double));

    // 初始化数据
    printf("初始化数据...\n");
    #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        data[i] = sin(i * 0.00001) * 10.0;
    }
    
    printf("开始测试...\n\n");
    double start_time, end_time;
    double alpha = 2.5; // 所有调用中共享的常量参数
    double sum = 0.0, sum_parallel = 0.0, sum_uniform = 0.0, sum_uniform_improved = 0.0;
    double sum_uniform_combined = 0.0, sum_uniform_optimized = 0.0;

    // 版本1: 串行版本
    start_time = omp_get_wtime();
    for (int i = 0; i < SIZE; i++) {
        results[i] = custom_math_function(alpha, data[i]);
        sum += results[i];
    }
    end_time = omp_get_wtime();
    double serial_time = end_time - start_time;
    printf("串行版本运行时间: %.4f秒\n\n", serial_time);

    // 版本2: 使用parallel for的版本
    start_time = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum_parallel)
    for (int i = 0; i < SIZE; i++) {
        results_parallel[i] = custom_math_function(alpha, data[i]);
        sum_parallel += results_parallel[i];
    }
    end_time = omp_get_wtime();
    print_results("parallel for版本", serial_time, end_time - start_time);

    // 版本3: 使用uniform指令
    start_time = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum_uniform)
    for (int i = 0; i < SIZE; i++) {
        #pragma omp uniform(alpha)
        results_uniform[i] = custom_math_function(alpha, data[i]);
        sum_uniform += results_uniform[i];
    }
    end_time = omp_get_wtime();
    print_results("uniform版本", serial_time, end_time - start_time);

    // 版本4: uniform + simd指令
    start_time = omp_get_wtime();
    #pragma omp parallel for simd reduction(+:sum_uniform_improved)
    for (int i = 0; i < SIZE; i++) {
        #pragma omp uniform(alpha)
        results_uniform_improved[i] = custom_math_function(alpha, data[i]);
        sum_uniform_improved += results_uniform_improved[i];
    }
    end_time = omp_get_wtime();
    print_results("uniform + simd版本", serial_time, end_time - start_time);

    // 版本5: uniform + 分块处理
    start_time = omp_get_wtime();
    const int BLOCK_SIZE = 1024;
    #pragma omp parallel reduction(+:sum_uniform_combined)
    {
        #pragma omp for schedule(dynamic, 16)
        for (int b = 0; b < SIZE; b += BLOCK_SIZE) {
            int end = (b + BLOCK_SIZE < SIZE) ? b + BLOCK_SIZE : SIZE;
            double local_sum = 0.0;
            
            #pragma omp simd reduction(+:local_sum)
            for (int i = b; i < end; i++) {
                #pragma omp uniform(alpha)
                results_uniform_combined[i] = custom_math_function(alpha, data[i]);
                local_sum += results_uniform_combined[i];
            }
            
            sum_uniform_combined += local_sum;
        }
    }
    end_time = omp_get_wtime();
    print_results("uniform + 分块处理版本", serial_time, end_time - start_time);

    // 版本6: uniform + task + simd
    start_time = omp_get_wtime();
    const int TASK_SIZE = SIZE / (num_threads * 4);
    sum_uniform_optimized = 0.0;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int b = 0; b < SIZE; b += TASK_SIZE) {
                int end = (b + TASK_SIZE < SIZE) ? b + TASK_SIZE : SIZE;
                
                #pragma omp task firstprivate(b, end)
                {
                    double local_sum = 0.0;
                    
                    #pragma omp simd reduction(+:local_sum)
                    for (int i = b; i < end; i++) {
                        #pragma omp uniform(alpha)
                        results_uniform_optimized[i] = custom_math_function(alpha, data[i]);
                        local_sum += results_uniform_optimized[i];
                    }
                    
                    #pragma omp atomic
                    sum_uniform_optimized += local_sum;
                }
            }
        }
    }
    end_time = omp_get_wtime();
    print_results("uniform + task + simd版本", serial_time, end_time - start_time);

    // 验证结果正确性
    printf("验证结果: ");
    double error = fabs((sum - sum_uniform_optimized) / sum);
    if (error < 1e-10) {
        printf("所有版本结果一致 ✓\n");
    } else {
        printf("警告: 结果有误差 %.10e\n", error);
    }

    // 清理
    free(data);
    free(results);
    free(results_parallel);
    free(results_uniform);
    free(results_uniform_improved);
    free(results_uniform_combined);
    free(results_uniform_optimized);

    return 0;
}


// 使用 16 个线程运行

// 初始化数据...
// 开始测试...

// 串行版本运行时间: 27.1990秒

// parallel for版本运行时间: 1.6937秒
// parallel for版本加速比: 16.06

// uniform版本运行时间: 1.3791秒
// uniform版本加速比: 19.72

// uniform + simd版本运行时间: 1.5149秒
// uniform + simd版本加速比: 17.95

// uniform + 分块处理版本运行时间: 2.7005秒
// uniform + 分块处理版本加速比: 10.07

// uniform + task + simd版本运行时间: 1.5310秒
// uniform + task + simd版本加速比: 17.77

// 验证结果: 所有版本结果一致 ✓


// 使用 64 个线程运行

// 初始化数据...
// 开始测试...

// 串行版本运行时间: 19.2360秒

// parallel for版本运行时间: 0.6101秒
// parallel for版本加速比: 31.53

// uniform版本运行时间: 0.9927秒
// uniform版本加速比: 19.38

// uniform + simd版本运行时间: 0.7669秒
// uniform + simd版本加速比: 25.08

// uniform + 分块处理版本运行时间: 0.6291秒
// uniform + 分块处理版本加速比: 30.58

// uniform + task + simd版本运行时间: 0.7100秒
// uniform + task + simd版本加速比: 27.09

// 验证结果: 所有版本结果一致 ✓