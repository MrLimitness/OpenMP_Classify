
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define ARRAY_SIZE 100000000
#define POLYNOMIAL_TERMS 20
#define ITERATIONS 50

// 多项式计算函数 - 计算多项式在不同点的值
// P(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
void polynomial_evaluation_serial(double* input, double* output, double* coefficients, int size, int terms) {
    for (int i = 0; i < size; i++) {
        double x = input[i];
        double result = 0.0;
        double x_power = 1.0;
        
        for (int j = 0; j < terms; j++) {
            result += coefficients[j] * x_power;
            x_power *= x;
        }
        
        output[i] = result;
    }
}

// 使用parallel for的并行版本
void polynomial_evaluation_parallel_for(double* input, double* output, double* coefficients, int size, int terms) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        double x = input[i];
        double result = 0.0;
        double x_power = 1.0;
        
        for (int j = 0; j < terms; j++) {
            result += coefficients[j] * x_power;
            x_power *= x;
        }
        
        output[i] = result;
    }
}

// 使用parallel for + unroll的并行版本
void polynomial_evaluation_unroll(double* input, double* output, double* coefficients, int size, int terms) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        double x = input[i];
        double result = 0.0;
        double x_power = 1.0;
        
        #pragma omp unroll(4)
        for (int j = 0; j < terms; j++) {
            result += coefficients[j] * x_power;
            x_power *= x;
        }
        
        output[i] = result;
    }
}

// 使用parallel for + simd + unroll的并行版本
void polynomial_evaluation_simd_unroll(double* input, double* output, double* coefficients, int size, int terms) {
    #pragma omp parallel for simd
    for (int i = 0; i < size; i++) {
        double x = input[i];
        double result = 0.0;
        double x_power = 1.0;
        
        #pragma omp unroll(4)
        for (int j = 0; j < terms; j++) {
            result += coefficients[j] * x_power;
            x_power *= x;
        }
        
        output[i] = result;
    }
}

// 完全展开内层循环的版本 (手动展开)
void polynomial_evaluation_manual_unroll(double* input, double* output, double* coefficients, int size, int terms) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        double x = input[i];
        double result = 0.0;
        double x1 = x;
        double x2 = x1 * x;
        double x3 = x2 * x;
        double x4 = x3 * x;
        
        // 假设terms = 20，手动展开为5个块
        // 第1块
        result += coefficients[0];
        result += coefficients[1] * x1;
        result += coefficients[2] * x2;
        result += coefficients[3] * x3;
        
        // 第2块
        result += coefficients[4] * x4;
        double x5 = x4 * x;
        result += coefficients[5] * x5;
        double x6 = x5 * x;
        result += coefficients[6] * x6;
        double x7 = x6 * x;
        result += coefficients[7] * x7;
        
        // 第3块
        double x8 = x7 * x;
        result += coefficients[8] * x8;
        double x9 = x8 * x;
        result += coefficients[9] * x9;
        double x10 = x9 * x;
        result += coefficients[10] * x10;
        double x11 = x10 * x;
        result += coefficients[11] * x11;
        
        // 第4块
        double x12 = x11 * x;
        result += coefficients[12] * x12;
        double x13 = x12 * x;
        result += coefficients[13] * x13;
        double x14 = x13 * x;
        result += coefficients[14] * x14;
        double x15 = x14 * x;
        result += coefficients[15] * x15;
        
        // 第5块
        double x16 = x15 * x;
        result += coefficients[16] * x16;
        double x17 = x16 * x;
        result += coefficients[17] * x17;
        double x18 = x17 * x;
        result += coefficients[18] * x18;
        double x19 = x18 * x;
        result += coefficients[19] * x19;
        
        output[i] = result;
    }
}

// 使用parallel for + simd + unroll + schedule的并行版本
void polynomial_evaluation_combined(double* input, double* output, double* coefficients, int size, int terms) {
    #pragma omp parallel for simd schedule(static) 
    for (int i = 0; i < size; i++) {
        double x = input[i];
        double result = 0.0;
        double x_power = 1.0;
        
        #pragma omp unroll(8)
        for (int j = 0; j < terms; j++) {
            result += coefficients[j] * x_power;
            x_power *= x;
        }
        
        output[i] = result;
    }
}

int main() {
    int num_threads = 64; // 可以修改为16/64/144等
    omp_set_num_threads(num_threads);
    
    printf("运行线程数: %d\n", num_threads);
    printf("数组大小: %d\n", ARRAY_SIZE);
    printf("多项式项数: %d\n", POLYNOMIAL_TERMS);
    printf("重复计算次数: %d\n\n", ITERATIONS);
    
    // 分配内存
    double* input = (double*)malloc(ARRAY_SIZE * sizeof(double));
    double* output = (double*)malloc(ARRAY_SIZE * sizeof(double));
    double* coefficients = (double*)malloc(POLYNOMIAL_TERMS * sizeof(double));
    
    // 初始化数据
    srand(123); // 固定随机种子以便比较结果
    for (int i = 0; i < ARRAY_SIZE; i++) {
        input[i] = (double)rand() / RAND_MAX * 2.0 - 1.0; // 范围在[-1, 1]
    }
    
    for (int i = 0; i < POLYNOMIAL_TERMS; i++) {
        coefficients[i] = (double)rand() / RAND_MAX * 10.0 - 5.0; // 范围在[-5, 5]
    }
    
    // 版本1: 串行计算
    double serial_start = omp_get_wtime();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        polynomial_evaluation_serial(input, output, coefficients, ARRAY_SIZE, POLYNOMIAL_TERMS);
    }
    
    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    printf("串行版本加速比: 1.00\n\n");
    
    // 记录串行结果用于验证
    double serial_sum = 0.0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        serial_sum += output[i];
    }
    printf("串行版本结果校验和: %.6e\n\n", serial_sum);
    
    // 版本2: 使用parallel for
    double parallel_for_start = omp_get_wtime();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        polynomial_evaluation_parallel_for(input, output, coefficients, ARRAY_SIZE, POLYNOMIAL_TERMS);
    }
    
    double parallel_for_end = omp_get_wtime();
    double parallel_for_time = parallel_for_end - parallel_for_start;
    printf("parallel for版本运行时间: %.4f 秒\n", parallel_for_time);
    printf("parallel for版本加速比: %.2f\n\n", serial_time / parallel_for_time);
    
    // 验证结果
    double parallel_for_sum = 0.0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        parallel_for_sum += output[i];
    }
    printf("parallel for版本结果校验和: %.6e\n", parallel_for_sum);
    printf("parallel for版本结果误差: %.6e\n\n", fabs(parallel_for_sum - serial_sum));
    
    // 版本3: 使用parallel for + unroll
    double unroll_start = omp_get_wtime();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        polynomial_evaluation_unroll(input, output, coefficients, ARRAY_SIZE, POLYNOMIAL_TERMS);
    }
    
    double unroll_end = omp_get_wtime();
    double unroll_time = unroll_end - unroll_start;
    printf("unroll版本运行时间: %.4f 秒\n", unroll_time);
    printf("unroll版本加速比: %.2f\n\n", serial_time / unroll_time);
    
    // 验证结果
    double unroll_sum = 0.0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        unroll_sum += output[i];
    }
    printf("unroll版本结果校验和: %.6e\n", unroll_sum);
    printf("unroll版本结果误差: %.6e\n\n", fabs(unroll_sum - serial_sum));
    
    // 版本4: 使用parallel for + simd + unroll
    double simd_unroll_start = omp_get_wtime();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        polynomial_evaluation_simd_unroll(input, output, coefficients, ARRAY_SIZE, POLYNOMIAL_TERMS);
    }
    
    double simd_unroll_end = omp_get_wtime();
    double simd_unroll_time = simd_unroll_end - simd_unroll_start;
    printf("simd+unroll版本运行时间: %.4f 秒\n", simd_unroll_time);
    printf("simd+unroll版本加速比: %.2f\n\n", serial_time / simd_unroll_time);
    
    // 验证结果
    double simd_unroll_sum = 0.0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        simd_unroll_sum += output[i];
    }
    printf("simd+unroll版本结果校验和: %.6e\n", simd_unroll_sum);
    printf("simd+unroll版本结果误差: %.6e\n\n", fabs(simd_unroll_sum - serial_sum));
    
    // 版本5: 手动循环展开
    double manual_unroll_start = omp_get_wtime();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        polynomial_evaluation_manual_unroll(input, output, coefficients, ARRAY_SIZE, POLYNOMIAL_TERMS);
    }
    
    double manual_unroll_end = omp_get_wtime();
    double manual_unroll_time = manual_unroll_end - manual_unroll_start;
    printf("手动展开版本运行时间: %.4f 秒\n", manual_unroll_time);
    printf("手动展开版本加速比: %.2f\n\n", serial_time / manual_unroll_time);
    
    // 验证结果
    double manual_unroll_sum = 0.0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        manual_unroll_sum += output[i];
    }
    printf("手动展开版本结果校验和: %.6e\n", manual_unroll_sum);
    printf("手动展开版本结果误差: %.6e\n\n", fabs(manual_unroll_sum - serial_sum));
    
    // 版本6: 使用组合优化策略
    double combined_start = omp_get_wtime();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        polynomial_evaluation_combined(input, output, coefficients, ARRAY_SIZE, POLYNOMIAL_TERMS);
    }
    
    double combined_end = omp_get_wtime();
    double combined_time = combined_end - combined_start;
    printf("组合优化版本运行时间: %.4f 秒\n", combined_time);
    printf("组合优化版本加速比: %.2f\n\n", serial_time / combined_time);
    
    // 验证结果
    double combined_sum = 0.0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        combined_sum += output[i];
    }
    printf("组合优化版本结果校验和: %.6e\n", combined_sum);
    printf("组合优化版本结果误差: %.6e\n\n", fabs(combined_sum - serial_sum));
    
    // 性能总结
    printf("======= 性能总结 =======\n");
    printf("串行版本:      %.4f 秒 (加速比: 1.00)\n", serial_time);
    printf("parallel for:  %.4f 秒 (加速比: %.2f)\n", parallel_for_time, serial_time / parallel_for_time);
    printf("unroll:        %.4f 秒 (加速比: %.2f)\n", unroll_time, serial_time / unroll_time);
    printf("simd+unroll:   %.4f 秒 (加速比: %.2f)\n", simd_unroll_time, serial_time / simd_unroll_time);
    printf("手动展开:      %.4f 秒 (加速比: %.2f)\n", manual_unroll_time, serial_time / manual_unroll_time);
    printf("组合优化:      %.4f 秒 (加速比: %.2f)\n", combined_time, serial_time / combined_time);
    
    // 释放内存
    free(input);
    free(output);
    free(coefficients);
    
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



// 运行线程数: 16
// 数组大小: 100000000
// 多项式项数: 20
// 重复计算次数: 50

// 串行版本运行时间: 113.5284 秒
// 串行版本加速比: 1.00

// 串行版本结果校验和: 2.655308e+07

// parallel for版本运行时间: 14.4790 秒
// parallel for版本加速比: 7.84

// parallel for版本结果校验和: 2.655308e+07
// parallel for版本结果误差: 0.000000e+00

// unroll版本运行时间: 18.8483 秒
// unroll版本加速比: 6.02

// unroll版本结果校验和: 2.655308e+07
// unroll版本结果误差: 0.000000e+00

// simd+unroll版本运行时间: 19.8744 秒
// simd+unroll版本加速比: 5.71

// simd+unroll版本结果校验和: 2.655308e+07
// simd+unroll版本结果误差: 0.000000e+00

// 手动展开版本运行时间: 11.5146 秒
// 手动展开版本加速比: 9.86

// 手动展开版本结果校验和: 2.655308e+07
// 手动展开版本结果误差: 0.000000e+00

// 组合优化版本运行时间: 14.9153 秒
// 组合优化版本加速比: 7.61

// 组合优化版本结果校验和: 2.655308e+07
// 组合优化版本结果误差: 0.000000e+00

// ======= 性能总结 =======
// 串行版本:      113.5284 秒 (加速比: 1.00)
// parallel for:  14.4790 秒 (加速比: 7.84)
// unroll:        18.8483 秒 (加速比: 6.02)
// simd+unroll:   19.8744 秒 (加速比: 5.71)
// 手动展开:      11.5146 秒 (加速比: 9.86)
// 组合优化:      14.9153 秒 (加速比: 7.61)
// sh-5.0$ gcc -fopenmp -foffload=disable -O2 OpenMP_Classify/1/unroll_example.c -o OpenMP_Classify/1/unroll_example -lm
// sh-5.0$ OpenMP_Classify/1/unroll_example
// 运行线程数: 64
// 数组大小: 100000000
// 多项式项数: 20
// 重复计算次数: 50

// 串行版本运行时间: 124.3718 秒
// 串行版本加速比: 1.00

// 串行版本结果校验和: 2.655308e+07

// parallel for版本运行时间: 4.1593 秒
// parallel for版本加速比: 29.90

// parallel for版本结果校验和: 2.655308e+07
// parallel for版本结果误差: 0.000000e+00

// unroll版本运行时间: 5.0647 秒
// unroll版本加速比: 24.56

// unroll版本结果校验和: 2.655308e+07
// unroll版本结果误差: 0.000000e+00

// simd+unroll版本运行时间: 4.9215 秒
// simd+unroll版本加速比: 25.27

// simd+unroll版本结果校验和: 2.655308e+07
// simd+unroll版本结果误差: 0.000000e+00

// 手动展开版本运行时间: 3.8372 秒
// 手动展开版本加速比: 32.41

// 手动展开版本结果校验和: 2.655308e+07
// 手动展开版本结果误差: 0.000000e+00

// 组合优化版本运行时间: 5.5622 秒
// 组合优化版本加速比: 22.36

// 组合优化版本结果校验和: 2.655308e+07
// 组合优化版本结果误差: 0.000000e+00

// ======= 性能总结 =======
// 串行版本:      124.3718 秒 (加速比: 1.00)
// parallel for:  4.1593 秒 (加速比: 29.90)
// unroll:        5.0647 秒 (加速比: 24.56)
// simd+unroll:   4.9215 秒 (加速比: 25.27)
// 手动展开:      3.8372 秒 (加速比: 32.41)
// 组合优化:      5.5622 秒 (加速比: 22.36)