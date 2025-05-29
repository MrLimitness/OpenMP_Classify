#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>

#define SIZE 8192
#define ITERATIONS 500
#define OFFSET 5

void init_arrays(double* a, double* b, int size) {
    for (int i = 0; i < size + OFFSET; i++) {
        a[i] = (double)rand() / RAND_MAX;
        b[i] = 0.0;
    }
}

double serial_version(double* a, double* b, int size) {
    double start_time = omp_get_wtime();

    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < size; i++) {
            double temp = 0.0;
            for (int j = 0; j < 1000; j++) {
                temp += sin(a[i+OFFSET]) * cos(a[i]) / (j+1.0);
            }
            b[i] = temp;
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

double parallel_for_version(double* a, double* b, int size, int num_threads) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(num_threads);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            double temp = 0.0;
            for (int j = 0; j < 1000; j++) {
                temp += sin(a[i+OFFSET]) * cos(a[i]) / (j+1.0);
            }
            b[i] = temp;
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

double linear_version1(double* a, double* b, int size, int num_threads) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(num_threads);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        int i;
        #pragma omp parallel for simd linear(i:1)
        for (i = 0; i < size; i++) {
            double temp = 0.0;
            for (int j = 0; j < 1000; j++) {
                temp += sin(a[i+OFFSET]) * cos(a[i]) / (j+1.0);
            }
            b[i] = temp;
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

double linear_version2(double* a, double* b, int size, int num_threads) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(num_threads);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        int i;
        #pragma omp parallel for simd linear(i:1) aligned(a,b:64)
        for (i = 0; i < size; i++) {
            double temp = 0.0;
            for (int j = 0; j < 1000; j++) {
                temp += sin(a[i+OFFSET]) * cos(a[i]) / (j+1.0);
            }
            b[i] = temp;
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

double linear_version3(double* a, double* b, int size, int num_threads) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(num_threads);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        int i;
        #pragma omp parallel for simd linear(i:1) simdlen(8)
        for (i = 0; i < size; i++) {
            double temp = 0.0;
            for (int j = 0; j < 1000; j++) {
                temp += sin(a[i+OFFSET]) * cos(a[i]) / (j+1.0);
            }
            b[i] = temp;
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

double linear_version4(double* a, double* b, int size, int num_threads) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(num_threads);
    int block_size = 128;

    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (int ii = 0; ii < size; ii += block_size) {
            int end = ii + block_size < size ? ii + block_size : size;
            int i;
            #pragma omp simd linear(i:1) aligned(a,b:64)
            for (i = ii; i < end; i++) {
                double temp = 0.0;
                for (int j = 0; j < 1000; j++) {
                    temp += sin(a[i+OFFSET]) * cos(a[i]) / (j+1.0);
                }
                b[i] = temp;
            }
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main(int argc, char* argv[]) {
    int num_threads = 16;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    printf("使用线程数: %d\n", num_threads);

    srand(time(NULL));

    double* a = (double*)aligned_alloc(64, (SIZE + OFFSET) * sizeof(double));
    double* b = (double*)aligned_alloc(64, SIZE * sizeof(double));

    if (!a || !b) {
        printf("内存分配失败！\n");
        return 1;
    }

    init_arrays(a, b, SIZE);
    printf("数组大小: %d, 迭代次数: %d\n", SIZE, ITERATIONS);

    printf("\n运行串行版本...\n");
    double serial_time = serial_version(a, b, SIZE);
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    printf("加速比: 1.00\n");

    memset(b, 0, SIZE * sizeof(double));
    printf("\n运行parallel for版本...\n");
    double parallel_time = parallel_for_version(a, b, SIZE, num_threads);
    printf("parallel for版本运行时间: %.4f 秒\n", parallel_time);
    printf("加速比: %.2f\n", serial_time / parallel_time);

    memset(b, 0, SIZE * sizeof(double));
    printf("\n运行linear版本1...\n");
    double linear_time1 = linear_version1(a, b, SIZE, num_threads);
    printf("linear版本1运行时间: %.4f 秒\n", linear_time1);
    printf("加速比: %.2f\n", serial_time / linear_time1);

    memset(b, 0, SIZE * sizeof(double));
    printf("\n运行linear版本2...\n");
    double linear_time2 = linear_version2(a, b, SIZE, num_threads);
    printf("linear版本2运行时间: %.4f 秒\n", linear_time2);
    printf("加速比: %.2f\n", serial_time / linear_time2);

    memset(b, 0, SIZE * sizeof(double));
    printf("\n运行linear版本3...\n");
    double linear_time3 = linear_version3(a, b, SIZE, num_threads);
    printf("linear版本3运行时间: %.4f 秒\n", linear_time3);
    printf("加速比: %.2f\n", serial_time / linear_time3);

    memset(b, 0, SIZE * sizeof(double));
    printf("\n运行linear版本4...\n");
    double linear_time4 = linear_version4(a, b, SIZE, num_threads);
    printf("linear版本4运行时间: %.4f 秒\n", linear_time4);
    printf("加速比: %.2f\n", serial_time / linear_time4);

    free(a);
    free(b);
    printf("\n所有测试完成！\n");

    return 0;
}
// 使用线程数: 16
// 数组大小: 8192, 迭代次数: 500

// 运行串行版本...
// 串行版本运行时间: 8.4867 秒
// 加速比: 1.00

// 运行parallel for版本...
// parallel for版本运行时间: 11.2524 秒
// 加速比: 0.75

// 运行linear版本1...
// linear版本1运行时间: 12.4596 秒
// 加速比: 0.68

// 运行linear版本2...
// linear版本2运行时间: 7.8953 秒
// 加速比: 1.07

// 运行linear版本3...
// linear版本3运行时间: 0.8982 秒
// 加速比: 9.45

// 运行linear版本4...
// linear版本4运行时间: 3.1757 秒
// 加速比: 2.67
