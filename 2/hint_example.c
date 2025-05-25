#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>

#define SIZE 1000
#define ITERATIONS 500

void init_arrays(double* A, double* B, double* C, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
        C[i] = 0.0;
    }
}

double serial_version(double* A, double* B, double* C, int size) {
    double start_time = omp_get_wtime();

    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < size; i++) {
            double val = A[i];
            for (int j = 0; j < size; j++) {
                C[i] += sin(val) * cos(B[j]) * sqrt(fabs(val - B[j])) + tan(val * B[j] * 0.01);
            }
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

double parallel_for_version(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();

    omp_set_num_threads(num_threads);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            double val = A[i];
            for (int j = 0; j < size; j++) {
                C[i] += sin(val) * cos(B[j]) * sqrt(fabs(val - B[j])) + tan(val * B[j] * 0.01);
            }
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// version 1: use aligned
double simd_hint_version1(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();

    omp_set_num_threads(num_threads);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            double val = A[i];
            double sum = 0.0;

            #pragma omp simd aligned(B:64) reduction(+:sum)
            for (int j = 0; j < size; j++) {
                sum += sin(val) * cos(B[j]) * sqrt(fabs(val - B[j])) + tan(val * B[j] * 0.01);
            }

            C[i] += sum;
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// version 2: use safelen
double simd_hint_version2(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();

    omp_set_num_threads(num_threads);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            double val = A[i];
            double sum = 0.0;

            #pragma omp simd safelen(64) reduction(+:sum)
            for (int j = 0; j < size; j++) {
                sum += sin(val) * cos(B[j]) * sqrt(fabs(val - B[j])) + tan(val * B[j] * 0.01);
            }

            C[i] += sum;
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// version 3: use simdlen
double simd_hint_version3(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();

    omp_set_num_threads(num_threads);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            double val = A[i];
            double sum = 0.0;

            #pragma omp simd simdlen(8) reduction(+:sum)
            for (int j = 0; j < size; j++) {
                sum += sin(val) * cos(B[j]) * sqrt(fabs(val - B[j])) + tan(val * B[j] * 0.01);
            }

            C[i] += sum;
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// version 4: combined hints
double simd_hint_version4(double* A, double* B, double* C, int size, int num_threads) {
    double start_time = omp_get_wtime();

    omp_set_num_threads(num_threads);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            double val = A[i];
            double sum = 0.0;

            #pragma omp simd simdlen(8) aligned(B:64) reduction(+:sum)
            for (int j = 0; j < size; j++) {
                sum += sin(val) * cos(B[j]) * sqrt(fabs(val - B[j])) + tan(val * B[j] * 0.01);
            }

            C[i] += sum;
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main(int argc, char* argv[]) {
    int num_threads = 64;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    printf("使用线程数: %d\n", num_threads);

    srand(time(NULL));

    double* A = (double*)aligned_alloc(64, SIZE * sizeof(double));
    double* B = (double*)aligned_alloc(64, SIZE * sizeof(double));
    double* C = (double*)aligned_alloc(64, SIZE * sizeof(double));

    if (!A || !B || !C) {
        printf("内存分配失败！\n");
        return 1;
    }

    init_arrays(A, B, C, SIZE);

    printf("数组大小: %d, 迭代次数: %d\n", SIZE, ITERATIONS);

    printf("\n运行串行版本...\n");
    double serial_time = serial_version(A, B, C, SIZE);
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    printf("加速比: 1.00\n");

    memset(C, 0, SIZE * sizeof(double));
    printf("\n运行parallel for版本...\n");
    double parallel_time = parallel_for_version(A, B, C, SIZE, num_threads);
    printf("parallel for版本运行时间: %.4f 秒\n", parallel_time);
    printf("加速比: %.2f\n", serial_time / parallel_time);

    memset(C, 0, SIZE * sizeof(double));
    printf("\n运行simd hint版本1 (aligned)...\n");
    double simd_time1 = simd_hint_version1(A, B, C, SIZE, num_threads);
    printf("simd hint版本1运行时间: %.4f 秒\n", simd_time1);
    printf("加速比: %.2f\n", serial_time / simd_time1);

    memset(C, 0, SIZE * sizeof(double));
    printf("\n运行simd hint版本2 (safelen)...\n");
    double simd_time2 = simd_hint_version2(A, B, C, SIZE, num_threads);
    printf("simd hint版本2运行时间: %.4f 秒\n", simd_time2);
    printf("加速比: %.2f\n", serial_time / simd_time2);

    memset(C, 0, SIZE * sizeof(double));
    printf("\n运行simd hint版本3 (simdlen)...\n");
    double simd_time3 = simd_hint_version3(A, B, C, SIZE, num_threads);
    printf("simd hint版本3运行时间: %.4f 秒\n", simd_time3);
    printf("加速比: %.2f\n", serial_time / simd_time3);

    memset(C, 0, SIZE * sizeof(double));
    printf("\n运行simd hint版本4 (组合)...\n");
    double simd_time4 = simd_hint_version4(A, B, C, SIZE, num_threads);
    printf("simd hint版本4运行时间: %.4f 秒\n", simd_time4);
    printf("加速比: %.2f\n", serial_time / simd_time4);

    free(A);
    free(B);
    free(C);

    printf("\n所有测试完成！\n");

    return 0;
}

// 使用线程数: 16
// 数组大小: 1000, 迭代次数: 500

// 运行串行版本...
// 串行版本运行时间: 12.6694 秒
// 加速比: 1.00

// 运行parallel for版本...
// parallel for版本运行时间: 12.4288 秒
// 加速比: 1.02

// 运行simd hint版本1 (aligned)...
// simd hint版本1运行时间: 13.1680 秒
// 加速比: 0.96

// 运行simd hint版本2 (safelen)...
// simd hint版本2运行时间: 11.0503 秒
// 加速比: 1.15

// 运行simd hint版本3 (simdlen)...
// simd hint版本3运行时间: 5.1570 秒
// 加速比: 2.46

// 运行simd hint版本4 (组合)...
// simd hint版本4运行时间: 5.4171 秒
// 加速比: 2.34


// 使用线程数: 64
// 数组大小: 1000, 迭代次数: 500

// 运行串行版本...
// 串行版本运行时间: 12.5436 秒
// 加速比: 1.00

// 运行parallel for版本...
// parallel for版本运行时间: 18.5980 秒
// 加速比: 0.67

// 运行simd hint版本1 (aligned)...
// simd hint版本1运行时间: 19.5765 秒
// 加速比: 0.64

// 运行simd hint版本2 (safelen)...
// simd hint版本2运行时间: 4.9459 秒
// 加速比: 2.54

// 运行simd hint版本3 (simdlen)...
// simd hint版本3运行时间: 0.2946 秒
// 加速比: 42.58

// 运行simd hint版本4 (组合)...
// simd hint版本4运行时间: 4.0290 秒
// 加速比: 3.11