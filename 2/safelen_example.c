#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define SIZE 1000000000
#define SKIP 4

// 串行版本
double serial_version(double* a, double* b, double* c) {
    double start_time = omp_get_wtime();

    for (int i = 0; i < SIZE - SKIP; i++) {
        c[i] = a[i] * a[i + SKIP] + b[i] * b[i + SKIP];
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// parallel for 多线程版本
double parallel_for_version(double* a, double* b, double* c, int num_threads) {
    double start_time = omp_get_wtime();

    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for (int i = 0; i < SIZE - SKIP; i++) {
        c[i] = a[i] * a[i + SKIP] + b[i] * b[i + SKIP];
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// 使用 simd safelen 优化版本（只使用向量化，不开线程）
double safelen_version(double* a, double* b, double* c) {
    double start_time = omp_get_wtime();

    // 使用 SIMD 指令并指定最大向量安全长度
    #pragma omp simd safelen(SKIP)
    for (int i = 0; i < SIZE - SKIP; i++) {
        c[i] = a[i] * a[i + SKIP] + b[i] * b[i + SKIP];
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

    double* a = (double*)malloc((SIZE + SKIP) * sizeof(double));
    double* b = (double*)malloc((SIZE + SKIP) * sizeof(double));
    double* c = (double*)malloc(SIZE * sizeof(double));

    // 初始化数组
    for (int i = 0; i < SIZE + SKIP; i++) {
        a[i] = (double)rand() / RAND_MAX;
        b[i] = (double)rand() / RAND_MAX;
    }
    memset(c, 0, SIZE * sizeof(double));

    printf("数组大小: %d, 跳步: %d\n", SIZE, SKIP);

    printf("\n运行串行版本...\n");
    double serial_time = serial_version(a, b, c);
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    printf("加速比: 1.00\n");

    memset(c, 0, SIZE * sizeof(double));
    printf("\n运行 parallel for 多线程版本...\n");
    double parallel_time = parallel_for_version(a, b, c, num_threads);
    printf("parallel for 运行时间: %.4f 秒\n", parallel_time);
    printf("加速比: %.2f\n", serial_time / parallel_time);

    memset(c, 0, SIZE * sizeof(double));
    printf("\n运行 safelen 向量化版本...\n");
    double safelen_time = safelen_version(a, b, c);
    printf("safelen 运行时间: %.4f 秒\n", safelen_time);
    printf("加速比: %.2f\n", serial_time / safelen_time);

    free(a);
    free(b);
    free(c);

    return 0;
}
// 使用线程数: 16
// 数组大小: 1000000000, 跳步: 2

// 运行串行版本...
// 串行版本运行时间: 9.4758 秒
// 加速比: 1.00

// 运行 parallel for 多线程版本...
// parallel for 运行时间: 0.8841 秒
// 加速比: 10.72

// 运行 safelen 向量化版本...
// safelen 运行时间: 4.5522 秒
// 加速比: 2.08

// 使用线程数: 64
// 数组大小: 1000000000, 跳步: 4

// 运行串行版本...
// 串行版本运行时间: 9.2096 秒
// 加速比: 1.00

// 运行 parallel for 多线程版本...
// parallel for 运行时间: 0.4085 秒
// 加速比: 22.55

// 运行 safelen 向量化版本...
// safelen 运行时间: 2.3073 秒
// 加速比: 3.99