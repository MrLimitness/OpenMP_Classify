```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 2048
#define TILE_SIZE 32

float **a, **b, **c;

void init_matrix(float ***matrix) {
    *matrix = (float **)malloc(N * sizeof(float *));
    for (int i = 0; i < N; i++) {
        (*matrix)[i] = (float *)malloc(N * sizeof(float));
        for (int j = 0; j < N; j++) {
            (*matrix)[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

void serial() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

void parallel_for() {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

void tile_version() {
    #pragma omp parallel for
    for (int i = 0; i < N; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            #pragma omp tile sizes(TILE_SIZE, TILE_SIZE)
            for (int ii = i; ii < i + TILE_SIZE; ii++) {
                for (int jj = j; jj < j + TILE_SIZE; jj++) {
                    float sum = 0.0;
                    for (int k = 0; k < N; k++) {
                        sum += a[ii][k] * b[k][jj];
                    }
                    c[ii][jj] = sum;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("使用方法: %s <版本号1-3> <线程数>\n", argv[0]);
        return 1;
    }

    int version = atoi(argv[1]);
    int threads = atoi(argv[2]);
    omp_set_num_threads(threads);

    init_matrix(&a);
    init_matrix(&b);
    init_matrix(&c);

    double start, end;
    double serial_time = 0.0;

    if (version == 1) {
        start = omp_get_wtime();
        serial();
        end = omp_get_wtime();
        serial_time = end - start;
        printf("串行版本运行时间: %.2f秒\n", serial_time);
    } else {
        // 先运行串行版本获取基准时间
        start = omp_get_wtime();
        serial();
        end = omp_get_wtime();
        serial_time = end - start;
    }

    // 重置结果矩阵
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            c[i][j] = 0.0;

    start = omp_get_wtime();
    switch(version) {
        case 1: serial(); break;
        case 2: parallel_for(); break;
        case 3: tile_version(); break;
        default: printf("无效版本号\n"); return 1;
    }
    end = omp_get_wtime();

    double parallel_time = end - start;
    if (version != 1) {
        printf("并行版本运行时间: %.2f秒\n", parallel_time);
        printf("加速比: %.2f倍\n", serial_time / parallel_time);
    }

    // 释放内存
    for (int i = 0; i < N; i++) {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }
    free(a);
    free(b);
    free(c);

    return 0;
}
```

使用方法：  
`gcc -O3 -fopenmp -o matmul matmul.c`  
运行命令：  
`./matmul <1-3> <线程数>` （1=串行，2=parallel for，3=tile版本）  

示例输出：  
`./matmul 1 1` 输出串行时间  
`./matmul 2 16` 输出并行时间和加速比  
`./matmul 3 16` 输出tile版本时间和加速比  

该脚本：  
1. 使用2048x2048大矩阵确保计算量  
2. 动态内存分配避免栈溢出  
3. 支持三个版本对比  
4. 可设置任意线程数（建议16/64/144）  
5. 自动计算加速比  
6. 中文输出结果