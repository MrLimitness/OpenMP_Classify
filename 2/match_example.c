```bash
#!/bin/bash

# 设置线程数，可修改为16/64/144等
THREADS=4

# 编译并运行串行版本
echo "编译串行版本..."
gcc -o serial serial.c
echo "运行串行版本..."
SERIAL_TIME=$(./serial)
echo "串行时间: ${SERIAL_TIME}秒"

# 编译并运行parallel for版本
echo "编译parallel for版本..."
gcc -o parallel_for parallel_for.c -fopenmp
echo "运行parallel for版本..."
PARALLEL_FOR_TIME=$(OMP_NUM_THREADS=$THREADS ./parallel_for)
SPEEDUP_FOR=$(echo "$SERIAL_TIME / $PARALLEL_FOR_TIME" | bc -l)
echo "parallel for时间: ${PARALLEL_FOR_TIME}秒, 加速比: ${SPEEDUP_FOR}"

# 编译并运行match版本1
echo "编译match版本1..."
gcc -o match1 match1.c -fopenmp
echo "运行match版本1..."
MATCH1_TIME=$(OMP_NUM_THREADS=$THREADS ./match1)
SPEEDUP_MATCH1=$(echo "$SERIAL_TIME / $MATCH1_TIME" | bc -l)
echo "match版本1时间: ${MATCH1_TIME}秒, 加速比: ${SPEEDUP_MATCH1}"

# 编译并运行match版本2
echo "编译match版本2..."
gcc -o match2 match2.c -fopenmp
echo "运行match版本2..."
MATCH2_TIME=$(OMP_NUM_THREADS=$THREADS ./match2)
SPEEDUP_MATCH2=$(echo "$SERIAL_TIME / $MATCH2_TIME" | bc -l)
echo "match版本2时间: ${MATCH2_TIME}秒, 加速比: ${SPEEDUP_MATCH2}"

# 清理生成的可执行文件
rm serial parallel_for match1 match2
```

```c
// serial.c
#include <stdio.h>
#include <time.h>
#define N 1024

int main() {
    double A[N][N], B[N][N], C[N][N];
    clock_t start = clock();

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }

    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i][j] += A[i][k] * B[k][j];

    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("运行时间: %.2f秒\n", time);
    return 0;
}

// parallel_for.c
#include <stdio.h>
#include <time.h>
#include <omp.h>
#define N 1024

int main() {
    double A[N][N], B[N][N], C[N][N];
    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i][j] += A[i][k] * B[k][j];

    double time = omp_get_wtime() - start;
    printf("运行时间: %.2f秒\n", time);
    return 0;
}

// match1.c
#include <stdio.h>
#include <time.h>
#include <omp.h>
#define N 1024

int main() {
    double A[N][N], B[N][N], C[N][N];
    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i][j] = i + j;
                B[i][j] = i - j;
            }

        #pragma omp master
        {
            for (int i = 0; i < N; i++)
                for (int k = 0; k < N; k++)
                    for (int j = 0; j < N; j++)
                        C[i][j] += A[i][k] * B[k][j];
        }
    }

    double time = omp_get_wtime() - start;
    printf("运行时间: %.2f秒\n", time);
    return 0;
}

// match2.c
#include <stdio.h>
#include <time.h>
#include <omp.h>
#define N 1024

int main() {
    double A[N][N], B[N][N], C[N][N];
    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    A[i][j] = i + j;

            #pragma omp section
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    B[i][j] = i - j;
        }

        #pragma omp master
        {
            #pragma omp parallel for
            for (int i = 0; i < N; i++)
                for (int k = 0; k < N; k++)
                    for (int j = 0; j < N; j++)
                        C[i][j] += A[i][k] * B[k][j];
        }
    }

    double time = omp_get_wtime() - start;
    printf("运行时间: %.2f秒\n", time);
    return 0;
}