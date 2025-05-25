```bash
#!/bin/bash

# 设置线程数（默认144，可通过命令行参数修改）
THREADS=${1:-144}
export OMP_NUM_THREADS=$THREADS
echo "当前使用线程数：$THREADS"

# 矩阵维度设置
SIZE=2000

cat <<EOF > omp_cancel_demo.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#define SIZE $SIZE

void init_matrix(float m[SIZE][SIZE]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

void serial_version() {
    float A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];
    init_matrix(A);
    init_matrix(B);

    double start = omp_get_wtime();
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("串行版本运行时间：%.2f秒\n", end - start);
}

void parallel_for_version() {
    float A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];
    init_matrix(A);
    init_matrix(B);

    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("并行for版本运行时间：%.2f秒\n", end - start);
}

void cancel_version1() {
    float A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];
    init_matrix(A);
    init_matrix(B);
    int found = 0;

    double start = omp_get_wtime();
    #pragma omp parallel shared(found)
    {
        #pragma omp for collapse(2)
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (found) continue;
                C[i][j] = 0;
                for (int k = 0; k < SIZE; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                    if (C[i][j] > 500.0f) {  // 触发阈值
                        #pragma omp atomic write
                        found = 1;
                        #pragma omp cancel for
                    }
                }
                #pragma omp cancellation point for
            }
        }
    }
    double end = omp_get_wtime();
    printf("带cancel版本1运行时间：%.2f秒\n", end - start);
}

int main() {
    printf("矩阵维度：%dx%d\n", SIZE, SIZE);
    
    double t1, t2, t3;
    
    printf("\n【串行版本】\n");
    double start = omp_get_wtime();
    serial_version();
    t1 = omp_get_wtime() - start;
    
    printf("\n【parallel for版本】\n");
    start = omp_get_wtime();
    parallel_for_version();
    t2 = omp_get_wtime() - start;
    
    printf("\n【cancel版本1】\n");
    start = omp_get_wtime();
    cancel_version1();
    t3 = omp_get_wtime() - start;
    
    printf("\n加速比统计：\n");
    printf("parallel for版本加速比：%.2f倍\n", t1 / t2);
    printf("cancel版本1加速比：%.2f倍\n", t1 / t3);
    
    return 0;
}
EOF

gcc -fopenmp -o omp_cancel_demo omp_cancel_demo.c -lm
export OMP_CANCELLATION=true

echo ""
./omp_cancel_demo

rm -f omp_cancel_demo omp_cancel_demo.c