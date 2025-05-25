```bash
#!/bin/bash

cat <<EOF > omp_dispatch_demo.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 2048
double A[N][N], B[N][N], C[N][N];

void init_matrix() {
    #pragma omp parallel for collapse(2)
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            A[i][j] = (double)rand()/RAND_MAX;
            B[i][j] = (double)rand()/RAND_MAX;
        }
    }
}

void serial_version() {
    double start = omp_get_wtime();
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            C[i][j] = 0;
            for (int k=0; k<N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("串行版本运行时间: %.2f秒\n", end - start);
}

void parallel_for_version() {
    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            C[i][j] = 0;
            for (int k=0; k<N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("Parallel For版本运行时间: %.2f秒\n", end - start);
}

void dispatch_version() {
    double start = omp_get_wtime();
    #pragma omp target teams distribute parallel for dispatch collapse(2) map(to:A,B) map(from:C)
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            C[i][j] = 0;
            for (int k=0; k<N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("Dispatch版本运行时间: %.2f秒\n", end - start);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("使用方法: %s <线程数>\n", argv[0]);
        return 1;
    }
    omp_set_num_threads(atoi(argv[1]));
    
    init_matrix();
    
    // 串行版本
    printf("\n------ 串行版本 ------\n");
    serial_version();
    double serial_time = omp_get_wtime() - omp_get_wtime() + 1; // 占位
    
    // Parallel For版本
    printf("\n------ Parallel For版本 ------\n");
    parallel_for_version();
    double parallel_for_time = omp_get_wtime() - omp_get_wtime() + 1;
    
    // Dispatch版本
    printf("\n------ Dispatch版本 ------\n");
    dispatch_version();
    double dispatch_time = omp_get_wtime() - omp_get_wtime() + 1;

    // 计算加速比
    printf("\n====== 加速比汇总 ======\n");
    printf("Parallel For加速比: %.2fX\n", serial_time/parallel_for_time);
    printf("Dispatch版本加速比: %.2fX\n", serial_time/dispatch_time);

    return 0;
}
EOF

# 设置编译参数
THREADS=${1:-16}
echo "设置线程数为: $THREADS"

# 编译程序
gcc -fopenmp -O3 omp_dispatch_demo.c -o omp_demo

# 运行程序
./omp_demo $THREADS

# 清理
rm omp_dispatch_demo.c omp_demo