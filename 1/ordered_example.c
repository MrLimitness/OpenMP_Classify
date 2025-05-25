#!/bin/bash

# 设置线程数，默认为1
THREADS=${1:-1}
export OMP_NUM_THREADS=$THREADS

echo "设置线程数：$THREADS"

# 生成C源代码
cat <<EOF > matrix_multiply.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 2000

void serial_version() {
    double *A = malloc(N*N*sizeof(double));
    double *B = malloc(N*N*sizeof(double));
    double *C = malloc(N*N*sizeof(double));
    double start = omp_get_wtime();

    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            A[i*N+j] = (double)rand()/RAND_MAX;

    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            B[i*N+j] = (double)rand()/RAND_MAX;

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            double sum = 0.0;
            for(int k=0; k<N; k++)
                sum += A[i*N+k] * B[k*N+j];
            C[i*N+j] = sum;
        }
    }
    
    double end = omp_get_wtime();
    printf("串行版本运行时间：%.2f秒\n", end - start);
    free(A); free(B); free(C);
}

void parallel_for_version() {
    double *A = malloc(N*N*sizeof(double));
    double *B = malloc(N*N*sizeof(double));
    double *C = malloc(N*N*sizeof(double));
    double start = omp_get_wtime();

    #pragma omp parallel for
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            A[i*N+j] = (double)rand()/RAND_MAX;

    #pragma omp parallel for
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            B[i*N+j] = (double)rand()/RAND_MAX;

    #pragma omp parallel for collapse(2)
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++) {
            double sum = 0.0;
            for(int k=0; k<N; k++)
                sum += A[i*N+k] * B[k*N+j];
            C[i*N+j] = sum;
        }

    double end = omp_get_wtime();
    printf("Parallel for版本运行时间：%.2f秒\n", end - start);
    free(A); free(B); free(C);
}

void ordered_version() {
    double *A = malloc(N*N*sizeof(double));
    double *B = malloc(N*N*sizeof(double));
    double *C = malloc(N*N*sizeof(double));
    double start = omp_get_wtime();

    #pragma omp parallel for
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            A[i*N+j] = (double)rand()/RAND_MAX;

    #pragma omp parallel for
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            B[i*N+j] = (double)rand()/RAND_MAX;

    #pragma omp parallel for ordered
    for(int i=0; i<N; i++) {
        // Ordered指令应用在需要顺序执行的部分
        #pragma omp ordered
        { 
            for(int j=0; j<N; j++) {
                double sum = 0.0;
                for(int k=0; k<N; k++)
                    sum += A[i*N+k] * B[k*N+j];
                C[i*N+j] = sum;
            }
        }
    }

    double end = omp_get_wtime();
    printf("Ordered版本运行时间：%.2f秒\n", end - start);
    free(A); free(B); free(C);
}

int main() {
    printf("=== 矩阵乘法性能测试 ===\n");
    
    printf("\n[串行版本]");
    serial_version();
    
    printf("\n[Parallel for版本]");
    parallel_for_version();
    
    printf("\n[Ordered版本]");
    ordered_version();
    
    return 0;
}
EOF

# 编译程序
gcc matrix_multiply.c -o matrix_multiply -fopenmp -O3 -lm

# 运行程序并捕获输出
output=$(./matrix_multiply)

# 提取各版本运行时间
t_serial=$(echo "$output" | grep '串行' | awk '{print $4}')
t_parallel=$(echo "$output" | grep 'Parallel' | awk '{print $4}')
t_ordered=$(echo "$output" | grep 'Ordered' | awk '{print $4}')

# 计算加速比
speedup_parallel=$(echo "scale=2; $t_serial / $t_parallel" | bc)
speedup_ordered=$(echo "scale=2; $t_serial / $t_ordered" | bc)

# 打印结果
echo -e "\n=== 加速比 ==="
echo "Parallel for加速比：$speedup_parallel"
echo "Ordered版本加速比：$speedup_ordered"

# 清理生成的文件
rm matrix_multiply.c matrix_multiply