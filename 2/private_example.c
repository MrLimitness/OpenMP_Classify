```bash
#!/bin/bash

# 设置默认线程数
THREADS=16

# 处理命令行参数
while getopts "t:" opt; do
  case $opt in
    t) THREADS=$OPTARG ;;
    *) echo "用法: $0 [-t 线程数]" >&2
       exit 1 ;;
  esac
done

# 导出OpenMP线程数
export OMP_NUM_THREADS=$THREADS

cat <<EOF > matrix_multiply.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 2000
double a[N][N], b[N][N], c[N][N];

void initialize() {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = (double)rand()/RAND_MAX;
            b[i][j] = (double)rand()/RAND_MAX;
        }
    }
}

// 串行版本
void serial_multiply() {
    double start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            c[i][j] = 0;
            for (int k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("串行版本运行时间: %.2f秒\n", end - start);
}

// parallel for版本
void parallel_for_version() {
    double start = omp_get_wtime();
    #pragma omp parallel for schedule(static) private(i, j, k)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            c[i][j] = 0;
            for (int k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("parallel for版本运行时间: %.2f秒\n加速比: %.2fX\n", end - start, serial_time/(end - start));
}

// 使用private的版本1
void private_version1() {
    double start = omp_get_wtime();
    #pragma omp parallel private(j, k)
    {
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                c[i][j] = 0;
                for (int k = 0; k < N; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
    double end = omp_get_wtime();
    printf("private版本1运行时间: %.2f秒\n加速比: %.2fX\n", end - start, serial_time/(end - start));
}

// 使用private的版本2（结合collapse）
void private_version2() {
    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2) private(k) schedule(static, 16)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            c[i][j] = 0;
            for (int k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("private版本2运行时间: %.2f秒\n加速比: %.2fX\n", end - start, serial_time/(end - start));
}

// 使用private的版本3（嵌套并行）
void private_version3() {
    double start = omp_get_wtime();
    #pragma omp parallel private(j)
    {
        #pragma omp for private(k)
        for (int i = 0; i < N; i++) {
            #pragma omp parallel for private(k)
            for (int j = 0; j < N; j++) {
                c[i][j] = 0;
                for (int k = 0; k < N; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
    double end = omp_get_wtime();
    printf("private版本3运行时间: %.2f秒\n加速比: %.2fX\n", end - start, serial_time/(end - start));
}

int main() {
    double base_start;
    initialize();
    
    // 先运行串行版本获取基准时间
    base_start = omp_get_wtime();
    serial_multiply();
    serial_time = omp_get_wtime() - base_start;
    
    // 运行各个并行版本
    parallel_for_version();
    private_version1();
    private_version2();
    private_version3();
    
    return 0;
}
EOF

# 编译并运行
gcc -fopenmp matrix_multiply.c -o matrix_multiply
./matrix_multiply