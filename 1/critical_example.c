```bash
#!/bin/bash

# 设置矩阵大小和线程数（可修改）
MATRIX_SIZE=1024
THREADS=${1:-16}  # 默认16线程，运行时可传入参数指定，如：./script.sh 64

# 生成C代码文件
cat > omp_matrix.c <<EOF
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N $MATRIX_SIZE

void init_matrix(double *mat) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i*N + j] = (double)rand() / RAND_MAX;
        }
    }
}

// 串行版本
void serial_matmul(double *A, double *B, double *C) {
    clock_t start = clock();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    printf("串行版本运行时间: %.2fs\n", (double)(clock() - start)/CLOCKS_PER_SEC);
}

// parallel for版本
void parallel_for_matmul(double *A, double *B, double *C) {
    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    printf("Parallel for版本运行时间: %.2fs\n", omp_get_wtime() - start);
}

// critical版本（带负载均衡）
void critical_matmul(double *A, double *B, double *C) {
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        double *local_C = (double*)calloc(N*N, sizeof(double));
        
        #pragma omp for nowait
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    local_C[i*N + j] += A[i*N + k] * B[k*N + j];
                }
            }
        }
        
        #pragma omp critical
        {
            for (int i = 0; i < N*N; i++) {
                C[i] += local_C[i];
            }
        }
        free(local_C);
    }
    printf("Critical版本运行时间: %.2fs\n", omp_get_wtime() - start);
}

int main() {
    double *A = (double*)malloc(N*N*sizeof(double));
    double *B = (double*)malloc(N*N*sizeof(double));
    double *C = (double*)malloc(N*N*sizeof(double));

    srand(time(NULL));
    init_matrix(A);
    init_matrix(B);

    // 串行版本
    printf("\n----- 串行版本运行中 -----\n");
    serial_matmul(A, B, C);

    // parallel for版本
    printf("\n----- Parallel for版本运行中 (%d线程) -----\n", $THREADS);
    omp_set_num_threads($THREADS);
    parallel_for_matmul(A, B, C);

    // critical版本
    printf("\n----- Critical版本运行中 (%d线程) -----\n", $THREADS);
    omp_set_num_threads($THREADS);
    critical_matmul(A, B, C);

    free(A);
    free(B);
    free(C);
    return 0;
}
EOF

# 编译并运行
gcc -fopenmp omp_matrix.c -o omp_matrix -O3
echo "============================= 运行测试 ============================="
./omp_matrix

# 计算加速比
echo "\n============================ 加速比统计 ============================"
serial_time=$(./omp_matrix | grep '串行' | awk '{print $4}' | tr -d 's')
parallel_time=$(./omp_matrix | grep 'Parallel for' | awk '{print $5}' | tr -d 's')
critical_time=$(./omp_matrix | grep 'Critical' | awk '{print $5}' | tr -d 's')

speedup_parallel=$(echo "scale=2; $serial_time / $parallel_time" | bc)
speedup_critical=$(echo "scale=2; $serial_time / $critical_time" | bc)

printf "Parallel for版本加速比: %.2fx\n" $speedup_parallel
printf "Critical版本加速比:     %.2fx\n" $speedup_critical
echo "=================================================================="