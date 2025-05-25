```bash
#!/bin/bash

# 设置矩阵大小和线程数
MATRIX_SIZE=2048
THREADS=${1:-16}  # 默认16线程

cat <<EOF > matmul.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N $MATRIX_SIZE

// 声明矩阵结构体和自定义映射器
#pragma omp declare mapper(struct Matrix) map(to: m.data[0:m.size*m.size])
struct Matrix {
    double *data;
    int size;
};

void init_matrix(struct Matrix *mat) {
    mat->size = N;
    mat->data = (double *)malloc(N * N * sizeof(double));
    for (int i = 0; i < N*N; i++) {
        mat->data[i] = (double)rand() / RAND_MAX;
    }
}

// 串行版本
void serial_matmul(struct Matrix *A, struct Matrix *B, struct Matrix *C) {
    double start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A->data[i*N + k] * B->data[k*N + j];
            }
            C->data[i*N + j] = sum;
        }
    }
    printf("串行版本运行时间: %.2f秒\n", omp_get_wtime() - start);
}

// Parallel for版本
void parallel_for_matmul(struct Matrix *A, struct Matrix *B, struct Matrix *C) {
    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A->data[i*N + k] * B->data[k*N + j];
            }
            C->data[i*N + j] = sum;
        }
    }
    printf("Parallel for版本运行时间: %.2f秒\n", omp_get_wtime() - start);
}

// Mapper版本
void mapper_matmul(struct Matrix *A, struct Matrix *B, struct Matrix *C) {
    double start = omp_get_wtime();
    #pragma omp target data map(to: A[0:1], B[0:1]) map(from: C[0:1])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A->data[i*N + k] * B->data[k*N + j];
                }
                C->data[i*N + j] = sum;
            }
        }
    }
    printf("Mapper版本运行时间: %.2f秒\n", omp_get_wtime() - start);
}

int main() {
    struct Matrix A, B, C;
    init_matrix(&A);
    init_matrix(&B);
    C.data = (double *)malloc(N * N * sizeof(double));

    // 串行版本
    serial_matmul(&A, &B, &C);

    // Parallel for版本
    omp_set_num_threads($THREADS);
    parallel_for_matmul(&A, &B, &C);

    // Mapper版本
    omp_set_num_threads($THREADS);
    mapper_matmul(&A, &B, &C);

    free(A.data);
    free(B.data);
    free(C.data);
    return 0;
}
EOF

# 编译和运行
echo "编译中..."
gcc -fopenmp -o matmul_serial matmul.c -D SERIAL
gcc -fopenmp -o matmul_parallel_for matmul.c
gcc -fopenmp -o matmul_mapper matmul.c -foffload=nvptx-none

echo "运行测试..."
echo "--------------------------------"

serial_time=$(./matmul_serial | grep '串行' | awk '{print $4}')
parallel_time=$(./matmul_parallel_for | grep 'Parallel' | awk '{print $4}')
mapper_time=$(./matmul_mapper | grep 'Mapper' | awk '{print $4}')

echo "加速比报告:"
printf "Parallel for版本加速比: %.2fX\n" $(echo "$serial_time / $parallel_time" | bc -l)
printf "Mapper版本加速比: %.2fX\n" $(echo "$serial_time / $mapper_time" | bc -l)

rm matmul.c matmul_serial matmul_parallel_for matmul_mapper