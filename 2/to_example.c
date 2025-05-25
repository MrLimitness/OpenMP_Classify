```bash
#!/bin/bash

# 设置线程数（默认为16）
THREADS=${1:-16}
export OMP_NUM_THREADS=$THREADS

# 矩阵维度设置（2000x2000矩阵大约需要10秒）
SIZE=2000

# 生成C程序文件
cat <<EOF > omp_matrix.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void serial_version() {
    double start = omp_get_wtime();
    int *A = malloc(SIZE*SIZE*sizeof(int));
    int *B = malloc(SIZE*SIZE*sizeof(int));
    int *C = malloc(SIZE*SIZE*sizeof(int));
    
    // 初始化矩阵
    #pragma omp parallel for
    for(int i=0; i<SIZE; i++){
        for(int j=0; j<SIZE; j++){
            A[i*SIZE+j] = i+j;
            B[i*SIZE+j] = i-j;
        }
    }

    // 串行计算
    for(int i=0; i<SIZE; i++){
        for(int j=0; j<SIZE; j++){
            C[i*SIZE+j] = 0;
            for(int k=0; k<SIZE; k++){
                C[i*SIZE+j] += A[i*SIZE+k] * B[k*SIZE+j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("串行版本运行时间: %.2f秒\n", end - start);
    free(A); free(B); free(C);
}

void parallel_for_version() {
    double start = omp_get_wtime();
    int *A = malloc(SIZE*SIZE*sizeof(int));
    int *B = malloc(SIZE*SIZE*sizeof(int));
    int *C = malloc(SIZE*SIZE*sizeof(int));
    
    #pragma omp parallel for
    for(int i=0; i<SIZE; i++){
        for(int j=0; j<SIZE; j++){
            A[i*SIZE+j] = i+j;
            B[i*SIZE+j] = i-j;
        }
    }

    #pragma omp parallel for
    for(int i=0; i<SIZE; i++){
        for(int j=0; j<SIZE; j++){
            C[i*SIZE+j] = 0;
            for(int k=0; k<SIZE; k++){
                C[i*SIZE+j] += A[i*SIZE+k] * B[k*SIZE+j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("parallel for版本运行时间: %.2f秒\n", end - start);
    free(A); free(B); free(C);
}

void target_version() {
    double start = omp_get_wtime();
    int *A = malloc(SIZE*SIZE*sizeof(int));
    int *B = malloc(SIZE*SIZE*sizeof(int));
    int *C = malloc(SIZE*SIZE*sizeof(int));
    
    #pragma omp parallel for
    for(int i=0; i<SIZE; i++){
        for(int j=0; j<SIZE; j++){
            A[i*SIZE+j] = i+j;
            B[i*SIZE+j] = i-j;
        }
    }

    #pragma omp target teams distribute parallel for map(to: A[0:SIZE*SIZE], B[0:SIZE*SIZE]) map(from: C[0:SIZE*SIZE])
    for(int i=0; i<SIZE; i++){
        for(int j=0; j<SIZE; j++){
            C[i*SIZE+j] = 0;
            for(int k=0; k<SIZE; k++){
                C[i*SIZE+j] += A[i*SIZE+k] * B[k*SIZE+j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("target版本运行时间: %.2f秒\n", end - start);
    free(A); free(B); free(C);
}

int main() {
    double serial_time, parallel_time, target_time;
    
    printf("矩阵维度: %dx%d\n", SIZE, SIZE);
    printf("使用线程数: %d\n", OMP_NUM_THREADS);
    printf("\n===== 串行版本 =====\n");
    serial_version();
    
    printf("\n===== parallel for版本 =====\n");
    parallel_for_version();
    
    printf("\n===== target版本 =====\n");
    target_version();
    
    return 0;
}
EOF

# 编译并运行程序
gcc -fopenmp -foffload=nvptx-none -O3 omp_matrix.c -o omp_matrix
if [ $? -eq 0 ]; then
    ./omp_matrix
else
    echo "编译失败，请检查OpenMP环境"
fi