```bash
#!/bin/bash

# 设置矩阵大小和线程数（可修改）
MATRIX_SIZE=1024
THREADS=(16 64 144)  # 可自主设置线程数

# 生成测试数据
echo "生成矩阵数据（${MATRIX_SIZE}x${MATRIX_SIZE}）..."
dd if=/dev/urandom of=matrix_a.bin bs=$((MATRIX_SIZE*MATRIX_SIZE*4)) count=1 >/dev/null 2>&1
dd if=/dev/urandom of=matrix_b.bin bs=$((MATRIX_SIZE*MATRIX_SIZE*4)) count=1 >/dev/null 2>&1

cat <<EOF > matrix_interop.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

#define N $MATRIX_SIZE

float* read_matrix(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    float* matrix = (float*)malloc(N*N*sizeof(float));
    fread(matrix, sizeof(float), N*N, fp);
    fclose(fp);
    return matrix;
}

void serial_matmul(float *A, float *B, float *C) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    
    gettimeofday(&end, NULL);
    printf("串行版本运行时间: %.3f秒\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6);
}

void parallel_for_matmul(float *A, float *B, float *C) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    
    gettimeofday(&end, NULL);
    printf("Parallel for版本运行时间: %.3f秒\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6);
}

void interop_matmul(float *A, float *B, float *C) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    #pragma omp interop init(targetsync: device(0))  // 初始化设备交互
    #pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N]) device(0)
    {
        #pragma omp target teams distribute parallel for collapse(2) device(0)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[i*N + k] * B[k*N + j];
                }
                C[i*N + j] = sum;
            }
        }
    }
    #pragma omp interop destroy(device(0))
    
    gettimeofday(&end, NULL);
    printf("Interop版本运行时间: %.3f秒\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6);
}

int main() {
    float *A = read_matrix("matrix_a.bin");
    float *B = read_matrix("matrix_b.bin");
    float *C = (float*)malloc(N*N*sizeof(float));

    // 串行版本
    printf("\n【串行版本】\n");
    serial_matmul(A, B, C);
    
    // Parallel for版本
    printf("\n【Parallel for版本】\n");
    parallel_for_matmul(A, B, C);
    
    // Interop版本
    printf("\n【Interop版本】\n");
    interop_matmul(A, B, C);

    free(A);
    free(B);
    free(C);
    return 0;
}
EOF

# 编译程序
echo "编译程序中..."
gcc -fopenmp matrix_interop.c -o matrix_interop -O3

# 运行测试
for threads in "${THREADS[@]}"; do
    echo -e "\n\n========================= 使用 ${threads} 线程 ========================="
    export OMP_NUM_THREADS=$threads
    ./matrix_interop | awk -v serial_time=0 -v last_time=0 '
    /串行版本运行时间/ {serial_time = \$3; last_time = \$3; print}
    /Parallel for版本运行时间/ {
        printf("加速比: %.2fx\n", serial_time/\$3)
        last_time = \$3
    }
    /Interop版本运行时间/ {
        printf("加速比: %.2fx\n", serial_time/\$3)
        last_time = \$3
    }
    {print}
    '
done

# 清理临时文件
rm matrix_a.bin matrix_b.bin matrix_interop 2>/dev/null