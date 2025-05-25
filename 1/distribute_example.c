```bash
#!/bin/bash

cat > omp_matrix.c << 'EOL'
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 4096
double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];

void init_matrix() {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }
    }
}

void serial_version() {
    double start = omp_get_wtime();
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
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
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("Parallel For版本运行时间: %.2f秒\n", end - start);
}

void distribute_version() {
    double start = omp_get_wtime();
    #pragma omp target teams distribute parallel for collapse(2) map(to:A,B) map(from:C)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("Distribute版本运行时间: %.2f秒\n", end - start);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("请用运行参数指定线程数（如：16/64/144）\n");
        return 1;
    }
    int threads = atoi(argv[1]);
    omp_set_num_threads(threads);

    init_matrix();
    
    printf("矩阵尺寸: %dx%d，线程数: %d\n", SIZE, SIZE, threads);
    
    serial_version();
    double serial_time = omp_get_wtime() - start;
    
    init_matrix();
    parallel_for_version();
    double parallel_for_time = omp_get_wtime() - start;
    
    init_matrix();
    distribute_version();
    double distribute_time = omp_get_wtime() - start;

    printf("\n加速比统计：\n");
    printf("Parallel For加速比: %.2fx\n", serial_time / parallel_for_time);
    printf("Distribute版本加速比: %.2fx\n", serial_time / distribute_time);
    
    return 0;
}
EOL

# 编译和运行（注：可能需要添加 -fopenmp-targets 选项）
echo "编译中..."
gcc -o omp_matrix omp_matrix.c -fopenmp -O3

echo "运行程序..."
./omp_matrix ${1:-16}  # 默认使用16线程，运行时可指定其他线程数