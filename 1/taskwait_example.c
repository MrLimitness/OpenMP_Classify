```bash
#!/bin/bash

cat > omp_matrix.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 2048  // 矩阵大小

void init_matrix(double **mat) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

int main() {
    double **A = (double **)malloc(N * sizeof(double *));
    double **B = (double **)malloc(N * sizeof(double *));
    double **C = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }

    srand(time(NULL));
    init_matrix(A);
    init_matrix(B);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
        }
    }

    double start = omp_get_wtime();

// 版本选择
#ifdef SERIAL
    // 串行版本
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
#elif defined PARALLEL_FOR
    // parallel for版本
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
#elif defined TASK_VERSION
    // taskwait版本
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < N; i++) {
                #pragma omp task
                {
                    for (int j = 0; j < N; j++) {
                        #pragma omp simd
                        for (int k = 0; k < N; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
            #pragma omp taskwait
        }
    }
#endif

    double end = omp_get_wtime();
    printf("版本运行时间: %.3f秒\n", end - start);

    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    return 0;
}
EOF

# 编译版本
gcc -DSERIAL -fopenmp -o serial omp_matrix.c
gcc -DPARALLEL_FOR -fopenmp -o parallel_for omp_matrix.c
gcc -DTASK_VERSION -fopenmp -o task_version omp_matrix.c

# 设置测试线程
THREADS="${THREADS:-16 64 144}"

# 运行测试
echo "------------------------------------------------"
serial_time=$(OMP_NUM_THREADS=1 ./serial | awk '{print $4}')
echo "串行版本运行时间: ${serial_time}秒"

for thread in $THREADS; do
    echo "------------------------------------------------"
    echo "测试线程数: $thread"
    
    # Parallel for版本
    pf_time=$(OMP_NUM_THREADS=$thread ./parallel_for | awk '{print $4}')
    pf_speedup=$(echo "scale=2; $serial_time / $pf_time" | bc)
    echo "Parallel for版本 时间: ${pf_time}秒 | 加速比: ${pf_speedup}"
    
    # Task版本
    task_time=$(OMP_NUM_THREADS=$thread ./task_version | awk '{print $4}')
    task_speedup=$(echo "scale=2; $serial_time / $task_time" | bc)
    echo "Taskwait版本    时间: ${task_time}秒 | 加速比: ${task_speedup}"
done

# 清理
rm -f serial parallel_for task_version omp_matrix.c