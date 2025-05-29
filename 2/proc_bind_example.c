#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#define N 1024
#define ITERATIONS 10

void matrix_multiply_serial(double **A, double **B, double **C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrix_multiply_parallel_for(double **A, double **B, double **C) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrix_multiply_proc_bind_close(double **A, double **B, double **C) {
    #pragma omp parallel for proc_bind(close)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrix_multiply_proc_bind_spread(double **A, double **B, double **C) {
    #pragma omp parallel for proc_bind(spread)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrix_multiply_proc_bind_master(double **A, double **B, double **C) {
    #pragma omp parallel for proc_bind(master)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrix_multiply_proc_bind_close_blocked(double **A, double **B, double **C) {
    #define BLOCK_SIZE 64
    #pragma omp parallel proc_bind(close)
    {
        #pragma omp for
        for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                    for (int i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                        for (int j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
                            if (kk == 0) C[i][j] = 0.0;
                            for (int k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                                C[i][j] += A[i][k] * B[k][j];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ✅ 新增：proc_bind(close) + OMP_PLACES 控制版本
void matrix_multiply_proc_bind_close_places(double **A, double **B, double **C) {
    #pragma omp parallel for proc_bind(close)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

double** allocate_matrix(int n) {
    double **matrix = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double *)malloc(n * sizeof(double));
    }
    return matrix;
}

void initialize_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
}

void free_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int verify_matrices(double **C1, double **C2, int n) {
    double epsilon = 1e-6;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(C1[i][j] - C2[i][j]) > epsilon) {
                return 0;
            }
        }
    }
    return 1;
}

void clear_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        memset(matrix[i], 0, n * sizeof(double));
    }
}

int main(int argc, char **argv) {
    int num_threads = 16;

    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }

    omp_set_num_threads(num_threads);

    printf("使用线程数: %d\n", num_threads);

    double **A = allocate_matrix(N);
    double **B = allocate_matrix(N);
    double **C_serial = allocate_matrix(N);
    double **C_result = allocate_matrix(N);

    srand(42);
    initialize_matrix(A, N);
    initialize_matrix(B, N);

    double start_time, end_time;
    double serial_time, parallel_time, speedup;

    printf("矩阵大小: %d x %d\n", N, N);
    printf("重复次数: %d\n\n", ITERATIONS);

    // 串行版本
    printf("运行串行版本...\n");
    start_time = omp_get_wtime();
    for (int i = 0; i < ITERATIONS; i++) {
        matrix_multiply_serial(A, B, C_serial);
    }
    end_time = omp_get_wtime();
    serial_time = end_time - start_time;
    printf("串行版本运行时间: %.2f 秒\n", serial_time);
    printf("加速比: 1.00\n\n");

    struct {
        const char *name;
        void (*func)(double **, double **, double **);
    } versions[] = {
        {"parallel for版本", matrix_multiply_parallel_for},
        {"proc_bind(close)版本", matrix_multiply_proc_bind_close},
        {"proc_bind(spread)版本", matrix_multiply_proc_bind_spread},
        {"proc_bind(master)版本", matrix_multiply_proc_bind_master},
        {"proc_bind(close) + 分块优化版本", matrix_multiply_proc_bind_close_blocked},
        {"proc_bind(close) + OMP_PLACES版本", matrix_multiply_proc_bind_close_places},
    };

    for (int v = 0; v < sizeof(versions) / sizeof(versions[0]); v++) {
        printf("运行%s...\n", versions[v].name);
        if (strstr(versions[v].name, "OMP_PLACES"))
            printf("（请确保运行前设置: export OMP_PLACES=cores）\n");

        clear_matrix(C_result, N);
        start_time = omp_get_wtime();
        for (int i = 0; i < ITERATIONS; i++) {
            versions[v].func(A, B, C_result);
        }
        end_time = omp_get_wtime();
        parallel_time = end_time - start_time;
        speedup = serial_time / parallel_time;
        printf("%s运行时间: %.2f 秒\n", versions[v].name, parallel_time);
        printf("加速比: %.2f\n", speedup);
        if (!verify_matrices(C_serial, C_result, N)) {
            printf("警告: %s结果不正确!\n", versions[v].name);
        }
        printf("\n");
    }

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C_serial, N);
    free_matrix(C_result, N);

    return 0;
}




// 使用线程数: 16
// 矩阵大小: 1024 x 1024
// 重复次数: 10

// 运行串行版本...
// 串行版本运行时间: 21.05 秒
// 加速比: 1.00

// 运行parallel for版本...
// parallel for版本运行时间: 2.71 秒
// 加速比: 7.76

// 运行proc_bind(close)版本...
// proc_bind(close)版本运行时间: 3.87 秒
// 加速比: 5.44

// 运行proc_bind(spread)版本...
// proc_bind(spread)版本运行时间: 4.22 秒
// 加速比: 4.98

// 运行proc_bind(master)版本...
// proc_bind(master)版本运行时间: 3.62 秒
// 加速比: 5.82

// 运行proc_bind(close) + 分块优化版本...
// proc_bind(close) + 分块优化版本运行时间: 1.64 秒
// 加速比: 12.81

// 运行proc_bind(close) + OMP_PLACES版本...
// （请确保运行前设置: export OMP_PLACES=cores）
// proc_bind(close) + OMP_PLACES版本运行时间: 2.31 秒
// 加速比: 9.11

// 使用线程数: 64
// 矩阵大小: 1024 x 1024
// 重复次数: 10

// 运行串行版本...
// 串行版本运行时间: 16.80 秒
// 加速比: 1.00

// 运行parallel for版本...
// parallel for版本运行时间: 1.18 秒
// 加速比: 14.25

// 运行proc_bind(close)版本...
// proc_bind(close)版本运行时间: 1.17 秒
// 加速比: 14.37

// 运行proc_bind(spread)版本...
// proc_bind(spread)版本运行时间: 0.85 秒
// 加速比: 19.68

// 运行proc_bind(master)版本...
// proc_bind(master)版本运行时间: 0.71 秒
// 加速比: 23.56

// 运行proc_bind(close) + 分块优化版本...
// proc_bind(close) + 分块优化版本运行时间: 2.38 秒
// 加速比: 7.07

// 运行proc_bind(close) + OMP_PLACES版本...
// （请确保运行前设置: export OMP_PLACES=cores）
// proc_bind(close) + OMP_PLACES版本运行时间: 1.00 秒
// 加速比: 16.76

// 使用线程数: 128
// 矩阵大小: 1024 x 1024
// 重复次数: 10

// 运行串行版本...
// 串行版本运行时间: 15.82 秒
// 加速比: 1.00

// 运行parallel for版本...
// parallel for版本运行时间: 0.97 秒
// 加速比: 16.29

// 运行proc_bind(close)版本...
// proc_bind(close)版本运行时间: 1.01 秒
// 加速比: 15.60

// 运行proc_bind(spread)版本...
// proc_bind(spread)版本运行时间: 0.96 秒
// 加速比: 16.57

// 运行proc_bind(master)版本...
// proc_bind(master)版本运行时间: 0.83 秒
// 加速比: 19.07

// 运行proc_bind(close) + 分块优化版本...
// proc_bind(close) + 分块优化版本运行时间: 1.53 秒
// 加速比: 10.34

// 运行proc_bind(close) + OMP_PLACES版本...
// （请确保运行前设置: export OMP_PLACES=cores）
// proc_bind(close) + OMP_PLACES版本运行时间: 0.76 秒
// 加速比: 20.78