#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N 4000

double **allocate_matrix(int n) {
    double **mat = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        mat[i] = (double *)malloc(n * sizeof(double));
    }
    return mat;
}

void free_matrix(double **mat, int n) {
    for (int i = 0; i < n; i++) {
        free(mat[i]);
    }
    free(mat);
}

void initialize_matrices(double **A, double **B, double **C, int n) {
    srand(12345);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }
    }
}

void reset_matrix(double **C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0.0;
}

double checksum(double **C, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            sum += C[i][j];
    return sum;
}

// 所有版本的矩阵乘法函数定义不变，略...

// 串行版本
void matrix_multiply_serial(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

// 其他并行版本（parallel_for、loop、loop_collapse、teams_loop、teams_loop_collapse）函数略写，和你代码保持一致即可

void test_version(const char *name, void (*matmul_func)(double **, double **, double **, int),
                  double **A, double **B, double **C, double serial_checksum, double serial_time) {
    reset_matrix(C, N);
    printf("运行%-40s...\n", name);
    double start = omp_get_wtime();
    matmul_func(A, B, C, N);
    double end = omp_get_wtime();
    double time = end - start;
    double check = checksum(C, N);
    double speedup = serial_time / time;
    double error = fabs(check - serial_checksum);
    printf("%-42s运行时间: %.4f 秒\n", name, time);
    printf("%-42s加速比:   %.2f\n", "", speedup);
    printf("%-42s校验和:   %.6e\n", "", check);
    printf("%-42s误差:     %.6e\n\n", "", error);
}

int main() {
    int num_threads = 16;
    omp_set_num_threads(num_threads);

    printf("矩阵大小: %d x %d\n", N, N);
    printf("线程数: %d\n\n", num_threads);

    // 分配内存和初始化
    double **A = allocate_matrix(N);
    double **B = allocate_matrix(N);
    double **C = allocate_matrix(N);
    initialize_matrices(A, B, C, N);

    // 串行版本作为基准
    printf("运行串行版本...\n");
    double start = omp_get_wtime();
    matrix_multiply_serial(A, B, C, N);
    double end = omp_get_wtime();
    double serial_time = end - start;
    double serial_sum = checksum(C, N);
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    printf("串行版本校验和:   %.6e\n\n", serial_sum);

    // 测试各个并行版本
    test_version("parallel for版本", matrix_multiply_parallel_for, A, B, C, serial_sum, serial_time);
    test_version("loop版本", matrix_multiply_loop, A, B, C, serial_sum, serial_time);
    test_version("loop+collapse版本", matrix_multiply_loop_collapse, A, B, C, serial_sum, serial_time);
    test_version("teams distribute parallel loop版本", matrix_multiply_teams_loop, A, B, C, serial_sum, serial_time);
    test_version("teams distribute parallel loop collapse版本", matrix_multiply_teams_loop_collapse, A, B, C, serial_sum, serial_time);

    // 总结
    printf("======= 性能测试完成 =======\n");

    // 清理内存
    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
