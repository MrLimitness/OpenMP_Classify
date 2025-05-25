```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 2048

double **A, **B, **C;

void initialize_matrices() {
    A = (double **)malloc(N * sizeof(double *));
    B = (double **)malloc(N * sizeof(double *));
    C = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand()/RAND_MAX;
            B[i][j] = (double)rand()/RAND_MAX;
            C[i][j] = 0.0;
        }
    }
}

void free_matrices() {
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
}

void reset_C() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = 0.0;
}

void serial_matmul() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void parallel_for_matmul() {
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void loop_matmul_v3() {
    #pragma omp parallel
    #pragma omp loop
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void loop_matmul_v4() {
    #pragma omp parallel
    #pragma omp loop collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void loop_matmul_v5() {
    #pragma omp parallel
    #pragma omp loop schedule(dynamic)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void loop_matmul_v6() {
    #pragma omp parallel
    #pragma omp loop schedule(guided) nowait
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("请传入线程数作为参数！示例：%s 16\n", argv[0]);
        return 1;
    }
    omp_set_num_threads(atoi(argv[1]));
    initialize_matrices();
    
    double start, t_serial, t_pfor, t_v3, t_v4, t_v5, t_v6;
    
    reset_C();
    start = omp_get_wtime();
    serial_matmul();
    t_serial = omp_get_wtime() - start;
    printf("串行版本运行时间：%.2f秒\n", t_serial);
    
    reset_C();
    start = omp_get_wtime();
    parallel_for_matmul();
    t_pfor = omp_get_wtime() - start;
    printf("Parallel for版本：%.2f秒（加速比：%.2fx）\n", t_pfor, t_serial/t_pfor);
    
    reset_C();
    start = omp_get_wtime();
    loop_matmul_v3();
    t_v3 = omp_get_wtime() - start;
    printf("Loop版本v3：%.2f秒（加速比：%.2fx）\n", t_v3, t_serial/t_v3);
    
    reset_C();
    start = omp_get_wtime();
    loop_matmul_v4();
    t_v4 = omp_get_wtime() - start;
    printf("Loop版本v4（collapse）：%.2f秒（加速比：%.2fx）\n", t_v4, t_serial/t_v4);
    
    reset_C();
    start = omp_get_wtime();
    loop_matmul_v5();
    t_v5 = omp_get_wtime() - start;
    printf("Loop版本v5（dynamic）：%.2f秒（加速比：%.2fx）\n", t_v5, t_serial/t_v5);
    
    reset_C();
    start = omp_get_wtime();
    loop_matmul_v6();
    t_v6 = omp_get_wtime() - start;
    printf("Loop版本v6（guided）：%.2f秒（加速比：%.2fx）\n", t_v6, t_serial/t_v6);
    
    free_matrices();
    return 0;
}