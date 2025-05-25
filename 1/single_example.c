```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1024

void init_matrix(double **matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

void serial_matrix_mult(double **A, double **B, double **C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void parallel_for_matrix_mult(double **A, double **B, double **C) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void single_task_matrix_mult(double **A, double **B, double **C) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < N; i++) {
                #pragma omp task firstprivate(i)
                {
                    for (int j = 0; j < N; j++) {
                        C[i][j] = 0;
                        for (int k = 0; k < N; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

void single_sections_matrix_mult(double **A, double **B, double **C) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp sections
            {
                #pragma omp section
                { 
                    for (int i = 0; i < N/2; i++) {
                        for (int j = 0; j < N; j++) {
                            C[i][j] = 0;
                            for (int k = 0; k < N; k++) {
                                C[i][j] += A[i][k] * B[k][j];
                            }
                        }
                    }
                }
                #pragma omp section
                { 
                    for (int i = N/2; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            C[i][j] = 0;
                            for (int k = 0; k < N; k++) {
                                C[i][j] += A[i][k] * B[k][j];
                            }
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("请输入线程数作为参数！\n");
        return 1;
    }
    int threads = atoi(argv[1]);
    omp_set_num_threads(threads);

    double **A = (double **)malloc(N * sizeof(double *));
    double **B = (double **)malloc(N * sizeof(double *));
    double **C_serial = (double **)malloc(N * sizeof(double *));
    double **C_parallel_for = (double **)malloc(N * sizeof(double *));
    double **C_single_task = (double **)malloc(N * sizeof(double *));
    double **C_single_sections = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C_serial[i] = (double *)malloc(N * sizeof(double));
        C_parallel_for[i] = (double *)malloc(N * sizeof(double));
        C_single_task[i] = (double *)malloc(N * sizeof(double));
        C_single_sections[i] = (double *)malloc(N * sizeof(double));
    }

    init_matrix(A);
    init_matrix(B);

    double start, end;

    start = omp_get_wtime();
    serial_matrix_mult(A, B, C_serial);
    end = omp_get_wtime();
    double serial_time = end - start;
    printf("串行版本运行时间: %.3f秒\n", serial_time);

    start = omp_get_wtime();
    parallel_for_matrix_mult(A, B, C_parallel_for);
    end = omp_get_wtime();
    double parallel_for_time = end - start;
    printf("Parallel for版本运行时间: %.3f秒 (加速比: %.2fx)\n", parallel_for_time, serial_time / parallel_for_time);

    start = omp_get_wtime();
    single_task_matrix_mult(A, B, C_single_task);
    end = omp_get_wtime();
    double single_task_time = end - start;
    printf("Single+Task版本运行时间: %.3f秒 (加速比: %.2fx)\n", single_task_time, serial_time / single_task_time);

    start = omp_get_wtime();
    single_sections_matrix_mult(A, B, C_single_sections);
    end = omp_get_wtime();
    double single_sections_time = end - start;
    printf("Single+Sections版本运行时间: %.3f秒 (加速比: %.2fx)\n", single_sections_time, serial_time / single_sections_time);

    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C_serial[i]);
        free(C_parallel_for[i]);
        free(C_single_task[i]);
        free(C_single_sections[i]);
    }
    free(A);
    free(B);
    free(C_serial);
    free(C_parallel_for);
    free(C_single_task);
    free(C_single_sections);

    return 0;
}