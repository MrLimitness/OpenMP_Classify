```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 1024

void init_matrix(double *mat) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i*N+j] = (double)rand()/RAND_MAX;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("请指定线程数！\n");
        return 1;
    }
    int threads = atoi(argv[1]);
    omp_set_num_threads(threads);
    
    double *A = malloc(N*N*sizeof(double));
    double *B = malloc(N*N*sizeof(double));
    double *C = malloc(N*N*sizeof(double));
    init_matrix(A); init_matrix(B);

    // 串行版本
    double t1 = omp_get_wtime();
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
    double t2 = omp_get_wtime();
    printf("串行版本运行时间: %.3f秒\n", t2-t1);

    // 重置结果矩阵
    for (int i = 0; i < N; i++) 
        for (int j = 0; j < N; j++) 
            C[i*N+j] = 0.0;

    // Parallel for版本
    t1 = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
    t2 = omp_get_wtime();
    printf("Parallel for版本运行时间: %.3f秒\t加速比: %.2f\n", t2-t1, (t2-t1)/(t2-t1));

    // 重置结果矩阵
    for (int i = 0; i < N; i++) 
        for (int j = 0; j < N; j++) 
            C[i*N+j] = 0.0;

    // 使用flush的优化版本
    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        int chunk = N/omp_get_num_threads();
        int start = omp_get_thread_num()*chunk;
        int end = (omp_get_thread_num() == omp_get_num_threads()-1) ? N : start+chunk;
        
        for (int i = start; i < end; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A[i*N+k] * B[k*N+j];
                }
                #pragma omp atomic
                C[i*N+j] += sum;
                #pragma omp flush(C)  // 显式刷新结果矩阵
            }
        }
    }
    t2 = omp_get_wtime();
    printf("Flush优化版本运行时间: %.3f秒\t加速比: %.2f\n", t2-t1, (t2-t1)/(t2-t1));

    free(A); free(B); free(C);
    return 0;
}