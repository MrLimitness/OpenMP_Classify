#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 2000
#define BLOCK_SIZE 64

// 初始化矩阵
void init_matrix(float *A, float *B, float *C, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i*n+j] = (float)rand() / RAND_MAX;
            B[i*n+j] = (float)rand() / RAND_MAX;
            C[i*n+j] = 0.0f;
        }
    }
}

// 计算矩阵元素和
float compute_sum(float *C, int n) {
    float sum = 0.0f;
    #pragma omp parallel for collapse(2) reduction(+:sum)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum += C[i*n+j];
        }
    }
    return sum;
}

void serial_version(float *A, float *B, float *C) {
    double start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }
    }
    double end = omp_get_wtime();
    float sum = compute_sum(C, N);
    printf("[Serial] Time: %.2f s, Sum: %f\n", end - start, sum);
}

void parallel_version(float *A, float *B, float *C, int num_threads) {
    double start = omp_get_wtime();
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }
    }
    double end = omp_get_wtime();
    float sum = compute_sum(C, N);
    printf("[Parallel] Threads: %d, Time: %.2f s, Sum: %f\n", num_threads, end - start, sum);
}

void usm_target_version(float *A, float *B, float *C) {
    double start = omp_get_wtime();
    #pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(tofrom: C[0:N*N])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[i*N+k] * B[k*N+j];
                }
                C[i*N+j] = sum;
            }
        }
    }
    double end = omp_get_wtime();
    float sum = compute_sum(C, N);
    printf("[USM Target] Time: %.2f s, Sum: %f\n", end - start, sum);
}

void usm_blocked_version(float *A, float *B, float *C) {
    double start = omp_get_wtime();
    #pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(tofrom: C[0:N*N])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; i += BLOCK_SIZE) {
            for (int j = 0; j < N; j += BLOCK_SIZE) {
                for (int k = 0; k < N; k += BLOCK_SIZE) {
                    for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii++) {
                        for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj++) {
                            float sum = 0.0f;
                            for (int kk = k; kk < k + BLOCK_SIZE && kk < N; kk++) {
                                sum += A[ii*N+kk] * B[kk*N+jj];
                            }
                            #pragma omp atomic
                            C[ii*N+jj] += sum;
                        }
                    }
                }
            }
        }
    }
    double end = omp_get_wtime();
    float sum = compute_sum(C, N);
    printf("[USM Blocked] Time: %.2f s, Sum: %f\n", end - start, sum);
}

int main(int argc, char *argv[]) {
    int num_threads = (argc > 1) ? atoi(argv[1]) : 16;
    printf("Matrix size: %d x %d, Threads: %d\n", N, N, num_threads);

    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));

    srand(42);
    init_matrix(A, B, C, N);
    serial_version(A, B, C);

    init_matrix(A, B, C, N);
    parallel_version(A, B, C, num_threads);

    int offload_available = 0;
    #pragma omp target map(from:offload_available)
    {
        offload_available = 1;
    }

    if (offload_available) {
        init_matrix(A, B, C, N);
        usm_target_version(A, B, C);

        init_matrix(A, B, C, N);
        usm_blocked_version(A, B, C);
    } else {
        printf("OpenMP target offloading not available.\n");
    }

    free(A);
    free(B);
    free(C);

    return 0;
}