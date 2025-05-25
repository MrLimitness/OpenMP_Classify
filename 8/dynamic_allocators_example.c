#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 2000

double** allocate_matrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void init_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = ((double)rand() / RAND_MAX) * 10.0;
        }
    }
}

// 串行版本
void matrix_multiply_serial(double** A, double** B, double** C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// parallel for 版本
void matrix_multiply_parallel_for(double** A, double** B, double** C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// dynamic_allocators 版本1 - 使用基本的dynamic调度
void matrix_multiply_dynamic(double** A, double** B, double** C, int n) {
    #pragma omp parallel for schedule(dynamic, 10)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// dynamic_allocators 版本2 - 使用dynamic调度与嵌套并行化
void matrix_multiply_dynamic_nested(double** A, double** B, double** C, int n) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop grainsize(10)
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    C[i][j] = 0.0;
                    for (int k = 0; k < n; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }
    }
}

// dynamic_allocators 版本3 - 使用dynamic和collapse
void matrix_multiply_dynamic_collapse(double** A, double** B, double** C, int n) {
    #pragma omp parallel for collapse(2) schedule(dynamic, 5)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// dynamic_allocators 版本4 - 任务块并行
void matrix_multiply_dynamic_chunked(double** A, double** B, double** C, int n) {
    int chunk_size = 50;
    
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1)
        for (int chunk_i = 0; chunk_i < n; chunk_i += chunk_size) {
            int end_i = chunk_i + chunk_size < n ? chunk_i + chunk_size : n;
            for (int chunk_j = 0; chunk_j < n; chunk_j += chunk_size) {
                int end_j = chunk_j + chunk_size < n ? chunk_j + chunk_size : n;
                
                for (int i = chunk_i; i < end_i; i++) {
                    for (int j = chunk_j; j < end_j; j++) {
                        C[i][j] = 0.0;
                        for (int k = 0; k < n; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int num_threads = 64; // 默认线程数
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);
    
    srand(time(NULL));
    
    double** A = allocate_matrix(N);
    double** B = allocate_matrix(N);
    double** C = allocate_matrix(N);
    
    init_matrix(A, N);
    init_matrix(B, N);
    
    double start_time, end_time;
    
    // 串行版本
    start_time = omp_get_wtime();
    matrix_multiply_serial(A, B, C, N);
    end_time = omp_get_wtime();
    double serial_time = end_time - start_time;
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    printf("串行版本加速比: %.2f\n", 1.0);
    
    // parallel for 版本
    start_time = omp_get_wtime();
    matrix_multiply_parallel_for(A, B, C, N);
    end_time = omp_get_wtime();
    double parallel_for_time = end_time - start_time;
    printf("parallel for 版本运行时间: %.4f 秒\n", parallel_for_time);
    printf("parallel for 版本加速比: %.2f\n", serial_time / parallel_for_time);
    
    // dynamic_allocators 版本1
    start_time = omp_get_wtime();
    matrix_multiply_dynamic(A, B, C, N);
    end_time = omp_get_wtime();
    double dynamic_time = end_time - start_time;
    printf("dynamic 版本运行时间: %.4f 秒\n", dynamic_time);
    printf("dynamic 版本加速比: %.2f\n", serial_time / dynamic_time);
    
    // dynamic_allocators 版本2
    start_time = omp_get_wtime();
    matrix_multiply_dynamic_nested(A, B, C, N);
    end_time = omp_get_wtime();
    double dynamic_nested_time = end_time - start_time;
    printf("dynamic nested 版本运行时间: %.4f 秒\n", dynamic_nested_time);
    printf("dynamic nested 版本加速比: %.2f\n", serial_time / dynamic_nested_time);
    
    // dynamic_allocators 版本3
    start_time = omp_get_wtime();
    matrix_multiply_dynamic_collapse(A, B, C, N);
    end_time = omp_get_wtime();
    double dynamic_collapse_time = end_time - start_time;
    printf("dynamic collapse 版本运行时间: %.4f 秒\n", dynamic_collapse_time);
    printf("dynamic collapse 版本加速比: %.2f\n", serial_time / dynamic_collapse_time);
    
    // dynamic_allocators 版本4
    start_time = omp_get_wtime();
    matrix_multiply_dynamic_chunked(A, B, C, N);
    end_time = omp_get_wtime();
    double dynamic_chunked_time = end_time - start_time;
    printf("dynamic chunked 版本运行时间: %.4f 秒\n", dynamic_chunked_time);
    printf("dynamic chunked 版本加速比: %.2f\n", serial_time / dynamic_chunked_time);
    
    printf("使用的线程数: %d\n", num_threads);
    
    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);
    
    return 0;
}


// 串行版本运行时间: 34.9637 秒
// 串行版本加速比: 1.00
// parallel for 版本运行时间: 3.0294 秒
// parallel for 版本加速比: 11.54
// dynamic 版本运行时间: 2.3531 秒
// dynamic 版本加速比: 14.86
// dynamic nested 版本运行时间: 2.9338 秒
// dynamic nested 版本加速比: 11.92
// dynamic collapse 版本运行时间: 2.1910 秒
// dynamic collapse 版本加速比: 15.96
// dynamic chunked 版本运行时间: 1.5619 秒
// dynamic chunked 版本加速比: 22.38
// 使用的线程数: 16


// 串行版本运行时间: 53.3455 秒
// 串行版本加速比: 1.00
// parallel for 版本运行时间: 1.5348 秒
// parallel for 版本加速比: 34.76
// dynamic 版本运行时间: 1.1559 秒
// dynamic 版本加速比: 46.15
// dynamic nested 版本运行时间: 1.0394 秒
// dynamic nested 版本加速比: 51.32
// dynamic collapse 版本运行时间: 1.1093 秒
// dynamic collapse 版本加速比: 48.09
// dynamic chunked 版本运行时间: 1.7959 秒
// dynamic chunked 版本加速比: 29.70
// 使用的线程数: 64