```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 2048
double A[N][N], B[N][N], C[N][N];

void initialize() {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }
    }
}

void serial_matmul() {
    double start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end = omp_get_wtime();
    printf("串行版本运行时间: %.2f秒\n", end - start);
}

void parallel_for_matmul() {
    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double temp = 0.0;
            for (int k = 0; k < N; k++) {
                temp += A[i][k] * B[k][j];
            }
            C[i][j] = temp;
        }
    }
    double end = omp_get_wtime();
    printf("Parallel for版本运行时间: %.2f秒 (加速比: %.2fx)\n", 
          end - start, (global_serial_time)/(end - start));
}

void depobj_matmul_v3() {
    double start = omp_get_wtime();
    omp_depend_t depobj[N];
    for (int i = 0; i < N; i++)
        omp_init_depend(&depobj[i]);

    #pragma omp parallel
    #pragma omp single
    for (int i = 0; i < N; i++) {
        #pragma omp task depend(out: depobj[i]) firstprivate(i)
        {
            for (int j = 0; j < N; j++) {
                double temp = 0.0;
                for (int k = 0; k < N; k++) {
                    temp += A[i][k] * B[k][j];
                }
                C[i][j] = temp;
            }
        }
    }
    double end = omp_get_wtime();
    printf("Depobj版本3运行时间: %.2f秒 (加速比: %.2fx)\n", 
          end - start, (global_serial_time)/(end - start));
}

void depobj_matmul_v4() {
    double start = omp_get_wtime();
    omp_depend_t depobj[N][N];
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            omp_init_depend(&depobj[i][j]);

    #pragma omp parallel
    #pragma omp single
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            #pragma omp task depend(out: depobj[i][j]) firstprivate(i, j)
            {
                double temp = 0.0;
                for (int k = 0; k < N; k++) {
                    temp += A[i][k] * B[k][j];
                }
                C[i][j] = temp;
            }
        }
    }
    double end = omp_get_wtime();
    printf("Depobj版本4运行时间: %.2f秒 (加速比: %.2fx)\n", 
          end - start, (global_serial_time)/(end - start));
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("使用方法: %s <线程数>\n", argv[0]);
        return 1;
    }
    omp_set_num_threads(atoi(argv[1]));
    
    initialize();
    double t1 = omp_get_wtime();
    serial_matmul();
    double t2 = omp_get_wtime();
    global_serial_time = t2 - t1;
    
    initialize();
    parallel_for_matmul();
    
    initialize();
    depobj_matmul_v3();
    
    initialize();
    depobj_matmul_v4();

    return 0;
}