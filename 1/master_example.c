```bash
#!/bin/bash
export OMP_NUM_THREADS=${1:-16}
MATRIX_SIZE=2000

cat << EOF > omp_master_example.c
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N $MATRIX_SIZE

void init_matrix(double mat[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

void serial_matmul(double A[N][N], double B[N][N], double C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void parallel_for_matmul(double A[N][N], double B[N][N], double C[N][N]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void master_matmul(double A[N][N], double B[N][N], double C[N][N]) {
    #pragma omp parallel
    {
        #pragma omp master
        {
            for (int i = 0; i < N/4; i++) {
                for (int j = 0; j < N; j++) {
                    C[i][j] = 0;
                    for (int k = 0; k < N; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }

        #pragma omp for nowait
        for (int i = N/4; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] = 0;
                for (int k = 0; k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    static double A[N][N], B[N][N], C[N][N];
    srand(time(NULL));
    init_matrix(A);
    init_matrix(B);

    double start = omp_get_wtime();
    if (argc > 1) {
        if (argv[1][0] == '1') serial_matmul(A, B, C);
        else if (argv[1][0] == '2') parallel_for_matmul(A, B, C);
        else if (argv[1][0] == '3') master_matmul(A, B, C);
    }
    double duration = omp_get_wtime() - start;

    printf("版本%s运行时间：%.2f秒\n", argv[1], duration);
    return 0;
}
EOF

gcc -fopenmp omp_master_example.c -O3 -o omp_master_example

echo "====================="
time_serial=$(./omp_master_example 1 | awk '{print $2}')
echo "串行版本参考时间：$time_serial秒"

echo "---------------------"
time_para=$(./omp_master_example 2 | awk '{print $2}')
speedup_para=$(echo "scale=2; $time_serial/$time_para" | bc)
echo "Parallel for加速比：$speedup_para倍"

echo "---------------------"
time_master=$(./omp_master_example 3 | awk '{print $2}')
speedup_master=$(echo "scale=2; $time_serial/$time_master" | bc)
echo "Master版本加速比：$speedup_master倍"
echo "====================="

rm -f omp_master_example omp_master_example.c