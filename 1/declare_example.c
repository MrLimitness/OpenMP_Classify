```bash
#!/bin/bash

if [ $# -eq 0 ]; then
    echo "请指定线程数，例如：$0 16"
    exit 1
fi
THREADS=$1
export OMP_NUM_THREADS=$THREADS

cat <<EOF >matrix_omp.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define N 2000

double A[N][N];
double B[N][N];
double C[N][N];

void init_matrix() {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand()/RAND_MAX;
            B[i][j] = (double)rand()/RAND_MAX;
        }
    }
}

void serial_version() {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void parallel_for_version() {
    memset(C, 0, sizeof(C));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

#pragma omp declare reduction(vec_add:double*:omp_out+=omp_in)
static double tmp;

void declare_version() {
    memset(C, 0, sizeof(C));
    #pragma omp parallel reduction(vec_add:C[:N][:N]) private(tmp)
    {
        #pragma omp for collapse(2)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                tmp = 0.0;
                for (int k = 0; k < N; k++)
                    tmp += A[i][k] * B[k][j];
                C[i][j] += tmp;
            }
    }
}

void declare_schedule_version() {
    memset(C, 0, sizeof(C));
    #pragma omp parallel for collapse(2) schedule(dynamic, 10) reduction(+:C[:N][:N])
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("请指定版本：serial/parallel_for/declare/declare_schedule\n");
        return 1;
    }

    srand(42);
    init_matrix();

    double st = omp_get_wtime();
    if (strcmp(argv[1], "serial") == 0) {
        serial_version();
    } else if (strcmp(argv[1], "parallel_for") == 0) {
        parallel_for_version();
    } else if (strcmp(argv[1], "declare") == 0) {
        declare_version();
    } else if (strcmp(argv[1], "declare_schedule") == 0) {
        declare_schedule_version();
    } else {
        printf("无效版本\n");
        return 1;
    }
    printf("%s版本运行时间：%.2f秒\n", argv[1], omp_get_wtime()-st);
    return 0;
}
EOF

gcc -fopenmp -O3 matrix_omp.c -o matrix_omp

echo "-------------------------"
echo "线程数：$THREADS"

declare -A TIMES
for ver in serial parallel_for declare declare_schedule; do
    TIMES[$ver]=$(./matrix_omp $ver | awk '{print $4}')
done

base_time=${TIMES[serial]}
echo "串行基准时间：${base_time}秒"

for ver in parallel_for declare declare_schedule; do
    speedup=$(echo "scale=2; $base_time / ${TIMES[$ver]}" | bc)
    echo "$ver版本加速比：${speedup}x"
done