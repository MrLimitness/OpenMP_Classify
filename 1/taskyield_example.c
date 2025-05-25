```bash
#!/bin/bash

# 设置线程数（可传入多个参数，如 ./script.sh 16 64 144）
threads="16 64 144"
if [ "$#" -ne 0 ]; then
    threads=$@
fi

# 矩阵大小设置（运行时间约10秒）
N=2000

cat > matrix_omp.c <<EOF
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N $N
double A[N][N];
double B[N][N];
double C[N][N];

void init_matrix() {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }
    }
}

void serial() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void parallel_for() {
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

void taskyield_impl() {
    #pragma omp parallel
    {
        #pragma omp master
        {
            #pragma omp taskgroup
            {
                for (int i = 0; i < N; ++i) {
                    #pragma omp task firstprivate(i) shared(A, B, C)
                    {
                        for (int j = 0; j < N; ++j) {
                            double sum = 0;
                            for (int k = 0; k < N; ++k) {
                                sum += A[i][k] * B[k][j];
                                if (k % 50 == 0) {
                                    #pragma omp taskyield
                                }
                            }
                            #pragma omp atomic
                            C[i][j] += sum;
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    init_matrix();
    double start = omp_get_wtime();

    if (argc > 1) {
        if (argv[1][0] == 's') serial();
        else if (argv[1][0] == 'p') parallel_for();
        else if (argv[1][0] == 't') taskyield_impl();
    }

    double end = omp_get_wtime();
    printf("运行时间: %.2f秒\n", end - start);
    return 0;
}
EOF

# 编译程序
gcc -fopenmp matrix_omp.c -o matrix_omp

# 运行测试
echo "-----------------------------"
echo "串行版本基准测试:"
serial_time=$(OMP_NUM_THREADS=1 ./matrix_omp s | grep -oP '运行时间: \K\d+\.\d+')
echo "串行时间: ${serial_time}秒"

for t in $threads; do
    echo "-----------------------------"
    echo "当前线程数: $t"
    export OMP_NUM_THREADS=$t
    
    # parallel for版本
    pf_time=$(./matrix_omp p | grep -oP '运行时间: \K\d+\.\d+')
    pf_speedup=$(echo "scale=2; $serial_time / $pf_time" | bc)
    echo "parallel_for版本加速比: $pf_speedup"

    # taskyield版本
    ty_time=$(./matrix_omp t | grep -oP '运行时间: \K\d+\.\d+')
    ty_speedup=$(echo "scale=2; $serial_time / $ty_time" | bc)
    echo "taskyield版本加速比: $ty_speedup"
done

echo "-----------------------------"
rm matrix_omp matrix_omp.c