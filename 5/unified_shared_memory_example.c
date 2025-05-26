
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <time.h>

// 定义矩阵大小
#define N 5000

// 串行版本：大型3D网格扩散模拟
void diffusion_simulation_serial(double ***grid, double ***next_grid, int n, int iterations) {
    const double alpha = 0.1; // 扩散系数
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                for (int k = 1; k < n-1; k++) {
                    // 6邻域扩散计算
                    next_grid[i][j][k] = grid[i][j][k] + alpha * (
                        grid[i+1][j][k] + grid[i-1][j][k] +
                        grid[i][j+1][k] + grid[i][j-1][k] +
                        grid[i][j][k+1] + grid[i][j][k-1] - 
                        6.0 * grid[i][j][k]
                    );
                }
            }
        }
        
        // 交换网格
        double ***temp = grid;
        grid = next_grid;
        next_grid = temp;
    }
}

// 基本parallel for版本：大型3D网格扩散模拟
void diffusion_simulation_parallel(double ***grid, double ***next_grid, int n, int iterations) {
    const double alpha = 0.1; // 扩散系数
    
    for (int iter = 0; iter < iterations; iter++) {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                for (int k = 1; k < n-1; k++) {
                    // 6邻域扩散计算
                    next_grid[i][j][k] = grid[i][j][k] + alpha * (
                        grid[i+1][j][k] + grid[i-1][j][k] +
                        grid[i][j+1][k] + grid[i][j-1][k] +
                        grid[i][j][k+1] + grid[i][j][k-1] - 
                        6.0 * grid[i][j][k]
                    );
                }
            }
        }
        
        // 交换网格
        double ***temp = grid;
        grid = next_grid;
        next_grid = temp;
    }
}

// 版本1：使用target和unified_shared_memory的扩散模拟
void diffusion_simulation_usm_target(double ***grid, double ***next_grid, int n, int iterations) {
    const double alpha = 0.1; // 扩散系数
    const int n3 = n*n*n;
    
    // 使用一维数组替代三维数组，以适应GPU计算
    double *grid_flat = (double *)malloc(n3 * sizeof(double));
    double *next_grid_flat = (double *)malloc(n3 * sizeof(double));
    
    // 将三维数组展平为一维
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                grid_flat[i*n*n + j*n + k] = grid[i][j][k];
                next_grid_flat[i*n*n + j*n + k] = 0.0;
            }
        }
    }
    
    // 在统一共享内存中重新分配数组
    double *usm_grid, *usm_next_grid;
    
    #pragma omp allocate(usm_grid, usm_next_grid) allocator(omp_unified_shared_memory)
    {
        usm_grid = (double *)malloc(n3 * sizeof(double));
        usm_next_grid = (double *)malloc(n3 * sizeof(double));
    }
    
    // 复制数据到USM数组
    memcpy(usm_grid, grid_flat, n3 * sizeof(double));
    memcpy(usm_next_grid, next_grid_flat, n3 * sizeof(double));
    
    for (int iter = 0; iter < iterations; iter++) {
        // 使用target并行计算
        #pragma omp target teams distribute parallel for collapse(3) map(tofrom: usm_grid[0:n3], usm_next_grid[0:n3])
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                for (int k = 1; k < n-1; k++) {
                    int idx = i*n*n + j*n + k;
                    int idx_ip = (i+1)*n*n + j*n + k;
                    int idx_im = (i-1)*n*n + j*n + k;
                    int idx_jp = i*n*n + (j+1)*n + k;
                    int idx_jm = i*n*n + (j-1)*n + k;
                    int idx_kp = i*n*n + j*n + (k+1);
                    int idx_km = i*n*n + j*n + (k-1);
                    
                    usm_next_grid[idx] = usm_grid[idx] + alpha * (
                        usm_grid[idx_ip] + usm_grid[idx_im] +
                        usm_grid[idx_jp] + usm_grid[idx_jm] +
                        usm_grid[idx_kp] + usm_grid[idx_km] - 
                        6.0 * usm_grid[idx]
                    );
                }
            }
        }
        
        // 交换网格
        double *temp = usm_grid;
        usm_grid = usm_next_grid;
        usm_next_grid = temp;
    }
    
    // 将最终结果从一维数组转回三维数组
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                grid[i][j][k] = usm_grid[i*n*n + j*n + k];
            }
        }
    }
    
    // 释放内存
    free(grid_flat);
    free(next_grid_flat);
    free(usm_grid);
    free(usm_next_grid);
}

// 版本2：使用unified_shared_memory和数据分块优化的扩散模拟
void diffusion_simulation_usm_tiled(double ***grid, double ***next_grid, int n, int iterations) {
    const double alpha = 0.1; // 扩散系数
    const int n3 = n*n*n;
    const int block_size = 16; // 分块大小
    
    // 展平三维数组到一维
    double *grid_flat = (double *)malloc(n3 * sizeof(double));
    double *next_grid_flat = (double *)malloc(n3 * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                grid_flat[i*n*n + j*n + k] = grid[i][j][k];
                next_grid_flat[i*n*n + j*n + k] = 0.0;
            }
        }
    }
    
    // 在统一共享内存中分配数组
    double *usm_grid, *usm_next_grid;
    
    #pragma omp allocate(usm_grid, usm_next_grid) allocator(omp_unified_shared_memory)
    {
        usm_grid = (double *)malloc(n3 * sizeof(double));
        usm_next_grid = (double *)malloc(n3 * sizeof(double));
    }
    
    memcpy(usm_grid, grid_flat, n3 * sizeof(double));
    memcpy(usm_next_grid, next_grid_flat, n3 * sizeof(double));
    
    for (int iter = 0; iter < iterations; iter++) {
        // 使用分块策略进行并行计算
        #pragma omp target teams distribute parallel for collapse(3) map(tofrom: usm_grid[0:n3], usm_next_grid[0:n3])
        for (int ib = 1; ib < n-1; ib += block_size) {
            for (int jb = 1; jb < n-1; jb += block_size) {
                for (int kb = 1; kb < n-1; kb += block_size) {
                    // 分块内的计算
                    for (int i = ib; i < ib + block_size && i < n-1; i++) {
                        for (int j = jb; j < jb + block_size && j < n-1; j++) {
                            for (int k = kb; k < kb + block_size && k < n-1; k++) {
                                int idx = i*n*n + j*n + k;
                                int idx_ip = (i+1)*n*n + j*n + k;
                                int idx_im = (i-1)*n*n + j*n + k;
                                int idx_jp = i*n*n + (j+1)*n + k;
                                int idx_jm = i*n*n + (j-1)*n + k;
                                int idx_kp = i*n*n + j*n + (k+1);
                                int idx_km = i*n*n + j*n + (k-1);
                                
                                usm_next_grid[idx] = usm_grid[idx] + alpha * (
                                    usm_grid[idx_ip] + usm_grid[idx_im] +
                                    usm_grid[idx_jp] + usm_grid[idx_jm] +
                                    usm_grid[idx_kp] + usm_grid[idx_km] - 
                                    6.0 * usm_grid[idx]
                                );
                            }
                        }
                    }
                }
            }
        }
        
        // 交换网格
        double *temp = usm_grid;
        usm_grid = usm_next_grid;
        usm_next_grid = temp;
    }
    
    // 将最终结果从一维数组转回三维数组
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                grid[i][j][k] = usm_grid[i*n*n + j*n + k];
            }
        }
    }
    
    free(grid_flat);
    free(next_grid_flat);
    free(usm_grid);
    free(usm_next_grid);
}

// 版本3：使用unified_shared_memory和数据重组优化的扩散模拟
void diffusion_simulation_usm_reordered(double ***grid, double ***next_grid, int n, int iterations) {
    const double alpha = 0.1; // 扩散系数
    const int n3 = n*n*n;
    
    // 展平三维数组到一维，但重新排列数据以提高访问局部性
    double *grid_flat = (double *)malloc(n3 * sizeof(double));
    double *next_grid_flat = (double *)malloc(n3 * sizeof(double));
    
    // 使用空间填充曲线重排数据（这里简化为Z顺序）
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                // 计算Z顺序索引（简化版本）
                int z_idx = 0;
                for (int bit = 0; bit < 10; bit++) { // 假设n < 2^10
                    z_idx |= ((i >> bit) & 1) << (3*bit);
                    z_idx |= ((j >> bit) & 1) << (3*bit + 1);
                    z_idx |= ((k >> bit) & 1) << (3*bit + 2);
                }
                z_idx = z_idx % n3; // 确保在范围内
                
                grid_flat[z_idx] = grid[i][j][k];
                next_grid_flat[z_idx] = 0.0;
                
                // 存储反向映射，用于最终结果的转换
                if (i < n && j < n && k < n) {
                    // 这里简化处理，实际中需要维护更复杂的映射结构
                }
            }
        }
    }
    
    // 在统一共享内存中分配数组
    double *usm_grid, *usm_next_grid;
    
    #pragma omp allocate(usm_grid, usm_next_grid) allocator(omp_unified_shared_memory)
    {
        usm_grid = (double *)malloc(n3 * sizeof(double));
        usm_next_grid = (double *)malloc(n3 * sizeof(double));
    }
    
    memcpy(usm_grid, grid_flat, n3 * sizeof(double));
    memcpy(usm_next_grid, next_grid_flat, n3 * sizeof(double));
    
    // 为了简化实现，此处不使用Z顺序进行计算，而是恢复到线性布局进行计算
    // 实际应用中应当在Z顺序上直接进行计算
    double *temp_grid = (double *)malloc(n3 * sizeof(double));
    double *temp_next_grid = (double *)malloc(n3 * sizeof(double));
    
    // 从Z顺序恢复到线性顺序
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                int linear_idx = i*n*n + j*n + k;
                
                // 计算Z顺序索引（简化版本）
                int z_idx = 0;
                for (int bit = 0; bit < 10; bit++) {
                    z_idx |= ((i >> bit) & 1) << (3*bit);
                    z_idx |= ((j >> bit) & 1) << (3*bit + 1);
                    z_idx |= ((k >> bit) & 1) << (3*bit + 2);
                }
                z_idx = z_idx % n3;
                
                temp_grid[linear_idx] = usm_grid[z_idx];
                temp_next_grid[linear_idx] = usm_next_grid[z_idx];
            }
        }
    }
    
    memcpy(usm_grid, temp_grid, n3 * sizeof(double));
    memcpy(usm_next_grid, temp_next_grid, n3 * sizeof(double));
    
    for (int iter = 0; iter < iterations; iter++) {
        // 使用target并行计算
        #pragma omp target teams distribute parallel for collapse(3) map(tofrom: usm_grid[0:n3], usm_next_grid[0:n3])
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                for (int k = 1; k < n-1; k++) {
                    int idx = i*n*n + j*n + k;
                    int idx_ip = (i+1)*n*n + j*n + k;
                    int idx_im = (i-1)*n*n + j*n + k;
                    int idx_jp = i*n*n + (j+1)*n + k;
                    int idx_jm = i*n*n + (j-1)*n + k;
                    int idx_kp = i*n*n + j*n + (k+1);
                    int idx_km = i*n*n + j*n + (k-1);
                    
                    usm_next_grid[idx] = usm_grid[idx] + alpha * (
                        usm_grid[idx_ip] + usm_grid[idx_im] +
                        usm_grid[idx_jp] + usm_grid[idx_jm] +
                        usm_grid[idx_kp] + usm_grid[idx_km] - 
                        6.0 * usm_grid[idx]
                    );
                }
            }
        }
        
        // 交换网格
        double *temp = usm_grid;
        usm_grid = usm_next_grid;
        usm_next_grid = temp;
    }
    
    // 将最终结果从线性布局转回Z顺序
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                int linear_idx = i*n*n + j*n + k;
                
                // 计算Z顺序索引
                int z_idx = 0;
                for (int bit = 0; bit < 10; bit++) {
                    z_idx |= ((i >> bit) & 1) << (3*bit);
                    z_idx |= ((j >> bit) & 1) << (3*bit + 1);
                    z_idx |= ((k >> bit) & 1) << (3*bit + 2);
                }
                z_idx = z_idx % n3;
                
                temp_grid[z_idx] = usm_grid[linear_idx];
            }
        }
    }
    
    memcpy(usm_grid, temp_grid, n3 * sizeof(double));
    
    // 将Z顺序中的最终结果转回三维数组
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                // 计算Z顺序索引
                int z_idx = 0;
                for (int bit = 0; bit < 10; bit++) {
                    z_idx |= ((i >> bit) & 1) << (3*bit);
                    z_idx |= ((j >> bit) & 1) << (3*bit + 1);
                    z_idx |= ((k >> bit) & 1) << (3*bit + 2);
                }
                z_idx = z_idx % n3;
                
                grid[i][j][k] = usm_grid[z_idx];
            }
        }
    }
    
    // 释放内存
    free(grid_flat);
    free(next_grid_flat);
    free(temp_grid);
    free(temp_next_grid);
    free(usm_grid);
    free(usm_next_grid);
}

// 版本4: 使用unified_shared_memory和流水线优化的扩散模拟
void diffusion_simulation_usm_pipelined(double ***grid, double ***next_grid, int n, int iterations) {
    const double alpha = 0.1; // 扩散系数
    const int n3 = n*n*n;
    const int n_slices = 4; // 流水线切片数量
    const int slice_size = (n-2) / n_slices; // 每个切片的大小
    
    // 展平三维数组到一维
    double *grid_flat = (double *)malloc(n3 * sizeof(double));
    double *next_grid_flat = (double *)malloc(n3 * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                grid_flat[i*n*n + j*n + k] = grid[i][j][k];
                next_grid_flat[i*n*n + j*n + k] = 0.0;
            }
        }
    }
    
    // 在统一共享内存中分配数组
    double *usm_grid, *usm_next_grid;
    
    #pragma omp allocate(usm_grid, usm_next_grid) allocator(omp_unified_shared_memory)
    {
        usm_grid = (double *)malloc(n3 * sizeof(double));
        usm_next_grid = (double *)malloc(n3 * sizeof(double));
    }
    
    memcpy(usm_grid, grid_flat, n3 * sizeof(double));
    memcpy(usm_next_grid, next_grid_flat, n3 * sizeof(double));
    
    for (int iter = 0; iter < iterations; iter++) {
        // 使用流水线方式并行计算
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int slice = 0; slice < n_slices; slice++) {
                    int start_i = 1 + slice * slice_size;
                    int end_i = (slice == n_slices-1) ? n-1 : start_i + slice_size;
                    
                    #pragma omp task depend(in: usm_grid[0:n3]) depend(out: usm_next_grid[start_i*n*n:(end_i-start_i)*n*n])
                    {
                        #pragma omp target teams distribute parallel for collapse(3) map(tofrom: usm_grid[0:n3], usm_next_grid[start_i*n*n:(end_i-start_i)*n*n])
                        for (int i = start_i; i < end_i; i++) {
                            for (int j = 1; j < n-1; j++) {
                                for (int k = 1; k < n-1; k++) {
                                    int idx = i*n*n + j*n + k;
                                    int idx_ip = (i+1)*n*n + j*n + k;
                                    int idx_im = (i-1)*n*n + j*n + k;
                                    int idx_jp = i*n*n + (j+1)*n + k;
                                    int idx_jm = i*n*n + (j-1)*n + k;
                                    int idx_kp = i*n*n + j*n + (k+1);
                                    int idx_km = i*n*n + j*n + (k-1);
                                    
                                    usm_next_grid[idx] = usm_grid[idx] + alpha * (
                                        usm_grid[idx_ip] + usm_grid[idx_im] +
                                        usm_grid[idx_jp] + usm_grid[idx_jm] +
                                        usm_grid[idx_kp] + usm_grid[idx_km] - 
                                        6.0 * usm_grid[idx]
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 交换网格
        double *temp = usm_grid;
        usm_grid = usm_next_grid;
        usm_next_grid = temp;
    }
    
    // 将最终结果从一维数组转回三维数组
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                grid[i][j][k] = usm_grid[i*n*n + j*n + k];
            }
        }
    }
    
    // 释放内存
    free(grid_flat);
    free(next_grid_flat);
    free(usm_grid);
    free(usm_next_grid);
}

// 分配3D网格内存
double*** allocate_3d_grid(int n) {
    double ***grid = (double ***)malloc(n * sizeof(double **));
    for (int i = 0; i < n; i++) {
        grid[i] = (double **)malloc(n * sizeof(double *));
        for (int j = 0; j < n; j++) {
            grid[i][j] = (double *)malloc(n * sizeof(double));
        }
    }
    return grid;
}

// 释放3D网格内存
void free_3d_grid(double ***grid, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            free(grid[i][j]);
        }
        free(grid[i]);
    }
    free(grid);
}

// 初始化3D网格
void initialize_grid(double ***grid, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                // 中心热源
                if (i > n/3 && i < 2*n/3 && j > n/3 && j < 2*n/3 && k > n/3 && k < 2*n/3) {
                    grid[i][j][k] = 100.0;
                } else {
                    grid[i][j][k] = 0.0;
                }
            }
        }
    }
}

// 计算网格校验和
double compute_checksum(double ***grid, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                sum += grid[i][j][k];
            }
        }
    }
    return sum;
}

int main() {
    int num_threads = 16; // 可以修改为16/64/144等
    omp_set_num_threads(num_threads);
    
    // 使用较小的尺寸以减少测试时间，实际运行时可以设置为100以上
    const int n = 100;
    const int iterations = 10;
    
    printf("3D网格扩散模拟\n");
    printf("网格大小: %d x %d x %d\n", n, n, n);
    printf("迭代次数: %d\n", iterations);
    printf("运行线程数: %d\n\n", num_threads);
    
    // 分配内存
    double ***grid = allocate_3d_grid(n);
    double ***next_grid = allocate_3d_grid(n);
    
    // 用于存储每个版本的结果的副本
    double ***result_serial = allocate_3d_grid(n);
    double ***result_parallel = allocate_3d_grid(n);
    double ***result_usm_target = allocate_3d_grid(n);
    double ***result_usm_tiled = allocate_3d_grid(n);
    double ***result_usm_reordered = allocate_3d_grid(n);
    double ***result_usm_pipelined = allocate_3d_grid(n);
    
    // 初始化网格
    initialize_grid(grid, n);
    
    // 复制初始网格到结果副本
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                result_serial[i][j][k] = grid[i][j][k];
                result_parallel[i][j][k] = grid[i][j][k];
                result_usm_target[i][j][k] = grid[i][j][k];
                result_usm_tiled[i][j][k] = grid[i][j][k];
                result_usm_reordered[i][j][k] = grid[i][j][k];
                result_usm_pipelined[i][j][k] = grid[i][j][k];
                next_grid[i][j][k] = 0.0;
            }
        }
    }
    
    // 测试串行版本
    printf("运行串行版本...\n");
    double start_time = omp_get_wtime();
    diffusion_simulation_serial(result_serial, next_grid, n, iterations);
    double end_time = omp_get_wtime();
    double serial_time = end_time - start_time;
    printf("串行版本运行时间: %.4f 秒\n", serial_time);
    printf("串行版本加速比: 1.00\n\n");
    
    // 计算串行结果的校验和
    double serial_checksum = compute_checksum(result_serial, n);
    printf("串行版本结果校验和: %.6e\n\n", serial_checksum);
    
    // 测试基本parallel for版本
    printf("运行基本parallel for版本...\n");
    start_time = omp_get_wtime();
    diffusion_simulation_parallel(result_parallel, next_grid, n, iterations);
    end_time = omp_get_wtime();
    double parallel_time = end_time - start_time;
    printf("基本parallel for版本运行时间: %.4f 秒\n", parallel_time);
    printf("基本parallel for版本加速比: %.2f\n\n", serial_time / parallel_time);
    
    // 验证结果
    double parallel_checksum = compute_checksum(result_parallel, n);
    printf("基本parallel for版本结果校验和: %.6e\n", parallel_checksum);
    printf("基本parallel for版本结果误差: %.6e\n\n", fabs(parallel_checksum - serial_checksum));
    
    // 测试unified_shared_memory target版本
    printf("运行unified_shared_memory target版本...\n");
    start_time = omp_get_wtime();
    diffusion_simulation_usm_target(result_usm_target, next_grid, n, iterations);
    end_time = omp_get_wtime();
    double usm_target_time = end_time - start_time;
    printf("unified_shared_memory target版本运行时间: %.4f 秒\n", usm_target_time);
    printf("unified_shared_memory target版本加速比: %.2f\n\n", serial_time / usm_target_time);
    
    // 验证结果
    double usm_target_checksum = compute_checksum(result_usm_target, n);
    printf("unified_shared_memory target版本结果校验和: %.6e\n", usm_target_checksum);
    printf("unified_shared_memory target版本结果误差: %.6e\n\n", fabs(usm_target_checksum - serial_checksum));
    
    // 测试unified_shared_memory tiled版本
    printf("运行unified_shared_memory tiled版本...\n");
    start_time = omp_get_wtime();
    diffusion_simulation_usm_tiled(result_usm_tiled, next_grid, n, iterations);
    end_time = omp_get_wtime();
    double usm_tiled_time = end_time - start_time;
    printf("unified_shared_memory tiled版本运行时间: %.4f 秒\n", usm_tiled_time);
    printf("unified_shared_memory tiled版本加速比: %.2f\n\n", serial_time / usm_tiled_time);
    
    // 验证结果
    double usm_tiled_checksum = compute_checksum(result_usm_tiled, n);
    printf("unified_shared_memory tiled版本结果校验和: %.6e\n", usm_tiled_checksum);
    printf("unified_shared_memory tiled版本结果误差: %.6e\n\n", fabs(usm_tiled_checksum - serial_checksum));
    
    // 测试unified_shared_memory reordered版本
    printf("运行unified_shared_memory reordered版本...\n");
    start_time = omp_get_wtime();
    diffusion_simulation_usm_reordered(result_usm_reordered, next_grid, n, iterations);
    end_time = omp_get_wtime();
    double usm_reordered_time = end_time - start_time;
    printf("unified_shared_memory reordered版本运行时间: %.4f 秒\n", usm_reordered_time);
    printf("unified_shared_memory reordered版本加速比: %.2f\n\n", serial_time / usm_reordered_time);
    
    // 验证结果
    double usm_reordered_checksum = compute_checksum(result_usm_reordered, n);
    printf("unified_shared_memory reordered版本结果校验和: %.6e\n", usm_reordered_checksum);
    printf("unified_shared_memory reordered版本结果误差: %.6e\n\n", fabs(usm_reordered_checksum - serial_checksum));
    
    // 测试unified_shared_memory pipelined版本
    printf("运行unified_shared_memory pipelined版本...\n");
    start_time = omp_get_wtime();
    diffusion_simulation_usm_pipelined(result_usm_pipelined, next_grid, n, iterations);
    end_time = omp_get_wtime();
    double usm_pipelined_time = end_time - start_time;
    printf("unified_shared_memory pipelined版本运行时间: %.4f 秒\n", usm_pipelined_time);
    printf("unified_shared_memory pipelined版本加速比: %.2f\n\n", serial_time / usm_pipelined_time);
    
    // 验证结果
    double usm_pipelined_checksum = compute_checksum(result_usm_pipelined, n);
    printf("unified_shared_memory pipelined版本结果校验和: %.6e\n", usm_pipelined_checksum);
    printf("unified_shared_memory pipelined版本结果误差: %.6e\n\n", fabs(usm_pipelined_checksum - serial_checksum));
    
    // 性能总结
    printf("======= 性能总结 =======\n");
    printf("串行版本:                            %.4f 秒 (加速比: 1.00)\n", serial_time);
    printf("基本parallel for版本:                %.4f 秒 (加速比: %.2f)\n", parallel_time, serial_time / parallel_time);
    printf("unified_shared_memory target版本:    %.4f 秒 (加速比: %.2f)\n", usm_target_time, serial_time / usm_target_time);
    printf("unified_shared_memory tiled版本:     %.4f 秒 (加速比: %.2f)\n", usm_tiled_time, serial_time / usm_tiled_time);
    printf("unified_shared_memory reordered版本: %.4f 秒 (加速比: %.2f)\n", usm_reordered_time, serial_time / usm_reordered_time);
    printf("unified_shared_memory pipelined版本: %.4f 秒 (加速比: %.2f)\n", usm_pipelined_time, serial_time / usm_pipelined_time);
    
    // 释放内存
    free_3d_grid(grid, n);
    free_3d_grid(next_grid, n);
    free_3d_grid(result_serial, n);
    free_3d_grid(result_parallel, n);
    free_3d_grid(result_usm_target, n);
    free_3d_grid(result_usm_tiled, n);
    free_3d_grid(result_usm_reordered, n);
    free_3d_grid(result_usm_pipelined, n);
    
    return 0;
}
