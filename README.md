# OpenMP官方示例仓库中文改版

## 环境准备
- GCC 15版本（可通过conda安装）

## 目录
- [指令类](#指令类)
- [子句类](#子句类)
- [内存管理相关](#内存管理相关)
- [任务依赖相关](#任务依赖相关)
- [设备相关扩展](#设备相关扩展)
- [同步原语](#同步原语)
- [运行时函数特性](#运行时函数特性)
- [指令修饰符](#指令修饰符)
- [特殊上下文限定](#特殊上下文限定)
### 原语介绍类
**目标**：理论学习、复现代码、应用实践（对比普通`parallel for`的优势）

## 指令类   
## simd
- **作用**：指示编译器对后续循环进行矢量化，将多次标量操作合并为一次 SIMD 指令。
- **场景**：  
  - 数值计算中对大规模数组或向量执行相同运算（如加减乘除）时。  
  - 循环体内无数据依赖或依赖可以通过 `safelen`/`aligned` 等方式告知编译器时。  
  - 希望在 CPU 支持 AVX、SSE 等指令集时，通过显式 `#pragma omp simd` 强制矢量化。  
- **性能提升**：  
  1. 充分利用向量寄存器一次处理多组数据，单指令吞吐量成倍提升。  
  2. 避免标量循环中每次只处理一个元素造成的流水线浪费，减少循环开销。  
  3. 结合内存对齐提示（`aligned`）可减少访存未对齐带来的性能损失。
- **实际测试效果**：
  1. 单独使用会降低效率
  2. 结合reduction或collapse可略微提速
- **建议**：
  1. 暂时不采用
## task
- **作用**：将后续代码块封装为可由线程池动态调度的任务，实现细粒度并行化。
- **场景**：  
  - 递归或树形算法（如并行快速排序、并行归并排序）中，根据数据划分动态生成子任务。  
  - 任务执行时间不可预知或粒度不一时，通过任务调度动态均衡负载。  
  - 在事件驱动或不规则并行场景下，将独立工作单元封装为任务，最大化并行利用率。  
- **性能提升**：  
  1. 细粒度并行调度减少线程空闲，通过任务窃取机制改善负载均衡。  
  2. 递归分解后各线程可并发执行不同分支，利用多核CPU资源加速计算。  
  3. 动态创建和销毁任务避免了过度同步，提升整体吞吐并降低等待时间。
- **实际测试效果**：
  1. 在快速排序算法中,task结合任务切分阈值、结合任务切分阈值和任务优先级、结合任务切分和任务池控制都能显著提升几个量级的加速效果
- **建议**：
  1. 值得采用  
## taskloop
- **作用**：将 `for` 循环中的迭代分解成一组任务，自动生成并行任务并分配给线程池执行。
- **场景**：  
  - 需要对大规模循环迭代做动态负载均衡时，将循环拆分为多个任务更灵活。  
  - 循环体内工作量不均匀时，避免静态分配导致的线程空闲，通过任务动态分配减少负载差异。  
  - 在递归/迭代混合模型中，将循环内可并行部分以任务方式并发执行。  
- **性能提升**：  
  1. 结合 `grainsize` 自动控制任务粒度，确保最佳任务数量与执行效率。  
  2. 动态调度循环迭代，减少线程因固定分块后工作量不均而空闲。  
  3. 减少手动拆分循环的开发成本，利用运行时优化自动决定任务分配。
- **实际测试效果**：
  1. 模拟复杂的科学计算或数据处理任务时，无论何种策略均不如parallel for
- **建议**：
  1. 不建议采用  
## teams
- **作用**：在 Offloading 场景（如 `target`）中创建一个或多个线程团队（team），用于在设备端（GPU/加速器）并行执行代码。
- **场景**：  
  - 将计算密集型内核 Offload 到 GPU 时，使用 `#pragma omp teams` 划分线程团队。  
  - 在多级并行模型中，先使用 `teams` 创建多个团队，再在每个团队内部使用 `parallel` 或 `distribute`。  
  - 需要将大规模并行任务分发到不同线程组，充分利用设备多级硬件并行能力。  
- **性能提升**：  
  1. 在设备端按硬件拓扑创建线程团队，提高对 GPU SM 或 CPU 核心集群的利用率。  
  2. 支持层次化并行，将全局并行（teams）与局部并行（parallel）结合，实现更细粒度和更高吞吐。  
  3. 利用 `thread_limit` 与 `proc_bind` 等配合，优化设备线程排列，减少跨核通信延迟。
- **实际测试效果**：
  1. teams + distribute + 分块优化可以达到相对于串行55倍的加速比，超过parallel for约两倍
- **建议**：
  1. 非常值得采用 
## sections
- **作用**：将并行区域内的多个代码段划分为若干互斥执行的“段”，每个段由不同线程并行执行。
- **场景**：  
  - 多组独立任务逻辑不同但可并行执行时，例如同时计算矩阵加法和傅里叶变换。  
  - 可将互不依赖的功能模块并行化，避免在一个循环中硬编码分支判断。  
  - 需要对若干分支执行不同算法且保证各自并行时使用。  
- **性能提升**：  
  1. 多线程同时执行不同代码段，减少连续串行执行各功能模块带来的延迟。  
  2. 避免循环迭代中分支判断开销，提高并行效率。  
  3. 各“段”可独立优化线程数与负载，改善整体并行度。
- **实际测试效果**：
  1. 加速效果远不如单独使用parallel for
- **建议**：
  1. 不值得采用  
## section
- **作用**：与 `sections` 配对，标记并行区域内单个代码段的开始，指示该段由一个线程执行。
- **场景**：  
  - 在 `sections` 块内，每个 `section` 对应一个任务单元，各线程各自执行一个 `section`。  
  - 需要对若干互斥段落进行并行执行，以避免在循环中嵌套多次判断。  
- **性能提升**：  
  1. 明确分支代码由独立线程并行执行，减少串行分支判断开销。  
  2. 各 `section` 可并行优化，不同线程可同时处理不同算法逻辑，加速整体处理。  
  3. 与 `nowait` 配合使用时，避免等候其他 `section` 完成后再进入后续代码，减少同步开销。
- **实际测试效果**：
  1. 加速效果远不如单独使用parallel for
- **建议**：
  1. 不值得采用    
## atomic
- **作用**：对单个内存位置执行原子更新，保证并行线程可安全地无锁更新共享变量。
- **场景**：  
  - 多线程对同一个计数器、标志或累加器执行自增/自减操作时。  
  - 需要在高并发条件下避免使用重量级 `critical` 导致的严重性能瓶颈。  
  - 计算过程中频繁更新共享状态，但无需复杂操作（仅简单加减或交换）时。  
- **性能提升**：  
  1. 原子操作相比 `critical` 开销更低，减少线程切换与锁争用。  
  2. 高并发小粒度更新场景下，降低同步延迟，提升并行吞吐。  
  3. 避免大范围互斥锁导致的串行化开销，使更多线程并发执行。
  - **实际测试效果**：
  1. 在面临数据竞争时，直接添加parallel for计算结果是错误的
  2. 结合schedule或者动态调度，能有效的处理数据竞争，提升运行速率
- **建议**：
  1. 在面对数据竞争的时候，非常值得采用  

## target
- **作用**：在主机代码中指定一个计算区域将在设备（如 GPU、加速器）上并行执行，实现代码 Offloading。
- **场景**：  
  - 大规模数值模拟、矩阵乘法等计算密集型内核需在 GPU 上运行以加速。  
  - 异构平台（CPU+GPU）中，将部分热点函数 Offload 到加速器，减少主机负担。  
  - 分阶段计算中，在设备端先执行大规模并行任务，再将结果返回主机继续后续处理。  
- **性能提升**：  
  1. 利用设备强大的并行计算能力，加速大规模浮点或整数运算。  
  2. 减少主机 CPU 负载，将相对独立的计算逻辑转移至加速器，提高整体吞吐。  
  3. 与 `teams`、`distribute`、`parallel` 等联合使用，实现多层次并行优化。
 - **实际测试效果**：
  1. 无论哪个版本，性能都超过了parallel for
  2. target blocking版本遥遥领先，但是需要涉及到对函数进行分块重构
- **建议**：
  1. 非常值得采用  

## dispatch
- **作用**：在 OpenMP 5.0+ 中与 `loop` 或 `distribute` 配合使用，控制循环迭代在不同层次（Teams vs. Threads）上的分配方式。
- **场景**：  
  - 在 Offloading 场景下，通过 `#pragma omp distribute dispatch` 将循环迭代先分配给设备上的 Teams，再由 Teams 内线程进一步分配。  
  - 需同时优化全球分布（跨 Teams）与局部分布（Teams 内线程）的迭代映射策略时。  
  - 对嵌套循环进行分层并行调度，希望在 Teams 与 Threads 层面实现最佳负载均衡。  
- **性能提升**：  
  1. 分层分配循环迭代，提高硬件资源（GPU SM、CPU 物理核心）的利用效率。  
  2. 结合 `schedule`、`proc_bind` 优化 Teams 与 Threads 的绑定，减少跨组通信与延迟。  
  3. 动态或静态分配可根据迭代负载自动调节，使设备与主机均衡工作，提升吞吐。
  - **实际测试效果**：
  1. 无论哪个版本，性能都没有超过parallel for
- **建议**：
  1. 不建议采用  

## loop
- **作用**：与 `distribute`、`simd` 等联合使用，专门针对 Fortran 或 C/C++ 中的循环进行并行分配，类似于 `for`/`do`。
- **场景**：  
  - 在多层并行（如在 `teams` 内部）对循环进行并行调度时，用 `loop` 代替普通 `for`。  
  - 需要结合 `collapse` 或 `tile` 对多重嵌套循环进行并行化和优化时。  
  - Offloading 场景下，使用 `#pragma omp target teams distribute parallel loop` 将循环推送到设备端多个层次并行。  
- **性能提升**：  
  1. 专门优化循环分发逻辑，减少迭代划分开销，提高并行度。  
  2. 与 `collapse`、`tile` 配合可同时优化多维循环并行度与内存局部性。  
  3. 在层次化并行中，实现跨 Teams 与 Threads 的迭代划分，提升设备端并行吞吐。

## tile
- **作用**：将多维嵌套循环分割成若干“块”（tile/block），以提高数据重用和内存局部性，同时结合并行分配。
- **场景**：  
  - 矩阵乘法或多维数组处理时，对 i/j 循环进行块划分，以在缓存中保持数据重用。  
  - 多层嵌套循环中希望通过 `#pragma omp loop tile(tile_size)` 同时实现并行和块划分。  
  - 数据访问模式按行或列连续时，使用 Tile 减少缓存未命中并改善预取效率。  
- **性能提升**：  
  1. 提高缓存命中率，减少数据在多次循环迭代中频繁载入/驱逐的开销。  
  2. 将块分配给不同线程时保证局部数据访问集中，减少跨线程内存带宽竞用。  
  3. 与 `dispatch`、`collapse` 等配合，可在多层并行与块划分间取得最佳平衡。
- **实际测试效果**：
  1. 仅手动展开这一版本可以超过parallel for一倍加速比
- **建议**：
  1. 也许值得采用，待定   
## unroll
- **作用**：提示编译器对循环体进行展开，将多次迭代合并在一个循环中执行，以减少循环控制开销。
- **场景**：  
  - 循环迭代次数固定且较少时，通过 `#pragma omp loop unroll(n)` 展开 `n` 次迭代减少循环跳转。  
  - 计算密集型循环希望减少分支与计数器更新开销时。  
  - 与 `simd`、`tile` 等联合使用时，通过展开配合矢量化获得更高吞吐。  
- **性能提升**：  
  1. 减少循环跳转和索引更新指令，降低指令流水线停顿。  
  2. 展开后的循环体更有利于向量化，提高单次循环处理元素数量。  
  3. 结合 `safelen`/`aligned` 提示，可进一步加大向量宽度，提升内存带宽利用。
- **实际测试效果**：
  1. 加速效果大多不如parallel for
  2. 仅手动展开这一版本略微超过了parallel for
- **建议**：
  1. 不值得采用   
---



## single
- **作用**：在并行区域内仅由一个线程执行后续代码块，其他线程跳过该块并在结束时等待（除非加 `nowait`）。
- **场景**：  
  - 并行区域中需要某段只初始化或设置一次（如分配内存、读写配置文件等）。  
  - 避免每个线程重复执行只需执行一次的串行操作，同时确保并行区域结构正确。  
- **性能提升**：无

## master
- **作用**：在并行区域内仅由主线程（线程 0）执行后续代码块，其他线程跳过且不等待（无隐式屏障）。
- **场景**：  
  - 需要使用主线程执行特定 I/O、日志写入或环境检查，而不希望其他线程被阻塞。  
  - 与 `single` 不同，`master` 不带隐式屏障，可以减少因等待而产生的开销，但本身不提供并行加速。  
- **性能提升**：无

## taskwait
- **作用**：在任务生成点插入一个同步点，等待当前线程生成的所有子任务完成后再继续执行。
- **场景**：  
  - 在任务图中需要在某处收集所有子任务结果，再进行后续串行操作时使用。  
  - 避免并行任务继续执行而导致对下游数据访问冲突。  
- **性能提升**：无

## taskyield
- **作用**：提示运行时当前任务可暂时让出 CPU，以便其他就绪任务获得执行机会，类似于 “协作式” 任务切换提示。
- **场景**：  
  - 当任务内部有长时间计算，又希望在空闲时让出给其他短任务执行，以改善负载均衡。  
  - 在不规则任务流中，通过显式 `taskyield` 减少某一任务长时间独占 CPU 资源。  
- **性能提升**：无

## declare
- **作用**：用于声明 OpenMP 相关属性或函数版本，如 `declare simd`, `declare target` 等元指令，影响编译器生成的代码版本或映射规则。
- **场景**：  
  - 为函数声明矢量化版本（`declare simd`）或设备端版本（`declare target`）。  
  - 在模块/库级别提供多个实现（如标量 vs. 矢量）供编译器或运行时选择。  
- **性能提升**：无

## metadirective
- **作用**：OpenMP 5.0+ 新增，用于根据运行时或编译时条件选择不同指令方案，类似条件编译的并行指令选择器。
- **场景**：  
  - 在代码中预先定义多种并行化实现（如针对不同硬件特性），使用 `metadirective` 在运行时选择最优实现。  
  - 需要根据环境（如线程数、设备类型）动态决定使用 `simd`、`teams` 或其他指令时。  
- **性能提升**：无

## masked
- **作用**：在 OpenMP 5.1+ 中用于 `#pragma omp masked`，根据遮罩表达式将并行区域分为 “inbranch” 与 “notinbranch”，控制不同线程执行不同分支。
- **场景**：  
  - 并行区域需对满足条件与不满足条件的线程执行不同代码路径时，使用 `masked` 分流逻辑。  
  - 需要在同一个并行上下文内实现条件执行而不分离为两个并行区域时。  
- **性能提升**：无

## scope
- **作用**：OpenMP 5.0+ 中用于指定某些元指令或属性的作用域，如 `declare simd` 的函数作用范围或 `declare target` 的模块作用范围。
- **场景**：  
  - 当对特定模块或函数集合应用 OpenMP 约束时，通过 `#pragma omp declare simd scope(function)` 限定作用范围。  
  - 需要在大型项目中分层管理 OpenMP 声明时，用 `scope` 控制属性生效的范围。  
- **性能提升**：无

## interop
- **作用**：OpenMP 5.2+ 引入，用于与其他并行模型（如 MPI、线程库）实现互操作，控制任务与并行区域在不同并行生态间协同。
- **场景**：  
  - 在混合编程（MPI+OpenMP、CUDA+OpenMP）中，需要控制任务或线程在不同并行后端间切换。  
  - 与第三方线程池协调并行调度时，使用 `interop` 定义接口和回调钩子。  
- **性能提升**：无

## declare mapper
- **作用**：用于 Offloading 场景中，为用户自定义数据类型定义主机与设备之间的数据映射规则。
- **场景**：  
  - 复杂结构体或自定义类需要在主机与设备之间传输时，使用 `declare mapper` 定义映射方法。  
  - 避免在每次 `map` 子句中手动指定成员，简化数据传输声明。  
- **性能提升**：无

## 子句类  

## reduction
- **作用**：在并行区域中对共享变量执行归约操作，将各线程的局部结果汇总到全局变量。
- **场景**：需要对向量求和、求最大/最小值、逻辑与/或等可归约运算时使用，如并行计算矩阵元素之和。
- **性能提升**：  
  1. 避免显式的原子操作或临界区锁竞争，减少同步开销。  
  2. 各线程维护私有副本后再合并，降低内存访问冲突。  
  3. 随着线程数增加，归约操作可近似线性加速。
- **实际测试效果**：
  1. 加速效果远超parallel for几十倍
  2. reduction for和reduction + 分块版本结果非常亮眼
- **建议**：
  1. 非常值得采用    
## nowait
- **作用**：在并行循环（如 `#pragma omp for` ）或任务（如 `#pragma omp task` ）后取消隐含的线程同步屏障，使线程无需等待即可继续执行后续代码。
- **场景**：当后续代码与当前循环/任务结果无依赖关系，或依赖关系由更精细的同步机制（如 `taskwait`）管理时使用。
- **性能提升**：  
  1. 消除不必要的全局同步点，减少线程空闲等待时间。  
  2. 提高流水线并行度，允许生产者/消费者型任务并发执行。  
  3. 在多阶段并行算法里，各阶段可重叠执行，提升整体吞吐量。
- **实际测试效果**：
  1. 加速效果比parallel for略好
  2. 线程越多，加速效果越好
- **建议**：
  1. 值得采用，但要考虑线程数   

## schedule
- **作用**：为并行循环指定迭代分配策略，如 `static`, `dynamic`, `guided` 等，以控制迭代到线程的映射方式。
- **场景**：  
  - 循环迭代工作负载不均匀时使用 `dynamic` 或 `guided`；  
  - 负载均衡性要求极高且迭代数固定时使用 `static` 分块。  
  - 需要指定分块大小时使用 `schedule(type, chunk)`。
- **性能提升**：  
  1. 动态调度（`dynamic`/`guided`）在负载不均时减少线程空闲，改善负载平衡。  
  2. 静态调度（`static`）在迭代数大且工作量均匀时，减少调度开销。  
  3. 自定义分块大小（`chunk`）可控制缓存局部性和同步开销，进一步优化性能。
- **实际测试效果**：
  1. 无论哪种策略，加速效果比parallel for好很多
  2. 不同schedule策略的性能差异主要体现在负载不均衡的情况下，Dynamic和Guided通常在工作负载不均匀时表现更好
- **建议**：
  1. 非常值得采用 

## num_threads
- **作用**：为并行区域（`parallel`）显式指定线程数，覆盖默认线程数设定。
- **场景**：  
  - 在嵌套并行或多级并行策略中，需要为特定区域使用不同线程数。  
  - 系统环境变量已设置较大线程数，但某一段代码仅需较少线程时显式控制。
- **性能提升**：  
  1. 避免线程过度创建或不足，减少上下文切换与资源争用。  
  2. 保证线程数与硬件核心数或 NUMA 拓扑匹配，提高计算资源利用率。  
  3. 在 NUMA 架构上，通过限制线程数到单一节点，改善内存访问延迟。
- **实际测试效果**：
  1.  num_threads嵌套版本实际加速效果都不如parallel for
  2.  配合parallel for，指定最大线程，可以取得比parallel for更好的效果
- **建议**：
  1.  一般不建议采用
  2. 配合简单的parallel for可取得略微提速

## proc_bind
- **作用**：控制线程与处理器（或核心/NUMA 节点）绑定策略，如 `master`, `close`, `spread` 等，以改善线程亲和性。
- **场景**：  
  - 程序对缓存敏感，需要线程固定在某些核心上以保持缓存热数据。  
  - 多线程在不同 NUMA 节点间移动会导致跨节点访问延迟，需要绑定到本地节点。  
  - 多级并行时，需要细粒度绑定策略来避免线程聚集或散布过度。
- **性能提升**：  
  1. 减少线程在核心/节点间迁移带来的上下文切换和缓存失效。  
  2. 提高内存局部性，尤其在 NUMA 系统上，减少远程内存访问延迟。  
  3. 对于对称多处理器( SMP )系统，合理绑定可平衡各核心负载，避免热点。
- **实际测试效果**：
  1.  直接使用，无论哪个亲和性策略，都不如parallel for
  2.  proc_bind(close) + OMP_PLACES版本性能超越了parallel for但不多
- **建议**：
  1.  先观望，后期考虑


## collapse
- **作用**：将多重嵌套循环“折叠”为一个逻辑循环进行并行分配，使外层和内层循环迭代整体参与调度。
- **场景**：  
  - 嵌套循环层数较深，且内层循环迭代数较少，单独并行内层循环负载不够时。  
  - 希望将 `for (i…) for (j…)` 一并成 `for (i×j…)` 来提高并行度。  
  - 数据访问模式允许跨嵌套维度的连续访问，以保持缓存局部性。
- **性能提升**：  
  1. 增加并行循环的总体迭代数，提升线程并行度，避免线程空闲。  
  2. 通过一次性划分更大循环区间，减少划分开销和同步次数。  
  3. 若访问模式合理，可利用内存预取和缓存行连续性，减少缓存未命中。
- **实际测试效果**：
  1. 相较于parallel for略微提速
- **建议**：
  1.  暂时不采用，建议后期采用
## grainsize
- **作用**：指定任务划分时的最小粒度，控制任务调度和窃取大小，常与 `taskloop` 联合使用。
- **场景**：  
  - 任务递归生成树深度较深时，需要控制子任务的最小工作量。  
  - 任务负载不均时，通过调节 `grainsize` 平衡并行度与调度开销。  
  - 在计算密集型递归分解算法（如并行快速排序、并行归并）中使用。
- **性能提升**：  
  1. 减少任务过细导致的调度开销，提高任务执行效率。  
  2. 避免单个任务过大导致负载不均，改善整体负载平衡。  
  3. 最优 `grainsize` 可结合硬件线程数与问题规模达到最大吞吐。
- **实际测试效果**：
  1. 相较于parallel for提速不明显，甚至不如parallel for
- **建议**：
  1. 不值得使用
## hint
- **作用**：向编译器或运行时提供额外优化提示，如 `subnorm`（特殊浮点格式）、`safelen` 等与向量化相关提示。
- **场景**：  
  - 使用 `#pragma omp simd hint([named_hint])` 时，可启用自定义向量化提示。  
  - 希望编译器在有条件地忽略某些依赖或假定对齐时，通过专门的 `hint` 优化循环。  
  - 结合特定硬件指令集（如 AVX-512）时，利用 `hint` 提高向量化效率。
- **性能提升**：  
  1. 帮助编译器做更激进的向量化决策，从而生成更高效的 SIMD 指令。  
  2. 在浮点操作中，可提供更宽松的规范（如允许非规范子正常数），减少指令开销。  
  3. 在循环自动向量化受限时，通过手动 `hint` 可以解锁硬件能力。
- **实际测试效果**：
  1. simd hint版本3 (simdlen)在多线程是提速效果显著（42倍）
  2. 其他策略效果不明显
- **建议**：
  1. 需要结合具体场景使用simd hint版本3 (simdlen)

## safelen
- **作用**：用于 `#pragma omp simd safelen(n)`，指示向量化时循环迭代间访问间隔至少 `n`，编译器据此判断是否可安全展开向量指令。
- **场景**：  
  - 循环中访问数组 `a[i+n]` 跳步访问时，明确告诉编译器不会发生依赖冲突。  
  - 手动优化向量化代码时，用于告诉编译器最远的距离，允许更大宽度向量化。  
  - 当编译器对数据依赖检测保守时，通过指定 `safelen` 强制向量展开。
- **性能提升**：  
  1. 提高自动向量化强度，利用更宽 SIMD 寄存器加速循环。  
  2. 减少编译器因不确定依赖而插入的标量或缩减化指令。  
  3. 在内存访问模式明确的情况下，可显著提升内存带宽利用率。
- **实际测试效果**：
  1. 加速效果有限
- **建议**：
  1. 不建议采用
## simdlen
- **作用**：在 `#pragma omp simd simdlen(n)` 中显式指定向量化宽度（迭代数），迫使编译器生成 `n` 跳的 SIMD 指令。
- **场景**：  
  - 确定底层硬件支持 8 跳或 16 跳向量指令时，强制使用对应宽度。  
  - 循环迭代对齐且可整除指定宽度时，提高向量化效率。  
  - 在编译器默认无法选取最优向量宽度时，通过 `simdlen` 手动指定。
- **性能提升**：  
  1. 保证生成的矢量指令满载硬件向量寄存器，提高单指令吞吐量。  
  2. 减少对标量尾部处理的开销，提高循环主体执行效率。  
  3. 与 `aligned` 联合使用可确保对齐，有效减少内存访问延迟。
- **实际测试效果**：
  1. 相较于parallel for略微提速
- **建议**：
  1.  暂时不采用，建议后期采用
## linear
- **作用**：用于 `#pragma omp simd linear(var:step)`，指示向量化循环时 `var` 在迭代间以 `step` 线性递增/递减，便于编译器展开向量化代码。
- **场景**：  
  - 循环中指针或索引随迭代按固定步长移动时，例如 `for(i) a[i] = b[i+offset]`。  
  - 需要手动帮助编译器识别简单递增关系，解除编译器保守依赖检查。  
  - 与 `simd`、`aligned` 结合使用时，让编译器生成更高效的矢量访存指令。
- **性能提升**：  
  1. 明确告诉编译器依赖关系，删除或简化依赖检查指令，加速向量展开。  
  2. 在循环内存访问模式规则时，提高 SIMD 单指令多数据执行效率。  
  3. 避免额外的循环归约或标量补尾，减少循环整体指令数。
- **实际测试效果**：
  1. 相较于parallel for性能有了极大提升，但要结合合适的策略（比如aligned）
- **建议**：
  1.  建议采用
## aligned
- **作用**：在 `#pragma omp simd aligned(ptr:align)` 中指定指针 `ptr` 按 `align` 字节对齐，帮助编译器生成对齐访存指令。
- **场景**：  
  - 已知数组或数据结构以 16、32、64 字节对齐时，通过 `aligned` 保证向量化访存对齐安全。  
  - 需要最大化缓存行利用率、减少跨缓存行访问导致的性能下降。  
  - 在多维数组计算或矩阵乘法中，通过对齐可减少缓存未命中。
- **性能提升**：  
  1. 生成对齐 SIMD 载入/存储指令，减少未对齐访问带来的性能损耗。  
  2. 避免编译器因保守假设而插入的对齐检查或调度屏障。  
  3. 在大数据量循环中，内存带宽与缓存效率显著提升。

## uniform
- **作用**：用于 OpenMP 5.1+ 的 `declare simd uniform(var)`，指示在生成 SIMD 版本的函数时，`var` 在所有向量 lane 中保持不变。
- **场景**：  
  - 多线程或多 lane 并行调用一个函数时，部分参数在所有 lane 中相同。  
  - 需要编译器生成更高效的向量化函数调用，减少冗余寄存器搬移。  
  - 用于加速数学函数或自定义内联函数，在矢量化调用中共享常量参数。
- **性能提升**：  
  1. 减少矢量化函数内部的参数加载与广播开销。  
  2. 提高矢量指令流水线利用率，减少分支或控制流开销。  
  3. 在大量数据并行计算中，加速函数调用带来的矢量瓶颈。
  - **实际测试效果**：
  1. 线程少时效果略高于parallel for
  2. 线程多时效果不如parallel for
 - **建议**：
  1. 不采用
---



## private
- **作用**：为每个线程分配一个局部私有副本，保证线程之间的变量互不干扰。
- **场景**：当并行区域中需要局部临时变量且不希望线程之间相互覆盖时使用。
- **性能提升**：无

## shared
- **作用**：将变量声明为共享，使所有线程可直接访问同一内存地址。
- **场景**：需要在线程间共享只读数据或并发读写但通过原子/锁机制保护时使用。
- **性能提升**：无

## default
- **作用**：指定并行区域中未显式声明的数据共享属性，如 `default(none)`、`default(shared)`。
- **场景**：用于强制要求开发者显式声明每个变量的作用域或在不声明时使用默认策略。
- **性能提升**：无

## firstprivate
- **作用**：为每个线程创建一个变量的私有副本，并将该副本初始化为原变量的值。
- **场景**：当每个线程初始化时需要从并行区域外继承初始值时使用。
- **性能提升**：无

## lastprivate
- **作用**：为每个线程创建变量私有副本，并在并行结构完成后，将最后一个迭代的副本值复制回原变量。
- **场景**：当并行循环结束后需要保存最后一次迭代结果到共享变量时使用。
- **性能提升**：无

## copyin
- **作用**：在并行区域开始时，将主线程（或创建线程）的线程私有数据复制到其他线程的对应私有副本中，仅对 `threadprivate` 变量生效。
- **场景**：多个并行段之间需要共享同一个 `threadprivate` 变量的初始值时使用。
- **性能提升**：无

## copyprivate
- **作用**：在 `single` 或 `master` 段中使用，将一个线程的私有变量广播到同一并行区域内其他线程的私有副本。
- **场景**：当多个线程的私有变量需要在单线程段结束后保持一致时使用，如并行初始化只做一次。
- **性能提升**：无

## defaultmap
- **作用**：在 OpenMP 5.0+ 中指定未显式 `map` 的变量在设备（如 GPU）上的映射策略，如 `defaultmap(to:scalar)`。
- **场景**：用于简化设备数据映射，设置默认映射规则，以减少显式 `map` 声明。
- **性能提升**：无

## depend
- **作用**：在 `task` 中声明任务间依赖关系，保证按指定顺序执行。
- **场景**：并行任务图中需要精确控制任务执行先后顺序时使用，如流水线阶段化并行。
- **性能提升**：无

## map / to / from / tofrom / is_device_ptr
- **作用**：  
  - `map`：指定将主机内存与设备内存之间的数据传输方向；  
  - `to`：将主机变量拷贝到设备；  
  - `from`：将设备变量拷贝回主机；  
  - `tofrom`：先 `to` 再 `from`；  
  - `is_device_ptr`：指定指针已经在设备上分配，并无需再分配/拷贝。  
- **场景**：OpenMP Offloading（如 GPU 加速）时，需要控制数据在主机和设备间的流向。例如 `#pragma omp target map(to: a[0:N])`。
- **性能提升**：无

## device
- **作用**：用于 `#pragma omp target device(k)`，指定并行区域运行在第 k 号设备（如 GPU）上。
- **场景**：多 GPU 系统中显式选择目标设备，或在异构平台上管理多个加速器。
- **性能提升**：无

## thread_limit
- **作用**：设置并行区域中可创建线程的最大值，用于控制并行嵌套层次的线程数上限。
- **场景**：嵌套并行时，避免线程数指数级增长导致资源耗尽。例如 `#pragma omp parallel thread_limit(4)`。
- **性能提升**：无

## if
- **作用**：条件化并行执行，根据表达式结果决定是否以并行方式执行区域，否则串行执行。
- **场景**：当问题规模较小时，通过 `if(n > threshold)` 避免并行开销；当运行环境不支持并行也可退化为串行。
- **性能提升**：无

## final
- **作用**：在 OpenMP 4.5+ 中用于 `task`，指示如果条件为真，则当前任务及其子任务按串行方式执行，不再生成子任务。
- **场景**：在递归任务生成时，当子问题规模较小时，使用 `final( depth>k )` 防止过多任务生成。
- **性能提升**：无

## mergeable
- **作用**：在 OpenMP 5.0+ 中用于 `task`, 提示运行时可将当前任务与其子任务合并执行，以减少调度开销。
- **场景**：生成大量细粒度任务时，允许运行时合并相邻任务，以提高执行效率。
- **性能提升**：无

## untied
- **作用**：用于 `task`，允许任务在不同线程间迁移执行，不强制绑定到创建它的线程。
- **场景**：当任务执行时间不确定或需要更灵活的负载均衡时使用，可让任务被其他空闲线程窃取。
- **性能提升**：无

## priority
- **作用**：在 OpenMP 5.0+ 中为 `task` 指定优先级，运行时可根据优先级调度任务以提高执行效率。
- **场景**：关键任务需要优先执行时使用，如图算法中需优先展开关键路径任务。
- **性能提升**：无

## task_reduction / in_reduction
- **作用**：  
  - `in_reduction`：在任务中声明归约变量；  
  - `task_reduction`：在并行任务图中跨任务执行归约操作。  
- **场景**：复杂任务树中对全局归约变量执行并行归约，如并行图遍历或并行数值算法。  
- **性能提升**：无

## match
- **作用**：用于 OpenMP 5.1+ 中的 `declare variant`，基于架构、编译器或运行时条件选择不同函数变体，支持多版本优化。
- **场景**：同一函数有多种实现（如标量版、AVX2、AVX-512），通过 `match` 选择最优版本并行调用。  
- **性能提升**：无

## filter
- **作用**：在 OpenMP 5.1+ 中与 `declare variant` 联合使用，根据标签过滤选择适当函数变体。  
- **场景**：根据架构特性（如 `target { x86_64-v2 }`）在编译时或运行时选择最优实现。  
- **性能提升**：无

## defaultmap
- **作用**：在 OpenMP 5.0+ 中为未显式声明 `map` 的变量指定默认映射策略（详见上文 “defaultmap”）。
- **场景**：简化 Offloading 数据映射规则，无需为每个变量都写 `map`。
- **性能提升**：无

## novariants
- **作用**：用于 OpenMP 5.1+ 中的 `declare variant`，指示不为某些目标生成变体版本。  
- **场景**：当某一架构不需要特定优化或已有足够高效实现时，通过 `novariants` 跳过变体生成。  
- **性能提升**：无

## reverse_offload
- **作用**：在 Offloading 场景下，当设备（如 GPU）需要调用主机函数时使用，允许从设备发起主机回调。  
- **场景**：在 GPU 内核中需要访问主机端 I/O 或复杂库函数时，设备发起对主机函数的调用。  
- **性能提升**：无

## indirect
- **作用**：在 OpenMP 5.0+ 中，配合 `declare variant` 和 `bind`，用于间接调用目标变体。  
- **场景**：需要根据运行时决策在多个实现间切换时使用，如同一函数在不同加速器上实例化。  
- **性能提升**：无

## inclusive/exclusive
- **作用**：在 OpenMP 5.0+ 的 `teams distribute` 或 `distribute` 中用于定义循环边界为包含式（inclusive）或排除式（exclusive）。  
- **场景**：为了明确由哪些迭代参与并行分配，以避免 off-by-one 错误。  
- **性能提升**：无

## severity / transparent
- **作用**：在 OpenMP 5.0+ 中与性能分析或调试相关，用于设置运行时消息严重性等级（`severity`）或指定变体函数为透明（`transparent`），以便性能分析工具更好地聚焦热点。  
- **场景**：性能剖析或调试 OpenMP 代码时，通过设置日志输出级别或标记某些函数不计入剖析统计。  
- **性能提升**：无

## use
- **作用**：OpenMP 5.1+ 中与模块（`module`）配合使用，指定引入特定 OpenMP 接口或功能模块。  
- **场景**：使用 Fortran 或 C/C++ 模块化 OpenMP 接口时，通过 `use omp_lib`、`use omp_lib, only: …` 等形式导入。  
- **性能提升**：无




## 内存管理相关
## allocator
- **作用**：定义并指定自定义内存分配器类型，用于后续 `allocate` 或数据映射时调用特定分配策略。  
- **场景**：需要在 NUMA 架构、多级内存体系（如 CPU/GPU 异构系统）或大数据结构分配前，选择合适的内存池（如高带宽内存、页锁定内存等）。  
- **性能提升**：  
  1. 通过使用针对特定硬件优化的分配器，减少动态分配开销。  
  2. 将数据放置在离计算单元更近的内存节点，降低访问延迟并提高带宽利用率。  
  3. 降低内存碎片率，改善缓存局部性，提升循环或内存密集型计算的效率。  

## allocate
- **作用**：使用某个已定义的 `allocator` 分配一块连续内存（在主机或设备）并返回指针，可在后续并行或 Offloading 区域直接使用。  
- **场景**：  
  - 在进入 `parallel`、`target` 或 `teams` 等并行区域之前，显式预先分配大数组或缓冲区；  
  - 在 GPU Offloading 场景下，用自定义分配器将数组直接分配到设备内存；  
  - 需要对一段内存进行一次性批量分配，并在多次并行操作中复用。  
- **性能提升**：  
  1. 预先分配并保持内存连续，避免并行区域运行时的重复分配与释放开销。  
  2. 利用自定义分配器在本地 NUMA 节点或高带宽内存中分配，降低远程访问延迟。  
  3. 在设备端（GPU）预分配大块显存，减少 Host↔Device 传输次数并提升数据传输带宽。  

## uses_allocators
- **作用**：将一个或多个 `allocator` 绑定到某个并行区域（如 `parallel`、`task`、`teams`）或数据映射区域，使该区域内的所有内存分配请求默认采用指定分配器。  
- **场景**：  
  - 在一个包含多次动态分配的并行头（`#pragma omp parallel uses_allocators(myAlloc:allocator)`）中，让所有线程或任务都使用同一分配策略；  
  - 在 Offloading 时，用 `uses_allocators` 将 GPU 专用分配器附加到 `target enter data` 或 `target data` 区域，确保后续映射和分配都走该分配器；  
  - 需要在单次并行区域内统一管理内存分配，以保证不同线程获得相同策略下的内存布局。  
- **性能提升**：  
  1. 统一分配策略，减少多次单独调用 `allocate` 或默认 `malloc` 带来的上下文切换与碎片化；  
  2. 提高线程或任务间的内存布局一致性，增强缓存预取和并行访问的局部性；  
  3. 在多层并行（如嵌套并行或 CPU→GPU Offload）中保持分配策略一致，避免不同阶段重复分配或跨内存域拷贝。  

## 任务依赖相关 
## depobj
- **作用**：声明并操作 OpenMP 5.0+ 中的依赖对象，用于管理一组任务之间的依赖关系。
- **场景**：在复杂任务图中，需要多个任务对同一依赖对象注册依赖（如 `depend(in:obj)` 或 `depend(out:obj)`），以便统一控制任务的执行顺序。
- **性能提升**：无

## detach
- **作用**：在任务内部使用 `detach(depobj)` 将当前任务分离，使此任务可提前返回并让后续依赖于该依赖对象的任务在满足条件时立即执行。
- **场景**：当一个任务完成必要的初始化工作后，需将剩余工作封装为依赖对象，让其他任务并行进行，而本任务主体可提前结束以释放执行资源。
- **性能提升**：无

## holds
- **作用**：用于在 `depend` 语句中表示当前任务在访问某个依赖对象时保持其持有状态，例如 `depend(inout:obj) holds(...)`。
- **场景**：当任务需要对子依赖对象进行多阶段或重复访问时，通过 `holds` 表示在整个执行阶段持续对该对象保持访问权限。
- **性能提升**：无

## init_complete
- **作用**：在 OpenMP 5.0+ 中与依赖对象配合使用，表示该依赖对象的初始化已完成，可被其他任务注册依赖。
- **场景**：在创建并初始化诸如锁、缓冲区或共享资源的过程中，当完成初始化后，调用 `init_complete(depobj)` 通知依赖对象已就绪。
- **性能提升**：无

## init
- **作用**：用于初始化 OpenMP 5.0+ 中的依赖对象，例如 `omp_init_depobj(&obj)`，将其置于可供后续 `depend` 动作使用的状态。
- **场景**：在使用任何依赖操作（如 `depend`, `detach`）之前，需要先调用 `init(depobj)` 来创建并初始化依赖对象。
- **性能提升**：无

## 设备相关扩展


## unified_shared_memory
- **作用**：指示在 OpenMP Offloading 场景下，所使用的内存是主机和设备共享的统一内存，无需显式拷贝。  
- **场景**：  
  - 在支持统一共享内存（UVM/USM）的硬件（如部分 NVIDIA GPU、AMD GPU 或 Intel CPU+GPU 一体架构）上，开启 OpenMP Offloading 时。  
  - 代码中需要频繁在主机和设备之间读写同一数据结构，并且希望避免重复的数据传输。  
  - 对于动态分配的大块数据，在进入并行或 `target` 区域前，通过 `allocate` 配合 `allocator(unified_shared_memory)` 进行分配。  
- **性能提升**：  
  1. 消除主机与设备之间的显式 `to/from` 数据拷贝开销，减少内存传输延迟。  
  2. 在访问频繁切换主机/设备的场景下，利用统一地址空间提高数据访问局部性。  
  3. 减少编程复杂度和同步点，使得并行执行更流畅，有助于提升整体吞吐和带宽利用率。  

---



## device_type
- **作用**：在 `#pragma omp target device_type(type)` 中指定目标设备类型（如 `host`、`gpu`、`accel` 等），用于选择要执行 Offloading 的设备类别。  
- **场景**：多设备系统中，需要显式指定将并行区域或 `target` 区域运行在哪种设备上，例如在拥有 CPU 和 GPU 的系统里选择仅在 CPU（host）上执行。  
- **性能提升**：无

## is_device_ptr
- **作用**：在 OpenMP Offloading 中，用于指明一个指针已经指向设备内存，不需要再进行额外的内存拷贝或分配。  
- **场景**：当通过低级接口（如 CUDA、HIP）或先前的 `allocate`/`attach` 操作获得了设备指针时，在后续的 `target` 或 `teams` 语句中使用，避免重复拷贝。  
- **性能提升**：无

## 同步原语


## atomic
- **作用**：对单个内存位置执行原子操作，使并行线程可以安全地进行无锁更新。
- **场景**：  
  - 多线程对同一计数器、累加器等进行频繁更新时。  
  - 需要避免使用重量级的 `critical` 区域但又要求原子性时。  
  - 在简单的累加、增减或更新标志等场景中替代互斥锁，以减少开销。  
- **性能提升**：  
  1. 原子操作开销低于 `critical` 区域，减少线程切换和锁争用。  
  2. 避免大范围临界区，降低同步延迟，提升并行吞吐量。  
  3. 在高并发小粒度更新场景下可显著降低总的同步开销。  
  - **实际测试效果**：
  1. 在面临数据竞争时，直接添加parallel for计算结果是错误的
  2. 结合schedule或者动态调度，能有效的处理数据竞争，提升运行速率
- **建议**：
  1. 在面对数据竞争的时候，非常值得采用  



## barrier
- **作用**：在并行区域中强制所有线程在此同步点等待，等待所有线程到达后再继续执行。
- **场景**：  
  - 在并行循环或并行段中需要保证执行顺序或等待其他线程完成某些阶段工作时。  
  - 需要将各线程汇总后再进行后续计算时使用。  
- **性能提升**：无

## critical
- **作用**：指定一个互斥区域，在同一时间只允许一个线程进入，以保护共享资源访问。
- **场景**：  
  - 需要对共享数据进行复杂更新或无法用 `atomic` 表达的操作时。  
  - 并行区域内对输出、日志、文件操作等不可并行化的临界段进行保护。  
- **性能提升**：无

## ordered
- **作用**：在并行循环中保证标记为 `ordered` 的代码块按照原始循环顺序依次执行。
- **场景**：  
  - 并行循环中需要按顺序访问或输出结果，例如保持输出顺序一致。  
  - 需要依赖于迭代顺序的逻辑计算。  
- **性能提升**：无

## flush
- **作用**：在并行程序中强制刷新线程本地缓存，将变量值与主内存同步。
- **场景**：  
  - 在没有明确同步构造（如 `barrier`、`critical`）时，手动确保可见性和顺序一致性。  
  - 在使用锁或原子以外的方式进行跨线程通信时，插入 `flush` 以防止编译器/硬件重排序。  
- **性能提升**：无

## 运行时函数特性
## requires
- **作用**：在 `#pragma omp requires` 中指定编译器或运行时对 OpenMP 规范的支持要求，如 `requires unified_shared_memory`、`requires atomic_precision` 等，用于在程序启动时验证运行时是否满足指定特性。
- **场景**：  
  - 编写需要特定 OpenMP 功能（如统一共享内存、严格原子精度等）的程序，需要在编译或运行时立即失败以避免错误行为。  
  - 确保在目标平台上 OpenMP 运行时具备某些扩展功能，再执行后续并行代码。  
- **性能提升**：无

## error
- **作用**：在 `#pragma omp error` 中指定当某种条件不满足时立即触发编译或运行时错误，通常与 `requires` 一起使用，用于强制执行编译期或运行时检查。
- **场景**：  
  - 当需要保证程序运行时不被允许在不支持的环境下继续执行，防止程序在不满足条件时产生未定义行为。  
  - 在开发阶段，通过此指令捕获配置不当或不支持的 OpenMP 特性，便于快速排错。  
- **性能提升**：无

## cancel
- **作用**：在 `#pragma omp cancel` 中用于取消当前并行区域、循环或任务的执行，可与 `cancel point` 一起使用，通过检查条件选择性结束并行工作。
- **场景**：  
  - 在并行循环或任务中，当满足某种中止条件（如找到目标元素或达到最优解）时，提前停止剩余并行计算。  
  - 实现类似“提前退出”机制，无需等待所有线程或任务完成即可跳出并行区域。  
- **性能提升**：无

## 指令修饰符


## dynamic_allocators
- **作用**：允许在并行或 Offloading 区域中动态选择内存分配器，使分配策略在运行时可配置和切换。  
- **场景**：  
  - 在 NUMA 架构或多级缓存系统下，根据运行时环境选择最佳分配器，将数据放置在距离计算单元更近的内存节点。  
  - 针对 CPU/GPU 异构平台，运行时通过环境变量或 API 切换到专门的设备分配器。  
  - 大规模并行程序需要在不同阶段使用不同分配策略来减少碎片并提高带宽利用率。  
- **性能提升**：  
  1. 根据硬件拓扑和内存层次动态适配，减少跨节点访问延迟。  
  2. 避免因使用默认分配导致的高碎片率，提高缓存局部性和带宽利用。  
  3. 可在不重编译的情况下通过环境或运行时调整，使程序针对不同平台自动调优。
- **实际测试效果**：
  1. dynamic chunked、Interchange+Schedule、Interchange+Collapse均有大幅提升
- **建议**：
  1. 值得采用
## interchange
- **作用**：在多层嵌套循环中指示编译器将循环层级交换，以优化并行度和数据访问模式。  
- **场景**：  
  - 存在多层嵌套循环时，外层迭代数较少、内层迭代数较多，希望将可并行的循环提升到最外层。  
  - 矩阵运算、张量计算等对缓存局部性敏感，需要通过交换循环顺序减少缓存未命中。  
  - 内外循环访问模式不同，想让数据沿连续内存方向访问，从而减少内存带宽瓶颈。  
- **性能提升**：  
  1. 将并行循环置于更高迭代级别，提高线程并行度，减少线程空闲。  
  2. 调整访问顺序以改善数据局部性，减少缓存未命中次数。  
  3. 根据循环访问特性实现更均衡的负载分配，避免某些线程过早结束或过度等待。
- **实际测试效果**：
  1. Interchange+Collapse和Interchange+Schedule版本大幅度超越纯Parallel for版本
 - **建议**：
  1. 值得采用
---



## reverse
- **作用**：在并行循环中指示以相反顺序进行迭代，例如从循环末尾向开头处理数据。  
- **场景**：  
  - 需要与原始串行循环方向保持一致，或者出于算法逻辑要求必须倒序处理。  
  - 在某些边界条件或依赖模式下，为了保证正确性而使用逆向遍历。  
- **性能提升**：无

## partial
- **作用**：用于指示在某些分布或并行化模式下，仅分配或处理部分迭代／任务，留下剩余部分由其他机制处理。  
- **场景**：  
  - 嵌套循环或分布式循环中需要只并行化一部分迭代，以避免冲突或满足数据依赖。  
  - 需要对大型数据集分阶段并行处理，将部分任务留给后续阶段或其他处理单元。  
- **性能提升**：无

## 特殊上下文限定

## notinbranch
- **作用**：用于 `#pragma omp masked` 中，指定只有不满足遮罩条件（mask）的线程才执行后续代码块。
- **场景**：当需要在同一个并行区域中对满足条件和不满足条件的线程分别执行不同逻辑时，使用 `notinbranch` 来标记不满足条件的分支。
- **性能提升**：无

## inbranch
- **作用**：用于 `#pragma omp masked` 中，指定只有满足遮罩条件（mask）的线程才执行后续代码块。
- **场景**：在需要将并行线程分为“活动”与“非活动”两组，分别执行不同代码路径时，使用 `inbranch` 来标记满足条件的分支。
- **性能提升**：无
