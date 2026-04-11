CUDA内存访问优化是一个很深的话题，下面我尽量全面地梳理所有主要手段。

---

## 一、全局内存（Global Memory）访问优化

**1. 合并访问（Coalesced Access）**
同一warp的32个线程访问连续、对齐的地址，硬件合并为最少的内存事务。理想情况下一次128B事务服务整个warp。反例是stride访问或随机访问，会导致事务数暴增。

**2. 对齐访问（Aligned Access）**
访问的起始地址对齐到128B（或32B的sector）边界。未对齐会浪费带宽，因为硬件以sector为单位读取，不对齐意味着额外sector被拉入但无用。

**3. 向量化访存（Vectorized Load/Store）**
使用 `float2`、`float4`、`int4`、`double2` 等宽类型，单条指令读写128-bit。好处是减少指令数、提升每事务有效字节数。需要保证地址对齐到向量宽度。

**4. 只读数据路径**

* `__ldg()` 内建函数：显式走只读数据缓存（L1 texture缓存），绕过L1常规路径，对不规则访问模式有更好的缓存行为。
* `const __restrict__` 指针修饰：告诉编译器该指针所指数据不会被写入且无别名，编译器自动选择 `__ldg()` 路径。

**5. L2缓存优化（Compute Capability 8.0+）**

* **L2 Persistence**：使用 `cudaAccessPolicyWindow` API将热点数据"钉"在L2缓存中，防止被冷数据驱逐。适合数据量小但反复访问的场景（如lookup table、embedding）。
* **L2 访问属性**：`cudaAccessProperty::cudaAccessPropertyPersisting` vs `Streaming`，精细控制不同数据的L2驻留策略。

**6. Sector化理解**
从Volta架构开始，L1缓存以32B sector为粒度工作（而非传统128B cache line整体fetch）。这意味着不连续访问的代价比Kepler/Maxwell时代降低了，但连续访问仍然最优。

---

## 二、共享内存（Shared Memory）优化

**7. Tiling（分块）**
将全局内存中的数据按tile搬到Shared Memory，在片上多次复用。矩阵乘法的经典优化——每个tile只从全局内存读一次，在Shared Memory中被复用 (O(N)) 次。

**8. Bank Conflict消除**
Shared Memory被划分为32个bank，每个bank宽4B（或在某些模式下8B）。同一warp中多个线程访问同一bank的不同地址会串行化。解决方式：

* **Padding**：给二维数组每行末尾加一个元素，如 `__shared__ float s[32][33]`，错开bank映射。
* **Swizzle/XOR索引**：用异或操作重映射索引，是更高级且不浪费空间的方案。

**9. 异步拷贝（Async Copy）**

* `cp.async`（CUDA 11+）：Global → Shared Memory的搬运由硬件DMA完成，不占用寄存器和计算单元。可以与计算完全重叠。
* `cuda::memcpy_async`（C++ API）：语义更清晰的封装。
* **多级流水线（Multi-stage Pipeline）**：分配多个Shared Memory buffer，一个在加载数据，一个在计算，实现软件流水线，极大隐藏全局内存延迟。

**10. Shared Memory容量配置**
使用 `cudaFuncSetAttribute` 将L1/Shared Memory的比例调向Shared Memory一侧（如从48KB提升到100KB+），适合Shared Memory需求大的kernel。Ampere架构Shared Memory最大可达164KB。

---

## 三、常量内存（Constant Memory）

**11. `__constant__` 内存**
总共64KB，硬件有专用缓存。当warp内所有线程读同一个地址时效率最高（广播机制，一次读取服务32个线程）。适合存储kernel参数、卷积核权重等小型只读数据。如果同一warp内线程读不同地址，则串行化，性能反而差。

---

## 四、纹理内存（Texture Memory）

**12. Texture / Surface 对象**

* 硬件针对2D空间局部性优化缓存（Morton/Z-order布局），适合图像处理、模板计算等二维访问模式。
* 自动处理边界条件（clamp、wrap模式），免去手写边界判断。
* 硬件插值（线性、双线性），对图像/体素采样类任务可直接利用。
* 现代GPU上 `__ldg()` 在大多数一维场景可以替代texture，但二维空间局部性场景texture仍有优势。

---

## 五、寄存器（Register）层面优化

**13. 寄存器复用与压力控制**

* 寄存器是最快的存储（零延迟），但每个SM总量有限（如65536个32-bit寄存器）。
* 寄存器用量过高 → 每个SM能驻留的warp数减少 → occupancy降低 → 延迟隐藏能力下降。
* 用 `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 引导编译器控制寄存器分配。
* 过度溢出到Local Memory（即register spill）会退化为全局内存访问，极其昂贵。

**14. 寄存器级数据复用**
让每个线程持有多个数据元素（增加ILP），在寄存器层面完成尽可能多的计算后再写回，减少对Shared/Global Memory的依赖。

---

## 六、数据布局优化

**15. SoA vs AoS**

* **AoS（Array of Structures）**：`struct { float x, y, z; } particles[N]`——同一粒子的字段连续存放。warp访问同一字段时地址不连续，无法合并。
* **SoA（Structure of Arrays）**：`float x[N], y[N], z[N]`——同一字段的所有粒子连续存放。warp访问时天然合并。
* **CUDA中几乎总是优选SoA**，差距可以达到数倍。

**16. Padding与对齐**
对二维数组的行宽做padding（`cudaMallocPitch` / `cudaMalloc3D`），保证每一行的起始地址对齐到合并访问边界。虽然浪费少量空间，但带宽利用率大幅提升。

**17. 数据重排（Data Reordering）**
对于不规则访问模式（如稀疏矩阵、图计算），通过预处理阶段对数据做重排：

* 空间填充曲线（Z-order / Hilbert curve）排列，提升缓存局部性。
* 对CSR稀疏矩阵按行长度分桶，让同一warp处理长度相近的行，减少负载不均和分支发散。

---

## 七、访存与计算重叠

**18. 双缓冲 / 多级流水线**
在Shared Memory或寄存器中分配两组buffer：

* 阶段1：buffer A 计算，buffer B 加载下一批数据
* 阶段2：buffer B 计算，buffer A 加载下一批数据

这是所有高性能GEMM实现（如cuBLAS、CUTLASS）的核心技术。

**19. CUDA Streams重叠**

* 不同stream之间的kernel执行、H2D拷贝、D2H拷贝可以并行。
* 把大数据分chunk，在多个stream中做"拷贝-计算-回拷"流水线。

**20. Prefetch**

* 对Global Memory使用 `__builtin_prefetch` 或手动用额外线程提前加载下一次迭代的数据。
* Unified Memory场景下用 `cudaMemPrefetchAsync` 显式触发页迁移，避免按需缺页的高延迟。

---

## 八、Unified Memory / 零拷贝

**21. Unified Memory优化**

* 默认的按需页迁移（on-demand paging）延迟很高。
* 用 `cudaMemPrefetchAsync` 提前迁移到目标设备。
* 用 `cudaMemAdvise`（如 `cudaMemAdviseSetReadMostly`、`cudaMemAdviseSetPreferredLocation`）给驱动提供访问模式提示。

**22. 零拷贝内存（Zero-Copy / Pinned Mapped Memory）**

* `cudaHostAlloc` + `cudaHostAllocMapped`：GPU直接通过PCIe访问主机内存，免去显式拷贝。
* 只适合数据量小或只访问一次的场景，否则PCIe带宽（~32GB/s）远低于显存带宽（~900GB/s+）。

**23. Pinned Memory（Page-locked）**

* `cudaHostAlloc` 或 `cudaMallocHost` 分配的锁页内存，H2D/D2H传输带宽比可分页内存高得多（可达2倍），且支持异步传输。

---

## 九、减少不必要的内存访问

**24. Kernel Fusion**
把相邻的producer-consumer kernel合并为一个，中间结果保留在寄存器或Shared Memory中，消除一次完整的Global Memory往返。这往往是最高收益的单项优化。

**25. In-place操作**
尽可能原地修改数据（如 element-wise 操作），避免分配额外的输出buffer，减少内存占用和访问量。

**26. Warp Shuffle（`__shfl_sync` 系列）**
warp内线程直接交换寄存器值，无需经过Shared Memory。适用于：

* Warp级归约（reduction）
* 前缀和（scan）
* 广播一个值给warp内所有线程
* 延迟约为1个时钟周期，比Shared Memory还快且不存在bank conflict问题。

**27. 协作组（Cooperative Groups）**
灵活定义比warp更小或跨block的同步组，精确控制同步粒度，避免不必要的 `__syncthreads()` 全block同步带来的等待开销。

---

## 十、总结：优化决策树

```
内存访问优化
├── 带宽利用率
│   ├── 合并访问 + 对齐
│   ├── 向量化 float4
│   ├── SoA数据布局
│   └── Padding（pitch分配）
├── 减少访问次数
│   ├── Shared Memory tiling
│   ├── Kernel fusion
│   ├── Warp shuffle代替shared memory
│   └── 寄存器级数据复用
├── 利用缓存层级
│   ├── __ldg() / const __restrict__
│   ├── L2 persistence
│   ├── Texture（二维局部性）
│   └── Constant Memory（广播读）
├── 隐藏延迟
│   ├── cp.async异步拷贝
│   ├── 双缓冲/多级流水线
│   ├── Prefetch
│   └── Stream重叠
└── 主机-设备传输
    ├── Pinned memory
    ├── 零拷贝（小数据）
    └── Unified Memory + prefetch/advise
```

每一项优化都应该在Nsight Compute的profiling数据指导下进行，关注Memory Throughput、L1/L2 Hit Rate、Shared Memory Efficiency、Stall Reasons等指标，避免盲目优化。
