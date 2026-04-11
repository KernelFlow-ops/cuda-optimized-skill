# NVIDIA CUTLASS Examples 全量优化技巧解析与汇总

> 基于 CUTLASS 4.x（截至 v4.3.5）的 `examples/` 目录中所有编号示例，逐一分析其使用的优化技巧，最后进行跨 example 的综合汇总。

---

## 第一部分：逐 Example 优化技巧解析

### 基础 GEMM 与工具类（Example 00–06）

| Example | 名称 | 描述 | 核心优化技巧 |
|---------|------|------|-------------|
| **00** | basic_gemm | 单精度 SGEMM，列优先布局 | ① 分层分块（Threadblock/Warp/Thread Tile）<br>② SIMT 核心运算<br>③ 线性组合 Epilogue（\(\alpha AB + \beta C\)）|
| **01** | cutlass_utilities | CUTLASS 工具函数演示 | 张量分配/初始化/IO 工具，无特殊算子优化 |
| **02** | dump_reg_shmem | 调试：打印寄存器和共享内存内容 | 诊断工具，用于验证数据布局和寄存器分配正确性 |
| **03** | visualize_layout | 可视化布局函数 | 布局理解工具，帮助分析数据在内存中的排列 |
| **04** | tile_iterator | 内存中 tile 迭代器 | ① Tile Iterator 抽象<br>② 合并访存（Coalesced Access）模式 |
| **05** | batched_gemm | 批量步进 GEMM | ① Batched Strided GEMM（多问题共享同一 kernel launch）<br>② 通过 Z 维度 grid 索引实现 batch 并行 |
| **06** | splitK_gemm | Split-K 并行归约 | ① **Split-K 并行归约**：沿 K 维度将工作划分到多个 CTA<br>② 两阶段执行（partitioned-K GEMM + batched reduction）<br>③ 适用于 Tall-Skinny 问题（M、N 小，K 大）|

---

### Tensor Core 架构专用示例（Example 07–08）

| Example | 名称 | 描述 | 核心优化技巧 |
|---------|------|------|-------------|
| **07** | volta_tensorop_gemm | Volta Tensor Core 混合精度 GEMM | ① **Tensor Core**（`wmma` 指令）<br>② **混合精度**：FP16 输入 / FP32 累加<br>③ Warp 级 MMA 分块 |
| **08** | turing_tensorop_gemm | Turing Tensor Core 整数 GEMM | ① **INT8 Tensor Core**（`mma.sync`）<br>② **INT8 输入 / INT32 累加**<br>③ 更大的 MMA 指令形状（如 8×8×16）|

---

### 卷积示例（Example 09, 16–25）

| Example | 名称 | 核心优化技巧 |
|---------|------|-------------|
| **09** | turing_tensorop_conv2dfprop | ① **Implicit GEMM 卷积**：将 conv 重构为 GEMM<br>② Turing INT8 Tensor Core<br>③ 隐式 Im2Col 迭代器（无需显式构造中间矩阵）|
| **16** | ampere_tensorop_conv2dfprop | ① Ampere Tensor Core（TF32/FP16）<br>② Implicit GEMM with Optimized 迭代器<br>③ 多阶段流水线（`cp.async`）|
| **17** | conv2d_dgrad (wgrad with split-K) | ① 反向传播梯度计算<br>② **Split-K + 卷积 wgrad 融合**<br>③ 跨步 DGRAD 卷积 |
| **18** | ampere_transposed_conv | ① 转置卷积（反卷积）<br>② 利用 DGRAD kernel 实现转置卷积 |
| **19** | group_conv | ① **分组卷积**<br>② 分析迭代器（Analytical Iterator）<br>③ 两种模式：kSingleGroup / kMultipleGroup |
| **20** | depthwise_conv | ① **深度可分离卷积**<br>② SIMT 专用路径<br>③ Input Channel = Output Channel = Group Number |
| **22** | ampere_tensorop_conv2dfprop (int) | ① Ampere INT8 Tensor Core 卷积<br>② 与 example 16 类似，但整数精度 |
| **25** | ampere_3x_conv_fprop | ① **CUTLASS 3.x API** 风格卷积<br>② CuTe Layout 抽象<br>③ 支持 Affine 和 Gather/Scatter 张量<br>④ 复用 3.x Collective 组件 |

---

### 数据布局与复数运算（Example 10–11）

| Example | 名称 | 核心优化技巧 |
|---------|------|-------------|
| **10** | planar_complex | ① **平面复数 GEMM**：实部虚部分开存储<br>② 四次实数 GEMM 组合实现复数乘法<br>③ 特殊布局映射（Planar Complex Layout）|
| **11** | planar_complex_array | ① 平面复数 + **batch 特定问题尺寸**<br>② Grouped GEMM 雏形（每个 batch 可有不同尺寸）|

---

### Epilogue 融合与算子融合（Example 12–13, 35, 49）

| Example | 名称 | 核心优化技巧 |
|---------|------|-------------|
| **12** | gemm_bias_relu | ① **GEMM + Bias + ReLU Epilogue 融合**<br>② `LinearCombinationRelu` Epilogue operator<br>③ 避免额外的全局内存读写往返 |
| **13** | two_tensor_op_fusion | ① **两个 GEMM（或卷积）融合在单个 kernel 中**<br>② 第一个 GEMM 的输出暂存在共享内存中<br>③ 直接馈入第二个 GEMM，无需写回全局内存<br>④ **Back-to-back GEMM 融合**<br>⑤ Turing 上支持将第一个卷积的累加器暂存在 shared memory |
| **35** | gemm_softmax | ① **GEMM + Softmax Epilogue 融合**<br>② Epilogue Visitor 模式（自定义 epilogue 回调）<br>③ 将所有归约计算融合到前一个 GEMM<br>④ `ThreadblockSwizzle` + `EpilogueVisitor` |
| **36** | gemm_layernorm | ① **GEMM + LayerNorm 融合**<br>② 将 layernorm 拆为两部分，分别融合到前后两个 GEMM<br>③ Shift-K 方差计算（解决平方和数值问题）|
| **37** | gemm_layernorm_epilogue_permutation | ① **GEMM + Epilogue 排列融合**<br>② 在 epilogue 中应用用户定义的排列布局映射 |
| **49** | hopper_gemm_with_collective_builder（EVT） | ① **Epilogue Visitor Tree (EVT)**<br>② CUTLASS 3.x API 的自定义 epilogue 融合<br>③ 支持 Sm90AuxLoad/Sm90SrcFetch<br>④ 用户无需手写 epilogue 即可组合复杂融合模式 |

---

### Ampere 架构专用（Example 14–15, 24）

| Example | 名称 | 核心优化技巧 |
|---------|------|-------------|
| **14** | ampere_tf32_tensorop_gemm | ① **TF32（TensorFloat-32）隐式转换**<br>② FP32 输入自动截断为 TF32 送入 Tensor Core<br>③ 相比纯 FP32 SIMT 大幅提速 |
| **15** | ampere_sparse_tensorop_gemm | ① **结构化稀疏（2:4 Sparse）Tensor Core**<br>② Ampere A100 特有的稀疏 MMA 指令<br>③ 2:4 稀疏模式：每 4 个元素中有 2 个非零<br>④ 理论上 2× Tensor Core 吞吐量 |
| **24** | ampere_gemm_operand_reduction | ① **GEMM 操作数沿 K 维度归约**<br>② 在计算 GEMM 的同时顺带归约一个操作数 |

---

### Ampere 多阶段流水线（Example 内嵌在 16+ 中）

Ampere 架构示例（如 14、16 等）普遍使用的优化：

- **`cp.async` 异步拷贝**：Global → Shared Memory 异步搬运
- **多阶段（Multistage）流水线**：3~7 级共享内存缓冲，比双缓冲更深
- **软件流水线**：MMA 计算与内存加载在同一 warp 中交叠执行

---

### Ada / Hopper GEMM 核心示例（Example 41–67）

| Example | 名称 | 核心优化技巧 |
|---------|------|-------------|
| **41** | ada_fp8_gemm | ① **FP8 Tensor Core**（Ada Lovelace）<br>② CUTLASS 2.x API 的 FP8 路径<br>③ E4M3/E5M2 数据格式 |
| **44** | multi_gemm_s8_tensor_op（Syrk/Herk） | ① 对称矩阵运算<br>② 利用对称性减少一半计算量 |
| **45** | dual_gemm | ① **双 GEMM**：两个 GEMM 共享同一个 A 矩阵<br>② 只加载 A 一次，复用数据 |
| **47** | ampere_gemm_universal_streamk | ① **Stream-K 负载均衡**<br>② 将 K 维度的工作连续流式分配给各 CTA<br>③ 解决问题尺寸不整除 tile 尺寸时的负载不均<br>④ 对非正则几何形状性能提升显著 |
| **48** | hopper_warp_specialized_gemm | ① **Warp 特化**（Hopper SM90）<br>② Producer/Consumer Warp Group 分离<br>③ **TMA（Tensor Memory Accelerator）异步加载**<br>④ `wgmma` 指令（Warp Group MMA）<br>⑤ 异步 Pipeline 类协调生产者/消费者 |
| **49** | hopper_gemm_collective_builder | ① **CollectiveBuilder API**（CUTLASS 3.x）<br>② EVT Epilogue 融合<br>③ 多种调度策略可选（Auto/Cooperative/Pingpong）|
| **50** | hopper_gemm_with_epilogue_swizzle | ① Hopper TMA + **CTA Swizzle**<br>② Epilogue 中的 swizzle 模式<br>③ 提升 L2 缓存命中率 |
| **51** | hopper_fp8_gemm | ① **Hopper FP8 GEMM**<br>② TMA + WGMMA + Threadblock Cluster<br>③ **FP8 快速累加模式**（速度 vs 精度权衡）<br>④ Warp 特化持久化 kernel |
| **52** | hopper_gemm_grouped | ① **Grouped GEMM**（每组问题尺寸不同）<br>② Hopper TMA + Warp 特化<br>③ Tile Scheduler 处理异构 batch |
| **53** | hopper_gemm_permute | ① **GEMM + 输出排列融合**<br>② 在 epilogue 中执行张量维度重排 |
| **54** | hopper_fp8_gemm_sparse | ① **FP8 + 结构化稀疏**（Hopper）<br>② 2:4 稀疏 + FP8 数据类型组合<br>③ 极致吞吐量（稀疏 2× + FP8 2×）|
| **55** | hopper_mixed_dtype_gemm | ① **混合数据类型 GEMM**（A、B 类型不同）<br>② 窄类型通过寄存器文件传输并上转换<br>③ TMA Warp 特化（Cooperative/Pingpong）<br>④ 支持 fp8×int4 等组合<br>⑤ **数据重排优化**（窄类型 tensor reorder）|
| **56** | hopper_ptr_array_batched_gemm | ① **指针数组 Batched GEMM**<br>② 每个 batch 可有不同内存地址<br>③ Hopper TMA + Warp 特化 |
| **57** | hopper_gemm_streamk | ① **Hopper Stream-K**<br>② Persistent Tile Scheduler + StreamK 负载均衡<br>③ CTA Rasterization |
| **58** | ada_grouped_gemm | ① Ada 架构 Grouped GEMM |
| **59** | hopper_gemm_with_topK_softmax | ① **GEMM + Top-K + Softmax Epilogue 融合**<br>② EVT 构建复杂融合图<br>③ 单 kernel 完成 GEMM→TopK→Softmax |

---

### Distributed / 多 GPU（Example 62–65）

| Example | 名称 | 核心优化技巧 |
|---------|------|-------------|
| **62–65** | distributed_gemm | ① **多 GPU 分布式 GEMM**<br>② 跨节点通信与计算重叠<br>③ NCCL / 自定义归约 |

---

### DeepGEMM / 块级缩放（Example 67–68）

| Example | 名称 | 核心优化技巧 |
|---------|------|-------------|
| **67** | hopper_fp8_warp_specialized_gemm_with_blockwise_scaling | ① **Block-wise Scaling**（按块缩放因子）<br>② FP8 + Per-block Scale Factor<br>③ 对齐 DeepSeek 实现的 256×128 tile<br>④ Warp 特化持久化 kernel |
| **68** | hopper_fp8_gemm_with_groupwise_scaling | ① **Group-wise Scaling**<br>② 更细粒度的量化缩放<br>③ 适用于 LLM 推理的量化 GEMM |

---

### Blackwell 架构示例（Example 70–93）

| Example | 名称 | 核心优化技巧 |
|---------|------|-------------|
| **70** | blackwell_gemm | ① **SM100 Tensor Core**（`tcgen05.mma`）<br>② **Tensor Memory (TMEM)**：每 SM 专属内存<br>③ 扩展 Warp 特化（MMA 与 Epilogue 分离到不同 warp）<br>④ CUTLASS 3.x CollectiveBuilder |
| **71** | blackwell_gemm_with_epilogue_evt | ① Blackwell EVT<br>② 兼容 mainloop + epilogue builder 调度策略<br>③ 自定义 epilogue 融合（bias、activation 等）|
| **72** | blackwell_narrow_precision_gemm | ① **Block-Scaled NVFP4**（4 位浮点）<br>② `tcgen05.mma.blockscaled` 指令<br>③ 2× FP8 吞吐量 / 4× WGMMA 吞吐量<br>④ OCP MXFP4/MXFP6/MXFP8 支持 |
| **73** | blackwell_cluster_gemm | ① **Preferred Cluster 特性**<br>② Threadblock Cluster 优化分配<br>③ 跨 SM 共享数据（DSMEM Multicast）|
| **74** | blackwell_conv | ① Blackwell 卷积（fprop/dgrad/wgrad）<br>② SM100 Tensor Core + Implicit GEMM<br>③ 支持 Stream-K 卷积 |
| **75** | blackwell_gemm_streamk | ① **Blackwell Stream-K**<br>② SM100 持久化 Tile Scheduler<br>③ 负载均衡 + CTA Rasterization |
| **76** | blackwell_sparse_gemm | ① **Blackwell 结构化稀疏 GEMM**<br>② SM100 稀疏 MMA 指令<br>③ 稀疏压缩器（Compressor）|
| **77** | blackwell_fmha / blackwell_mla | ① **Fused Multi-Head Attention (FMHA)**<br>② **Multi-Head Latent Attention (MLA)**<br>③ Warp 特化 + Pipeline 协调<br>④ TMA 异步加载 Q/K/V<br>⑤ 在线 Softmax（不需要完整 attention matrix）<br>⑥ Cluster 级归约<br>⑦ GQA 支持<br>⑧ Softmax skip correction |
| **78** | blackwell_gemm_with_broadcast | ① GEMM + 广播（broadcast）操作融合 |
| **79** | blackwell_gemm_mixed_input | ① Blackwell 混合输入 GEMM |
| **80–82** | blackwell_grouped_gemm 系列 | ① Blackwell Grouped GEMM<br>② 块级缩放分组 GEMM<br>③ TMA 3D Load<br>④ 异步 TMA Descriptor 更新 |
| **83–85** | blackwell_gemm_blockscaled 系列 | ① 块级缩放 GEMM 变体<br>② 稀疏 + 块缩放组合 |
| **86** | blackwell_mixed_dtype_gemm | ① **Blackwell 混合数据类型 GEMM**<br>② Legacy mixed input mainloop |
| **87** | blackwell_geforce_gemm_blockwise | ① **SM120（GeForce RTX 50 系列）块级 GEMM**<br>② 与 SM100 datacenter GPU 不同的计算能力 |
| **89** | sm103_fp4_ultra_gemm | ① **SM103（B300）Ultra FP4 GEMM**<br>② Block-scaled 数据类型 |
| **90** | sm103_fp4_ultra_grouped_gemm | ① SM103 Ultra FP4 分组 GEMM |
| **91** | fp4_gemv | ① **FP4 GEMV**（矩阵向量乘）<br>② 块级缩放 GEMV kernel |
| **92** | blackwell_moe_gemm | ① **MoE（Mixture of Experts）GEMM**<br>② TMA 3D Load（weights）+ CPASYNC（tokens）<br>③ 仅一个问题维度跨 group 变化<br>④ Ragged Contiguous Grouped GEMM<br>⑤ TensorMap 异步更新<br>⑥ 低延迟推理优化 |
| **93** | blackwell_low_latency_gqa | ① **低延迟 GQA（Grouped Query Attention）**<br>② Flash Decoding + Cluster 归约<br>③ 生成阶段专用 kernel |

---

### CuTe / Python DSL 示例

| Example 目录 | 核心优化技巧 |
|-------------|-------------|
| `examples/cute/` | CuTe Layout 代数、Tensor 操作、MMA Atom 原语教学 |
| `examples/python/CuTeDSL/ampere/` | Ampere elementwise、HSTU Attention |
| `examples/python/CuTeDSL/hopper/` | Hopper 持久化 Dense GEMM（静态调度）|
| `examples/python/CuTeDSL/blackwell/` | ① Blackwell Dense GEMM Persistent<br>② Mixed-input GEMM<br>③ Blockwise GEMM（含 Contiguous Grouped / Masked Grouped）<br>④ FMHA Backward<br>⑤ MLA<br>⑥ **Programmatic Dependent Launch (PDL)**<br>⑦ Pipeline Producer/Consumer API |

---

## 第二部分：优化技巧全量汇总与交叉分析

### 一、按优化维度分类

#### 1. 数据搬运优化

| 技巧 | 说明 | 使用 Examples |
|------|------|-------------|
| `cp.async` 异步拷贝 | Global→Shared 异步搬运，不阻塞 warp | 14, 16, 22+（所有 Ampere+）|
| TMA（Tensor Memory Accelerator） | 硬件加速的 Global→Shared 搬运，自动计算地址和 swizzle | 48–59, 67–93（所有 Hopper/Blackwell）|
| CPASYNC（与 TMA 组合） | 用于 token 加载的异步拷贝 | 92 (MoE) |
| TMA 3D Load | 三维张量的 TMA 加载 | 80–82, 92 |
| TMA Multicast | 数据多播到 Cluster 内多个 SM | 48+, 73, 77 |
| 异步 TMA Descriptor 更新 | 运行时更新 TMA 描述符 | 92 |

#### 2. 计算优化

| 技巧 | 说明 | 使用 Examples |
|------|------|-------------|
| SIMT Core | 标量浮点运算 | 00 |
| `wmma` Tensor Core（Volta） | 16×16×16 FP16 MMA | 07 |
| `mma.sync` Tensor Core（Turing/Ampere） | 高吞吐量矩阵乘加 | 08, 14–16, 22 |
| `wgmma` Warp Group MMA（Hopper） | Warp Group 级 MMA | 48–59, 67–68 |
| `tcgen05.mma`（Blackwell） | 第 5 代 Tensor Core | 70–93 |
| `tcgen05.mma.blockscaled` | Block-scaled MMA（2× fp8 吞吐） | 72, 83–85, 89–90 |
| 结构化稀疏 2:4 MMA | 稀疏 Tensor Core | 15, 54, 76 |
| TF32 隐式转换 | FP32→TF32 自动截断 | 14 |
| 混合精度（FP16/BF16/FP8/INT8/INT4/FP4） | 低精度输入、高精度累加 | 07, 08, 41, 51, 55, 67–68, 72, 89–91 |
| FP8 快速累加 | 跳过中间精度提升 | 51, 67 |

#### 3. 并行度与调度优化

| 技巧 | 说明 | 使用 Examples |
|------|------|-------------|
| Split-K 并行归约 | K 维度分到多个 CTA | 06, 17 |
| Stream-K 负载均衡 | K 维度工作流式分配 | 47, 57, 75 |
| Threadblock Cluster | 多 SM 组成集群共享 DSMEM | 48+, 73, 77, 92 |
| Persistent Kernel（持久化） | CTA 驻留 SM 处理多个 tile | 48+, 51, 57, 67, 70+ |
| CTA Swizzle / Rasterization | 线程块→问题分区的空间局部映射 | 35, 47, 50, 57, 75 |
| Preferred Cluster | 优化 cluster 分配策略 | 73 |
| Tile Scheduler | 统一的 tile 分配调度框架 | 49+, 52, 57, 70+ |
| Partial SM Allocation | 部分 SM 分配执行 GEMM | 4.x 特性 |

#### 4. 流水线优化

| 技巧 | 说明 | 使用 Examples |
|------|------|-------------|
| 双缓冲（Double Buffering） | Shared Memory + Register 两级双缓冲 | 00–08（所有 pre-Ampere）|
| 多阶段流水线（Multistage） | N 级 SMEM 缓冲（N=3~7） | 14, 16, 22+（Ampere+）|
| Warp 特化（Producer/Consumer） | 数据搬运 warp 与计算 warp 分离 | 48–59, 67–93（Hopper/Blackwell）|
| Ping-Pong 设计 | 两个 Consumer 交替 MMA/Epilogue | 48, 49, 51, 55, 70+ |
| Cooperative 设计 | 两个 Consumer 协作同一 tile | 48, 49, 51 |
| 异步 Pipeline 类 | 屏障同步的多级环形缓冲 | 48+, 所有 Hopper/Blackwell |
| Warp Group 寄存器重分配 | Producer 减寄存器 / Consumer 增寄存器 | 48+, Hopper/Blackwell |
| Programmatic Dependent Launch (PDL) | Kernel 间依赖调度 | Python DSL Blackwell 示例 |

#### 5. 存储层次优化

| 技巧 | 说明 | 使用 Examples |
|------|------|-------------|
| Shared Memory Swizzle | XOR 地址变换消除 bank conflict | 所有使用 SMEM 的示例 |
| Tensor Memory (TMEM) | Blackwell 每 SM 专属内存 | 70–93 |
| DSMEM（分布式共享内存） | Cluster 内跨 SM 共享 SMEM | 48+, 73, 77 |
| L2 缓存 Swizzle | CTA 映射优化 L2 命中率 | 35, 47, 50, 57 |
| L2 缓存驱逐优先级 | 精细粒度 L2 缓存控制 | Python DSL 4.3.0+ |
| 合并访存（Epilogue） | Epilogue 通过 SMEM 中转实现合并写回 | 所有示例 |
| 向量化内存操作 | 对齐保证下使用 128 位 LDG/STG | 通过 Alignment 模板参数控制 |

#### 6. Epilogue 融合

| 技巧 | 说明 | 使用 Examples |
|------|------|-------------|
| Linear Combination | \(\alpha AB + \beta C\) | 00, 05, 06, 07, 08 |
| + Bias + ReLU | 融合偏置加和激活 | 12 |
| + GELU/SiLU/Sigmoid/HardSwish/LeakyReLU | 各种激活函数融合 | 多个（通过 Epilogue 模板参数）|
| + Softmax | 融合归约操作 | 35, 59 |
| + LayerNorm | 拆分融合到前后 GEMM | 36 |
| + Permutation | 输出排列融合 | 37, 53 |
| + Top-K | 选择 Top-K 元素 | 59 |
| + Broadcast | 广播操作融合 | 78 |
| Epilogue Visitor Tree (EVT) | 用户自定义计算图融合 | 49, 59, 71 |
| Back-to-back GEMM | 两个 GEMM 通过 SMEM 串联 | 13 |

#### 7. 问题形态适配

| 技巧 | 说明 | 使用 Examples |
|------|------|-------------|
| Batched Strided GEMM | 固定步长的批量 GEMM | 05 |
| Ptr-Array Batched GEMM | 指针数组的批量 GEMM | 56 |
| Grouped GEMM | 每组不同问题尺寸 | 52, 58, 80–82 |
| MoE GEMM | Mixture of Experts 专用分组 | 92 |
| Implicit GEMM Convolution | 卷积→GEMM 映射 | 09, 16–22, 25, 74 |
| 结构化稀疏 GEMM | 2:4 稀疏模式 | 15, 54, 76 |
| Block-Scaled GEMM | 逐块量化缩放 | 67–68, 72, 83–85, 87, 89–90 |
| GEMV（矩阵向量乘） | 向量级运算 | 91 |
| FMHA / MLA | 融合多头注意力 | 77, 93 |

---

### 二、按 GPU 架构演进的技巧图谱

```
Volta (SM70)
 └─ wmma Tensor Core, 混合精度 (FP16→FP32)
    └─ Example: 07

Turing (SM75)
 └─ mma.sync, INT8 Tensor Core, Implicit GEMM Conv
    └─ Examples: 08, 09

Ampere (SM80)
 └─ cp.async 异步拷贝
 └─ 多阶段流水线 (Multistage)
 └─ TF32 隐式转换
 └─ 2:4 结构化稀疏
 └─ Mainloop 融合卷积
    └─ Examples: 14, 15, 16, 22, 24, 25, 47

Ada (SM89)
 └─ FP8 Tensor Core (E4M3, E5M2)
    └─ Examples: 41, 58

Hopper (SM90)
 └─ TMA (Tensor Memory Accelerator)
 └─ wgmma (Warp Group MMA)
 └─ Warp 特化 (Producer/Consumer)
 └─ Threadblock Cluster + DSMEM
 └─ Persistent Kernel
 └─ Ping-Pong / Cooperative 调度
 └─ Stream-K
 └─ Warp Group 寄存器重分配
 └─ 异步 Pipeline 类
 └─ FP8 + Sparse
 └─ Mixed Dtype
 └─ Block/Group-wise Scaling
 └─ EVT Epilogue
    └─ Examples: 48–59, 67–68

Blackwell SM100 (Datacenter)
 └─ tcgen05.mma + tcgen05.mma.blockscaled
 └─ Tensor Memory (TMEM)
 └─ 扩展 Warp 特化 (MMA/Epilogue 分 warp)
 └─ NVFP4, MXFP4/6/8 块级缩放
 └─ Preferred Cluster
 └─ FMHA/MLA with Cluster Reduction
 └─ MoE Low-Latency GEMM
 └─ Sparse Compressor
    └─ Examples: 70–93

Blackwell SM103 (B300)
 └─ Ultra FP4 block-scaled
    └─ Examples: 89, 90

Blackwell SM120 (GeForce RTX 50)
 └─ 块级 GEMM (与 SM100 不同的计算能力)
    └─ Example: 87
```

---

### 三、优化技巧的关键组合模式

在高性能 kernel 中，多种技巧通常**组合使用**才能达到峰值性能：

**模式 A：Hopper 峰值 GEMM（如 Example 51）**
```
TMA 异步加载 → 多级 Pipeline → Warp 特化 (Ping-Pong)
→ wgmma Tensor Core → 寄存器重分配
→ Persistent Kernel → CTA Swizzle
→ EVT Epilogue 融合
```

**模式 B：Blackwell 块级缩放 GEMM（如 Example 72）**
```
TMA → TMEM → tcgen05.mma.blockscaled
→ 扩展 Warp 特化 → Preferred Cluster
→ Block-Scaled FP4/FP8 → Stream-K
```

**模式 C：融合注意力（如 Example 77）**
```
TMA 加载 Q/K/V → Warp 特化 Pipeline
→ Back-to-back GEMM (Q×K^T → ×V)
→ 在线 Softmax (tiling-friendly)
→ Cluster 级归约 → Flash Decoding
```

**模式 D：MoE 低延迟推理（如 Example 92）**
```
TMA 3D Load (权重) + CPASYNC (tokens)
→ Ragged Contiguous Grouped GEMM
→ 异步 TMA Descriptor 更新
→ MoEProblemShape 简化 API
```

---

### 四、性能影响力排序（定性估计）

根据在 CUTLASS 中的实际性能贡献，各优化技巧的影响力大致排序如下：

1. **Tensor Core 利用**（从 SIMT→Tensor Core 可带来 10×+ 提升）
2. **流水线深度与 Warp 特化**（从无流水线到多级 Ping-Pong 可提升 2–3×）
3. **数据精度降低**（FP32→FP16→FP8→FP4 每降一级约 2× 吞吐）
4. **结构化稀疏**（2:4 稀疏额外 2× 吞吐，但需要模型适配）
5. **Epilogue 融合**（消除全局内存往返，bandwidth-bound 场景提升 30–50%+）
6. **CTA Swizzle / Stream-K**（减少 L2 miss 和负载不均，提升 10–30%）
7. **Shared Memory Swizzle**（消除 bank conflict，提升 10–20%）
8. **Threadblock Cluster + DSMEM**（减少冗余加载，提升 5–15%）
9. **持久化 Kernel**（减少 kernel launch 开销，提升 5–10%）
10. **自动调优（nvMatmulHeuristics）**（找到最优配置，vs 随机配置可差 2–3×）

---

### 五、总结

CUTLASS 的 90+ 个示例构成了一个从入门到极致优化的完整技术谱系。核心认知可归纳为：

1. **分层分块是基础**：所有性能优化都建立在 Grid→CTA→Warp→Thread 四层分块之上。

2. **"喂饱 Tensor Core"是目标**：几乎所有高级优化（流水线、Warp 特化、TMA、TMEM、Cluster）的本质都是确保 Tensor Core 不会因数据等待而空闲。

3. **Epilogue 融合是必备**：现代深度学习工作负载中，单独的 GEMM 很少出现；与激活函数、归一化、attention 等操作的融合是实际部署的关键。

4. **架构特性驱动优化演进**：每一代 GPU 架构引入新的硬件特性（cp.async → TMA → TMEM），CUTLASS 通过新的 Example 展示如何利用这些特性。

5. **没有万能配置**：不同的问题形状（大/小、方/长、密/稀）需要不同的 tile 尺寸、调度策略和 split-K 参数，自动调优工具（如 nvMatmulHeuristics）至关重要。


## 详细文档请看下面链接
https://docs.nvidia.com/cutlass/latest/
