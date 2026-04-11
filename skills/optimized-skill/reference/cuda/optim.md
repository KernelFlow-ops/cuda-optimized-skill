# CUDA 算子迭代优化方案（NCU 驱动）

## 〇、前置：环境信息采集

在一切优化开始之前，先确定硬件天花板和 kernel 的当前状态。

```bash
# 1. 获取显卡型号与关键硬件参数
nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.mem --format=csv

# 2. 用 deviceQuery 获取详细参数（SM 数、每 SM 寄存器数、Shared Memory 容量、L2 大小等）
./deviceQuery

# 3. 对目标 kernel 采集完整 NCU 报告
ncu --set full -o report_v0 ./your_binary
```

需要记录的关键硬件参数（后续优化的天花板）：

| 参数 | 含义 | 示例（RTX 4090） |
|---|---|---|
| SM 数量 | 并行单元数 | 128 |
| 峰值显存带宽 | 全局内存天花板 | 1008 GB/s |
| 峰值 FP32 算力 | 计算天花板 | 82.6 TFLOPS |
| 峰值 Tensor Core 算力 | 矩阵运算天花板 | 330 TFLOPS (FP16) |
| L2 缓存大小 | 缓存策略参考 | 72 MB |
| 每 SM Shared Memory 上限 | Tiling 策略参考 | 100 KB（可配至 228 KB） |
| 每 SM 寄存器总数 | Occupancy 计算 | 65536 × 32-bit |

---

## 一、迭代优化总流程

核心思想：**Profile → 定位瓶颈 → 分类施策 → 验证 → 重复**。

```
┌─────────────────────────────────────────────────────┐
│                  编写 / 修改 Kernel                    │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│           NCU Profiling（--set full）                 │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│         第一步：Roofline 定位 —— 计算瓶颈还是带宽瓶颈？  │
│                                                     │
│  Arithmetic Intensity = FLOPs / Bytes               │
│  与硬件 Roofline 交叉点比较：                          │
│    · 落在带宽斜坡 → 内存受限（Memory Bound）            │
│    · 落在算力平顶 → 计算受限（Compute Bound）           │
│    · 两者都远未达到 → 延迟受限（Latency Bound）         │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│    第二步：根据瓶颈类型进入对应优化路径（见下文）          │
│    ——可多个方向同时进行                                │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│         第三步：修改代码，重新 NCU Profiling             │
│         对比 report_vN 与 report_v(N-1)              │
│         确认指标改善、无性能回退                         │
└──────────────────────┬──────────────────────────────┘
                       ▼
              性能达标？──否──→ 回到第一步
                 │
                 是
                 ▼
               完成
```

---

## 二、NCU 关键指标速查与瓶颈判定

### 2.1 一级判定：Roofline 模型

NCU 的 **Speed Of Light (SOL)** 面板直接给出：

| 指标 | 含义 | 判定 |
|---|---|---|
| `SM SOL %` | 计算单元利用率占峰值百分比 | 高 → 计算受限 |
| `Memory SOL %` | 显存带宽利用率占峰值百分比 | 高 → 带宽受限 |
| 两者都低 | — | 延迟受限 |

**判定规则**：
- \(\text{SM SOL} > 60\%\) 且远高于 Memory SOL → **Compute Bound**
- \(\text{Memory SOL} > 60\%\) 且远高于 SM SOL → **Memory Bound**
- 两者均 \(< 40\%\) → **Latency Bound**（通常因 occupancy 过低、同步过多、分支发散等）

### 2.2 二级诊断：细分指标

确定一级瓶颈后，进入对应的二级指标：

#### Memory Bound 时重点查看：

| NCU 指标 | 问题信号 | 对应优化方向 |
|---|---|---|
| `Global Load/Store Efficiency` | < 100% | 未合并访问，检查 SoA/AoS、对齐、stride |
| `L1 Hit Rate` | 过低 | 数据局部性差，考虑 tiling / `__ldg()` |
| `L2 Hit Rate` | 过低 | 工作集超出 L2；考虑 L2 persistence / 减小 tile |
| `Shared Memory Efficiency` | < 100% | 存在 bank conflict |
| `DRAM Throughput` | 接近峰值但 kernel 仍慢 | 已达带宽极限，需减少访存量（fusion / 算法改进） |
| `Sectors/Request` | > 1（理想值为 1） | 未对齐或未合并 |

#### Compute Bound 时重点查看：

| NCU 指标 | 问题信号 | 对应优化方向 |
|---|---|---|
| `Warp Execution Efficiency` | < 100% | 分支发散 |
| `Eligible Warps Per Cycle` | 过低 | ILP 不足或 occupancy 过低 |
| `Issue Slot Utilization` | < 50% | 指令调度不饱和 |
| `FP32/FP16/Tensor Pipe Utilization` | 不均衡 | 未使用合适的计算管线（如该用 Tensor Core 没用） |
| `Register Spill (Local Memory)` | > 0 | 寄存器溢出 |

#### Latency Bound 时重点查看：

| NCU 指标 | 问题信号 | 对应优化方向 |
|---|---|---|
| `Occupancy (Achieved vs Theoretical)` | 差距大 | 寄存器 / Shared Memory 用量过高 |
| `Stall Reasons` 面板 | 高 `Stall Not Selected` / `Stall Barrier` | 同步开销过大 |
| `Stall Long Scoreboard` | 高 | 全局内存延迟未隐藏 |
| `Stall Short Scoreboard` | 高 | Shared Memory / L1 延迟未隐藏 |

---

## 三、分瓶颈优化手册

### 3.1 Memory Bound 优化路径

按优先级排序，**同一轮迭代中可同时执行多项**：

**第一优先级：减少访存量**
1. **Kernel Fusion**：将前后依赖的 kernel 合并，中间结果留在寄存器 / Shared Memory 中，消除一次完整 Global Memory 往返。这通常是收益最大的单项优化。
2. **Warp Shuffle 替代 Shared Memory**：warp 内的归约、scan、广播改用 `__shfl_sync` 系列，免去 "写 Shared → `__syncthreads()` → 读 Shared" 的三步开销。
3. **寄存器级数据复用**：每个线程处理多个元素，在寄存器中完成尽可能多的计算后再写回。

**第二优先级：提升带宽利用率**
4. **合并访问 + SoA 布局**：确保同一 warp 访问连续地址。将 AoS 改为 SoA。
5. **向量化访存**：使用 `float4` / `int4` 等宽类型，单条指令搬运 128-bit。
6. **对齐与 Padding**：用 `cudaMallocPitch` 或手动 padding 保证行首对齐。

**第三优先级：利用缓存层级**
7. **Shared Memory Tiling**：经典分块，从 Global Memory 搬入 Shared Memory 后复用 \(O(N)\) 次。
8. **消除 Bank Conflict**：padding（如 `float s[32][33]`）或 swizzle/XOR 索引。
9. **`__ldg()` / `const __restrict__`**：走只读纹理缓存路径，对不规则访问更友好。
10. **L2 Persistence（CC 8.0+）**：对热点小数据用 `cudaAccessPolicyWindow` 钉在 L2 中。

**第四优先级：隐藏延迟**
11. **`cp.async` + 多级流水线**：Global → Shared 的搬运走硬件 DMA，与计算完全重叠。分配多级 buffer 做软件流水。
12. **双缓冲**：一组 buffer 计算、另一组 buffer 加载，交替进行。
13. **Prefetch / `cudaMemPrefetchAsync`**：Unified Memory 场景下提前触发页迁移。

**第五优先级：主机-设备传输**
14. **Pinned Memory**：`cudaMallocHost` 分配锁页内存，传输带宽提升可达 2×，且支持异步。
15. **Stream 流水线**：多 stream 将大数据分 chunk 做 "H2D → Compute → D2H" 流水线。

### 3.2 Compute Bound 优化路径

**第一优先级：使用正确的硬件单元**
1. **Tensor Core**：矩阵类运算必须用 Tensor Core（通过 WMMA / MMA PTX / cuBLAS / CUTLASS）。不用约等于浪费半数以上算力。
2. **SFU Intrinsics**：超越函数使用 `__sinf`、`__cosf`、`__expf` 等走 SFU 流水线。
3. **快速数学**：`--use_fast_math` 或单独开启 `--ftz=true`、`--prec-div=false`。

**第二优先级：减少指令数**
4. **FMA**：确保 `a*b+c` 被编译为一条 FMA 指令（`__fmaf_rn()`）。
5. **强度削减**：用移位替代 2 的幂乘除、`rsqrtf()` 替代 `1/sqrtf()`、乘法替代除法（`__frcp_rn`）。
6. **查表（LUT）**：复杂函数在输入范围有限时预计算放 Shared / Constant Memory。
7. **位操作 Intrinsics**：`__popc()`、`__clz()`、`__ffs()` 等直接映射单周期硬件指令。

**第三优先级：消除分支发散**
8. **谓词化与 Select**：短分支改为三目运算符或算术表达式，让编译器生成无分支代码。
9. **按 warp 重组数据**：预排序让同类数据聚集到同一 warp，减少 divergence。
10. **Warp Vote Early Exit**：`__all_sync()` / `__any_sync()` 检测后整个 warp 提前退出。

**第四优先级：提升 ILP / 调度效率**
11. **循环展开**：`#pragma unroll`，暴露更多独立指令。
12. **循环分裂 / 合并 / 交换**：根据具体情况选择，减少寄存器压力或提升数据复用。
13. **软件流水线**：循环体内交错不同迭代的加载、计算、存储阶段。

### 3.3 Latency Bound 优化路径

**核心目标：给调度器更多就绪 warp 或更多独立指令。**

1. **提升 Occupancy**：
   - 减少每线程寄存器用量（`__launch_bounds__`、减小展开因子、拆分 kernel）。
   - 减少每 block Shared Memory 用量，或调整 L1/Shared 比例（`cudaFuncSetAttribute`）。
   - 调整 block size（通常 128/256/512 中实测选最优）。

2. **减少同步开销**：
   - 减少不必要的 `__syncthreads()`——如果数据访问模式无跨 warp 依赖则删除。
   - 用 `__syncwarp()` 替代 `__syncthreads()` 缩小同步粒度。
   - 用 Cooperative Groups 定义最小必要同步组。
   - 用 Warp Shuffle 彻底消除 "写 Shared → sync → 读 Shared" 模式。

3. **隐藏全局内存延迟**：
   - `cp.async` + 多级流水线。
   - 增加每线程独立工作量（ILP），让调度器在等待内存时有计算指令可发射。

4. **减少寄存器溢出**：
   - 用 `--ptxas-options=-v` 检查 spill 量。
   - 缩减活跃变量数、减小展开因子、拆分 kernel。

---

## 四、迭代实操模板

每一轮迭代遵循以下步骤：

### 第 N 轮迭代

```
1. Profiling
   ncu --set full -o report_vN ./binary

2. 记录关键指标
   ┌────────────────────────┬──────────┬──────────┐
   │ 指标                    │ v(N-1)   │ vN       │
   ├────────────────────────┼──────────┼──────────┤
   │ Kernel 耗时 (μs)       │          │          │
   │ SM SOL %               │          │          │
   │ Memory SOL %           │          │          │
   │ Achieved Occupancy     │          │          │
   │ Global Load Efficiency │          │          │
   │ Shared Efficiency      │          │          │
   │ Register Spill (bytes) │          │          │
   │ Warp Exec Efficiency   │          │          │
   └────────────────────────┴──────────┴──────────┘

3. 瓶颈判定
   · 一级：Roofline 分类 → Memory / Compute / Latency Bound
   · 二级：查细分指标定位具体问题

4. 制定本轮优化计划（可多项并行）
   · 优化项 A：[描述]，预期影响 [指标]
   · 优化项 B：[描述]，预期影响 [指标]
   · ...

5. 实施修改

6. 验证
   · 正确性：对比 baseline 输出
   · 性能：ncu --set full -o report_v(N+1)
   · 回归检查：确认未优化指标没有恶化

7. 判断是否继续
   · 已达硬件峰值 80%+ → 可停止
   · 仍有明显瓶颈 → 进入第 N+1 轮
```

---

## 五、架构相关注意事项

不同 Compute Capability 的关键差异，影响优化策略选择：

| 特性 | CC 7.x (Volta/Turing) | CC 8.x (Ampere) | CC 9.0 (Hopper) |
|---|---|---|---|
| Tensor Core 代次 | 第1/2代 | 第3代 | 第4代（支持 FP8） |
| Shared Memory 上限 | 96 KB | 164 KB | 228 KB |
| L2 缓存 | 6 MB | 40-80 MB | 50 MB |
| `cp.async` | ✗ (Volta) / ✓ (Turing有限) | ✓ | ✓（+ TMA） |
| L2 Persistence | ✗ | ✓ | ✓ |
| Thread Block Cluster | ✗ | ✗ | ✓ |
| Warp Specialization 硬件支持 | ✗ | ✗ | ✓（wgmma + TMA） |

**务必根据目标 GPU 的 CC 版本选择可用的优化手段。**

---

## 六、终止条件与常见陷阱

### 终止条件
- Kernel 耗时已达理论最优的 80\%~90\%（通过 Roofline 上界估算）。
- SOL 指标中 SM 或 Memory 已接近峰值，继续优化收益递减。
- 算法本身已无法进一步降低计算量或访存量。

### 常见陷阱
- **Occupancy 陷阱**：盲目追求高 occupancy 导致每线程可用寄存器减少、ILP 下降，反而变慢。应以实测为准。
- **过度展开**：循环展开过大导致寄存器溢出，溢出到 Local Memory 后性能断崖式下降。
- **忽略 L2 行为**：Ampere+ 上 L2 缓存很大，有时看似 Global Memory 访问效率低，但实际被 L2 吸收了，优化方向应转向 L2 命中率而非 Shared Memory tiling。
- **Fusion 过度**：kernel 过大导致寄存器压力爆炸，反而不如拆成两个 kernel。需要权衡。
- **忽略验证**：每轮必须验证计算正确性。快速数学选项（`--use_fast_math`）可能引入精度问题。
