---
name: ncu-rep-analyze
description: Profile a CUDA, CUTLASS, or Triton operator with Nsight Compute or analyze an existing `.ncu-rep` report to diagnose bottlenecks and produce actionable optimization guidance. Use when Claude needs to 解释 NCU 指标、定位 kernel 为什么慢、生成 fresh `.ncu-rep`、判断 memory/latency/compute/occupancy 瓶颈，或把报告结论整理成下一轮算子优化可直接使用的建议。
---

# NCU Profiling and Analysis

这个 skill 负责回答“这个 kernel 为什么慢”。

当前适用后端：
- `cuda`
- `cutlass`
- `triton`

说明：
- 如果输入是当前算子文件，优先为当前版本生成 fresh `.ncu-rep`
- 如果输入是现成 `.ncu-rep`，则解释现有报告
- Triton 同样可以走 NCU，方式是通过 `ncu --target-processes all ... python benchmark.py <kernel.py>` 抓取 Python 进程中 launch 的 GPU kernel

## 共享文档入口

### CUDA
- `../reference/cuda/optim.md`
- `../reference/cuda/memory-optim.md`
- `../reference/cuda/compute-optim.md`
- `../reference/cuda/sync-optim.md`

### CUTLASS
- `../reference/cutlass/cutlass-optim.md`

### Triton
- `../reference/triton/triton-optim.md`

## 文件策略

### 如果输入是算子源码

- 生成与当前算子同目录、同 stem 的 `.ncu-rep`
- 报告和分析结果都放在算子旁边
- 不要复用旧版本算子的 `.ncu-rep`

### 如果输入是 `.ncu-rep`

- 直接分析现有报告
- 若对应源码已经明显变化，要标记“报告可能过期”

## 推荐 profiling 流程

先做一轮有针对性的采样，避免默认 `--set full` 过重。

### 第一轮：目标化 section

```bash
ncu --target-processes all \
    --profile-from-start on \
    --launch-skip 20 \
    --launch-count 1 \
    --section LaunchStatistics \
    --section Occupancy \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section SchedulerStatistics \
    -o {kernel_dir}/{kernel_stem} -f \
    python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py <solution_file> \
    --backend=<cuda|cutlass|triton> [--DIM=VALUE ...] --repeat=22
```

### 第二轮：只在第一轮不够时再深挖

可以按需升级为：
- `--set full`
- `--set roofline`
- 额外 `--metrics ...`

## 报告读取

先看摘要：

```bash
ncu --import <file.ncu-rep> --print-summary per-kernel
```

再按需查询具体指标：

```bash
ncu --import <file.ncu-rep> --page details
```

## 诊断顺序

1. 先看 `SpeedOfLight`
2. 再看 `Occupancy` 和 `LaunchStatistics`
3. 再看 `MemoryWorkloadAnalysis`
4. 最后看 `SchedulerStatistics`

## 瓶颈分类规则

| 类别 | 典型信号 | 第一建议 |
| --- | --- | --- |
| `DRAM_MEMORY_BOUND` | DRAM 高、SM 低、sector/request 差 | 先修 coalescing，再看 vectorization / tiling |
| `L1_PRESSURE_BOUND` | L1/TEX 压力高、shared path 紧张、可能有 bank conflict | shared memory tiling、transpose、padding 或 swizzling |
| `LATENCY_BOUND` | SM 低、Memory 也不高、occupancy 尚可、eligible warps 低 | ILP、unroll、double buffering、减少长依赖链 |
| `COMPUTE_BOUND` | SM 高、SM Busy 高、Memory 不是主问题 | Tensor Core、低精度、MMA 路径 |
| `OCCUPANCY_BOUND` | achieved occupancy 低，且限制因子明确 | 降 registers/smem、改 block size、改 tile |
| `HOST_OR_LAUNCH_BOUND` | kernel 很短、网格很小、GPU 指标都不高 | 不要继续盲改 kernel，转去更上层时序分析 |
| `MIXED_BOUND` | 多项都一般，没有单一主症状 | 只选最明确的一类先验证 |

## 后端特化要求

### CUTLASS
不要只给泛化的 CUDA 建议，要尽量映射到 CUTLASS pattern：
- Tensor Core 路径 / 数据类型
- threadblock / warp / mma tile shape
- stage count / multistage pipeline
- epilogue fusion / EVT
- split-K / stream-K
- swizzle / scheduler
- 架构特性（Ampere / Hopper / Blackwell）

### Triton
不要只给泛化的 CUDA 建议，要尽量映射到 Triton choices：
- BLOCK_M / BLOCK_N / BLOCK_K
- `num_warps`
- `num_stages`
- coalescing / vectorization hints
- fusion
- swizzle / persistent / split-K
- autotune 配置空间是否过大或过小

## 不要做的事

- 不要把 NCU expert system 建议当成直接处方
- 不要用别的 kernel 的 `.ncu-rep` 冒充当前版本分析
- 不要因为 NCU 失败就“凭感觉”输出高置信度瓶颈结论

## 输出格式

输出一份结构化分析，至少包含：
- 报告路径
- backend
- kernel 名称
- 是否 fresh profile
- targeted NCU 命令与报告路径
- full NCU 命令与报告路径
- 关键指标摘要
- 主瓶颈类型
- 判断依据
- 高优先级优化建议
- 是否需要转更上层时序分析或 correctness 排查

## 失败处理

最终最好方案交付时，必须带当前最好版本对应的 full NCU 报告信息，不能只给 targeted sections 结果。

如果 `ncu` 不可用、权限不足或被环境阻止：
- 明确写出失败原因
- 标记本轮 profiling 失败
- 不要静默跳过
- 给出人工修复方向后停止
