---
name: ncu-rep-analyze
description: Profile a CUDA kernel with Nsight Compute or analyze an existing `.ncu-rep` report to diagnose bottlenecks and produce actionable optimization guidance. Use when Claude needs to 解释 NCU 指标、定位 kernel 为什么慢、生成 fresh `.ncu-rep`、判断 memory/latency/compute/occupancy 瓶颈，或把报告结论整理成下一轮算子优化可直接使用的建议。
---

# NCU Profiling and Analysis

这个 skill 负责回答“这个 kernel 为什么慢”。
如果输入是 `.cu`，优先为当前版本 kernel 生成 fresh `.ncu-rep`；如果输入是现成 `.ncu-rep`，则解释现有报告。

## 共享文档入口

优先查这些现有文档：
- 总体迭代 workflow: `../reference/optim.md`
- memory 优化: `../reference/memory-optim.md`
- compute 优化: `../reference/compute-optim.md`
- sync 优化: `../reference/sync-optim.md`

## 文件策略

### 如果输入是 `.cu`

- 生成与当前 kernel 同目录、同 stem 的 `.ncu-rep`
- 报告和分析结果都放在 kernel 旁边
- 不要复用旧版本 kernel 的 `.ncu-rep`

### 如果输入是 `.ncu-rep`

- 直接分析现有报告
- 若对应 `.cu` 已经明显变化，要标记“报告可能过期”

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
    python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py <cu_file> \
    [--DIM=VALUE ...] --repeat=22
```

### 第二轮：只在第一轮不够时再深挖

可以按需升级为：
- `--set full`
- `--set roofline`
- 额外 `--metrics ...`

只在需要更深结论时才加重 profiling。

## 报告读取

先看摘要：

```bash
ncu --import <file.ncu-rep> --print-summary per-kernel
```

再按需查询具体指标：

```bash
ncu --import <file.ncu-rep> --page details
```

或直接针对关键指标重查：

```bash
ncu --metrics sm__throughput.avg_pct_of_peak_sustained_elapsed,\
dram__throughput.avg_pct_of_peak_sustained_elapsed,\
sm__warps_active.avg_pct_of_peak_sustained_elapsed \
    <program>
```

## 诊断顺序

1. 先看 `SpeedOfLight`：
   - `SM` 高还是 `Memory` 高
   - 两者是否都低

2. 再看 `Occupancy` 和 `LaunchStatistics`：
   - achieved occupancy
   - 理论 occupancy
   - 限制因子是 registers、shared memory 还是 block size

3. 再看 `MemoryWorkloadAnalysis`：
   - global load/store pattern
   - sector/request
   - L1/L2/DRAM 压力
   - shared memory bank conflict

4. 最后看 `SchedulerStatistics`：
   - warp stall
   - eligible warps
   - issue efficiency

## 瓶颈分类规则

| 类别 | 典型信号 | 第一建议 |
| --- | --- | --- |
| `DRAM_MEMORY_BOUND` | DRAM 高、SM 低、sector/request 差 | 先修 coalescing，再看 vectorization / tiling |
| `L1_PRESSURE_BOUND` | L1/TEX 压力高、shared path 紧张、可能有 bank conflict | shared memory tiling、transpose、padding 或 swizzling |
| `LATENCY_BOUND` | SM 低、Memory 也不高、occupancy 尚可、eligible warps 低 | ILP、unroll、double buffering、减少长依赖链 |
| `COMPUTE_BOUND` | SM 高、SM Busy 高、Memory 不是主问题 | FMA、低精度、Tensor Core |
| `OCCUPANCY_BOUND` | achieved occupancy 低，且限制因子明确 | 降 registers/smem、改 block size、`__launch_bounds__` |
| `HOST_OR_LAUNCH_BOUND` | kernel 很短、网格很小、GPU 指标都不高 | 不要继续盲改 kernel，转去更上层时序分析 |
| `MIXED_BOUND` | 多项都一般，没有单一主症状 | 只选最明确的一类先验证 |

## 具体判断要点

### Memory coalescing

重点看：
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
- `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`

经验解释：
- `1-4` sectors/request 通常不错
- `8-16` 说明已经有明显问题
- `32+` 往往接近随机访问

### Shared memory bank conflict

重点看：
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`

若 conflict / wavefront 明显大于 `1`，就要优先怀疑：
- tile 布局
- transpose 读写模式
- warp 到数据的映射

### Occupancy

不要只看“低不低”，还要看为什么低：
- registers 限制
- shared memory 限制
- block size 限制

如果 theoretical occupancy 也低，说明资源配置本身就是瓶颈。
如果 theoretical 高但 achieved 低，更多是调度或依赖链问题。

## 什么时候转向更上层时序分析

出现这些情况时，不要继续只盯着 NCU：
- low SM 且 low Memory
- kernel 本身很短，但调用次数非常多
- 明显怀疑 CPU 端准备、同步、分配或 launch gap

这时要明确指出：当前问题可能不是单个 kernel 内部瓶颈，不能继续只靠 NCU 下结论。

## 不要做的事

- 不要把 NCU expert system 建议当成直接处方
- 不要用别的 kernel 的 `.ncu-rep` 冒充当前版本分析
- 不要因为 NCU 失败就“凭感觉”输出高置信度瓶颈结论

## 输出格式

输出一份结构化分析，至少包含：
- 报告路径
- kernel 名称
- 是否 fresh profile
- targeted NCU 命令与报告路径
- full NCU 命令与报告路径
- 关键指标摘要
- 主瓶颈类型
- 判断依据
- 高优先级优化建议
- 是否需要转更上层时序分析或 correctness 排查

如果从 `.cu` 生成了新报告，还要把：
- targeted NCU 命令
- targeted `.ncu-rep` 路径
- full NCU 命令
- full `.ncu-rep` 路径
- `ncu --import <file.ncu-rep> --print-summary per-kernel` 这类导入查看命令
- `_analysis.md` 或 summary 文件路径

一起交付。

## 失败处理

最终最好方案交付时，必须带当前最好版本对应的 full NCU 报告信息，不能只给 targeted sections 结果。

如果 `ncu` 不可用、权限不足或被环境阻止：
- 明确写出失败原因
- 标记本轮 profiling 失败
- 不要静默跳过
- 给出人工修复方向后停止
