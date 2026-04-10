---
name: operator-optimize-loop
description: Run a CUDA operator optimization loop that enforces correctness validation, benchmark, targeted/full NCU profiling, per-iteration artifact capture, and final best-version selection. Use when Claude needs to iterate on a `.cu` kernel for a user-specified number of rounds, compare versions by benchmark results, keep full Nsight Compute evidence for the winning version, and prepare the next optimization step from the latest reports.
---

# CUDA Operator Optimization Loop

这个 skill 是 `skills/optimized-skill` 的统一主入口，用来把现有能力串成闭环：

1. correctness validation
2. benchmark
3. targeted NCU analysis
4. full NCU analysis
5. 生成本轮优化方案
6. 生成下一版算子
7. 再次 benchmark/NCU
8. 在用户指定轮数内选出最优版本

## 复用的现有能力

优先复用以下内容，不要重写已有逻辑：
- benchmark: `skills/optimized-skill/kernel-benchmark/scripts/benchmark.py`
- benchmark skill 约定: `skills/optimized-skill/kernel-benchmark/SKILL.md`
- NCU 分析约定: `skills/optimized-skill/ncu-rep-analyze/SKILL.md`
- 优化知识库:
  - `skills/optimized-skill/reference/optim.md`
  - `skills/optimized-skill/reference/memory-optim.md`
  - `skills/optimized-skill/reference/compute-optim.md`
  - `skills/optimized-skill/reference/sync-optim.md`

## 输入

必需：
- `.cu` 文件路径
- `--max-iterations=<N>`

非常重要！！！  如果用户没有显式提供 `--max-iterations`，先要求用户明确给出轮数，再执行，不要自行使用默认值。 非常重要！！！

强烈建议：
- `--ref=<reference.py>`，否则不能宣称 correctness 已验证

常用可选参数：
- `--M=... --N=... --K=...` 等维度参数
- `--warmup`
- `--repeat`
- `--arch`
- `--gpu`
- `--ptr-size`
- `--seed`
- `--run-dir`

## 每轮迭代的强制流程

对 `v0` 到 `vN` 的每一轮都必须执行：

1. 调用 `optimize_loop.py` 对当前版本做完整评测。
2. 读取本轮输出的：
   - `benchmark_result.json`
   - `iteration_summary.md`
   - targeted/full NCU 的 summary/details 文本
3. 按照“首轮广覆盖、后续针对性修正”的规则制定本轮优化方向。
4. 写出本轮 `optimization_proposal.md`。
5. 基于 proposal 生成下一版 kernel。
6. 继续下一轮，直到达到 `max_iterations` 或提前停止。

## 首轮与后续轮的优化策略

### 第一次优化（baseline -> v1）

第一次优化要尽可能多地吸收 `reference/` 中已经整理好的优化方法，但前提是这些方法与当前 kernel 的算法形态相容，且不会明显互相冲突。

优先顺序：
1. 先从 `optim.md` 获取整体迭代思路。
2. 再从以下文档中尽可能覆盖当前 kernel 能合理采用的优化项：
   - `reference/memory-optim.md`
   - `reference/compute-optim.md`
   - `reference/sync-optim.md`
3. 首轮允许做“覆盖面较广”的组合优化，但不要为了堆技巧而引入明显不适配的改法。

首轮目标：
- 先做一版高质量、覆盖较广的候选实现。
- 尽量把通用收益高的 memory / compute / sync 优化一次性吃进去。
- 明确评估是否可以加入双缓冲 / 多级流水线：在 Shared Memory 或寄存器中准备两组 buffer，让当前批次计算与下一批次加载重叠。
- 产出 first-pass 的 targeted/full NCU 报告，作为后续轮次的基准。

### 后续迭代（v1 之后）

后续迭代不要再追求广泛铺开，而要针对“上一轮 full NCU 报告暴露的最主要不足”做定向修正。

规则：
- 先看上一轮 full NCU，再参考 targeted NCU。
- 每一轮只优先解决 1 到 2 个最明确的瓶颈。
- 每一轮都要显式评估“双缓冲 / 多级流水线”是否适合当前 kernel：
  - 阶段1：buffer A 计算，buffer B 加载下一批数据
  - 阶段2：buffer B 计算，buffer A 加载下一批数据
- 如果适合，就把双缓冲 / 多级流水线纳入本轮优化候选；如果不适合，也要明确写出原因，例如数据复用不足、tile 结构不匹配、寄存器或 shared memory 压力过高。
- 优化方向要和具体 NCU 信号绑定，例如：
  - coalescing 差 -> 优先修访存布局 / vectorization / tiling
  - occupancy 低 -> 优先修寄存器、smem、block size
  - latency bound -> 优先修 ILP、依赖链、同步粒度，并优先考虑双缓冲 / 多级流水线隐藏访存延迟
  - compute bound -> 优先修 Tensor Core、FMA、低精度路径
- 不要在后续轮次继续无差别叠加新技巧；每轮改动都要能解释“它是在修上一轮 NCU 的哪个短板”。

## 标准执行命令

```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <cu_file> \
    --max-iterations=<N> [--ref=<ref.py>] [--DIM=VALUE ...] \
    --warmup=10 --repeat=20
```

如果已经有 run 目录，要继续同一轮次序列：

```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <next_version.cu> \
    --run-dir=<existing_run_dir> --iteration=<i> --max-iterations=<N> \
    [--ref=<ref.py>] [--DIM=VALUE ...] --warmup=10 --repeat=20
```

## 产物约定

每次 run 都应生成一个 run 目录，目录下至少包含：
- `run_manifest.json`
- `final_summary.md`
- `iter_v0/`, `iter_v1/`, ...

每轮目录至少包含：
- 当前版本 `.cu`
- `benchmark_result.json`
- `benchmark.stdout.txt`
- `benchmark.stderr.txt`
- `targeted.ncu-rep`
- `full.ncu-rep`
- `targeted_summary.txt`
- `targeted_details.txt`
- `full_summary.txt`
- `full_details.txt`
- `iteration_summary.md`
- `optimization_proposal.md`

## 最优版本选择规则

只有满足以下条件的版本才允许参与排名：
- benchmark 成功
- 如果给了 reference，则 correctness 必须通过
- full `.ncu-rep` 必须存在

排序规则：
1. 主排序：kernel median latency 最低
2. 次排序：kernel average latency 最低
3. 再次排序：更早达到该性能的版本优先

## Claude 在循环中的行为要求

- 首轮优化按 `reference/` 做尽可能广覆盖但仍然合理的组合优化；从第二轮开始，每轮先读上一轮 full NCU 报告，再做针对性修正。
- 每轮改动都要能解释它解决的是哪一个具体瓶颈，不要无差别继续堆优化技巧。
- 每一轮都要检查双缓冲 / 多级流水线是否值得引入或继续保留，并在 proposal 中写明判断依据。
- 不要把 targeted sections 的结论当作最终结论；最终交付必须引用 winning version 的 full NCU 报告。
- correctness 失败的版本可以保留在 run 目录中，但必须明确标记为 rejected，不能参与 best 评选。
- 如果 `ncu` 不可用、导入失败或 full 报告缺失，要明确写出失败原因并停止把该版本当作最终答案。
- 优化建议要尽量和 `memory / compute / sync` 三类文档中的已有策略对应起来。

## 最终回答必须包含

- 最佳版本路径
- baseline 与最佳版本的 benchmark 对比
- 最佳版本 full NCU 报告路径
- 最佳版本 targeted/full NCU 命令
- 主瓶颈判断
- 采用的关键优化思路
- 被淘汰版本及其淘汰原因（如 correctness fail、NCU 不完整、性能不如当前 best）

## 常见提前停止条件

出现以下情况可以提前停止：
- correctness 失败且当前问题明显需要先修正确性
- `ncu` 无法运行或环境不允许 profiling
- 连续多轮没有任何可解释的性能改善
- 已经达到用户要求或收益明显递减
