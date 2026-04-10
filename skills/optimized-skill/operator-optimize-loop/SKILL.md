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

如果用户没有显式提供 `--max-iterations`，先要求用户明确给出轮数，再执行，不要自行使用默认值。

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
3. 根据 NCU 结论在 `reference/*.md` 里选择最有把握的优化方向。
4. 写出本轮 `optimization_proposal.md`。
5. 基于 proposal 生成下一版 kernel。
6. 继续下一轮，直到达到 `max_iterations` 或提前停止。

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

- 每轮先读报告，再改 kernel，不要跳过诊断直接盲改。
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
