# optimized-skill

这个目录提供一组面向 CUDA 算子优化的 skills，覆盖从正确性验证到 benchmark、NCU 分析、迭代优化和最终最佳版本汇总的完整流程。

## 包含的 skill

- `kernel-benchmark/`
  - 编译 `.cu`，可选做 correctness 验证，并输出 benchmark。
- `ncu-rep-analyze/`
  - 生成或分析 `.ncu-rep`，定位主瓶颈并给出优化方向。
- `operator-optimize-loop/`
  - 统一编排入口，执行 correctness -> benchmark -> targeted/full NCU -> 优化方案 -> 下一版算子 -> 再评测 的闭环。
- `reference/`
  - 优化知识库，按 memory / compute / sync 分类。

## 推荐入口

需要完整迭代优化时，优先使用：

- `skills/optimized-skill/operator-optimize-loop/SKILL.md`

需要单独做正确性和性能测试时，使用：

- `skills/optimized-skill/kernel-benchmark/SKILL.md`

需要单独做 NCU 报告生成和分析时，使用：

- `skills/optimized-skill/ncu-rep-analyze/SKILL.md`

## 完整工作流

主入口 skill 期望执行以下流程：

1. 对当前 kernel 做 correctness 验证
2. 跑 benchmark，记录 baseline
3. 跑 targeted NCU
4. 跑 full NCU
5. 根据报告选择最有把握的优化方向
6. 生成下一版 kernel
7. 对新版本重复 correctness、benchmark、targeted/full NCU
8. 在用户设定的迭代轮数内选出最优版本
9. 最终交付最优版本、full NCU 报告和 benchmark 对比

## 用户可配置参数

主入口支持的核心参数：

- `cu_file`
- `--ref=<reference.py>`
- `--M=... --N=... --K=...` 等维度参数
- `--max-iterations=<N>`
- `--warmup=<N>`
- `--repeat=<N>`
- `--arch=<sm_xx>`
- `--gpu=<idx>`
- `--ptr-size=<N>`
- `--seed=<N>`
- `--run-dir=<dir>`

其中：
- `max_iterations` 控制最多迭代多少轮，且必须由用户显式提供。
- `ref` 强烈建议提供；没有 reference 时，不应宣称 correctness 已验证。

## 命令示例

### 1. 只做 correctness + benchmark

```bash
python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py path/to/kernel.cu \
    --ref=path/to/reference.py --M=4096 --N=4096 --K=4096 \
    --warmup=10 --repeat=20
```

### 2. 做一次完整迭代评测并生成 run 目录

注意：`--max-iterations` 是必填项，不传会直接报错。

```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py path/to/kernel.cu \
    --ref=path/to/reference.py --M=4096 --N=4096 --K=4096 \
    --max-iterations=3 --warmup=10 --repeat=20
```

### 3. 在已有 run 目录上评测下一版 kernel

```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py path/to/kernel_v1.cu \
    --run-dir=path/to/optimize_runs/run_20260410_000000 --iteration=1 \
    --ref=path/to/reference.py --M=4096 --N=4096 --K=4096 \
    --max-iterations=3 --warmup=10 --repeat=20
```

## 输出产物

每次完整 run 都会生成独立目录，默认在当前 kernel 附近：

```text
optimize_runs/
  run_YYYYMMDD_HHMMSS/
    run_manifest.json
    final_summary.md
    iter_v0/
      <kernel>_v0.cu
      benchmark_result.json
      benchmark.stdout.txt
      benchmark.stderr.txt
      targeted.ncu-rep
      full.ncu-rep
      targeted_summary.txt
      targeted_details.txt
      full_summary.txt
      full_details.txt
      iteration_summary.md
      optimization_proposal.md
    iter_v1/
      ...
```

### 关键产物说明

- `run_manifest.json`
  - 记录输入参数、每轮结果、当前 best iteration。
- `final_summary.md`
  - 汇总所有轮次，列出 baseline 与最佳版本，并给出 full NCU 报告路径。
- `benchmark_result.json`
  - 结构化 benchmark 结果，便于脚本和后续分析复用。
- `iteration_summary.md`
  - 当前轮的命令、状态、产物路径和 follow-up 要点。
- `optimization_proposal.md`
  - 当前轮准备尝试的优化假设。

## 如何确定最终最优版本

只有满足以下条件的版本才允许参与 best 评选：

- benchmark 成功
- 如果提供了 reference，则 correctness 通过
- full `.ncu-rep` 存在

排序规则：

1. kernel median latency 最低
2. 若接近，则看 kernel average latency
3. 若仍接近，则更早达到该性能的版本优先

最终回答必须引用：

- 最佳版本 `.cu` 路径
- 最佳版本 full `.ncu-rep` 路径
- 最佳版本 targeted/full NCU 命令
- baseline 与最佳版本的 benchmark 对比
- 主瓶颈与最终采用的优化思路

## 使用建议

- 每轮先读 targeted/full NCU 报告，再改 kernel。
- correctness 失败的版本不能参与性能结论。
- 最终交付不能只引用 targeted 结果，必须带 winning version 的 full NCU 报告。
- 优化建议优先从 `reference/` 下已有知识库中选择，不要每轮都发明新套路。

## 常见失败场景

### 1. correctness 失败

说明优化还不成立。此时先修正确性，不应继续拿该版本做最佳方案。

### 2. `ncu` 不可用

会导致 profiling 无法完成。此时必须明确说明失败原因，不能静默跳过 full report。

### 3. benchmark 噪声过大

优先统一：
- 输入规模
- `warmup`
- `repeat`
- GPU 选择

### 4. 多轮没有收益

可以提前停止，把当前 best 作为最终候选，并说明后续收益递减。
