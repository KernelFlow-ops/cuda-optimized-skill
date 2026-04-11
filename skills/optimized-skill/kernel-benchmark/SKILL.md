---
name: kernel-benchmark
description: Compile, validate, and benchmark a CUDA, CUTLASS, or Triton operator against an optional Python reference by driving `skills/optimized-skill/kernel-benchmark/scripts/benchmark.py`. Use when Claude needs to 验证算子正确性、做 baseline 性能测试、比较 operator 与 reference 的 speedup、对 CUDA/CUTLASS 读取 `extern "C" void solve(...)` 签名推断参数，或在进入后续分析前先得到稳定 benchmark 结果。
---

# Kernel Benchmark

通过 `skills/optimized-skill/kernel-benchmark/scripts/benchmark.py` 编译、验证、压测三类算子：

- `cuda`: `extern "C" void solve(...)` 风格的 CUDA `.cu`
- `cutlass`: 同样暴露 `solve(...)` 入口的 CUTLASS `.cu`
- `triton`: 暴露 `setup(...)` + `run_kernel(...)` 的 Triton `.py`

所有命令都从仓库根目录运行。

## 共享文档入口

按后端查对应文档：

### CUDA
- `../reference/cuda/optim.md`
- `../reference/cuda/memory-optim.md`
- `../reference/cuda/compute-optim.md`
- `../reference/cuda/sync-optim.md`

### CUTLASS
- `../reference/cutlass/cutlass-optim.md`

### Triton
- `../reference/triton/triton-optim.md`

## 输入

- 必需：solution 文件
  - CUDA / CUTLASS: `.cu`
  - Triton: `.py`
- 可选：`--backend=cuda|cutlass|triton`
- 可选：reference `.py`
- 可选：维度参数，如 `--M=... --N=... --K=...`
- 可选：`--warmup`、`--repeat`、`--arch`、`--gpu`、`--ptr-size`
- 可选：`--atol`、`--rtol`、`--seed`
- 可选：`--nvcc-bin=<path or command>`，指定 `nvcc`（CUDA / CUTLASS）
- 可选：`--json-out=<file>`，把结构化结果写成 JSON，便于上层编排脚本复用

说明：
- `--backend` 不给时，脚本按文件后缀推断：`.py -> triton`，否则默认 `cuda`
- `cutlass` 与 `cuda` 共用 `.cu` benchmark 链路

## 各后端输入约定

### CUDA / CUTLASS

脚本从 `extern "C" void solve(...)` 读取签名并自动推断参数。

要求：
- 必须暴露 `solve(...)`
- 指针参数按当前 benchmark 约定自动分配 torch CUDA tensor
- 非 const 指针视为输出 tensor

### Triton

Triton 模块必须满足：

```python
def setup(*, seed=None, **dims):
    return {
        "inputs": {...},
        "outputs": ["out"],
    }


def run_kernel(**inputs):
    ...
```

约定：
- `setup()` 返回本轮 benchmark/validation 所需输入
- `outputs` 中列出会被原地写入的 tensor 名称
- `reference(**kwargs)` 与 `run_kernel(**kwargs)` 共享同一套输入语义

## reference 推断规则

如果用户没显式给 `--ref`，按这些位置寻找：
- solution 同目录下的 `*_ref.py`
- 算法目录下语义匹配的 reference 文件

找不到 reference 时：
- 仍可 benchmark kernel/operator
- 但不要声称 correctness 已验证

## 维度参数推断规则

### CUDA / CUTLASS

脚本会从 `solve(...)` 读取整型参数名。
若用户没给值，需要补一个合理默认值再运行。

常用默认：
- matmul / GEMM: `M=4096, N=4096, K=4096`
- reduction / element-wise: `N=1000000`
- transpose: `M=4096, N=4096`

### Triton

脚本不会自动从源码解析形状参数名。
需要基于 `setup()` 语义和上下文补足维度参数，不要乱猜极端大尺寸。

## 标准命令

### CUDA

```bash
python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py <kernel.cu> \
    --backend=cuda [--DIM=VALUE ...] --warmup=10 --repeat=20 [--nvcc-bin=<nvcc>]
```

### CUTLASS

```bash
python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py <kernel.cu> \
    --backend=cutlass [--DIM=VALUE ...] --warmup=10 --repeat=20 [--nvcc-bin=<nvcc>]
```

### Triton

```bash
python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py <kernel.py> \
    --backend=triton [--DIM=VALUE ...] --warmup=10 --repeat=20
```

### 验证加 benchmark

```bash
python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py <solution_file> \
    --backend=<backend> --ref=<ref_file> [--DIM=VALUE ...] --warmup=10 --repeat=20
```

### 输出结构化 JSON

```bash
python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py <solution_file> \
    --backend=<backend> --ref=<ref_file> [--DIM=VALUE ...] --json-out=benchmark_result.json
```

## benchmark.py 的实际行为

### CUDA / CUTLASS
脚本会：
- 从 `solve(...)` 签名推断参数
- 自动检测 GPU 架构，未指定时默认用当前设备 capability
- 用 `nvcc` 编译为共享库
- 有 reference 时先跑 correctness，再 benchmark reference 和 kernel
- 没有 reference 时，先打印输入输出 preview，再 benchmark kernel

### Triton
脚本会：
- 导入 Triton module
- 调用 `setup()` 构造输入
- 有 reference 时先跑 correctness，再 benchmark reference 和 Triton kernel
- 没有 reference 时，先打印输入输出 preview，再 benchmark Triton kernel

### 统一 JSON 输出
若给了 `--json-out`，则额外产出结构化结果，至少包含：
- `solution_file`
- `cu_file`（兼容旧字段）
- `backend`
- `ref_file`
- `has_reference`
- `dims`
- `gpu_name`
- `arch`
- `correctness`
- `kernel`
- `reference`
- `speedup_vs_reference`

## 如何解读结果

优先看这些信息：
- correctness 是否通过
- `Average` 和 `Median` 是否稳定
- `Speedup` 是否真的大于 reference
- `~Bandwidth` 是否与瓶颈判断一致

经验规则：
- `Average`、`Median` 差很多，通常说明 benchmark 不稳定
- 快了但 correctness 失败，不算优化成功
- 规模不同的 benchmark 不能直接比较

## 失败处理

### correctness 失败

立即停止后续性能结论，优先反馈：
- 失败的输出 tensor
- 最大误差和首个错误位置
- 建议先做内存、竞争和同步问题排查

### 编译失败（CUDA / CUTLASS）

直接返回 `nvcc` 错误，不要继续做性能判断。

### benchmark 噪声过大

优先统一：
- 输入规模
- `warmup`
- `repeat`
- GPU 选择

## 输出约定

返回一份简洁报告：
- solution 路径
- backend
- reference 路径或“未提供”
- 维度参数
- GPU / arch
- 完整实际执行命令
- correctness 结果
- kernel latency summary
- reference latency summary
- speedup
- 是否建议进入后续 profiling / 分析

完整实际执行命令必须原样回显，至少包含：
- solution 路径
- `--backend`
- `--ref`
- 所有 `--DIM=VALUE`
- `--arch`
- `--warmup`
- `--repeat`
- `--json-out`（若使用）
