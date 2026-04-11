#!/usr/bin/env python3
"""Generic operator benchmark with optional correctness validation.

Supported backends:
  - cuda: raw CUDA `.cu` kernels exposing `extern "C" void solve(...)`
  - cutlass: CUTLASS-based `.cu` kernels exposing the same `solve(...)` entry
  - triton: Python modules exposing `setup(...)` + `run_kernel(...)`

When --ref is provided:
  1. Validates kernel correctness against the reference implementation.
     Exits immediately if validation fails.
  2. Benchmarks the reference implementation.
  3. Benchmarks the target kernel/module.
  4. Prints a combined summary with speedup.

When --ref is omitted:
  Benchmarks the target kernel/module only.
"""

import re
import os
import sys
import json
import copy
import subprocess
import ctypes
import argparse
import importlib.util
import torch

# ---------------------------------------------------------------------------
# Type tables
# ---------------------------------------------------------------------------

SUPPORTED_TYPES = {
    "float*": ("float*", ctypes.c_void_p),
    "double*": ("double*", ctypes.c_void_p),
    "unsigned char*": ("unsigned char*", ctypes.c_void_p),
    "unsigned short*": ("unsigned short*", ctypes.c_void_p),
    "unsigned int*": ("unsigned int*", ctypes.c_void_p),
    "char*": ("char*", ctypes.c_void_p),
    "short*": ("short*", ctypes.c_void_p),
    "long*": ("long*", ctypes.c_void_p),
    "int*": ("int*", ctypes.c_void_p),
    "int": ("int", ctypes.c_int),
    "long": ("long", ctypes.c_long),
    "size_t": ("size_t", ctypes.c_size_t),
    "unsigned int": ("unsigned int", ctypes.c_uint),
    "unsigned short": ("unsigned short", ctypes.c_ushort),
    "unsigned char": ("unsigned char", ctypes.c_ubyte),
    "char": ("char", ctypes.c_char),
    "short": ("short", ctypes.c_short),
}

DTYPE_MAP = {
    "float*": torch.float32,
    "double*": torch.float64,
    "int*": torch.int32,
    "long*": torch.int64,
    "short*": torch.int16,
    "char*": torch.int8,
    "unsigned char*": torch.uint8,
    "unsigned short*": getattr(torch, "uint16", torch.int16),
    "unsigned int*": getattr(torch, "uint32", torch.int32),
}

INT_TYPES = {"int", "long", "size_t", "unsigned int"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_solve_signature(cu_file: str):
    """Extract parameter list from `extern "C" void solve(...)` in a .cu file."""
    with open(cu_file, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r'extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f'Cannot find \'extern "C" void solve(...)\' in {cu_file}')

    raw = match.group(1)
    raw = re.sub(r"/\*.*?\*/", "", raw)
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = " ".join(raw.split())

    params = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        is_const = "const" in token
        token_clean = re.sub(r"\s+", " ", token.replace("const", "").strip())
        matched = False
        for key in sorted(SUPPORTED_TYPES.keys(), key=len, reverse=True):
            base = key.replace("*", r"\s*\*")
            m = re.match(rf"({base})\s+(\w+)", token_clean)
            if m:
                params.append((key, m.group(2), is_const))
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot parse parameter: '{token.strip()}'")

    return params



def detect_arch(device_index: int | None = None) -> str:
    """Auto-detect GPU compute capability and return sm_XX string."""
    if torch.cuda.is_available():
        if device_index is None:
            device_index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device_index)
        return f"sm_{major}{minor}"
    return "sm_80"


_STRIP_INCLUDES = re.compile(r'^\s*#\s*include\s*<__clang_cuda[^>]*>\s*$', re.MULTILINE)



def _preprocess_cu(cu_file: str) -> str:
    """Strip clang-specific includes that break nvcc. Returns path to clean file."""
    with open(cu_file, "r", encoding="utf-8") as f:
        src = f.read()
    cleaned = _STRIP_INCLUDES.sub("", src)
    if cleaned == src:
        return cu_file
    tmp = cu_file + ".nvcc_clean.cu"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(cleaned)
    return tmp



def compile_cu(cu_file: str, output_so: str, arch: str, nvcc_bin: str):
    """Compile .cu to a shared library."""
    clean_file = _preprocess_cu(cu_file)
    cmd = [nvcc_bin, "-shared", "-std=c++17", f"-arch={arch}", "-O3", "-o", output_so, clean_file]
    if os.name != "nt":
        cmd[2:2] = ["-Xcompiler", "-fPIC"]
    else:
        cmd[2:2] = [
            "-allow-unsupported-compiler",
            "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        ]
    print(f"[compile] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except OSError as exc:
        if clean_file != cu_file and os.path.exists(clean_file):
            os.remove(clean_file)
        print(f"Compilation failed:\n{exc}", file=sys.stderr)
        sys.exit(1)
    if clean_file != cu_file and os.path.exists(clean_file):
        os.remove(clean_file)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"[compile] -> {output_so}")



def load_python_module(module_file: str, module_name: str):
    """Import a Python module from a file path and return its module object."""
    if not os.path.exists(module_file):
        raise FileNotFoundError(f"Module file not found: {module_file}")
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {module_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



def load_reference(ref_file: str):
    """Import a Python reference file and return its module."""
    mod = load_python_module(ref_file, "_ref_module")
    if not hasattr(mod, "reference"):
        raise AttributeError(f"'{ref_file}' must define a `reference(**kwargs)` function.")
    return mod



def infer_backend(solution_file: str, backend: str) -> str:
    if backend != "auto":
        return backend
    suffix = os.path.splitext(solution_file)[1].lower()
    if suffix == ".py":
        return "triton"
    return "cuda"



def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return copy.deepcopy(value)



def _determine_ptr_elems(int_values: list, ptr_size_override: int) -> int:
    """Calculate number of elements for pointer buffers from dimension values."""
    if ptr_size_override > 0:
        ptr_elems = ptr_size_override
    elif len(int_values) == 0:
        ptr_elems = 1024 * 1024
    elif len(int_values) == 1:
        ptr_elems = int_values[0]
    else:
        sv = sorted(int_values, reverse=True)
        ptr_elems = sv[0] * sv[1]
    return min(ptr_elems, 256 * 1024 * 1024)



def _fmt_vals(vals, width=10):
    """Format a list of numeric values for compact display."""
    return "[" + ", ".join(f"{v:>{width}.4f}" for v in vals) + "]"



def _color(text: str, ok: bool) -> str:
    """ANSI color: green for pass, red for fail (only when stdout is a tty)."""
    if not sys.stdout.isatty():
        return text
    code = "\033[92m" if ok else "\033[91m"
    return f"{code}{text}\033[0m"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_iterations(fn, warmup: int, repeat: int) -> list:
    """Run fn for warmup + repeat iterations and return per-iter ms timings."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    avg_ms = start_event.elapsed_time(end_event) / repeat
    return [avg_ms] * repeat



def _stats(times_ms: list):
    avg = sum(times_ms) / len(times_ms)
    med = sorted(times_ms)[len(times_ms) // 2]
    return avg, med, min(times_ms), max(times_ms)



def _stats_dict(times_ms: list):
    avg, med, mn, mx = _stats(times_ms)
    return {
        "average_ms": avg,
        "median_ms": med,
        "min_ms": mn,
        "max_ms": mx,
    }



def _write_json_out(path: str, payload: dict):
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------


def _print_results(label, avg, med, mn, mx, total_ptr_bytes, ptr_elems, solution_file, dim_values, arch, ref_avg=None):
    """Print benchmark results table; append speedup line when ref_avg is given."""
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print()
    print("=" * 55)
    print(f"  {label}")
    print(f"  Target       : {os.path.basename(solution_file)}")
    print(f"  GPU          : {gpu_name}")
    print(f"  Arch         : {arch}")
    print(f"  Dims         : {dim_values}")
    print(f"  Buf/ptr      : {ptr_elems} elems")
    print("-" * 55)
    print(f"  Average      : {avg:>10.4f} ms")
    print(f"  Median       : {med:>10.4f} ms")
    print(f"  Min          : {mn:>10.4f} ms")
    print(f"  Max          : {mx:>10.4f} ms")
    if avg > 0:
        bw = total_ptr_bytes / (avg / 1000) / 1e9
        print(f"  ~Bandwidth   : {bw:>10.2f} GB/s  (all tensors, rough)")
    if ref_avg is not None and avg > 0:
        speedup = ref_avg / avg
        print(f"  Speedup      : {speedup:>10.2f}x  vs reference")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_outputs(kernel_tensors, ref_tensors, output_params, atol, rtol):
    """Compare kernel and reference output tensors. Returns True if all pass."""
    PREVIEW = 8
    print(f"\n[validate] {len(output_params)} output tensor(s)\n")

    all_pass = True
    for pname, ptype in output_params:
        kt = kernel_tensors[pname].float()
        rt = ref_tensors[pname].float()

        match = torch.allclose(kt, rt, atol=atol, rtol=rtol)
        if not match:
            all_pass = False

        max_diff = (kt - rt).abs().max().item()
        mean_diff = (kt - rt).abs().mean().item()
        rel_err = ((kt - rt).abs() / rt.abs().clamp(min=1e-8)).mean().item()

        status_str = _color("PASS" if match else "FAIL", match)
        print(f"  [{status_str}]  {pname}  ({ptype})")
        print(f"         max |delta|   = {max_diff:.6e}")
        print(f"         mean |delta|  = {mean_diff:.6e}")
        print(f"         mean rel  = {rel_err:.6e}")

        if not match:
            diff_mask = ~torch.isclose(kt, rt, atol=atol, rtol=rtol)
            bad_idx = diff_mask.nonzero(as_tuple=True)[0]
            n_bad = bad_idx.numel()
            print(f"         mismatches: {n_bad} / {kt.numel()}")
            if n_bad > 0:
                idx = bad_idx[0].item()
                print(f"         first bad   @ idx={idx}:  kernel={kt[idx].item():.6f}  ref={rt[idx].item():.6f}")

        k_preview = kernel_tensors[pname][:PREVIEW].float().cpu().tolist()
        r_preview = ref_tensors[pname][:PREVIEW].float().cpu().tolist()
        print(f"         kernel[:{PREVIEW}] = {_fmt_vals(k_preview)}")
        print(f"         ref   [:{PREVIEW}] = {_fmt_vals(r_preview)}")
        print()

    return all_pass


# ---------------------------------------------------------------------------
# Setup: compile + allocate buffers
# ---------------------------------------------------------------------------


def _setup_cuda(solution_file, dim_values, ptr_size_override, arch, nvcc_bin, seed=None, backend_name="cuda"):
    params = parse_solve_signature(solution_file)
    sig_str = ", ".join(f"{'const ' if c else ''}{t} {n}" for t, n, c in params)
    print(f"[signature] solve({sig_str})\n")

    lib_ext = ".dll" if os.name == "nt" else ".so"
    so_file = os.path.splitext(solution_file)[0] + lib_ext
    compile_cu(solution_file, so_file, arch, nvcc_bin)
    lib = ctypes.CDLL(so_file)

    for ptype, pname, _ in params:
        if ptype in INT_TYPES and pname not in dim_values:
            raise ValueError(f"Missing dimension: --{pname}=<value>  (required by kernel signature)")

    int_vals = [dim_values[pname] for ptype, pname, _ in params if ptype in INT_TYPES]
    ptr_elems = _determine_ptr_elems(int_vals, ptr_size_override)

    if seed is not None:
        torch.manual_seed(seed)

    tensor_inputs = {}
    reference_inputs = {}
    output_specs = []
    kernel_call_args = []
    argtypes = []

    print("[buffers]")
    for ptype, pname, is_const in params:
        if ptype in DTYPE_MAP:
            dtype = DTYPE_MAP[ptype]
            if dtype.is_floating_point:
                tensor = torch.randn(ptr_elems, device="cuda", dtype=dtype)
            else:
                tensor = torch.zeros(ptr_elems, device="cuda", dtype=dtype).random_()
            tensor_inputs[pname] = tensor
            reference_inputs[pname] = tensor
            if not is_const:
                output_specs.append((pname, ptype))
            kernel_call_args.append(ctypes.c_void_p(tensor.data_ptr()))
            argtypes.append(ctypes.c_void_p)
            role = "input" if is_const else "output"
            eb = tensor.element_size()
            print(
                f"  {pname:>10s} : {ptype:<16s} [{role:>6s}] "
                f"{ptr_elems} elems  ({ptr_elems * eb / 1024 / 1024:.1f} MB)"
            )
        elif ptype in SUPPORTED_TYPES:
            _, ctype = SUPPORTED_TYPES[ptype]
            val = dim_values[pname]
            reference_inputs[pname] = val
            kernel_call_args.append(ctype(val))
            argtypes.append(ctype)
            print(f"  {pname:>10s} : {ptype:<16s} = {val}")

    lib.solve.restype = None
    lib.solve.argtypes = argtypes

    total_ptr_bytes = sum(t.nelement() * t.element_size() for t in tensor_inputs.values())

    return {
        "backend": backend_name,
        "signature": [
            {"type": ptype, "name": pname, "is_const": is_const}
            for ptype, pname, is_const in params
        ],
        "callable": lambda: lib.solve(*kernel_call_args),
        "tensor_inputs": tensor_inputs,
        "reference_inputs": reference_inputs,
        "output_specs": output_specs,
        "ptr_elems": ptr_elems,
        "total_ptr_bytes": total_ptr_bytes,
        "preview_tensors": [
            {
                "name": pname,
                "type": ptype,
                "role": "input" if is_const else "output",
                "tensor": tensor_inputs[pname],
            }
            for ptype, pname, is_const in params if ptype in DTYPE_MAP
        ],
    }



def _setup_triton(solution_file, dim_values, seed=None):
    module = load_python_module(solution_file, "_triton_kernel_module")
    if not hasattr(module, "setup"):
        raise AttributeError(f"'{solution_file}' must define a `setup(**kwargs)` function for Triton benchmarking.")
    if not hasattr(module, "run_kernel"):
        raise AttributeError(f"'{solution_file}' must define a `run_kernel(**kwargs)` function for Triton benchmarking.")

    setup_kwargs = dict(dim_values)
    setup_kwargs["seed"] = seed
    prepared = module.setup(**setup_kwargs)
    if not isinstance(prepared, dict):
        raise TypeError("Triton setup() must return a dict")

    reference_inputs = prepared.get("inputs")
    outputs = prepared.get("outputs")
    if not isinstance(reference_inputs, dict):
        raise TypeError("Triton setup() must return dict['inputs'] as a mapping")
    if not isinstance(outputs, (list, tuple)):
        raise TypeError("Triton setup() must return dict['outputs'] as a list/tuple")

    for name in outputs:
        if name not in reference_inputs:
            raise ValueError(f"Triton output '{name}' not found in setup()['inputs']")
        if not isinstance(reference_inputs[name], torch.Tensor):
            raise TypeError(f"Triton output '{name}' must be a torch.Tensor")

    tensor_inputs = {
        name: value for name, value in reference_inputs.items() if isinstance(value, torch.Tensor)
    }
    ptr_elems = sum(value.numel() for value in tensor_inputs.values())
    total_ptr_bytes = sum(value.nelement() * value.element_size() for value in tensor_inputs.values())

    preview_tensors = []
    for name, value in tensor_inputs.items():
        preview_tensors.append(
            {
                "name": name,
                "type": str(value.dtype).replace("torch.", ""),
                "role": "output" if name in outputs else "input",
                "tensor": value,
            }
        )

    signature = []
    for name, value in reference_inputs.items():
        if isinstance(value, torch.Tensor):
            signature.append({
                "type": f"tensor[{str(value.dtype).replace('torch.', '')}]",
                "name": name,
                "is_const": name not in outputs,
            })
        elif isinstance(value, int):
            signature.append({"type": "int", "name": name, "is_const": True})
        elif isinstance(value, float):
            signature.append({"type": "float", "name": name, "is_const": True})
        else:
            signature.append({"type": type(value).__name__, "name": name, "is_const": True})

    print("[triton setup]")
    for item in signature:
        role = "input" if item["is_const"] else "output"
        print(f"  {item['name']:>10s} : {item['type']:<24s} [{role:>6s}]")

    output_specs = []
    for name in outputs:
        dtype_name = str(reference_inputs[name].dtype).replace("torch.", "")
        output_specs.append((name, dtype_name))

    return {
        "backend": "triton",
        "signature": signature,
        "callable": lambda: module.run_kernel(**reference_inputs),
        "tensor_inputs": tensor_inputs,
        "reference_inputs": reference_inputs,
        "output_specs": output_specs,
        "ptr_elems": ptr_elems,
        "total_ptr_bytes": total_ptr_bytes,
        "preview_tensors": preview_tensors,
    }



def _setup_backend(solution_file, backend, dim_values, ptr_size_override, arch, nvcc_bin, seed=None):
    if backend == "triton":
        return _setup_triton(solution_file, dim_values, seed=seed)
    if backend in {"cuda", "cutlass"}:
        return _setup_cuda(
            solution_file,
            dim_values,
            ptr_size_override,
            arch,
            nvcc_bin,
            seed=seed,
            backend_name=backend,
        )
    raise ValueError(f"Unsupported backend: {backend}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(solution_file, ref_file, dim_values, warmup, repeat, ptr_size_override, arch, atol, rtol, seed, json_out="", nvcc_bin="nvcc", backend="auto"):
    """Main benchmark pipeline."""
    resolved_backend = infer_backend(solution_file, backend)
    has_ref = bool(ref_file)

    ref_fn = None
    _atol = atol
    _rtol = rtol

    if has_ref:
        ref_mod = load_reference(ref_file)
        ref_fn = ref_mod.reference
        _atol = float(getattr(ref_mod, "atol", atol))
        _rtol = float(getattr(ref_mod, "rtol", rtol))
        print(f"[reference] {ref_file}  (atol={_atol}, rtol={_rtol})\n")

    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    result = {
        "solution_file": os.path.abspath(solution_file),
        "cu_file": os.path.abspath(solution_file),
        "backend": resolved_backend,
        "ref_file": os.path.abspath(ref_file) if has_ref else "",
        "has_reference": has_ref,
        "dims": dim_values,
        "warmup": warmup,
        "repeat": repeat,
        "ptr_size_override": ptr_size_override,
        "gpu_index": gpu_index,
        "gpu_name": gpu_name,
        "arch": arch,
        "seed": seed,
        "correctness": {
            "checked": has_ref,
            "passed": None,
            "atol": _atol if has_ref else None,
            "rtol": _rtol if has_ref else None,
            "output_tensor_count": 0,
        },
        "kernel": None,
        "reference": None,
        "speedup_vs_reference": None,
    }

    state = _setup_backend(
        solution_file,
        resolved_backend,
        dim_values,
        ptr_size_override,
        arch,
        nvcc_bin,
        seed=seed if has_ref else None,
    )
    result["signature"] = state["signature"]
    result["ptr_elems"] = state["ptr_elems"]
    result["total_ptr_bytes"] = state["total_ptr_bytes"]
    result["correctness"]["output_tensor_count"] = len(state["output_specs"])

    if not state["output_specs"] and has_ref:
        print("\n[warn] No output tensors detected. Nothing to validate.", file=sys.stderr)

    if has_ref:
        ref_inputs = {
            name: clone_value(value) for name, value in state["reference_inputs"].items()
        }

        print("\n[kernel]    running ... ", end="", flush=True)
        state["callable"]()
        torch.cuda.synchronize()
        print("done")

        print("[reference] running ... ", end="", flush=True)
        ref_fn(**ref_inputs)
        torch.cuda.synchronize()
        print("done")

        kernel_outputs = {
            name: tensor for name, tensor in state["tensor_inputs"].items() if name in {spec[0] for spec in state["output_specs"]}
        }
        ref_outputs = {
            name: tensor for name, tensor in ref_inputs.items() if isinstance(tensor, torch.Tensor) and name in kernel_outputs
        }

        validation_passed = _validate_outputs(
            kernel_outputs,
            ref_outputs,
            state["output_specs"],
            _atol,
            _rtol,
        )

        print("=" * 60)
        print(f"  Target     : {os.path.basename(solution_file)}")
        print(f"  Backend    : {resolved_backend}")
        print(f"  Reference  : {os.path.basename(ref_file)}")
        print(f"  GPU        : {gpu_name}")
        print(f"  Arch       : {arch}")
        print(f"  Dims       : {dim_values}")
        print(f"  Buf/ptr    : {state['ptr_elems']} elems")
        print(f"  Tolerance  : atol={_atol}  rtol={_rtol}")
        print("-" * 60)
        result_str = "ALL PASS" if validation_passed else "FAILED"
        print(f"  Result     : {_color(result_str, validation_passed)}")
        print("=" * 60)

        result["correctness"]["passed"] = validation_passed
        if not validation_passed:
            _write_json_out(json_out, result)
            sys.exit(1)

    times_ref = None
    if has_ref:
        ref_bench_inputs = {
            name: clone_value(value) for name, value in state["reference_inputs"].items()
        }
        print(f"\n[warmup] reference  {warmup} iterations ...")
        times_ref = _time_iterations(lambda: ref_fn(**ref_bench_inputs), warmup, repeat)
        print(f"[bench]  reference  {repeat} iterations ... done")

    if not has_ref:
        preview = 8
        print(f"\n[preview] first {preview} elements before kernel call:")
        for item in state["preview_tensors"]:
            tag = "IN " if item["role"] == "input" else "OUT"
            print(f"  {tag} {item['name']:>6s} = {_fmt_vals(item['tensor'][:preview].float().cpu().tolist())}")

        state["callable"]()
        torch.cuda.synchronize()

        print(f"\n[preview] first {preview} elements after 1 kernel call:")
        for item in state["preview_tensors"]:
            tag = "IN " if item["role"] == "input" else "OUT"
            print(f"  {tag} {item['name']:>6s} = {_fmt_vals(item['tensor'][:preview].float().cpu().tolist())}")

    print(f"\n[warmup] kernel  {warmup} iterations ...")
    times_kernel = _time_iterations(state["callable"], warmup, repeat)
    print(f"[bench]  kernel  {repeat} iterations ... done")

    avg_k, med_k, mn_k, mx_k = _stats(times_kernel)
    result["kernel"] = _stats_dict(times_kernel)
    result["kernel"]["bandwidth_gbps_rough"] = (
        state["total_ptr_bytes"] / (avg_k / 1000) / 1e9 if avg_k > 0 else None
    )

    target_label = f"{resolved_backend.upper()} Kernel"
    if has_ref:
        avg_r, med_r, mn_r, mx_r = _stats(times_ref)
        result["reference"] = _stats_dict(times_ref)
        result["reference"]["bandwidth_gbps_rough"] = (
            state["total_ptr_bytes"] / (avg_r / 1000) / 1e9 if avg_r > 0 else None
        )
        result["speedup_vs_reference"] = avg_r / avg_k if avg_k > 0 else None
        _print_results(
            target_label,
            avg_k,
            med_k,
            mn_k,
            mx_k,
            state["total_ptr_bytes"],
            state["ptr_elems"],
            solution_file,
            dim_values,
            arch,
            ref_avg=avg_r,
        )
        _print_results(
            f"Reference ({os.path.basename(ref_file)})",
            avg_r,
            med_r,
            mn_r,
            mx_r,
            state["total_ptr_bytes"],
            state["ptr_elems"],
            solution_file,
            dim_values,
            arch,
        )
    else:
        _print_results(
            target_label,
            avg_k,
            med_k,
            mn_k,
            mx_k,
            state["total_ptr_bytes"],
            state["ptr_elems"],
            solution_file,
            dim_values,
            arch,
        )

    _write_json_out(json_out, result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generic operator benchmark (CUDA/CUTLASS/Triton, with optional validation)",
        epilog=(
            "Dimension args: pass --NAME=VALUE for each shape/scalar arg.\n"
            "For CUDA/CUTLASS, the kernel must expose extern \"C\" void solve(...).\n"
            "For Triton, the module must define setup(**kwargs) and run_kernel(**kwargs).\n"
            "ref.py must define `reference(**kwargs)` and may set module-level atol/rtol."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("solution_file", help="Path to solution file (.cu or .py)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "cuda", "cutlass", "triton"], help="Backend type")
    parser.add_argument("--ref", type=str, default="", help="Path to reference .py file; enables validation + reference benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--repeat", type=int, default=20, help="Benchmark iterations (default: 20)")
    parser.add_argument("--ptr-size", type=int, default=0, help="Override element count for all CUDA/CUTLASS pointer buffers")
    parser.add_argument("--arch", type=str, default="", help="GPU arch, e.g. sm_90 (auto-detected if omitted)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (default: 0)")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for validation (default: 1e-4)")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for validation (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for input tensors when validating (default: 42)")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to write structured benchmark results as JSON")
    parser.add_argument("--nvcc-bin", type=str, default="nvcc", help="NVCC executable or full path")

    args, unknown = parser.parse_known_args()

    dim_values = {}
    for item in unknown:
        if item.startswith("--") and "=" in item:
            key, val = item[2:].split("=", 1)
            dim_values[key] = int(val)
        else:
            print(f"Warning: ignoring unknown arg '{item}'", file=sys.stderr)

    torch.cuda.set_device(args.gpu)
    arch = args.arch if args.arch else detect_arch(args.gpu)

    run(
        solution_file=args.solution_file,
        ref_file=args.ref,
        dim_values=dim_values,
        warmup=args.warmup,
        repeat=args.repeat,
        ptr_size_override=args.ptr_size,
        arch=arch,
        atol=args.atol,
        rtol=args.rtol,
        seed=args.seed,
        json_out=args.json_out,
        nvcc_bin=args.nvcc_bin,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
