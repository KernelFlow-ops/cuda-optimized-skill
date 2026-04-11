#!/usr/bin/env python3
"""Generic CUDA kernel benchmark (with optional correctness validation).

When --ref is provided:
  1. Validates kernel correctness against the reference implementation.
     Exits immediately if validation fails.
  2. Benchmarks the reference implementation.
  3. Benchmarks the CUDA kernel.
  4. Prints a combined summary with speedup.

When --ref is omitted:
  Benchmarks the CUDA kernel only.

Usage:
    python benchmark.py <solution.cu> [--DIM=VALUE ...] [options]
    python benchmark.py <solution.cu> --ref=<ref.py> [--DIM=VALUE ...] [options]

Examples:
    # Benchmark only
    python benchmark.py VectorAddition/solution.cu --N=1000000
    python benchmark.py MatMul/solution.cu --M=1024 --N=1024 --K=1024

    # Validate + benchmark
    python benchmark.py VectorAddition/solution.cu --ref=refs/vector_add.py --N=1000000
    python benchmark.py MatMul/solution.cu --ref=refs/matmul.py --M=1024 --N=1024 --K=1024

    # Custom warmup / repeat
    python benchmark.py solution.cu --N=4096 --warmup=10 --repeat=100

ref.py format
-------------
    import torch

    def reference(*, A, B, C, M, K, N, **kwargs):
        C[:] = (A.reshape(M, K) @ B.reshape(K, N)).reshape(-1)

    # Optional tolerance overrides
    atol = 1e-4
    rtol = 1e-3
"""

import re
import os
import sys
import json
import subprocess
import ctypes
import argparse
import importlib.util
import torch

# ---------------------------------------------------------------------------
# Type tables
# ---------------------------------------------------------------------------

SUPPORTED_TYPES = {
    "float*":          ("float*",          ctypes.c_void_p),
    "double*":         ("double*",         ctypes.c_void_p),
    "unsigned char*":  ("unsigned char*",  ctypes.c_void_p),
    "unsigned short*": ("unsigned short*", ctypes.c_void_p),
    "unsigned int*":   ("unsigned int*",   ctypes.c_void_p),
    "char*":           ("char*",           ctypes.c_void_p),
    "short*":          ("short*",          ctypes.c_void_p),
    "long*":           ("long*",           ctypes.c_void_p),
    "int*":            ("int*",            ctypes.c_void_p),
    "int":             ("int",             ctypes.c_int),
    "long":            ("long",            ctypes.c_long),
    "size_t":          ("size_t",          ctypes.c_size_t),
    "unsigned int":    ("unsigned int",    ctypes.c_uint),
    "unsigned short":  ("unsigned short",  ctypes.c_ushort),
    "unsigned char":   ("unsigned char",   ctypes.c_ubyte),
    "char":            ("char",            ctypes.c_char),
    "short":           ("short",           ctypes.c_short),
}

DTYPE_MAP = {
    "float*":          torch.float32,
    "double*":         torch.float64,
    "int*":            torch.int32,
    "long*":           torch.int64,
    "short*":          torch.int16,
    "char*":           torch.int8,
    "unsigned char*":  torch.uint8,
    "unsigned short*": getattr(torch, "uint16", torch.int16),
    "unsigned int*":   getattr(torch, "uint32", torch.int32),
}

INT_TYPES = {"int", "long", "size_t", "unsigned int"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_solve_signature(cu_file: str):
    """Extract parameter list from `extern "C" void solve(...)` in a .cu file."""
    with open(cu_file, "r") as f:
        content = f.read()

    pattern = r'extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(
            f'Cannot find \'extern "C" void solve(...)\' in {cu_file}'
        )

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


_STRIP_INCLUDES = re.compile(
    r'^\s*#\s*include\s*<__clang_cuda[^>]*>\s*$', re.MULTILINE
)


def _preprocess_cu(cu_file: str) -> str:
    """Strip clang-specific includes that break nvcc. Returns path to clean file."""
    with open(cu_file, "r") as f:
        src = f.read()
    cleaned = _STRIP_INCLUDES.sub("", src)
    if cleaned == src:
        return cu_file
    tmp = cu_file + ".nvcc_clean.cu"
    with open(tmp, "w") as f:
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


def load_reference(ref_file: str):
    """Import a Python reference file and return its module."""
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Reference file not found: {ref_file}")
    spec = importlib.util.spec_from_file_location("_ref_module", ref_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "reference"):
        raise AttributeError(
            f"'{ref_file}' must define a `reference(**kwargs)` function."
        )
    return mod


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
    """Run fn for warmup + repeat iterations and return per-iter ms timings.

    start_event / end_event are placed outside the loop so that only GPU
    execution time is measured and CPU scheduling overhead between iterations
    is excluded.  The total elapsed time is divided by ``repeat`` to get the
    average per-iteration latency.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

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

def _print_results(label, avg, med, mn, mx, total_ptr_bytes, ptr_elems,
                   cu_file, dim_values, arch, ref_avg=None):
    """Print benchmark results table; append speedup line when ref_avg is given."""
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print()
    print("=" * 55)
    print(f"  {label}")
    print(f"  Kernel       : {os.path.basename(cu_file)}")
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
        print(f"  ~Bandwidth   : {bw:>10.2f} GB/s  (all ptrs, rough)")
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

        max_diff  = (kt - rt).abs().max().item()
        mean_diff = (kt - rt).abs().mean().item()
        rel_err   = ((kt - rt).abs() / rt.abs().clamp(min=1e-8)).mean().item()

        status_str = _color("PASS" if match else "FAIL", match)
        print(f"  [{status_str}]  {pname}  ({ptype})")
        print(f"         max |delta|   = {max_diff:.6e}")
        print(f"         mean |delta|  = {mean_diff:.6e}")
        print(f"         mean rel  = {rel_err:.6e}")

        if not match:
            diff_mask = ~torch.isclose(kt, rt, atol=atol, rtol=rtol)
            bad_idx   = diff_mask.nonzero(as_tuple=True)[0]
            n_bad     = bad_idx.numel()
            print(f"         mismatches: {n_bad} / {kt.numel()}")
            if n_bad > 0:
                idx = bad_idx[0].item()
                print(f"         first bad   @ idx={idx}:  "
                      f"kernel={kt[idx].item():.6f}  ref={rt[idx].item():.6f}")

        k_preview = kernel_tensors[pname][:PREVIEW].float().cpu().tolist()
        r_preview = ref_tensors[pname][:PREVIEW].float().cpu().tolist()
        print(f"         kernel[:{PREVIEW}] = {_fmt_vals(k_preview)}")
        print(f"         ref   [:{PREVIEW}] = {_fmt_vals(r_preview)}")
        print()

    return all_pass


# ---------------------------------------------------------------------------
# Setup: compile + allocate buffers
# ---------------------------------------------------------------------------

def _setup(cu_file, dim_values, ptr_size_override, arch, nvcc_bin, seed=None):
    """Parse signature, compile kernel, allocate tensors.

    Returns:
        lib: loaded shared library
        params: parsed parameter list
        kernel_tensors: dict of GPU tensors keyed by param name
        kernel_call_args: ctypes arg list for lib.solve(...)
        argtypes: ctypes argtypes list
        output_params: list of (pname, ptype) for non-const pointer params
        ptr_elems: element count used for pointer buffers
        total_ptr_bytes: total bytes across all pointer tensors
    """
    # -- signature + compile --------------------------------------------------
    params = parse_solve_signature(cu_file)
    sig_str = ", ".join(f"{'const ' if c else ''}{t} {n}" for t, n, c in params)
    print(f"[signature] solve({sig_str})\n")

    lib_ext = ".dll" if os.name == "nt" else ".so"
    so_file = os.path.splitext(cu_file)[0] + lib_ext
    compile_cu(cu_file, so_file, arch, nvcc_bin)
    lib = ctypes.CDLL(so_file)

    # -- validate dimensions --------------------------------------------------
    for ptype, pname, _ in params:
        if ptype in INT_TYPES and pname not in dim_values:
            raise ValueError(
                f"Missing dimension: --{pname}=<value>  (required by kernel signature)"
            )

    int_vals = [dim_values[pname]
                for ptype, pname, _ in params if ptype in INT_TYPES]
    ptr_elems = _determine_ptr_elems(int_vals, ptr_size_override)

    # -- allocate tensors -----------------------------------------------------
    if seed is not None:
        torch.manual_seed(seed)

    kernel_tensors: dict = {}
    output_params = []
    kernel_call_args = []
    argtypes = []

    print("[buffers]")
    for ptype, pname, is_const in params:
        if ptype in DTYPE_MAP:
            dtype = DTYPE_MAP[ptype]
            if dtype.is_floating_point:
                t = torch.randn(ptr_elems, device="cuda", dtype=dtype)
            else:
                t = torch.zeros(ptr_elems, device="cuda", dtype=dtype).random_()
            kernel_tensors[pname] = t
            if not is_const:
                output_params.append((pname, ptype))
            kernel_call_args.append(ctypes.c_void_p(t.data_ptr()))
            argtypes.append(ctypes.c_void_p)
            role = "input" if is_const else "output"
            eb   = t.element_size()
            print(
                f"  {pname:>10s} : {ptype:<16s} [{role:>6s}] "
                f"{ptr_elems} elems  ({ptr_elems * eb / 1024 / 1024:.1f} MB)"
            )
        elif ptype in SUPPORTED_TYPES:
            _, ctype = SUPPORTED_TYPES[ptype]
            val = dim_values[pname]
            kernel_call_args.append(ctype(val))
            argtypes.append(ctype)
            print(f"  {pname:>10s} : {ptype:<16s} = {val}")

    lib.solve.restype  = None
    lib.solve.argtypes = argtypes

    total_ptr_bytes = sum(t.nelement() * t.element_size()
                          for t in kernel_tensors.values())

    return (lib, params, kernel_tensors, kernel_call_args,
            argtypes, output_params, ptr_elems, total_ptr_bytes)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cu_file, ref_file, dim_values, warmup, repeat,
        ptr_size_override, arch, atol, rtol, seed, json_out="", nvcc_bin="nvcc"):
    """Main benchmark pipeline.

    Steps:
      1. If ref provided: run kernel + ref once and validate correctness.
         Exit immediately on failure.
      2. If ref provided: benchmark reference.
      3. Benchmark kernel.
      4. Print summary results.
    """
    has_ref = bool(ref_file)

    # -- load reference module ------------------------------------------------
    ref_fn = None
    ref_kwargs = None
    ref_tensors = None
    _atol = atol
    _rtol = rtol

    if has_ref:
        ref_mod = load_reference(ref_file)
        ref_fn  = ref_mod.reference
        _atol   = float(getattr(ref_mod, "atol", atol))
        _rtol   = float(getattr(ref_mod, "rtol", rtol))
        print(f"[reference] {ref_file}  (atol={_atol}, rtol={_rtol})\n")

    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    result = {
        "cu_file": os.path.abspath(cu_file),
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

    # -- compile + allocate ---------------------------------------------------
    (lib, params, kernel_tensors, kernel_call_args,
     argtypes, output_params, ptr_elems, total_ptr_bytes) = _setup(
        cu_file,
        dim_values,
        ptr_size_override,
        arch,
        nvcc_bin,
        seed=seed if has_ref else None,
    )
    result["signature"] = [
        {"type": ptype, "name": pname, "is_const": is_const}
        for ptype, pname, is_const in params
    ]
    result["ptr_elems"] = ptr_elems
    result["total_ptr_bytes"] = total_ptr_bytes
    result["correctness"]["output_tensor_count"] = len(output_params)

    if not output_params and has_ref:
        print("\n[warn] No output tensors detected (all pointer params are const). "
              "Nothing to validate.", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Step 1: correctness check (only when ref is provided)
    # -------------------------------------------------------------------------
    if has_ref:
        # Build ref tensors as clones from the same seed-initialised kernel tensors
        ref_tensors = {pname: t.clone() for pname, t in kernel_tensors.items()}

        # Build ref kwargs
        ref_kwargs = {}
        for ptype, pname, _ in params:
            if ptype in DTYPE_MAP:
                ref_kwargs[pname] = ref_tensors[pname]
            else:
                ref_kwargs[pname] = dim_values[pname]

        print("\n[kernel]    running ... ", end="", flush=True)
        lib.solve(*kernel_call_args)
        torch.cuda.synchronize()
        print("done")

        print("[reference] running ... ", end="", flush=True)
        ref_fn(**ref_kwargs)
        torch.cuda.synchronize()
        print("done")

        validation_passed = _validate_outputs(
            kernel_tensors, ref_tensors, output_params, _atol, _rtol
        )

        # -- validation summary -----------------------------------------------
        print("=" * 60)
        print(f"  Kernel    : {os.path.basename(cu_file)}")
        print(f"  Reference : {os.path.basename(ref_file)}")
        print(f"  GPU       : {gpu_name}")
        print(f"  Arch      : {arch}")
        print(f"  Dims      : {dim_values}")
        print(f"  Buf/ptr   : {ptr_elems} elems")
        print(f"  Tolerance : atol={_atol}  rtol={_rtol}")
        print("-" * 60)
        result_str = "ALL PASS" if validation_passed else "FAILED"
        print(f"  Result    : {_color(result_str, validation_passed)}")
        print("=" * 60)

        result["correctness"]["passed"] = validation_passed
        if not validation_passed:
            _write_json_out(json_out, result)
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 2: benchmark reference (only when ref is provided)
    # -------------------------------------------------------------------------
    times_ref = None
    if has_ref:
        print(f"\n[warmup] reference  {warmup} iterations ...")
        times_ref = _time_iterations(lambda: ref_fn(**ref_kwargs), warmup, repeat)
        print(f"[bench]  reference  {repeat} iterations ... done")

    # -------------------------------------------------------------------------
    # Step 3: benchmark kernel
    # -------------------------------------------------------------------------
    if not has_ref:
        # Without ref, show before/after previews for a quick sanity check
        PREVIEW = 8
        tensor_info = [
            (pname, ptype, "input" if is_const else "output", kernel_tensors[pname])
            for ptype, pname, is_const in params if ptype in DTYPE_MAP
        ]

        print(f"\n[preview] first {PREVIEW} elements before kernel call:")
        for name, ptype, role, t in tensor_info:
            tag = "IN " if role == "input" else "OUT"
            print(f"  {tag} {name:>6s} = {_fmt_vals(t[:PREVIEW].cpu().tolist())}")

        lib.solve(*kernel_call_args)
        torch.cuda.synchronize()

        print(f"\n[preview] first {PREVIEW} elements after 1 kernel call:")
        for name, ptype, role, t in tensor_info:
            tag = "IN " if role == "input" else "OUT"
            print(f"  {tag} {name:>6s} = {_fmt_vals(t[:PREVIEW].cpu().tolist())}")

    print(f"\n[warmup] kernel  {warmup} iterations ...")
    times_kernel = _time_iterations(lambda: lib.solve(*kernel_call_args), warmup, repeat)
    print(f"[bench]  kernel  {repeat} iterations ... done")

    # -------------------------------------------------------------------------
    # Step 4: print summary
    # -------------------------------------------------------------------------
    avg_k, med_k, mn_k, mx_k = _stats(times_kernel)

    result["kernel"] = _stats_dict(times_kernel)
    result["kernel"]["bandwidth_gbps_rough"] = (
        total_ptr_bytes / (avg_k / 1000) / 1e9 if avg_k > 0 else None
    )

    if has_ref:
        avg_r, med_r, mn_r, mx_r = _stats(times_ref)
        result["reference"] = _stats_dict(times_ref)
        result["reference"]["bandwidth_gbps_rough"] = (
            total_ptr_bytes / (avg_r / 1000) / 1e9 if avg_r > 0 else None
        )
        result["speedup_vs_reference"] = avg_r / avg_k if avg_k > 0 else None
        _print_results(
            "CUDA Kernel", avg_k, med_k, mn_k, mx_k,
            total_ptr_bytes, ptr_elems, cu_file, dim_values, arch, ref_avg=avg_r,
        )
        _print_results(
            f"Reference ({os.path.basename(ref_file)})",
            avg_r, med_r, mn_r, mx_r,
            total_ptr_bytes, ptr_elems, cu_file, dim_values, arch,
        )
    else:
        _print_results(
            "CUDA Kernel", avg_k, med_k, mn_k, mx_k,
            total_ptr_bytes, ptr_elems, cu_file, dim_values, arch,
        )

    _write_json_out(json_out, result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generic CUDA kernel benchmark (with optional validation)",
        epilog=(
            "Dimension args: pass --NAME=VALUE for each int param in solve().\n"
            "ref.py must define `reference(**kwargs)` and may set module-level atol/rtol."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cu_file", help="Path to .cu solution file")
    parser.add_argument("--ref", type=str, default="",
                        help="Path to reference .py file; enables validation + reference benchmark")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--repeat", type=int, default=20,
                        help="Benchmark iterations (default: 20)")
    parser.add_argument("--ptr-size", type=int, default=0,
                        help="Override element count for all pointer buffers")
    parser.add_argument("--arch", type=str, default="",
                        help="GPU arch, e.g. sm_90 (auto-detected if omitted)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index (default: 0)")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Absolute tolerance for validation (default: 1e-4)")
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="Relative tolerance for validation (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for input tensors when validating (default: 42)")
    parser.add_argument("--json-out", type=str, default="",
                        help="Optional path to write structured benchmark results as JSON")
    parser.add_argument("--nvcc-bin", type=str, default="nvcc",
                        help="NVCC executable or full path")

    args, unknown = parser.parse_known_args()

    dim_values: dict = {}
    for u in unknown:
        if u.startswith("--") and "=" in u:
            key, val = u[2:].split("=", 1)
            dim_values[key] = int(val)
        else:
            print(f"Warning: ignoring unknown arg '{u}'", file=sys.stderr)

    torch.cuda.set_device(args.gpu)
    arch = args.arch if args.arch else detect_arch(args.gpu)

    run(
        cu_file           = args.cu_file,
        ref_file          = args.ref,
        dim_values        = dim_values,
        warmup            = args.warmup,
        repeat            = args.repeat,
        ptr_size_override = args.ptr_size,
        arch              = arch,
        atol              = args.atol,
        rtol              = args.rtol,
        seed              = args.seed,
        json_out          = args.json_out,
        nvcc_bin          = args.nvcc_bin,
    )


if __name__ == "__main__":
    main()
