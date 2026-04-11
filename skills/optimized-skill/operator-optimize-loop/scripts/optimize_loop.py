#!/usr/bin/env python3
"""Evaluate one CUDA kernel iteration inside an optimization loop.

This script does the repeatable mechanics for each version:
1. Snapshot the current kernel into a run directory.
2. Run correctness validation + benchmark via benchmark.py.
3. Generate targeted and full Nsight Compute reports.
4. Import summary/details text from the generated .ncu-rep files.
5. Update a run manifest and final summary so Claude can compare iterations.

Code generation and optimization decisions are intentionally left to the skill.
The skill edits or creates the next kernel version, then invokes this script again
for the next iteration in the same run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


TARGETED_SECTIONS = [
    "LaunchStats",
    "Occupancy",
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "SchedulerStats",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def valid_report_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_command(cmd: list[str], stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except OSError as exc:
        result = subprocess.CompletedProcess(cmd, 127, "", str(exc))
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    return result


def trim_output(text: str, max_lines: int = 20) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + ["..."])


def add_requirement(
    requirements: list[dict[str, Any]],
    errors: list[str],
    name: str,
    ok: bool,
    detail: str,
    *,
    required: bool = True,
) -> None:
    requirements.append(
        {
            "name": name,
            "ok": ok,
            "detail": detail,
            "required": required,
        }
    )
    if required and not ok:
        errors.append(f"{name}: {detail}")


def run_probe(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except OSError as exc:
        return {
            "command": shell_join(cmd),
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": shell_join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def candidate_has_path(candidate: str) -> bool:
    return any(sep in candidate for sep in ("\\", "/"))


def find_cuda_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    return roots


def find_ncu_roots() -> list[Path]:
    roots: list[Path] = []
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        nvidia_dir = Path(program_files) / "NVIDIA Corporation"
        if nvidia_dir.exists():
            roots.extend(sorted(nvidia_dir.glob("Nsight Compute*")))
    return roots


def resolve_executable(candidate: str, tool_name: str) -> str:
    candidate = candidate.strip().strip('"')
    direct = Path(candidate).expanduser()
    if direct.exists():
        return str(direct.resolve())

    resolved = shutil.which(candidate)
    if resolved:
        return resolved

    if candidate_has_path(candidate):
        return ""

    extra_names = [candidate]
    if os.name == "nt" and not Path(candidate).suffix:
        extra_names.extend([f"{candidate}.exe", f"{candidate}.bat", f"{candidate}.cmd"])

    search_roots: list[Path] = []
    if tool_name == "nvcc":
        search_roots.extend(root / "bin" for root in find_cuda_roots())
    elif tool_name == "ncu":
        search_roots.extend(find_ncu_roots())

    for root in search_roots:
        for name in extra_names:
            probe = root / name
            if probe.exists():
                return str(probe.resolve())
    return ""


def probe_executable(candidate: str, tool_name: str, version_args: list[str]) -> dict[str, Any]:
    resolved = resolve_executable(candidate, tool_name)
    info: dict[str, Any] = {
        "requested": candidate,
        "resolved": resolved,
        "exists": bool(resolved),
        "version_command": "",
        "version_returncode": None,
        "version_output": "",
    }
    if not resolved:
        return info

    probe = run_probe([resolved, *version_args])
    output = (probe["stdout"] or probe["stderr"]).strip()
    info["version_command"] = probe["command"]
    info["version_returncode"] = probe["returncode"]
    info["version_output"] = trim_output(output)
    return info


def probe_nvidia_smi() -> dict[str, Any]:
    resolved = shutil.which("nvidia-smi")
    info: dict[str, Any] = {
        "exists": bool(resolved),
        "resolved": resolved or "",
        "query_command": "",
        "returncode": None,
        "query_output": "",
        "gpus": [],
    }
    if not resolved:
        return info

    primary = run_probe(
        [resolved, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"]
    )
    probe = primary
    if primary["returncode"] != 0 or not primary["stdout"].strip():
        fallback = run_probe([resolved, "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if fallback["returncode"] == 0 and fallback["stdout"].strip():
            probe = fallback

    info["query_command"] = probe["command"]
    info["returncode"] = probe["returncode"]
    info["query_output"] = trim_output((probe["stdout"] or probe["stderr"]).strip())

    if probe["returncode"] == 0:
        for line in probe["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                info["gpus"].append(
                    {
                        "name": parts[0],
                        "compute_capability": parts[1],
                        "driver_version": parts[2],
                    }
                )
            elif len(parts) >= 2:
                info["gpus"].append(
                    {
                        "name": parts[0],
                        "compute_capability": "",
                        "driver_version": parts[1],
                    }
                )
    return info


def probe_torch_cuda(gpu_index: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "importable": False,
        "version": "",
        "cuda_version": "",
        "cuda_available": False,
        "device_count": 0,
        "selected_gpu_index": gpu_index,
        "selected_gpu_name": "",
        "selected_gpu_compute_capability": "",
        "selected_sm": "",
        "error": "",
    }
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        info["error"] = str(exc)
        return info

    info["importable"] = True
    info["version"] = getattr(torch, "__version__", "")
    info["cuda_version"] = getattr(torch.version, "cuda", "") or ""

    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = int(torch.cuda.device_count())
            if 0 <= gpu_index < info["device_count"]:
                info["selected_gpu_name"] = torch.cuda.get_device_name(gpu_index)
                major, minor = torch.cuda.get_device_capability(gpu_index)
                info["selected_gpu_compute_capability"] = f"{major}.{minor}"
                info["selected_sm"] = f"sm_{major}{minor}"
    except Exception as exc:  # pragma: no cover - environment dependent
        info["error"] = str(exc)
    return info


def collect_preflight(
    args: argparse.Namespace,
    benchmark_script: Path,
    cu_file: Path,
    ref_file: Path | None,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    requirements: list[dict[str, Any]] = []

    preflight: dict[str, Any] = {
        "checked_at": now_iso(),
        "ready": False,
        "python_executable": sys.executable,
        "python_version": sys.version.splitlines()[0],
        "selected_gpu_index": args.gpu,
        "env_vars": {
            "CUDA_PATH": os.environ.get("CUDA_PATH", ""),
            "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
            "CUDA_ROOT": os.environ.get("CUDA_ROOT", ""),
        },
        "requirements": requirements,
        "warnings": warnings,
        "errors": errors,
    }

    add_requirement(
        requirements,
        errors,
        "kernel file",
        cu_file.exists(),
        str(cu_file),
    )
    add_requirement(
        requirements,
        errors,
        "kernel suffix",
        cu_file.suffix.lower() == ".cu",
        cu_file.suffix or "(no suffix)",
    )
    add_requirement(
        requirements,
        errors,
        "benchmark.py",
        benchmark_script.exists(),
        str(benchmark_script),
    )
    add_requirement(
        requirements,
        errors,
        "reference file",
        ref_file is None or ref_file.exists(),
        "not provided" if ref_file is None else str(ref_file),
    )

    torch_info = probe_torch_cuda(args.gpu)
    preflight["torch"] = torch_info
    add_requirement(
        requirements,
        errors,
        "PyTorch import",
        torch_info["importable"],
        torch_info["version"] if torch_info["importable"] else (torch_info["error"] or "torch import failed"),
    )
    add_requirement(
        requirements,
        errors,
        "CUDA runtime",
        torch_info["cuda_available"],
        (
            f"torch CUDA {torch_info['cuda_version']}"
            if torch_info["cuda_available"]
            else (torch_info["error"] or "torch.cuda.is_available() returned false")
        ),
    )
    selected_gpu_ok = (
        torch_info["cuda_available"] and 0 <= args.gpu < int(torch_info["device_count"])
    )
    add_requirement(
        requirements,
        errors,
        f"GPU index {args.gpu}",
        selected_gpu_ok,
        (
            f"{torch_info['selected_gpu_name']} ({torch_info['selected_sm']})"
            if selected_gpu_ok
            else f"available device count: {torch_info['device_count']}"
        ),
    )

    nvidia_smi_info = probe_nvidia_smi()
    preflight["nvidia_smi"] = nvidia_smi_info
    if not nvidia_smi_info["exists"]:
        warnings.append("nvidia-smi not found; GPU model falls back to PyTorch detection.")
    elif nvidia_smi_info.get("returncode") not in (None, 0):
        warnings.append("nvidia-smi is present but GPU query failed.")

    gpu_info: dict[str, Any] = {
        "name": torch_info.get("selected_gpu_name", ""),
        "compute_capability": torch_info.get("selected_gpu_compute_capability", ""),
        "sm": torch_info.get("selected_sm", ""),
        "driver_version": "",
        "source": "torch" if torch_info.get("selected_gpu_name") else "",
    }
    if nvidia_smi_info.get("gpus") and args.gpu < len(nvidia_smi_info["gpus"]):
        smi_gpu = nvidia_smi_info["gpus"][args.gpu]
        if smi_gpu.get("name"):
            gpu_info["name"] = smi_gpu["name"]
            gpu_info["source"] = "nvidia-smi"
        if smi_gpu.get("compute_capability"):
            gpu_info["compute_capability"] = smi_gpu["compute_capability"]
            if not gpu_info["sm"] and "." in smi_gpu["compute_capability"]:
                major, minor = smi_gpu["compute_capability"].split(".", 1)
                gpu_info["sm"] = f"sm_{major}{minor}"
        if smi_gpu.get("driver_version"):
            gpu_info["driver_version"] = smi_gpu["driver_version"]
    preflight["gpu"] = gpu_info

    nvcc_info = probe_executable(args.nvcc_bin, "nvcc", ["--version"])
    preflight["nvcc"] = nvcc_info
    add_requirement(
        requirements,
        errors,
        "nvcc executable",
        nvcc_info["exists"],
        nvcc_info["resolved"] or f"cannot resolve {args.nvcc_bin}",
    )
    if nvcc_info["exists"] and nvcc_info.get("version_returncode") not in (None, 0):
        warnings.append("nvcc exists but `--version` did not exit cleanly.")

    ncu_info = probe_executable(args.ncu_bin, "ncu", ["--version"])
    preflight["ncu"] = ncu_info
    add_requirement(
        requirements,
        errors,
        "ncu executable",
        ncu_info["exists"],
        ncu_info["resolved"] or f"cannot resolve {args.ncu_bin}",
    )
    if ncu_info["exists"] and ncu_info.get("version_returncode") not in (None, 0):
        warnings.append("ncu exists but `--version` did not exit cleanly.")

    if args.arch and gpu_info.get("sm") and args.arch != gpu_info["sm"]:
        warnings.append(
            f"--arch={args.arch} does not match selected GPU capability {gpu_info['sm']}."
        )

    preflight["ready"] = not errors
    return preflight


def render_preflight_markdown(preflight: dict[str, Any]) -> str:
    lines = [
        "# CUDA Optimization Loop Preflight",
        "",
        "## Status",
        f"- ready: {'yes' if preflight.get('ready') else 'no'}",
        f"- checked at: {preflight.get('checked_at', '')}",
        f"- python: {preflight.get('python_executable', '')}",
        f"- python version: {preflight.get('python_version', '')}",
        f"- selected gpu index: {preflight.get('selected_gpu_index')}",
        "",
        "## Required environment",
        "",
        "| Requirement | Status | Detail |",
        "| --- | --- | --- |",
    ]

    for item in preflight.get("requirements", []):
        status = "ok" if item.get("ok") else "missing"
        detail = str(item.get("detail", "")).replace("\n", "<br>")
        lines.append(f"| {item.get('name')} | {status} | {detail} |")

    gpu = preflight.get("gpu") or {}
    torch_info = preflight.get("torch") or {}
    nvidia_smi = preflight.get("nvidia_smi") or {}
    nvcc = preflight.get("nvcc") or {}
    ncu = preflight.get("ncu") or {}

    lines.extend(
        [
            "",
            "## GPU",
            f"- model: {gpu.get('name') or 'unknown'}",
            f"- compute capability: {gpu.get('compute_capability') or 'unknown'}",
            f"- sm: {gpu.get('sm') or 'unknown'}",
            f"- driver version: {gpu.get('driver_version') or 'unknown'}",
            f"- source: {gpu.get('source') or 'unknown'}",
            f"- torch: {torch_info.get('version') or 'not importable'}",
            f"- torch cuda: {torch_info.get('cuda_version') or 'unknown'}",
            f"- device count: {torch_info.get('device_count')}",
            f"- nvidia-smi: {nvidia_smi.get('resolved') or 'not found'}",
            "",
            "## Tools",
            f"- nvcc requested: {nvcc.get('requested', '')}",
            f"- nvcc resolved: {nvcc.get('resolved') or 'not found'}",
            f"- nvcc version: {nvcc.get('version_output') or 'n/a'}",
            f"- ncu requested: {ncu.get('requested', '')}",
            f"- ncu resolved: {ncu.get('resolved') or 'not found'}",
            f"- ncu version: {ncu.get('version_output') or 'n/a'}",
            "",
            "## Environment variables",
            f"- CUDA_PATH: {preflight.get('env_vars', {}).get('CUDA_PATH') or '(unset)'}",
            f"- CUDA_HOME: {preflight.get('env_vars', {}).get('CUDA_HOME') or '(unset)'}",
            f"- CUDA_ROOT: {preflight.get('env_vars', {}).get('CUDA_ROOT') or '(unset)'}",
            "",
            "## Errors",
        ]
    )

    if preflight.get("errors"):
        lines.extend(f"- {item}" for item in preflight["errors"])
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings"])
    if preflight.get("warnings"):
        lines.extend(f"- {item}" for item in preflight["warnings"])
    else:
        lines.append("- none")

    return "\n".join(lines)


def ensure_run_dir(cu_file: Path, run_dir_arg: str) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (cu_file.resolve().parent / "optimize_runs" / f"run_{ts}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_manifest(manifest_path: Path, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path, None)
    if manifest is None:
        manifest = {
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "run_dir": str(run_dir),
            "source_cu_file": str(Path(args.cu_file).resolve()),
            "reference_file": str(Path(args.ref).resolve()) if args.ref else "",
            "max_iterations": args.max_iterations,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "gpu": args.gpu,
            "arch": args.arch,
            "ptr_size": args.ptr_size,
            "seed": args.seed,
            "dims_args": list(args.dim_args),
            "preflight": {},
            "iterations": [],
            "best_iteration": None,
            "best_kernel_path": "",
        }
    return manifest


def pick_iteration_index(manifest: dict[str, Any], requested_iteration: int) -> int:
    if requested_iteration >= 0:
        return requested_iteration
    return len(manifest.get("iterations", []))


def build_benchmark_cmd(args: argparse.Namespace, benchmark_script: Path, snapshot_cu: Path, benchmark_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(benchmark_script),
        str(snapshot_cu),
        f"--warmup={args.warmup}",
        f"--repeat={args.repeat}",
        f"--gpu={args.gpu}",
        f"--seed={args.seed}",
        f"--atol={args.atol}",
        f"--rtol={args.rtol}",
        f"--json-out={benchmark_json}",
        f"--nvcc-bin={args.nvcc_bin}",
    ]
    if args.ref:
        cmd.append(f"--ref={Path(args.ref).resolve()}")
    if args.arch:
        cmd.append(f"--arch={args.arch}")
    if args.ptr_size > 0:
        cmd.append(f"--ptr-size={args.ptr_size}")
    cmd.extend(args.dim_args)
    return cmd


def build_targeted_ncu_cmd(args: argparse.Namespace, bench_cmd: list[str], out_prefix: Path) -> list[str]:
    cmd = [
        args.ncu_bin,
        "--target-processes",
        "all",
        "--profile-from-start",
        "on",
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
    ]
    if args.kernel_name_regex:
        cmd.extend(["--kernel-name-base", "demangled", "-k", f"regex:{args.kernel_name_regex}"])
    for section in TARGETED_SECTIONS:
        cmd.extend(["--section", section])
    cmd.extend(["-o", str(out_prefix), "-f"])
    cmd.extend(bench_cmd)
    return cmd


def build_full_ncu_cmd(args: argparse.Namespace, bench_cmd: list[str], out_prefix: Path) -> list[str]:
    cmd = [
        args.ncu_bin,
        "--target-processes",
        "all",
        "--profile-from-start",
        "on",
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
        "--set",
        "full",
    ]
    if args.kernel_name_regex:
        cmd.extend(["--kernel-name-base", "demangled", "-k", f"regex:{args.kernel_name_regex}"])
    cmd.extend(["-o", str(out_prefix), "-f"])
    cmd.extend(bench_cmd)
    return cmd


def import_ncu_report(args: argparse.Namespace, rep_path: Path, summary_txt: Path, details_txt: Path) -> dict[str, Any]:
    summary_cmd = [args.ncu_bin, "--import", str(rep_path), "--print-summary", "per-kernel"]
    details_cmd = [args.ncu_bin, "--import", str(rep_path), "--page", "details"]

    summary_res = run_command(summary_cmd, summary_txt, summary_txt.with_suffix(".stderr.txt"))
    details_res = run_command(details_cmd, details_txt, details_txt.with_suffix(".stderr.txt"))

    return {
        "summary_command": shell_join(summary_cmd),
        "summary_txt": str(summary_txt),
        "summary_rc": summary_res.returncode,
        "details_command": shell_join(details_cmd),
        "details_txt": str(details_txt),
        "details_rc": details_res.returncode,
    }


def choose_best_iteration(iterations: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for item in iterations:
        bench = item.get("benchmark_result") or {}
        kernel = bench.get("kernel") or {}
        correctness = bench.get("correctness") or {}
        if not item.get("full_report_exists"):
            continue
        if bench.get("has_reference") and not correctness.get("passed"):
            continue
        if kernel.get("median_ms") is None or kernel.get("average_ms") is None:
            continue
        candidates.append(item)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item["benchmark_result"]["kernel"]["median_ms"],
            item["benchmark_result"]["kernel"]["average_ms"],
            item["iteration"],
        ),
    )


def render_iteration_markdown(record: dict[str, Any]) -> str:
    bench = record.get("benchmark_result") or {}
    correctness = bench.get("correctness") or {}
    kernel = bench.get("kernel") or {}
    reference = bench.get("reference") or {}
    lines = [
        f"# Iteration v{record['iteration']}",
        "",
        "## Status",
        f"- benchmark rc: {record['benchmark_rc']}",
        f"- targeted ncu rc: {record.get('targeted_ncu_rc')}",
        f"- full ncu rc: {record.get('full_ncu_rc')}",
        f"- snapshot kernel: {record['snapshot_cu']}",
        f"- correctness checked: {correctness.get('checked')}",
        f"- correctness passed: {correctness.get('passed')}",
        "",
        "## Commands",
        f"- benchmark: `{record['benchmark_command']}`",
        f"- targeted ncu: `{record.get('targeted_ncu_command', '')}`",
        f"- full ncu: `{record.get('full_ncu_command', '')}`",
        "",
        "## Benchmark",
        f"- kernel average ms: {kernel.get('average_ms')}",
        f"- kernel median ms: {kernel.get('median_ms')}",
        f"- kernel min ms: {kernel.get('min_ms')}",
        f"- kernel max ms: {kernel.get('max_ms')}",
        f"- speedup vs reference: {bench.get('speedup_vs_reference')}",
        f"- reference average ms: {reference.get('average_ms')}",
        "",
        "## Artifacts",
        f"- benchmark json: {record['benchmark_json']}",
        f"- targeted report: {record.get('targeted_report')}",
        f"- full report: {record.get('full_report')}",
        f"- targeted summary: {record.get('targeted_import', {}).get('summary_txt')}",
        f"- full summary: {record.get('full_import', {}).get('summary_txt')}",
        f"- targeted details: {record.get('targeted_import', {}).get('details_txt')}",
        f"- full details: {record.get('full_import', {}).get('details_txt')}",
        "",
        "## Claude follow-up",
        "- Read the targeted/full summaries and details before deciding the next optimization.",
        "- Write the optimization hypothesis into optimization_proposal.md for this iteration.",
        "- Only promote this version as best when correctness passes and the full report exists.",
        "",
    ]
    return "\n".join(lines)


def render_final_summary(manifest: dict[str, Any]) -> str:
    preflight = manifest.get("preflight") or {}
    lines = [
        "# CUDA Optimization Loop Summary",
        "",
        "## Run info",
        f"- run dir: {manifest['run_dir']}",
        f"- source cu file: {manifest['source_cu_file']}",
        f"- reference file: {manifest['reference_file'] or 'not provided'}",
        f"- max iterations: {manifest['max_iterations']}",
        f"- warmup: {manifest['warmup']}",
        f"- repeat: {manifest['repeat']}",
        f"- gpu: {manifest['gpu']}",
        f"- arch: {manifest['arch'] or 'auto'}",
        f"- preflight ready: {'yes' if preflight.get('ready') else 'no'}" if preflight else "- preflight ready: not run",
        "",
        "## Environment",
        f"- gpu name: {preflight.get('gpu_name') or 'unknown'}" if preflight else "- gpu name: unknown",
        f"- compute capability: {preflight.get('gpu_compute_capability') or 'unknown'}" if preflight else "- compute capability: unknown",
        f"- nvcc: {preflight.get('nvcc_bin') or 'unknown'}" if preflight else "- nvcc: unknown",
        f"- ncu: {preflight.get('ncu_bin') or 'unknown'}" if preflight else "- ncu: unknown",
        f"- preflight report: {preflight.get('markdown_path')}" if preflight.get("markdown_path") else "- preflight report: not generated",
        "",
        "## Iterations",
        "",
        "| Iter | Correctness | Kernel median ms | Kernel avg ms | Full NCU | Snapshot |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]

    for item in manifest.get("iterations", []):
        bench = item.get("benchmark_result") or {}
        correctness = bench.get("correctness") or {}
        kernel = bench.get("kernel") or {}
        correctness_value = correctness.get("passed")
        if not bench.get("has_reference"):
            correctness_text = "not checked"
        else:
            correctness_text = "pass" if correctness_value else "fail"
        lines.append(
            "| v{iteration} | {correctness} | {median} | {avg} | {full_report} | {snapshot} |".format(
                iteration=item.get("iteration"),
                correctness=correctness_text,
                median=kernel.get("median_ms", "-"),
                avg=kernel.get("average_ms", "-"),
                full_report="yes" if item.get("full_report_exists") else "no",
                snapshot=item.get("snapshot_cu", "-"),
            )
        )

    best_iteration = manifest.get("best_iteration")
    lines.extend(["", "## Best version"])
    if best_iteration is None:
        lines.append("- No eligible best version yet. Need a correctness-passing iteration with a full NCU report.")
    else:
        best = next(item for item in manifest["iterations"] if item["iteration"] == best_iteration)
        bench = best.get("benchmark_result") or {}
        kernel = bench.get("kernel") or {}
        lines.extend(
            [
                f"- best iteration: v{best_iteration}",
                f"- best kernel path: {best.get('snapshot_cu')}",
                f"- full NCU report: {best.get('full_report')}",
                f"- targeted NCU report: {best.get('targeted_report')}",
                f"- kernel median ms: {kernel.get('median_ms')}",
                f"- kernel average ms: {kernel.get('average_ms')}",
                f"- speedup vs reference: {bench.get('speedup_vs_reference')}",
                f"- full NCU import summary: {best.get('full_import', {}).get('summary_txt')}",
                f"- full NCU import details: {best.get('full_import', {}).get('details_txt')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Required final answer checklist",
            "- Compare baseline vs best benchmark numbers.",
            "- Cite the best full NCU report path.",
            "- Summarize the bottleneck and the winning optimization idea.",
            "- Mention any failed iterations and why they were rejected.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Evaluate one iteration in a CUDA optimization loop")
    parser.add_argument("cu_file", help="Path to the current .cu kernel file")
    parser.add_argument("--ref", type=str, default="", help="Optional Python reference file")
    parser.add_argument("--run-dir", type=str, default="", help="Existing or new run directory")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration index; defaults to next index")
    parser.add_argument("--max-iterations", type=int, required=True, help="Required maximum iterations for the run")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for benchmark.py")
    parser.add_argument("--repeat", type=int, default=20, help="Benchmark repeat count for benchmark.py")
    parser.add_argument("--ptr-size", type=int, default=0, help="Pointer buffer element override for benchmark.py")
    parser.add_argument("--arch", type=str, default="", help="GPU arch, e.g. sm_90")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--atol", type=float, default=1e-4, help="Correctness absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Correctness relative tolerance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed passed to benchmark.py")
    parser.add_argument("--launch-skip", type=int, default=20, help="NCU launch-skip value")
    parser.add_argument("--launch-count", type=int, default=1, help="NCU launch-count value")
    parser.add_argument("--nvcc-bin", type=str, default="nvcc", help="NVCC executable or full path")
    parser.add_argument("--ncu-bin", type=str, default="ncu", help="Nsight Compute executable")
    parser.add_argument("--kernel-name-regex", type=str, default="", help="Optional NCU kernel filter regex")
    parser.add_argument("--preflight-only", action="store_true", help="Only run environment checks and exit")

    args, unknown = parser.parse_known_args()
    args.dim_args = [item for item in unknown if item.startswith("--") and "=" in item]
    return args, unknown


def main() -> int:
    args, unknown = parse_args()
    if any(not (item.startswith("--") and "=" in item) for item in unknown):
        bad = [item for item in unknown if not (item.startswith("--") and "=" in item)]
        print(f"Unsupported extra args: {bad}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[4]
    benchmark_script = repo_root / "skills" / "optimized-skill" / "kernel-benchmark" / "scripts" / "benchmark.py"
    cu_file = Path(args.cu_file).resolve()
    ref_file = Path(args.ref).resolve() if args.ref else None

    run_dir = ensure_run_dir(cu_file, args.run_dir)
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "final_summary.md"
    manifest = load_manifest(manifest_path, args, run_dir)
    preflight = collect_preflight(args, benchmark_script, cu_file, ref_file)
    preflight_json = run_dir / "preflight_check.json"
    preflight_md = run_dir / "preflight_check.md"
    write_json(preflight_json, preflight)
    write_text(preflight_md, render_preflight_markdown(preflight))
    manifest["preflight"] = {
        "checked_at": preflight["checked_at"],
        "ready": preflight["ready"],
        "gpu_name": (preflight.get("gpu") or {}).get("name", ""),
        "gpu_compute_capability": (preflight.get("gpu") or {}).get("compute_capability", ""),
        "nvcc_bin": (preflight.get("nvcc") or {}).get("resolved", ""),
        "ncu_bin": (preflight.get("ncu") or {}).get("resolved", ""),
        "json_path": str(preflight_json),
        "markdown_path": str(preflight_md),
        "errors": list(preflight.get("errors", [])),
        "warnings": list(preflight.get("warnings", [])),
    }
    write_json(manifest_path, manifest)
    write_text(summary_path, render_final_summary(manifest))

    if preflight.get("nvcc", {}).get("resolved"):
        args.nvcc_bin = preflight["nvcc"]["resolved"]
    if preflight.get("ncu", {}).get("resolved"):
        args.ncu_bin = preflight["ncu"]["resolved"]

    if args.preflight_only or not preflight.get("ready"):
        if not preflight.get("ready"):
            print(f"Preflight failed. See {preflight_md}", file=sys.stderr)
            for item in preflight.get("errors", []):
                print(f"- {item}", file=sys.stderr)
            return 2
        return 0

    iteration = pick_iteration_index(manifest, args.iteration)
    iter_dir = run_dir / f"iter_v{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    snapshot_cu = iter_dir / f"{cu_file.stem}_v{iteration}{cu_file.suffix}"
    shutil.copy2(cu_file, snapshot_cu)
    if ref_file:
        shutil.copy2(ref_file, iter_dir / ref_file.name)

    benchmark_json = iter_dir / "benchmark_result.json"
    benchmark_stdout = iter_dir / "benchmark.stdout.txt"
    benchmark_stderr = iter_dir / "benchmark.stderr.txt"
    bench_cmd = build_benchmark_cmd(args, benchmark_script, snapshot_cu, benchmark_json)
    bench_res = run_command(bench_cmd, benchmark_stdout, benchmark_stderr)
    bench_json = read_json(benchmark_json, {})

    record: dict[str, Any] = {
        "iteration": iteration,
        "created_at": now_iso(),
        "snapshot_cu": str(snapshot_cu),
        "benchmark_command": shell_join(bench_cmd),
        "benchmark_stdout": str(benchmark_stdout),
        "benchmark_stderr": str(benchmark_stderr),
        "benchmark_json": str(benchmark_json),
        "benchmark_rc": bench_res.returncode,
        "benchmark_result": bench_json,
        "targeted_ncu_rc": None,
        "full_ncu_rc": None,
        "targeted_report": "",
        "full_report": "",
        "targeted_report_exists": False,
        "full_report_exists": False,
        "targeted_import": {},
        "full_import": {},
    }

    correctness = (bench_json or {}).get("correctness") or {}
    correctness_failed = bool(bench_json.get("has_reference")) and correctness.get("passed") is False

    if bench_res.returncode == 0 and not correctness_failed:
        targeted_prefix = iter_dir / "targeted"
        targeted_stdout = iter_dir / "targeted_ncu.stdout.txt"
        targeted_stderr = iter_dir / "targeted_ncu.stderr.txt"
        targeted_cmd = build_targeted_ncu_cmd(args, bench_cmd, targeted_prefix)
        targeted_res = run_command(targeted_cmd, targeted_stdout, targeted_stderr)
        targeted_rep = targeted_prefix.with_suffix(".ncu-rep")
        record["targeted_ncu_command"] = shell_join(targeted_cmd)
        record["targeted_ncu_stdout"] = str(targeted_stdout)
        record["targeted_ncu_stderr"] = str(targeted_stderr)
        record["targeted_ncu_rc"] = targeted_res.returncode
        record["targeted_report"] = str(targeted_rep)
        record["targeted_report_exists"] = valid_report_exists(targeted_rep)
        if record["targeted_report_exists"]:
            record["targeted_import"] = import_ncu_report(
                args,
                targeted_rep,
                iter_dir / "targeted_summary.txt",
                iter_dir / "targeted_details.txt",
            )

        full_prefix = iter_dir / "full"
        full_stdout = iter_dir / "full_ncu.stdout.txt"
        full_stderr = iter_dir / "full_ncu.stderr.txt"
        full_cmd = build_full_ncu_cmd(args, bench_cmd, full_prefix)
        full_res = run_command(full_cmd, full_stdout, full_stderr)
        full_rep = full_prefix.with_suffix(".ncu-rep")
        record["full_ncu_command"] = shell_join(full_cmd)
        record["full_ncu_stdout"] = str(full_stdout)
        record["full_ncu_stderr"] = str(full_stderr)
        record["full_ncu_rc"] = full_res.returncode
        record["full_report"] = str(full_rep)
        record["full_report_exists"] = valid_report_exists(full_rep)
        if record["full_report_exists"]:
            record["full_import"] = import_ncu_report(
                args,
                full_rep,
                iter_dir / "full_summary.txt",
                iter_dir / "full_details.txt",
            )
    else:
        record["targeted_ncu_command"] = ""
        record["full_ncu_command"] = ""

    write_text(iter_dir / "iteration_summary.md", render_iteration_markdown(record))
    proposal_path = iter_dir / "optimization_proposal.md"
    if not proposal_path.exists():
        write_text(
            proposal_path,
            "# Optimization proposal\n\n- Fill in the bottleneck diagnosis from the targeted/full NCU reports.\n- Describe the next kernel change for v{iteration_plus_one}.\n".format(
                iteration_plus_one=iteration + 1
            ),
        )

    iterations = [item for item in manifest.get("iterations", []) if item.get("iteration") != iteration]
    iterations.append(record)
    iterations.sort(key=lambda item: item["iteration"])
    manifest["iterations"] = iterations
    best = choose_best_iteration(iterations)
    manifest["best_iteration"] = None if best is None else best["iteration"]
    manifest["best_kernel_path"] = "" if best is None else best["snapshot_cu"]
    manifest["updated_at"] = now_iso()

    write_json(manifest_path, manifest)
    write_text(summary_path, render_final_summary(manifest))

    if bench_res.returncode != 0:
        return bench_res.returncode
    if record.get("targeted_ncu_rc") not in (None, 0):
        return record["targeted_ncu_rc"]
    if record.get("full_ncu_rc") not in (None, 0):
        return record["full_ncu_rc"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
