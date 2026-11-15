#!/usr/bin/env python3
"""Repeat power-method runs and save measured CPU/CUDA timings without claiming speedups in advance."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def one_run(backend: str, n: int, iterations: int, threads: int) -> dict:
    executable = ROOT / "bin" / f"power_{backend}"
    proc = subprocess.run([str(executable), "--n", str(n), "--iterations", str(iterations),
                           "--threads", str(threads)], text=True, capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"{executable.name} failed (exit {proc.returncode}):\n{proc.stderr}")
    result_lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if len(result_lines) != 1:
        raise RuntimeError(f"Expected one RESULT record:\n{proc.stdout}")
    values = dict(field.split("=", 1) for field in result_lines[0].split()[1:])
    for col in ("compute_ms", "total_ms", "eigenvalue", "relative_residual"):
        values[col] = float(values[col])
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--iterations", type=int, default=48)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--threads", type=int, default=256)
    p.add_argument("--mode", choices=["auto", "cpu", "both"], default="auto")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()
    if args.repeats < 1 or any(n < 1 or n > 4096 for n in args.sizes):
        p.error("Repetitions must be positive and sizes must lie in [1,4096]")
    if not (ROOT / "bin/power_cpu").exists():
        p.error("Build CPU executable first: make cpu")
    backends = ["cpu"]
    cuda_binary = ROOT / "bin/power_cuda"
    if args.mode == "both" and not cuda_binary.exists():
        p.error("CUDA executable missing; run `make gpu` first")
    if args.mode != "cpu" and cuda_binary.exists():
        probe = subprocess.run([str(cuda_binary), "--n", "1", "--iterations", "1"],
                               text=True, capture_output=True)
        if probe.returncode == 0:
            backends.append("cuda")
        elif args.mode == "both":
            p.error("CUDA binary cannot run on this device: " + probe.stderr.strip())
        else:
            print("CUDA hardware unavailable; collecting CPU measurements only.")
    elif args.mode == "auto":
        print("CUDA executable not built; collecting CPU measurements only.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.sizes:
        for backend in backends:
            one_run(backend, n, args.iterations, args.threads)  # warm-up, not recorded
            for rep in range(1, args.repeats + 1):
                result = one_run(backend, n, args.iterations, args.threads)
                if abs(result["eigenvalue"] - 3.0) > 5e-4 or result["relative_residual"] > 2e-4:
                    raise RuntimeError(f"Accuracy validation failed: backend={backend}, n={n}: {result}")
                row = {
                    "backend": backend, "n": n, "iterations": args.iterations,
                    "threads": args.threads, "repeat": rep,
                    "compute_ms": result["compute_ms"], "total_ms": result["total_ms"],
                    "eigenvalue": result["eigenvalue"],
                    "relative_residual": result["relative_residual"],
                    "device": result.get("device", "CPU"),
                }
                rows.append(row)
                print(f"{backend.upper()} n={n} rep={rep} compute_ms={row['compute_ms']:.4f} total_ms={row['total_ms']:.4f}")
    path = args.output_dir / "raw_benchmarks.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for n in args.sizes:
        for backend in backends:
            these = [r for r in rows if r["n"] == n and r["backend"] == backend]
            summary.append({
                "n": n,
                "backend": backend,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "median_compute_ms": statistics.median(r["compute_ms"] for r in these),
                "median_total_ms": statistics.median(r["total_ms"] for r in these),
                "device": these[0]["device"],
            })
    path2 = args.output_dir / "summary.csv"
    with path2.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("Saved", path, "and", path2)
    for n in args.sizes:
        cpu = next(r for r in summary if r["n"] == n and r["backend"] == "cpu")
        gpu = next((r for r in summary if r["n"] == n and r["backend"] == "cuda"), None)
        if gpu:
            compute_speed = cpu["median_compute_ms"] / gpu["median_compute_ms"]
            e2e_speed = cpu["median_compute_ms"] / gpu["median_total_ms"]
            print(f"n={n} CPU={cpu['median_compute_ms']:.3f}ms GPU-kernel={gpu['median_compute_ms']:.3f}ms "
                  f"GPU-end-to-end={gpu['median_total_ms']:.3f}ms kernel-only-ratio={compute_speed:.3f}x "
                  f"end-to-end-ratio={e2e_speed:.3f}x")
        else:
            print(f"n={n} CPU={cpu['median_compute_ms']:.3f}ms (GPU not measured)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
