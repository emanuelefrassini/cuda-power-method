#!/usr/bin/env python3
"""Plot CPU and GPU timings from actual benchmark summary records."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("results/summary.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = p.parse_args()
    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu = {int(row["n"]): row for row in rows if row["backend"] == "cpu"}
    gpu = {int(row["n"]): row for row in rows if row["backend"] == "cuda"}
    if not cpu:
        raise SystemExit("CSV missing CPU rows")
    sizes = sorted(cpu)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(sizes, [float(cpu[n]["median_compute_ms"]) for n in sizes], "o-", label="CPU compute")
    if gpu:
        overlap = sorted(cpu.keys() & gpu.keys())
        ax.plot(overlap, [float(gpu[n]["median_compute_ms"]) for n in overlap], "s-", label="GPU kernel time")
        ax.plot(overlap, [float(gpu[n]["median_total_ms"]) for n in overlap], "^-", label="GPU end-to-end")
    ax.set(xlabel="Dense matrix dimension N", ylabel="Median time (ms)", title="Power iteration: measured execution time")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "timings.png", dpi=170)
    plt.close(fig)
    if gpu:
        overlap = sorted(cpu.keys() & gpu.keys())
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(overlap, [float(cpu[n]["median_compute_ms"]) / float(gpu[n]["median_compute_ms"])
                          for n in overlap], "o-", label="CPU compute / GPU kernels")
        ax.plot(overlap, [float(cpu[n]["median_compute_ms"]) / float(gpu[n]["median_total_ms"])
                          for n in overlap], "s-", label="CPU compute / GPU end-to-end")
        ax.axhline(1.0, linewidth=1, linestyle="--")
        ax.set(xlabel="Dense matrix dimension N", ylabel="Speedup ratio", title="GPU speedup: end-to-end versus kernel-only")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_dir / "speedup.png", dpi=170)
        plt.close(fig)
    print("Saved", args.output_dir / "timings.png")
    if gpu:
        print("Saved", args.output_dir / "speedup.png")


if __name__ == "__main__":
    main()
