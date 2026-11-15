#!/usr/bin/env python3
"""Generate a reproducible eigenpair residual plot using the CPU executable."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    opts = p.parse_args()
    ks = [1, 2, 4, 6, 8, 12, 16, 24, 32, 40, 48]
    errors = []
    for k in ks:
        run = subprocess.run([str(ROOT / "bin/power_cpu"), "--n", str(opts.n), "--iterations", str(k)],
                             check=True, capture_output=True, text=True)
        fields = dict(word.split("=", 1) for word in run.stdout.strip().split()[1:])
        errors.append(float(fields["relative_residual"]))
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(ks, errors, "o-")
    ax.set(xlabel="Power iterations", ylabel="Relative eigenpair residual", title=f"Power-method convergence (CPU, N={opts.n})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = opts.output_dir / "cpu_convergence.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("Saved", path)

if __name__ == "__main__":
    main()
