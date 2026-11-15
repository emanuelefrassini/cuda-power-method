#!/usr/bin/env python3
"""Check known eigenspectrum, convergence and CUDA/CPU agreement."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def execute(backend: str, n: int, iters: int, threads: int = 256) -> dict[str, str]:
    exe = ROOT / "bin" / f"power_{backend}"
    proc = subprocess.run(
        [str(exe), "--n", str(n), "--iterations", str(iters), "--threads", str(threads)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise AssertionError(f"{exe.name} exited {proc.returncode}: {proc.stderr}")
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, proc.stdout
    return dict(item.split("=", 1) for item in lines[0].split()[1:])


def check(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)
    print(f"PASS {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "cuda"], required=True)
    backend = parser.parse_args().backend
    for n in (1, 7, 31, 32, 33, 127, 257):
        for threads in ((128, 256) if backend == "cuda" else (256,)):
            result = execute(backend, n, 48, threads)
            check(abs(float(result["eigenvalue"]) - 3.0) < 5e-4,
                  f"{backend} n={n} threads={threads}: dominant eigenvalue matches 3")
            check(float(result["relative_residual"]) < 2e-4,
                  f"{backend} n={n} threads={threads}: relative residual < 2e-4")
    early = execute(backend, 127, 4)
    late = execute(backend, 127, 40)
    check(float(late["relative_residual"]) < float(early["relative_residual"]) / 100,
          f"{backend}: eigenvector residual improves with power iterations")
    if backend == "cuda":
        for n in (33, 127, 257):
            gpu = execute("cuda", n, 48)
            cpu = execute("cpu", n, 48)
            check(abs(float(gpu["eigenvalue"]) - float(cpu["eigenvalue"])) < 5e-4,
                  f"n={n}: CUDA and CPU eigenvalues agree")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"FAIL: {err}", file=sys.stderr)
        sys.exit(1)
