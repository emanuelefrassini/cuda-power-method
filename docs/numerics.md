# Numerical design and verification

## Power iteration

The power method approximates the dominant eigenvector of a matrix \(A\) using
\(v_{k+1} = Av_k / \|Av_k\|_2\). Given the final unit vector \(v\), the
Rayleigh quotient is \(\lambda = (v^TAv)/(v^Tv)\). The diagnostic
\(\|Av-\lambda v\|_2 / \|Av\|_2\) measures the eigenpair residual.

A small change in the eigenvalue estimate alone is not a reliable indication
of eigenvector convergence. The implementation uses a fixed number of steps
and reports both the eigenvalue error and the residual. The final diagnostic
matvec is part of the measured compute time for both backends.

## Deterministic test matrix

To make correctness checkable without an external eigensolver, the matrix is
\(A = 2I + \frac{1}{N}\mathbf{1}\mathbf{1}^T\). It is dense, symmetric,
and positive definite. Its eigenvalues are exactly 3 (once) and 2
(with multiplicity \(N-1\)). The initial vector has a nonzero component
in the dominant eigenspace and is not an eigenvector.

The GPU and CPU versions both explicitly perform dense \(O(N^2)\)
matrix-vector products rather than using the matrix's special rank-one
structure. This deliberately measures a generic dense power-method pathway,
although a specialized rank-one computation could be much faster on *either*
backend. This matrix is a synthetic verification workload, not a measured
scientific dataset.

## CUDA implementation

- Matrix stored in row-major single-precision floats.
- One warp processes each matrix row using coalesced column reads and
  warp-shuffle reduction; arbitrary N is supported.
- Norm squared is computed using shared-memory block reductions followed
  by `atomicAdd` across blocks.
- Normalization applies the inverse square root without copying the scalar
  norm to the CPU each iteration.
- The eigenvalue and residual are evaluated on the host after the final
  vector and matrix product are copied back. This diagnostic occurs after
  compute timing and is the same calculation used for the CPU results.

## Timing methodology

`compute_ms` measures the iterative algorithm plus final matvec:
- CPU: wall-clock steady-clock time around the numerical loop.
- CUDA: CUDA-event time around kernel execution and scalar reset operations.

`total_ms` for CUDA is host wall time from GPU allocations through input
transfers, kernel work, synchronization and output transfers. It excludes
common host-side matrix and vector creation but includes device setup costs.
The CPU `total_ms` is its compute time (it has no device transfers).
These are **not strictly equivalent** accounting boundaries, so timing
comparisons should say exactly which measurement is used. Do not interpret
short GPU kernel times as end-to-end application speedup.

Repetitions are independent program processes. Warm-up runs are discarded.
Reported timings are medians; CPU frequency scaling, GPU warm-up, memory
capacity, launch overhead and hardware vary considerably.

## Limitations

- Designed for dense small/medium matrices; GPU memory usage scales as
  \(4N^2\) bytes for the matrix (plus vectors and runtime overhead).
- No dynamic tolerance-based early stopping; both backends execute exactly
  the configured iteration count.
- Power iteration is intended for a distinct dominant eigenvalue; a tiny
  spectral gap converges slowly and a badly chosen initial vector can fail.
- Floating-point reduction order and `atomicAdd` are nondeterministic,
  so expect small CPU/GPU differences. Accuracy is validated with a
  residual and an analytically known eigenvalue.
- No GPU performance or compatibility is claimed until tested on actual CUDA
  hardware using `make test-gpu` and benchmark scripts.
