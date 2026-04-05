# Benchmark Plot Guide

Generated graphics:
- `*_runtime.png`: total runtime comparison for Serial, best OpenMP, and SIMD.
- `*_speedup.png`: percent speedup versus Serial for best OpenMP and SIMD.
- `*_omp_scaling.png`: OpenMP speedup for each tested thread count.
- `*_runtime_breakdown.png`: step/CFL/TV runtime share at the largest mesh size.
- `*_openmp_verification_ms.png`: CFL and TV time in milliseconds across OpenMP thread counts.
- `*_openmp_verification_pct.png`: CFL and TV share of runtime across OpenMP thread counts.
- `best_openmp_summary.csv`: best OpenMP thread count per mesh size.

Flux/scheme pairs found in this CSV:
- `burgers` / `lax_friedrichs`
- `burgers` / `leapfrog`