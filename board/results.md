# Benchmark results: TV + CFL check on the Ultra96

## TL;DR

- **Burgers** (cheapest flux): after fixing an inefficiency in the CPU-side benchmark code,
  **plain CPU-only beats the FPGA at 4 threads** (by 9-44%, depending on N). The FPGA only wins
  at 1-2 threads.
- **Buckley-Leverett** (heaviest flux, division in the CFL formula): **the FPGA wins at every
  thread count**, by 1.41-1.62x, including at 4 threads.
- **Mechanism, confirmed with real numbers on both fluxes:** the FPGA kernel's check time is
  essentially **flux-independent** (~337 us at N=100k/150MHz for both Burgers and
  Buckley-Leverett, within 0.1%), because it is a fully pipelined design - one element per clock
  regardless of whether that clock's arithmetic is a subtract or a division. The CPU-side check
  is **flux-dependent**: it scaled from 297 us (Burgers, 4T) to 1048 us (Buckley-Leverett, 4T),
  a 3.5x increase, because it actually has to execute the heavier formula N times on real cycles.
  This is why a cheap flux lets a fast CPU win, and a heavy flux hands the win to the FPGA.

Hardware: Ultra96 (ZU3EG). CPU: 4x Cortex-A53 at 1.2 GHz. FPGA clock (PL) as listed per section.
Lax-Friedrichs, periodic, `double`, N = 10,000 and 100,000. Timestep chosen so CFL is about 0.5
(no violations) unless noted. "1T/2T/4T" = threads used for the solver step (and for the CPU-side
check in CPU-only mode). Every run's sanity check passed: same final state in every mode, FPGA TV
and CFL equal to the CPU's, no violations flagged. Run-to-run noise is about 6-9%.

---

# Part 1: Burgers flux (`f(u) = 0.5u^2`, `|f'(u)| = |u|`)

## 1.1 Initial measurements, before a CPU-side fix (2026-09-25)

The very first measurements used a CPU benchmark (`step_lf`, `check_fused`) that called a
`pidx()` index-wrapping function with two `if` branches on every array access, to handle the
periodic boundary. Four FPGA configurations were compared:

| Label | Bitstream | PL clock | Reads per clock |
|---|---|---|---|
| old@100 | `tv_cfl.bit` | 100 MHz | 1 double |
| old@150 | `tv_cfl.bit` | 150 MHz | 1 double |
| wide@100 | `tv_cfl_wide150.bit` | 100 MHz | 2 doubles |
| wide@150 | `tv_cfl_wide150.bit` | 150 MHz | 2 doubles |

**The check alone (us per call):**

| N | old@100 | old@150 | wide@100 | wide@150 | CPU 1T | CPU 2T | CPU 4T |
|---|---|---|---|---|---|---|---|
| 10,000 | 104.5 | 69.9 | 55.0 | 37.0 | 116.4-116.7 | 61.3-61.4 | 31.0-31.4 |
| 100,000 | 1004.5 | 669.9 | 505.0 | 337.0 | 1156-1162 | 588.6-588.8 | 296.1-297.0 |

FPGA speed-up at N=100k relative to old@100: clock alone 1.50x, width alone 1.99x, both 2.98x
(matches the ideal cycle counts almost exactly: 1000, 667, 500, 333 us plus ~5 us launch overhead).

**Whole iteration, N = 100,000 (ms/step), overlapped mode:**

| Config | 1T | 2T | 4T |
|---|---|---|---|
| CPU only (branchy) | 4.37 | 1.97 | 1.02-1.09 |
| old@100 | 3.31 | 1.47 | 1.02 |
| old@150 | 3.31 | 1.48 | 0.88 |
| wide@100 | 3.31 | 1.47 | 0.90 |
| wide@150 | 3.31 | 1.52 | 0.94 |

At this stage the FPGA (overlapped) looked like a modest win over CPU-only at every thread count:
about 1.2-1.35x at 1-2T, 1.1-1.2x at 4T. **This conclusion did not survive fixing the CPU code
(1.2, below).**

## 1.2 The CPU fix: removing the periodic-index branches

The only two points in the whole array that ever need index wraparound are `i=0` and `i=N-1` -
every other point's neighbours are already in bounds. Handling those two points once, outside
the loop, and running the rest of the loop with plain array indexing (no `pidx()` branch at all)
was verified bit-identical to the original for N = 2 through 20,001 at 1-4 threads, for both the
step and the check.

**Effect on the isolated CPU numbers, N = 100,000, 4T:**

| | Step | Check |
|---|---|---|
| Before (branchy) | 661.3 us | 296.1-297.0 us |
| After (branchless) | 334.7 us | 161.2 us |

Roughly **halved**, at every thread count, for both the step and the check.

**Effect on the whole-iteration comparison, wide@150, N = 100,000 (ms/step):**

| Mode | 1T | 2T | 4T |
|---|---|---|---|
| CPU only | 1.992 | 1.075 | **0.640** |
| FPGA, waiting | 1.727 | 1.129 | 0.847 |
| FPGA, overlapped | 1.478 | 0.937 | **0.724** |

**N = 10,000 (ms/step):**

| Mode | 1T | 2T | 4T |
|---|---|---|---|
| CPU only | 0.189 | 0.101 | **0.052** |
| FPGA, waiting | 0.210 | 0.139 | 0.098 |
| FPGA, overlapped | 0.178 | 0.105 | **0.075** |

**At 4 threads, CPU-only now wins in both cases** (0.640 vs 0.724 ms at N=100k: CPU 1.13x faster;
0.052 vs 0.075 ms at N=10k: CPU 1.44x faster). The FPGA still wins at 1-2 threads (1.2-1.35x).

Why: the fix helped CPU-only in *two* places (its own step and its own check), but only helped
the FPGA-overlapped step (the FPGA's own check time is fixed, ~337 us regardless of how fast the
CPU got). Once the CPU check got cheap enough, the FPGA's fixed per-check cost (mainly the cache
flush + launch, not the kernel's own arithmetic) stopped being worth paying at 4 threads.

## 1.3 Clock experiment: does raising the FPGA clock help? (150 -> 166.665 MHz)

Raised the clock to test whether the FPGA's own port was self-limited (a memory-bandwidth
question), using the same `wide` (2 doubles/clock) kernel, real routed timing verified safe
(+0.866 ns slack at 164.582 MHz target; PYNQ's PLL divider actually delivered 166.665 MHz, still
comfortably inside the verified margin).

| | Check only (FPGA), N=100k | Predicted from clock ratio |
|---|---|---|
| 150 MHz (149.9985 actual) | 337.0 us | - |
| 166.665 MHz | **303.4 us** | 303.3 us |

**The check scaled almost exactly with clock** (303.4 measured vs 303.3 predicted) - confirming
the FPGA's own port is *not* self-limited by the board's shared LPDDR4 channel (Ultra96: single
32-bit channel at 1066 MT/s, ~4.26 GB/s total) when running in isolation.

**But the whole-iteration number did not move at all:**

| N=100,000, 4T | 150 MHz | 166.665 MHz |
|---|---|---|
| FPGA overlapped | 0.724 ms | 0.728 ms (flat, within noise) |

Because the check (337 us) was *already* shorter than the (contention-inflated) step
(~557-570 us) even at 150 MHz - it was already fully hidden. Shrinking an already-hidden check
further does nothing for the overlapped time, which is set by `max(step, check)`. This is a clean
confirmation that **for Burgers at 4 threads, the CPU step - not the check - is the bottleneck**,
and no amount of further FPGA clock/width tuning would change that outcome.

## 1.4 Other things tried for Burgers

- **A cache-coherent FPGA port** (routing the kernel through the PS's `HPC0` port instead of the
  plain `HP0`, with `AxCACHE`/`AxUSER` set for shareable transactions): built and tested with the
  flush disabled. Result: **did not achieve real coherency** - the FPGA still read stale data
  without a flush, identically to the non-coherent build. Root cause (from AMD's docs, not fully
  verified against this specific board/image): real HPC coherency needs the CCI-400 interconnect's
  snoop control explicitly enabled at the system level (a register write the FSBL/boot firmware
  would need to do, or a runtime privileged register poke we did not attempt) plus the CPU-side
  buffer being mapped as "outer shareable" in the page tables - neither of which Vivado/HLS
  settings alone provide. Not pursued further given the flush turned out to be cheap anyway (next
  point).
- **Software flush variants** (clean-only vs. clean+invalidate, one thread vs. each thread
  cleaning its own chunk): **no measurable difference** between any of the four variants tested.
- **Direct flush cost measurement:** ~11 us (N=10k) to ~58 us (N=100k) - much cheaper than an
  earlier *inferred* estimate of 130-150 us (which was wrong: it was computed by subtracting
  isolated benchmarks from the whole-loop time, and most of that gap turned out to be something
  else - see 1.5).
- **More outstanding checks (a deeper buffer ring):** considered but not built. Reasoning: the
  FPGA processes checks strictly one at a time (no parallelism gain from more buffers), and the
  measured "wait" time (the CPU blocking on a not-yet-finished check) was already ~0.4 us in
  every configuration tested - there was nothing to relieve. It would also directly worsen
  detection latency (how many steps a real instability could go unnoticed), for no measured
  throughput benefit.

## 1.5 What actually inflates the step in overlapped mode (phase profiling)

Breaking the overlapped loop into step/flush/launch/wait phases (median of many iterations, to
filter OS-noise outliers) found:

| N=100,000, 4T | Step, isolated | Step, inside CPU-only loop | **Step, inside FPGA-overlapped loop** |
|---|---|---|---|
| | 338.8 us | 358.9 us | **~557-561 us** |

The same step function, same N, same threads - but **~65% slower specifically when running
concurrently with the FPGA's own memory read**, and this held at both 150 MHz and 166.665 MHz
(557 vs 561 us - unaffected by the FPGA getting faster). Best explanation: DRAM bandwidth
contention between the CPU's own streaming step and the FPGA's concurrent DMA read, competing for
the same single, fairly narrow LPDDR4 channel. Not fully isolated/proven (no direct bandwidth-
utilization measurement was taken), but the pattern (present only during concurrent access, absent
in isolation or CPU-only mode) fits.

## 1.6 Burgers final numbers (wide, 150 MHz, post-fix) - the ones that matter

| N | Mode | 1T | 2T | 4T |
|---|---|---|---|---|
| 10,000 | CPU only | 0.189 | 0.101 | **0.052** |
| | FPGA overlapped | 0.178 | 0.105 | 0.075 |
| 100,000 | CPU only | 1.992 | 1.075 | **0.640** |
| | FPGA overlapped | 1.478 | 0.937 | 0.724 |

**Conclusion for Burgers: use plain CPU-only at 4 threads.** The FPGA only helps if you are
limited to 1-2 threads.

---

# Part 2: Buckley-Leverett flux (`f(u) = u^2/(u^2+0.25(1-u)^2)`)

## 2.1 Why this flux

What determines whether the FPGA wins is `|f'(u)|` specifically - the formula used *only* by the
CPU-side check (`check_fused`), since the step's `f(u)` cost is shared by both modes and cancels
out of the comparison. Buckley-Leverett's derivative,
`|f'(u)| = |0.5 u (1-u) / (u^2 + 0.25(1-u)^2)^2|`, needs one division plus ~6-7 surrounding
multiplies/adds to compute the denominator - the heaviest of the project's 7 fluxes, versus a
trivial `|u|` for Burgers. It is also one of the project's own two-phase-flow flux functions, not
an arbitrary pick.

## 2.2 Kernel build

Single-flux kernel (`TV_CFL_FIXED_FLUX=4`, same 2-doubles/clock, double-precision design as the
Burgers `wide` kernel), built and verified at 150 MHz (149.9985 MHz actual):

| | Result |
|---|---|
| Pipelining | Target II=1, achieved II=1 (same throughput guarantee as Burgers) |
| Pipeline depth | 97 cycles (vs. Burgers' 29) - one-time startup latency only, negligible |
| Routed timing | +1.220 ns slack at 148.148 MHz target - "All user specified timing constraints are met" |
| DSPs | 167 / 360 (46%) - two 64-bit dividers plus 12 multipliers, vs. Burgers' 23 DSPs (6%) |
| LUTs | 25,713 / 70,560 (36.4%) |
| Flip-flops | 30,710 / 141,120 (21.8%) |

Correctness verified in C-sim (kernel vs. reference) and on real hardware (below).

The CPU-side benchmark (`bench_e2e.cpp`) was extended with a `CPU_FLUX_BUCKLEY` build option
implementing the same `f(u)` and `|f'(u)|` formulas, verified bit-identical to an independent
reference across N = 2-20,001 and 1-4 threads before running on the board.

## 2.3 Results (150 MHz, wide kernel, N=10,000 and 100,000)

**The check alone (us per call):**

| N | FPGA | CPU 1T | CPU 2T | CPU 4T |
|---|---|---|---|---|
| 10,000 | 37.4 | 447.2 | 211.5 | 106.3 |
| 100,000 | 337.4 | 4468.8 | 2094.9 | **1047.7** |

**Whole iteration (ms/step):**

| N | Mode | 1T | 2T | 4T |
|---|---|---|---|---|
| 10,000 | CPU only | 1.072 | 0.528 | 0.266 |
| | FPGA overlapped | 0.675 | 0.353 | **0.189** |
| 100,000 | CPU only | 10.919 | 5.377 | 2.791 |
| | FPGA overlapped | 6.721 | 3.457 | **1.836** |

**FPGA-overlap speed-up vs. CPU-only, every thread count:**

| N | 1T | 2T | 4T |
|---|---|---|---|
| 10,000 | 1.59x | 1.50x | **1.41x** |
| 100,000 | 1.62x | 1.56x | **1.52x** |

Sanity checks passed on real hardware: same final state in every mode, FPGA TV and CFL matched
the CPU's, no violations flagged.

## 2.4 The mechanism, confirmed directly

| N=100,000, 4T | Burgers | Buckley-Leverett | Change |
|---|---|---|---|
| CPU check | 297 us | 1047.7 us | **3.5x more expensive** |
| FPGA check | 337.0 us | 337.4 us | **unchanged (0.1%)** |

The FPGA's check time is flux-independent because it is a fully pipelined design - one element
per clock regardless of whether that clock's arithmetic is a subtraction or a division with
several multiplies. The CPU's check time is flux-dependent because it actually executes the
formula, on real cycles, once per element. This is the entire reason Burgers (cheap `|f'(u)|`)
let a fast 4-thread CPU win, while Buckley-Leverett (expensive `|f'(u)|`) hands a clear,
consistent win to the FPGA at every thread count tested.

The step also got proportionally more expensive on the CPU (Buckley-Leverett's `f(u)` also has a
division, called twice per point by the step) - 6453 us vs. Burgers' ~1235 us at 1 thread. This
doesn't change who wins the FPGA-vs-CPU comparison (it's common to both modes), but it means the
whole simulation is much slower in absolute terms with this flux - a separate, real cost of
choosing it for production use, independent of the FPGA question.

---

# Overall conclusion

The FPGA-assisted check is not universally faster or slower than the CPU - **it depends entirely
on how expensive the flux's derivative is to evaluate.** For the cheapest flux in this project
(Burgers), a modestly-optimized 4-thread CPU wins outright. For the most expensive one tested
(Buckley-Leverett, with a division in its CFL formula), the FPGA wins clearly and consistently,
because its pipelined arithmetic is insensitive to formula complexity in a way scalar CPU code
never is. Log and flood-wave (also carrying a division or square root) would be expected to show
a similar FPGA advantage, though this has not been measured.

## Caveats

- Only two fluxes, one data profile, N = 10k and 100k tested. Energy and CPU load were not measured.
- Run-to-run noise is about 6-9%; treat differences smaller than that with caution.
- The DRAM-contention explanation for the Burgers step's overlapped-mode inflation (1.5) is a
  plausible mechanism supported by the pattern observed, not a directly measured/isolated cause.
- The coherent-port attempt (1.4) is a genuine capability of this chip (documented in AMD's
  UG1085/DS925), just not one this project got working; it was not exhaustively debugged.
- Stopping a real simulation run on a detected violation is not implemented - the benchmark only
  counts violations, it does not act on them. The one-step detection-lag bound (from the
  two-buffer overlapped design) is structural, not yet exercised end-to-end.
