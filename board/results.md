# Benchmark results: TV + CFL check on the Ultra96

## TL;DR

- **Burgers** (cheapest flux): after fixing an inefficiency in the CPU-side benchmark code,
  **plain CPU-only beats the FPGA at 4 threads** (by 9-44%, depending on N). The FPGA only wins
  at 1-2 threads.
- **Buckley-Leverett** (heaviest flux, division plus ~6-7 ops in the CFL formula): **the FPGA
  wins at every thread count**, by 1.41-1.62x.
- **Flood wave** (a single square root): FPGA wins at every thread count, by **1.32-1.50x** -
  solidly, though a bit less than Buckley-Leverett.
- **Log** (a single division): FPGA wins, but only barely - **1.04-1.06x** - because `ln(u)`
  (used by the *step*, not the check) is so expensive that the check is a tiny slice of the
  total iteration, capping how much hiding it can possibly help. See Part 4.
- **Mechanism, confirmed across all four fluxes:** the FPGA kernel's check time is essentially
  **flux-independent** (~337 us at N=100k/150MHz for all four, within noise), because it is a
  fully pipelined design - one element per clock regardless of whether that clock's arithmetic is
  a subtract, a square root, or a division. The CPU-side check is **flux-dependent**: it scaled
  from 297 us (Burgers, 4T) to 1048 us (Buckley-Leverett, 4T), a 3.5x increase, because it
  actually has to execute the heavier formula N times on real cycles. This is why a cheap flux
  lets a fast CPU win, and a heavy flux hands the win to the FPGA - see Part 5 for the full
  four-flux comparison and the one case (Log) where it's more subtle than that.

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

# Part 3: Flood wave flux (`f(u) = max(u,0)^1.5`, `|f'(u)| = 1.5*sqrt(u)` for `u>0`)

## 3.1 Why this flux

A middle case between Burgers (trivial check) and Buckley-Leverett (division plus ~6-7
surrounding ops): `|f'(u)|` here is **one square root and nothing else**. Chosen to test whether
a single expensive operation, on its own, is enough to flip the result - without Buckley-
Leverett's extra multiply/add cluster around it. No domain restriction is needed: both `f(u)` and
`f'(u)` are well-defined at `u=0` (the usual test profile touches 0 and even dips to about
`-0.0018` at the boundary from floating-point rounding; both sides guard `u>0` identically, so
this is handled consistently, not a bug).

## 3.2 Kernel build

| | Result |
|---|---|
| Pipelining | Target II=1, achieved II=1 |
| Pipeline depth | 71 cycles (two 57-cycle square roots - one per element/clock in the 2-wide kernel) |
| Routed timing | +1.289 ns slack at 148.148 MHz target - "All user specified timing constraints are met" |
| DSPs | 45 / 360 (12%) |
| LUTs | 20,993 / 70,560 (29.75%) |
| Flip-flops | 26,638 / 141,120 (18.88%) |

Verified in C-sim and against an independent reference (step and check, N = 2-20,001, 1-4
threads) before running on the board.

## 3.3 Results (150 MHz, wide kernel)

**The check alone (us per call):**

| N | FPGA | CPU 1T | CPU 2T | CPU 4T |
|---|---|---|---|---|
| 10,000 | 37.2 | 200.2 | 146.3 | 85.3 |
| 100,000 | 337.3 | 1998.0 | 1439.3 | **837.7** |

**Whole iteration (ms/step):**

| N | Mode | 1T | 2T | 4T |
|---|---|---|---|---|
| 10,000 | CPU only | 0.557 | 0.414 | 0.241 |
| | FPGA overlapped | 0.397 | 0.300 | **0.183** |
| 100,000 | CPU only | 5.456 | 4.131 | 2.437 |
| | FPGA overlapped | 3.635 | 2.819 | **1.758** |

**FPGA-overlap speed-up vs. CPU-only, every thread count:**

| N | 1T | 2T | 4T |
|---|---|---|---|
| 10,000 | 1.40x | 1.38x | **1.32x** |
| 100,000 | 1.50x | 1.47x | **1.39x** |

Sanity checks passed on real hardware: same final state in every mode, FPGA TV and CFL matched
the CPU's, no violations flagged.

A solid win at every thread count - a bit smaller than Buckley-Leverett's (1.41-1.62x), consistent
with a single square root being less CPU-expensive than a division plus six extra multiplies/adds.

---

# Part 4: Log flux (`f(u) = ln(u)`, `|f'(u)| = 1/u` for `u>0`)

## 4.1 Why this flux, and a profile change it needed

Chosen to isolate whether Buckley-Leverett's win came from the division itself or from its
surrounding arithmetic: `|f'(u)| = 1/u` is **one division and nothing else**. Unlike the other
three fluxes, `ln(u)` is undefined at `u=0`, and the usual test profile touches exactly 0 - so
this flux's CPU-side benchmark uses the usual profile shifted up by 0.1 (min ~0.098, same shape
otherwise). This only affects the CPU-side driver script; the HLS kernel is unaffected, since it
only ever computes `f'(u) = 1/u` (already guarded for `u<=0`), never `ln(u)` itself.

## 4.2 Kernel build

| | Result |
|---|---|
| Pipelining | Target II=1, achieved II=1 |
| Pipeline depth | 46 cycles (two 40-cycle dividers - one per element/clock) |
| Routed timing | +1.866 ns slack at 148.148 MHz target - the most margin of all four kernels built |
| DSPs | 23 / 360 (6%) |
| LUTs | 20,697 / 70,560 (29.33%) |
| Flip-flops | 22,768 / 141,120 (16.13%) |

Verified in C-sim and against an independent reference (step and check, N = 2-20,001, 1-4
threads, on the shifted profile) before running on the board.

## 4.3 Results (150 MHz, wide kernel)

**The check alone (us per call):**

| N | FPGA | CPU 1T | CPU 2T | CPU 4T |
|---|---|---|---|---|
| 10,000 | 37.0 | 272.2 | 120.8 | 61.3 |
| 100,000 | 337.1 | 2716.8 | 1190.4 | **596.3** |

The check alone is a clear FPGA win here - **1.77x faster than the 4-thread CPU at N=100k**
(596.3 vs. 337.1 us), comparable in spirit to Flood wave's check-alone margin.

**But the step is enormous** (`ln(u)` called twice per point by the solver step):

| N=100,000 | CPU step, 4T | CPU check, 4T |
|---|---|---|
| | **8701.7 us** | 596.3 us |

**Whole iteration (ms/step):**

| N | Mode | 1T | 2T | 4T |
|---|---|---|---|---|
| 10,000 | CPU only | 3.633 | 1.866 | 0.937 |
| | FPGA overlapped | 3.476 | 1.779 | **0.903** |
| 100,000 | CPU only | 36.557 | 18.552 | 9.347 |
| | FPGA overlapped | 34.379 | 17.520 | **8.905** |

**FPGA-overlap speed-up vs. CPU-only:**

| N | 1T | 2T | 4T |
|---|---|---|---|
| 10,000 | 1.05x | 1.05x | **1.04x** |
| 100,000 | 1.06x | 1.06x | **1.05x** |

Sanity checks passed on real hardware: same final state in every mode, FPGA TV and CFL matched
the CPU's, no violations flagged. FPGA still wins, but only barely - see 4.4 for why.

## 4.4 Why a 1.77x faster check barely moves the whole iteration

The overlapped iteration time is roughly `max(step, check) + overhead`. Once the check is fully
hidden behind the step (true here: 337 us FPGA check vs. 8702 us CPU step), the *most* overlap can
ever save is however large a share of the total the check used to occupy in CPU-only mode:

| Flux | CPU step, 4T | CPU check, 4T | Check's share of the CPU-only iteration |
|---|---|---|---|
| Burgers | 335 us | 297 us | 47% |
| Buckley-Leverett | 1631 us | 1048 us | 39% |
| Flood wave | 1567 us | 838 us | 35% |
| **Log** | **8702 us** | 596 us | **6%** |

For Log, that ceiling is about 6% - almost exactly the 4-6% speed-up actually measured. The check
genuinely got 1.77x faster on the FPGA; it just never mattered much, because `ln(u)` made the step
so dominant that the check was a small slice of the pie to begin with.

**Why is the step/check cost gap so much bigger for Log than the other three?** Subtracting the
Burgers baseline (both `f(u)` and `f'(u)` trivial) isolates each flux's own cost:

| Flux | Step extra (≈ cost of `f(u)`) | Check extra (≈ cost of `f'(u)`) | f(u) is how many x costlier than f'(u) |
|---|---|---|---|
| Buckley-Leverett | 1296 us | 751 us | 1.7x |
| Flood wave | 1232 us | 541 us | 2.3x |
| **Log** | **8367 us** | 299 us | **28x** |

Buckley-Leverett and Flood wave's `f(u)` and `f'(u)` are built from the same family of operations
(multiplies/adds plus at most one division or square root), so differentiating doesn't change
their computational class - the derivative of a power-law/rational function is another function
of similar complexity. `ln(u)` is different in kind: it's a transcendental function, computed via
range reduction plus a polynomial/rational approximation in the math library, far costlier than a
single hardware divide. Its derivative, `1/u`, happens to collapse to one of the *cheapest*
possible operations - a special property of the natural logarithm, not something that generalizes
to power-law fluxes. That mismatch between an expensive `f(u)` and a cheap `f'(u)` is unique to
Log among the four fluxes tested, and it's the entire reason its real, measured check speedup
(1.77x) translates into such a small whole-iteration speedup (1.05x).

---

# Part 5: Cross-flux comparison

**4 threads, N = 100,000, all four fluxes, final numbers:**

| Flux | `\|f'(u)\|` | Check: FPGA vs. CPU-4T | **Whole iteration: FPGA vs. CPU-only** |
|---|---|---|---|
| Burgers | `\|u\|` | FPGA 1.13x **slower** | **CPU wins, 1.13x** |
| Log | `1/u` | FPGA 1.77x faster | FPGA wins, only **1.05x** |
| Flood wave | `1.5*sqrt(u)` | FPGA 2.48x faster | FPGA wins, **1.39x** |
| Buckley-Leverett | division + ~6 ops | FPGA 3.11x faster | FPGA wins, **1.52x** |

The ranking is exactly what check-formula complexity predicts - cheaper check, more likely the CPU
wins; heavier check, more likely the FPGA wins. Log is the outlier that reveals the full picture
isn't *just* about `f'(u)`'s cost: the whole-iteration speed-up is capped by how large a share of
the total iteration the check occupies, which also depends on `f(u)`'s cost (used only by the
step). A flux can have a genuinely fast FPGA check and still show almost no end-to-end benefit, if
its step is disproportionately expensive for unrelated reasons (here, a transcendental function).

---

# Overall conclusion

The FPGA-assisted check is not universally faster or slower than the CPU - **it depends on how
expensive the flux's `f'(u)` (the check) is relative to the CPU's cycle budget, and separately, on
how large a share of the total iteration that check represents**, which depends on `f(u)` (the
step) too. Four fluxes spanning trivial to heavy were measured on real hardware:

- **Burgers** (trivial check): 4-thread CPU wins outright, 1.13-1.44x.
- **Log** (cheap check, but a transcendental step): FPGA wins, but barely, 1.04-1.06x - a fast
  check diluted by an expensive, unrelated step.
- **Flood wave** (one square root): FPGA wins solidly, 1.32-1.50x.
- **Buckley-Leverett** (division plus several ops, the heaviest check): FPGA wins most clearly
  and consistently, 1.41-1.62x.

The FPGA's advantage comes from being a fully pipelined design whose per-element throughput is
insensitive to formula complexity (always ~337 us at N=100k/150MHz, regardless of flux) - a
property scalar CPU code can never have, since it genuinely executes however many cycles the
formula needs, once per element, every time.

## Caveats

- Four fluxes, one data profile each, N = 10k and 100k tested. Energy and CPU load were not measured.
- Run-to-run noise is about 6-9%; treat differences smaller than that with caution.
- The DRAM-contention explanation for the Burgers step's overlapped-mode inflation (1.5) is a
  plausible mechanism supported by the pattern observed, not a directly measured/isolated cause.
  Not re-tested for the other three fluxes.
- The coherent-port attempt (1.4) is a genuine capability of this chip (documented in AMD's
  UG1085/DS925), just not one this project got working; it was not exhaustively debugged.
- Stopping a real simulation run on a detected violation is not implemented - the benchmark only
  counts violations, it does not act on them. The one-step detection-lag bound (from the
  two-buffer overlapped design) is structural, not yet exercised end-to-end.
- LWR, cubic, and trilinear (the three remaining, cheap fluxes) were not built or measured; based
  on the mechanism confirmed here, they would be expected to pattern with Burgers (CPU wins at 4
  threads), not with the three heavier fluxes.
