# Handoff: TV/CFL HLS kernel for the Ultra96

Status as of 2026-09-18. Branch: `match-python-fluxes`. **Nothing is committed.**

## Goal

Run the total-variation (TV) and CFL checks for the 1D conservation-law solvers on
the FPGA fabric of an **Ultra96** (Zynq UltraScale+ ZU3EG, part
`xczu3eg-sbva484-1-e`). The CPU (ARM) advances the solution; the FPGA checks it.
A run should stop when **either** check is violated:

- CFL number `max|f'(u)|*dt/dx > 1`
- TV increased versus the previous step (needs a tolerance, see "Open decisions")

Idea for later: the checks only monitor stability (dt is fixed in the host code),
so the CPU can run a few steps ahead of the FPGA and discard steps after a
violation. This needs a ring of 2-3 snapshot buffers.

## What the kernel does

`tv_cfl_kernel` ([hls/tv_cfl_kernel.cpp](hls/tv_cfl_kernel.cpp)) makes one streaming
pass over `u[0..N-1]` and returns `results[0] = TV`, `results[1] = CFL`.
The array is periodic. Flux is selected at run time by `flux_id` (0-6: Burgers, LWR,
flood wave, cubic, Buckley-Leverett, log, trilinear).

- 16 partial accumulators for TV (sums) and 16 for CFL (maxes), dealt round-robin
  (`(i-1) % 16`). This breaks the loop-carried floating-point dependency so the
  main loop reaches **II=1**. Buckets must be >= FP add latency.
- Interfaces: `m_axi` for `u` (bundle `gmem0`) and `results` (bundle `gmem1`);
  `s_axilite` (bundle `control`) for `N`, `dt`, `dx`, `flux_id` and start/done.
  No explicit `offset=` is set on the `m_axi` ports; the generated register map
  (`hls_out/hls/syn/verilog/tv_cfl_kernel_control_s_axi.v` header) still contains
  address registers for `u` (0x10/0x14) and `results` (0x1c/0x20).
- Precision is set by `real_t` in [hls/tv_cfl_kernel.hpp](hls/tv_cfl_kernel.hpp).

## The target is an Ultra96, not an Alveo card

The pre-existing files were written for an Alveo U200 over PCIe. None of these apply
to the Ultra96 and should be treated as not usable as-is:

- [vitis/Makefile](vitis/Makefile) (U200 platform, XRT, Linux commands)
- [vitis/connectivity.cfg](vitis/connectivity.cfg) (`DDR[0]`/`DDR[1]` banks)
- [host/host_tv_cfl.cpp](host/host_tv_cfl.cpp) (PCIe buffer copies via XRT)

The Ultra96 has an ARM CPU and FPGA fabric sharing one DDR. There is no PCIe copy.
The kernel reads the buffer straight from shared DDR.

## Chosen flow: "Route A"

Vitis HLS -> export IP -> Vivado block design with the Zynq PS -> bitstream ->
load and drive from Python on the board (PYNQ image). Route B (v++ with a custom
embedded platform + cross-compiled XRT host) was rejected as too much setup for a
first pass.

## What has been done

1. **Changed files** (uncommitted; see `git status`):
   - `hls/tv_cfl_kernel.hpp`: added the `real_t` typedef (default **double**;
     `-DTV_CFL_USE_FLOAT` selects float).
   - `hls/tv_cfl_kernel.cpp`: uses `real_t`. (An attempted `offset=slave` addition on the
     `m_axi` ports never applied and is not in the file.)
   - `hls/tb_tv_cfl.cpp`: uses `real_t`; tolerance is 1e-10 for double, 1e-4 for float.
   - `hls/hls_config.cfg`: **new**; Vitis 2025.1 config for the Ultra96 part, 10 ns clock,
     IP-catalog packaging.
   - `.gitignore`: added `hls_out/`.
2. **C-sim** passes 9/9 in both double and float against the CPU reference.
3. **Synthesis on the Ultra96 part** succeeded (double build, current state):

   | Metric | Value |
   |---|---|
   | Main loop II | 1 (target 1) |
   | Pipeline depth | 63 cycles |
   | Clock | 7.61 ns est. + 2.70 ns uncertainty (target 10 ns) |
   | DSP | 103 / 360 (28%) |
   | LUT | 17,254 / 70,560 (24%) |
   | FF | 17,579 / 141,120 (12%) |
   | BRAM | 4 / 432 |

   Expected run time is about N + 63 cycles per call.
4. A **float** build was attempted but is not valid. See "Known issues".

## How to reproduce

Tooling is Vitis/Vivado **2025.1** at `D:\2025.1`. Zynq UltraScale+ device support was
installed separately (the initial install had only 7-series and Versal). `vitis_hls`
does not exist in 2025.1; use `v++` / `vitis-run`. Run from Git Bash in `hls/`:

```bash
export PATH=/c/msys64/mingw64/bin:/d/2025.1/Vitis/bin:$PATH   # mingw needed for g++

# C-sim with plain g++ (fast)
g++ -O2 -std=c++17 -DHLS_CSIM -I. -o tb tb_tv_cfl.cpp tv_cfl_kernel.cpp && ./tb

# Synthesis for the Ultra96 (~1 min); output in hls_out/
v++ -c --mode hls --config hls_config.cfg --work_dir hls_out

# C-sim through Vitis (attempted once; failed because the part was not installed then,
# not re-run since)
vitis-run --mode hls --csim --config hls_config.cfg --work_dir hls_out

# Package as IP for Vivado (NOT yet run)
vitis-run --mode hls --package --config hls_config.cfg --work_dir hls_out
```

Outputs: `hls/hls_out/hls/syn/report/tv_cfl_kernel_csynth.rpt` (resources, timing,
ports), `..._Pipeline_VITIS_LOOP_87_2_csynth.rpt` (loop II/depth),
`hls/hls_out/hls/syn/verilog/` and `vhdl/` (generated RTL).

On this PC, g++ fails silently unless `C:\msys64\mingw64\bin` is on PATH.

## Known issues

- **`flux_functions.hpp` is still double-only.** An edit converting it to `real_t`
  was applied and later disappeared (the file matched git HEAD again; cause unknown,
  possibly an editor revert). Consequence: `-DTV_CFL_USE_FLOAT` is not a true float
  design, because the flux math is silently promoted to double. An earlier "float"
  synthesis (104 DSPs) was therefore mostly double hardware and its numbers are not
  meaningful for a real float design. The current double build is unaffected.
- **All seven flux branches are synthesized** (divider, square root, several multipliers),
  because `flux_id` is a run-time register. The comment in `flux_functions.hpp`
  claiming the tool specializes the switch is wrong. A single-flux build would
  save area.
- **Timing is marginal:** 7.61 + 2.70 = 10.31 ns vs. the 10 ns target. Vivado
  implementation may miss 100 MHz; fallback is a 12 ns clock.
- `hls/dfx_runtime.txt` is a stray file written by the tool; safe to delete.
- Pre-existing and unrelated: `vitis/tb_tv_cfl.exe` and `errors.txt` are tracked in git.

## Open decisions

- **Float vs. double for the checks.** Current choice: double, to match the CPU
  solvers, avoid conversions, and allow a tight tolerance. It fits easily.
  Float is viable for CFL; for the "TV increased" test it needs a tolerance
  (e.g. flag only if `TV_new > TV_old*(1+1e-5)`), since TVD schemes change TV very
  little per step and float rounding noise can be comparable.
- Whether the violation logic (CFL > 1 or TV increased) lives in the kernel (a flag
  output) or on the CPU. Not implemented yet.
- Which board image is on the Ultra96 (stock Avnet vs. PYNQ) and V1 vs. V2. Not yet
  answered.

## Suggested next steps

1. Package the IP (`--package`), then open Vivado and build a block design:
   Zynq UltraScale+ PS + this IP + AXI interconnect (kernel master -> a PS HP port,
   control slave <- PS GP port). Generate a bitstream. Expect the 10 ns clock to need
   checking after implementation.
2. Decide the violation logic and add it (kernel flag, with the TV tolerance).
3. Optionally try a fixed-flux Burgers build to measure the area saving, and a
   wider memory port (e.g. 4 elements per cycle, roughly 4x faster loop).
4. When the board is available: load the bitstream from PYNQ, set the buffer
   address and parameters through the AXI-Lite registers, start the kernel, read
   the result. Handle ARM cache coherence for the shared buffer.
5. Later: ring of snapshot buffers so the CPU can run ahead of the check.

## Update: violation flag added (2026-09-19)

- `tv_cfl_kernel` now returns an `int` status and takes two new inputs, `tv_prev` and
  `tol`. Status bit0 = CFL > 1, bit1 = TV > `tv_prev*(1+tol)` (negative `tv_prev`
  disables the TV test). The check runs once at the end of each pass, after the
  16-bucket reduction; the main loop, buckets and pragmas are unchanged.
- `m_axi` `depth` for `u` reduced 1048576 -> 131072 (simulation-only setting; the
  testbench pads its buffer to match). Max N usable in co-sim is now 131072.
- C-sim: 13/13 pass. Synthesis (Ultra96 part): loop II=1, depth 63, 7.61 ns (+2.70),
  103 DSP (28%), 18,055 LUT (25%), 17,952 FF (12%).
- **Register offsets changed.** New map (from `tv_cfl_kernel_control_s_axi.v`):
  return/status 0x10, u 0x18/0x1c, results 0x24/0x28, N 0x30, dt 0x38/0x3c,
  dx 0x44/0x48, flux_id 0x50, tv_prev 0x58/0x5c, tol 0x64/0x68.
  The earlier map in this file/conversation is obsolete.
- The C/RTL co-simulation PASS reported earlier was for the kernel WITHOUT the flag and
  is stale; co-sim has not been re-run on the current source.
- `host/host_tv_cfl.cpp` still calls the old signature (it is Alveo/XRT code and is not
  built here).
- Still not implemented: the CPU-side loop that reads the status each step and stops the
  run, and the ring of snapshot buffers that lets the CPU run ahead.
