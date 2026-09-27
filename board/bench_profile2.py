# bench_profile2.py - where does the ~144us/step gap between "isolated step+check" and the real
# CPU-only loop actually go? And in the FPGA-overlapped loop, is any phase's mean dragged up by a
# few slow outliers (median << mean, big max), or is it consistently that slow every iteration?
#
# Run on the Ultra96 (PYNQ 2.4, Python 3.6) from a Jupyter Terminal:
#   python3 bench_profile2.py tv_cfl_wide150.bit 150
#
# Needs the CURRENT bench_e2e.cpp (with bench_iter_profiled_cpu and the 12-wide bench_iter_profiled)
# and one bitstream pair in the same folder.
#
# Not tested on hardware yet.
import sys, os, ctypes, subprocess
import numpy as np
from pynq import Overlay, Xlnk, Clocks, Bitstream

BIT_NAME = "tv_cfl_wide150.bit"
PL_MHZ = 150.0
if len(sys.argv) > 2 and sys.argv[1].endswith(".bit"):
    BIT_NAME, PL_MHZ = sys.argv[1], float(sys.argv[2])
BASE = 0xA0000000

print(subprocess.check_output(
    ["g++", "-O3", "-std=c++14", "-fopenmp", "-shared", "-fPIC", "bench_e2e.cpp", "-o", "libbench.so"],
    stderr=subprocess.STDOUT, universal_newlines=True) or "libbench.so built")

BIT = os.path.abspath(BIT_NAME)
try:
    ol = Overlay(BIT)
    print("Overlay loaded:", list(ol.ip_dict.keys()))
except Exception as e:
    print("Overlay() failed:", repr(e), "-> plain download")
    Bitstream(BIT).download()
Clocks.fclk0_mhz = PL_MHZ
print("PL clock:", Clocks.fclk0_mhz, "MHz")

lib = ctypes.CDLL(os.path.abspath("libbench.so"))
dp = ctypes.POINTER(ctypes.c_double)
cd, ci, cu = ctypes.c_double, ctypes.c_int, ctypes.c_uint64
lib.fpga_init.argtypes = [cu]; lib.fpga_init.restype = ci
lib.bench_cpu_step.argtypes = [ci, ci, ci, dp, dp, cd, cd]; lib.bench_cpu_step.restype = cd
lib.bench_cpu_check.argtypes = [ci, ci, ci, dp, cd, cd]; lib.bench_cpu_check.restype = cd
lib.bench_iter.argtypes = [ci, ci, ci, ci, cd, cd, dp, cu, dp, cu, dp, cu, ci, dp]; lib.bench_iter.restype = cd
lib.bench_iter_profiled.argtypes = [ci, ci, ci, cd, cd, dp, cu, dp, cu, dp, cu, ci, dp]
lib.bench_iter_profiled.restype = cd
lib.bench_iter_profiled_cpu.argtypes = [ci, ci, ci, cd, cd, dp, dp, dp]
lib.bench_iter_profiled_cpu.restype = cd
print("fpga_init:", lib.fpga_init(BASE), "(0 = registers mapped)")

NMAX = 131072
xlnk = Xlnk()
ub = [xlnk.cma_array(shape=(NMAX + 2,), dtype=np.float64, cacheable=1) for _ in range(2)]
res = xlnk.cma_array(shape=(2,), dtype=np.float64)
P = lambda a: ctypes.cast(a.ctypes.data, dp)

def init_state(N):
    x = np.linspace(0.0, 38.0, N)
    ub[0][:N] = np.where(x < 15.01, -0.015 * x * (x - 15.0), 0.0)
    ub[1][:N] = 0.0

def row(label, med, mean, mx):
    flag = "  <- outliers?" if mean > 1.5 * med and mx > 3 * med else ""
    print("  %-8s median %8.2f  mean %8.2f  max %9.2f%s" % (label, med, mean, mx, flag))

print("\n============= PART 1: CPU-only, isolated calls vs. the real loop =============")
for N, steps in ((10000, 400), (100000, 150)):
    dx = 38.0 / (N - 1)
    dt = 0.5 * dx / 0.85
    print("\n--- N = %d ---" % N)
    for th in (1, 2, 4):
        iso_step = lib.bench_cpu_step(N, th, 30, P(ub[0]), P(ub[1]), dt, dx)
        iso_check = lib.bench_cpu_check(N, th, 30, P(ub[0]), dt, dx)
        init_state(N)
        phase = (ctypes.c_double * 6)()
        real_sum = lib.bench_iter_profiled_cpu(th, N, steps, dt, dx, P(ub[0]), P(ub[1]), phase)
        init_state(N)
        ms = lib.bench_iter(0, th, N, steps, dt, dx, P(ub[0]), 0, P(ub[1]), 0, None, 0, 0, (ctypes.c_double * 6)())
        print(" threads=%d  isolated step=%.1fus check=%.1fus sum=%.1fus | real-loop sum=%.1fus | bench_iter=%.1fus (whole-iter ms*1000)"
              % (th, iso_step, iso_check, iso_step + iso_check, real_sum, ms * 1000.0))
        row("step", phase[0], phase[1], phase[2])
        row("check", phase[3], phase[4], phase[5])

print("\n============= PART 2: FPGA-overlapped phase breakdown, median/mean/max =============")
for N, steps in ((10000, 400), (100000, 150)):
    dx = 38.0 / (N - 1)
    dt = 0.5 * dx / 0.85
    print("\n--- N = %d ---" % N)
    for th in (1, 2, 4):
        init_state(N)
        phase = (ctypes.c_double * 12)()
        total_us = lib.bench_iter_profiled(th, N, steps, dt, dx, P(ub[0]), ub[0].physical_address,
                                           P(ub[1]), ub[1].physical_address, P(res),
                                           res.physical_address, 1, phase)
        init_state(N)
        out = (ctypes.c_double * 6)()
        ms = lib.bench_iter(2, th, N, steps, dt, dx, P(ub[0]), ub[0].physical_address, P(ub[1]),
                            ub[1].physical_address, P(res), res.physical_address, 1, out)
        print(" threads=%d  sum of medians=%.1fus | bench_iter=%.1fus  (gap=%.1fus)"
              % (th, total_us, ms * 1000.0, ms * 1000.0 - total_us))
        row("step", phase[0], phase[1], phase[2])
        row("flush", phase[3], phase[4], phase[5])
        row("launch", phase[6], phase[7], phase[8])
        row("wait", phase[9], phase[10], phase[11])
