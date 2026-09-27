# bench_flush.py - how much does making the CPU's array visible to the FPGA cost, and can it be cheaper?
# Run on the Ultra96 (PYNQ 2.4, Python 3.6) from a Jupyter Terminal:
#
#   python3 bench_flush.py tv_cfl_wide150.bit 150
#
# Needs bench_e2e.cpp (the updated one) and the bitstream pair in the same folder.
# Prints tables and writes bench_flush_<bitstream>_<MHz>MHz.csv - paste me the printed tables.
#
# Flush modes compared (all on the same FPGA check):
#   0  one thread flushes the whole array after the step, clean+invalidate   (what bench_all.py did)
#   1  one thread flushes the whole array after the step, clean only
#   2  every thread cleans its own chunk right after computing it, clean only
#   3  every thread flushes its own chunk right after computing it, clean+invalidate
#   4  NO flush (diagnostic: the FPGA should read stale data, so its answers should be WRONG)
#
# Not tested on hardware yet.
import sys, os, ctypes, subprocess, csv
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
lib.fpga_init.argtypes = [cu]
lib.fpga_init.restype = ci
lib.set_flush_mode.argtypes = [ci]
lib.set_flush_mode.restype = None
lib.bench_cpu_step.argtypes = [ci, ci, ci, dp, dp, cd, cd]
lib.bench_cpu_step.restype = cd
lib.bench_flush_cost.argtypes = [ci, ci, ci, ci, dp, dp, cd, cd]
lib.bench_flush_cost.restype = cd
lib.bench_step_flush.argtypes = [ci, ci, ci, ci, dp, dp, cd, cd]
lib.bench_step_flush.restype = cd
lib.bench_iter.argtypes = [ci, ci, ci, ci, cd, cd, dp, cu, dp, cu, dp, cu, ci, dp]
lib.bench_iter.restype = cd
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

MODES = ((0, "0 separate, clean+invalidate (old)"), (1, "1 separate, clean only"),
         (2, "2 per-thread, clean only"), (3, "3 per-thread, clean+invalidate"), (4, "4 NO flush (expect wrong)"))

def iteration(mode, th, N, steps, dt, dx):
    init_state(N)
    out = (ctypes.c_double * 6)()
    ms = lib.bench_iter(mode, th, N, steps, dt, dx, P(ub[0]), ub[0].physical_address, P(ub[1]),
                        ub[1].physical_address, P(res), res.physical_address, 1, out)
    ok = ms > 0 and abs(out[1] - out[3]) <= 1e-9 * abs(out[3]) and abs(out[2] - out[4]) <= 1e-9 * abs(out[4])
    return ms, ok

rows = []
for N, steps in ((100000, 150), (10000, 400)):
    dx = 38.0 / (N - 1)
    dt = 0.5 * dx / 0.85
    init_state(N)
    print("\n=== N = %d (%d timed steps per run) ===" % (N, steps))

    fc0 = lib.bench_flush_cost(N, 0, 4, 40, P(ub[0]), P(ub[1]), dt, dx)
    fc1 = lib.bench_flush_cost(N, 1, 4, 40, P(ub[0]), P(ub[1]), dt, dx)
    print("Flush alone, one thread, right after a step (us):   clean+invalidate %8.1f | clean only %8.1f" % (fc0, fc1))
    print("Step time (us)                        1T %9.1f | 2T %9.1f | 4T %9.1f" % tuple(
        lib.bench_step_flush(N, -1, th, 30, P(ub[0]), P(ub[1]), dt, dx) for th in (1, 2, 4)))
    print("Step + per-thread clean only (us)     1T %9.1f | 2T %9.1f | 4T %9.1f" % tuple(
        lib.bench_step_flush(N, 1, th, 30, P(ub[0]), P(ub[1]), dt, dx) for th in (1, 2, 4)))
    print("Step + per-thread clean+inval (us)    1T %9.1f | 2T %9.1f | 4T %9.1f" % tuple(
        lib.bench_step_flush(N, 0, th, 30, P(ub[0]), P(ub[1]), dt, dx) for th in (1, 2, 4)))

    cpu = [iteration(0, th, N, steps, dt, dx)[0] for th in (1, 2, 4)]
    print("CPU-only reference (ms/step)          1T %9.3f | 2T %9.3f | 4T %9.3f" % tuple(cpu))
    rows.append([N, "cpu_only", "-", cpu[0], cpu[1], cpu[2], True])

    for fmode, fname in ((2, "FPGA overlapped"), (1, "FPGA waiting")):
        print("\n%s, whole iteration (ms/step)              1T        2T        4T    FPGA answers correct?" % fname)
        for fm, label in MODES:
            lib.set_flush_mode(fm)
            r = [iteration(fmode, th, N, steps, dt, dx) for th in (1, 2, 4)]
            allok = all(x[1] for x in r)
            print("  %-36s %9.3f %9.3f %9.3f    %s" % (label, r[0][0], r[1][0], r[2][0], "yes" if allok else "NO"))
            rows.append([N, fname, fm, r[0][0], r[1][0], r[2][0], allok])
    lib.set_flush_mode(0)

csv_name = "bench_flush_%s_%dMHz.csv" % (BIT_NAME.replace(".bit", ""), int(Clocks.fclk0_mhz))
with open(csv_name, "w") as f:
    w = csv.writer(f)
    w.writerow(["N", "setup", "flush_mode", "ms_1T", "ms_2T", "ms_4T", "fpga_answers_correct"])
    w.writerows(rows)
print("\nwrote", csv_name)
