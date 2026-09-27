# bench_all.py - complete benchmark on the Ultra96 (PYNQ 2.4, Python 3.6), Jupyter cells.
# Each "# %%" marker starts a new notebook cell.
#
# Put these in the notebook folder on the board:
#   bench_e2e.cpp  bench_all.py(cells)  and the bitstream pair you want to measure
#   (tv_cfl_wide150.bit + tv_cfl_wide150.hwh, or tv_cfl_buckley150.bit + tv_cfl_buckley150.hwh)
#
# Easiest: run it as a script from a Jupyter Terminal, one command per (bitstream, clock):
#   python3 bench_all.py tv_cfl_wide150.bit 150
# (or paste the cells into a notebook and edit BIT_NAME / PL_MHZ; restart the kernel between bitstreams)
# Each run prints a table and writes bench_<bitstream>_<MHz>.csv - paste me the printed table.
#
# Not tested on hardware yet.

# %% Cell 1: settings, compile the C++ library, load the bitstream, set the PL clock
import os, time, ctypes, subprocess, csv
import numpy as np
from pynq import Overlay, Xlnk, Clocks, Bitstream

BIT_NAME = "tv_cfl_wide150.bit"   # or "tv_cfl_buckley150.bit" for the Buckley-Leverett kernel
PL_MHZ = 100.0                # 100.0 or 150.0 (wide150 is built for 150)
import sys
if len(sys.argv) > 2 and sys.argv[1].endswith(".bit"):    # python3 bench_all.py tv_cfl_wide150.bit 150
    BIT_NAME, PL_MHZ = sys.argv[1], float(sys.argv[2])
BASE = 0xA0000000             # kernel register block (Vivado address map)

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
lib.fpga_init.argtypes = [ctypes.c_uint64]
lib.fpga_init.restype = ctypes.c_int
lib.bench_cpu_step.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, dp, dp, ctypes.c_double, ctypes.c_double]
lib.bench_cpu_step.restype = ctypes.c_double
lib.bench_cpu_check.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, dp, ctypes.c_double, ctypes.c_double]
lib.bench_cpu_check.restype = ctypes.c_double
lib.bench_fpga_check.argtypes = [ctypes.c_int, ctypes.c_int, dp, ctypes.c_uint64, dp, ctypes.c_uint64,
                                 ctypes.c_double, ctypes.c_double, ctypes.c_int]
lib.bench_fpga_check.restype = ctypes.c_double
lib.bench_iter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                           dp, ctypes.c_uint64, dp, ctypes.c_uint64, dp, ctypes.c_uint64, ctypes.c_int, dp]
lib.bench_iter.restype = ctypes.c_double
rc = lib.fpga_init(BASE)
print("fpga_init:", rc, "(0 = registers mapped)")

# %% Cell 2: buffers (two solver states + a results buffer), and the initial condition
NMAX = 131072
xlnk = Xlnk()
CACHEABLE = 1
try:
    ub = [xlnk.cma_array(shape=(NMAX + 2,), dtype=np.float64, cacheable=1) for _ in range(2)]
except Exception as e:
    print("cacheable buffers not available (%r): using uncached ones - CPU numbers will be pessimistic" % (e,))
    CACHEABLE = 0
    ub = [xlnk.cma_array(shape=(NMAX + 2,), dtype=np.float64) for _ in range(2)]
res = xlnk.cma_array(shape=(2,), dtype=np.float64)          # small, uncached: the FPGA writes TV, CFL here
print("cacheable u buffers:", bool(CACHEABLE), "| physical:", hex(ub[0].physical_address), hex(ub[1].physical_address))

def P(a):                       # pointer to a numpy/cma array's data
    return ctypes.cast(a.ctypes.data, dp)

def init_state(N):
    x = np.linspace(0.0, 38.0, N)
    ub[0][:N] = np.where(x < 15.01, -0.015 * x * (x - 15.0), 0.0)
    ub[1][:N] = 0.0
    if hasattr(ub[0], "flush"):
        ub[0].flush()

# %% Cell 3: run everything and print the table
def run_size(N, steps):
    dx = 38.0 / (N - 1)
    dt = 0.5 * dx / 0.85                     # CFL about 0.5, so the run is stable (no violations)
    init_state(N)
    r = {"N": N, "steps": steps}
    for th in (1, 2, 4):
        r["cpu_step_us_%dT" % th] = lib.bench_cpu_step(N, th, 30, P(ub[0]), P(ub[1]), dt, dx)
    for th in (1, 2, 4):
        r["cpu_check_us_%dT" % th] = lib.bench_cpu_check(N, th, 30, P(ub[0]), dt, dx)
    if hasattr(ub[0], "flush"):
        ub[0].flush()
    r["fpga_check_us"] = lib.bench_fpga_check(N, 30, P(ub[0]), ub[0].physical_address, P(res),
                                              res.physical_address, dt, dx, CACHEABLE)
    checks = []
    for mode, name in ((0, "cpu_only"), (1, "fpga_sync"), (2, "fpga_overlap")):
        for th in (1, 2, 4):
            init_state(N)
            out = (ctypes.c_double * 6)()
            ms = lib.bench_iter(mode, th, N, steps, dt, dx, P(ub[0]), ub[0].physical_address,
                                P(ub[1]), ub[1].physical_address, P(res), res.physical_address, CACHEABLE, out)
            r["%s_ms_%dT" % (name, th)] = ms
            checks.append((name, th, out[0], out[1], out[2], out[3], out[4], int(out[5])))
    r["_checks"] = checks
    return r

results = [run_size(10000, 400), run_size(100000, 150)]

def fmt(x):
    return "%10.1f" % x if x >= 0 else "     ERROR(%d)" % x

print("\nPL clock %.1f MHz | bitstream %s | cacheable buffers: %s" % (Clocks.fclk0_mhz, BIT_NAME, bool(CACHEABLE)))
for r in results:
    print("\n=== N = %d (%d timed steps per run) ===" % (r["N"], r["steps"]))
    print("Check only (us per call):   FPGA %s | CPU 1T %s | CPU 2T %s | CPU 4T %s" % (
        fmt(r["fpga_check_us"]), fmt(r["cpu_check_us_1T"]), fmt(r["cpu_check_us_2T"]), fmt(r["cpu_check_us_4T"])))
    print("Step only  (us per call):   CPU 1T %s | CPU 2T %s | CPU 4T %s" % (
        fmt(r["cpu_step_us_1T"]), fmt(r["cpu_step_us_2T"]), fmt(r["cpu_step_us_4T"])))
    print("Whole iteration (ms/step)     1 thread   2 threads   4 threads")
    for name in ("cpu_only", "fpga_sync", "fpga_overlap"):
        print("  %-14s %10.3f %10.3f %10.3f" % (name, r[name + "_ms_1T"], r[name + "_ms_2T"], r[name + "_ms_4T"]))
    ref = [c for c in r["_checks"] if c[0] == "cpu_only" and c[1] == 1][0][2]
    same = all(abs(c[2] - ref) <= 1e-9 * abs(ref) for c in r["_checks"])
    fp = [c for c in r["_checks"] if c[0] != "cpu_only"]
    tv_ok = all(abs(c[3] - c[5]) <= 1e-9 * abs(c[5]) for c in fp)
    cfl_ok = all(abs(c[4] - c[6]) <= 1e-9 * abs(c[6]) for c in fp)
    viol = sum(c[7] for c in r["_checks"])
    print("  sanity: same final state in every mode: %s | FPGA TV matches CPU: %s | FPGA CFL matches CPU: %s | violations flagged: %d"
          % (same, tv_ok, cfl_ok, viol))

csv_name = "bench_%s_%dMHz.csv" % (BIT_NAME.replace(".bit", ""), int(Clocks.fclk0_mhz))
with open(csv_name, "w") as f:
    w = csv.writer(f)
    keys = [k for k in results[0].keys() if not k.startswith("_")]
    w.writerow(keys)
    for r in results:
        w.writerow([r[k] for k in keys])
print("\nwrote", csv_name)
