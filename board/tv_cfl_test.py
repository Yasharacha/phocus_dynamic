# tv_cfl_test.py - run on the Ultra96 (PYNQ 2.4, Python 3.6) in a Jupyter notebook.
# Each "# %%" marker starts a new notebook cell. Written for the Burgers-only kernel.
#
# Files needed in the SAME folder as this notebook on the board (same base name):
#   tv_cfl_wide150.bit     tv_cfl_wide150.hwh      Burgers, 2 elements/clock, 150 MHz
#   tv_cfl_buckley150.bit  tv_cfl_buckley150.hwh   Buckley-Leverett, 2 elements/clock, 150 MHz
# (all of them live in the board folder of the project on the PC)
#
# Not tested on hardware yet. PYNQ 2.4 is old: no pynq.allocate, no capture_output.

# %% Cell 1: load the bitstream
import os, time, struct
import numpy as np
from pynq import Overlay, Xlnk, MMIO, Bitstream

BIT_NAME = "tv_cfl_wide150.bit"  # or "tv_cfl_buckley150.bit"
PL_MHZ = 150.0                   # clock to run at: 150.0 for either bitstream
import sys
if len(sys.argv) > 2 and sys.argv[1].endswith(".bit"):    # python3 tv_cfl_test.py tv_cfl_wide150.bit 150
    BIT_NAME, PL_MHZ = sys.argv[1], float(sys.argv[2])
BIT = os.path.abspath(BIT_NAME)
print("bitstream:", BIT, "exists:", os.path.exists(BIT))
print("hwh exists:", os.path.exists(BIT.replace(".bit", ".hwh")))

BASE = 0xA0000000          # kernel register block, from the Vivado address map (64 KB)
ol = None
try:
    ol = Overlay(BIT)
    print("Overlay loaded. IPs found:", list(ol.ip_dict.keys()))
    for name, info in ol.ip_dict.items():
        print("  ", name, hex(info["phys_addr"]))
        if "tv_cfl" in name:
            BASE = info["phys_addr"]
except Exception as e:
    print("Overlay() failed:", repr(e))
    print("Falling back to a plain bitstream download; assuming BASE =", hex(BASE))
    Bitstream(BIT).download()
print("Kernel base address:", hex(BASE))

# Run at a clock Vivado verified. The board boots with the PL clock at ~187.5 MHz, which
# neither build was timed for (tv_cfl: closed at 100 MHz with +3.47 ns slack; wide150: see its report).
from pynq import Clocks
print("PL clock before:", Clocks.fclk0_mhz, "MHz")
Clocks.fclk0_mhz = PL_MHZ
print("PL clock now   :", Clocks.fclk0_mhz, "MHz")

# %% Cell 2: register map + helpers
mmio = MMIO(BASE, 0x10000)
CTRL, RET = 0x00, 0x10                 # ap_ctrl_hs: bit0 start, bit1 done, bit2 idle
U_LO, U_HI = 0x18, 0x1c                # pointer u
R_LO, R_HI = 0x24, 0x28                # pointer results
N_REG = 0x30
DT_LO, DT_HI = 0x38, 0x3c              # doubles are split into two 32-bit words
DX_LO, DX_HI = 0x44, 0x48
FLUX = 0x50                            # ignored: the Burgers-only build is fixed
TVP_LO, TVP_HI = 0x58, 0x5c            # tv_prev (negative = skip the TV test)
TOL_LO, TOL_HI = 0x64, 0x68

def write_double(lo, hi, x):
    l, h = struct.unpack("<II", struct.pack("<d", float(x)))
    mmio.write(lo, l)
    mmio.write(hi, h)

xlnk = Xlnk()

def run_kernel(u_np, dt, dx, tv_prev=-1.0, tol=1e-5, timeout_s=5.0):
    """Returns (tv, cfl, status, seconds). status bit0 = CFL>1, bit1 = TV increased."""
    N = len(u_np)
    # even length: the wide build reads 2 doubles per beat, so an odd N needs one spare element
    u_buf = xlnk.cma_array(shape=(N + (N & 1),), dtype=np.float64)
    res_buf = xlnk.cma_array(shape=(2,), dtype=np.float64)
    u_buf[:] = 0.0
    u_buf[:N] = u_np
    res_buf[:] = 0.0
    for b in (u_buf, res_buf):
        if hasattr(b, "flush"):
            b.flush()                  # make sure the FPGA sees the CPU's data

    pa = u_buf.physical_address
    ra = res_buf.physical_address
    mmio.write(U_LO, pa & 0xFFFFFFFF); mmio.write(U_HI, pa >> 32)
    mmio.write(R_LO, ra & 0xFFFFFFFF); mmio.write(R_HI, ra >> 32)
    mmio.write(N_REG, N)
    write_double(DT_LO, DT_HI, dt)
    write_double(DX_LO, DX_HI, dx)
    write_double(TVP_LO, TVP_HI, tv_prev)
    write_double(TOL_LO, TOL_HI, tol)
    mmio.write(FLUX, 0)

    mmio.read(CTRL)                    # reading clears any stale 'done' bit
    t0 = time.perf_counter()
    mmio.write(CTRL, 0x1)              # ap_start
    while not (mmio.read(CTRL) & 0x2): # wait for ap_done
        if time.perf_counter() - t0 > timeout_s:
            raise RuntimeError("kernel did not finish within %.1f s" % timeout_s)
    dt_s = time.perf_counter() - t0
    status = mmio.read(RET)
    if hasattr(res_buf, "invalidate"):
        res_buf.invalidate()
    tv, cfl = float(res_buf[0]), float(res_buf[1])
    for b in (u_buf, res_buf):
        try:
            b.freebuffer()
        except Exception:
            pass
    return tv, cfl, status, dt_s

def reference(u, dt, dx):
    tv = float(np.sum(np.abs(np.roll(u, -1) - u)))   # periodic total variation
    cfl = float(np.max(np.abs(u))) * dt / dx          # Burgers: |f'(u)| = |u|
    return tv, cfl

def make_u(N):
    x = np.linspace(0.0, 38.0, N)
    return np.where(x < 15.01, -0.015 * x * (x - 15.0), 0.0)

print("idle bit before start:", bool(mmio.read(CTRL) & 0x4))

# %% Cell 3: tests (FPGA result vs numpy reference)
all_ok = True
print("--- values and CFL flag ---")
for N, dt in [(20, 1.0), (1000, 1.0), (1000, 0.01), (100000, 1.0)]:
    dx = 38.0 / (N - 1)
    u = make_u(N)
    rtv, rcfl = reference(u, dt, dx)
    tv, cfl, status, secs = run_kernel(u, dt, dx)
    t0 = time.perf_counter(); reference(u, dt, dx); np_s = time.perf_counter() - t0
    tv_ok = abs(tv - rtv) <= 1e-10 * rtv + 1e-14
    cfl_ok = abs(cfl - rcfl) <= 1e-10 * rcfl + 1e-14
    want = 1 if rcfl > 1.0 else 0
    flag_ok = (status == want)
    all_ok = all_ok and tv_ok and cfl_ok and flag_ok
    print("N=%-6d dt=%-5g TV %s (%.8f vs %.8f)  CFL %s (%.6f vs %.6f)  status %d (want %d) %s"
          % (N, dt, "OK" if tv_ok else "FAIL", tv, rtv, "OK" if cfl_ok else "FAIL",
             cfl, rcfl, status, want, "OK" if flag_ok else "FAIL"))
    print("         kernel+overhead %.3f ms | expected ~%.3f ms at %.1f MHz | numpy %.3f ms"
          % (secs * 1e3, N / (Clocks.fclk0_mhz * 1e6) * 1e3, Clocks.fclk0_mhz, np_s * 1e3))

print("--- TV-increase flag (N=1000, dt=0.01 so CFL<1) ---")
N, dt = 1000, 0.01
dx = 38.0 / (N - 1)
u = make_u(N)
rtv, _ = reference(u, dt, dx)
for scale, want in [(0.5, 2), (1.0, 0), (2.0, 0), (-1.0, 0)]:
    tv, cfl, status, secs = run_kernel(u, dt, dx, tv_prev=(rtv * scale if scale > 0 else -1.0))
    ok = (status == want)
    all_ok = all_ok and ok
    print("tv_prev = %.4f x TV -> status %d (want %d) %s" % (scale, status, want, "OK" if ok else "FAIL"))

print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")

# %% Cell 4: clock sweep for the current bitstream (N = 100000, median of 5)
N = 100000
u = make_u(N)
dx = 38.0 / (N - 1)
for mhz in (100.0, 125.0, 150.0):
    Clocks.fclk0_mhz = mhz
    times = []
    ok = True
    for _ in range(5):
        tv, cfl, st, secs = run_kernel(u, 1.0, dx)
        rtv, rcfl = reference(u, 1.0, dx)
        ok = ok and abs(tv - rtv) <= 1e-10 * rtv and abs(cfl - rcfl) <= 1e-10 * rcfl
        times.append(secs)
    times.sort()
    print("PL clock %.1f MHz (reads back %.1f): median %.3f ms  correct=%s"
          % (mhz, Clocks.fclk0_mhz, times[2] * 1e3, ok))
Clocks.fclk0_mhz = PL_MHZ
