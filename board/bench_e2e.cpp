// bench_e2e.cpp - end-to-end benchmark for the TV/CFL FPGA offload on the Ultra96.
//
// Build ON THE BOARD (a shared library that bench_all.py loads through ctypes):
//   g++ -O3 -std=c++14 -fopenmp -shared -fPIC bench_e2e.cpp -o libbench.so
//
// Everything timed here runs in C++ (no Python in the loops). Physical buffers come from
// PYNQ's Xlnk().cma_array and are passed in as (virtual pointer, physical address) pairs.
//
// Solver: Burgers, Lax-Friedrichs, periodic (same arithmetic as cpu_baseline.cpp / the repo).
// Check : total variation + CFL number, done either on the CPU or by the FPGA kernel.
//
// Modes of bench_iter():
//   0 = CPU only          : step, then the check on the CPU (same thread count)
//   1 = FPGA, synchronous : step, flush, launch FPGA check, wait for it
//   2 = FPGA, overlapped  : step n+1 runs on the CPU while the FPGA checks step n
//
// Kernel register map (Vivado base address 0xA0000000, from the HLS control_s_axi.v):
//   0x00 ctrl (bit0 start, bit1 done, bit2 idle)   0x10 return value (violation bitmask)
//   0x18/0x1c u pointer   0x24/0x28 results pointer   0x30 N
//   0x38/0x3c dt   0x44/0x48 dx   0x50 flux_id   0x58/0x5c tv_prev   0x64/0x68 tol
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <omp.h>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#define HAVE_MMAP 1
#endif

using clk = std::chrono::steady_clock;
static inline double now_us() {
    return std::chrono::duration<double, std::micro>(clk::now().time_since_epoch()).count();
}
static volatile double g_sink = 0.0;   // keeps the compiler from dropping benchmark work

// ------------------------------------------------------------------ solver + CPU check
static inline int pidx(int i, int N) {
    if (i < 0)  return i + N;
    if (i >= N) return i - N;
    return i;
}
// CPU_FLUX_* selects a flux's f(u) and |f'(u)| in place of Burgers', so the CPU side matches
// whichever flux the FPGA bitstream was built for. f(u) is what the solver step uses;
// cpu_fprime_abs(u) is what the CFL check uses - both must match flux_functions.hpp's
// corresponding FluxID case exactly for the FPGA-vs-CPU sanity check (TV/CFL agreement) to hold.
// Define at most one of these when compiling; none selected defaults to Burgers.
#if defined(CPU_FLUX_BUCKLEY)
static inline double flux(double u) {
    const double u2 = u * u, a = 1.0 - u;
    const double denom = u2 + 0.25 * a * a;
    return (denom > 1e-14) ? (u2 / denom) : 0.0;
}
static inline double cpu_fprime_abs(double u) {
    const double u2 = u * u, a = 1.0 - u;
    const double denom = u2 + 0.25 * a * a;
    if (denom <= 1e-14) return 0.0;
    const double fp = 0.5 * u * (1.0 - u) / (denom * denom);
    return std::fabs(fp);
}
#elif defined(CPU_FLUX_FLOODWAVE)
// f(u) = max(u,0)^1.5 = u*sqrt(u) for u>0 (0 otherwise); f'(u) = 1.5*sqrt(u) for u>0 (0 otherwise).
// No domain restriction needed: both are well-defined, including at u=0, for the usual test profile.
static inline double flux(double u) { return (u > 0.0) ? u * std::sqrt(u) : 0.0; }
static inline double cpu_fprime_abs(double u) { return (u > 0.0) ? 1.5 * std::sqrt(u) : 0.0; }
#elif defined(CPU_FLUX_LOG)
// f(u) = ln(u); f'(u) = 1/u. Only defined for u>0 - ln(0) is -infinity, so this flux needs a
// strictly-positive test profile (bench_all_log.py shifts the usual initial condition up by
// 0.1). The u<=0 fallback below only matters if that profile guarantee is violated.
static inline double flux(double u) { return (u > 0.0) ? std::log(u) : 0.0; }
static inline double cpu_fprime_abs(double u) { return (u > 0.0) ? 1.0 / u : 0.0; }
#else
static inline double flux(double u) { return 0.5 * u * u; }
static inline double cpu_fprime_abs(double u) { return std::fabs(u); }   // Burgers: |f'(u)| = |u|
#endif

// The periodic wraparound only ever happens at the two global endpoints (i=0 needs u[N-1],
// i=N-1 needs u[0]); every other point's neighbours are already in bounds. So instead of an
// index-wrapping branch on every point (pidx(), removed from this hot path), the two endpoints
// are handled once, directly, and the interior loop runs with plain array indexing - no branch
// per point at all. Measured on the Ultra96 (Cortex-A53): roughly 2x faster than the pidx()
// version, at every thread count. Assumes N >= 2 (already required elsewhere: dx = 38/(N-1)).
static void step_range(const double* u, double* un, int N, double dt, double dx, int lo, int hi) {
    int core_lo = lo, core_hi = hi;
    if (lo == 0) {
        const int ip = 1, im = N - 1;      // i=0: right neighbour never wraps (N>=2), left does
        un[0] = 0.5 * (u[ip] + u[im]) - 0.5 * (dt / dx) * (flux(u[ip]) - flux(u[im]));
        core_lo = 1;
    }
    if (hi == N) {
        const int i = N - 1, ip = 0, im = N - 2;   // i=N-1: left never wraps, right does
        un[i] = 0.5 * (u[ip] + u[im]) - 0.5 * (dt / dx) * (flux(u[ip]) - flux(u[im]));
        core_hi = N - 1;
    }
    for (int i = core_lo; i < core_hi; ++i)
        un[i] = 0.5 * (u[i + 1] + u[i - 1]) - 0.5 * (dt / dx) * (flux(u[i + 1]) - flux(u[i - 1]));
}
static inline void cache_op(const void* p, size_t bytes, int op);   // defined in the cache section below

// fop >= 0: each thread also cache-cleans the output chunk it just wrote (cache_op mode fop).
static void step_lf(const double* u, double* un, int N, double dt, double dx, int threads,
                    int fop = -1) {
    if (threads <= 1) {
        step_range(u, un, N, dt, dx, 0, N);
        if (fop >= 0) cache_op(un, (size_t)N * sizeof(double), fop);
        return;
    }
#pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        int lo = (int)((long long)N * t / nt), hi = (int)((long long)N * (t + 1) / nt);
        step_range(u, un, N, dt, dx, lo, hi);
        if (fop >= 0) cache_op(un + lo, (size_t)(hi - lo) * sizeof(double), fop);
    }
}

// ---- Optimized variants, to test where step_lf's time actually goes ----
//
// variant 1: precompute flux(u[i]) once per point (the original calls flux() twice per point,
// once as each point's neighbour's im and again as its neighbour's ip).
static void step_range_fcache(const double* u, const double* f, double* un, int N, double dt,
                              double dx, int lo, int hi) {
    for (int i = lo; i < hi; ++i) {
        int im = pidx(i - 1, N), ip = pidx(i + 1, N);
        un[i] = 0.5 * (u[ip] + u[im]) - 0.5 * (dt / dx) * (f[ip] - f[im]);
    }
}
static void step_lf_fcache(const double* u, double* un, double* fbuf, int N, double dt, double dx,
                           int threads) {
    if (threads <= 1) {
        for (int i = 0; i < N; ++i) fbuf[i] = flux(u[i]);
        step_range_fcache(u, fbuf, un, N, dt, dx, 0, N);
        return;
    }
#pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        int lo = (int)((long long)N * t / nt), hi = (int)((long long)N * (t + 1) / nt);
        for (int i = lo; i < hi; ++i) fbuf[i] = flux(u[i]);
#pragma omp barrier
        step_range_fcache(u, fbuf, un, N, dt, dx, lo, hi);
    }
}

// variant 2: no periodic_index branch. u/un point one element INTO a buffer with one ghost cell
// on each side (index -1 and N are valid); the caller mirrors u[-1]=u[N-1] and u[N]=u[0] once,
// outside the parallel region, and un's ghost cells don't need setting (never read this step).
static void step_range_nobranch(const double* u, double* un, double dt, double dx, int lo, int hi) {
    for (int i = lo; i < hi; ++i)
        un[i] = 0.5 * (u[i + 1] + u[i - 1]) - 0.5 * (dt / dx) * (flux(u[i + 1]) - flux(u[i - 1]));
}
static void step_lf_nobranch(double* u_g, double* un_g, int N, double dt, double dx, int threads) {
    // u_g, un_g point at index 0 of an (N+2)-length allocation; index 1..N is u[0..N-1].
    double* u  = u_g + 1;
    double* un = un_g + 1;
    u[-1] = u[N - 1];
    u[N]  = u[0];
    if (threads <= 1) { step_range_nobranch(u, un, dt, dx, 0, N); return; }
#pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        int lo = (int)((long long)N * t / nt), hi = (int)((long long)N * (t + 1) / nt);
        step_range_nobranch(u, un, dt, dx, lo, hi);
    }
}

// Same fix as step_range: only i=N-1's neighbour wraps (to u[0]); handled once, outside the
// (now branchless) loop over i=0..N-2.
static void check_fused(const double* u, int N, double dt, double dx, int threads,
                        double* tv, double* cfl) {
    double s = 0.0, m = 0.0;
    if (threads <= 1) {
        for (int i = 0; i < N - 1; ++i) {
            s += std::fabs(u[i + 1] - u[i]);
            m = std::max(m, cpu_fprime_abs(u[i]));
        }
    } else {
#pragma omp parallel for num_threads(threads) reduction(+ : s) reduction(max : m) schedule(static)
        for (int i = 0; i < N - 1; ++i) {
            s += std::fabs(u[i + 1] - u[i]);
            m = std::max(m, cpu_fprime_abs(u[i]));
        }
    }
    s += std::fabs(u[0] - u[N - 1]);   // the wrap term, done once
    m = std::max(m, cpu_fprime_abs(u[N - 1]));
    *tv = s;
    *cfl = m * dt / dx;   // |f'(u)| via cpu_fprime_abs (Burgers |u|, or Buckley-Leverett under CPU_FLUX_BUCKLEY)
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static double meanv(const std::vector<double>& v) {
    double s = 0.0; for (double x : v) s += x; return v.empty() ? 0.0 : s / v.size();
}
static double maxv(const std::vector<double>& v) {
    double m = 0.0; for (double x : v) m = std::max(m, x); return m;
}
// Fills out[0..2] = median, mean, max of v (in that order). Lets a caller see whether a phase's
// mean is dragged up by a few slow outlier iterations (mean >> median, large max) or is
// genuinely, consistently slower than its median suggests (mean close to median, small max).
static void stats3(std::vector<double>& v, double* out) {
    out[1] = meanv(v);
    out[2] = maxv(v);
    out[0] = median(v);   // sorts v; do this last
}

// ------------------------------------------------------------------ cache maintenance
// The FPGA reads DDR through a non-coherent port, so data the CPU wrote (and that may still
// sit in its cache) must be pushed out first. aarch64 allows this from user space.
// op 0 = clean + invalidate (dc civac): the data reaches DRAM and leaves the CPU cache.
// op 1 = clean only (dc cvac): the data reaches DRAM and stays cached, so the CPU's next
//        reads of it (the following time step) still hit the cache.
static inline void cache_op(const void* p, size_t bytes, int op) {
#if defined(__aarch64__)
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)63;
    uintptr_t e = (uintptr_t)p + bytes;
    if (op == 0) { for (; a < e; a += 64) asm volatile("dc civac, %0" ::"r"(a) : "memory"); }
    else         { for (; a < e; a += 64) asm volatile("dc cvac, %0" ::"r"(a) : "memory"); }
    asm volatile("dsb sy" ::: "memory");
#else
    (void)p; (void)bytes; (void)op;
#endif
}

// Flush mode used by bench_iter for the FPGA modes (set with set_flush_mode):
//   0 = one thread flushes the whole array after the step, clean+invalidate  (original)
//   1 = one thread flushes the whole array after the step, clean only
//   2 = every thread cleans its own chunk right after computing it, clean only
//   3 = every thread flushes its own chunk right after computing it, clean+invalidate
//   4 = no flush at all (diagnostic: the FPGA is expected to read stale data)
static int g_flush_mode = 0;
extern "C" void set_flush_mode(int m) { g_flush_mode = m; }

// ------------------------------------------------------------------ FPGA kernel access
static volatile uint8_t* g_regs = nullptr;

extern "C" int fpga_init(uint64_t base) {
#ifdef HAVE_MMAP
    if (g_regs) return 0;
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return -1;
    void* p = mmap(nullptr, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)base);
    if (p == MAP_FAILED) return -2;
    g_regs = (volatile uint8_t*)p;
    return 0;
#else
    (void)base;
    return -3;   // no /dev/mem on this platform
#endif
}
static inline void w32(uint32_t off, uint32_t v) { *(volatile uint32_t*)(g_regs + off) = v; }
static inline uint32_t r32(uint32_t off) { return *(volatile uint32_t*)(g_regs + off); }
static inline void w64d(uint32_t off, double d) {
    uint64_t b; std::memcpy(&b, &d, 8);
    w32(off, (uint32_t)b);
    w32(off + 4, (uint32_t)(b >> 32));
}

static void fpga_launch(uint64_t u_phys, uint64_t res_phys, int N, double dt, double dx,
                        double tv_prev, double tol) {
    w32(0x18, (uint32_t)u_phys);   w32(0x1c, (uint32_t)(u_phys >> 32));
    w32(0x24, (uint32_t)res_phys); w32(0x28, (uint32_t)(res_phys >> 32));
    w32(0x30, (uint32_t)N);
    w64d(0x38, dt);
    w64d(0x44, dx);
    w32(0x50, 0);
    w64d(0x58, tv_prev);
    w64d(0x64, tol);
    (void)r32(0x00);               // reading the control register clears a stale 'done'
#if defined(__aarch64__)
    asm volatile("dsb sy" ::: "memory");
#endif
    w32(0x00, 1);                  // ap_start
}
// returns 0 when done, -1 on timeout (~5 s)
static int fpga_wait() {
    double t0 = now_us();
    uint64_t spins = 0;
    while (!(r32(0x00) & 0x2)) {
        if ((++spins & 0xFFFFF) == 0 && now_us() - t0 > 5e6) return -1;
    }
    return 0;
}

// ------------------------------------------------------------------ public benchmarks
// Median microseconds for one step (buffers: b0 -> b1).
extern "C" double bench_cpu_step(int N, int threads, int reps, double* b0, double* b1,
                                 double dt, double dx) {
    std::vector<double> t;
    for (int i = 0; i < 3; ++i) step_lf(b0, b1, N, dt, dx, threads);
    for (int i = 0; i < reps; ++i) {
        double a = now_us();
        step_lf(b0, b1, N, dt, dx, threads);
        t.push_back(now_us() - a);
        g_sink = b1[N / 2];
    }
    return median(t);
}

// Median microseconds for one CPU-side fused TV+CFL check.
// Median us for the flux-cached step variant. fbuf must have room for N doubles.
extern "C" double bench_cpu_step_fcache(int N, int threads, int reps, double* b0, double* b1,
                                        double* fbuf, double dt, double dx) {
    std::vector<double> t;
    for (int i = 0; i < 3; ++i) step_lf_fcache(b0, b1, fbuf, N, dt, dx, threads);
    for (int i = 0; i < reps; ++i) {
        double a = now_us();
        step_lf_fcache(b0, b1, fbuf, N, dt, dx, threads);
        t.push_back(now_us() - a);
        g_sink = b1[N / 2];
    }
    return median(t);
}

// Median us for the branchless step variant. b0_g/b1_g must be (N+2)-long; index 1..N is u[0..N-1].
extern "C" double bench_cpu_step_nobranch(int N, int threads, int reps, double* b0_g, double* b1_g,
                                          double dt, double dx) {
    std::vector<double> t;
    for (int i = 0; i < 3; ++i) step_lf_nobranch(b0_g, b1_g, N, dt, dx, threads);
    for (int i = 0; i < reps; ++i) {
        double a = now_us();
        step_lf_nobranch(b0_g, b1_g, N, dt, dx, threads);
        t.push_back(now_us() - a);
        g_sink = b1_g[1 + N / 2];
    }
    return median(t);
}

extern "C" double bench_cpu_check(int N, int threads, int reps, const double* b0, double dt, double dx) {
    std::vector<double> t;
    double tv = 0, cfl = 0;
    for (int i = 0; i < 3; ++i) check_fused(b0, N, dt, dx, threads, &tv, &cfl);
    for (int i = 0; i < reps; ++i) {
        double a = now_us();
        check_fused(b0, N, dt, dx, threads, &tv, &cfl);
        t.push_back(now_us() - a);
        g_sink = tv + cfl;
    }
    return median(t);
}

// Median microseconds for cache_op(op) over a freshly written array of N doubles, one thread.
// (Each rep first runs an untimed step b0 -> b1, so b1 is dirty in the cache, as in the loop.)
extern "C" double bench_flush_cost(int N, int op, int threads, int reps, double* b0, double* b1,
                                   double dt, double dx) {
    std::vector<double> t;
    for (int i = -2; i < reps; ++i) {
        step_lf(b0, b1, N, dt, dx, threads);
        double a = now_us();
        cache_op(b1, (size_t)N * sizeof(double), op);
        double d = now_us() - a;
        if (i >= 0) t.push_back(d);
        g_sink = b1[N / 2];
    }
    return median(t);
}

// Median microseconds for one step where every thread also cleans its own output chunk (fop).
extern "C" double bench_step_flush(int N, int fop, int threads, int reps, double* b0, double* b1,
                                   double dt, double dx) {
    std::vector<double> t;
    for (int i = -3; i < reps; ++i) {
        double a = now_us();
        step_lf(b0, b1, N, dt, dx, threads, fop);
        double d = now_us() - a;
        if (i >= 0) t.push_back(d);
        g_sink = b1[N / 2];
    }
    return median(t);
}

// Median microseconds for one FPGA check (launch + wait), data already in b0/p0.
// Returns -1 on timeout, -3 if the registers are not mapped.
extern "C" double bench_fpga_check(int N, int reps, const double* b0, uint64_t p0,
                                   volatile double* res, uint64_t rp, double dt, double dx,
                                   int cacheable) {
    if (!g_regs) return -3;
    if (cacheable) cache_op(b0, (size_t)N * sizeof(double), 0);
    std::vector<double> t;
    for (int i = -3; i < reps; ++i) {
        double a = now_us();
        fpga_launch(p0, rp, N, dt, dx, -1.0, 1e-5);
        if (fpga_wait()) return -1;
        double d = now_us() - a;
        if (i >= 0) t.push_back(d);
    }
    g_sink = res[0] + res[1];
    return median(t);
}

// Like bench_iter(mode=2, ...) but also fills phase[0..3] with the median MICROSECONDS spent per
// iteration in: step, cache-flush, register-launch-of-the-FPGA, and waiting for the previous
// check to finish (0 whenever the FPGA was already done, i.e. fully hidden).
// phase[] gets 4 groups of 3: (step, flush, launch, wait) x (median, mean, max), all in us.
// A phase whose mean is much bigger than its median (with a big max) means a few slow outlier
// iterations are dragging the average up; a phase whose mean tracks its median closely means it
// is consistently that slow every iteration, not just occasionally.
extern "C" double bench_iter_profiled(int threads, int N, int steps, double dt, double dx,
                                      double* b0, uint64_t p0, double* b1, uint64_t p1,
                                      volatile double* res, uint64_t rp, int cacheable,
                                      double* phase) {
    if (!g_regs) return -3;
    double* bv[2] = {b0, b1};
    uint64_t bp[2] = {p0, p1};
    const size_t bytes = (size_t)N * sizeof(double);
    const int warm = 5;
    int cur = 0;
    bool busy = false;
    int sep_op = cacheable ? 0 : -1;
    std::vector<double> t_step, t_flush, t_launch, t_wait;
    t_step.reserve(steps); t_flush.reserve(steps); t_launch.reserve(steps); t_wait.reserve(steps);

    for (int k = 0; k < warm + steps; ++k) {
        int nxt = 1 - cur;
        double a = now_us();
        step_lf(bv[cur], bv[nxt], N, dt, dx, threads);
        double b = now_us();
        if (sep_op >= 0) cache_op(bv[nxt], bytes, sep_op);
        double c = now_us();
        double waited = 0.0;
        if (busy) {
            double d0 = now_us();
            if (fpga_wait()) return -1;
            waited = now_us() - d0;
        }
        double e = now_us();
        fpga_launch(bp[nxt], rp, N, dt, dx, -1.0, 1e-5);
        busy = true;
        double f = now_us();
        if (k >= warm) {
            t_step.push_back(b - a);
            t_flush.push_back(c - b);
            t_wait.push_back(waited);
            t_launch.push_back(f - e);
        }
        cur = nxt;
    }
    if (busy) fpga_wait();
    stats3(t_step,   phase + 0);
    stats3(t_flush,  phase + 3);
    stats3(t_launch, phase + 6);
    stats3(t_wait,   phase + 9);
    return phase[0] + phase[3] + phase[6] + phase[9];   // sum of the four medians
}

// Same idea for CPU-only mode: breaks the REAL loop (not an isolated call) into its step and
// check phases. phase[] gets 2 groups of 3: (step, check) x (median, mean, max), in us.
// Comparing this against bench_cpu_step()/bench_cpu_check() (which time an isolated call
// repeated many times) shows whether the real loop costs more than those isolated numbers
// suggest, and if so, whether it is the step or the check phase that differs.
extern "C" double bench_iter_profiled_cpu(int threads, int N, int steps, double dt, double dx,
                                          double* b0, double* b1, double* phase) {
    double* bv[2] = {b0, b1};
    const int warm = 5;
    int cur = 0;
    double tv = 0.0, cfl = 0.0;
    std::vector<double> t_step, t_check;
    t_step.reserve(steps); t_check.reserve(steps);

    for (int k = 0; k < warm + steps; ++k) {
        int nxt = 1 - cur;
        double a = now_us();
        step_lf(bv[cur], bv[nxt], N, dt, dx, threads);
        double b = now_us();
        check_fused(bv[nxt], N, dt, dx, threads, &tv, &cfl);
        double c = now_us();
        g_sink = tv + cfl;
        if (k >= warm) { t_step.push_back(b - a); t_check.push_back(c - b); }
        cur = nxt;
    }
    stats3(t_step,  phase + 0);
    stats3(t_check, phase + 3);
    return phase[0] + phase[3];
}

// Runs (5 warm-up + `steps`) solver steps and returns milliseconds per step.
// out[0]=checksum of final u, out[1]=FPGA TV, out[2]=FPGA CFL, out[3]=CPU TV, out[4]=CPU CFL,
// out[5]=number of steps whose check flagged a violation (CFL bit set; TV test is disabled).
// Returns -1 on FPGA timeout, -3 if the FPGA registers are not mapped (modes 1, 2).
extern "C" double bench_iter(int mode, int threads, int N, int steps, double dt, double dx,
                             double* b0, uint64_t p0, double* b1, uint64_t p1,
                             volatile double* res, uint64_t rp, int cacheable, double* out) {
    if (mode != 0 && !g_regs) return -3;
    double* bv[2] = {b0, b1};
    uint64_t bp[2] = {p0, p1};
    const size_t bytes = (size_t)N * sizeof(double);
    const int warm = 5;
    int cur = 0, viol = 0;
    bool busy = false;
    double t0 = 0.0, tv = 0.0, cfl = 0.0;
    int fused_op = -1, sep_op = -1;             // how the array is made visible to the FPGA
    if (mode != 0 && cacheable) {
        switch (g_flush_mode) {
            case 0: sep_op = 0; break;
            case 1: sep_op = 1; break;
            case 2: fused_op = 1; break;
            case 3: fused_op = 0; break;
            default: break;                     // 4: no flush (diagnostic)
        }
    }

    for (int k = 0; k < warm + steps; ++k) {
        if (k == warm) t0 = now_us();
        int nxt = 1 - cur;
        step_lf(bv[cur], bv[nxt], N, dt, dx, threads, fused_op);
        if (mode == 0) {
            check_fused(bv[nxt], N, dt, dx, threads, &tv, &cfl);
            g_sink = tv + cfl;
            if (cfl > 1.0) ++viol;
        } else {
            if (sep_op >= 0) cache_op(bv[nxt], bytes, sep_op);
            if (mode == 1) {
                fpga_launch(bp[nxt], rp, N, dt, dx, -1.0, 1e-5);
                if (fpga_wait()) return -1;
                if (r32(0x10) & 1) ++viol;
            } else {
                if (busy) {                       // finish the check of the previous state
                    if (fpga_wait()) return -1;
                    if (r32(0x10) & 1) ++viol;
                }
                fpga_launch(bp[nxt], rp, N, dt, dx, -1.0, 1e-5);
                busy = true;                      // runs while the next step computes
            }
        }
        cur = nxt;
    }
    if (mode == 2 && busy) {
        if (fpga_wait()) return -1;
        if (r32(0x10) & 1) ++viol;
    }
    double t1 = now_us();

    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += bv[cur][i];
    double ctv = 0, ccfl = 0;
    check_fused(bv[cur], N, dt, dx, 1, &ctv, &ccfl);
    out[0] = sum;
    out[1] = (mode != 0) ? res[0] : ctv;
    out[2] = (mode != 0) ? res[1] : ccfl;
    out[3] = ctv;
    out[4] = ccfl;
    out[5] = (double)viol;
    return (t1 - t0) / 1e3 / steps;
}
