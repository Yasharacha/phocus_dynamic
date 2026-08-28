// host_tv_cfl.cpp
// XRT native API host driver.
//
// Demonstrates the full CPU/FPGA co-execution loop:
//   CPU  : Lax-Friedrichs time-step
//   FPGA : TV + CFL reduction (replaces compute_cfl_serial + total_variation_serial)
//
// Build (after generating tv_cfl.xclbin via Vitis):
//   g++ -O2 -std=c++17 -o host_tv_cfl host_tv_cfl.cpp \
//       -I$(XILINX_XRT)/include \
//       -L$(XILINX_XRT)/lib -lxrt_coreutil -pthread
//
// Run:
//   ./host_tv_cfl tv_cfl.xclbin <flux_id> <N> <n_steps>
//   e.g.: ./host_tv_cfl tv_cfl.xclbin 0 100000 25   # Burgers, N=100k

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// XRT native API headers (installed with Vitis / XRT runtime)
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Flux IDs must match flux_functions.hpp
enum FluxID {
    FLUX_BURGERS     = 0,
    FLUX_LWR         = 1,
    FLUX_FLOOD_WAVE  = 2,
    FLUX_CUBIC       = 3,
    FLUX_BUCKLEY_LEV = 4,
    FLUX_LOG         = 5,
    FLUX_TRILINEAR   = 6
};

// ---- CPU step kernel (unchanged from existing code) -------------------------

static inline int periodic(int i, int N)
{
    if (i < 0)  return i + N;
    if (i >= N) return i - N;
    return i;
}

static inline double burgers_flux(double u) { return 0.5 * u * u; }

static void step_lax_friedrichs(const std::vector<double>& u,
                                 std::vector<double>& u_new,
                                 double dt, double dx)
{
    const int N = (int)u.size();
    for (int i = 0; i < N; ++i) {
        int im = periodic(i - 1, N);
        int ip = periodic(i + 1, N);
        u_new[i] = 0.5 * (u[ip] + u[im])
                 - 0.5 * (dt / dx) * (burgers_flux(u[ip]) - burgers_flux(u[im]));
    }
}

// ---- Timing helper ----------------------------------------------------------

using hrc = std::chrono::high_resolution_clock;

struct Timer {
    hrc::time_point t0 = hrc::now();
    double ms() const {
        return std::chrono::duration<double, std::milli>(hrc::now() - t0).count();
    }
};

// ---- Main -------------------------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <xclbin> <flux_id> <N> <n_steps>\n"
            "  flux_id: 0=Burgers 1=LWR 2=FloodWave 3=Cubic 4=Buckley 5=Log 6=Trilinear\n",
            argv[0]);
        return 1;
    }

    const std::string xclbin_path = argv[1];
    const int  flux_id = std::stoi(argv[2]);
    const int  N       = std::stoi(argv[3]);
    const int  n_steps = std::stoi(argv[4]);

    // Simulation parameters (Burgers-like defaults)
    const double x_min = 0.0, x_max = 38.0;
    const double dx = (x_max - x_min) / (N - 1);
    const double dt = 1.0;  // dt=1.0 for Burgers; adjust for other fluxes

    printf("Config: flux_id=%d  N=%d  n_steps=%d  dt=%.4f  dx=%.6f\n",
           flux_id, N, n_steps, dt, dx);

    // ---- Initialise XRT device and load xclbin ----
    printf("Loading device 0 ...\n");
    auto device = xrt::device(0);
    printf("Loading xclbin: %s\n", xclbin_path.c_str());
    auto uuid   = device.load_xclbin(xclbin_path);
    auto kernel = xrt::kernel(device, uuid, "tv_cfl_kernel");

    // ---- Allocate device buffers ----
    // group_id(0) = port u       → gmem0
    // group_id(1) = port results → gmem1
    const size_t u_bytes       = N * sizeof(double);
    const size_t results_bytes = 2 * sizeof(double);

    auto buf_u       = xrt::bo(device, u_bytes,       kernel.group_id(0));
    auto buf_results = xrt::bo(device, results_bytes,  kernel.group_id(1));

    // Host-side mirrors
    double* h_u       = buf_u.map<double*>();
    double* h_results = buf_results.map<double*>();

    // ---- Initial condition (Burgers) ----
    std::vector<double> u_cpu(N), u_new_cpu(N);
    for (int i = 0; i < N; ++i) {
        double x = x_min + i * dx;
        u_cpu[i] = (x < 15.01) ? -0.015 * x * (x - 15.0) : 0.0;
    }

    // ---- Time-loop timing accumulators ----
    double ms_step   = 0.0;   // CPU Lax-Friedrichs
    double ms_h2d    = 0.0;   // Host → device transfer
    double ms_kernel = 0.0;   // FPGA kernel execution
    double ms_d2h    = 0.0;   // Device → host transfer

    printf("\nRunning %d steps ...\n\n", n_steps);

    for (int step = 0; step < n_steps; ++step) {

        // 1. CPU: advance one time step
        {
            Timer t;
            step_lax_friedrichs(u_cpu, u_new_cpu, dt, dx);
            ms_step += t.ms();
        }
        u_cpu.swap(u_new_cpu);

        // 2. Copy current state to device buffer and sync
        {
            Timer t;
            std::memcpy(h_u, u_cpu.data(), u_bytes);
            buf_u.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            ms_h2d += t.ms();
        }

        // 3. Launch FPGA kernel and wait
        {
            Timer t;
            auto run = kernel(buf_u, buf_results, N, dt, dx, flux_id);
            run.wait();
            ms_kernel += t.ms();
        }

        // 4. Sync results back to host
        {
            Timer t;
            buf_results.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            ms_d2h += t.ms();
        }

        const double tv  = h_results[0];
        const double cfl = h_results[1];

        if (step < 5 || step == n_steps - 1) {
            printf("  step %4d  TV=%.6f  CFL=%.6f\n", step + 1, tv, cfl);
        }
    }

    // ---- Print timing summary ----
    const double total_fpga = ms_h2d + ms_kernel + ms_d2h;
    const double total_all  = ms_step + total_fpga;
    auto pct = [&](double ms){ return 100.0 * ms / total_all; };

    printf("\n=== Timing summary (%d steps, N=%d) ===\n", n_steps, N);
    printf("  CPU step kernel : %8.3f ms  (%.1f%%)\n", ms_step,   pct(ms_step));
    printf("  H→D transfer    : %8.3f ms  (%.1f%%)\n", ms_h2d,    pct(ms_h2d));
    printf("  FPGA kernel     : %8.3f ms  (%.1f%%)\n", ms_kernel, pct(ms_kernel));
    printf("  D→H transfer    : %8.3f ms  (%.1f%%)\n", ms_d2h,    pct(ms_d2h));
    printf("  ─────────────────────────────────────\n");
    printf("  TOTAL           : %8.3f ms\n", total_all);
    printf("\n  Per-step FPGA overhead : %.3f ms  (H2D+kernel+D2H)\n",
           total_fpga / n_steps);

    return 0;
}
