// tb_tv_cfl.cpp
// C-simulation testbench for tv_cfl_kernel.
//
// Build and run with Vitis HLS:
//   vitis_hls -f run_csim.tcl
//
// Or compile standalone (no HLS headers needed for C-sim):
//   g++ -O2 -o tb_tv_cfl tb_tv_cfl.cpp tv_cfl_kernel.cpp -I. && ./tb_tv_cfl

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tv_cfl_kernel.hpp"

// ---- CPU reference implementations (matches combined.cpp exactly) ----------

static double ref_total_variation(const std::vector<double>& u)
{
    const int N = (int)u.size();
    if (N <= 1) return 0.0;
    double tv = 0.0;
    for (int i = 0; i < N - 1; ++i)
        tv += std::abs(u[i + 1] - u[i]);
    tv += std::abs(u[0] - u[N - 1]);   // periodic wrap
    return tv;
}

static double ref_compute_cfl(const std::vector<double>& u,
                               double dt, double dx, int flux_id)
{
    double max_speed = 0.0;
    for (double ui : u)
        max_speed = std::max(max_speed, flux_prime_abs(ui, flux_id));
    return max_speed * dt / dx;
}

// ---- Helper: generate Burgers-like initial condition -------------------------

static std::vector<double> make_test_data(int N, double x_min, double x_max)
{
    std::vector<double> u(N);
    const double dx = (x_max - x_min) / (N - 1);
    for (int i = 0; i < N; ++i) {
        double x = x_min + i * dx;
        u[i] = (x < 15.01) ? -0.015 * x * (x - 15.0) : 0.0;
    }
    return u;
}

// ---- Test runner -------------------------------------------------------------

struct TestCase {
    const char* name;
    int         flux_id;
    int         N;
    double      dt;
    double      dx;
};

static bool run_test(const TestCase& tc)
{
    const auto u = make_test_data(tc.N, 0.0, 38.0);

    // Allocate results buffer (2 doubles)
    double results[2] = {0.0, 0.0};

    // Call kernel under test
    tv_cfl_kernel(u.data(), results, tc.N, tc.dt, tc.dx, tc.flux_id);

    const double kernel_tv  = results[0];
    const double kernel_cfl = results[1];

    // Reference
    const double ref_tv  = ref_total_variation(u);
    const double ref_cfl = ref_compute_cfl(u, tc.dt, tc.dx, tc.flux_id);

    // Tolerances — double FP, ~1 ULP per accumulation step
    const double tv_tol  = ref_tv  * 1e-10 + 1e-14;
    const double cfl_tol = ref_cfl * 1e-10 + 1e-14;

    const bool tv_ok  = std::abs(kernel_tv  - ref_tv)  <= tv_tol;
    const bool cfl_ok = std::abs(kernel_cfl - ref_cfl) <= cfl_tol;

    const bool pass = tv_ok && cfl_ok;

    printf("[%s] N=%-7d  TV  kernel=%.8f  ref=%.8f  err=%.2e  %s\n",
           tc.name, tc.N, kernel_tv,  ref_tv,  std::abs(kernel_tv  - ref_tv),
           tv_ok  ? "OK" : "FAIL");
    printf("[%s] N=%-7d  CFL kernel=%.8f  ref=%.8f  err=%.2e  %s\n",
           tc.name, tc.N, kernel_cfl, ref_cfl, std::abs(kernel_cfl - ref_cfl),
           cfl_ok ? "OK" : "FAIL");

    return pass;
}

int main()
{
    const double dt_burgers  = 1.0;
    const double dt_buckley  = 0.35;
    const double dt_lwr      = 0.275;
    const double dt_trilinear = 0.5;

    // dx = (38 - 0) / (N - 1)  for each N
    auto dx = [](int N){ return 38.0 / (N - 1); };

    const TestCase tests[] = {
        // name,           flux_id,           N,       dt,               dx
        {"burgers  ",  FLUX_BURGERS,          20,      dt_burgers,       dx(20)},
        {"burgers  ",  FLUX_BURGERS,          1000,    dt_burgers,       dx(1000)},
        {"burgers  ",  FLUX_BURGERS,          100000,  dt_burgers,       dx(100000)},
        {"lwr      ",  FLUX_LWR,              1000,    dt_lwr,           dx(1000)},
        {"buckley  ",  FLUX_BUCKLEY_LEV,      1000,    dt_buckley,       dx(1000)},
        {"trilinear",  FLUX_TRILINEAR,        1000,    dt_trilinear,     dx(1000)},
        {"cubic    ",  FLUX_CUBIC,            1000,    0.625,            dx(1000)},
        {"floodwave",  FLUX_FLOOD_WAVE,       1000,    0.5,              dx(1000)},
        {"log      ",  FLUX_LOG,              1000,    0.025,            dx(1000)},
    };

    int n_pass = 0;
    int n_total = (int)(sizeof(tests) / sizeof(tests[0]));

    printf("=== tv_cfl_kernel C-sim testbench ===\n\n");
    for (int i = 0; i < n_total; ++i) {
        if (run_test(tests[i])) ++n_pass;
        printf("\n");
    }

    printf("Result: %d / %d tests passed\n", n_pass, n_total);
    return (n_pass == n_total) ? 0 : 1;
}
