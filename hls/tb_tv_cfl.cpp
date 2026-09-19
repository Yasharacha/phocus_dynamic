// tb_tv_cfl.cpp
// C-simulation testbench for tv_cfl_kernel.
//
// Build and run with Vitis HLS:
//   vitis_hls -f run_csim.tcl
//
// Or compile standalone (no HLS headers needed for C-sim):
//   g++ -O2 -o tb_tv_cfl tb_tv_cfl.cpp tv_cfl_kernel.cpp -I. && ./tb_tv_cfl

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tv_cfl_kernel.hpp"

// ---- CPU reference implementations (matches combined.cpp exactly) ----------

static double ref_total_variation(const std::vector<real_t>& u)
{
    const int N = (int)u.size();
    if (N <= 1) return 0.0;
    double tv = 0.0;
    for (int i = 0; i < N - 1; ++i)
        tv += std::abs(u[i + 1] - u[i]);
    tv += std::abs(u[0] - u[N - 1]);   // periodic wrap
    return tv;
}

static double ref_compute_cfl(const std::vector<real_t>& u,
                               double dt, double dx, int flux_id)
{
    double max_speed = 0.0;
    for (real_t ui : u)
        max_speed = std::max(max_speed, (double)flux_prime_abs(ui, flux_id));
    return max_speed * dt / dx;
}

// ---- Helper: generate Burgers-like initial condition -------------------------

static std::vector<real_t> make_test_data(int N, double x_min, double x_max)
{
    std::vector<real_t> u(N);
    const double dx = (x_max - x_min) / (N - 1);
    for (int i = 0; i < N; ++i) {
        double x = x_min + i * dx;
        u[i] = (real_t)((x < 15.01) ? -0.015 * x * (x - 15.0) : 0.0);
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

    // Allocate results buffer (2 real_t)
    real_t results[2] = {0, 0};

    // Call kernel under test
    // HLS co-sim copies `depth` elements (m_axi depth=131072 in the kernel) out of the
    // pointer, so the buffer handed to the kernel must be at least that large.
    std::vector<real_t> u_padded(131072, 0);
    std::copy(u.begin(), u.end(), u_padded.begin());
    // tv_prev < 0 disables the TV-increase test here; the flag tests below cover it.
    const int status = tv_cfl_kernel(u_padded.data(), results, tc.N,
                                     (real_t)tc.dt, (real_t)tc.dx, tc.flux_id,
                                     (real_t)-1, (real_t)1e-5);

    const double kernel_tv  = results[0];
    const double kernel_cfl = results[1];

    // Reference
    const double ref_tv  = ref_total_variation(u);
    const double ref_cfl = ref_compute_cfl(u, tc.dt, tc.dx, tc.flux_id);

    // Tolerances - reference is computed in double from the same real_t inputs.
    // float: N-term sums accumulate rounding (16 partials keep it small).
#ifdef TV_CFL_USE_FLOAT
    const double rel = 1e-4,  abs_tol = 1e-6;
#else
    const double rel = 1e-10, abs_tol = 1e-14;
#endif
    const double tv_tol  = ref_tv  * rel + abs_tol;
    const double cfl_tol = ref_cfl * rel + abs_tol;

    const bool tv_ok  = std::abs(kernel_tv  - ref_tv)  <= tv_tol;
    const bool cfl_ok = std::abs(kernel_cfl - ref_cfl) <= cfl_tol;

    // Violation flag: bit0 must equal (CFL > 1); bit1 must be clear (test disabled).
    const int expected_status = (ref_cfl > 1.0) ? 1 : 0;
    const bool flag_ok = (status == expected_status);

    const bool pass = tv_ok && cfl_ok && flag_ok;

    printf("[%s] N=%-7d  TV  kernel=%.8f  ref=%.8f  err=%.2e  %s\n",
           tc.name, tc.N, kernel_tv,  ref_tv,  std::abs(kernel_tv  - ref_tv),
           tv_ok  ? "OK" : "FAIL");
    printf("[%s] N=%-7d  CFL kernel=%.8f  ref=%.8f  err=%.2e  %s\n",
           tc.name, tc.N, kernel_cfl, ref_cfl, std::abs(kernel_cfl - ref_cfl),
           cfl_ok ? "OK" : "FAIL");
    printf("[%s] N=%-7d  flag kernel=%d  expected=%d  %s\n",
           tc.name, tc.N, status, expected_status, flag_ok ? "OK" : "FAIL");

    return pass;
}

// TV-increase test: same data, different tv_prev, expected bit1 known from the reference.
static bool run_tv_flag_test(const char* label, double tv_prev_scale, int expect_bit1)
{
    const int N = 1000;
    const auto u = make_test_data(N, 0.0, 38.0);
    const double dx_ = 38.0 / (N - 1);
    real_t results[2] = {0, 0};
    std::vector<real_t> u_padded(131072, 0);
    std::copy(u.begin(), u.end(), u_padded.begin());
    const double tv_prev = ref_total_variation(u) * tv_prev_scale;
    // dt = 0.01 keeps CFL well below 1 so only bit1 can be set.
    const int status = tv_cfl_kernel(u_padded.data(), results, N, (real_t)0.01,
                                     (real_t)dx_, FLUX_BURGERS, (real_t)tv_prev, (real_t)1e-5);
    const bool ok = (status == expect_bit1 * 2);
    printf("[%s] tv_prev=%.6f  flag kernel=%d  expected=%d  %s\n",
           label, tv_prev, status, expect_bit1 * 2, ok ? "OK" : "FAIL");
    return ok;
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
#ifndef __RTL_SIMULATION__   // skip the large case in RTL co-sim (too slow)
        {"burgers  ",  FLUX_BURGERS,          100000,  dt_burgers,       dx(100000)},
#endif
        {"lwr      ",  FLUX_LWR,              1000,    dt_lwr,           dx(1000)},
        {"buckley  ",  FLUX_BUCKLEY_LEV,      1000,    dt_buckley,       dx(1000)},
        {"trilinear",  FLUX_TRILINEAR,        1000,    dt_trilinear,     dx(1000)},
        {"cubic    ",  FLUX_CUBIC,            1000,    0.625,            dx(1000)},
        {"floodwave",  FLUX_FLOOD_WAVE,       1000,    0.5,              dx(1000)},
        {"log      ",  FLUX_LOG,              1000,    0.025,            dx(1000)},
        {"burg-cfl<1", FLUX_BURGERS,          1000,    0.01,             dx(1000)},
    };

    int n_pass = 0;
    int n_total = (int)(sizeof(tests) / sizeof(tests[0]));

    printf("=== tv_cfl_kernel C-sim testbench ===\n\n");
    for (int i = 0; i < n_total; ++i) {
        if (run_test(tests[i])) ++n_pass;
        printf("\n");
    }

    printf("--- TV-increase flag tests ---\n");
    n_total += 3;
    if (run_tv_flag_test("tv_prev = TV/2  ", 0.5, 1)) ++n_pass;   // TV doubled: violation
    if (run_tv_flag_test("tv_prev = TV    ", 1.0, 0)) ++n_pass;   // unchanged: within tolerance
    if (run_tv_flag_test("tv_prev = 2*TV  ", 2.0, 0)) ++n_pass;   // TV fell: fine
    printf("\n");

    printf("Result: %d / %d tests passed\n", n_pass, n_total);
    return (n_pass == n_total) ? 0 : 1;
}
