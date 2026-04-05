#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <omp.h>

using steady_clock_t = std::chrono::steady_clock;

struct TimingStats {
    double step_ms = 0.0;
    double cfl_ms  = 0.0;
    double tv_ms   = 0.0;
    double total_ms() const { return step_ms + cfl_ms + tv_ms; }
};

struct ScopedTimer {
    steady_clock_t ::time_point t0;
    double* accum_ms;
    explicit ScopedTimer(double* a) : t0(steady_clock_t ::now()), accum_ms(a) {}
    ~ScopedTimer() {
        auto t1 = steady_clock_t::now();
        *accum_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

inline int periodic_index(int i, int N) {
    if (i < 0)   return i + N;
    if (i >= N)  return i - N;
    return i;
}

inline double flux(double u) { return std::log(u); }

inline double flux_prime(double u) {
    if (u <= 0.0) return 0.0;
    return 1.0 / u;
}

void step_lax_friedrichs(const std::vector<double>& u, std::vector<double>& u_new, double dt, double dx) {
    int N = (int)u.size();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        int im = periodic_index(i - 1, N);
        int ip = periodic_index(i + 1, N);
        double f_plus  = flux(u[ip]);
        double f_minus = flux(u[im]);
        u_new[i] = 0.5 * (u[ip] + u[im]) - 0.5 * (dt / dx) * (f_plus - f_minus);
    }
}

void step_leapfrog(const std::vector<double>& u_old,
                   const std::vector<double>& u,
                   std::vector<double>& u_new,
                   double dt, double dx)
{
    int N = (int)u.size();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        int im = periodic_index(i - 1, N);
        int ip = periodic_index(i + 1, N);
        double f_plus  = flux(u[ip]);
        double f_minus = flux(u[im]);
        u_new[i] = u_old[i] - (dt / dx) * (f_plus - f_minus);
    }
}

double max_abs(const std::vector<double>& u) {
    double m = 0.0;
    for (double val : u) m = std::max(m, std::abs(flux_prime(val)));
    return m;
}

double compute_cfl(const std::vector<double>& u, double dt, double dx) {
    double amax = max_abs(u);
    return amax * dt / dx;
}

double total_variation(const std::vector<double>& u) {
    int N = (int)u.size();
    double tv = 0.0;
    for (int i = 0; i < N; ++i) {
        int ip = periodic_index(i + 1, N);
        tv += std::abs(u[ip] - u[i]);
    }
    return tv;
}

double initial_condition(double x) {
    return (x < 15.0) ? 1.5 : 0.5;
}

static void print_timing(const char* label, const TimingStats& s) {
    double tot = s.total_ms();
    auto pct = [&](double ms) { return (tot > 0.0) ? (100.0 * ms / tot) : 0.0; };

    std::cout << "\n=== Timing: " << label << " ===\n";
    std::cout << "step kernel : " << s.step_ms << " ms (" << pct(s.step_ms) << "%)\n";
    std::cout << "CFL check   : " << s.cfl_ms  << " ms (" << pct(s.cfl_ms)  << "%)\n";
    std::cout << "TV check    : " << s.tv_ms   << " ms (" << pct(s.tv_ms)   << "%)\n";
    std::cout << "TOTAL       : " << tot       << " ms\n";
}

int main() {
    const double x_min   = 0.0;
    const double x_max   = 38.0;
    const int    N       = 20;
    const double dx      = (x_max - x_min) / (N - 1);
    const double dt      = 1.0;
    const int    n_steps = 25;

    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = x_min + i * dx;

    TimingStats lf_stats;
    std::vector<double> u_lf(N), u_lf_new(N);
    for (int i = 0; i < N; ++i) u_lf[i] = initial_condition(x[i]);

    double t = 0.0;
    for (int n = 0; n < n_steps; ++n) {
        { ScopedTimer timer(&lf_stats.step_ms);
          step_lax_friedrichs(u_lf, u_lf_new, dt, dx);
        }
        double CFL = 0.0, TV = 0.0;
        { ScopedTimer timer(&lf_stats.cfl_ms);
          CFL = compute_cfl(u_lf, dt, dx);
        }
        { ScopedTimer timer(&lf_stats.tv_ms);
          TV = total_variation(u_lf_new);
        }
        t += dt;
        u_lf.swap(u_lf_new);
    }
    print_timing("Lax-Friedrichs", lf_stats);

    TimingStats lfrog_stats;
    std::vector<double> u_old(N), u(N), u_new(N);
    for (int i = 0; i < N; ++i) u_old[i] = initial_condition(x[i]);

    { ScopedTimer timer(&lfrog_stats.step_ms);
      step_lax_friedrichs(u_old, u, dt, dx);
    }
    t = dt;

    for (int n = 1; n < n_steps; ++n) {
        { ScopedTimer timer(&lfrog_stats.step_ms);
          step_leapfrog(u_old, u, u_new, dt, dx);
        }
        double CFL = 0.0, TV = 0.0;
        { ScopedTimer timer(&lfrog_stats.cfl_ms);
          CFL = compute_cfl(u, dt, dx);
        }
        { ScopedTimer timer(&lfrog_stats.tv_ms);
          TV = total_variation(u_new);
        }
        t += dt;
        std::cout << "[Leapfrog] step " << n+1 << ", t=" << t << ", CFL=" << CFL << ", TV=" << TV << "\n";
        u_old.swap(u);
        u.swap(u_new);
    }
    print_timing("Leapfrog (incl. LF bootstrap in step_ms)", lfrog_stats);

    return 0;
}
