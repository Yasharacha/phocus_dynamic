#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using steady_clock_t = std::chrono::steady_clock;

inline double flux(double u) { return u * (1.0 - u); }

double initial_condition(double x) {
    if (x < 15.01) return -0.015 * x * (x - 15.0);
    return 0.0;
}

void step_lax_friedrichs_serial(const std::vector<double>& u, std::vector<double>& u_new, double dt, double dx) {
    const int N = static_cast<int>(u.size());
    for (int i = 0; i < N; ++i) {
        const int im = (i == 0) ? (N - 1) : (i - 1);
        const int ip = (i == N - 1) ? 0 : (i + 1);
        u_new[i] = 0.5 * (u[ip] + u[im]) - 0.5 * (dt / dx) * (flux(u[ip]) - flux(u[im]));
    }
}

void step_lax_friedrichs_simd(const std::vector<double>& u, std::vector<double>& u_new, double dt, double dx) {
    const int N = static_cast<int>(u.size());
    const double coeff = 0.5 * (dt / dx);

    if (N == 1) {
        u_new[0] = u[0];
        return;
    }

    u_new[0] = 0.5 * (u[1] + u[N - 1]) - coeff * (flux(u[1]) - flux(u[N - 1]));

    #pragma omp simd
    for (int i = 1; i < N - 1; ++i) {
        const double up = u[i + 1];
        const double um = u[i - 1];
        u_new[i] = 0.5 * (up + um) - coeff * (flux(up) - flux(um));
    }

    u_new[N - 1] = 0.5 * (u[0] + u[N - 2]) - coeff * (flux(u[0]) - flux(u[N - 2]));
}

void step_leapfrog_serial(const std::vector<double>& u_old, const std::vector<double>& u, std::vector<double>& u_new,
                          double dt, double dx) {
    const int N = static_cast<int>(u.size());
    for (int i = 0; i < N; ++i) {
        const int im = (i == 0) ? (N - 1) : (i - 1);
        const int ip = (i == N - 1) ? 0 : (i + 1);
        u_new[i] = u_old[i] - (dt / dx) * (flux(u[ip]) - flux(u[im]));
    }
}

void step_leapfrog_simd(const std::vector<double>& u_old, const std::vector<double>& u, std::vector<double>& u_new,
                        double dt, double dx) {
    const int N = static_cast<int>(u.size());
    const double coeff = dt / dx;

    if (N == 1) {
        u_new[0] = u_old[0];
        return;
    }

    u_new[0] = u_old[0] - coeff * (flux(u[1]) - flux(u[N - 1]));

    #pragma omp simd
    for (int i = 1; i < N - 1; ++i) {
        const double up = u[i + 1];
        const double um = u[i - 1];
        u_new[i] = u_old[i] - coeff * (flux(up) - flux(um));
    }

    u_new[N - 1] = u_old[N - 1] - coeff * (flux(u[0]) - flux(u[N - 2]));
}

double compute_cfl_serial(const std::vector<double>& u, double dt, double dx) {
    double umax = 0.0;
    for (double ui : u) umax = std::max(umax, std::abs(1.0 - 2.0 * ui));
    return umax * dt / dx;
}

double compute_cfl_simd(const std::vector<double>& u, double dt, double dx) {
    const int N = static_cast<int>(u.size());
    double umax = 0.0;

    #pragma omp simd reduction(max:umax)
    for (int i = 0; i < N; ++i) {
        umax = std::max(umax, std::abs(1.0 - 2.0 * u[i]));
    }

    return umax * dt / dx;
}

double total_variation_serial(const std::vector<double>& u) {
    const int N = static_cast<int>(u.size());
    double tv = 0.0;
    for (int i = 0; i < N; ++i) {
        const int ip = (i == N - 1) ? 0 : (i + 1);
        tv += std::abs(u[ip] - u[i]);
    }
    return tv;
}

double total_variation_simd(const std::vector<double>& u) {
    const int N = static_cast<int>(u.size());
    if (N == 0) return 0.0;
    if (N == 1) return 0.0;

    double tv = 0.0;

    #pragma omp simd reduction(+:tv)
    for (int i = 0; i < N - 1; ++i) {
        tv += std::abs(u[i + 1] - u[i]);
    }

    tv += std::abs(u[0] - u[N - 1]);
    return tv;
}

struct Timings {
    double lf_ms = 0.0;
    double leapfrog_ms = 0.0;
};

Timings run_serial_case(int N, int n_steps, double dt) {
    const double x_min = 0.0, x_max = 38.0;
    const double dx = (x_max - x_min) / (N - 1);

    std::vector<double> x(N), u0(N);
    for (int i = 0; i < N; ++i) {
        x[i] = x_min + i * dx;
        u0[i] = initial_condition(x[i]);
    }

    Timings out;

    {
        std::vector<double> u = u0, u_new(N);
        auto t0 = steady_clock_t::now();
        for (int n = 0; n < n_steps; ++n) {
            step_lax_friedrichs_serial(u, u_new, dt, dx);
            (void)compute_cfl_serial(u_new, dt, dx);
            (void)total_variation_serial(u_new);
            u.swap(u_new);
        }
        auto t1 = steady_clock_t::now();
        out.lf_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    {
        std::vector<double> u_old = u0, u(N), u_new(N);
        auto t0 = steady_clock_t::now();
        step_lax_friedrichs_serial(u_old, u, dt, dx);
        (void)compute_cfl_serial(u, dt, dx);
        (void)total_variation_serial(u);
        for (int n = 1; n < n_steps; ++n) {
            step_leapfrog_serial(u_old, u, u_new, dt, dx);
            (void)compute_cfl_serial(u_new, dt, dx);
            (void)total_variation_serial(u_new);
            u_old.swap(u);
            u.swap(u_new);
        }
        auto t1 = steady_clock_t::now();
        out.leapfrog_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return out;
}

Timings run_simd_case(int N, int n_steps, double dt) {
    const double x_min = 0.0, x_max = 38.0;
    const double dx = (x_max - x_min) / (N - 1);

    std::vector<double> x(N), u0(N);
    for (int i = 0; i < N; ++i) {
        x[i] = x_min + i * dx;
        u0[i] = initial_condition(x[i]);
    }

    Timings out;

    {
        std::vector<double> u = u0, u_new(N);
        auto t0 = steady_clock_t::now();
        for (int n = 0; n < n_steps; ++n) {
            step_lax_friedrichs_simd(u, u_new, dt, dx);
            (void)compute_cfl_simd(u_new, dt, dx);
            (void)total_variation_simd(u_new);
            u.swap(u_new);
        }
        auto t1 = steady_clock_t::now();
        out.lf_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    {
        std::vector<double> u_old = u0, u(N), u_new(N);
        auto t0 = steady_clock_t::now();
        step_lax_friedrichs_simd(u_old, u, dt, dx);
        (void)compute_cfl_simd(u, dt, dx);
        (void)total_variation_simd(u);
        for (int n = 1; n < n_steps; ++n) {
            step_leapfrog_simd(u_old, u, u_new, dt, dx);
            (void)compute_cfl_simd(u_new, dt, dx);
            (void)total_variation_simd(u_new);
            u_old.swap(u);
            u.swap(u_new);
        }
        auto t1 = steady_clock_t::now();
        out.leapfrog_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return out;
}

int main() {
    const double dt = 0.275;
    const int n_steps = 25;
    const int reps = 3;

    std::vector<int> n_values = {20, 1000, 10000, 100000, 500000, 1000000};

    std::ofstream csv("benchmark_lwr_full_simd.csv");
    if (!csv) {
        std::cerr << "Failed to open CSV output file.\n";
        return 1;
    }

    csv << "N,dt,n_steps,reps,serial_lf_ms,simd_lf_ms,lf_speedup,serial_leapfrog_ms,simd_leapfrog_ms,leapfrog_speedup\n";

    std::cout << std::left
              << std::setw(10) << "N"
              << std::setw(12) << "serial_LF"
              << std::setw(12) << "simd_LF"
              << std::setw(12) << "LF_spdup"
              << std::setw(14) << "serial_Leap"
              << std::setw(12) << "simd_Leap"
              << std::setw(12) << "Leap_spdup"
              << "\n";
    std::cout << std::string(82, '-') << "\n";

    for (int N : n_values) {
        double serial_lf_sum = 0.0, serial_leap_sum = 0.0;
        double simd_lf_sum = 0.0, simd_leap_sum = 0.0;

        for (int r = 0; r < reps; ++r) {
            Timings ts = run_serial_case(N, n_steps, dt);
            serial_lf_sum += ts.lf_ms;
            serial_leap_sum += ts.leapfrog_ms;

            Timings tv = run_simd_case(N, n_steps, dt);
            simd_lf_sum += tv.lf_ms;
            simd_leap_sum += tv.leapfrog_ms;
        }

        const double serial_lf_avg = serial_lf_sum / reps;
        const double serial_leap_avg = serial_leap_sum / reps;
        const double simd_lf_avg = simd_lf_sum / reps;
        const double simd_leap_avg = simd_leap_sum / reps;

        const double lf_speedup = (simd_lf_avg > 0.0) ? (serial_lf_avg / simd_lf_avg) : 0.0;
        const double leap_speedup = (simd_leap_avg > 0.0) ? (serial_leap_avg / simd_leap_avg) : 0.0;

        csv << N << ','
            << dt << ','
            << n_steps << ','
            << reps << ','
            << serial_lf_avg << ','
            << simd_lf_avg << ','
            << lf_speedup << ','
            << serial_leap_avg << ','
            << simd_leap_avg << ','
            << leap_speedup << '\n';

        std::cout << std::left
                  << std::setw(10) << N
                  << std::setw(12) << std::fixed << std::setprecision(3) << serial_lf_avg
                  << std::setw(12) << std::fixed << std::setprecision(3) << simd_lf_avg
                  << std::setw(12) << std::fixed << std::setprecision(3) << lf_speedup
                  << std::setw(14) << std::fixed << std::setprecision(3) << serial_leap_avg
                  << std::setw(12) << std::fixed << std::setprecision(3) << simd_leap_avg
                  << std::setw(12) << std::fixed << std::setprecision(3) << leap_speedup
                  << "\n";
    }

    std::cout << "\nWrote: benchmark_lwr_full_simd.csv\n";
    return 0;
}
