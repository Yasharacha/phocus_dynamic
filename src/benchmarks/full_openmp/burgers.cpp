#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

using steady_clock_t = std::chrono::steady_clock;

inline int periodic_index(int i, int N) {
    if (i < 0) return i + N;
    if (i >= N) return i - N;
    return i;
}

inline double flux(double u) { return 0.5 * u * u; }

double initial_condition(double x) {
    if (x < 15.01) return -0.015 * x * (x - 15.0);
    return 0.0;
}

void step_lax_friedrichs_serial(const std::vector<double>& u, std::vector<double>& u_new, double dt, double dx) {
    const int N = static_cast<int>(u.size());
    for (int i = 0; i < N; ++i) {
        int im = periodic_index(i - 1, N);
        int ip = periodic_index(i + 1, N);
        u_new[i] = 0.5 * (u[ip] + u[im]) - 0.5 * (dt / dx) * (flux(u[ip]) - flux(u[im]));
    }
}

void step_lax_friedrichs_omp(const std::vector<double>& u, std::vector<double>& u_new, double dt, double dx) {
    const int N = static_cast<int>(u.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        int im = periodic_index(i - 1, N);
        int ip = periodic_index(i + 1, N);
        u_new[i] = 0.5 * (u[ip] + u[im]) - 0.5 * (dt / dx) * (flux(u[ip]) - flux(u[im]));
    }
}

void step_leapfrog_serial(const std::vector<double>& u_old, const std::vector<double>& u, std::vector<double>& u_new,
                          double dt, double dx) {
    const int N = static_cast<int>(u.size());
    for (int i = 0; i < N; ++i) {
        int im = periodic_index(i - 1, N);
        int ip = periodic_index(i + 1, N);
        u_new[i] = u_old[i] - (dt / dx) * (flux(u[ip]) - flux(u[im]));
    }
}

void step_leapfrog_omp(const std::vector<double>& u_old, const std::vector<double>& u, std::vector<double>& u_new,
                       double dt, double dx) {
    const int N = static_cast<int>(u.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        int im = periodic_index(i - 1, N);
        int ip = periodic_index(i + 1, N);
        u_new[i] = u_old[i] - (dt / dx) * (flux(u[ip]) - flux(u[im]));
    }
}

double compute_cfl_serial(const std::vector<double>& u, double dt, double dx) {
    double umax = 0.0;
    for (double ui : u) umax = std::max(umax, std::abs(ui));
    return umax * dt / dx;
}

double compute_cfl_omp(const std::vector<double>& u, double dt, double dx) {
    const int N = static_cast<int>(u.size());
    double umax = 0.0;
    #pragma omp parallel for reduction(max:umax) schedule(static)
    for (int i = 0; i < N; ++i) {
        umax = std::max(umax, std::abs(u[i]));
    }
    return umax * dt / dx;
}

double total_variation_serial(const std::vector<double>& u) {
    const int N = static_cast<int>(u.size());
    double tv = 0.0;
    for (int i = 0; i < N; ++i) {
        int ip = periodic_index(i + 1, N);
        tv += std::abs(u[ip] - u[i]);
    }
    return tv;
}

double total_variation_omp(const std::vector<double>& u) {
    const int N = static_cast<int>(u.size());
    double tv = 0.0;
    #pragma omp parallel for reduction(+:tv) schedule(static)
    for (int i = 0; i < N; ++i) {
        int ip = periodic_index(i + 1, N);
        tv += std::abs(u[ip] - u[i]);
    }
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

Timings run_omp_case(int N, int n_steps, double dt) {
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
            step_lax_friedrichs_omp(u, u_new, dt, dx);
            (void)compute_cfl_omp(u_new, dt, dx);
            (void)total_variation_omp(u_new);
            u.swap(u_new);
        }
        auto t1 = steady_clock_t::now();
        out.lf_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    {
        std::vector<double> u_old = u0, u(N), u_new(N);
        auto t0 = steady_clock_t::now();
        step_lax_friedrichs_omp(u_old, u, dt, dx);
        (void)compute_cfl_omp(u, dt, dx);
        (void)total_variation_omp(u);
        for (int n = 1; n < n_steps; ++n) {
            step_leapfrog_omp(u_old, u, u_new, dt, dx);
            (void)compute_cfl_omp(u_new, dt, dx);
            (void)total_variation_omp(u_new);
            u_old.swap(u);
            u.swap(u_new);
        }
        auto t1 = steady_clock_t::now();
        out.leapfrog_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return out;
}

int main() {
    // Keep dt fixed across all N/thread sweeps.
    const double dt = 1.0;
    const int n_steps = 25;
    const int reps = 3;

    std::vector<int> n_values = {20, 1000, 10000, 100000, 500000, 1000000};

    int max_threads = omp_get_max_threads();
    std::vector<int> thread_values = {1, 2, 4, 8, 16};
    thread_values.erase(
        std::remove_if(thread_values.begin(), thread_values.end(),
                       [&](int t) { return t > max_threads; }),
        thread_values.end()
    );
    if (thread_values.empty() || thread_values.back() != max_threads) {
        thread_values.push_back(max_threads);
    }
    std::sort(thread_values.begin(), thread_values.end());
    thread_values.erase(std::unique(thread_values.begin(), thread_values.end()), thread_values.end());

    std::ofstream csv("benchmark_burgers_full_openmp.csv");
    if (!csv) {
        std::cerr << "Failed to open CSV output file.\n";
        return 1;
    }
    csv << "N,threads,dt,n_steps,reps,serial_lf_ms,omp_lf_ms,lf_speedup,serial_leapfrog_ms,omp_leapfrog_ms,leapfrog_speedup\n";

    std::cout << std::left
              << std::setw(10) << "N"
              << std::setw(10) << "threads"
              << std::setw(12) << "serial_LF"
              << std::setw(12) << "omp_LF"
              << std::setw(12) << "LF_spdup"
              << std::setw(14) << "serial_Leap"
              << std::setw(12) << "omp_Leap"
              << std::setw(12) << "Leap_spdup"
              << "\n";
    std::cout << std::string(94, '-') << "\n";

    for (int N : n_values) {
        double serial_lf_sum = 0.0, serial_leap_sum = 0.0;
        for (int r = 0; r < reps; ++r) {
            Timings t = run_serial_case(N, n_steps, dt);
            serial_lf_sum += t.lf_ms;
            serial_leap_sum += t.leapfrog_ms;
        }
        double serial_lf_avg = serial_lf_sum / reps;
        double serial_leap_avg = serial_leap_sum / reps;

        for (int threads : thread_values) {
            omp_set_num_threads(threads);

            double omp_lf_sum = 0.0, omp_leap_sum = 0.0;
            for (int r = 0; r < reps; ++r) {
                Timings t = run_omp_case(N, n_steps, dt);
                omp_lf_sum += t.lf_ms;
                omp_leap_sum += t.leapfrog_ms;
            }

            double omp_lf_avg = omp_lf_sum / reps;
            double omp_leap_avg = omp_leap_sum / reps;

            double lf_speedup = (omp_lf_avg > 0.0) ? (serial_lf_avg / omp_lf_avg) : 0.0;
            double leap_speedup = (omp_leap_avg > 0.0) ? (serial_leap_avg / omp_leap_avg) : 0.0;

            csv << N << ','
                << threads << ','
                << dt << ','
                << n_steps << ','
                << reps << ','
                << serial_lf_avg << ','
                << omp_lf_avg << ','
                << lf_speedup << ','
                << serial_leap_avg << ','
                << omp_leap_avg << ','
                << leap_speedup << '\n';

            std::cout << std::left
                      << std::setw(10) << N
                      << std::setw(10) << threads
                      << std::setw(12) << std::fixed << std::setprecision(3) << serial_lf_avg
                      << std::setw(12) << std::fixed << std::setprecision(3) << omp_lf_avg
                      << std::setw(12) << std::fixed << std::setprecision(3) << lf_speedup
                      << std::setw(14) << std::fixed << std::setprecision(3) << serial_leap_avg
                      << std::setw(12) << std::fixed << std::setprecision(3) << omp_leap_avg
                      << std::setw(12) << std::fixed << std::setprecision(3) << leap_speedup
                      << "\n";
        }
    }

    std::cout << "\nWrote: benchmark_burgers_full_openmp.csv\n";
    return 0;
}
