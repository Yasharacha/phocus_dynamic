#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>

using steady_clock_t = std::chrono::steady_clock;

inline int periodic_index(int i, int N) {
    if (i < 0)   return i + N;
    if (i >= N)  return i - N;
    return i;
}

inline double flux(double u) {
    double up = std::max(u, 0.0);
    return up * std::sqrt(up);
}

double initial_condition(double x) {
    if (x < 15.01) return -0.015 * x * (x - 15.0) + 1.0;
    else return 1.0;
}

// ---- Serial step ----
void step_lax_friedrichs_serial(const std::vector<double>& u,
                                 std::vector<double>& u_new,
                                 double dt, double dx)
{
    int N = (int)u.size();
    for (int i = 0; i < N; ++i) {
        int im = periodic_index(i - 1, N);
        int ip = periodic_index(i + 1, N);
        u_new[i] = 0.5*(u[ip]+u[im]) - 0.5*(dt/dx)*(flux(u[ip])-flux(u[im]));
    }
}

// ---- OpenMP step ----
void step_lax_friedrichs_omp(const std::vector<double>& u,
                               std::vector<double>& u_new,
                               double dt, double dx)
{
    int N = (int)u.size();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        int im = periodic_index(i - 1, N);
        int ip = periodic_index(i + 1, N);
        u_new[i] = 0.5*(u[ip]+u[im]) - 0.5*(dt/dx)*(flux(u[ip])-flux(u[im]));
    }
}

// Run n_steps of a given step function and return total wall time in ms
template<typename StepFn>
double time_run(StepFn step_fn, int N, int n_steps, double dt, double dx) {
    const double x_min = 0.0, x_max = 38.0;
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = x_min + i * (x_max - x_min) / (N - 1);

    std::vector<double> u(N), u_new(N);
    for (int i = 0; i < N; ++i) u[i] = initial_condition(x[i]);

    auto t0 = steady_clock_t::now();
    for (int n = 0; n < n_steps; ++n) {
        step_fn(u, u_new, dt, dx);
        u.swap(u_new);
    }
    auto t1 = steady_clock_t::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    const double dt     = 1.0;
    const int    REPS   = 5;   // repeat each run to stabilise timing

    // N values to benchmark: small → large
    std::vector<int> ns = {20, 100, 500, 1000, 5000, 10000,
                           50000, 100000, 500000, 1000000};

    // timestep counts to sweep
    std::vector<int> steps_list = {10, 25, 100, 500};

    // thread counts to test
    std::vector<int> thread_counts = {1, 2, 4, 8};

    const int col_w = 10 + 14 + (int)thread_counts.size() * 26;

    for (int n_steps : steps_list) {
        std::cout << "\n=== n_steps = " << n_steps << " ===\n";

        // header
        std::cout << std::left
                  << std::setw(10) << "N"
                  << std::setw(14) << "serial(ms)";
        for (int t : thread_counts)
            std::cout << std::setw(14) << ("omp_" + std::to_string(t) + "t(ms)")
                      << std::setw(12) << ("speedup_" + std::to_string(t) + "t");
        std::cout << "\n" << std::string(col_w, '-') << "\n";

        for (int N : ns) {
            const double dx = 38.0 / (N - 1);

            // serial timing (average over REPS)
            double serial_ms = 0.0;
            for (int r = 0; r < REPS; ++r)
                serial_ms += time_run(step_lax_friedrichs_serial, N, n_steps, dt, dx);
            serial_ms /= REPS;

            std::cout << std::left << std::setw(10) << N
                      << std::setw(14) << std::fixed << std::setprecision(4) << serial_ms;

            for (int nthreads : thread_counts) {
                omp_set_num_threads(nthreads);

                double omp_ms = 0.0;
                for (int r = 0; r < REPS; ++r)
                    omp_ms += time_run(step_lax_friedrichs_omp, N, n_steps, dt, dx);
                omp_ms /= REPS;

                double speedup = serial_ms / omp_ms;
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << omp_ms
                          << std::setw(12) << std::fixed << std::setprecision(3) << speedup;
            }
            std::cout << "\n";
        }
    }

    return 0;
}
