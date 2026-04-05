#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <omp.h>

using steady_clock_t = std::chrono::steady_clock;

inline int periodic_index(int i, int N) {
    if (i < 0)   return i + N;
    if (i >= N)  return i - N;
    return i;
}

inline double flux(double u) {
    const double u2 = u * u;
    const double a = 1.0 - u;
    const double denom = u2 + 0.25 * a * a;
    if (denom <= 1e-14) return 0.0;
    return u2 / denom;
}

inline double flux_prime(double u) {
    const double u2 = u * u;
    const double a = 1.0 - u;
    const double denom = u2 + 0.25 * a * a;
    if (denom <= 1e-14) return 0.0;
    return 0.5 * u * (1.0 - u) / (denom * denom);
}

double initial_condition(double x) {
    return 100.0 * std::sin(0.0001 * x);
}

void step_serial(const std::vector<double>& u,
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

void step_omp(const std::vector<double>& u,
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

double compute_cfl(const std::vector<double>& u, double dt, double dx) {
    double amax = 0.0;
    for (double val : u) amax = std::max(amax, std::abs(flux_prime(val)));
    return amax * dt / dx;
}

double total_variation(const std::vector<double>& u) {
    int N = (int)u.size();
    double tv = 0.0;
    for (int i = 0; i < N; ++i)
        tv += std::abs(u[periodic_index(i+1,N)] - u[i]);
    return tv;
}

double run_serial(int N, int n_steps, double dt, double dx) {
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = i * dx;
    std::vector<double> u(N), u_new(N);
    for (int i = 0; i < N; ++i) u[i] = initial_condition(x[i]);

    volatile double sink = 0.0;

    auto t0 = steady_clock_t::now();
    for (int n = 0; n < n_steps; ++n) {
        step_serial(u, u_new, dt, dx);
        sink += compute_cfl(u_new, dt, dx);
        sink += total_variation(u_new);
        u.swap(u_new);
    }
    auto t1 = steady_clock_t::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

template<typename StepFn>
double run_pipeline(StepFn step_fn, int N, int n_steps, double dt, double dx) {
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = i * dx;

    std::vector<double> pipe_buf(N);

    std::mutex              mtx;
    std::condition_variable cv;
    bool buffer_ready    = false;
    bool buffer_consumed = true;
    bool all_done        = false;

    volatile double sink = 0.0;

    std::thread consumer([&]() {
        for (int n = 0; n < n_steps; ++n) {
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&]{ return buffer_ready || all_done; });
                if (all_done) return;
                buffer_ready = false;
            }
            sink += compute_cfl(pipe_buf, dt, dx);
            sink += total_variation(pipe_buf);
            {
                std::lock_guard<std::mutex> lk(mtx);
                buffer_consumed = true;
            }
            cv.notify_one();
        }
    });

    std::vector<double> u(N), u_new(N);
    for (int i = 0; i < N; ++i) u[i] = initial_condition(x[i]);

    auto t0 = steady_clock_t::now();
    for (int n = 0; n < n_steps; ++n) {
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&]{ return buffer_consumed; });
            buffer_consumed = false;
        }

        step_fn(u, u_new, dt, dx);

        {
            std::lock_guard<std::mutex> lk(mtx);
            pipe_buf = u_new;
            buffer_ready = true;
        }
        cv.notify_one();

        u.swap(u_new);
    }
    auto t1 = steady_clock_t::now();

    consumer.join();
    (void)sink;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    const double dt   = 1.0;
    const int    REPS = 5;

    std::vector<int> ns         = {20, 100, 500, 1000, 5000, 10000,
                                    50000, 100000, 500000, 1000000};
    std::vector<int> steps_list = {10, 25, 100, 500};
    std::vector<int> omp_threads = {2, 4, 8};

    const int CW = 14;
    const int SW = 10;

    for (int n_steps : steps_list) {
        std::cout << "\n=== n_steps = " << n_steps << " ===\n";

        std::cout << std::left << std::setw(10) << "N"
                  << std::setw(CW) << "serial(ms)"
                  << std::setw(CW) << "pipe(ms)"
                  << std::setw(SW) << "spdup";
        for (int t : omp_threads)
            std::cout << std::setw(CW) << ("p+omp" + std::to_string(t) + "t(ms)")
                      << std::setw(SW) << ("spdup_" + std::to_string(t) + "t");
        std::cout << "\n"
                  << std::string(10 + CW + CW + SW + (int)omp_threads.size()*(CW+SW), '-')
                  << "\n";

        for (int N : ns) {
            const double dx = 38.0 / (N - 1);

            double serial_ms = 0.0;
            for (int r = 0; r < REPS; ++r)
                serial_ms += run_serial(N, n_steps, dt, dx);
            serial_ms /= REPS;

            double pipe_ms = 0.0;
            for (int r = 0; r < REPS; ++r)
                pipe_ms += run_pipeline(step_serial, N, n_steps, dt, dx);
            pipe_ms /= REPS;

            std::cout << std::left
                      << std::setw(10) << N
                      << std::setw(CW) << std::fixed << std::setprecision(4) << serial_ms
                      << std::setw(CW) << std::fixed << std::setprecision(4) << pipe_ms
                      << std::setw(SW) << std::fixed << std::setprecision(3) << (serial_ms / pipe_ms);

            for (int nthreads : omp_threads) {
                omp_set_num_threads(nthreads);

                double pomp_ms = 0.0;
                for (int r = 0; r < REPS; ++r)
                    pomp_ms += run_pipeline(step_omp, N, n_steps, dt, dx);
                pomp_ms /= REPS;

                std::cout << std::setw(CW) << std::fixed << std::setprecision(4) << pomp_ms
                          << std::setw(SW) << std::fixed << std::setprecision(3) << (serial_ms / pomp_ms);
            }
            std::cout << "\n";
        }
    }

    return 0;
}
