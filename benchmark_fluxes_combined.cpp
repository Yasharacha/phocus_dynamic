#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>

using steady_clock_t = std::chrono::steady_clock;

enum class Approach {
    Serial,
    OpenMP,
    Simd
};

struct TimingBreakdown {
    double step_ms = 0.0;
    double cfl_ms = 0.0;
    double tv_ms = 0.0;

    double total_ms() const {
        return step_ms + cfl_ms + tv_ms;
    }
};

struct SchemeRunResult {
    TimingBreakdown lf;
    TimingBreakdown leapfrog;
};

struct AveragedSchemeRunResult {
    TimingBreakdown lf;
    TimingBreakdown leapfrog;
};

struct FluxConfig {
    std::string name;
    double x_min;
    double x_max;
    double dt;
    int n_steps;
};

inline int periodic_index(int i, int N) {
    if (i < 0) return i + N;
    if (i >= N) return i - N;
    return i;
}

struct ScopedTimer {
    steady_clock_t::time_point t0;
    double* accum_ms;

    explicit ScopedTimer(double* accum) : t0(steady_clock_t::now()), accum_ms(accum) {}

    ~ScopedTimer() {
        const auto t1 = steady_clock_t::now();
        *accum_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

struct BuckleyLeverettFlux {
    static double flux(double u) {
        const double u2 = u * u;
        const double a = 1.0 - u;
        const double denom = u2 + 0.25 * a * a;
        if (denom <= 1e-14) return 0.0;
        return u2 / denom;
    }

    static double flux_prime(double u) {
        const double u2 = u * u;
        const double a = 1.0 - u;
        const double denom = u2 + 0.25 * a * a;
        if (denom <= 1e-14) return 0.0;
        return 0.5 * u * (1.0 - u) / (denom * denom);
    }

    static double initial_condition(double x) {
        return 100.0 * std::sin(0.0001 * x);
    }

    static FluxConfig config() {
        return {"buckleyLeverett", 0.0, 38.0, 0.35, 30};
    }
};

struct CubicFlux {
    static double flux(double u) {
        return u * u * u;
    }

    static double flux_prime(double u) {
        return 3.0 * u * u;
    }

    static double initial_condition(double x) {
        if (x < 15.0) return 0.015 * x * (15.0 - x);
        return 0.0;
    }

    static FluxConfig config() {
        return {"cubic", 0.0, 38.0, 0.625, 25};
    }
};

struct BurgersFlux {
    static double flux(double u) {
        return 0.5 * u * u;
    }

    static double flux_prime(double u) {
        return u;
    }

    static double initial_condition(double x) {
        if (x < 15.01) return -0.015 * x * (x - 15.0);
        return 0.0;
    }

    static FluxConfig config() {
        return {"burgers", 0.0, 38.0, 1.0, 25};
    }
};

struct LwrFlux {
    static double flux(double u) {
        return u * (1.0 - u);
    }

    static double flux_prime(double u) {
        return 1.0 - 2.0 * u;
    }

    static double initial_condition(double x) {
        if (x < 15.01) return -0.015 * x * (x - 15.0);
        return 0.0;
    }

    static FluxConfig config() {
        return {"lwr", 0.0, 38.0, 0.275, 25};
    }
};

struct LogFlux {
    static double flux(double u) {
        return std::log(u);
    }

    static double flux_prime(double u) {
        if (u <= 0.0) return 0.0;
        return 1.0 / u;
    }

    static double initial_condition(double x) {
        return (x < 15.0) ? 1.5 : 0.5;
    }

    static FluxConfig config() {
        return {"log", 0.0, 38.0, 0.025, 1000};
    }
};

struct TrilinearFlux {
    static double flux(double u) {
        return -std::max(-u, 0.0) + std::max(u - 1.0, 0.0);
    }

    static double flux_prime(double) {
        // Match the standalone trilinear benchmark's fixed CFL = dt / dx policy.
        return 1.0;
    }

    static double initial_condition(double x) {
        if (x < 15.01) return -0.015 * x * (x - 15.0);
        return 0.0;
    }

    static FluxConfig config() {
        return {"trilinear", 0.0, 38.0, 0.5, 35};
    }
};

template <typename Flux>
double compute_cfl_serial(const std::vector<double>& u, double dt, double dx) {
    double max_speed = 0.0;
    for (double ui : u) {
        max_speed = std::max(max_speed, std::abs(Flux::flux_prime(ui)));
    }
    return max_speed * dt / dx;
}

template <typename Flux>
double compute_cfl_openmp(const std::vector<double>& u, double dt, double dx) {
    const int N = static_cast<int>(u.size());
    double max_speed = 0.0;
    #pragma omp parallel for reduction(max:max_speed) schedule(static)
    for (int i = 0; i < N; ++i) {
        max_speed = std::max(max_speed, std::abs(Flux::flux_prime(u[i])));
    }
    return max_speed * dt / dx;
}

template <typename Flux>
double compute_cfl_simd(const std::vector<double>& u, double dt, double dx) {
    const int N = static_cast<int>(u.size());
    double max_speed = 0.0;
    #pragma omp simd reduction(max:max_speed)
    for (int i = 0; i < N; ++i) {
        max_speed = std::max(max_speed, std::abs(Flux::flux_prime(u[i])));
    }
    return max_speed * dt / dx;
}

double total_variation_serial(const std::vector<double>& u) {
    const int N = static_cast<int>(u.size());
    if (N <= 1) return 0.0;

    double tv = 0.0;
    for (int i = 0; i < N - 1; ++i) {
        tv += std::abs(u[i + 1] - u[i]);
    }
    tv += std::abs(u[0] - u[N - 1]);
    return tv;
}

double total_variation_openmp(const std::vector<double>& u) {
    const int N = static_cast<int>(u.size());
    if (N <= 1) return 0.0;

    double tv = 0.0;
    #pragma omp parallel for reduction(+:tv) schedule(static)
    for (int i = 0; i < N - 1; ++i) {
        tv += std::abs(u[i + 1] - u[i]);
    }
    tv += std::abs(u[0] - u[N - 1]);
    return tv;
}

double total_variation_simd(const std::vector<double>& u) {
    const int N = static_cast<int>(u.size());
    if (N <= 1) return 0.0;

    double tv = 0.0;
    #pragma omp simd reduction(+:tv)
    for (int i = 0; i < N - 1; ++i) {
        tv += std::abs(u[i + 1] - u[i]);
    }
    tv += std::abs(u[0] - u[N - 1]);
    return tv;
}

template <typename Flux>
void step_lax_friedrichs_serial(const std::vector<double>& u, std::vector<double>& u_new, double dt, double dx) {
    const int N = static_cast<int>(u.size());
    for (int i = 0; i < N; ++i) {
        const int im = periodic_index(i - 1, N);
        const int ip = periodic_index(i + 1, N);
        u_new[i] = 0.5 * (u[ip] + u[im]) - 0.5 * (dt / dx) * (Flux::flux(u[ip]) - Flux::flux(u[im]));
    }
}

template <typename Flux>
void step_lax_friedrichs_openmp(const std::vector<double>& u, std::vector<double>& u_new, double dt, double dx) {
    const int N = static_cast<int>(u.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const int im = periodic_index(i - 1, N);
        const int ip = periodic_index(i + 1, N);
        u_new[i] = 0.5 * (u[ip] + u[im]) - 0.5 * (dt / dx) * (Flux::flux(u[ip]) - Flux::flux(u[im]));
    }
}

template <typename Flux>
void step_lax_friedrichs_simd(const std::vector<double>& u, std::vector<double>& u_new, double dt, double dx) {
    const int N = static_cast<int>(u.size());
    const double coeff = 0.5 * (dt / dx);

    if (N == 1) {
        u_new[0] = u[0];
        return;
    }

    u_new[0] = 0.5 * (u[1] + u[N - 1]) - coeff * (Flux::flux(u[1]) - Flux::flux(u[N - 1]));

    #pragma omp simd
    for (int i = 1; i < N - 1; ++i) {
        const double up = u[i + 1];
        const double um = u[i - 1];
        u_new[i] = 0.5 * (up + um) - coeff * (Flux::flux(up) - Flux::flux(um));
    }

    u_new[N - 1] = 0.5 * (u[0] + u[N - 2]) - coeff * (Flux::flux(u[0]) - Flux::flux(u[N - 2]));
}

template <typename Flux>
void step_leapfrog_serial(const std::vector<double>& u_old,
                          const std::vector<double>& u,
                          std::vector<double>& u_new,
                          double dt,
                          double dx) {
    const int N = static_cast<int>(u.size());
    for (int i = 0; i < N; ++i) {
        const int im = periodic_index(i - 1, N);
        const int ip = periodic_index(i + 1, N);
        u_new[i] = u_old[i] - (dt / dx) * (Flux::flux(u[ip]) - Flux::flux(u[im]));
    }
}

template <typename Flux>
void step_leapfrog_openmp(const std::vector<double>& u_old,
                          const std::vector<double>& u,
                          std::vector<double>& u_new,
                          double dt,
                          double dx) {
    const int N = static_cast<int>(u.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const int im = periodic_index(i - 1, N);
        const int ip = periodic_index(i + 1, N);
        u_new[i] = u_old[i] - (dt / dx) * (Flux::flux(u[ip]) - Flux::flux(u[im]));
    }
}

template <typename Flux>
void step_leapfrog_simd(const std::vector<double>& u_old,
                        const std::vector<double>& u,
                        std::vector<double>& u_new,
                        double dt,
                        double dx) {
    const int N = static_cast<int>(u.size());
    const double coeff = dt / dx;

    if (N == 1) {
        u_new[0] = u_old[0];
        return;
    }

    u_new[0] = u_old[0] - coeff * (Flux::flux(u[1]) - Flux::flux(u[N - 1]));

    #pragma omp simd
    for (int i = 1; i < N - 1; ++i) {
        const double up = u[i + 1];
        const double um = u[i - 1];
        u_new[i] = u_old[i] - coeff * (Flux::flux(up) - Flux::flux(um));
    }

    u_new[N - 1] = u_old[N - 1] - coeff * (Flux::flux(u[0]) - Flux::flux(u[N - 2]));
}

template <typename Flux>
TimingBreakdown run_lax_friedrichs(int N, Approach approach, double dt, int n_steps) {
    const FluxConfig cfg = Flux::config();
    const double dx = (cfg.x_max - cfg.x_min) / (N - 1);

    std::vector<double> x(N), u(N), u_new(N);
    for (int i = 0; i < N; ++i) {
        x[i] = cfg.x_min + i * dx;
        u[i] = Flux::initial_condition(x[i]);
    }

    TimingBreakdown stats;
    for (int n = 0; n < n_steps; ++n) {
        {
            ScopedTimer timer(&stats.step_ms);
            if (approach == Approach::Serial) {
                step_lax_friedrichs_serial<Flux>(u, u_new, dt, dx);
            } else if (approach == Approach::OpenMP) {
                step_lax_friedrichs_openmp<Flux>(u, u_new, dt, dx);
            } else {
                step_lax_friedrichs_simd<Flux>(u, u_new, dt, dx);
            }
        }

        {
            ScopedTimer timer(&stats.cfl_ms);
            if (approach == Approach::Serial) {
                (void)compute_cfl_serial<Flux>(u_new, dt, dx);
            } else if (approach == Approach::OpenMP) {
                (void)compute_cfl_openmp<Flux>(u_new, dt, dx);
            } else {
                (void)compute_cfl_simd<Flux>(u_new, dt, dx);
            }
        }

        {
            ScopedTimer timer(&stats.tv_ms);
            if (approach == Approach::Serial) {
                (void)total_variation_serial(u_new);
            } else if (approach == Approach::OpenMP) {
                (void)total_variation_openmp(u_new);
            } else {
                (void)total_variation_simd(u_new);
            }
        }

        u.swap(u_new);
    }

    return stats;
}

template <typename Flux>
TimingBreakdown run_leapfrog(int N, Approach approach, double dt, int n_steps) {
    const FluxConfig cfg = Flux::config();
    const double dx = (cfg.x_max - cfg.x_min) / (N - 1);

    std::vector<double> x(N), u0(N);
    for (int i = 0; i < N; ++i) {
        x[i] = cfg.x_min + i * dx;
        u0[i] = Flux::initial_condition(x[i]);
    }

    std::vector<double> u_old = u0;
    std::vector<double> u(N), u_new(N);
    TimingBreakdown stats;

    {
        ScopedTimer timer(&stats.step_ms);
        if (approach == Approach::Serial) {
            step_lax_friedrichs_serial<Flux>(u_old, u, dt, dx);
        } else if (approach == Approach::OpenMP) {
            step_lax_friedrichs_openmp<Flux>(u_old, u, dt, dx);
        } else {
            step_lax_friedrichs_simd<Flux>(u_old, u, dt, dx);
        }
    }

    {
        ScopedTimer timer(&stats.cfl_ms);
        if (approach == Approach::Serial) {
            (void)compute_cfl_serial<Flux>(u, dt, dx);
        } else if (approach == Approach::OpenMP) {
            (void)compute_cfl_openmp<Flux>(u, dt, dx);
        } else {
            (void)compute_cfl_simd<Flux>(u, dt, dx);
        }
    }

    {
        ScopedTimer timer(&stats.tv_ms);
        if (approach == Approach::Serial) {
            (void)total_variation_serial(u);
        } else if (approach == Approach::OpenMP) {
            (void)total_variation_openmp(u);
        } else {
            (void)total_variation_simd(u);
        }
    }

    for (int n = 1; n < n_steps; ++n) {
        {
            ScopedTimer timer(&stats.step_ms);
            if (approach == Approach::Serial) {
                step_leapfrog_serial<Flux>(u_old, u, u_new, dt, dx);
            } else if (approach == Approach::OpenMP) {
                step_leapfrog_openmp<Flux>(u_old, u, u_new, dt, dx);
            } else {
                step_leapfrog_simd<Flux>(u_old, u, u_new, dt, dx);
            }
        }

        {
            ScopedTimer timer(&stats.cfl_ms);
            if (approach == Approach::Serial) {
                (void)compute_cfl_serial<Flux>(u_new, dt, dx);
            } else if (approach == Approach::OpenMP) {
                (void)compute_cfl_openmp<Flux>(u_new, dt, dx);
            } else {
                (void)compute_cfl_simd<Flux>(u_new, dt, dx);
            }
        }

        {
            ScopedTimer timer(&stats.tv_ms);
            if (approach == Approach::Serial) {
                (void)total_variation_serial(u_new);
            } else if (approach == Approach::OpenMP) {
                (void)total_variation_openmp(u_new);
            } else {
                (void)total_variation_simd(u_new);
            }
        }

        u_old.swap(u);
        u.swap(u_new);
    }

    return stats;
}

TimingBreakdown average_timing(const std::vector<TimingBreakdown>& samples) {
    TimingBreakdown avg;
    for (const TimingBreakdown& sample : samples) {
        avg.step_ms += sample.step_ms;
        avg.cfl_ms += sample.cfl_ms;
        avg.tv_ms += sample.tv_ms;
    }

    const double denom = static_cast<double>(samples.size());
    avg.step_ms /= denom;
    avg.cfl_ms /= denom;
    avg.tv_ms /= denom;
    return avg;
}

template <typename Flux>
AveragedSchemeRunResult run_average_for_approach(int N, Approach approach, int reps) {
    const FluxConfig cfg = Flux::config();

    std::vector<TimingBreakdown> lf_samples;
    std::vector<TimingBreakdown> leapfrog_samples;
    lf_samples.reserve(reps);
    leapfrog_samples.reserve(reps);

    for (int r = 0; r < reps; ++r) {
        lf_samples.push_back(run_lax_friedrichs<Flux>(N, approach, cfg.dt, cfg.n_steps));
        leapfrog_samples.push_back(run_leapfrog<Flux>(N, approach, cfg.dt, cfg.n_steps));
    }

    AveragedSchemeRunResult out;
    out.lf = average_timing(lf_samples);
    out.leapfrog = average_timing(leapfrog_samples);
    return out;
}

double speedup_percent(double serial_ms, double optimized_ms) {
    if (serial_ms <= 0.0) return 0.0;
    return 100.0 * (serial_ms - optimized_ms) / serial_ms;
}

double runtime_share_percent(double part_ms, double total_ms) {
    if (total_ms <= 0.0) return 0.0;
    return 100.0 * part_ms / total_ms;
}

void write_csv_row(std::ofstream& csv,
                   const std::string& flux_name,
                   const char* scheme,
                   int N,
                   int omp_threads,
                   double dt,
                   int n_steps,
                   int reps,
                   const TimingBreakdown& serial_stats,
                   const TimingBreakdown& omp_stats,
                   const TimingBreakdown& simd_stats) {
    const double serial_total = serial_stats.total_ms();
    const double omp_total = omp_stats.total_ms();
    const double simd_total = simd_stats.total_ms();

    csv << flux_name << ','
        << scheme << ','
        << N << ','
        << omp_threads << ','
        << dt << ','
        << n_steps << ','
        << reps << ','
        << serial_total << ','
        << omp_total << ','
        << simd_total << ','
        << speedup_percent(serial_total, omp_total) << ','
        << speedup_percent(serial_total, simd_total) << ','
        << runtime_share_percent(serial_stats.cfl_ms, serial_total) << ','
        << runtime_share_percent(serial_stats.tv_ms, serial_total) << ','
        << runtime_share_percent(omp_stats.cfl_ms, omp_total) << ','
        << runtime_share_percent(omp_stats.tv_ms, omp_total) << ','
        << runtime_share_percent(simd_stats.cfl_ms, simd_total) << ','
        << runtime_share_percent(simd_stats.tv_ms, simd_total)
        << '\n';
}

template <typename Flux>
void benchmark_flux(std::ofstream& csv,
                    const std::vector<int>& n_values,
                    const std::vector<int>& thread_values,
                    int reps) {
    const FluxConfig cfg = Flux::config();

    std::cout << "\nFlux: " << cfg.name << "\n";
    for (int N : n_values) {
        const AveragedSchemeRunResult serial = run_average_for_approach<Flux>(N, Approach::Serial, reps);
        const AveragedSchemeRunResult simd = run_average_for_approach<Flux>(N, Approach::Simd, reps);

        for (int threads : thread_values) {
            omp_set_num_threads(threads);
            const AveragedSchemeRunResult openmp = run_average_for_approach<Flux>(N, Approach::OpenMP, reps);

            write_csv_row(csv, cfg.name, "lax_friedrichs", N, threads, cfg.dt, cfg.n_steps, reps,
                          serial.lf, openmp.lf, simd.lf);
            write_csv_row(csv, cfg.name, "leapfrog", N, threads, cfg.dt, cfg.n_steps, reps,
                          serial.leapfrog, openmp.leapfrog, simd.leapfrog);

            std::cout << "  N=" << std::setw(8) << N
                      << " threads=" << std::setw(3) << threads
                      << " serial=" << std::setw(10) << std::fixed << std::setprecision(3) << serial.lf.total_ms()
                      << " omp=" << std::setw(10) << openmp.lf.total_ms()
                      << " simd=" << std::setw(10) << simd.lf.total_ms()
                      << "\n";
        }
    }
}

std::optional<std::string> parse_flux_filter(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--flux" && i + 1 < argc) {
            return std::string(argv[++i]);
        }
    }
    return std::nullopt;
}

std::string parse_output_path(int argc, char** argv, const std::string& default_path) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            return std::string(argv[++i]);
        }
    }
    return default_path;
}

std::vector<int> parse_n_values(int argc, char** argv, const std::vector<int>& default_values) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--n-values" && i + 1 < argc) {
            std::vector<int> values;
            std::stringstream ss(argv[++i]);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    values.push_back(std::stoi(item));
                }
            }
            if (!values.empty()) {
                return values;
            }
        }
    }
    return default_values;
}

int main(int argc, char** argv) {
    const int reps = 3;
    const std::vector<int> default_n_values = {20, 1000, 10000, 100000, 500000, 1000000};
    const std::optional<std::string> flux_filter = parse_flux_filter(argc, argv);
    const std::string output_path = parse_output_path(argc, argv, "benchmark_fluxes_combined.csv");
    const std::vector<int> n_values = parse_n_values(argc, argv, default_n_values);

    int max_threads = omp_get_max_threads();
    std::vector<int> thread_values = {1, 2, 4, 8, 16};
    thread_values.erase(
        std::remove_if(thread_values.begin(), thread_values.end(), [&](int t) { return t > max_threads; }),
        thread_values.end()
    );
    if (thread_values.empty() || thread_values.back() != max_threads) {
        thread_values.push_back(max_threads);
    }
    std::sort(thread_values.begin(), thread_values.end());
    thread_values.erase(std::unique(thread_values.begin(), thread_values.end()), thread_values.end());

    std::ofstream csv(output_path);
    if (!csv) {
        std::cerr << "Failed to open " << output_path << " for writing.\n";
        return 1;
    }

    csv << "flux,scheme,N,omp_threads,dt,n_steps,reps,"
           "serial_total_ms,omp_total_ms,simd_total_ms,"
           "omp_speedup_pct,simd_speedup_pct,"
           "serial_cfl_pct,serial_tv_pct,"
           "omp_cfl_pct,omp_tv_pct,"
           "simd_cfl_pct,simd_tv_pct\n";

    bool ran_any_flux = false;
    if (!flux_filter || *flux_filter == "buckleyLeverett") {
        benchmark_flux<BuckleyLeverettFlux>(csv, n_values, thread_values, reps);
        ran_any_flux = true;
    }
    if (!flux_filter || *flux_filter == "cubic") {
        benchmark_flux<CubicFlux>(csv, n_values, thread_values, reps);
        ran_any_flux = true;
    }
    if (!flux_filter || *flux_filter == "burgers") {
        benchmark_flux<BurgersFlux>(csv, n_values, thread_values, reps);
        ran_any_flux = true;
    }
    if (!flux_filter || *flux_filter == "lwr") {
        benchmark_flux<LwrFlux>(csv, n_values, thread_values, reps);
        ran_any_flux = true;
    }
    if (!flux_filter || *flux_filter == "log") {
        benchmark_flux<LogFlux>(csv, n_values, thread_values, reps);
        ran_any_flux = true;
    }
    if (!flux_filter || *flux_filter == "trilinear") {
        benchmark_flux<TrilinearFlux>(csv, n_values, thread_values, reps);
        ran_any_flux = true;
    }

    if (!ran_any_flux) {
        std::cerr << "Unknown flux selection. Use --flux buckleyLeverett, cubic, burgers, lwr, log, or trilinear.\n";
        return 1;
    }

    std::cout << "\nWrote: " << output_path << "\n";
    return 0;
}
