#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

const double PI = 3.14159265358979323846;

// Flux for Burgers: f(u) = 0.5 u^2
inline double flux(double u) {
    return 0.5 * u * u;
}

// Initial condition: e.g., sine wave or square pulse
double initial_condition(double x) {
    // Example: sine wave
    return std::sin(2.0 * PI * x);
    // or a square pulse:
    // if (0.25 <= x && x <= 0.75) return 1.0;
    // return 0.0;
}

// Periodic index helper
inline int periodic_index(int i, int N) {
    if (i < 0)   return i + N;
    if (i >= N)  return i - N;
    return i;
}

// Compute max |u|
double max_abs(const std::vector<double>& u) {
    double m = 0.0;
    for (double val : u)
        m = std::max(m, std::abs(val));
    return m;
}

// Total variation TV(u) = sum_i |u_{i+1} - u_i|
double total_variation(const std::vector<double>& u) {
    int N = (int)u.size();
    double tv = 0.0;
    for (int i = 0; i < N; ++i) {
        int ip = periodic_index(i + 1, N);
        tv += std::abs(u[ip] - u[i]);
    }
    return tv;
}

// One Lax–Friedrichs step
void step_lax_friedrichs(const std::vector<double>& u,
                         std::vector<double>& u_new,
                         double dt, double dx)
{
    int N = (int)u.size();
    for (int i = 0; i < N; ++i) {
        int im = periodic_index(i - 1, N);
        int ip = periodic_index(i + 1, N);

        double f_plus  = flux(u[ip]);
        double f_minus = flux(u[im]);

        u_new[i] = 0.5 * (u[ip] + u[im])
                 - 0.5 * (dt / dx) * (f_plus - f_minus);
    }
}

// One Leapfrog step (needs u_old and u at current time)
void step_leapfrog(const std::vector<double>& u_old,
                   const std::vector<double>& u,
                   std::vector<double>& u_new,
                   double dt, double dx)
{
    int N = (int)u.size();
    for (int i = 0; i < N; ++i) {
        int im = periodic_index(i - 1, N);
        int ip = periodic_index(i + 1, N);

        double f_plus  = flux(u[ip]);
        double f_minus = flux(u[im]);

        u_new[i] = u_old[i] - (dt / dx) * (f_plus - f_minus);
    }
}

int main() {
    // ----- Parameters -----
    const int    N           = 200;    // number of spatial points
    const double L           = 1.0;    // domain length [0,L]
    const double dx          = L / (N - 1);
    const double CFL_target  = 0.5;
    const double t_final     = 0.2;
    const bool   use_lf      = true;   // toggle: true=LF, false=Leapfrog

    // ----- Grid -----
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = i * dx;

    // ----- State -----
    std::vector<double> u(N), u_new(N);
    for (int i = 0; i < N; ++i)
        u[i] = initial_condition(x[i]);

    double t = 0.0;

    if (use_lf) {
        // ===== Lax–Friedrichs evolution =====
        while (t < t_final) {
            double umax = max_abs(u);
            double dt = (umax > 0.0) ? CFL_target * dx / umax : 1e-3;

            if (t + dt > t_final)
                dt = t_final - t;

            step_lax_friedrichs(u, u_new, dt, dx);

            double tv  = total_variation(u_new);
            double CFL = (umax > 0.0) ? umax * dt / dx : 0.0;

            std::cout << "t=" << t + dt
                      << "  TV=" << tv
                      << "  CFL=" << CFL << "\n";

            u.swap(u_new);
            t += dt;
        }
    } else {
        // ===== Leapfrog evolution =====

        // First step: compute u^1 using Lax–Friedrichs as a starter
        std::vector<double> u_old = u;        // u^0
        double umax0 = max_abs(u_old);
        double dt = (umax0 > 0.0) ? CFL_target * dx / umax0 : 1e-3;
        if (t + dt > t_final) dt = t_final - t;

        step_lax_friedrichs(u_old, u, dt, dx);  // u = u^1
        t += dt;

        std::cout << "After first LF step: t=" << t
                  << ", TV=" << total_variation(u)
                  << ", CFL=" << (umax0 * dt / dx) << "\n";

        // Now leapfrog for the remaining time
        std::vector<double> u_new(N);
        while (t < t_final) {
            double umax = max_abs(u);  // speed at time level n
            double dt = (umax > 0.0) ? CFL_target * dx / umax : 1e-3;
            if (t + dt > t_final)
                dt = t_final - t;

            step_leapfrog(u_old, u, u_new, dt, dx);

            double tv  = total_variation(u_new);
            double CFL = (umax > 0.0) ? umax * dt / dx : 0.0;

            std::cout << "t=" << t + dt
                      << "  TV=" << tv
                      << "  CFL=" << CFL << "\n";

            // Rotate time levels: n-1 <- n, n <- n+1
            u_old.swap(u);
            u.swap(u_new);

            t += dt;
        }
    }

    // At this point, `u` holds the final solution.
    // You can write it to a file for plotting.
    return 0;
}
