#pragma once
// flux_functions.hpp
// Flux derivative dispatch for all 7 supported fluxes.
//
// HLS note: flux_id is an AXI-Lite register — constant during the entire
// kernel execution. The synthesiser will specialise the switch and inline
// only the live branch, so there is no runtime MUX penalty.

#ifdef __SYNTHESIS__
#  include <hls_math.h>
#  define HLS_SQRT(x) hls::sqrt(x)
#else
#  include <cmath>
#  define HLS_SQRT(x) std::sqrt(x)
#endif

// Numeric IDs must match what the host passes as flux_id.
enum FluxID {
    FLUX_BURGERS      = 0,  // f(u) = 0.5*u^2        f'(u) = u
    FLUX_LWR          = 1,  // f(u) = u*(1-u)         f'(u) = 1 - 2*u
    FLUX_FLOOD_WAVE   = 2,  // f(u) = max(u,0)^1.5    f'(u) = 1.5*sqrt(u)  [u>0]
    FLUX_CUBIC        = 3,  // f(u) = u^3             f'(u) = 3*u^2
    FLUX_BUCKLEY_LEV  = 4,  // f(u) = u^2/(u^2 + 0.25*(1-u)^2)
    FLUX_LOG          = 5,  // f(u) = ln(u)           f'(u) = 1/u          [u>0]
    FLUX_TRILINEAR    = 6   // max wave speed = 1     f'(u) = 1  (constant)
};

// Returns |f'(u)| for the requested flux.
// Called once per element inside the pipelined loop — must be II=1 compatible.
inline double flux_prime_abs(double u, int flux_id)
{
#pragma HLS INLINE
    double result = 0.0;
#ifdef TV_CFL_FIXED_FLUX
    // Build a single-flux kernel: fixing the flux at compile time lets the compiler fold
    // the switch below to one branch, so only that flux's arithmetic is synthesized.
    flux_id = TV_CFL_FIXED_FLUX;
#endif
    switch (flux_id) {
        case FLUX_BURGERS: {
            result = (u >= 0.0) ? u : -u;   // |f'| = |u|
            break;
        }
        case FLUX_LWR: {
            double fp = 1.0 - 2.0 * u;
            result = (fp >= 0.0) ? fp : -fp; // |1 - 2u|
            break;
        }
        case FLUX_FLOOD_WAVE: {
            result = (u > 0.0) ? 1.5 * HLS_SQRT(u) : 0.0;
            break;
        }
        case FLUX_CUBIC: {
            result = 3.0 * u * u;            // always >= 0
            break;
        }
        case FLUX_BUCKLEY_LEV: {
            double u2    = u * u;
            double a     = 1.0 - u;
            double denom = u2 + 0.25 * a * a;
            if (denom > 1e-14) {
                double fp = 0.5 * u * (1.0 - u) / (denom * denom);
                result = (fp >= 0.0) ? fp : -fp;
            }
            break;
        }
        case FLUX_LOG: {
            result = (u > 0.0) ? 1.0 / u : 0.0;
            break;
        }
        case FLUX_TRILINEAR: {
            result = 1.0;
            break;
        }
        default:
            result = 0.0;
    }
    return result;
}
