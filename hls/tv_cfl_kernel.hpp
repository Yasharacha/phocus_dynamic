#pragma once
// tv_cfl_kernel.hpp
// Public interface for the TV+CFL Vitis HLS kernel.
//
// The kernel performs a single-pass streaming reduction over the state array u[]:
//   TV  = sum_i |u[i+1] - u[i]|   (periodic: u[N] = u[0])
//   CFL = max_i |f'(u[i])| * dt / dx

// Working precision: double, matching the CPU solvers. Measured on the Ultra96
// (ZU3EG) this uses ~28% of DSPs with II=1. Compile with -DTV_CFL_USE_FLOAT for
// a float kernel (NOTE: flux_functions.hpp is still double-only, so that build
// is not a true float design until flux_functions.hpp is converted to real_t).
#ifdef TV_CFL_USE_FLOAT
typedef float real_t;
#else
typedef double real_t;
#endif

#include "flux_functions.hpp"

// Number of independent partial accumulators.
// Must be >= FP add latency (~14 cycles for double, fewer for float).
// 16 gives comfortable margin and maps cleanly to ARRAY_PARTITION complete.
#define TV_CFL_NUM_PARTIALS 16

// Kernel top-level function.
// Ports:
//   u       [in]  AXI4 master — state array, N doubles, sequential burst read
//   results [out] AXI4 master — 2 doubles: [0]=TV, [1]=CFL (allocated by host)
//   N            [in]  AXI4-Lite — number of mesh points
//   dt           [in]  AXI4-Lite — time step
//   dx           [in]  AXI4-Lite — grid spacing
//   flux_id      [in]  AXI4-Lite — selects flux derivative (see FluxID enum)
//   tv_prev      [in]  AXI4-Lite — TV from the previous check; negative = no history
//   tol          [in]  AXI4-Lite — relative tolerance for the "TV increased" test
// Return value (AXI-Lite): violation bitmask. 0 = OK, bit0 = CFL > 1, bit1 = TV increased.
extern "C" int tv_cfl_kernel(
    const real_t* u,
    real_t*       results,
    int           N,
    real_t        dt,
    real_t        dx,
    int           flux_id,
    real_t        tv_prev,
    real_t        tol
);
