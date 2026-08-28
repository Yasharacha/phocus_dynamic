#pragma once
// tv_cfl_kernel.hpp
// Public interface for the TV+CFL Vitis HLS kernel.
//
// The kernel performs a single-pass streaming reduction over the state array u[]:
//   TV  = sum_i |u[i+1] - u[i]|   (periodic: u[N] = u[0])
//   CFL = max_i |f'(u[i])| * dt / dx

#include "flux_functions.hpp"

// Number of independent partial accumulators.
// Must be >= FP add latency (~14 cycles for double on DSP48E2).
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
extern "C" void tv_cfl_kernel(
    const double* u,
    double*       results,
    int           N,
    double        dt,
    double        dx,
    int           flux_id
);
