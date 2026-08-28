// tv_cfl_kernel.cpp
// Vitis HLS kernel: fused Total Variation + CFL reduction.
//
// Design notes
// ============
// Both TV and CFL are O(N) reductions over the state array u[].  A naive loop
// accumulating a double has II=14 (the FP adder latency on DSP48E2).  We break
// the loop-carried dependency by maintaining TV_CFL_NUM_PARTIALS independent
// accumulators that are interleaved round-robin.  Each partial chain only sees
// one new value every TV_CFL_NUM_PARTIALS iterations, which is long enough for
// the previous result to have exited the FP pipeline, giving II=1.
//
// Memory interface
// ================
//   u[]       AXI4 master, group gmem0.  Sequential burst read; HLS will
//              schedule prefetch and issue 256-beat bursts automatically when
//              the loop body has II=1.
//   results  AXI4 master, group gmem1.  Two writes at the very end; kept on
//              a separate bundle so the two channels do not contend.
//   scalars  AXI4-Lite (s_axilite) on bundle "control".
//
// The `extern "C"` linkage is required by Vitis so the symbol survives C++
// name mangling in the .xclbin packaging step.

#include "tv_cfl_kernel.hpp"

extern "C" void tv_cfl_kernel(
    const double* u,
    double*       results,
    int           N,
    double        dt,
    double        dx,
    int           flux_id
)
{
    // ---- AXI4 master interfaces ----
    // max_read_burst_length=256: issue up to 256-beat (2 KB) bursts.
    // num_read_outstanding=4:    up to 4 in-flight read transactions so
    //                            memory latency is fully hidden.
    #pragma HLS INTERFACE m_axi port=u       bundle=gmem0  \
        depth=1048576                                       \
        max_read_burst_length=256                          \
        num_read_outstanding=4
    #pragma HLS INTERFACE m_axi port=results bundle=gmem1  \
        depth=2                                             \
        max_write_burst_length=2                           \
        num_write_outstanding=1

    // ---- AXI4-Lite scalar interfaces ----
    #pragma HLS INTERFACE s_axilite port=N        bundle=control
    #pragma HLS INTERFACE s_axilite port=dt       bundle=control
    #pragma HLS INTERFACE s_axilite port=dx       bundle=control
    #pragma HLS INTERFACE s_axilite port=flux_id  bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control

    // ---- Partial accumulator arrays ----
    // ARRAY_PARTITION complete: each element lives in its own register so the
    // synthesis tool can address all TV_CFL_NUM_PARTIALS buckets in one cycle.
    double tv_parts[TV_CFL_NUM_PARTIALS];
    double cfl_parts[TV_CFL_NUM_PARTIALS];
    #pragma HLS ARRAY_PARTITION variable=tv_parts  complete dim=1
    #pragma HLS ARRAY_PARTITION variable=cfl_parts complete dim=1

    // Initialise  unrolled so HLS emits parallel register writes.
    for (int p = 0; p < TV_CFL_NUM_PARTIALS; ++p) {
        #pragma HLS UNROLL
        tv_parts[p]  = 0.0;
        cfl_parts[p] = 0.0;
    }

    // Cache the first element so we can close the periodic boundary at the end
    // without a second memory read.
    const double first_val = u[0];
    double prev = first_val;

    // ---- Main streaming reduction ----
    // Reads u[1..N-1], computes:
    //   TV contribution:  |u[i] - u[i-1]|
    //   CFL contribution: |f'(u[i-1])|
    // using the partial accumulator indexed by (i-1) % NUM_PARTIALS.
    //
    // II=1 is achievable because:
    //   1. tv_parts[p] and cfl_parts[p] are independent registers (full partition).
    //   2. Each register is updated at most once every NUM_PARTIALS iterations,
    //      which exceeds the ~14-cycle FP latency for double arithmetic.
    //   3. The memory read u[i] is sequential → HLS issues burst transactions.
    for (int i = 1; i < N; ++i) {
        #pragma HLS PIPELINE II=1

        const double curr = u[i];
        const int    part = (i - 1) % TV_CFL_NUM_PARTIALS;

        // TV: accumulate |curr - prev| into the assigned partial bucket.
        const double diff    = curr - prev;
        const double abs_diff = (diff >= 0.0) ? diff : -diff;
        tv_parts[part] += abs_diff;

        // CFL: update max |f'(prev)| in the same bucket.
        const double speed = flux_prime_abs(prev, flux_id);
        if (speed > cfl_parts[part]) cfl_parts[part] = speed;

        prev = curr;
        // `curr` is the new `prev` for the next iteration.
        // There is no dependency on tv_parts/cfl_parts here  only on `prev`,
        // which is a scalar register with 1-cycle latency.
    }

    // ---- Periodic boundary closure ----
    // Handle the wrap-around pair: (u[0], u[N-1]).
    // prev == u[N-1] at this point.
    {
        const double wrap_diff = first_val - prev;
        const double abs_wrap  = (wrap_diff >= 0.0) ? wrap_diff : -wrap_diff;
        tv_parts[0] += abs_wrap;

        const double last_speed = flux_prime_abs(prev, flux_id);
        if (last_speed > cfl_parts[0]) cfl_parts[0] = last_speed;
    }

    // ---- Final tree reduction across partials ----
    // Unrolled: synthesiser builds a balanced adder/comparator tree in logic,
    // adding only a few extra cycles of latency after the main loop.
    double tv_total  = 0.0;
    double cfl_total = 0.0;
    for (int p = 0; p < TV_CFL_NUM_PARTIALS; ++p) {
        #pragma HLS UNROLL
        tv_total += tv_parts[p];
        if (cfl_parts[p] > cfl_total) cfl_total = cfl_parts[p];
    }

    // ---- Write results to global memory ----
    results[0] = tv_total;
    results[1] = cfl_total * dt / dx;   // dimensionless CFL number
}
