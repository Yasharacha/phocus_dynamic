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

// Reinterpret 64 raw bits as a double (bit copy, no numeric conversion).
static inline real_t bits_to_real(ap_uint<64> b)
{
    union { uint64_t i; real_t d; } cv;
    cv.i = b.to_uint64();
    return cv.d;
}

extern "C" int tv_cfl_kernel(
    const word_t* u,
    real_t*       results,
    int           N,
    real_t        dt,
    real_t        dx,
    int           flux_id,
    real_t        tv_prev,
    real_t        tol
)
{
    // ---- AXI4 master interfaces ----
    // max_read_burst_length=256: issue up to 256-beat (2 KB) bursts.
    // num_read_outstanding=4:    up to 4 in-flight read transactions so
    //                            memory latency is fully hidden.
    #pragma HLS INTERFACE m_axi port=u       bundle=gmem0  \
        depth=65536                                        \
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
    #pragma HLS INTERFACE s_axilite port=tv_prev  bundle=control
    #pragma HLS INTERFACE s_axilite port=tol      bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control

    if (N <= 0) {
        results[0] = 0;
        results[1] = 0;
        return 0;
    }

    // ---- Partial accumulator arrays ----
    // ARRAY_PARTITION complete: each element lives in its own register so the
    // synthesis tool can address all TV_CFL_NUM_PARTIALS buckets in one cycle.
    real_t tv_parts[TV_CFL_NUM_PARTIALS];
    real_t cfl_parts[TV_CFL_NUM_PARTIALS];
    #pragma HLS ARRAY_PARTITION variable=tv_parts  complete dim=1
    #pragma HLS ARRAY_PARTITION variable=cfl_parts complete dim=1

    // Initialise  unrolled so HLS emits parallel register writes.
    for (int p = 0; p < TV_CFL_NUM_PARTIALS; ++p) {
        #pragma HLS UNROLL
        tv_parts[p]  = 0;
        cfl_parts[p] = 0;
    }

    // ---- Main streaming reduction: two elements per clock ----
    // The u port is 128 bits wide (pair_t = two doubles), so each iteration reads u[2j] and
    // u[2j+1] in one beat. Per iteration:
    //   TV : |u[2j]-u[2j-1]| + |u[2j+1]-u[2j]|   (first term skipped at j == 0)
    //   CFL: max(|f'(u[2j])|, |f'(u[2j+1])|)
    // Both go into bucket j % NUM_PARTIALS. The two-term TV sum is formed first, so each
    // bucket still sees a single add per visit and II=1 holds (a bucket is revisited only
    // every NUM_PARTIALS iterations, longer than the FP add latency).
    // For odd N the last word's second element is ignored; the buffer needs one spare element.
    const int  n_words = (N + 1) / 2;
    const bool odd     = (N & 1) != 0;
    real_t prev      = 0;   // last element of the previous word
    real_t first_val = 0;   // u[0], kept for the periodic wrap
    for (int j = 0; j < n_words; ++j) {
        #pragma HLS PIPELINE II=1
        const word_t w  = u[j];                       // one 128-bit read = two doubles
        const real_t e0 = bits_to_real(w.range(63, 0));
        const real_t e1 = bits_to_real(w.range(127, 64));
        const bool has_e1 = !(odd && (j == n_words - 1));
        const int  part   = j % TV_CFL_NUM_PARTIALS;

        // TV
        const real_t df0 = e0 - prev;
        const real_t df1 = e1 - e0;
        const real_t ad0 = (j == 0) ? (real_t)0 : ((df0 >= 0) ? df0 : -df0);
        const real_t ad1 = has_e1 ? ((df1 >= 0) ? df1 : -df1) : (real_t)0;
        tv_parts[part] += ad0 + ad1;

        // CFL
        const real_t sp0 = flux_prime_abs(e0, flux_id);
        const real_t sp1 = has_e1 ? flux_prime_abs(e1, flux_id) : (real_t)0;
        const real_t sp  = (sp1 > sp0) ? sp1 : sp0;
        if (sp > cfl_parts[part]) cfl_parts[part] = sp;

        if (j == 0) first_val = e0;
        prev = has_e1 ? e1 : e0;   // u[2j+1], or u[2j] for the odd last element
    }

    // ---- Periodic boundary closure ----
    // The wrap pair is (u[0], u[N-1]); prev == u[N-1] here. Every element's CFL speed was
    // already taken inside the loop.
    {
        const real_t wrap_diff = first_val - prev;
        const real_t abs_wrap  = (wrap_diff >= 0) ? wrap_diff : -wrap_diff;
        tv_parts[0] += abs_wrap;
    }

    // ---- Final reduction across partials (sequential chain, see below) ----
    // Unrolled, but NOT a tree: HLS keeps floating-point adds in source order, so the
    // 16 TV adds form a dependent chain (one shared adder, ~5 cycles each). The CFL max
    // is a similar compare chain. This runs once per call, ~100 cycles total, which is
    // negligible next to the N-cycle main loop.
    real_t tv_total  = 0;
    real_t cfl_total = 0;
    for (int p = 0; p < TV_CFL_NUM_PARTIALS; ++p) {
        #pragma HLS UNROLL
        tv_total += tv_parts[p];
        if (cfl_parts[p] > cfl_total) cfl_total = cfl_parts[p];
    }

    // ---- Violation check ----
    // bit 0: CFL number > 1.
    // bit 1: TV grew versus the previous check, beyond a relative tolerance.
    //        A negative tv_prev disables this test (first call, no history).
    const real_t cfl_number = cfl_total * dt / dx;   // dimensionless CFL number
    int status = 0;
    if (cfl_number > (real_t)1) status |= 1;
    if (tv_prev >= (real_t)0 && tv_total > tv_prev * ((real_t)1 + tol)) status |= 2;

    // ---- Write results to global memory ----
    results[0] = tv_total;
    results[1] = cfl_number;
    return status;   // 0 = OK; returned through the AXI-Lite control bank
}
