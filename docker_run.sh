#!/bin/bash
set -e

OUTPUT_DIR="/app/results"
mkdir -p "$OUTPUT_DIR"

echo "=== System Info ==="
echo "CPU cores: $(nproc)"
echo "GCC version: $(g++ --version | head -1)"
echo ""

# Run the combined benchmark for all fluxes
echo "=== Running benchmarks ==="
./benchmark_fluxes_combined --output "$OUTPUT_DIR/benchmark_fluxes_all.csv" "$@"

echo ""
echo "=== Generating plots ==="
/opt/plotenv/bin/python scripts/plot_benchmark_csv.py "$OUTPUT_DIR/benchmark_fluxes_all.csv" \
    --output-dir "$OUTPUT_DIR/plots"

/opt/plotenv/bin/python scripts/plot_speedup_by_threads.py "$OUTPUT_DIR/benchmark_fluxes_all.csv" \
    --output-dir "$OUTPUT_DIR/plots"

echo ""
echo "=== Done ==="
echo "Results in: $OUTPUT_DIR/"
echo "CSV:   $OUTPUT_DIR/benchmark_fluxes_all.csv"
echo "Plots: $OUTPUT_DIR/plots/"
ls "$OUTPUT_DIR/plots/" | wc -l | xargs -I{} echo "{} plot files generated"
