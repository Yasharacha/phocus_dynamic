# Phocus Dynamic

Numerical solvers and performance benchmarks for 1D conservation law PDEs using finite difference schemes (Lax-Friedrichs and Leapfrog) with periodic boundary conditions.

## Flux Functions

| Flux | f(u) | Domain |
|------|------|--------|
| Burgers | 0.5 u^2 | [0, 38] |
| LWR Traffic Flow | u(1 - u) | [0, 38] |
| Flood Wave | max(u,0)^(3/2) | [0, 38] |
| Cubic | u^3 | [0, 38] |
| Buckley-Leverett | u^2 / (u^2 + 0.25(1-u)^2) | [0, 38] |
| Logarithmic | ln(u) | [0, 38] |
| Trilinear | -max(-u,0) + max(u-1,0) | [0, 38] |

## Project Structure

```
phocus_dynamic/
├── src/
│   ├── solvers/              # Core solver implementations (serial)
│   ├── multithreaded/        # OpenMP multithreaded solvers
│   └── benchmarks/
│       ├── combined.cpp      # Unified benchmark for all 7 fluxes
│       ├── openmp/           # OpenMP thread-scaling benchmarks
│       ├── pipeline/         # Pipeline parallelism benchmarks
│       ├── full_openmp/      # Full OpenMP benchmarks with CSV output
│       └── full_simd/        # SIMD vectorization benchmarks
├── scripts/                  # Python plotting scripts
├── results/                  # Benchmark CSV output files
├── plots/                    # Generated benchmark plots
├── Dockerfile                # Standardized benchmark environment
├── docker_run.sh             # Entrypoint for Docker container
└── enviroment.yml            # Conda environment file
```

## Building Locally

Each `.cpp` file is a standalone compilation unit. Compile with GCC and OpenMP support:

```bash
# macOS (requires Homebrew GCC)
g++-14 -O3 -fopenmp -std=c++17 src/benchmarks/combined.cpp -o benchmark_combined

# Linux
g++ -O3 -fopenmp -std=c++17 src/benchmarks/combined.cpp -o benchmark_combined
```

Apple Clang does not support `-fopenmp`. Install GCC via Homebrew: `brew install gcc`.

### Running the combined benchmark

```bash
# All 7 fluxes
./benchmark_combined

# Single flux
./benchmark_combined --flux burgers

# Custom output path
./benchmark_combined --output results/my_results.csv

# Custom mesh sizes
./benchmark_combined --N 100,1000,10000
```

Thread counts (1, 2, 4, 8, 16, 20) are automatically filtered to the number of available cores.

### Generating plots

```bash
pip install matplotlib

# Per-flux detailed plots (runtime, speedup, scaling, breakdown)
python scripts/plot_benchmark_csv.py results/benchmark_fluxes_all.csv --output-dir plots/

# Thread-scaling comparison across all fluxes
python scripts/plot_speedup_by_threads.py results/benchmark_fluxes_all.csv --output-dir plots/
```

## Docker

Docker provides a standardized environment (GCC 13, Linux) so benchmarks produce consistent results across macOS and Windows.

### Build

```bash
docker build -t phocus .
```

### Run all benchmarks

```bash
docker run -v $(pwd)/output:/app/results phocus
```

Results (CSV + plots) will appear in `./output/`.

### Run a single flux

```bash
docker run -v $(pwd)/output:/app/results phocus --flux burgers
```

### Run with custom mesh sizes

```bash
docker run -v $(pwd)/output:/app/results phocus --N 1000,10000,100000
```

### Control thread count

```bash
docker run -v $(pwd)/output:/app/results -e OMP_NUM_THREADS=8 phocus
```

### Multi-core cloud runs

To test with 16 or 20 threads, run on a machine with enough cores (e.g. an AWS c5.4xlarge with 16 vCPUs or c5.12xlarge with 48 vCPUs):

```bash
docker run --cpus=20 -v $(pwd)/output:/app/results phocus
```

## Benchmark Output

The combined benchmark produces a CSV with columns:

| Column | Description |
|--------|-------------|
| `flux` | Flux function name |
| `scheme` | `lax_friedrichs` or `leapfrog` |
| `N` | Mesh size |
| `omp_threads` | Number of OpenMP threads |
| `serial_total_ms` | Serial runtime (ms) |
| `omp_total_ms` | OpenMP runtime (ms) |
| `simd_total_ms` | SIMD runtime (ms) |
| `omp_speedup_pct` | OpenMP speedup vs serial (%) |
| `simd_speedup_pct` | SIMD speedup vs serial (%) |

## Plot Types

| Plot | Description |
|------|-------------|
| `*_runtime.png` | Serial vs best OpenMP vs SIMD runtime |
| `*_speedup.png` | Speedup percentage vs serial |
| `*_omp_scaling.png` | OpenMP speedup by thread count |
| `*_runtime_breakdown.png` | Step/CFL/TV time share at largest N |
| `speedup_by_threads_*.png` | All fluxes compared, x=N, lines=threads |
| `runtime_by_threads_*.png` | Absolute runtime, x=N, lines=threads |
