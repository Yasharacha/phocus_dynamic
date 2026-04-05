FROM gcc:13

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/plotenv && \
    /opt/plotenv/bin/pip install --no-cache-dir matplotlib

WORKDIR /app

COPY src/benchmarks/combined.cpp ./
COPY scripts/plot_benchmark_csv.py scripts/plot_speedup_by_threads.py ./scripts/

RUN g++ -O3 -fopenmp -std=c++17 -o benchmark_fluxes_combined combined.cpp

COPY docker_run.sh ./
RUN chmod +x docker_run.sh

ENTRYPOINT ["./docker_run.sh"]
