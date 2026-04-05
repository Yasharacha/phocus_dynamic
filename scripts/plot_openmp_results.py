import re
import matplotlib.pyplot as plt
import numpy as np

def parse_benchmark(filepath):
    """Parse OpenMP benchmark output into structured data.
    Returns dict: {n_steps: {N: {serial_ms, omp_1t, omp_2t, omp_4t, omp_8t, spd_1, spd_2, spd_4, spd_8}}}
    """
    results = {}
    current_steps = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'=== n_steps = (\d+) ===', line)
            if m:
                current_steps = int(m.group(1))
                results[current_steps] = {}
                continue

            if current_steps is None or line.startswith('N') or line.startswith('-') or not line:
                continue

            parts = line.split()
            if len(parts) >= 10:
                N = int(parts[0])
                serial = float(parts[1])
                omp1 = float(parts[2])
                spd1 = float(parts[3])
                omp2 = float(parts[4])
                spd2 = float(parts[5])
                omp4 = float(parts[6])
                spd4 = float(parts[7])
                omp8 = float(parts[8])
                spd8 = float(parts[9])
                results[current_steps][N] = {
                    'serial': serial,
                    'speedup_1t': spd1,
                    'speedup_2t': spd2,
                    'speedup_4t': spd4,
                    'speedup_8t': spd8,
                }
    return results

# Parse all four benchmarks
cases = {
    'Burgers': '/tmp/results_omp_burgers.txt',
    'LWR Traffic Flow': '/tmp/results_omp_lwr.txt',
    'Flood Wave': '/tmp/results_omp_floodwave.txt',
    'Cubic': '/tmp/results_omp_cubic.txt',
}

all_results = {}
for name, path in cases.items():
    try:
        all_results[name] = parse_benchmark(path)
    except FileNotFoundError:
        print(f"Warning: {path} not found, skipping {name}")

# Pick n_steps=100 as the representative case
target_steps = 100
thread_labels = ['1t', '2t', '4t', '8t']
thread_counts = [1, 2, 4, 8]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, results) in enumerate(all_results.items()):
    ax = axes[idx]

    if target_steps not in results:
        ax.set_title(f'{name} (no data for n_steps={target_steps})')
        continue

    data = results[target_steps]
    Ns = sorted(data.keys())

    for i, (tl, tc) in enumerate(zip(thread_labels, thread_counts)):
        speedups = [data[N][f'speedup_{tl}'] for N in Ns]
        ax.semilogx(Ns, speedups, marker=markers[i], color=colors[i],
                    label=f'{tc} thread{"s" if tc > 1 else ""}', linewidth=2, markersize=6)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='baseline')
    ax.set_xlabel('N (grid points)')
    ax.set_ylabel('Speedup vs serial')
    ax.set_title(f'{name}')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

fig.suptitle(f'OpenMP Speedup vs Serial (n_steps={target_steps}, 5 reps avg)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('openmp_speedup_results.png', dpi=150, bbox_inches='tight')
print("Saved openmp_speedup_results.png")
plt.close()

# Also make a comparison plot: best speedup (8t) across all cases
fig2, ax2 = plt.subplots(figsize=(10, 6))
for idx, (name, results) in enumerate(all_results.items()):
    if target_steps not in results:
        continue
    data = results[target_steps]
    Ns = sorted(data.keys())
    speedups_8t = [data[N]['speedup_8t'] for N in Ns]
    ax2.semilogx(Ns, speedups_8t, marker=markers[idx], color=colors[idx],
                 label=name, linewidth=2, markersize=7)

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('N (grid points)', fontsize=12)
ax2.set_ylabel('Speedup (8 threads vs serial)', fontsize=12)
ax2.set_title(f'OpenMP 8-Thread Speedup Comparison (n_steps={target_steps})',
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('openmp_speedup_comparison.png', dpi=150, bbox_inches='tight')
print("Saved openmp_speedup_comparison.png")
plt.close()
