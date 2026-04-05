"""
Generate speedup-vs-N plots where each line represents a different thread count.
X-axis: mesh points (N), Y-axis: OpenMP speedup (%), one subplot per flux.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    4: "#2ca02c",
    8: "#d62728",
    16: "#9467bd",
    20: "#8c564b",
}


def load_csv(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            try:
                rows.append({
                    "flux": raw["flux"],
                    "scheme": raw["scheme"],
                    "N": int(raw["N"]),
                    "threads": int(raw["omp_threads"]),
                    "serial_ms": float(raw["serial_total_ms"]),
                    "omp_ms": float(raw["omp_total_ms"]),
                    "simd_ms": float(raw["simd_total_ms"]),
                    "omp_speedup": float(raw["omp_speedup_pct"]),
                    "simd_speedup": float(raw["simd_speedup_pct"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def plot_speedup_grid(rows, output_dir: Path, scheme_filter="lax_friedrichs"):
    """One plot per flux, x=N, lines=thread counts."""
    by_flux = defaultdict(list)
    for r in rows:
        if r["scheme"] == scheme_filter:
            by_flux[r["flux"]].append(r)

    fluxes = sorted(by_flux.keys())
    n_fluxes = len(fluxes)
    ncols = min(3, n_fluxes)
    nrows = (n_fluxes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for idx, flux in enumerate(fluxes):
        ax = axes[idx // ncols][idx % ncols]
        flux_rows = by_flux[flux]

        by_thread = defaultdict(list)
        for r in flux_rows:
            by_thread[r["threads"]].append(r)

        for t in sorted(by_thread.keys()):
            t_rows = sorted(by_thread[t], key=lambda r: r["N"])
            ns = [r["N"] for r in t_rows]
            speedups = [r["omp_speedup"] for r in t_rows]
            color = COLORS.get(t, None)
            ax.plot(ns, speedups, marker="o", linewidth=2, label=f"{t} threads", color=color)

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Mesh Points (N)")
        ax.set_ylabel("OpenMP Speedup vs Serial (%)")
        ax.set_title(f"{flux} ({scheme_filter})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_fluxes, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("OpenMP Speedup by Thread Count", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"speedup_by_threads_{scheme_filter}.png", dpi=200)
    plt.close(fig)


def plot_runtime_grid(rows, output_dir: Path, scheme_filter="lax_friedrichs"):
    """One plot per flux, x=N, lines=thread counts showing absolute runtime."""
    by_flux = defaultdict(list)
    for r in rows:
        if r["scheme"] == scheme_filter:
            by_flux[r["flux"]].append(r)

    fluxes = sorted(by_flux.keys())
    n_fluxes = len(fluxes)
    ncols = min(3, n_fluxes)
    nrows = (n_fluxes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for idx, flux in enumerate(fluxes):
        ax = axes[idx // ncols][idx % ncols]
        flux_rows = by_flux[flux]

        by_thread = defaultdict(list)
        for r in flux_rows:
            by_thread[r["threads"]].append(r)

        # Plot serial baseline (same for all thread counts, take from thread=1)
        if 1 in by_thread:
            t_rows = sorted(by_thread[1], key=lambda r: r["N"])
            ns = [r["N"] for r in t_rows]
            serial = [r["serial_ms"] for r in t_rows]
            ax.plot(ns, serial, marker="s", linewidth=2.5, label="Serial", color="black", linestyle="--")

        for t in sorted(by_thread.keys()):
            t_rows = sorted(by_thread[t], key=lambda r: r["N"])
            ns = [r["N"] for r in t_rows]
            omp_ms = [r["omp_ms"] for r in t_rows]
            color = COLORS.get(t, None)
            ax.plot(ns, omp_ms, marker="o", linewidth=2, label=f"OMP {t}t", color=color)

        # SIMD line (same for all thread counts)
        if 1 in by_thread:
            t_rows = sorted(by_thread[1], key=lambda r: r["N"])
            ns = [r["N"] for r in t_rows]
            simd = [r["simd_ms"] for r in t_rows]
            ax.plot(ns, simd, marker="^", linewidth=2, label="SIMD", color="#17becf", linestyle="-.")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mesh Points (N)")
        ax.set_ylabel("Runtime (ms)")
        ax.set_title(f"{flux} ({scheme_filter})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)

    for idx in range(n_fluxes, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Runtime by Thread Count", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"runtime_by_threads_{scheme_filter}.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_plots_all"))
    args = parser.parse_args()

    rows = load_csv(args.csv_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scheme in ["lax_friedrichs", "leapfrog"]:
        plot_speedup_grid(rows, args.output_dir, scheme)
        plot_runtime_grid(rows, args.output_dir, scheme)

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
