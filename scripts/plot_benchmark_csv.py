import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class BenchmarkRow:
    flux: str
    scheme: str
    N: int
    omp_threads: int
    dt: float
    n_steps: int
    reps: int
    serial_total_ms: float
    omp_total_ms: float
    simd_total_ms: float
    omp_speedup_pct: float
    simd_speedup_pct: float
    serial_cfl_pct: float
    serial_tv_pct: float
    omp_cfl_pct: float
    omp_tv_pct: float
    simd_cfl_pct: float
    simd_tv_pct: float

    @property
    def serial_step_pct(self) -> float:
        return max(0.0, 100.0 - self.serial_cfl_pct - self.serial_tv_pct)

    @property
    def omp_step_pct(self) -> float:
        return max(0.0, 100.0 - self.omp_cfl_pct - self.omp_tv_pct)

    @property
    def simd_step_pct(self) -> float:
        return max(0.0, 100.0 - self.simd_cfl_pct - self.simd_tv_pct)


def load_rows(csv_path: Path) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                BenchmarkRow(
                    flux=raw["flux"],
                    scheme=raw["scheme"],
                    N=int(raw["N"]),
                    omp_threads=int(raw["omp_threads"]),
                    dt=float(raw["dt"]),
                    n_steps=int(raw["n_steps"]),
                    reps=int(raw["reps"]),
                    serial_total_ms=float(raw["serial_total_ms"]),
                    omp_total_ms=float(raw["omp_total_ms"]),
                    simd_total_ms=float(raw["simd_total_ms"]),
                    omp_speedup_pct=float(raw["omp_speedup_pct"]),
                    simd_speedup_pct=float(raw["simd_speedup_pct"]),
                    serial_cfl_pct=float(raw["serial_cfl_pct"]),
                    serial_tv_pct=float(raw["serial_tv_pct"]),
                    omp_cfl_pct=float(raw["omp_cfl_pct"]),
                    omp_tv_pct=float(raw["omp_tv_pct"]),
                    simd_cfl_pct=float(raw["simd_cfl_pct"]),
                    simd_tv_pct=float(raw["simd_tv_pct"]),
                )
            )
    return rows


def slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def group_rows(rows: list[BenchmarkRow]) -> dict[tuple[str, str], list[BenchmarkRow]]:
    grouped: dict[tuple[str, str], list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.flux, row.scheme)].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: (row.N, row.omp_threads))
    return grouped


def best_omp_by_n(rows: list[BenchmarkRow]) -> dict[int, BenchmarkRow]:
    best: dict[int, BenchmarkRow] = {}
    for row in rows:
        current = best.get(row.N)
        if current is None or row.omp_total_ms < current.omp_total_ms:
            best[row.N] = row
    return best


def rows_by_thread(rows: list[BenchmarkRow]) -> dict[int, list[BenchmarkRow]]:
    grouped: dict[int, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[row.omp_threads].append(row)
    for thread_count in grouped:
        grouped[thread_count].sort(key=lambda row: row.N)
    return grouped


def save_runtime_plot(rows: list[BenchmarkRow], output_dir: Path, flux: str, scheme: str) -> None:
    best_rows = [best_omp_by_n(rows)[n] for n in sorted(best_omp_by_n(rows))]
    ns = [row.N for row in best_rows]
    serial = [row.serial_total_ms for row in best_rows]
    omp = [row.omp_total_ms for row in best_rows]
    simd = [row.simd_total_ms for row in best_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ns, serial, marker="o", linewidth=2.5, label="Serial", color="#1f4e79")
    ax.plot(ns, omp, marker="o", linewidth=2.5, label="Best OpenMP", color="#d04a02")
    ax.plot(ns, simd, marker="o", linewidth=2.5, label="SIMD", color="#3a7d44")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mesh Size N")
    ax.set_ylabel("Total Runtime (ms)")
    ax.set_title(f"{flux} {scheme}: Runtime by Approach")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    for row in best_rows:
        ax.annotate(f"{row.omp_threads}t", (row.N, row.omp_total_ms), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#d04a02")

    fig.tight_layout()
    fig.savefig(output_dir / f"{slugify(flux)}_{slugify(scheme)}_runtime.png", dpi=200)
    plt.close(fig)


def save_speedup_plot(rows: list[BenchmarkRow], output_dir: Path, flux: str, scheme: str) -> None:
    best_rows = [best_omp_by_n(rows)[n] for n in sorted(best_omp_by_n(rows))]
    ns = [row.N for row in best_rows]
    omp_speedup = [row.omp_speedup_pct for row in best_rows]
    simd_speedup = [row.simd_speedup_pct for row in best_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ns, omp_speedup, marker="o", linewidth=2.5, label="Best OpenMP Speedup", color="#d04a02")
    ax.plot(ns, simd_speedup, marker="o", linewidth=2.5, label="SIMD Speedup", color="#3a7d44")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Mesh Size N")
    ax.set_ylabel("Speedup vs Serial (%)")
    ax.set_title(f"{flux} {scheme}: Best Speedup vs Serial")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    for row in best_rows:
        ax.annotate(f"{row.omp_threads}t", (row.N, row.omp_speedup_pct), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#d04a02")

    fig.tight_layout()
    fig.savefig(output_dir / f"{slugify(flux)}_{slugify(scheme)}_speedup.png", dpi=200)
    plt.close(fig)


def save_omp_scaling_plot(rows: list[BenchmarkRow], output_dir: Path, flux: str, scheme: str) -> None:
    by_thread = rows_by_thread(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    for thread_count, thread_rows in sorted(by_thread.items()):
        ns = [row.N for row in thread_rows]
        speedups = [row.omp_speedup_pct for row in thread_rows]
        ax.plot(ns, speedups, marker="o", linewidth=2, label=f"{thread_count} threads")

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Mesh Size N")
    ax.set_ylabel("OpenMP Speedup vs Serial (%)")
    ax.set_title(f"{flux} {scheme}: OpenMP Scaling by Thread Count")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / f"{slugify(flux)}_{slugify(scheme)}_omp_scaling.png", dpi=200)
    plt.close(fig)


def save_breakdown_plot(rows: list[BenchmarkRow], output_dir: Path, flux: str, scheme: str) -> None:
    best_row = best_omp_by_n(rows)[max(row.N for row in rows)]
    labels = ["Serial", f"Best OpenMP\n({best_row.omp_threads}t)", "SIMD"]

    step_values = [best_row.serial_step_pct, best_row.omp_step_pct, best_row.simd_step_pct]
    cfl_values = [best_row.serial_cfl_pct, best_row.omp_cfl_pct, best_row.simd_cfl_pct]
    tv_values = [best_row.serial_tv_pct, best_row.omp_tv_pct, best_row.simd_tv_pct]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = range(len(labels))
    ax.bar(x, step_values, label="Step Kernel", color="#4c78a8")
    ax.bar(x, cfl_values, bottom=step_values, label="CFL", color="#f58518")
    stacked = [step_values[i] + cfl_values[i] for i in range(len(labels))]
    ax.bar(x, tv_values, bottom=stacked, label="TV", color="#54a24b")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of Runtime (%)")
    ax.set_title(f"{flux} {scheme}: Runtime Breakdown at Largest N={best_row.N}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / f"{slugify(flux)}_{slugify(scheme)}_runtime_breakdown.png", dpi=200)
    plt.close(fig)


def save_openmp_verification_plot(rows: list[BenchmarkRow], output_dir: Path, flux: str, scheme: str) -> None:
    by_n: dict[int, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        by_n[row.N].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    cfl_ax, tv_ax = axes

    for N in sorted(by_n):
        n_rows = sorted(by_n[N], key=lambda row: row.omp_threads)
        thread_counts = [row.omp_threads for row in n_rows]
        cfl_ms = [row.omp_total_ms * row.omp_cfl_pct / 100.0 for row in n_rows]
        tv_ms = [row.omp_total_ms * row.omp_tv_pct / 100.0 for row in n_rows]

        cfl_ax.plot(thread_counts, cfl_ms, marker="o", linewidth=2, label=f"N={N}")
        tv_ax.plot(thread_counts, tv_ms, marker="o", linewidth=2, label=f"N={N}")

    cfl_ax.set_title(f"{flux} {scheme}: OpenMP CFL Time")
    cfl_ax.set_xlabel("OpenMP Threads")
    cfl_ax.set_ylabel("CFL Time (ms)")
    cfl_ax.grid(True, alpha=0.25)

    tv_ax.set_title(f"{flux} {scheme}: OpenMP TV Time")
    tv_ax.set_xlabel("OpenMP Threads")
    tv_ax.set_ylabel("TV Time (ms)")
    tv_ax.grid(True, alpha=0.25)
    tv_ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / f"{slugify(flux)}_{slugify(scheme)}_openmp_verification_ms.png", dpi=200)
    plt.close(fig)


def save_openmp_verification_share_plot(rows: list[BenchmarkRow], output_dir: Path, flux: str, scheme: str) -> None:
    by_n: dict[int, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        by_n[row.N].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    cfl_ax, tv_ax = axes

    for N in sorted(by_n):
        n_rows = sorted(by_n[N], key=lambda row: row.omp_threads)
        thread_counts = [row.omp_threads for row in n_rows]
        cfl_pct = [row.omp_cfl_pct for row in n_rows]
        tv_pct = [row.omp_tv_pct for row in n_rows]

        cfl_ax.plot(thread_counts, cfl_pct, marker="o", linewidth=2, label=f"N={N}")
        tv_ax.plot(thread_counts, tv_pct, marker="o", linewidth=2, label=f"N={N}")

    cfl_ax.set_title(f"{flux} {scheme}: OpenMP CFL Share")
    cfl_ax.set_xlabel("OpenMP Threads")
    cfl_ax.set_ylabel("CFL Share of Runtime (%)")
    cfl_ax.grid(True, alpha=0.25)

    tv_ax.set_title(f"{flux} {scheme}: OpenMP TV Share")
    tv_ax.set_xlabel("OpenMP Threads")
    tv_ax.set_ylabel("TV Share of Runtime (%)")
    tv_ax.grid(True, alpha=0.25)
    tv_ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / f"{slugify(flux)}_{slugify(scheme)}_openmp_verification_pct.png", dpi=200)
    plt.close(fig)


def write_best_openmp_summary(grouped_rows: dict[tuple[str, str], list[BenchmarkRow]], output_dir: Path) -> None:
    summary_path = output_dir / "best_openmp_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "flux",
                "scheme",
                "N",
                "best_omp_threads",
                "serial_total_ms",
                "best_omp_total_ms",
                "simd_total_ms",
                "best_omp_speedup_pct",
                "simd_speedup_pct",
            ]
        )
        for (flux, scheme), rows in sorted(grouped_rows.items()):
            best_rows = best_omp_by_n(rows)
            for N in sorted(best_rows):
                row = best_rows[N]
                writer.writerow(
                    [
                        flux,
                        scheme,
                        N,
                        row.omp_threads,
                        row.serial_total_ms,
                        row.omp_total_ms,
                        row.simd_total_ms,
                        row.omp_speedup_pct,
                        row.simd_speedup_pct,
                    ]
                )


def write_plot_readme(grouped_rows: dict[tuple[str, str], list[BenchmarkRow]], output_dir: Path) -> None:
    lines = [
        "# Benchmark Plot Guide",
        "",
        "Generated graphics:",
        "- `*_runtime.png`: total runtime comparison for Serial, best OpenMP, and SIMD.",
        "- `*_speedup.png`: percent speedup versus Serial for best OpenMP and SIMD.",
        "- `*_omp_scaling.png`: OpenMP speedup for each tested thread count.",
        "- `*_runtime_breakdown.png`: step/CFL/TV runtime share at the largest mesh size.",
        "- `*_openmp_verification_ms.png`: CFL and TV time in milliseconds across OpenMP thread counts.",
        "- `*_openmp_verification_pct.png`: CFL and TV share of runtime across OpenMP thread counts.",
        "- `best_openmp_summary.csv`: best OpenMP thread count per mesh size.",
        "",
        "Flux/scheme pairs found in this CSV:",
    ]

    for flux, scheme in sorted(grouped_rows):
        lines.append(f"- `{flux}` / `{scheme}`")

    (output_dir / "README.md").write_text("\n".join(lines))


def generate_plots(rows: list[BenchmarkRow], output_dir: Path) -> None:
    grouped = group_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    for (flux, scheme), scheme_rows in grouped.items():
        save_runtime_plot(scheme_rows, output_dir, flux, scheme)
        save_speedup_plot(scheme_rows, output_dir, flux, scheme)
        save_omp_scaling_plot(scheme_rows, output_dir, flux, scheme)
        save_breakdown_plot(scheme_rows, output_dir, flux, scheme)
        save_openmp_verification_plot(scheme_rows, output_dir, flux, scheme)
        save_openmp_verification_share_plot(scheme_rows, output_dir, flux, scheme)

    write_best_openmp_summary(grouped, output_dir)
    write_plot_readme(grouped, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark CSV files for flux/runtime comparisons.")
    parser.add_argument("csv_path", type=Path, help="Path to the benchmark CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_plots"),
        help="Directory where plots and summaries will be written.",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    if not rows:
        raise SystemExit("No rows found in CSV.")

    generate_plots(rows, args.output_dir)
    print(f"Wrote plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
