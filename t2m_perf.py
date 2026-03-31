import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PerfRow:
    chunk: int
    job_name: str
    queue: str
    run_seconds: int
    chsy: float
    sypd: float
    asypd: float


def _parse_hms_to_seconds(s: str) -> int:
    parts = s.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid duration '{s}'")
    h, m, sec = (int(p) for p in parts)
    return h * 3600 + m * 60 + sec


def _parse_float(s: str) -> float:
    return float(s.replace(",", "").strip())


def read_perf_tsv(path: str) -> list[PerfRow]:
    rows: list[PerfRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"Chunk", "Job Name", "Queue", "Run", "CHSY", "SYPD", "ASYPD"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}. Found: {reader.fieldnames}")

        for r in reader:
            rows.append(
                PerfRow(
                    chunk=int(r["Chunk"]),
                    job_name=str(r["Job Name"]),
                    queue=str(r["Queue"]),
                    run_seconds=_parse_hms_to_seconds(r["Run"]),
                    chsy=_parse_float(r["CHSY"]),
                    sypd=_parse_float(r["SYPD"]),
                    asypd=_parse_float(r["ASYPD"]),
                )
            )
    rows.sort(key=lambda x: x.chunk)
    return rows


def _seconds_to_hhmm(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h:02d}:{m:02d}"


def summarize(rows: list[PerfRow]) -> dict[str, str | int | float]:
    run_s = np.array([r.run_seconds for r in rows], dtype=float)
    sypd = np.array([r.sypd for r in rows], dtype=float)
    asypd = np.array([r.asypd for r in rows], dtype=float)

    return {
        "chunks": int(len(rows)),
        "avg_run_time_hhmm": _seconds_to_hhmm(float(run_s.mean())),
        "median_run_time_hhmm": _seconds_to_hhmm(float(np.median(run_s))),
        "mean_sypd": float(sypd.mean()),
        "mean_asypd": float(asypd.mean()),
    }


def write_summary_csv(path_out: str, summary_319: dict, summary_399: dict) -> None:
    out = Path(path_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    keys = [
        "chunks",
        "avg_run_time_hhmm",
        "median_run_time_hhmm",
        "mean_sypd",
        "mean_asypd",
    ]

    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "tco319", "tco399"])
        for k in keys:
            w.writerow([k, summary_319[k], summary_399[k]])


def plot_runtime_per_chunk(rows_319: list[PerfRow], rows_399: list[PerfRow], out_png: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib import failed. This is often caused by a broken user-site installation on HPC.\n"
            "Try running one of:\n"
            "  PYTHONNOUSERSITE=1 python3 /work/ab0995/a270135/Analysis/FESOM/MRana/t2m_perf.py\n"
            "  python3 -s /work/ab0995/a270135/Analysis/FESOM/MRana/t2m_perf.py\n"
            "Or fix/remove the user-site matplotlib in ~/.local.\n"
            f"Original error: {e}"
        )

    chunks_319 = np.array([r.chunk for r in rows_319], dtype=int)
    run_319 = np.array([r.run_seconds for r in rows_319], dtype=float) / 3600.0

    chunks_399 = np.array([r.chunk for r in rows_399], dtype=int)
    run_399 = np.array([r.run_seconds for r in rows_399], dtype=float) / 3600.0

    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.plot(chunks_319, run_319, label="TCO319", linewidth=1.8)
    ax.plot(chunks_399, run_399, label="TCO399", linewidth=1.8)
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Run time (hours)")
    ax.set_title("Runtime per chunk")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_per_chunk_3exp(
    rows_a3be: list[PerfRow],
    rows_a3df: list[PerfRow],
    rows_a3bo: list[PerfRow],
    out_png: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib import failed. This is often caused by a broken user-site installation on HPC.\n"
            "Try running one of:\n"
            "  PYTHONNOUSERSITE=1 python3 /work/ab0995/a270135/Analysis/FESOM/MRana/t2m_perf.py\n"
            "  python3 -s /work/ab0995/a270135/Analysis/FESOM/MRana/t2m_perf.py\n"
            "Or fix/remove the user-site matplotlib in ~/.local.\n"
            f"Original error: {e}"
        )

    chunks_a3be = np.array([r.chunk for r in rows_a3be], dtype=int)
    run_a3be = np.array([r.run_seconds for r in rows_a3be], dtype=float) / 3600.0

    chunks_a3df = np.array([r.chunk for r in rows_a3df], dtype=int)
    run_a3df = np.array([r.run_seconds for r in rows_a3df], dtype=float) / 3600.0

    chunks_a3bo = np.array([r.chunk for r in rows_a3bo], dtype=int)
    run_a3bo = np.array([r.run_seconds for r in rows_a3bo], dtype=float) / 3600.0

    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.plot(chunks_a3be, run_a3be, label="TCO319 (1200s)", linewidth=1.8)
    ax.plot(chunks_a3df, run_a3df, label="TCO319 (900s)", linewidth=1.8)
    ax.plot(chunks_a3bo, run_a3bo, label="TCO399 (900s)", linewidth=1.8)
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Run time (hours)")
    ax.set_title("Runtime per chunk")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--perf319",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/csv/TCO319_ConsideredPerformance_a3be.csv",
    )
    ap.add_argument(
        "--perf399",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/csv/TCO399_ConsideredPerformance_a3bo.csv",
    )
    ap.add_argument(
        "--perf319_900",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/csv/TCO319_ConsideredPerformance_a3df.csv",
    )
    ap.add_argument(
        "--out_summary_csv",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/csv/t2m_perf_summary.csv",
    )
    ap.add_argument(
        "--out_runtime_png",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/figures/t2m_perf_runtime_per_chunk.png",
    )
    ap.add_argument(
        "--out_runtime_png_3exp",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/figures/t2m_perf_runtime_per_chunk_3exp.png",
    )

    args = ap.parse_args()

    rows_319 = read_perf_tsv(args.perf319)
    rows_399 = read_perf_tsv(args.perf399)
    rows_319_900 = read_perf_tsv(args.perf319_900)

    summary_319 = summarize(rows_319)
    summary_399 = summarize(rows_399)

    write_summary_csv(args.out_summary_csv, summary_319, summary_399)
    try:
        plot_runtime_per_chunk(rows_319, rows_399, args.out_runtime_png)
    except RuntimeError as e:
        print(str(e))
        args.out_runtime_png = ""

    try:
        plot_runtime_per_chunk_3exp(rows_319, rows_319_900, rows_399, args.out_runtime_png_3exp)
    except RuntimeError as e:
        print(str(e))
        args.out_runtime_png_3exp = ""

    print("saved:", args.out_summary_csv)
    if args.out_runtime_png:
        print("saved:", args.out_runtime_png)
    if args.out_runtime_png_3exp:
        print("saved:", args.out_runtime_png_3exp)


if __name__ == "__main__":
    main()
