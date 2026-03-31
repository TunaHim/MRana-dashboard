import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


@dataclass(frozen=True)
class MonthPoint:
    dt: date
    mean319_c: float
    mean399_c: float


def _guess_data_var(ds: xr.Dataset) -> str:
    candidates = [
        v
        for v in ds.data_vars
        if v.lower() in {"t2m", "2t", "mean2t"} or "2m" in v.lower() or "t2" in v.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise ValueError(f"Could not uniquely determine temperature variable. data_vars={list(ds.data_vars)}")


def _read_month_global_mean_degC(path_grib: str, var_name: str | None = None) -> float:
    ds = xr.open_dataset(path_grib, engine="cfgrib")
    if var_name is None:
        var_name = _guess_data_var(ds)

    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=0)

    vals = np.asarray(da.values, dtype=float)
    # Treat fill values as NaN if present
    fill = da.attrs.get("missingValue")
    if fill is not None:
        vals = np.where(vals == float(fill), np.nan, vals)

    mean_k = float(np.nanmean(vals))
    return mean_k - 273.15


def _iter_months(start_ym: str, end_ym: str):
    sy, sm = (int(x) for x in start_ym.split("-"))
    ey, em = (int(x) for x in end_ym.split("-"))

    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def _fit_trend(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    coeff = np.polyfit(x, y, deg=1)
    return np.polyval(coeff, x)


def _trend_slope_degC_per_year(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    coeff = np.polyfit(x, y, deg=1)
    slope_degC_per_month = float(coeff[0])
    return slope_degC_per_month * 12.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1990-01", help="Start YYYY-MM")
    ap.add_argument("--end", default="2014-12", help="End YYYY-MM")
    ap.add_argument("--progress_every", type=int, default=12)
    ap.add_argument(
        "--max_months",
        type=int,
        default=0,
        help="If >0, stop after this many months (debug/quick run)",
    )
    ap.add_argument(
        "--path319",
        default="/work/ab0995/a270135/MN5/projt319/a3be_grib/varyears/{year}/sfc_mean2t/mean2t_{year}{month:02d}.grib",
    )
    ap.add_argument(
        "--path399",
        default="/work/ab0995/a270135/MN5/projt399/a3bo_grib/varyears/{year}/sfc_mean2t/mean2t_{year}{month:02d}.grib",
    )
    ap.add_argument(
        "--out_csv",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/csv/t2m_global_mean_monthly_1990_2014.csv",
    )
    ap.add_argument(
        "--out_png",
        default="/work/ab0995/a270135/Analysis/FESOM/MRana/figures/t2m_global_mean_monthly_1990_2014.png",
    )

    args = ap.parse_args()

    points: list[MonthPoint] = []
    for i, (year, month) in enumerate(_iter_months(args.start, args.end), start=1):
        if args.max_months and i > args.max_months:
            break
        p319 = args.path319.format(year=year, month=month)
        p399 = args.path399.format(year=year, month=month)

        m319 = _read_month_global_mean_degC(p319)
        m399 = _read_month_global_mean_degC(p399)
        points.append(MonthPoint(dt=date(year, month, 15), mean319_c=m319, mean399_c=m399))

        if args.progress_every and (i % args.progress_every == 0):
            print(f"processed {i} months ... latest={year}-{month:02d}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "tco319_mean2t_degC", "tco399_mean2t_degC", "diff_399_minus_319_degC"])
        for p in points:
            w.writerow([
                p.dt.isoformat(),
                f"{p.mean319_c:.6f}",
                f"{p.mean399_c:.6f}",
                f"{(p.mean399_c - p.mean319_c):.6f}",
            ])

    dates = np.array([np.datetime64(p.dt.isoformat()) for p in points])
    y319 = np.array([p.mean319_c for p in points], dtype=float)
    y399 = np.array([p.mean399_c for p in points], dtype=float)
    ydiff = y399 - y319

    t319 = _fit_trend(y319)
    t399 = _fit_trend(y399)
    tdiff = _fit_trend(ydiff)

    slope319 = _trend_slope_degC_per_year(y319)
    slope399 = _trend_slope_degC_per_year(y399)
    slopediff = _trend_slope_degC_per_year(ydiff)

    fig = plt.figure(figsize=(16, 8))
    ax = plt.gca()

    ax.plot(dates, y319, label="TCO319", linewidth=2.0)
    ax.plot(dates, y399, label="TCO399", linewidth=2.0)
    ax.plot(dates, ydiff, label="Δ (399−319)", linewidth=2.0)

    ax.plot(dates, t319, linestyle="--", linewidth=1.4)
    ax.plot(dates, t399, linestyle="--", linewidth=1.4)
    ax.plot(dates, tdiff, linestyle="--", linewidth=1.4)

    ax.set_title("Global mean monthly 2m temperature (1990–2014)", fontsize=20)
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("Temperature (°C)", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=14)

    trend_text = (
        "Trend (°C/year):\n"
        f"  TCO319: {slope319:+.4f}\n"
        f"  TCO399: {slope399:+.4f}\n"
        f"  Δ(399−319): {slopediff:+.4f}"
    )
    ax.text(
        0.72,
        0.52,
        trend_text,
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=14,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("saved:", str(out_csv))
    print("saved:", str(out_png))


if __name__ == "__main__":
    main()
